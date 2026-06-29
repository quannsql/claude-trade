import asyncio
import csv
import hashlib
import json
import logging
import os
import sqlite3
import collections
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import secrets
import base64

import eth_account
from eth_account.signers.local import LocalAccount
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from hyperliquid.info import Info

from bot_engine import API_URL, PRIVATE_KEY, logger as bot_logger, run_bot_async

try:
    import psycopg
except ImportError:  # Optional; local dashboard still works with SQLite.
    psycopg = None


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
LOCAL_DB_PATH = DATA_DIR / "dashboard_history.sqlite3"
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
AUTO_START_BOT = os.environ.get("AUTO_START_BOT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
STARTING_EQUITY = 100.0
DASHBOARD_USERNAME = os.environ.get("DASHBOARD_USERNAME", "admin")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")

app = FastAPI()

@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    if not DASHBOARD_PASSWORD:
        return await call_next(request)
        
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Basic "):
        return Response(
            content="Unauthorized", 
            status_code=401, 
            headers={"WWW-Authenticate": "Basic"}
        )
        
    encoded_credentials = auth_header.split(" ", 1)[1]
    try:
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        username, password = decoded_credentials.split(":", 1)
    except Exception:
        return Response(content="Invalid credentials", status_code=401, headers={"WWW-Authenticate": "Basic"})
        
    is_correct_username = secrets.compare_digest(username.encode("utf8"), DASHBOARD_USERNAME.encode("utf8"))
    is_correct_password = secrets.compare_digest(password.encode("utf8"), DASHBOARD_PASSWORD.encode("utf8"))
    
    if not (is_correct_username and is_correct_password):
        return Response(content="Invalid credentials", status_code=401, headers={"WWW-Authenticate": "Basic"})
        
    return await call_next(request)


# ---------------------------------------------------------
# Logging setup for WebSockets
# ---------------------------------------------------------
class WebSocketLogHandler(logging.Handler):
    def __init__(self, capacity=200):
        super().__init__()
        self.connected_websockets = set()
        self.history = collections.deque(maxlen=capacity)

    def emit(self, record):
        log_entry = self.format(record)
        self.history.append(log_entry)
        for ws in list(self.connected_websockets):
            try:
                asyncio.create_task(ws.send_text(log_entry))
            except Exception:
                self.connected_websockets.remove(ws)


ws_handler = WebSocketLogHandler()
formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
ws_handler.setFormatter(formatter)
bot_logger.addHandler(ws_handler)


# ---------------------------------------------------------
# Hyperliquid Info for API
# ---------------------------------------------------------
info = Info(API_URL, skip_ws=True)
user_address = None
MAIN_ADDRESS = os.environ.get("HL_MAIN_ADDRESS", "").strip()

if MAIN_ADDRESS:
    user_address = MAIN_ADDRESS
elif PRIVATE_KEY:
    try:
        account: LocalAccount = eth_account.Account.from_key(PRIVATE_KEY)
        user_address = account.address
    except Exception:
        pass


# ---------------------------------------------------------
# Lightweight storage for dashboard history
# ---------------------------------------------------------
storage_initialized = False
storage_error: Optional[str] = None


def _storage_backend() -> str:
    if DATABASE_URL and psycopg is not None:
        return "postgres"
    return "sqlite"


def _database_url() -> str:
    if DATABASE_URL.startswith("postgres://"):
        return "postgresql://" + DATABASE_URL[len("postgres://") :]
    return DATABASE_URL


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None

    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 10_000_000_000:
            raw = raw / 1000
        return datetime.fromtimestamp(raw, tz=timezone.utc)

    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _chart_time(value: Any) -> Optional[int]:
    parsed = _parse_datetime(value)
    return int(parsed.timestamp()) if parsed else None


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _fill_id(fill: dict[str, Any]) -> str:
    for key in ("hash", "tid", "oid"):
        value = fill.get(key)
        if value:
            return str(value)

    raw = "|".join(
        str(fill.get(key, ""))
        for key in ("time", "coin", "dir", "px", "sz", "closedPnl")
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def init_storage() -> None:
    global storage_initialized, storage_error
    if storage_initialized:
        return

    DATA_DIR.mkdir(exist_ok=True)
    try:
        if _storage_backend() == "postgres":
            with psycopg.connect(_database_url()) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS account_snapshots (
                            id BIGSERIAL PRIMARY KEY,
                            captured_at TIMESTAMPTZ NOT NULL,
                            address TEXT,
                            account_value DOUBLE PRECISION,
                            total_margin_used DOUBLE PRECISION,
                            withdrawable DOUBLE PRECISION,
                            payload JSONB NOT NULL
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS fills (
                            fill_id TEXT PRIMARY KEY,
                            address TEXT,
                            fill_time TIMESTAMPTZ,
                            coin TEXT,
                            dir TEXT,
                            px DOUBLE PRECISION,
                            sz DOUBLE PRECISION,
                            closed_pnl DOUBLE PRECISION,
                            payload JSONB NOT NULL
                        )
                        """
                    )
        else:
            with sqlite3.connect(LOCAL_DB_PATH) as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS account_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        captured_at TEXT NOT NULL,
                        address TEXT,
                        account_value REAL,
                        total_margin_used REAL,
                        withdrawable REAL,
                        payload TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS fills (
                        fill_id TEXT PRIMARY KEY,
                        address TEXT,
                        fill_time TEXT,
                        coin TEXT,
                        dir TEXT,
                        px REAL,
                        sz REAL,
                        closed_pnl REAL,
                        payload TEXT NOT NULL
                    );
                    """
                )

        storage_initialized = True
        storage_error = None
    except Exception as exc:
        storage_error = str(exc)
        bot_logger.warning("Dashboard history storage disabled: %s", exc)


def persist_dashboard_state(
    address: str, margin_summary: dict[str, Any], fills: list[dict[str, Any]]
) -> None:
    init_storage()
    if not storage_initialized:
        return

    captured_at = datetime.now(timezone.utc).isoformat()
    account_value = _safe_float(margin_summary.get("accountValue"))
    total_margin_used = _safe_float(margin_summary.get("totalMarginUsed"))
    withdrawable = _safe_float(margin_summary.get("withdrawable"))
    payload = _json_dump(margin_summary)

    if _storage_backend() == "postgres":
        with psycopg.connect(_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO account_snapshots (
                        captured_at, address, account_value, total_margin_used,
                        withdrawable, payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        captured_at,
                        address,
                        account_value,
                        total_margin_used,
                        withdrawable,
                        payload,
                    ),
                )

                for fill in fills:
                    fill_time = _parse_datetime(fill.get("time"))
                    cur.execute(
                        """
                        INSERT INTO fills (
                            fill_id, address, fill_time, coin, dir, px, sz,
                            closed_pnl, payload
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (fill_id) DO UPDATE SET
                            address = EXCLUDED.address,
                            fill_time = EXCLUDED.fill_time,
                            coin = EXCLUDED.coin,
                            dir = EXCLUDED.dir,
                            px = EXCLUDED.px,
                            sz = EXCLUDED.sz,
                            closed_pnl = EXCLUDED.closed_pnl,
                            payload = EXCLUDED.payload
                        """,
                        (
                            _fill_id(fill),
                            address,
                            fill_time,
                            fill.get("coin"),
                            fill.get("dir"),
                            _safe_float(fill.get("px")),
                            _safe_float(fill.get("sz")),
                            _safe_float(fill.get("closedPnl")),
                            _json_dump(fill),
                        ),
                    )
        return

    with sqlite3.connect(LOCAL_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO account_snapshots (
                captured_at, address, account_value, total_margin_used,
                withdrawable, payload
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                captured_at,
                address,
                account_value,
                total_margin_used,
                withdrawable,
                payload,
            ),
        )

        for fill in fills:
            fill_time = _parse_datetime(fill.get("time"))
            conn.execute(
                """
                INSERT OR REPLACE INTO fills (
                    fill_id, address, fill_time, coin, dir, px, sz,
                    closed_pnl, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _fill_id(fill),
                    address,
                    fill_time.isoformat() if fill_time else None,
                    fill.get("coin"),
                    fill.get("dir"),
                    _safe_float(fill.get("px")),
                    _safe_float(fill.get("sz")),
                    _safe_float(fill.get("closedPnl")),
                    _json_dump(fill),
                ),
            )


def _query_history(limit: int) -> dict[str, Any]:
    init_storage()
    if not storage_initialized:
        return {
            "storage": _storage_backend(),
            "storage_ready": False,
            "error": storage_error,
            "snapshots": [],
            "fills": [],
        }

    limit = max(1, min(limit, 500))
    if _storage_backend() == "postgres":
        with psycopg.connect(_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT captured_at, address, account_value,
                           total_margin_used, withdrawable
                    FROM account_snapshots
                    ORDER BY captured_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                snapshots = [
                    {
                        "captured_at": row[0].isoformat() if row[0] else None,
                        "address": row[1],
                        "account_value": row[2],
                        "total_margin_used": row[3],
                        "withdrawable": row[4],
                    }
                    for row in cur.fetchall()
                ]
                cur.execute(
                    """
                    SELECT fill_id, fill_time, coin, dir, px, sz, closed_pnl
                    FROM fills
                    ORDER BY fill_time DESC NULLS LAST
                    LIMIT %s
                    """,
                    (limit,),
                )
                fills = [
                    {
                        "fill_id": row[0],
                        "fill_time": row[1].isoformat() if row[1] else None,
                        "coin": row[2],
                        "dir": row[3],
                        "px": row[4],
                        "sz": row[5],
                        "closed_pnl": row[6],
                    }
                    for row in cur.fetchall()
                ]
    else:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            snapshots = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT captured_at, address, account_value,
                           total_margin_used, withdrawable
                    FROM account_snapshots
                    ORDER BY captured_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            ]
            fills = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT fill_id, fill_time, coin, dir, px, sz, closed_pnl
                    FROM fills
                    ORDER BY fill_time DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            ]

    return {
        "storage": _storage_backend(),
        "storage_ready": True,
        "error": None,
        "snapshots": snapshots,
        "fills": fills,
    }


# ---------------------------------------------------------
# Backtest chart data
# ---------------------------------------------------------
def _discover_symbols() -> list[str]:
    RESULTS_DIR.mkdir(exist_ok=True)
    return sorted(
        path.name[: -len("_trades.csv")]
        for path in RESULTS_DIR.glob("*_trades.csv")
        if path.name.endswith("_trades.csv")
    )


def _empty_backtest_payload(symbols: list[str], selected_symbol: Optional[str]) -> dict[str, Any]:
    return {
        "symbols": symbols,
        "selected_symbol": selected_symbol,
        "metrics": {
            "total_trades": 0,
            "win_rate_pct": 0,
            "net_pnl_usd": 0,
            "profit_factor": None,
            "max_drawdown_pct": 0,
            "final_equity_usd": STARTING_EQUITY,
            "trades_per_day": 0,
            "total_fees_usd": 0,
            "expectancy_usd": 0,
            "best_trade_usd": 0,
            "worst_trade_usd": 0,
        },
        "series": {"equity": [], "drawdown": [], "pnl": [], "daily_pnl": []},
        "exit_reason_counts": {},
        "trades": [],
        "chart_image": None,
    }


def _load_backtest_payload(symbol: str, symbols: list[str]) -> dict[str, Any]:
    path = RESULTS_DIR / f"{symbol}_trades.csv"
    if not path.exists():
        return _empty_backtest_payload(symbols, symbol)

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return _empty_backtest_payload(symbols, symbol)

    first_time = _chart_time(rows[0].get("entry_time")) or int(datetime.now().timestamp())
    baseline_time = max(first_time - 60, 1)
    equity = STARTING_EQUITY
    running_max = STARTING_EQUITY
    last_chart_time = baseline_time

    equity_series = [{"time": baseline_time, "value": round(STARTING_EQUITY, 4)}]
    drawdown_series = [{"time": baseline_time, "value": 0.0}]
    cum_pnl_series = [{"time": baseline_time, "value": 0.0}]
    pnl_series = []
    daily_pnl: dict[str, float] = {}
    exit_reason_counts: dict[str, int] = {}
    table_rows = []
    net_values = []
    fees_total = 0.0
    wins = 0
    parsed_times: list[datetime] = []

    for index, row in enumerate(rows, start=1):
        net_pnl = _safe_float(row.get("net_pnl"))
        equity = _safe_float(row.get("equity_after"), equity + net_pnl)
        running_max = max(running_max, equity)
        drawdown_pct = ((equity - running_max) / running_max * 100) if running_max else 0.0

        chart_time = _chart_time(row.get("exit_time") or row.get("entry_time"))
        if chart_time is None:
            chart_time = last_chart_time + 60
        if chart_time <= last_chart_time:
            chart_time = last_chart_time + 1
        last_chart_time = chart_time

        exit_dt = _parse_datetime(row.get("exit_time"))
        entry_dt = _parse_datetime(row.get("entry_time"))
        if exit_dt:
            parsed_times.append(exit_dt)
            day_key = exit_dt.date().isoformat()
        elif entry_dt:
            parsed_times.append(entry_dt)
            day_key = entry_dt.date().isoformat()
        else:
            day_key = str(index)
        daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + net_pnl

        is_win = _boolish(row.get("win")) if row.get("win") else net_pnl > 0
        if is_win:
            wins += 1

        reason = row.get("exit_reason") or "unknown"
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
        fees_total += _safe_float(row.get("fees"))
        net_values.append(net_pnl)

        equity_series.append({"time": chart_time, "value": round(equity, 4)})
        drawdown_series.append({"time": chart_time, "value": round(drawdown_pct, 4)})
        cum_pnl_series.append({"time": chart_time, "value": round(equity - STARTING_EQUITY, 4)})
        pnl_series.append(
            {
                "time": chart_time,
                "value": round(net_pnl, 6),
                "color": "#15803d" if net_pnl >= 0 else "#dc2626",
            }
        )

        table_rows.append(
            {
                "index": index,
                "symbol": row.get("symbol") or symbol,
                "direction": row.get("direction", ""),
                "entry_time": row.get("entry_time", ""),
                "exit_time": row.get("exit_time", ""),
                "entry_price": _safe_float(row.get("entry_price")),
                "exit_price": _safe_float(row.get("exit_price")),
                "exit_reason": reason,
                "score": _safe_int(row.get("score")),
                "net_pnl": round(net_pnl, 6),
                "equity_after": round(equity, 6),
                "win": is_win,
            }
        )

    losses = len(rows) - wins
    gross_profit = sum(value for value in net_values if value > 0)
    gross_loss = -sum(value for value in net_values if value < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    max_drawdown = min((point["value"] for point in drawdown_series), default=0.0)

    days_span = 1
    if parsed_times:
        span = max(parsed_times) - min(parsed_times)
        days_span = max(span.days, 1)

    daily_series = [
        {"time": day, "value": round(value, 6), "color": "#15803d" if value >= 0 else "#dc2626"}
        for day, value in sorted(daily_pnl.items())
    ]

    chart_path = RESULTS_DIR / f"{symbol}_chart.png"
    chart_image = f"/results/{chart_path.name}" if chart_path.exists() else None

    return {
        "symbols": symbols,
        "selected_symbol": symbol,
        "metrics": {
            "total_trades": len(rows),
            "win_rate_pct": round(wins / len(rows) * 100, 2) if rows else 0,
            "losses": losses,
            "net_pnl_usd": round(sum(net_values), 4),
            "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
            "max_drawdown_pct": round(max_drawdown, 2),
            "final_equity_usd": round(equity, 4),
            "trades_per_day": round(len(rows) / days_span, 2),
            "total_fees_usd": round(fees_total, 4),
            "expectancy_usd": round(sum(net_values) / len(rows), 6) if rows else 0,
            "best_trade_usd": round(max(net_values), 6) if net_values else 0,
            "worst_trade_usd": round(min(net_values), 6) if net_values else 0,
        },
        "series": {
            "equity": equity_series,
            "drawdown": drawdown_series,
            "cum_pnl": cum_pnl_series,
            "pnl": pnl_series,
            "daily_pnl": daily_series,
        },
        "exit_reason_counts": dict(
            sorted(exit_reason_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "trades": list(reversed(table_rows[-80:])),
        "chart_image": chart_image,
    }


def _load_live_chart_payload(address: str, symbols: list[str]) -> dict[str, Any]:
    init_storage()
    if not storage_initialized or not address:
        return _empty_backtest_payload(symbols, "LIVE")

    snapshots = []
    fills = []
    
    if _storage_backend() == "postgres":
        with psycopg.connect(_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT captured_at, account_value FROM account_snapshots WHERE address = %s ORDER BY captured_at ASC", (address,))
                snapshots = cur.fetchall()
                cur.execute("SELECT fill_time, coin, dir, px, sz, closed_pnl FROM fills WHERE address = %s ORDER BY fill_time ASC", (address,))
                fills = cur.fetchall()
    else:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            snapshots = conn.execute("SELECT captured_at, account_value FROM account_snapshots WHERE address = ? ORDER BY captured_at ASC", (address,)).fetchall()
            fills = conn.execute("SELECT fill_time, coin, dir, px, sz, closed_pnl FROM fills WHERE address = ? ORDER BY fill_time ASC", (address,)).fetchall()

    equity_series = []
    drawdown_series = []
    running_max = STARTING_EQUITY
    final_equity = STARTING_EQUITY
    last_chart_time = 0
    
    for row in snapshots:
        dt = _parse_datetime(row[0])
        val = _safe_float(row[1])
        if dt and val > 0:
            final_equity = val
            running_max = max(running_max, val)
            dd = ((val - running_max) / running_max * 100) if running_max else 0.0
            ts = int(dt.timestamp())
            if ts <= last_chart_time:
                ts = last_chart_time + 1
            last_chart_time = ts
            
            equity_series.append({"time": ts, "value": round(val, 4)})
            drawdown_series.append({"time": ts, "value": round(dd, 4)})

    if not equity_series:
        baseline_time = int(datetime.now().timestamp())
        equity_series.append({"time": baseline_time, "value": STARTING_EQUITY})
        drawdown_series.append({"time": baseline_time, "value": 0.0})
    
    pnl_series = []
    cum_pnl_series = []
    cumulative_pnl = 0.0
    daily_pnl = {}
    table_rows = []
    net_values = []
    wins = 0
    parsed_times = []
    last_fill_time = 0
    
    for index, row in enumerate(fills, start=1):
        dt = _parse_datetime(row[0])
        coin = row[1]
        direction = row[2]
        px = _safe_float(row[3])
        sz = _safe_float(row[4])
        pnl = _safe_float(row[5])
        
        if dt:
            parsed_times.append(dt)
            day_key = dt.date().isoformat()
            ts = int(dt.timestamp())
        else:
            day_key = str(index)
            ts = int(datetime.now().timestamp())
            
        if ts <= last_fill_time:
            ts = last_fill_time + 1
        last_fill_time = ts
            
        daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + pnl
        is_win = pnl > 0
        if is_win:
            wins += 1
            
        net_values.append(pnl)
        cumulative_pnl += pnl
        
        cum_pnl_series.append({
            "time": ts,
            "value": round(cumulative_pnl, 6)
        })
        
        pnl_series.append({
            "time": ts,
            "value": round(pnl, 6),
            "color": "#15803d" if pnl >= 0 else "#dc2626",
        })
        
        table_rows.append({
            "index": index,
            "symbol": coin,
            "direction": direction,
            "entry_time": "",
            "exit_time": dt.isoformat() if dt else "",
            "entry_price": 0,
            "exit_price": px,
            "exit_reason": "live_fill",
            "score": 0,
            "net_pnl": round(pnl, 6),
            "equity_after": 0,
            "win": is_win,
        })
        
    losses = len(fills) - wins
    gross_profit = sum(v for v in net_values if v > 0)
    gross_loss = -sum(v for v in net_values if v < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    max_drawdown = min((point["value"] for point in drawdown_series), default=0.0)

    days_span = 1
    if parsed_times:
        span = max(parsed_times) - min(parsed_times)
        days_span = max(span.days, 1)
        
    daily_series = [
        {"time": day, "value": round(value, 6), "color": "#15803d" if value >= 0 else "#dc2626"}
        for day, value in sorted(daily_pnl.items())
    ]
    
    return {
        "symbols": ["LIVE"] + symbols,
        "selected_symbol": "LIVE",
        "metrics": {
            "total_trades": len(fills),
            "win_rate_pct": round(wins / len(fills) * 100, 2) if fills else 0,
            "losses": losses,
            "net_pnl_usd": round(sum(net_values), 4),
            "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
            "max_drawdown_pct": round(max_drawdown, 2),
            "final_equity_usd": round(final_equity, 4),
            "trades_per_day": round(len(fills) / days_span, 2),
            "total_fees_usd": 0,
            "expectancy_usd": round(sum(net_values) / len(fills), 6) if fills else 0,
            "best_trade_usd": round(max(net_values), 6) if net_values else 0,
            "worst_trade_usd": round(min(net_values), 6) if net_values else 0,
        },
        "series": {
            "equity": equity_series,
            "drawdown": drawdown_series,
            "cum_pnl": cum_pnl_series,
            "pnl": pnl_series,
            "daily_pnl": daily_series,
        },
        "exit_reason_counts": {"live_fill": len(fills)} if fills else {},
        "trades": list(reversed(table_rows[-80:])),
        "chart_image": None,
    }


# ---------------------------------------------------------
# Background Task
# ---------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    init_storage()
    if AUTO_START_BOT:
        asyncio.create_task(run_bot_async())
    else:
        bot_logger.info("AUTO_START_BOT=0, dashboard started without the live bot.")


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/api/state")
def get_state():
    """Fetch balance and open orders from Hyperliquid API."""
    if not user_address:
        return {"error": "Wallet not configured.", "storage": _query_history(20)}

    try:
        user_state = info.user_state(user_address)
        open_orders = info.open_orders(user_address)
        fills = info.user_fills(user_address) or []
        margin_summary = user_state.get("marginSummary", {})
        positions = user_state.get("assetPositions", [])

        persist_dashboard_state(user_address, margin_summary, fills[:200])

        return {
            "address": user_address,
            "margin_summary": margin_summary,
            "positions": positions,
            "open_orders": open_orders,
            "fills": fills[:50],
            "storage": {
                "backend": _storage_backend(),
                "ready": storage_initialized,
                "error": storage_error,
            },
        }
    except Exception as e:
        return {"error": str(e), "storage": _query_history(20)}


@app.get("/api/backtest")
def get_backtest(symbol: Optional[str] = Query(default=None)):
    symbols = _discover_symbols()
    if not symbols:
        return _empty_backtest_payload([], None)

    selected_symbol = symbol if symbol in symbols else symbols[0]
    return _load_backtest_payload(selected_symbol, symbols)


@app.get("/api/live_chart")
def get_live_chart():
    symbols = _discover_symbols()
    return _load_live_chart_payload(user_address, symbols)


@app.get("/api/history")
def get_history(limit: int = Query(default=200, ge=1, le=500)):
    return _query_history(limit)


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    if DASHBOARD_PASSWORD:
        auth_header = websocket.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            await websocket.close(code=1008)
            return
            
        encoded_credentials = auth_header.split(" ", 1)[1]
        try:
            decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
            username, password = decoded_credentials.split(":", 1)
            
            is_correct_username = secrets.compare_digest(username.encode("utf8"), DASHBOARD_USERNAME.encode("utf8"))
            is_correct_password = secrets.compare_digest(password.encode("utf8"), DASHBOARD_PASSWORD.encode("utf8"))
            
            if not (is_correct_username and is_correct_password):
                await websocket.close(code=1008)
                return
        except Exception:
            await websocket.close(code=1008)
            return

    await websocket.accept()
    ws_handler.connected_websockets.add(websocket)
    for log_entry in ws_handler.history:
        try:
            await websocket.send_text(log_entry)
        except Exception:
            pass
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_handler.connected_websockets.remove(websocket)


# ---------------------------------------------------------
# Static Files
# ---------------------------------------------------------
WEB_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.get("/")
def read_root():
    return FileResponse(WEB_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
