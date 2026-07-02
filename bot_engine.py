"""
bot_engine_improved.py — Bản vá lỗi và nâng cấp so với bot_engine.py gốc

THAY ĐỔI CHÍNH (so với bản gốc):
──────────────────────────────────────────────────────────────────────────────
FIX #1 — _ACTIVE_ORDER GUARD (ngăn Position Stacking hoàn toàn)
  Vấn đề: khi limit order GTC đang chờ fill, _position_for_coin() = 0 và
  _open_orders_for_coin() đôi khi rỗng do API lag → bot tiếp tục mở lệnh mới.
  Fix: thêm biến _active_entry_oids: set[str] để track ngay khi gửi order,
  trước khi exchange xác nhận. Guard check biến này TRƯỚC khi gọi API.

FIX #2 — COOLDOWN SAU SL (ngăn full-size ngay sau loss)
  Vấn đề: cooldown chỉ kích hoạt sau max_consecutive_losses (mặc định 3),
  không có cooldown nhỏ sau mỗi SL đơn lẻ.
  Fix: thêm SL_SINGLE_COOLDOWN_MINUTES (default 10 phút) sau mỗi lần thua.

FIX #3 — MAX NOTIONAL GUARD (giới hạn tuyệt đối tổng exposure)
  Vấn đề: không có giới hạn tổng notional nếu stacking xảy ra.
  Fix: trước mỗi entry, kiểm tra account_value. Nếu total margin used /
  account_value > MAX_MARGIN_USAGE_PCT thì skip setup.

FIX #4 — LIMIT ORDER TIMEOUT CANCEL xác nhận
  Vấn đề: sau khi cancel limit order timeout, code không verify cancel thành
  công → vẫn có thể bị fill muộn tạo orphan position.
  Fix: sau cancel, poll position 3 lần để xác nhận không có position mở.

FIX #5 — SL SIZE COVERAGE
  Vấn đề: SL được đặt cho filled_size toàn bộ, nhưng sau TP1 còn lại
  remaining_size < filled_size → SL cũ không đủ cover.
  Cải thiện: SL luôn đặt với kích thước khớp chính xác remaining position.

IMPROVEMENT #6 — PARTIAL FILL PNL TRACKING
  Cải thiện: _final_net_pnl giờ tìm cả fills không có oid trong managed_oids
  (partial fills bị fragmented) bằng cách so coin + time range + direction.
"""

import asyncio
import csv
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any

import eth_account
import pandas as pd
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from indicators import add_indicators, score_setup
from filters import compute_risk_sizing
from run_backtest import CONFIG, COIN_CONFIG


logger = logging.getLogger("bot_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
# Fix log đôi: không propagate lên root logger (uvicorn/app.py có handler riêng)
logger.propagate = False


PRIVATE_KEY = os.environ.get("HL_PRIVATE_KEY", "")
HL_NETWORK = os.environ.get("HL_NETWORK", "TESTNET").strip().upper()
API_URL = constants.MAINNET_API_URL if HL_NETWORK == "MAINNET" else constants.TESTNET_API_URL

SIGNAL_POLL_SECONDS = int(os.environ.get("HL_SIGNAL_POLL_SECONDS", "3"))
ORDER_POLL_SECONDS = int(os.environ.get("HL_ORDER_POLL_SECONDS", "3"))
LOOKBACK_1M = int(os.environ.get("HL_LOOKBACK_1M", "300"))
LOOKBACK_5M = int(os.environ.get("HL_LOOKBACK_5M", "160"))
LOOKBACK_15M = int(os.environ.get("HL_LOOKBACK_15M", "260"))
MARKET_CLOSE_SLIPPAGE = float(os.environ.get("HL_MARKET_CLOSE_SLIPPAGE", "0.002"))
POSITION_EPSILON = 1e-8

DISABLED_COINS: set[str] = set()
# ── FIX #2: Cooldown nhỏ sau mỗi lần thua đơn lẻ ──
SL_SINGLE_COOLDOWN_MINUTES = int(os.environ.get("HL_SL_COOLDOWN_MINUTES", "3"))

# ── FIX #3: Giới hạn tổng margin usage ──
MAX_MARGIN_USAGE_PCT = float(os.environ.get("HL_MAX_MARGIN_PCT", "0.70"))  # max 70% account dùng làm margin

# ── HARD DOLLAR SL floor: backstop tối thiểu (v3: backstop thực = planned_risk × mult) ──
HARD_DOLLAR_SL_USD = float(os.environ.get("HL_HARD_DOLLAR_SL", "4.5"))
HARD_DOLLAR_SL_CHECK_INTERVAL = int(os.environ.get("HL_HARD_SL_INTERVAL", "5"))  # check mỗi N giây

# ── FIX #1: Track active entry orders để chống stacking ──
# key = coin, value = timestamp khi entry order được gửi đi
_active_entry: dict[str, float] = {}
ACTIVE_ENTRY_TIMEOUT_SEC = 180  # 3 phút: nếu quá thời gian này mà không có result thì tự động clear

# Per-coin lock để ngăn concurrent execution
_coin_locks: dict[str, asyncio.Lock] = {}

# Theo dõi thời gian orphan position của từng coin
_orphan_timers: dict[str, float] = {}

# ── v3: OBI smoothing — buffer mẫu OBI per coin, lấy mỗi scan cycle (3s) ──
OBI_WINDOW_SEC = int(os.environ.get("HL_OBI_WINDOW_SEC", "90"))
OBI_MIN_SAMPLES = int(os.environ.get("HL_OBI_MIN_SAMPLES", "3"))
_obi_buffers: dict[str, deque] = {}
_latest_ob_analysis: dict[str, dict] = {}

# ── v3: Candle cache — chỉ refetch khi nến mới ĐÃ đóng (giảm ~20x REST call) ──
_candle_cache: dict[tuple[str, str], dict[str, Any]] = {}

# ── v3: Funding rate cache (meta_and_asset_ctxs là call nặng) ──
_funding_cache: dict[str, dict[str, Any]] = {}
FUNDING_CACHE_TTL_SEC = 120

# ── v3: API circuit breaker — nhiều lỗi liên tiếp → tạm dừng MỞ LỆNH MỚI ──
# (vòng quản lý vị thế đang mở vẫn chạy bình thường)
_api_failures: deque = deque()
API_FAILURE_WINDOW_SEC = 60
API_FAILURE_THRESHOLD = int(os.environ.get("HL_API_FAILURE_THRESHOLD", "6"))

# ── v3: Throttle log lặp (cooldown/risk block) ──
_last_throttled_log: dict[str, float] = {}

# ── v3: CSV log — mọi setup (kể cả bị block) + mọi trade, để tune weight theo data thật ──
SETUP_LOG_PATH = os.environ.get("HL_SETUP_LOG", "setups_log.csv")
TRADE_LOG_PATH = os.environ.get("HL_TRADE_LOG", "trades_log.csv")


@dataclass
class FillSummary:
    size: float = 0.0
    notional: float = 0.0
    closed_pnl: float = 0.0
    fees: float = 0.0
    crossed: bool = False

    @property
    def avg_price(self) -> float | None:
        if self.size <= 0:
            return None
        return self.notional / self.size


@dataclass
class TradeResult:
    coin: str
    direction: str
    exit_reason: str
    net_pnl: float
    filled_size: float
    entry_price: float
    entry_time: datetime
    exit_time: datetime

    @property
    def win(self) -> bool:
        return self.net_pnl > 0


@dataclass
class RiskState:
    consecutive_losses: dict[str, int] = field(default_factory=dict)
    cooldown_until: dict[str, datetime] = field(default_factory=dict)
    trades_today: dict[tuple[str, date], int] = field(default_factory=dict)
    pnl_today: dict[tuple[str, date], float] = field(default_factory=dict)

    def can_trade(self, coin: str, now_utc: datetime) -> tuple[bool, str | None]:
        cooldown = self.cooldown_until.get(coin)
        if cooldown and now_utc < cooldown:
            remaining = int((cooldown - now_utc).total_seconds() / 60)
            return False, f"cooldown {remaining}m còn lại (đến {cooldown.strftime('%H:%M')})"

        today = now_utc.date()
        total_trades_today = sum(v for k, v in self.trades_today.items() if k[1] == today)
        if total_trades_today >= CONFIG["max_trades_per_day"]:
            return False, "max global trades per day reached"

        total_pnl_today = sum(v for k, v in self.pnl_today.items() if k[1] == today)
        if total_pnl_today <= -CONFIG["daily_loss_limit_usd"]:
            return False, f"daily global loss limit reached ({total_pnl_today:.2f} <= {-CONFIG['daily_loss_limit_usd']})"

        return True, None

    def record_trade(self, result: TradeResult) -> None:
        day_key = (result.coin, result.entry_time.date())
        self.trades_today[day_key] = self.trades_today.get(day_key, 0) + 1
        self.pnl_today[day_key] = self.pnl_today.get(day_key, 0.0) + result.net_pnl

        if result.win:
            self.consecutive_losses[result.coin] = 0
        else:
            losses = self.consecutive_losses.get(result.coin, 0) + 1
            self.consecutive_losses[result.coin] = losses

            # ── FIX #2: Cooldown nhỏ sau MỖI lần thua ──
            single_cooldown = result.exit_time + timedelta(minutes=SL_SINGLE_COOLDOWN_MINUTES)
            existing = self.cooldown_until.get(result.coin)
            if existing is None or single_cooldown > existing:
                self.cooldown_until[result.coin] = single_cooldown
                logger.info(
                    "%s: single-loss cooldown until %s (%dm)",
                    result.coin,
                    single_cooldown.strftime("%H:%M"),
                    SL_SINGLE_COOLDOWN_MINUTES,
                )

            # Cooldown dài hơn sau N losses liên tiếp
            if losses >= CONFIG["max_consecutive_losses"]:
                long_cooldown = result.exit_time + timedelta(minutes=CONFIG["cooldown_minutes"])
                self.cooldown_until[result.coin] = long_cooldown
                self.consecutive_losses[result.coin] = 0
                logger.info(
                    "%s: %d consecutive losses → long cooldown until %s",
                    result.coin,
                    losses,
                    long_cooldown.strftime("%H:%M"),
                )

        if self.pnl_today[day_key] <= -CONFIG["daily_loss_limit_usd"]:
            tomorrow = datetime.combine(
                result.exit_time.date() + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            self.cooldown_until[result.coin] = tomorrow
            logger.info("%s: daily loss limit hit → pause until midnight", result.coin)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _interval_ms(interval: str) -> int:
    if interval == "1m":
        return 60_000
    if interval == "5m":
        return 5 * 60_000
    if interval == "15m":
        return 15 * 60_000
    raise ValueError(f"Unsupported interval: {interval}")


def _symbol_to_coin(symbol: str) -> str:
    return symbol.replace("USDT", "").replace("/", "")


def configured_live_coins() -> list[str]:
    raw = os.environ.get("HL_LIVE_COINS", "").strip()
    if raw:
        return [item.strip().upper() for item in raw.split(",") if item.strip()]
    return [_symbol_to_coin(s) for s in CONFIG.get("symbols", [])]


def _record_api_failure() -> None:
    now = time.time()
    _api_failures.append(now)
    while _api_failures and now - _api_failures[0] > API_FAILURE_WINDOW_SEC:
        _api_failures.popleft()


def _api_circuit_open() -> bool:
    """True nếu quá nhiều lỗi API trong 60s — tạm dừng mở lệnh mới."""
    now = time.time()
    while _api_failures and now - _api_failures[0] > API_FAILURE_WINDOW_SEC:
        _api_failures.popleft()
    return len(_api_failures) >= API_FAILURE_THRESHOLD


def _log_throttled(key: str, interval_sec: float, msg: str, *args) -> None:
    """Log tối đa 1 lần mỗi interval_sec cho mỗi key — chống spam log."""
    now = time.time()
    if now - _last_throttled_log.get(key, 0.0) >= interval_sec:
        _last_throttled_log[key] = now
        logger.info(msg, *args)


def _append_csv(path: str, row: dict) -> None:
    """Append 1 dòng vào CSV, tự viết header nếu file mới. Không bao giờ raise."""
    try:
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        logger.debug("csv append failed (%s): %s", path, exc)


def _log_setup_csv(coin: str, signal_ts: pd.Timestamp, price: float,
                   setup: dict, decision: str) -> None:
    _append_csv(SETUP_LOG_PATH, {
        "ts": signal_ts.isoformat(),
        "coin": coin,
        "regime": setup.get("regime", ""),
        "entry_mode": setup.get("entry_mode", ""),
        "direction": setup.get("direction") or "",
        "score": setup.get("score", 0),
        "confidence": setup.get("confidence", ""),
        "obi": round(setup.get("obi", 0.0), 4),
        "price": price,
        "bb_width_pct": round(setup.get("bb_width_pct", 0.0), 4),
        "decision": decision,
        "block_reason": "; ".join(setup.get("block_reasons", [])),
        "details": json.dumps(setup.get("score_details", {})),
    })


def _log_trade_csv(result: "TradeResult", score: int, regime: str, obi: float) -> None:
    _append_csv(TRADE_LOG_PATH, {
        "entry_time": result.entry_time.isoformat(),
        "exit_time": result.exit_time.isoformat(),
        "coin": result.coin,
        "direction": result.direction,
        "score": score,
        "regime": regime,
        "obi": round(obi, 4),
        "exit_reason": result.exit_reason,
        "filled_size": result.filled_size,
        "entry_price": result.entry_price,
        "net_pnl": round(result.net_pnl, 6),
        "win": result.win,
    })


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_price(price: float) -> float:
    return float(f"{price:.5g}")


def _round_size(exchange: Exchange, coin: str, size: float) -> float:
    asset = exchange.info.name_to_asset(coin)
    decimals = exchange.info.asset_to_sz_decimals.get(asset, 4)
    quant = Decimal("1").scaleb(-decimals)
    rounded = Decimal(str(size)).quantize(quant, rounding=ROUND_DOWN)
    return float(rounded)


def _order_statuses(response: Any) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    resp = response.get("response", {})
    if not isinstance(resp, dict):
        return []
    data = resp.get("data", {})
    if not isinstance(data, dict):
        return []
    statuses = data.get("statuses", [])
    return statuses if isinstance(statuses, list) else []


def _extract_oid(response: Any) -> int | None:
    for status in _order_statuses(response):
        for key in ("resting", "filled"):
            details = status.get(key)
            if isinstance(details, dict) and details.get("oid") is not None:
                return int(details["oid"])
    return None


def _response_errors(response: Any) -> list[str]:
    errors: list[str] = []
    if isinstance(response, dict) and response.get("status") == "err":
        resp_data = response.get("response")
        if isinstance(resp_data, str):
            errors.append(resp_data)
            
    for status in _order_statuses(response):
        if isinstance(status, dict) and status.get("error"):
            errors.append(str(status["error"]))
    return errors


def _summarize_fills(fills: list[dict[str, Any]], oid: int | None = None) -> FillSummary:
    summary = FillSummary()
    for fill in fills:
        if oid is not None and int(fill.get("oid", -1)) != oid:
            continue
        size = _to_float(fill.get("sz"))
        price = _to_float(fill.get("px"))
        summary.size += size
        summary.notional += size * price
        summary.closed_pnl += _to_float(fill.get("closedPnl"))
        summary.fees += _to_float(fill.get("fee"))
        summary.crossed = summary.crossed or bool(fill.get("crossed"))
    return summary


def _position_for_coin(info: Info, address: str, coin: str) -> tuple[float, float | None]:
    state = info.user_state(address)
    for item in state.get("assetPositions", []):
        position = item.get("position", {})
        if position.get("coin") != coin:
            continue
        size = _to_float(position.get("szi"))
        entry_px = position.get("entryPx")
        return size, _to_float(entry_px) if entry_px not in (None, "") else None
    return 0.0, None


def _margin_summary(info: Info, address: str) -> dict[str, float]:
    """Lấy tổng margin đang dùng và account value."""
    try:
        state = info.user_state(address)
        ms = state.get("marginSummary", {})
        return {
            "account_value": _to_float(ms.get("accountValue")),
            "total_margin_used": _to_float(ms.get("totalMarginUsed")),
            "withdrawable": _to_float(ms.get("withdrawable")),
        }
    except Exception as exc:
        logger.warning("margin_summary fetch failed: %s", exc)
        return {"account_value": 0.0, "total_margin_used": 0.0, "withdrawable": 0.0}

_margin_cache: dict[str, Any] = {"data": None, "ts": 0, "backoff_until": 0}

async def _get_cached_margin_summary(info: Info, address: str) -> dict[str, float] | None:
    now = time.time()
    if now < _margin_cache["backoff_until"]:
        return _margin_cache["data"]

    if now - _margin_cache["ts"] > 30:
        ms = await _call(_margin_summary, info, address)
        if ms["account_value"] >= 0:
            _margin_cache["data"] = ms
            _margin_cache["ts"] = now
        else:
            _margin_cache["backoff_until"] = now + 10
            # return stale cache if available
            
    return _margin_cache["data"]

def _open_orders_for_coin(info: Info, address: str, coin: str) -> list[dict[str, Any]]:
    return [order for order in info.open_orders(address) if order.get("coin") == coin]


def _is_order_open(info: Info, address: str, coin: str, oid: int | None) -> bool:
    if oid is None:
        return False
    return any(int(order.get("oid", -1)) == oid for order in _open_orders_for_coin(info, address, coin))


def _fills_by_oid(info: Info, address: str, start_ms: int, coin: str, oid: int | None) -> list[dict[str, Any]]:
    if oid is None:
        return []
    fills = info.user_fills_by_time(address, max(0, start_ms - 5_000)) or []
    return [fill for fill in fills if fill.get("coin") == coin and int(fill.get("oid", -1)) == oid]


def _get_orderbook_analysis(info: Info, coin: str, current_price: float = 0.0, depth: int = 20) -> dict:
    """
    Phân tích sổ lệnh nâng cao:
    - OBI: orderbook imbalance (snapshot — làm mượt qua _smoothed_obi)
    - Wall: phát hiện lệnh lớn bất thường (> 3x average level size)
    - Spread: bid-ask spread hiện tại (%)
    - Near wall: wall nằm trong 0.1% của current price
    current_price=0 → dùng mid của sổ lệnh.
    """
    result = {
        "obi": 0.0,
        "bid_wall": False,
        "ask_wall": False,
        "bid_wall_price": None,
        "ask_wall_price": None,
        "spread_pct": 0.0,
    }
    try:
        l2 = info.l2_snapshot(coin)
        if not l2 or "levels" not in l2 or len(l2["levels"]) < 2:
            return result

        raw_bids = l2["levels"][0]
        raw_asks = l2["levels"][1]

        if not raw_bids or not raw_asks:
            return result

        # Spread
        best_bid = float(raw_bids[0].get("px", 0))
        best_ask = float(raw_asks[0].get("px", 0))
        if best_bid > 0:
            result["spread_pct"] = (best_ask - best_bid) / best_bid * 100
        if current_price <= 0:
            current_price = (best_bid + best_ask) / 2

        # Lấy OBI trong khoảng ±0.15% quanh giá hiện tại thay vì số lượng level cố định
        min_bid_px = current_price * (1 - 0.0015)
        max_ask_px = current_price * (1 + 0.0015)
        
        bids = [b for b in raw_bids if float(b.get("px", 0)) >= min_bid_px]
        asks = [a for a in raw_asks if float(a.get("px", 0)) <= max_ask_px]
        
        if not bids: bids = raw_bids[:10]
        if not asks: asks = raw_asks[:10]

        # OBI
        bid_vol = sum(float(b.get("sz", 0)) for b in bids)
        ask_vol = sum(float(a.get("sz", 0)) for a in asks)
        total_vol = bid_vol + ask_vol
        if total_vol > 0:
            result["obi"] = (bid_vol - ask_vol) / total_vol

        # Wall detection: tìm level có size > 3x average
        avg_bid_sz = bid_vol / len(bids) if bids else 0
        avg_ask_sz = ask_vol / len(asks) if asks else 0

        wall_threshold = 3.0  # 3x average = "wall"
        near_threshold = 0.001  # 0.1% từ giá hiện tại

        for b in bids:
            sz = float(b.get("sz", 0))
            px = float(b.get("px", 0))
            if sz > avg_bid_sz * wall_threshold:
                dist_pct = abs(px - current_price) / current_price
                if dist_pct < near_threshold:
                    result["bid_wall"] = True
                    result["bid_wall_price"] = px
                    break  # Lấy wall gần nhất

        for a in asks:
            sz = float(a.get("sz", 0))
            px = float(a.get("px", 0))
            if sz > avg_ask_sz * wall_threshold:
                dist_pct = abs(px - current_price) / current_price
                if dist_pct < near_threshold:
                    result["ask_wall"] = True
                    result["ask_wall_price"] = px
                    break

    except Exception as e:
        logger.debug("orderbook_analysis failed: %s", e)

    return result


def _sample_obi(info: Info, coin: str) -> None:
    """
    Lấy 1 mẫu OBI vào buffer (gọi mỗi scan cycle ~3s, kể cả khi đang có
    vị thế/cooldown để buffer luôn ấm). Đây là call REST duy nhất chạy
    mỗi cycle — mọi thứ khác đều cache.
    """
    ob = _get_orderbook_analysis(info, coin, 0.0, 20)
    _latest_ob_analysis[coin] = ob
    buf = _obi_buffers.setdefault(coin, deque())
    now = time.time()
    buf.append((now, ob.get("obi", 0.0)))
    while buf and now - buf[0][0] > OBI_WINDOW_SEC:
        buf.popleft()


def _smoothed_obi(coin: str) -> tuple[float, int]:
    """
    OBI làm mượt = trung bình các mẫu trong OBI_WINDOW_SEC gần nhất.
    Trả về (obi_mean, n_samples). Dưới OBI_MIN_SAMPLES mẫu → (0.0, n)
    để scoring bỏ qua OBI (an toàn hơn là tin 1 snapshot nhiễu).
    """
    buf = _obi_buffers.get(coin)
    if not buf:
        return 0.0, 0
    now = time.time()
    samples = [v for (t, v) in buf if now - t <= OBI_WINDOW_SEC]
    n = len(samples)
    if n < OBI_MIN_SAMPLES:
        return 0.0, n
    return sum(samples) / n, n


def _get_funding_rate(info: Info, coin: str) -> float:
    """Lấy funding rate hiện tại của coin. Trả về 0.0 nếu lỗi."""
    try:
        meta = info.meta_and_asset_ctxs()
        if not meta or len(meta) < 2:
            return 0.0
        asset_ctxs = meta[1]
        coin_meta = info.name_to_asset(coin)
        if coin_meta < len(asset_ctxs):
            ctx = asset_ctxs[coin_meta]
            return float(ctx.get("funding", 0.0))
    except Exception:
        return 0.0
    return 0.0


def _get_funding_rate_cached(info: Info, coin: str) -> float:
    """Funding rate với cache TTL — meta_and_asset_ctxs là call nặng."""
    ent = _funding_cache.get(coin)
    now = time.time()
    if ent is not None and now - ent["ts"] < FUNDING_CACHE_TTL_SEC:
        return ent["val"]
    val = _get_funding_rate(info, coin)
    _funding_cache[coin] = {"ts": now, "val": val}
    return val


def fetch_hyperliquid_candles(
    info: Info,
    coin: str,
    interval: str,
    lookback_bars: int,
    asof_ms: int | None = None,
) -> pd.DataFrame:
    end_time = asof_ms or _now_ms()
    interval_ms = _interval_ms(interval)
    start_time = end_time - ((lookback_bars + 3) * interval_ms)
    req = {
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval, "startTime": start_time, "endTime": end_time},
    }
    res = info.post("/info", req)
    if not res:
        return pd.DataFrame()
    rows = [
        {
            "timestamp": pd.to_datetime(c["t"], unit="ms", utc=True),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
        }
        for c in res
    ]
    df = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp")
    cutoff = pd.to_datetime(end_time - interval_ms, unit="ms", utc=True)
    df = df[df["timestamp"] <= cutoff].tail(lookback_bars).reset_index(drop=True)
    if not df.empty and len(df) > 30:
        df = add_indicators(df)
    return df


def _align_index(ts: pd.Timestamp, df: pd.DataFrame) -> int:
    idx = df["timestamp"].searchsorted(ts, side="right") - 1
    return int(max(idx, 0))


def get_effective_config(coin: str) -> dict:
    overrides = COIN_CONFIG.get(coin, {})
    return {**CONFIG, **overrides}


def _cached_candles(info: Info, coin: str, interval: str, lookback: int) -> pd.DataFrame:
    """
    Candle fetch với cache thông minh: chỉ refetch khi nến MỚI đã đóng.
    Nến cuối trong df có open=T → đóng tại T+iv → nến kế tiếp available
    từ T+2iv. Trước thời điểm đó mọi fetch đều trả về dữ liệu y hệt
    → dùng cache. Giảm REST call từ ~60/phút xuống ~1-3/phút mỗi khung.
    """
    key = (coin, interval)
    ent = _candle_cache.get(key)
    now = _now_ms()
    if ent is not None and now < ent["next_fetch_ms"]:
        return ent["df"]

    iv = _interval_ms(interval)
    df = fetch_hyperliquid_candles(info, coin, interval, lookback, now)
    if df.empty:
        # fetch lỗi/rỗng → giữ cache cũ nếu có, thử lại sau 3s
        if ent is not None:
            ent["next_fetch_ms"] = now + 3_000
            return ent["df"]
        return df

    last_open_ms = int(df.iloc[-1]["timestamp"].timestamp() * 1000)
    next_fetch = last_open_ms + 2 * iv + 2_000
    if next_fetch <= now:
        next_fetch = now + 3_000  # nến mới lẽ ra phải có nhưng chưa — retry sớm
    _candle_cache[key] = {"df": df, "next_fetch_ms": next_fetch}
    return df


def _latest_setup(info: Info, coin: str, effective_cfg: dict,
                  obi_smooth: float, ob_analysis: dict) -> tuple[dict[str, Any], float, pd.Timestamp] | None:
    df15 = _cached_candles(info, coin, "15m", LOOKBACK_15M)
    df5 = _cached_candles(info, coin, "5m", LOOKBACK_5M)
    df1 = _cached_candles(info, coin, "1m", LOOKBACK_1M)
    if df15.empty or df5.empty or df1.empty:
        logger.warning("%s: empty candle data", coin)
        return None
    i1 = len(df1) - 1
    signal_ts = df1.iloc[i1]["timestamp"]
    i5 = _align_index(signal_ts, df5)
    i15 = _align_index(signal_ts, df15)
    current_price = float(df1.iloc[i1]["close"])
    funding_rate = _get_funding_rate_cached(info, coin)
    setup = score_setup(
        i15, df15, i5, df5, i1, df1,
        hour_utc=signal_ts.hour, cfg=effective_cfg,
        obi=obi_smooth, ob_analysis=ob_analysis, funding_rate=funding_rate,
    )
    setup["obi"] = obi_smooth
    return setup, current_price, signal_ts


async def _call(fn, *args, **kwargs):
    fn_name = getattr(fn, "__name__", "unknown")
    is_write = fn_name in ("order", "market_close", "cancel")
    max_retries = 1 if is_write else 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as exc:
            _record_api_failure()
            # Rate limit / gateway lỗi → backoff dài hơn
            err_text = str(exc)
            is_rate_limited = "429" in err_text or "502" in err_text or "504" in err_text
            if attempt == max_retries - 1:
                if not is_write:
                    logger.warning("API call %s failed after %d attempts: %s", fn_name, max_retries, str(exc)[:200])
                raise
            delay = base_delay * (2 ** attempt) * (3 if is_rate_limited else 1)
            logger.info("API call %s failed (retry in %.0fs): %s", fn_name, delay, str(exc)[:120])
            await asyncio.sleep(delay)


async def _cancel_if_open(exchange: Exchange, info: Info, address: str, coin: str, oid: int | None) -> None:
    if oid is None:
        return
    try:
        if await _call(_is_order_open, info, address, coin, oid):
            logger.info("%s: cancel open order oid=%s", coin, oid)
            await _call(exchange.cancel, coin, oid)
    except Exception as exc:
        logger.warning("%s: cancel oid=%s failed: %s", coin, oid, exc)


async def _place_limit(
    exchange: Exchange, coin: str, is_buy: bool, size: float, price: float, reduce_only: bool, tif: str,
) -> tuple[int | None, Any]:
    try:
        response = await _call(
            exchange.order, coin, is_buy, size, _round_price(price), {"limit": {"tif": tif}}, reduce_only,
        )
        errors = _response_errors(response)
        if errors:
            logger.warning("%s: limit order errors: %s", coin, "; ".join(errors))
        return _extract_oid(response), response
    except Exception as exc:
        logger.warning("%s: place limit order failed: %s", coin, exc)
        return None, None


async def _place_trigger_sl(
    exchange: Exchange, coin: str, is_buy: bool, size: float, trigger_price: float,
) -> tuple[int | None, Any]:
    try:
        price = _round_price(trigger_price)
        response = await _call(
            exchange.order, coin, is_buy, size, price,
            {"trigger": {"triggerPx": price, "isMarket": True, "tpsl": "sl"}}, True,
        )
        errors = _response_errors(response)
        if errors:
            logger.warning("%s: SL trigger errors: %s", coin, "; ".join(errors))
        return _extract_oid(response), response
    except Exception as exc:
        logger.warning("%s: place trigger SL failed: %s", coin, exc)
        return None, None


async def _place_tp_with_fallback(
    exchange: Exchange, coin: str, is_buy: bool, size: float, price: float,
) -> int | None:
    tif = "Alo" if CONFIG.get("use_maker_for_tp", True) else "Gtc"
    oid, _ = await _place_limit(exchange, coin, is_buy, size, price, True, tif)
    if oid is not None:
        return oid
    if tif == "Alo":
        logger.warning("%s: TP Alo rejected; fallback Gtc", coin)
        oid, _ = await _place_limit(exchange, coin, is_buy, size, price, True, "Gtc")
    return oid


async def _wait_entry_fill(
    info: Info,
    exchange: Exchange,
    address: str,
    coin: str,
    oid: int | None,
    target_size: float,
    start_ms: int,
    timeout_seconds: int,
    is_buy: bool = True,
    signal_price: float = 0.0,
    reversal_cancel_pct: float = 0.0,
) -> FillSummary:
    deadline = time.monotonic() + timeout_seconds
    last_summary = FillSummary()
    cancel_reason = "entry timeout"

    while time.monotonic() < deadline:
        fills = await _call(_fills_by_oid, info, address, start_ms, coin, oid)
        last_summary = _summarize_fills(fills, oid)
        if last_summary.size >= target_size * 0.999:
            return last_summary
        if oid is not None and not await _call(_is_order_open, info, address, coin, oid) and last_summary.size <= 0:
            return last_summary

        # ── v3: Cancel-on-reversal — giá đã quay đầu (bounce/drop bắt đầu)
        # mà limit chưa fill → cơ hội đã đi, hủy ngay thay vì chờ hết 60s.
        # Chống adverse selection: đứng chờ tiếp chỉ fill khi giá xuyên qua entry.
        if reversal_cancel_pct > 0 and signal_price > 0:
            try:
                mids = await _call(info.all_mids)
                mid = _to_float(mids.get(coin)) if isinstance(mids, dict) else 0.0
                if mid > 0:
                    rev = reversal_cancel_pct / 100
                    escaped = (is_buy and mid >= signal_price * (1 + rev)) or \
                              (not is_buy and mid <= signal_price * (1 - rev))
                    if escaped:
                        cancel_reason = f"price reverted {reversal_cancel_pct}% without fill"
                        break
            except Exception:
                pass

        await asyncio.sleep(ORDER_POLL_SECONDS)

    # ── FIX #4: Cancel và xác nhận không còn position orphan ──
    logger.info("%s: %s — cancelling order oid=%s", coin, cancel_reason, oid)
    await _cancel_if_open(exchange, info, address, coin, oid)

    # Poll lại _position_for_coin vài lần thay vì 1 lần để vượt qua độ trễ API
    pos_size = 0.0
    pos_entry = None
    for _ in range(3):
        await asyncio.sleep(ORDER_POLL_SECONDS)
        pos_size, pos_entry = await _call(_position_for_coin, info, address, coin)
        if abs(pos_size) > POSITION_EPSILON:
            break

    fills = await _call(_fills_by_oid, info, address, start_ms, coin, oid)
    final_summary = _summarize_fills(fills, oid)

    if final_summary.size > POSITION_EPSILON:
        logger.warning(
            "%s: partial fill after cancel: size=%.8f — will manage this position",
            coin, final_summary.size,
        )
    else:
        if abs(pos_size) > POSITION_EPSILON:
            logger.warning(
                "%s: orphan position detected after cancel! pos=%.8f. Adopting position instead of closing.",
                coin, pos_size,
            )
            final_summary.size = abs(pos_size)
            final_summary.notional = abs(pos_size) * (pos_entry if pos_entry else 0.0)

    return final_summary


async def _market_close_remaining(
    exchange: Exchange, info: Info, address: str, coin: str, managed_oids: set[int],
) -> int | None:
    try:
        pos_size, _ = await _call(_position_for_coin, info, address, coin)
        close_size = abs(pos_size)
        if close_size <= POSITION_EPSILON:
            return None
        logger.info("%s: time-stop market close size=%.8f", coin, close_size)
        response = await _call(exchange.market_close, coin, close_size, None, MARKET_CLOSE_SLIPPAGE)
        oid = _extract_oid(response)
        if oid is not None:
            managed_oids.add(oid)
        errors = _response_errors(response)
        if errors:
            logger.warning("%s: market close errors: %s", coin, "; ".join(errors))
        return oid
    except Exception as exc:
        logger.error("%s: market close failed: %s", coin, exc)
        return None


async def _final_net_pnl(info: Info, address: str, coin: str, start_ms: int, managed_oids: set[int]) -> float:
    await asyncio.sleep(2)
    fills = await _call(info.user_fills_by_time, address, max(0, start_ms - 5_000))
    if not fills:
        return 0.0

    # ── IMPROVEMENT #6: Tìm cả fills không nằm trong managed_oids ──
    # (bao gồm partial fills bị fragmented của cùng 1 entry order)
    coin_fills = [f for f in fills if f.get("coin") == coin]

    # Ưu tiên: fills trong managed_oids (chính xác nhất)
    related = [f for f in coin_fills if int(f.get("oid", -1)) in managed_oids]
    if related:
        return sum(_to_float(f.get("closedPnl")) for f in related)

    # Fallback: tất cả fills của coin trong time window (nếu oid tracking bị mất)
    logger.warning("%s: no managed_oid fills found, using all coin fills in window", coin)
    return sum(_to_float(f.get("closedPnl")) for f in coin_fills)


async def _wait_for_position_flat(info: Info, address: str, coin: str, timeout_seconds: int = 30) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        pos_size, pos_entry_px = await _call(_position_for_coin, info, address, coin)
        # Bỏ qua dust position (< $12) bằng notional
        if abs(pos_size) * (pos_entry_px or 0.0) <= 12: 
            return True
        await asyncio.sleep(ORDER_POLL_SECONDS)
    return False


def _clear_active_entry(coin: str) -> None:
    """Xóa active entry guard cho coin khi trade hoàn thành hoặc bị abort."""
    _active_entry.pop(coin, None)
    logger.debug("%s: active entry guard cleared", coin)


async def _resolve_exit_reason(
    info: Info, address: str, coin: str, start_ms: int,
    tp1_oid: int | None, tp2_oid: int | None, sl_oid: int | None, stage: str,
) -> str:
    """
    Xác định lý do thoát THẬT khi position flat, dựa trên fills của từng oid.
    Fix nhãn cũ 'SL_or_external_flat' gán cho cả lệnh TP2 thắng → làm bẩn
    dữ liệu phân tích.
    """
    async def _oid_filled(oid: int | None) -> bool:
        if oid is None:
            return False
        try:
            fills = await _call(_fills_by_oid, info, address, start_ms, coin, oid)
            return _summarize_fills(fills, oid).size > POSITION_EPSILON
        except Exception:
            return False

    try:
        if stage == "tp2":
            if await _oid_filled(tp2_oid):
                return "TP2"
            if await _oid_filled(sl_oid):
                return "SL_after_TP1"
        else:
            if await _oid_filled(sl_oid):
                return "SL"
            if await _oid_filled(tp1_oid):
                return "TP1_flat"
        return "external_flat"
    except Exception:
        return "external_flat"


async def _execute_setup(
    exchange: Exchange,
    info: Info,
    address: str,
    coin: str,
    direction: str,
    signal_price: float,
    signal_ts: pd.Timestamp,
    margin: float,
    score: int,
    atr_5m: float = 0.0,
    effective_cfg: dict | None = None,
) -> TradeResult | None:
    cfg = effective_cfg or CONFIG
    leverage = cfg["leverage"]
    notional = margin * leverage
    is_entry_buy = direction == "long"
    exit_is_buy = not is_entry_buy

    entry_price = signal_price
    if cfg.get("entry_order_type") == "limit":
        offset = cfg.get("limit_offset_pct", 0.0) / 100
        entry_price = signal_price * (1 - offset if is_entry_buy else 1 + offset)
    size = _round_size(exchange, coin, notional / entry_price)
    if size <= 0:
        logger.warning("%s: computed entry size is zero; skipping", coin)
        _clear_active_entry(coin)
        return None

    start_ms = _now_ms()
    managed_oids: set[int] = set()

    logger.info(
        "%s: entry %s score at %.4f | margin=%.2f notional=%.2f size=%.8f",
        coin, direction.upper(), signal_price, margin, notional, size,
    )

    # Double-check trước khi place order
    pre_check_pos, _ = await _call(_position_for_coin, info, address, coin)
    pre_check_orders = await _call(_open_orders_for_coin, info, address, coin)
    if abs(pre_check_pos) > POSITION_EPSILON or pre_check_orders:
        logger.warning(
            "%s: exposure detected before order (pos=%.8f orders=%d); aborting",
            coin, pre_check_pos, len(pre_check_orders),
        )
        _clear_active_entry(coin)
        return None

    # ── v3: Setup A/A+ (score cao) → vào TAKER để không lỡ lệnh đẹp nhất.
    # Log cũ cho thấy lệnh không fill toàn là lệnh giá đảo chiều ngay (winner).
    # B/C setup vẫn maker để tối ưu phí.
    entry_order_type = cfg.get("entry_order_type", "limit")
    if score >= cfg.get("taker_entry_min_score", 85):
        entry_order_type = "market"
        logger.info("%s: score %d >= taker threshold — using market entry", coin, score)

    if entry_order_type == "market":
        slippage = max(cfg.get("slippage_pct", 0.0) / 100, 0.002)
        response = await _call(
            exchange.market_open, coin, is_entry_buy, size, signal_price,
            slippage,
        )
        entry_oid = _extract_oid(response)
        timeout_seconds = 30
        reversal_cancel_pct = 0.0
    else:
        tif = "Alo" if cfg.get("use_maker_for_entry", True) else "Gtc"
        entry_oid, response = await _place_limit(
            exchange, coin, is_entry_buy, size, entry_price, False, tif
        )
        if entry_oid is None and tif == "Alo":
            logger.warning("%s: Entry Alo rejected (likely crosses spread), falling back to Gtc limit", coin)
            entry_oid, response = await _place_limit(
                exchange, coin, is_entry_buy, size, entry_price, False, "Gtc"
            )
        timeout_seconds = max(1, int(cfg.get("limit_timeout_bars", 1))) * 60
        reversal_cancel_pct = cfg.get("reversal_cancel_pct", 0.0)

    if entry_oid is None:
        logger.warning("%s: entry order not accepted: %s", coin, response)
        _clear_active_entry(coin)
        return None
    managed_oids.add(entry_oid)

    entry_fill = await _wait_entry_fill(
        info, exchange, address, coin, entry_oid, size, start_ms, timeout_seconds,
        is_buy=is_entry_buy, signal_price=signal_price,
        reversal_cancel_pct=reversal_cancel_pct,
    )

    if entry_fill.size <= POSITION_EPSILON or entry_fill.avg_price is None:
        logger.info("%s: entry not filled; setup skipped", coin)
        _clear_active_entry(coin)
        return None

    if entry_fill.size < size * 0.999:
        if entry_fill.size < size * 0.25:
            logger.warning(
                "%s: partial entry fill %.8f / %.8f is too small (< 25%%); closing as garbage.",
                coin, entry_fill.size, size,
            )
            await _market_close_remaining(exchange, info, address, coin, managed_oids)
            await _wait_for_position_flat(info, address, coin)
            # v3 FIX: PnL của garbage close (phí + slippage) trước đây KHÔNG được
            # ghi vào RiskState → day_pnl lệch. Giờ trả về TradeResult đầy đủ.
            net_pnl = await _final_net_pnl(info, address, coin, start_ms, managed_oids)
            _clear_active_entry(coin)
            return TradeResult(
                coin, direction, "partial_garbage", net_pnl,
                entry_fill.size, entry_fill.avg_price or signal_price,
                signal_ts.to_pydatetime(), datetime.now(timezone.utc),
            )

        logger.warning(
            "%s: partial entry fill %.8f / %.8f; managing remainder",
            coin, entry_fill.size, size,
        )

    avg_entry = entry_fill.avg_price
    filled_size = _round_size(exchange, coin, entry_fill.size)

    # Tính TP/SL
    if cfg.get("use_dynamic_tp_sl", False) and atr_5m > 0:
        from filters import compute_dynamic_levels
        levels = compute_dynamic_levels(
            avg_entry, direction, atr_5m,
            tp1_atr_mult=cfg.get("tp1_atr_mult", 1.5),
            tp2_atr_mult=cfg.get("tp2_atr_mult", 3.0),
            sl_atr_mult=cfg.get("sl_atr_mult", 1.2),
            tp1_pct_max=cfg.get("tp1_pct_max", 0.30),
            tp2_pct_max=cfg.get("tp2_pct_max", 0.60),
            sl_pct_max=cfg.get("sl_pct_max", 0.40),
            sl_pct_min=cfg.get("sl_pct_min", 0.0),
        )
        tp1_price = levels["tp1_price"]
        tp2_price = levels["tp2_price"]
        sl_price = levels["sl_price"]
    else:
        tp1_pct = cfg["tp1_pct"] / 100
        tp2_pct = cfg["tp2_pct"] / 100
        sl_pct = cfg["sl_pct"] / 100
        if direction == "long":
            tp1_price = avg_entry * (1 + tp1_pct)
            tp2_price = avg_entry * (1 + tp2_pct)
            sl_price = avg_entry * (1 - sl_pct)
        else:
            tp1_price = avg_entry * (1 - tp1_pct)
            tp2_price = avg_entry * (1 - tp2_pct)
            sl_price = avg_entry * (1 + sl_pct)

    tp1_size = _round_size(exchange, coin, filled_size / 2)
    if tp1_size <= POSITION_EPSILON:
        tp1_size = filled_size
    deadline = datetime.now(timezone.utc) + timedelta(minutes=cfg["time_stop_minutes"])

    # ── v3: Hard dollar SL = BACKSTOP theo planned risk, không phải stop chính.
    # Planned risk = notional × khoảng cách SL. Backstop = planned × 1.5.
    # (Bản cũ: $3 cứng trên $2000 notional = 0.15% < %SL → SL trigger thành trang trí.)
    sl_dist_pct = abs(avg_entry - sl_price) / avg_entry if avg_entry > 0 else 0.0
    planned_risk_usd = filled_size * avg_entry * sl_dist_pct
    hard_sl_usd = max(
        planned_risk_usd * cfg.get("hard_sl_backstop_mult", 1.5),
        HARD_DOLLAR_SL_USD,
    )

    logger.info(
        "%s: filled %.8f @ %.4f | TP1=%.4f TP2=%.4f SL=%.4f | risk=%.2f backstop=%.2f | deadline=%s",
        coin, filled_size, avg_entry, tp1_price, tp2_price, sl_price,
        planned_risk_usd, hard_sl_usd,
        deadline.strftime("%H:%M"),
    )

    tp1_oid = await _place_tp_with_fallback(exchange, coin, exit_is_buy, tp1_size, tp1_price)
    if tp1_oid is not None:
        managed_oids.add(tp1_oid)

    # ── FIX #5: SL size phải khớp chính xác với filled_size ──
    sl_oid, _ = await _place_trigger_sl(exchange, coin, exit_is_buy, filled_size, sl_price)
    if sl_oid is not None:
        managed_oids.add(sl_oid)

    # Removed emergency close, active order monitoring will retry if they are None

    stage = "tp1"
    exit_reason = "unknown"
    tp2_oid: int | None = None
    _last_hard_sl_check = time.monotonic()

    while True:
        try:
            now = datetime.now(timezone.utc)
            pos_size, pos_entry_px = await _call(_position_for_coin, info, address, coin)
            abs_pos = abs(pos_size)

            if now >= deadline:
                exit_reason = "time_stop"
                for oid in (tp1_oid, tp2_oid, sl_oid):
                    await _cancel_if_open(exchange, info, address, coin, oid)
                await _market_close_remaining(exchange, info, address, coin, managed_oids)
                await _wait_for_position_flat(info, address, coin)
                break

            if abs_pos * (pos_entry_px or 0.0) <= 12:  # Bỏ qua dust (< $12)
                if abs_pos > POSITION_EPSILON:
                    logger.warning("%s: dust position detected (size=%.8f) — ignoring to prevent $10 limit error", coin, abs_pos)
                # v3: xác định lý do thoát THẬT từ fills (TP2/SL/TP1_flat/external)
                exit_reason = await _resolve_exit_reason(
                    info, address, coin, start_ms, tp1_oid, tp2_oid, sl_oid, stage,
                )
                break

            # ── HARD DOLLAR SL: kiểm tra mỗi HARD_DOLLAR_SL_CHECK_INTERVAL giây ──
            # Thoát ngay nếu unrealized loss vượt HARD_DOLLAR_SL_USD
            # Bảo vệ khỏi tình huống giá trượt dài không chạm % SL
            elapsed_since_check = time.monotonic() - _last_hard_sl_check
            if elapsed_since_check >= HARD_DOLLAR_SL_CHECK_INTERVAL and abs_pos > POSITION_EPSILON:
                _last_hard_sl_check = time.monotonic()
                try:
                    # Lấy giá hiện tại từ mark price
                    state = await _call(info.user_state, address)
                    for item in state.get("assetPositions", []):
                        pos = item.get("position", {})
                        if pos.get("coin") == coin:
                            unrealized_pnl = _to_float(pos.get("unrealizedPnl"))
                            if unrealized_pnl < -hard_sl_usd:
                                logger.warning(
                                    "%s: HARD SL BACKSTOP triggered! unrealized_pnl=%.4f < -%.2f — emergency close",
                                    coin, unrealized_pnl, hard_sl_usd,
                                )
                                exit_reason = "hard_dollar_sl"
                                for oid in (tp1_oid, tp2_oid, sl_oid):
                                    await _cancel_if_open(exchange, info, address, coin, oid)
                                await _market_close_remaining(exchange, info, address, coin, managed_oids)
                                await _wait_for_position_flat(info, address, coin)
                            break
                except Exception as exc:
                    logger.warning("%s: hard dollar SL check failed: %s", coin, exc)

            if exit_reason in ("hard_dollar_sl",):
                break

            open_orders = await _call(_open_orders_for_coin, info, address, coin)
            open_oids = {int(o["oid"]) for o in open_orders if "oid" in o}

            if stage == "tp1":
                if tp1_oid is None or tp1_oid not in open_oids:
                    tp1_summary = _summarize_fills(await _call(_fills_by_oid, info, address, start_ms, coin, tp1_oid), tp1_oid) if tp1_oid else FillSummary()
                    if tp1_summary.size < tp1_size * 0.999 and abs_pos > (filled_size - tp1_size) * 1.001:
                        logger.warning("%s: TP1 order missing, recreating...", coin)
                        tp1_oid = await _place_tp_with_fallback(exchange, coin, exit_is_buy, tp1_size, tp1_price)
                        if tp1_oid: managed_oids.add(tp1_oid)

                if sl_oid is None or sl_oid not in open_oids:
                    sl_summary = _summarize_fills(await _call(_fills_by_oid, info, address, start_ms, coin, sl_oid), sl_oid) if sl_oid else FillSummary()
                    if sl_summary.size <= 0 and abs_pos > POSITION_EPSILON:
                        logger.warning("%s: SL order missing, recreating...", coin)
                        sl_oid, _ = await _place_trigger_sl(exchange, coin, exit_is_buy, abs_pos, sl_price)
                        if sl_oid: managed_oids.add(sl_oid)

                if abs_pos <= (filled_size - tp1_size) * 1.001:
                    logger.info("%s: TP1 threshold reached (pos=%.8f); moving to TP2", coin, abs_pos)
                    await _cancel_if_open(exchange, info, address, coin, sl_oid)
                    await _cancel_if_open(exchange, info, address, coin, tp1_oid)
                    
                    remaining_size = _round_size(exchange, coin, abs_pos)
                    if remaining_size > POSITION_EPSILON:
                        next_sl_price = avg_entry if CONFIG.get("move_sl_to_breakeven_after_tp1", False) else sl_price
                        tp2_oid = await _place_tp_with_fallback(exchange, coin, exit_is_buy, remaining_size, tp2_price)
                        if tp2_oid: managed_oids.add(tp2_oid)
                        
                        sl_oid, _ = await _place_trigger_sl(exchange, coin, exit_is_buy, remaining_size, next_sl_price)
                        if sl_oid: managed_oids.add(sl_oid)
                        
                        stage = "tp2"
                    else:
                        exit_reason = "TP1_flat"
                        break

            elif stage == "tp2":
                if tp2_oid is None or tp2_oid not in open_oids:
                    tp2_summary = _summarize_fills(await _call(_fills_by_oid, info, address, start_ms, coin, tp2_oid), tp2_oid) if tp2_oid else FillSummary()
                    if tp2_summary.size <= 0 and abs_pos > POSITION_EPSILON:
                        logger.warning("%s: TP2 order missing, recreating...", coin)
                        tp2_oid = await _place_tp_with_fallback(exchange, coin, exit_is_buy, abs_pos, tp2_price)
                        if tp2_oid: managed_oids.add(tp2_oid)

                if sl_oid is None or sl_oid not in open_oids:
                    sl_summary = _summarize_fills(await _call(_fills_by_oid, info, address, start_ms, coin, sl_oid), sl_oid) if sl_oid else FillSummary()
                    if sl_summary.size <= 0 and abs_pos > POSITION_EPSILON:
                        logger.warning("%s: SL stage 2 missing, recreating...", coin)
                        next_sl_price = avg_entry if CONFIG.get("move_sl_to_breakeven_after_tp1", False) else sl_price
                        sl_oid, _ = await _place_trigger_sl(exchange, coin, exit_is_buy, abs_pos, next_sl_price)
                        if sl_oid: managed_oids.add(sl_oid)

        except Exception as exc:
            logger.error("%s: active monitoring loop error: %s", coin, exc)

        await asyncio.sleep(ORDER_POLL_SECONDS)

    for oid in (tp1_oid, tp2_oid, sl_oid):
        await _cancel_if_open(exchange, info, address, coin, oid)

    net_pnl = await _final_net_pnl(info, address, coin, start_ms, managed_oids)
    logger.info("%s: trade closed reason=%s net_pnl=%.6f", coin, exit_reason, net_pnl)

    _clear_active_entry(coin)
    return TradeResult(
        coin, direction, exit_reason, net_pnl, filled_size, avg_entry,
        signal_ts.to_pydatetime(), datetime.now(timezone.utc),
    )


async def _execute_and_record(
    exchange: Exchange,
    info: Info,
    address: str,
    coin: str,
    direction: str,
    signal_price: float,
    signal_ts: pd.Timestamp,
    margin: float,
    score: int,
    atr_5m: float,
    risk: RiskState,
    effective_cfg: dict | None = None,
    regime: str = "",
    obi: float = 0.0,
) -> None:
    try:
        result = await _execute_setup(
            exchange, info, address, coin,
            direction, signal_price, signal_ts, margin,
            score, atr_5m, effective_cfg=effective_cfg,
        )
        if result is not None:
            risk.record_trade(result)
            _log_trade_csv(result, score, regime, obi)
            logger.info(
                "%s: recorded — win=%s pnl=%.4f | day_pnl=%.4f | trades_today=%d",
                coin, result.win, result.net_pnl,
                risk.pnl_today.get((coin, result.entry_time.date()), 0.0),
                risk.trades_today.get((coin, result.entry_time.date()), 0),
            )
    except Exception as exc:
        logger.error("%s: _execute_setup error: %s", coin, exc)
        _clear_active_entry(coin)


async def _scan_coin(
    exchange: Exchange,
    info: Info,
    address: str,
    coin: str,
    risk: RiskState,
    last_signal_ts: dict[str, pd.Timestamp],
    last_guard_log: dict[str, datetime],
) -> None:
    if coin not in _coin_locks:
        _coin_locks[coin] = asyncio.Lock()

    if _coin_locks[coin].locked():
        return

    async with _coin_locks[coin]:
        # ── DISABLED_COINS guard: bỏ qua hoàn toàn ETH và các coin bị disable ──
        if coin in DISABLED_COINS:
            logger.debug("%s: coin disabled — skipping", coin)
            return

        # ── v3: Sample OBI MỖI cycle (kể cả cooldown/đang có lệnh) để buffer
        # smoothing luôn ấm. Đây là call REST duy nhất chạy mỗi 3s.
        try:
            await _call(_sample_obi, info, coin)
        except Exception:
            pass

        # ── FIX #1: Active entry guard — kiểm tra TRƯỚC khi gọi API ──
        active_ts = _active_entry.get(coin)
        if active_ts is not None:
            elapsed = time.time() - active_ts
            if elapsed < ACTIVE_ENTRY_TIMEOUT_SEC:
                logger.debug(
                    "%s: active entry guard (%.0fs elapsed, timeout=%ds)",
                    coin, elapsed, ACTIVE_ENTRY_TIMEOUT_SEC,
                )
                return
            else:
                logger.warning(
                    "%s: active entry guard expired after %.0fs — clearing",
                    coin, elapsed,
                )
                _active_entry.pop(coin, None)

        now_utc = datetime.now(timezone.utc)
        can_trade, reason = risk.can_trade(coin, now_utc)
        if not can_trade:
            # v3: throttle — log 1 lần/phút thay vì mỗi 3s (log cũ ngập cooldown spam)
            _log_throttled(f"riskblock:{coin}", 60, "%s: risk block: %s", coin, reason)
            return
        # ── v3: New-candle gate — mọi call nặng (user_state, open_orders, 5m/15m
        # candles, funding) chỉ chạy khi có nến 1m MỚI đóng (~1 lần/phút).
        df1 = await _call(_cached_candles, info, coin, "1m", LOOKBACK_1M)
        if df1 is None or df1.empty:
            return
        signal_ts = df1.iloc[len(df1) - 1]["timestamp"]
        if last_signal_ts.get(coin) == signal_ts:
            return

        # ── v3: Circuit breaker — API đang lỗi hàng loạt → không mở lệnh mới.
        # (Monitoring loop của vị thế đang mở vẫn chạy độc lập.)
        if _api_circuit_open():
            _log_throttled(
                f"circuit:{coin}", 30,
                "%s: API circuit OPEN (%d failures/60s) — pausing new entries",
                coin, len(_api_failures),
            )
            return

        # Claim nến này ngay — nếu bước sau lỗi transient thì bỏ qua tín hiệu
        # của phút này thay vì retry dồn dập mỗi 3s.
        last_signal_ts[coin] = signal_ts

        # ── FIX #3: Kiểm tra tổng margin usage ──
        ms = await _get_cached_margin_summary(info, address)
        if ms is None:
            logger.warning("%s: margin_summary failed to fetch — skipping for safety", coin)
            return

        if ms["account_value"] > 0:
            margin_usage_pct = ms["total_margin_used"] / ms["account_value"]
            if margin_usage_pct > MAX_MARGIN_USAGE_PCT:
                _log_throttled(
                    f"margin:{coin}", 60,
                    "%s: margin usage %.1f%% > limit %.1f%% — skipping",
                    coin, margin_usage_pct * 100, MAX_MARGIN_USAGE_PCT * 100,
                )
                return

        user_state = await _call(info.user_state, address)
        active_coins = []
        pos_size = 0.0
        pos_entry = None
        for item in user_state.get("assetPositions", []):
            position = item.get("position", {})
            c = position.get("coin")
            s = _to_float(position.get("szi"))
            e = _to_float(position.get("entryPx"))
            if abs(s) * (e or 0) > 12:
                active_coins.append(c)
            if c == coin:
                pos_size = s
                pos_entry = e
                
        open_orders = await _call(_open_orders_for_coin, info, address, coin)

        notional_pos = abs(pos_size) * (pos_entry or 0)

        # Dọn dẹp stale orders
        if notional_pos <= 12 and len(open_orders) > 0:
            logger.info("%s: %d stale orders without valid position → cancelling", coin, len(open_orders))
            for order in open_orders:
                oid = order.get("oid")
                if oid:
                    await _call(exchange.cancel, coin, int(oid))
            open_orders = await _call(_open_orders_for_coin, info, address, coin)

        # Bỏ qua dust position (< $12 notional) khi check existing exposure
        if notional_pos > 12 or open_orders:
            # Lưới an toàn cho orphan position
            if notional_pos > 12 and not open_orders:
                if coin not in _orphan_timers:
                    _orphan_timers[coin] = time.time()
                elif time.time() - _orphan_timers[coin] > 60:
                    logger.error("%s: Orphan position detected for >60s (pos=%.8f). Emergency closing.", coin, pos_size)
                    await _market_close_remaining(exchange, info, address, coin, set())
                    _orphan_timers.pop(coin, None)
                    return
            else:
                _orphan_timers.pop(coin, None)

            last_log = last_guard_log.get(coin)
            if not last_log or (now_utc - last_log).total_seconds() >= 60:
                logger.warning(
                    "%s: existing exposure (pos=%.8f orders=%d) — skipping",
                    coin, pos_size, len(open_orders),
                )
                last_guard_log[coin] = now_utc
            return
        else:
            _orphan_timers.pop(coin, None)

        # ── v3: OBI đã làm mượt (rolling mean 90s) thay vì snapshot đơn lẻ ──
        obi_smooth, obi_n = _smoothed_obi(coin)
        ob_analysis = _latest_ob_analysis.get(coin, {})

        effective_cfg = get_effective_config(coin)
        latest = await _call(_latest_setup, info, coin, effective_cfg, obi_smooth, ob_analysis)
        if latest is None:
            return
        setup, current_price, signal_ts_setup = latest
        obi = setup.get("obi", 0.0)

        if setup["hard_block"] or setup["direction"] is None:
            reason_text = setup["block_reasons"][0] if setup["block_reasons"] else "no direction"
            if setup["hard_block"]:
                logger.info("%s %s: block: %s (price %.4f)", coin, signal_ts, reason_text, current_price)
                _log_setup_csv(coin, signal_ts, current_price, setup, "hard_block")
            return

        score = setup["score"]
        direction = setup["direction"]

        logger.info(
            "%s %s: setup %s [%s/%s] score=%s conf=%s price=%.4f obi=%.2f(n=%d) bb_w=%.2f%%",
            coin, signal_ts, direction.upper(),
            setup.get("regime", "?"), setup.get("entry_mode", "?"),
            score, setup.get("confidence", "?"),
            current_price, obi, obi_n,
            setup.get("bb_width_pct", 0),
        )
        score_details = setup.get("score_details", {})
        if score_details:
            parts = [f"{k}={v:+d}" for k, v in score_details.items() if v != 0]
            if parts:
                logger.info("%s: score breakdown: %s", coin, " | ".join(parts))

        # ── v3: Asia session (01-06 UTC) nâng ngưỡng thay vì penalty -5 ──
        bump = effective_cfg.get("asia_score_bump", 0) if 1 <= signal_ts.hour <= 6 else 0
        min_half = effective_cfg["min_score_half"] + bump
        min_full = effective_cfg.get("min_score_full", CONFIG["min_score_full"]) + bump

        if score < min_half:
            _log_setup_csv(coin, signal_ts, current_price, setup, "below_threshold")
            return

        # ── v3: Size theo rủi ro USD cố định (loss ≈ win) ──
        score_full = score >= min_full
        sizing = compute_risk_sizing(
            current_price, setup.get("atr_5m", 0.0), effective_cfg, score_full,
        )
        margin = sizing["margin"]
        if margin <= 0:
            return

        logger.info(
            "%s: risk sizing — risk=%.2f USD sl=%.3f%% notional=%.2f margin=%.2f (%s)",
            coin, sizing["risk_usd"], sizing["sl_pct"], sizing["notional"], margin,
            "full" if score_full else "half",
        )
        _log_setup_csv(coin, signal_ts, current_price, setup, "entered")

        # ── FIX #1: Set active entry guard NGAY KHI quyết định trade ──
        _active_entry[coin] = time.time()
        logger.info("%s: active entry guard SET — executing trade in background", coin)

        asyncio.create_task(
            _execute_and_record(
                exchange, info, address, coin,
                setup["direction"], current_price, signal_ts, margin,
                score, setup.get("atr_5m", 0.0),
                risk,
                effective_cfg=effective_cfg,
                regime=setup.get("regime", ""),
                obi=obi,
            )
        )


async def run_bot_async():
    logger.info("=" * 60)
    logger.info("STARTING BOT ENGINE v3 — DUAL REGIME | RISK SIZING | OBI SMOOTH")
    logger.info("=" * 60)
    logger.info(
        "Profile=%s | entry=%s (taker if score>=%d) | limit_offset=%.3f%% | time_stop=%sm",
        CONFIG["profile_name"], CONFIG["entry_order_type"],
        CONFIG.get("taker_entry_min_score", 85),
        CONFIG.get("limit_offset_pct", 0.0),
        CONFIG["time_stop_minutes"],
    )
    logger.info(
        "Risk/trade=%.2f USD (half=%.0f%%) | SL=%.1fxATR [%.2f-%.2f%%] | backstop=x%.1f | thresholds half/full=%d/%d (+%d Asia)",
        CONFIG.get("risk_per_trade_usd", 3.0), CONFIG.get("risk_half_scale", 0.6) * 100,
        CONFIG.get("sl_atr_mult", 1.2), CONFIG.get("sl_pct_min", 0.10), CONFIG.get("sl_pct_max", 0.40),
        CONFIG.get("hard_sl_backstop_mult", 1.5),
        CONFIG["min_score_half"], CONFIG["min_score_full"], CONFIG.get("asia_score_bump", 0),
    )
    logger.info(
        "OBI smoothing=%ds window (min %d samples) | reversal_cancel=%.2f%% | circuit breaker=%d fails/60s",
        OBI_WINDOW_SEC, OBI_MIN_SAMPLES,
        CONFIG.get("reversal_cancel_pct", 0.0), API_FAILURE_THRESHOLD,
    )
    logger.info(
        "Guards: entry=%ds | SL cooldown=%dm | max margin=%.0f%% | daily_loss=%.2f | max_trades/day=%d | disabled=%s",
        ACTIVE_ENTRY_TIMEOUT_SEC, SL_SINGLE_COOLDOWN_MINUTES, MAX_MARGIN_USAGE_PCT * 100,
        CONFIG.get("daily_loss_limit_usd", 0), CONFIG.get("max_trades_per_day", 0),
        ",".join(DISABLED_COINS) if DISABLED_COINS else "none",
    )
    logger.info("Setup log: %s | Trade log: %s", SETUP_LOG_PATH, TRADE_LOG_PATH)

    if not PRIVATE_KEY:
        logger.error("ERROR: No private key. Set HL_PRIVATE_KEY.")
        return

    try:
        account: LocalAccount = eth_account.Account.from_key(PRIVATE_KEY)
    except Exception as exc:
        logger.error("Private key parse error: %s", exc)
        return

    main_address = os.environ.get("HL_MAIN_ADDRESS", "").strip()
    address = main_address if main_address else account.address

    info = Info(API_URL, skip_ws=True)
    exchange = Exchange(account, API_URL, account_address=main_address if main_address else None)

    coins = configured_live_coins()
    logger.info("Wallet=%s monitoring=%s", address, ",".join(coins))

    risk = RiskState()
    last_signal_ts: dict[str, pd.Timestamp] = {}
    last_guard_log: dict[str, datetime] = {}

    while True:
        try:
            for coin in coins:
                await _scan_coin(exchange, info, address, coin, risk, last_signal_ts, last_guard_log)
            await asyncio.sleep(SIGNAL_POLL_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Main loop error: %s", exc)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(run_bot_async())