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
import logging
import os
import time
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

from indicators import add_indicators, check_entry_conditions
from run_backtest import CONFIG, COIN_CONFIG


logger = logging.getLogger("bot_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


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
SL_SINGLE_COOLDOWN_MINUTES = int(os.environ.get("HL_SL_COOLDOWN_MINUTES", "10"))

# ── FIX #3: Giới hạn tổng margin usage ──
MAX_MARGIN_USAGE_PCT = float(os.environ.get("HL_MAX_MARGIN_PCT", "0.70"))  # max 70% account dùng làm margin

# ── HARD DOLLAR SL: đóng ngay nếu unrealized loss vượt ngưỡng này ──
# Tránh tình huống như ETH -$7 vì price trượt dài không chạm % SL
HARD_DOLLAR_SL_USD = float(os.environ.get("HL_HARD_DOLLAR_SL", "3.0"))
HARD_DOLLAR_SL_CHECK_INTERVAL = int(os.environ.get("HL_HARD_SL_INTERVAL", "5"))  # check mỗi N giây

# ── FIX #1: Track active entry orders để chống stacking ──
# key = coin, value = timestamp khi entry order được gửi đi
_active_entry: dict[str, float] = {}
ACTIVE_ENTRY_TIMEOUT_SEC = 180  # 3 phút: nếu quá thời gian này mà không có result thì tự động clear

# Per-coin lock để ngăn concurrent execution
_coin_locks: dict[str, asyncio.Lock] = {}

# Theo dõi thời gian orphan position của từng coin
_orphan_timers: dict[str, float] = {}

# ── OBI Delta Tracking ──
_obi_history: dict[str, list[tuple[float, float]]] = {}

# ── Margin Summary Throttle ──
_cached_margin_summary = {"data": {"account_value": 0.0, "total_margin_used": 0.0, "withdrawable": 0.0}, "last_time": 0.0}



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

        day_key = (coin, now_utc.date())
        if self.trades_today.get(day_key, 0) >= CONFIG["max_trades_per_day"]:
            return False, "max trades per day reached"

        if self.pnl_today.get(day_key, 0.0) <= -CONFIG["daily_loss_limit_usd"]:
            return False, "daily loss limit reached"

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


def _margin_summary(info: Info, address: str, force: bool = False) -> dict[str, float]:
    """Lấy tổng margin đang dùng và account value (Throttled 30s)."""
    global _cached_margin_summary
    import time
    now = time.time()
    if not force and now - _cached_margin_summary["last_time"] < 30.0 and _cached_margin_summary["data"]["account_value"] > 0:
        return _cached_margin_summary["data"]
        
    try:
        state = info.user_state(address)
        ms = state.get("marginSummary", {})
        data = {
            "account_value": _to_float(ms.get("accountValue")),
            "total_margin_used": _to_float(ms.get("totalMarginUsed")),
            "withdrawable": _to_float(ms.get("withdrawable")),
        }
        _cached_margin_summary["data"] = data
        _cached_margin_summary["last_time"] = now
        return data
    except Exception as exc:
        logger.warning("margin_summary fetch failed: %s", exc)
        return _cached_margin_summary["data"]


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


def _get_orderbook_analysis(info: Info, coin: str, current_price: float, depth: int = 20) -> dict:
    """
    Phân tích sổ lệnh nâng cao:
    - OBI: orderbook imbalance (đã có)
    - Wall: phát hiện lệnh lớn bất thường (> 3x average level size)
    - Spread: bid-ask spread hiện tại (%)
    - Near wall: wall nằm trong 0.1% của current price
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

        bids = l2["levels"][0][:depth]
        asks = l2["levels"][1][:depth]

        if not bids or not asks:
            return result

        # Spread
        best_bid = float(bids[0].get("px", 0))
        best_ask = float(asks[0].get("px", 0))
        if best_bid > 0:
            result["spread_pct"] = (best_ask - best_bid) / best_bid * 100

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
        result["best_bid"] = best_bid
        result["best_ask"] = best_ask

    except Exception as e:
        logger.debug("orderbook_analysis failed: %s", e)

    return result


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


def _latest_setup(info: Info, coin: str, effective_cfg: dict) -> tuple[dict[str, Any], float, pd.Timestamp] | None:
    asof_ms = _now_ms()
    df15 = fetch_hyperliquid_candles(info, coin, "15m", LOOKBACK_15M, asof_ms)
    df1 = fetch_hyperliquid_candles(info, coin, "1m", LOOKBACK_1M, asof_ms)
    if df15.empty or df1.empty:
        logger.warning("%s: empty candle data", coin)
        return None
    i1 = len(df1) - 1
    signal_ts = df1.iloc[i1]["timestamp"]
    i15 = _align_index(signal_ts, df15)
    current_price = float(df1.iloc[i1]["close"])
    ob_analysis = _get_orderbook_analysis(info, coin, current_price, 20)
    obi = ob_analysis.get("obi", 0.0)
    
    # Track OBI history for OBI Delta
    import time
    now_ts = time.time()
    if coin not in _obi_history:
        _obi_history[coin] = []
    _obi_history[coin].append((now_ts, obi))
    _obi_history[coin] = [(t, o) for t, o in _obi_history[coin] if now_ts - t <= 65]
    
    obi_delta = 0.0
    if len(_obi_history[coin]) > 1:
        oldest_obi = _obi_history[coin][0][1]
        obi_delta = obi - oldest_obi

    setup = check_entry_conditions(i15, df15, i1, df1, obi=obi, obi_delta=obi_delta, cfg=effective_cfg)
    setup["obi"] = obi
    setup["obi_delta"] = obi_delta
    setup["best_bid"] = ob_analysis.get("best_bid", current_price)
    setup["best_ask"] = ob_analysis.get("best_ask", current_price)
    
    return setup, current_price, signal_ts


async def _call(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


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
) -> FillSummary:
    deadline = time.monotonic() + timeout_seconds
    last_summary = FillSummary()

    while time.monotonic() < deadline:
        fills = await _call(_fills_by_oid, info, address, start_ms, coin, oid)
        last_summary = _summarize_fills(fills, oid)
        if last_summary.size >= target_size * 0.999:
            return last_summary
        if oid is not None and not await _call(_is_order_open, info, address, coin, oid) and last_summary.size <= 0:
            return last_summary
        await asyncio.sleep(ORDER_POLL_SECONDS)

    # ── FIX #4: Cancel và xác nhận không còn position orphan ──
    logger.info("%s: entry timeout — cancelling order oid=%s", coin, oid)
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
        pos_size, _ = await _call(_position_for_coin, info, address, coin)
        # Bỏ qua dust position (< $10)
        # Giả định giá BTC ~ 60k -> $10 ~ 0.00016. Tạm dùng ngưỡng 0.0002 (khoảng $12) làm dust cho BTC.
        if abs(pos_size) <= 0.0002: 
            return True
        await asyncio.sleep(ORDER_POLL_SECONDS)
    return False


def _clear_active_entry(coin: str) -> None:
    """Xóa active entry guard cho coin khi trade hoàn thành hoặc bị abort."""
    _active_entry.pop(coin, None)
    logger.debug("%s: active entry guard cleared", coin)


async def _execute_setup(
    exchange: Exchange,
    info: Info,
    address: str,
    coin: str,
    setup: dict,
    signal_price: float,
    signal_ts: pd.Timestamp,
    margin: float,
    effective_cfg: dict | None = None,
) -> TradeResult | None:
    cfg = effective_cfg or CONFIG
    leverage = cfg["leverage"]
    notional = margin * leverage
    direction = setup["direction"]
    is_entry_buy = direction == "long"
    exit_is_buy = not is_entry_buy

    best_bid = setup.get("best_bid", signal_price)
    best_ask = setup.get("best_ask", signal_price)

    def get_tick(price):
        s = str(price)
        if '.' in s:
            return 10 ** -(len(s) - s.index('.') - 1)
        return 1.0

    if is_entry_buy:
        entry_price = best_bid + get_tick(best_bid)
        # Ensure it doesn't cross spread
        if entry_price >= best_ask:
            entry_price = best_bid
    else:
        entry_price = best_ask - get_tick(best_ask)
        if entry_price <= best_bid:
            entry_price = best_ask

    entry_price = _round_price(entry_price)
    size = _round_size(exchange, coin, notional / entry_price)
    if size <= 0:
        logger.warning("%s: computed entry size is zero; skipping", coin)
        _clear_active_entry(coin)
        return None

    start_ms = _now_ms()
    managed_oids = set()

    logger.info(
        "%s: entry %s at %.4f (Best Bid=%.4f, Best Ask=%.4f) | margin=%.2f notional=%.2f size=%.8f",
        coin, direction.upper(), entry_price, best_bid, best_ask, margin, notional, size,
    )

    pre_check_pos, _ = await _call(_position_for_coin, info, address, coin)
    pre_check_orders = await _call(_open_orders_for_coin, info, address, coin)
    if abs(pre_check_pos) > POSITION_EPSILON or pre_check_orders:
        logger.warning("%s: exposure detected before order; aborting", coin)
        _clear_active_entry(coin)
        return None

    # Post-Only Maker
    entry_oid, response = await _place_limit(
        exchange, coin, is_entry_buy, size, entry_price, False, "Alo"
    )
    if entry_oid is None:
        logger.warning("%s: Entry ALO rejected (likely crosses spread), skipping", coin)
        _clear_active_entry(coin)
        return None
        
    managed_oids.add(entry_oid)

    timeout_seconds = 20  # 15-20s timeout
    entry_fill = await _wait_entry_fill(
        info, exchange, address, coin, entry_oid, size, start_ms, timeout_seconds,
    )

    if entry_fill.size <= POSITION_EPSILON or entry_fill.avg_price is None:
        logger.info("%s: entry not filled within timeout; setup skipped", coin)
        _clear_active_entry(coin)
        return None

    if entry_fill.size < size * 0.999:
        if entry_fill.size < size * 0.25:
            logger.warning("%s: partial fill < 25%%; closing as garbage.", coin)
            await _market_close_remaining(exchange, info, address, coin, managed_oids)
            _clear_active_entry(coin)
            return None
        logger.warning("%s: partial entry fill %.8f / %.8f; managing remainder", coin, entry_fill.size, size)

    avg_entry = entry_fill.avg_price
    filled_size = _round_size(exchange, coin, entry_fill.size)

    # Dynamic SL/TP based on 1m ATR
    atr_1m = setup.get("atr_1m", 0.0)
    if atr_1m <= 0:
        atr_1m = avg_entry * 0.005 # fallback

    sl_dist = 1.5 * atr_1m
    
    # Calculate TP price for a $2.5 target profit
    target_profit_usd = cfg.get("target_profit_usd", 2.5)
    tp_dist = target_profit_usd / filled_size
    
    if direction == "long":
        sl_price = avg_entry - sl_dist
        tp_price = avg_entry + tp_dist
    else:
        sl_price = avg_entry + sl_dist
        tp_price = avg_entry - tp_dist

    deadline = datetime.now(timezone.utc) + timedelta(minutes=cfg["time_stop_minutes"])

    logger.info(
        "%s: filled %.8f @ %.4f | SL=%.4f TP=%.4f | deadline=%s",
        coin, filled_size, avg_entry, sl_price, tp_price, deadline.strftime("%H:%M"),
    )

    # Place SL trigger
    sl_oid, _ = await _place_trigger_sl(exchange, coin, exit_is_buy, filled_size, sl_price)
    if sl_oid is not None:
        managed_oids.add(sl_oid)

    # Place TP Limit
    tp_oid = await _place_tp_with_fallback(exchange, coin, exit_is_buy, filled_size, tp_price)
    if tp_oid is not None:
        managed_oids.add(tp_oid)

    exit_reason = "unknown"
    breakeven_pnl_usd = cfg.get("breakeven_pnl_usd", 1.5)
    breakeven_triggered = False

    while True:
        try:
            now = datetime.now(timezone.utc)
            pos_size, pos_entry_px = await _call(_position_for_coin, info, address, coin)
            abs_pos = abs(pos_size)

            if now >= deadline:
                exit_reason = "time_stop"
                await _cancel_if_open(exchange, info, address, coin, sl_oid)
                await _cancel_if_open(exchange, info, address, coin, tp_oid)
                await _market_close_remaining(exchange, info, address, coin, managed_oids)
                await _wait_for_position_flat(info, address, coin)
                break

            if abs_pos <= 0.0002:
                exit_reason = "SL_or_TP_hit"
                break

            # Calculate unrealized PnL
            state = await _call(info.user_state, address)
            unrealized_pnl = 0.0
            for item in state.get("assetPositions", []):
                pos = item.get("position", {})
                if pos.get("coin") == coin:
                    unrealized_pnl = float(pos.get("unrealizedPnl", 0.0))
                    break

            # Breakeven Trailing Stop
            if not breakeven_triggered and unrealized_pnl >= breakeven_pnl_usd:
                logger.info("%s: Unrealized PnL $%.2f >= $%.2f. Triggering breakeven SL at entry.", coin, unrealized_pnl, breakeven_pnl_usd)
                breakeven_triggered = True
                
                # Cancel old SL and create new SL at entry price
                await _cancel_if_open(exchange, info, address, coin, sl_oid)
                # Slightly offset to cover fees
                fee_offset = (avg_entry * 0.0005) # approx 0.05% for fees
                new_sl_price = avg_entry + fee_offset if direction == "long" else avg_entry - fee_offset
                sl_oid, _ = await _place_trigger_sl(exchange, coin, exit_is_buy, abs_pos, new_sl_price)
                if sl_oid is not None:
                    managed_oids.add(sl_oid)
                    
            # Check hard dollar SL (if it drops rapidly)
            if unrealized_pnl < -cfg.get("max_loss_per_trade_usd", 3.0):
                logger.warning("%s: HARD DOLLAR SL triggered! unrealized_pnl=%.4f — emergency close", coin, unrealized_pnl)
                exit_reason = "hard_dollar_sl"
                await _cancel_if_open(exchange, info, address, coin, sl_oid)
                await _cancel_if_open(exchange, info, address, coin, tp_oid)
                await _market_close_remaining(exchange, info, address, coin, managed_oids)
                await _wait_for_position_flat(info, address, coin)
                break

            # Check if orders are missing
            open_orders = await _call(_open_orders_for_coin, info, address, coin)
            open_oids = {int(o["oid"]) for o in open_orders if "oid" in o}
            
            if tp_oid is not None and tp_oid not in open_oids:
                tp_summary = _summarize_fills(await _call(_fills_by_oid, info, address, start_ms, coin, tp_oid), tp_oid)
                if tp_summary.size <= 0 and abs_pos > POSITION_EPSILON:
                    logger.warning("%s: TP order missing, recreating...", coin)
                    tp_oid = await _place_tp_with_fallback(exchange, coin, exit_is_buy, abs_pos, tp_price)
                    if tp_oid: managed_oids.add(tp_oid)

            if sl_oid is not None and sl_oid not in open_oids:
                sl_summary = _summarize_fills(await _call(_fills_by_oid, info, address, start_ms, coin, sl_oid), sl_oid)
                if sl_summary.size <= 0 and abs_pos > POSITION_EPSILON:
                    logger.warning("%s: SL order missing, recreating...", coin)
                    sl_oid, _ = await _place_trigger_sl(exchange, coin, exit_is_buy, abs_pos, avg_entry if breakeven_triggered else sl_price)
                    if sl_oid: managed_oids.add(sl_oid)
                    
        except Exception as exc:
            logger.error("%s: active monitoring loop error: %s", coin, exc)

        await asyncio.sleep(ORDER_POLL_SECONDS)

    await _cancel_if_open(exchange, info, address, coin, sl_oid)
    await _cancel_if_open(exchange, info, address, coin, tp_oid)
    
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
    setup: dict,
    signal_price: float,
    signal_ts: pd.Timestamp,
    margin: float,
    risk: RiskState,
    effective_cfg: dict | None = None,
) -> None:
    try:
        result = await _execute_setup(
            exchange, info, address, coin,
            setup, signal_price, signal_ts, margin,
            effective_cfg=effective_cfg,
        )
        if result is not None:
            risk.record_trade(result)
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

        # ── FIX #1: Active entry guard — kiểm tra TRƯỚC khi gọi API ──
        # Nếu đã gửi entry order trong ACTIVE_ENTRY_TIMEOUT_SEC giây qua,
        # KHÔNG được mở lệnh mới bất kể API trả về gì.
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
                # Timeout — clear guard và tiến hành kiểm tra bình thường
                logger.warning(
                    "%s: active entry guard expired after %.0fs — clearing",
                    coin, elapsed,
                )
                _active_entry.pop(coin, None)

        now_utc = datetime.now(timezone.utc)
        can_trade, reason = risk.can_trade(coin, now_utc)
        if not can_trade:
            logger.info("%s: risk block: %s", coin, reason)
            return

        # ── FIX #3: Kiểm tra tổng margin usage ──
        ms = await _call(_margin_summary, info, address)
        if ms["account_value"] > 0:
            margin_usage_pct = ms["total_margin_used"] / ms["account_value"]
            if margin_usage_pct > MAX_MARGIN_USAGE_PCT:
                logger.info(
                    "%s: margin usage %.1f%% > limit %.1f%% — skipping",
                    coin, margin_usage_pct * 100, MAX_MARGIN_USAGE_PCT * 100,
                )
                return

        pos_size, pos_entry = await _call(_position_for_coin, info, address, coin)
        open_orders = await _call(_open_orders_for_coin, info, address, coin)

        # Dọn dẹp stale orders
        if abs(pos_size) <= 0.0002 and len(open_orders) > 0:
            logger.info("%s: %d stale orders without valid position → cancelling", coin, len(open_orders))
            for order in open_orders:
                oid = order.get("oid")
                if oid:
                    await _call(exchange.cancel, coin, int(oid))
            open_orders = await _call(_open_orders_for_coin, info, address, coin)

        # Bỏ qua dust position (< 0.0002 BTC) khi check existing exposure
        if abs(pos_size) > 0.0002 or open_orders:
            # Lưới an toàn cho orphan position
            if abs(pos_size) > 0.0002 and not open_orders:
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

        effective_cfg = get_effective_config(coin)
        latest = await _call(_latest_setup, info, coin, effective_cfg)
        if latest is None:
            return
        setup, current_price, signal_ts = latest
        obi = setup.get("obi", 0.0)

        if last_signal_ts.get(coin) == signal_ts:
            return
        last_signal_ts[coin] = signal_ts

        if len(setup.get("block_reasons", [])) > 0 or setup.get("direction") is None:
            reason_text = setup["block_reasons"][0] if setup["block_reasons"] else "no direction"
            logger.info("%s %s: block: %s (price %.4f)", coin, signal_ts, reason_text, current_price)
            return

        direction = setup["direction"]
        
        logger.info(
            "%s %s: setup %s conf=%s price=%.4f obi=%.2f obi_delta=%.2f atr_1m=%.4f",
            coin, signal_ts, direction.upper(), setup.get("confidence", "?"),
            current_price, obi, setup.get("obi_delta", 0.0), setup.get("atr_1m", 0.0)
        )

        margin_full = effective_cfg.get("margin_full", CONFIG.get("margin_full", 100.0))
        margin_half = effective_cfg.get("margin_half", CONFIG.get("margin_half", 50.0))
        
        margin = margin_full if setup.get("confidence") == "A+" else margin_half

        # ── FIX #1: Set active entry guard NGAY KHI quyết định trade ──
        # Từ đây đến khi _execute_setup hoàn thành, mọi poll cycle sẽ bị chặn.
        _active_entry[coin] = time.time()
        logger.info("%s: active entry guard SET — executing trade in background", coin)

        asyncio.create_task(
            _execute_and_record(
                exchange, info, address, coin,
                setup, current_price, signal_ts, margin,
                risk,
                effective_cfg=effective_cfg,
            )
        )


async def run_bot_async():
    logger.info("=" * 60)
    logger.info("STARTING BOT ENGINE — BTC ONLY MODE")
    logger.info("=" * 60)
    logger.info(
        "Profile=%s | entry=%s | TP_USD=%.2f | BE_USD=%.2f | SL_ATR=%.2f | time_stop=%sm",
        CONFIG.get("profile_name", "unknown"), CONFIG.get("entry_order_type", "limit"),
        CONFIG.get("target_profit_usd", 0.0), CONFIG.get("breakeven_pnl_usd", 0.0), CONFIG.get("sl_atr_mult", 0.0),
        CONFIG.get("time_stop_minutes", 30),
    )
    logger.info(
        "FIX #1: Active entry guard=%ds | FIX #2: SL cooldown=%dm | FIX #3: Max margin=%.0f%%",
        ACTIVE_ENTRY_TIMEOUT_SEC, SL_SINGLE_COOLDOWN_MINUTES, MAX_MARGIN_USAGE_PCT * 100,
    )
    logger.info(
        "HARD DOLLAR SL=%.2f USD | DISABLED COINS=%s | daily_loss_limit=%.2f USD | max_trades/day=%d",
        HARD_DOLLAR_SL_USD, ",".join(DISABLED_COINS) if DISABLED_COINS else "none",
        CONFIG.get("daily_loss_limit_usd", 0), CONFIG.get("max_trades_per_day", 0),
    )

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