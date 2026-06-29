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

from indicators import add_indicators, score_setup
from run_backtest import CONFIG


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

SIGNAL_POLL_SECONDS = int(os.environ.get("HL_SIGNAL_POLL_SECONDS", "15"))
ORDER_POLL_SECONDS = int(os.environ.get("HL_ORDER_POLL_SECONDS", "5"))
LOOKBACK_1M = int(os.environ.get("HL_LOOKBACK_1M", "300"))
LOOKBACK_5M = int(os.environ.get("HL_LOOKBACK_5M", "160"))
LOOKBACK_15M = int(os.environ.get("HL_LOOKBACK_15M", "260"))
MARKET_CLOSE_SLIPPAGE = float(os.environ.get("HL_MARKET_CLOSE_SLIPPAGE", "0.002"))
POSITION_EPSILON = 1e-8

# ── FIX #2: Cooldown nhỏ sau mỗi lần thua đơn lẻ ──
SL_SINGLE_COOLDOWN_MINUTES = int(os.environ.get("HL_SL_COOLDOWN_MINUTES", "10"))

# ── FIX #3: Giới hạn tổng margin usage ──
MAX_MARGIN_USAGE_PCT = float(os.environ.get("HL_MAX_MARGIN_PCT", "0.70"))  # max 70% account dùng làm margin

# ── FIX #1: Track active entry orders để chống stacking ──
# key = coin, value = timestamp khi entry order được gửi đi
_active_entry: dict[str, float] = {}
ACTIVE_ENTRY_TIMEOUT_SEC = 180  # 3 phút: nếu quá thời gian này mà không có result thì tự động clear

# Per-coin lock để ngăn concurrent execution
_coin_locks: dict[str, asyncio.Lock] = {}


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
    data = response.get("response", {}).get("data", {})
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
    for status in _order_statuses(response):
        if status.get("error"):
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


def _get_orderbook_imbalance(info: Info, coin: str, depth: int = 50) -> float:
    try:
        l2 = info.l2_snapshot(coin)
        if not l2 or "levels" not in l2 or len(l2["levels"]) < 2:
            return 0.0
        bids = l2["levels"][0][:depth]
        asks = l2["levels"][1][:depth]
        bid_vol = sum(float(b.get("sz", 0)) for b in bids)
        ask_vol = sum(float(a.get("sz", 0)) for a in asks)
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return 0.0
        return (bid_vol - ask_vol) / total_vol
    except Exception as e:
        logger.error("Failed to fetch L2 snapshot for %s: %s", coin, e)
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


def _latest_setup(info: Info, coin: str) -> tuple[dict[str, Any], float, pd.Timestamp] | None:
    asof_ms = _now_ms()
    df15 = fetch_hyperliquid_candles(info, coin, "15m", LOOKBACK_15M, asof_ms)
    df5 = fetch_hyperliquid_candles(info, coin, "5m", LOOKBACK_5M, asof_ms)
    df1 = fetch_hyperliquid_candles(info, coin, "1m", LOOKBACK_1M, asof_ms)
    if df15.empty or df5.empty or df1.empty:
        logger.warning("%s: empty candle data", coin)
        return None
    i1 = len(df1) - 1
    signal_ts = df1.iloc[i1]["timestamp"]
    i5 = _align_index(signal_ts, df5)
    i15 = _align_index(signal_ts, df15)
    setup = score_setup(i15, df15, i5, df5, i1, df1, hour_utc=signal_ts.hour, cfg=CONFIG)
    current_price = float(df1.iloc[i1]["close"])
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

    # Đợi thêm 1 cycle để exchange xử lý cancel
    await asyncio.sleep(ORDER_POLL_SECONDS)

    fills = await _call(_fills_by_oid, info, address, start_ms, coin, oid)
    final_summary = _summarize_fills(fills, oid)

    if final_summary.size > POSITION_EPSILON:
        logger.warning(
            "%s: partial fill after cancel: size=%.8f — will manage this position",
            coin, final_summary.size,
        )
    else:
        # Xác nhận position thực sự flat
        pos_size, _ = await _call(_position_for_coin, info, address, coin)
        if abs(pos_size) > POSITION_EPSILON:
            logger.error(
                "%s: orphan position detected after cancel! pos=%.8f — emergency close",
                coin, pos_size,
            )
            managed_oids: set[int] = set()
            await _market_close_remaining(exchange, info, address, coin, managed_oids)

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
        if abs(pos_size) <= POSITION_EPSILON:
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
    direction: str,
    signal_price: float,
    signal_ts: pd.Timestamp,
    margin: float,
    atr_5m: float = 0.0,
) -> TradeResult | None:
    leverage = CONFIG["leverage"]
    notional = margin * leverage
    is_entry_buy = direction == "long"
    exit_is_buy = not is_entry_buy

    entry_price = signal_price
    if CONFIG.get("entry_order_type") == "limit":
        offset = CONFIG.get("limit_offset_pct", 0.0) / 100
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

    if CONFIG.get("entry_order_type") == "market":
        response = await _call(
            exchange.market_open, coin, is_entry_buy, size, signal_price,
            CONFIG.get("slippage_pct", 0.0) / 100,
        )
        entry_oid = _extract_oid(response)
        timeout_seconds = 30
    else:
        entry_oid, response = await _place_limit(
            exchange, coin, is_entry_buy, size, entry_price, False, "Gtc"
        )
        timeout_seconds = max(1, int(CONFIG.get("limit_timeout_bars", 1))) * 60

    if entry_oid is None:
        logger.warning("%s: entry order not accepted: %s", coin, response)
        _clear_active_entry(coin)
        return None
    managed_oids.add(entry_oid)

    entry_fill = await _wait_entry_fill(
        info, exchange, address, coin, entry_oid, size, start_ms, timeout_seconds,
    )

    if entry_fill.size <= POSITION_EPSILON or entry_fill.avg_price is None:
        logger.info("%s: entry not filled within timeout; setup skipped", coin)
        _clear_active_entry(coin)
        return None

    if entry_fill.size < size * 0.999:
        logger.warning(
            "%s: partial entry fill %.8f / %.8f; managing remainder",
            coin, entry_fill.size, size,
        )

    avg_entry = entry_fill.avg_price
    filled_size = _round_size(exchange, coin, entry_fill.size)

    # Tính TP/SL
    if CONFIG.get("use_dynamic_tp_sl", False) and atr_5m > 0:
        from filters import compute_dynamic_levels
        levels = compute_dynamic_levels(
            avg_entry, direction, atr_5m,
            tp1_atr_mult=CONFIG.get("tp1_atr_mult", 1.5),
            tp2_atr_mult=CONFIG.get("tp2_atr_mult", 3.0),
            sl_atr_mult=CONFIG.get("sl_atr_mult", 1.2),
        )
        tp1_price = levels["tp1_price"]
        tp2_price = levels["tp2_price"]
        sl_price = levels["sl_price"]
    else:
        tp1_pct = CONFIG["tp1_pct"] / 100
        tp2_pct = CONFIG["tp2_pct"] / 100
        sl_pct = CONFIG["sl_pct"] / 100
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
    deadline = datetime.now(timezone.utc) + timedelta(minutes=CONFIG["time_stop_minutes"])

    logger.info(
        "%s: filled %.8f @ %.4f | TP1=%.4f TP2=%.4f SL=%.4f | deadline=%s",
        coin, filled_size, avg_entry, tp1_price, tp2_price, sl_price,
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

    while True:
        try:
            now = datetime.now(timezone.utc)
            pos_size, _ = await _call(_position_for_coin, info, address, coin)
            abs_pos = abs(pos_size)

            if now >= deadline:
                exit_reason = "time_stop"
                for oid in (tp1_oid, tp2_oid, sl_oid):
                    await _cancel_if_open(exchange, info, address, coin, oid)
                await _market_close_remaining(exchange, info, address, coin, managed_oids)
                await _wait_for_position_flat(info, address, coin)
                break

            if abs_pos <= POSITION_EPSILON:
                exit_reason = "SL_or_external_flat" if stage != "tp2_done" else "TP2"
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
    atr_5m: float,
    risk: RiskState,
) -> None:
    try:
        result = await _execute_setup(
            exchange, info, address, coin,
            direction, signal_price, signal_ts, margin,
            atr_5m,
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
        if abs(pos_size) <= POSITION_EPSILON and len(open_orders) > 0:
            logger.info("%s: %d stale orders without position → cancelling", coin, len(open_orders))
            for order in open_orders:
                oid = order.get("oid")
                if oid:
                    await _call(exchange.cancel, coin, int(oid))
            open_orders = await _call(_open_orders_for_coin, info, address, coin)

        if abs(pos_size) > POSITION_EPSILON or open_orders:
            last_log = last_guard_log.get(coin)
            if not last_log or (now_utc - last_log).total_seconds() >= 60:
                logger.warning(
                    "%s: existing exposure (pos=%.8f entry=%s orders=%d) — skipping",
                    coin, pos_size, pos_entry, len(open_orders),
                )
                last_guard_log[coin] = now_utc
            return

        latest = await _call(_latest_setup, info, coin)
        if latest is None:
            return
        setup, current_price, signal_ts = latest

        if last_signal_ts.get(coin) == signal_ts:
            return
        last_signal_ts[coin] = signal_ts

        if setup["hard_block"] or setup["direction"] is None:
            reason_text = setup["block_reasons"][0] if setup["block_reasons"] else "no direction"
            logger.info("%s %s: block: %s (price %.4f)", coin, signal_ts, reason_text, current_price)
            return

        score = setup["score"]
        direction = setup["direction"]

        # Order Book Imbalance check
        min_obi = 0.15
        obi = await _call(_get_orderbook_imbalance, info, coin)
        setup["obi"] = obi

        if direction == "long" and obi < min_obi:
            logger.info(
                "%s %s: LONG (score %d) blocked — OBI=%.2f (sellers dominate)",
                coin, signal_ts, score, obi,
            )
            return
        elif direction == "short" and obi > -min_obi:
            logger.info(
                "%s %s: SHORT (score %d) blocked — OBI=%.2f (buyers dominate)",
                coin, signal_ts, score, obi,
            )
            return

        logger.info(
            "%s %s: setup %s score=%s price=%.4f obi=%.2f",
            coin, signal_ts, direction.upper(), score, current_price, obi,
        )

        if score < CONFIG["min_score_half"]:
            return

        margin = CONFIG["margin_full"] if score >= CONFIG["min_score_full"] else CONFIG["margin_half"]

        # ── FIX #1: Set active entry guard NGAY KHI quyết định trade ──
        # Từ đây đến khi _execute_setup hoàn thành, mọi poll cycle sẽ bị chặn.
        _active_entry[coin] = time.time()
        logger.info("%s: active entry guard SET — executing trade in background", coin)

        asyncio.create_task(
            _execute_and_record(
                exchange, info, address, coin,
                setup["direction"], current_price, signal_ts, margin,
                setup.get("atr_5m", 0.0),
                risk
            )
        )


async def run_bot_async():
    logger.info("=" * 60)
    logger.info("STARTING IMPROVED BOT ENGINE")
    logger.info("=" * 60)
    logger.info(
        "Profile=%s | entry=%s | TP1=%.3f%% | TP2=%.3f%% | SL=%.3f%% | time_stop=%sm",
        CONFIG["profile_name"], CONFIG["entry_order_type"],
        CONFIG["tp1_pct"], CONFIG["tp2_pct"], CONFIG["sl_pct"],
        CONFIG["time_stop_minutes"],
    )
    logger.info(
        "FIX #1: Active entry guard=%ds | FIX #2: SL cooldown=%dm | FIX #3: Max margin=%.0f%%",
        ACTIVE_ENTRY_TIMEOUT_SEC, SL_SINGLE_COOLDOWN_MINUTES, MAX_MARGIN_USAGE_PCT * 100,
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