"""
Strategy brain for the scalping bot.

The module is intentionally execution-free: it reads prepared candles plus
optional live order-book features, then returns one setup object that backtest
and live trading can both consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from filters import calculate_adx, momentum_strength


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct_diff(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _empty_result() -> dict[str, Any]:
    return {
        "direction": None,
        "score": 0,
        "hard_block": False,
        "block_reasons": [],
        "regime": "unknown",
        "engine": "none",
        "momentum": 0.0,
        "atr_1m": 0.0,
        "atr_5m": 0.0,
        "bb_width_pct": 0.0,
        "bb_squeeze": False,
        "confidence": "C",
        "score_details": {},
        "obi": 0.0,
        "spread_pct": 0.0,
        "orderbook_pressure": "unknown",
        "expected_net_tp_usd": 0.0,
    }


def _add_detail(details: dict[str, int], key: str, value: int) -> int:
    details[key] = details.get(key, 0) + value
    return value


def _candle_pattern(row: pd.Series, prev: pd.Series) -> str:
    body = abs(_num(row["close"]) - _num(row["open"]))
    candle_range = _num(row["high"]) - _num(row["low"])
    if candle_range <= 0:
        return "none"

    close = _num(row["close"])
    open_ = _num(row["open"])
    high = _num(row["high"])
    low = _num(row["low"])
    prev_close = _num(prev["close"])
    prev_open = _num(prev["open"])

    upper_wick = high - max(close, open_)
    lower_wick = min(close, open_) - low
    bullish = close > open_
    bearish = close < open_
    body_ratio = body / candle_range if candle_range else 0.0

    prev_top = max(prev_close, prev_open)
    prev_bottom = min(prev_close, prev_open)

    if bullish and prev_close < prev_open and close > prev_top and open_ < prev_bottom:
        return "bullish_engulfing"
    if bearish and prev_close > prev_open and close < prev_bottom and open_ > prev_top:
        return "bearish_engulfing"
    if lower_wick >= 2.0 * max(body, 1e-12) and upper_wick < max(body, candle_range * 0.25):
        return "hammer"
    if upper_wick >= 2.0 * max(body, 1e-12) and lower_wick < max(body, candle_range * 0.25):
        return "shooting_star"
    if bullish and body_ratio > 0.68:
        return "strong_bullish_close"
    if bearish and body_ratio > 0.68:
        return "strong_bearish_close"
    if body_ratio < 0.12:
        return "doji"
    return "none"


def _rsi_divergence(df: pd.DataFrame, idx: int, direction: str, lookback: int = 10) -> bool:
    if idx < lookback * 2:
        return False
    current_price = _num(df.iloc[idx].get("close"))
    current_rsi = _num(df.iloc[idx].get("rsi"), np.nan)
    if pd.isna(current_rsi):
        return False
    prev = df.iloc[idx - lookback * 2 : idx - lookback + 1]
    if len(prev) < 3:
        return False
    if direction == "long":
        return current_price <= prev["close"].min() and current_rsi > prev["rsi"].min()
    return current_price >= prev["close"].max() and current_rsi < prev["rsi"].max()


def _adx_at(df: pd.DataFrame, idx: int) -> float:
    if "adx" in df.columns:
        return _num(df.iloc[idx].get("adx"))
    start = max(0, idx - 70)
    window = df.iloc[start : idx + 1].copy().reset_index(drop=True)
    series = calculate_adx(window)
    if series is None or series.empty:
        return 0.0
    return _num(series.iloc[-1])


def _ema_slope_pct(df: pd.DataFrame, idx: int, column: str = "ema50", bars: int = 5) -> float:
    if idx < bars:
        return 0.0
    now = _num(df.iloc[idx].get(column), np.nan)
    then = _num(df.iloc[idx - bars].get(column), np.nan)
    if pd.isna(now) or pd.isna(then) or then == 0:
        return 0.0
    return _pct_diff(now, then)


def _trend_dir(row: pd.Series, slope_pct: float = 0.0) -> str:
    close = _num(row.get("close"))
    ema9 = _num(row.get("ema9"), np.nan)
    ema21 = _num(row.get("ema21"), np.nan)
    ema50 = _num(row.get("ema50"), np.nan)
    if any(pd.isna(v) for v in (ema9, ema21, ema50)):
        return "flat"
    if ema9 > ema21 and close > ema21 and (ema21 >= ema50 or slope_pct > 0.03):
        return "up"
    if ema9 < ema21 and close < ema21 and (ema21 <= ema50 or slope_pct < -0.03):
        return "down"
    return "flat"


@dataclass
class RegimeContext:
    regime: str
    trend_5m: str
    trend_15m: str
    impulse_dir: str
    adx_15m: float
    momentum_5m: float
    bb_width_pct: float
    bb_squeeze: bool
    atr_1m: float
    atr_5m: float
    reason: str


def detect_market_regime(
    i15: int,
    df15: pd.DataFrame,
    i5: int,
    df5: pd.DataFrame,
    i1: int,
    df1: pd.DataFrame,
    cfg: dict[str, Any],
) -> RegimeContext:
    row1 = df1.iloc[i1]
    row5 = df5.iloc[i5]
    row15 = df15.iloc[i15]

    bb_width = _num(row1.get("bb_width_pct"))
    bb_rank = _num(row1.get("bb_width_rank"), 0.5)
    atr_1m = _num(row1.get("atr"))
    atr_5m = _num(row5.get("atr"))
    adx_15m = _adx_at(df15, i15)
    mom_5m = momentum_strength(df5, i5, lookback=5)
    mom_5m = _num(mom_5m)

    slope_15m = _ema_slope_pct(df15, i15, "ema50", 5)
    slope_5m = _ema_slope_pct(df5, i5, "ema50", 5)
    trend_15m = _trend_dir(row15, slope_15m)
    trend_5m = _trend_dir(row5, slope_5m)

    impulse_bb = cfg.get("impulse_bb_width_pct", cfg.get("bb_width_max_pct", 1.2) * 1.35)
    impulse_mom = cfg.get("impulse_momentum_atr", 1.8)
    range_bb = cfg.get("range_bb_width_pct", min(cfg.get("bb_width_max_pct", 1.2), 1.1))

    impulse_dir = "none"
    if abs(mom_5m) >= impulse_mom:
        impulse_dir = "up" if mom_5m > 0 else "down"
    elif i1 > 0 and atr_1m > 0:
        last_move = _num(row1["close"]) - _num(df1.iloc[i1 - 1]["close"])
        if abs(last_move) > atr_1m * 1.4:
            impulse_dir = "up" if last_move > 0 else "down"

    aligned_trend = trend_5m == trend_15m and trend_5m in {"up", "down"}
    is_impulse = (bb_width >= impulse_bb and impulse_dir != "none") or (
        aligned_trend and abs(mom_5m) >= impulse_mom and adx_15m >= 20
    )

    if is_impulse:
        return RegimeContext(
            "impulse",
            trend_5m,
            trend_15m,
            impulse_dir,
            adx_15m,
            mom_5m,
            bb_width,
            bb_rank < 0.15,
            atr_1m,
            atr_5m,
            "wide_bb_or_fast_momentum",
        )

    if aligned_trend and adx_15m >= cfg.get("trend_adx_min", 20.0):
        return RegimeContext(
            "trend",
            trend_5m,
            trend_15m,
            "none",
            adx_15m,
            mom_5m,
            bb_width,
            bb_rank < 0.15,
            atr_1m,
            atr_5m,
            "aligned_5m_15m_trend",
        )

    if bb_width <= range_bb and adx_15m <= cfg.get("range_adx_max", 24.0) and abs(mom_5m) <= 1.35:
        return RegimeContext(
            "range",
            trend_5m,
            trend_15m,
            "none",
            adx_15m,
            mom_5m,
            bb_width,
            bb_rank < 0.15,
            atr_1m,
            atr_5m,
            "compressed_or_choppy",
        )

    return RegimeContext(
        "transition",
        trend_5m,
        trend_15m,
        "none",
        adx_15m,
        mom_5m,
        bb_width,
        bb_rank < 0.15,
        atr_1m,
        atr_5m,
        "mixed_context",
    )


def _score_range_reversion(
    i5: int,
    df5: pd.DataFrame,
    i1: int,
    df1: pd.DataFrame,
    ctx: RegimeContext,
    cfg: dict[str, Any],
) -> tuple[str | None, int, dict[str, int], list[str]]:
    row1 = df1.iloc[i1]
    prev1 = df1.iloc[i1 - 1]
    row5 = df5.iloc[i5]
    prev5 = df5.iloc[max(0, i5 - 1)]

    close = _num(row1["close"])
    bb_upper = _num(row1.get("bb_upper"), np.nan)
    bb_lower = _num(row1.get("bb_lower"), np.nan)
    bb_pct_b = _num(row1.get("bb_pct_b"), 0.5)
    rsi1 = _num(row1.get("rsi"), 50.0)
    rsi5 = _num(row5.get("rsi"), 50.0)
    ema9 = _num(row1.get("ema9"), close)
    vwap = _num(row1.get("vwap"), close)
    pattern1 = _candle_pattern(row1, prev1)
    pattern5 = _candle_pattern(row5, prev5)

    if pd.isna(bb_upper) or pd.isna(bb_lower) or close <= 0:
        return None, 0, {}, []

    vwap_stretch = abs(_pct_diff(close, vwap)) if vwap > 0 else 0.0
    ema9_dist = _pct_diff(close, ema9) if ema9 > 0 else 0.0

    direction = None
    score = 0
    details: dict[str, int] = {}
    notes: list[str] = []

    long_extreme = close <= bb_lower or rsi1 <= cfg.get("range_rsi_long", 32) or (
        close < vwap and vwap_stretch >= cfg.get("vwap_stretch_min_pct", 0.28)
    ) or ema9_dist <= -cfg.get("ema_reversion_min_pct", 0.28)
    short_extreme = close >= bb_upper or rsi1 >= cfg.get("range_rsi_short", 68) or (
        close > vwap and vwap_stretch >= cfg.get("vwap_stretch_min_pct", 0.28)
    ) or ema9_dist >= cfg.get("ema_reversion_min_pct", 0.28)

    if long_extreme and not short_extreme:
        direction = "long"
    elif short_extreme and not long_extreme:
        direction = "short"
    elif close <= bb_lower:
        direction = "long"
    elif close >= bb_upper:
        direction = "short"
    else:
        return None, 0, {}, []

    _add_detail(details, "range_base", 26)
    score += 26

    if ctx.regime == "range":
        score += _add_detail(details, "regime_range", 18)
    else:
        score += _add_detail(details, "regime_not_range", -18)
        notes.append("range_reversion_outside_range")

    if direction == "long":
        if rsi1 <= 28:
            score += _add_detail(details, "rsi_1m", 18)
        elif rsi1 <= 38:
            score += _add_detail(details, "rsi_1m", 9)
        if rsi5 <= 38:
            score += _add_detail(details, "rsi_5m", 14)
        elif rsi5 <= 46:
            score += _add_detail(details, "rsi_5m", 7)
        if pattern1 in {"hammer", "bullish_engulfing"}:
            score += _add_detail(details, "candle_1m", 12)
        if pattern5 in {"hammer", "bullish_engulfing"}:
            score += _add_detail(details, "candle_5m", 10)
        if bb_pct_b < -cfg.get("bb_pct_b_deep_threshold", 0.15):
            score += _add_detail(details, "deep_bb_break", -16)
        elif bb_pct_b <= 0.08:
            score += _add_detail(details, "bb_depth", 10)
        if ctx.trend_15m == "down" and pattern1 not in {"hammer", "bullish_engulfing"}:
            score += _add_detail(details, "trend_against", -16)
            notes.append("long_against_15m_downtrend")
    else:
        if rsi1 >= 72:
            score += _add_detail(details, "rsi_1m", 18)
        elif rsi1 >= 62:
            score += _add_detail(details, "rsi_1m", 9)
        if rsi5 >= 62:
            score += _add_detail(details, "rsi_5m", 14)
        elif rsi5 >= 54:
            score += _add_detail(details, "rsi_5m", 7)
        if pattern1 in {"shooting_star", "bearish_engulfing"}:
            score += _add_detail(details, "candle_1m", 12)
        if pattern5 in {"shooting_star", "bearish_engulfing"}:
            score += _add_detail(details, "candle_5m", 10)
        if bb_pct_b > 1 + cfg.get("bb_pct_b_deep_threshold", 0.15):
            score += _add_detail(details, "deep_bb_break", -16)
        elif bb_pct_b >= 0.92:
            score += _add_detail(details, "bb_depth", 10)
        if ctx.trend_15m == "up" and pattern1 not in {"shooting_star", "bearish_engulfing"}:
            score += _add_detail(details, "trend_against", -16)
            notes.append("short_against_15m_uptrend")

    if vwap_stretch >= 0.45:
        score += _add_detail(details, "vwap_stretch", 14)
    elif vwap_stretch >= 0.30:
        score += _add_detail(details, "vwap_stretch", 7)

    if _rsi_divergence(df1, i1, direction):
        score += _add_detail(details, "rsi_divergence", 12)

    return direction, score, details, notes


def _score_trend_pullback(
    i5: int,
    df5: pd.DataFrame,
    i1: int,
    df1: pd.DataFrame,
    ctx: RegimeContext,
    cfg: dict[str, Any],
) -> tuple[str | None, int, dict[str, int], list[str]]:
    if ctx.regime != "trend":
        return None, 0, {}, []
    if ctx.trend_5m != ctx.trend_15m or ctx.trend_5m not in {"up", "down"}:
        return None, 0, {}, []

    row1 = df1.iloc[i1]
    prev1 = df1.iloc[i1 - 1]
    row5 = df5.iloc[i5]
    close = _num(row1["close"])
    ema9 = _num(row1.get("ema9"), np.nan)
    ema21 = _num(row1.get("ema21"), np.nan)
    vwap = _num(row1.get("vwap"), np.nan)
    rsi1 = _num(row1.get("rsi"), 50.0)
    rsi5 = _num(row5.get("rsi"), 50.0)
    macd_hist = _num(row5.get("macd_hist"), 0.0)
    pattern1 = _candle_pattern(row1, prev1)

    if close <= 0 or pd.isna(ema9) or pd.isna(ema21):
        return None, 0, {}, []

    direction = "long" if ctx.trend_5m == "up" else "short"
    score = 0
    details: dict[str, int] = {}
    notes: list[str] = []

    score += _add_detail(details, "trend_alignment", 30)
    if ctx.regime == "trend":
        score += _add_detail(details, "regime_trend", 18)
    elif ctx.regime == "transition":
        score += _add_detail(details, "regime_transition", 8)
    elif ctx.regime == "impulse":
        score += _add_detail(details, "regime_impulse", 4)

    dist_ema9 = _pct_diff(close, ema9)
    dist_ema21 = _pct_diff(close, ema21)
    dist_vwap = _pct_diff(close, vwap) if not pd.isna(vwap) and vwap > 0 else 999.0
    pullback_pct = cfg.get("trend_pullback_max_pct", 0.34)

    if direction == "long":
        in_pullback_zone = (
            -pullback_pct <= dist_ema21 <= pullback_pct
            or -pullback_pct <= dist_vwap <= pullback_pct
            or (ema21 <= close <= ema9 * 1.002)
        )
        resume = _num(row1["close"]) > _num(row1["open"]) or pattern1 in {"hammer", "bullish_engulfing"}
        not_exhausted = rsi1 <= cfg.get("trend_long_rsi_ceiling", 68) and rsi5 <= 72
        momentum_ok = macd_hist >= -abs(_num(row5.get("atr"), 0.0)) * 0.001
    else:
        in_pullback_zone = (
            -pullback_pct <= dist_ema21 <= pullback_pct
            or -pullback_pct <= dist_vwap <= pullback_pct
            or (ema9 * 0.998 <= close <= ema21)
        )
        resume = _num(row1["close"]) < _num(row1["open"]) or pattern1 in {"shooting_star", "bearish_engulfing"}
        not_exhausted = rsi1 >= cfg.get("trend_short_rsi_floor", 32) and rsi5 >= 28
        momentum_ok = macd_hist <= abs(_num(row5.get("atr"), 0.0)) * 0.001

    if in_pullback_zone:
        score += _add_detail(details, "pullback_zone", 22)
    else:
        score += _add_detail(details, "pullback_zone", -12)
        notes.append("not_in_pullback_zone")

    if resume:
        score += _add_detail(details, "resume_candle", 12)
    else:
        score += _add_detail(details, "resume_candle", -6)

    if not_exhausted:
        score += _add_detail(details, "not_exhausted", 8)
    else:
        score += _add_detail(details, "not_exhausted", -10)
        notes.append("trend_entry_exhausted")

    if momentum_ok:
        score += _add_detail(details, "macd_context", 7)

    return direction, score, details, notes


def _score_impulse(
    i1: int,
    df1: pd.DataFrame,
    ctx: RegimeContext,
    cfg: dict[str, Any],
) -> tuple[str | None, int, dict[str, int], list[str]]:
    if ctx.regime != "impulse" or ctx.impulse_dir not in {"up", "down"}:
        return None, 0, {}, []

    row1 = df1.iloc[i1]
    prev1 = df1.iloc[i1 - 1]
    direction = "long" if ctx.impulse_dir == "up" else "short"
    pattern1 = _candle_pattern(row1, prev1)
    close = _num(row1["close"])
    open_ = _num(row1["open"])

    score = 34
    details = {"impulse_base": 34}
    notes: list[str] = []

    if direction == "long" and close > open_ and pattern1 != "shooting_star":
        score += _add_detail(details, "impulse_candle", 14)
    elif direction == "short" and close < open_ and pattern1 != "hammer":
        score += _add_detail(details, "impulse_candle", 14)
    else:
        score += _add_detail(details, "impulse_candle", -12)
        notes.append("weak_impulse_candle")

    if ctx.trend_5m == ctx.impulse_dir or ctx.trend_15m == ctx.impulse_dir:
        score += _add_detail(details, "impulse_trend_support", 12)
    else:
        score += _add_detail(details, "impulse_trend_support", -10)

    return direction, score, details, notes


def _apply_orderbook(
    direction: str,
    score: int,
    details: dict[str, int],
    result: dict[str, Any],
    obi: float,
    ob_analysis: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> int:
    ob = ob_analysis or {}
    obi_avg = _num(ob.get("obi_avg"), obi)
    spread_pct = _num(ob.get("spread_pct"))
    pressure = ob.get("pressure") or ("buy" if obi_avg > 0.16 else "sell" if obi_avg < -0.16 else "neutral")

    result["obi"] = obi_avg
    result["spread_pct"] = spread_pct
    result["orderbook_pressure"] = pressure

    if not ob and abs(obi) < 1e-9:
        return score

    max_spread = cfg.get("max_spread_pct", 0.03)
    if spread_pct > max_spread:
        result["hard_block"] = True
        result["block_reasons"].append(f"spread {spread_pct:.4f}% > max {max_spread:.4f}%")
        return score

    block_obi = cfg.get("obi_hard_block", 0.30)
    soft_obi = cfg.get("obi_soft_align", 0.12)
    aligned = (direction == "long" and obi_avg > soft_obi) or (direction == "short" and obi_avg < -soft_obi)
    opposed = (direction == "long" and obi_avg < -block_obi) or (direction == "short" and obi_avg > block_obi)

    if opposed:
        result["hard_block"] = True
        result["block_reasons"].append(f"orderbook pressure against {direction}: obi={obi_avg:.2f}")
        return score

    if aligned:
        score += _add_detail(details, "obi_aligned", 12)
    elif abs(obi_avg) <= soft_obi:
        score += _add_detail(details, "obi_neutral", 4)
    else:
        score += _add_detail(details, "obi_mild_against", -8)

    if direction == "long" and _bool(ob.get("bid_wall")):
        score += _add_detail(details, "bid_wall", 8)
    if direction == "short" and _bool(ob.get("ask_wall")):
        score += _add_detail(details, "ask_wall", 8)

    micro_bias = _num(ob.get("microprice_bias"))
    if direction == "long" and micro_bias > 0:
        score += _add_detail(details, "microprice", 6)
    elif direction == "short" and micro_bias < 0:
        score += _add_detail(details, "microprice", 6)
    elif abs(micro_bias) > 0:
        score += _add_detail(details, "microprice", -5)

    if _bool(ob.get("pressure_flip")):
        flip_to = ob.get("pressure_flip_to", "neutral")
        if (direction == "long" and flip_to == "buy") or (direction == "short" and flip_to == "sell"):
            score += _add_detail(details, "pressure_flip", 10)

    return score


def _apply_fee_edge(direction: str, score: int, details: dict[str, int], result: dict[str, Any], cfg: dict[str, Any]) -> int:
    margin = cfg.get("margin_half", cfg.get("margin_full", 50.0))
    notional = margin * cfg.get("leverage", 20)
    tp1_pct = cfg.get("tp1_pct", 0.10) / 100.0
    tp2_pct = cfg.get("tp2_pct", 0.18) / 100.0
    maker = cfg.get("maker_fee_pct", 0.015) / 100.0
    taker = cfg.get("taker_fee_pct", 0.045) / 100.0
    entry_fee = maker if cfg.get("use_maker_for_entry", True) else taker
    tp_fee = maker if cfg.get("use_maker_for_tp", True) else taker
    expected_gross = notional * ((tp1_pct * 0.5) + (tp2_pct * 0.5))
    expected_fees = notional * entry_fee + notional * tp_fee
    expected_net = expected_gross - expected_fees
    result["expected_net_tp_usd"] = expected_net

    min_edge = cfg.get("min_expected_tp_net_usd", 0.35)
    if expected_net < min_edge:
        score += _add_detail(details, "fee_edge", -12)
    else:
        score += _add_detail(details, "fee_edge", 4)
    return score


def _confidence(score: int) -> str:
    if score >= 95:
        return "A+"
    if score >= 78:
        return "A"
    if score >= 62:
        return "B"
    return "C"


def score_strategy_setup(
    i15: int,
    df15: pd.DataFrame,
    i5: int,
    df5: pd.DataFrame,
    i1: int,
    df1: pd.DataFrame,
    consecutive_losses: int = 0,
    hour_utc: int = 12,
    cfg: dict[str, Any] | None = None,
    obi: float = 0.0,
    ob_analysis: dict[str, Any] | None = None,
    funding_rate: float = 0.0,
) -> dict[str, Any]:
    result = _empty_result()
    cfg = cfg or {}

    if i15 < 200 or i5 < 35 or i1 < 50:
        return result

    row1 = df1.iloc[i1]
    row5 = df5.iloc[i5]
    row15 = df15.iloc[i15]
    required = [row15.get("ema200"), row5.get("macd"), row1.get("bb_width_pct"), row1.get("atr")]
    if any(v is None or pd.isna(v) for v in required):
        return result

    ctx = detect_market_regime(i15, df15, i5, df5, i1, df1, cfg)
    result.update(
        {
            "regime": ctx.regime,
            "momentum": ctx.momentum_5m,
            "atr_1m": ctx.atr_1m,
            "atr_5m": ctx.atr_5m,
            "bb_width_pct": ctx.bb_width_pct,
            "bb_squeeze": ctx.bb_squeeze,
            "trend_5m": ctx.trend_5m,
            "trend_15m": ctx.trend_15m,
            "adx_15m": ctx.adx_15m,
        }
    )

    candidates: list[dict[str, Any]] = []
    engines = [
        ("trend_pullback", _score_trend_pullback(i5, df5, i1, df1, ctx, cfg)),
        ("range_reversion", _score_range_reversion(i5, df5, i1, df1, ctx, cfg)),
        ("impulse_continuation", _score_impulse(i1, df1, ctx, cfg)),
    ]

    for engine, (direction, score, details, notes) in engines:
        if direction is None:
            continue
        candidates.append(
            {
                "engine": engine,
                "direction": direction,
                "score": score,
                "details": details,
                "notes": notes,
            }
        )

    if not candidates:
        return result

    setup = max(candidates, key=lambda item: item["score"])
    direction = setup["direction"]
    score = int(setup["score"])
    details = dict(setup["details"])
    result["engine"] = setup["engine"]
    result["direction"] = direction

    if setup["engine"] == "range_reversion" and ctx.regime == "impulse":
        result["hard_block"] = True
        result["block_reasons"].append("no fade during impulse")

    if setup["engine"] == "range_reversion" and ctx.regime != "range":
        result["hard_block"] = True
        result["block_reasons"].append(f"range engine blocked in {ctx.regime} regime")

    if setup["engine"] == "range_reversion" and ctx.bb_width_pct > cfg.get("fade_bb_width_hard_max_pct", cfg.get("bb_width_max_pct", 1.2) * 1.6):
        result["hard_block"] = True
        result["block_reasons"].append(f"bb_width too wide for fade: {ctx.bb_width_pct:.2f}%")

    score = _apply_orderbook(direction, score, details, result, obi, ob_analysis, cfg)
    score = _apply_fee_edge(direction, score, details, result, cfg)

    if funding_rate and cfg.get("use_funding_rate", True):
        if direction == "long" and funding_rate < -0.0001:
            score += _add_detail(details, "funding_bias", 6)
        elif direction == "short" and funding_rate > 0.0001:
            score += _add_detail(details, "funding_bias", 6)
        elif direction == "long" and funding_rate > 0.0005:
            score += _add_detail(details, "funding_bias", -8)
        elif direction == "short" and funding_rate < -0.0005:
            score += _add_detail(details, "funding_bias", -8)

    if cfg.get("use_time_filter", True):
        if hour_utc in range(7, 18):
            score += _add_detail(details, "time_filter", 6)
        elif hour_utc in range(1, 6):
            score += _add_detail(details, "time_filter", -4)

    if consecutive_losses >= cfg.get("max_consecutive_losses", 3):
        result["hard_block"] = True
        result["block_reasons"].append("consecutive loss cooldown")

    for note in setup["notes"]:
        if note in {"not_in_pullback_zone", "trend_entry_exhausted"} and setup["engine"] == "trend_pullback":
            score += _add_detail(details, "quality_note", -4)

    result["score"] = int(score)
    result["score_details"] = details
    result["confidence"] = _confidence(int(score))
    return result
