"""
filters.py
==========
Advanced market regime detection and entry quality filters.
These supplement the base scoring engine in indicators.py to reduce
noise trades and improve win rate.
"""

import pandas as pd
import numpy as np


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series | None:
    """
    Calculate Average Directional Index (ADX) from OHLC data.
    Returns the ADX series, or None if insufficient data.
    """
    if len(df) < period + 2:
        return None

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # True Range
    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Directional Movement
    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    for i in range(1, len(df)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Smoothed averages (Wilder's smoothing)
    atr = np.zeros(len(df))
    plus_di_smooth = np.zeros(len(df))
    minus_di_smooth = np.zeros(len(df))

    atr[period] = np.mean(tr[1 : period + 1])
    plus_di_smooth[period] = np.mean(plus_dm[1 : period + 1])
    minus_di_smooth[period] = np.mean(minus_dm[1 : period + 1])

    for i in range(period + 1, len(df)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_di_smooth[i] = (plus_di_smooth[i - 1] * (period - 1) + plus_dm[i]) / period
        minus_di_smooth[i] = (minus_di_smooth[i - 1] * (period - 1) + minus_dm[i]) / period

    # +DI / -DI
    plus_di = np.zeros(len(df))
    minus_di = np.zeros(len(df))
    dx = np.zeros(len(df))

    for i in range(period, len(df)):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_di_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_di_smooth[i] / atr[i]
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX (smoothed DX)
    adx = np.zeros(len(df))
    first_adx_idx = 2 * period
    if first_adx_idx < len(df):
        adx[first_adx_idx] = np.mean(dx[period : first_adx_idx + 1])
        for i in range(first_adx_idx + 1, len(df)):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return pd.Series(adx, index=df.index)


def detect_regime(df15: pd.DataFrame, i15: int, adx_period: int = 14) -> str:
    """
    Detect market regime using ADX on 15m timeframe.

    Returns:
        'trending'      - ADX > 25, strong directional movement
        'ranging'        - ADX < 18, choppy sideways market
        'transitioning'  - ADX 18-25, unclear
    """
    lookback_needed = adx_period * 3 + 5
    start_idx = max(0, i15 - lookback_needed)
    window = df15.iloc[start_idx : i15 + 1].copy().reset_index(drop=True)

    adx_series = calculate_adx(window, adx_period)
    if adx_series is None or len(adx_series) == 0:
        return "unknown"

    adx_val = adx_series.iloc[-1]

    if adx_val > 25:
        return "trending"
    elif adx_val < 18:
        return "ranging"
    else:
        return "transitioning"


def detect_trend_regime(df15: pd.DataFrame, i15: int, df5: pd.DataFrame, i5: int) -> str:
    """
    Phân loại regime cho dual-mode engine:
        'trend_up'   — EMA21(5m) dốc lên rõ + EMA9>EMA21>EMA50 xếp tầng trên 5m (nhanh hơn)
        'trend_down' — ngược lại
        'ranging'    — còn lại (mặc định, an toàn cho mean-reversion)

    Trend regime yêu cầu CẢ HAI điều kiện (slope 5m + stack 5m) để tránh
    flip-flop khi thị trường chỉ nhích nhẹ, kết hợp 15m như xác nhận phụ.
    """
    if i15 < 6 or i5 < 5 or i15 >= len(df15) or i5 >= len(df5):
        return "ranging"

    row15 = df15.iloc[i15]
    row5 = df5.iloc[i5]

    ema21_5_now = row5.get("ema21")
    ema21_5_5ago = df5.iloc[i5 - 5].get("ema21")
    slope_5m_pct = 0.0
    if (ema21_5_now is not None and ema21_5_5ago is not None
            and not pd.isna(ema21_5_now) and not pd.isna(ema21_5_5ago)
            and ema21_5_5ago > 0):
        slope_5m_pct = (ema21_5_now - ema21_5_5ago) / ema21_5_5ago * 100

    ema9_5 = row5.get("ema9")
    ema21_5 = row5.get("ema21")
    ema50_5 = row5.get("ema50")
    if any(v is None or pd.isna(v) for v in (ema9_5, ema21_5, ema50_5)):
        return "ranging"

    stack_up = ema9_5 > ema21_5 > ema50_5
    stack_down = ema9_5 < ema21_5 < ema50_5

    if slope_5m_pct > 0.08 and stack_up:
        return "trend_up"
    if slope_5m_pct < -0.08 and stack_down:
        return "trend_down"
    return "ranging"


def compute_risk_sizing(price: float, atr_5m: float, cfg: dict, score_full: bool = True) -> dict:
    """
    Size vị thế theo rủi ro USD cố định (loss ≈ win ≈ risk_per_trade_usd):

        sl_pct   = clamp(sl_atr_mult × ATR%, sl_pct_min, sl_pct_max)
        notional = risk_usd / sl_pct

    → khi SL % chạm, lỗ đúng ~risk_usd bất kể volatility. Hard dollar SL chỉ
    còn là backstop (risk_usd × hard_sl_backstop_mult), không phải stop chính.

    Trả về dict {notional, margin, sl_pct, risk_usd}.
    """
    leverage = cfg.get("leverage", 20)
    risk_usd = cfg.get("risk_per_trade_usd", 3.0)
    if not score_full:
        risk_usd *= cfg.get("risk_half_scale", 0.6)

    sl_mult = cfg.get("sl_atr_mult", 1.5)
    sl_pct_min = cfg.get("sl_pct_min", 0.10)
    sl_pct_max = cfg.get("sl_pct_max", 0.40)

    if price <= 0:
        return {"notional": 0.0, "margin": 0.0, "sl_pct": sl_pct_min, "risk_usd": risk_usd}

    atr_pct = (atr_5m / price * 100) if atr_5m > 0 else 0.0
    sl_pct = atr_pct * sl_mult
    if sl_pct <= 0:
        sl_pct = cfg.get("sl_pct", 0.20)
    sl_pct = min(max(sl_pct, sl_pct_min), sl_pct_max)

    notional = risk_usd / (sl_pct / 100)
    max_notional = cfg.get("margin_full", 100.0) * leverage
    min_notional = cfg.get("min_notional_usd", 20.0)
    notional = min(max(notional, min_notional), max_notional)

    return {
        "notional": notional,
        "margin": notional / leverage,
        "sl_pct": sl_pct,
        "risk_usd": risk_usd,
    }


def estimate_fee_edge(price: float, atr_5m: float, cfg: dict) -> dict:
    """
    FEE GATE — kiểm tra lệnh có đủ edge để trả phí round-trip không.

    Bài học từ live 02/07: BNB ATR ~0.06% → TP1 ~0.06% trong khi phí
    taker-taker = 0.09% → mọi lệnh đều ÂM ngay từ lúc đặt. Fee ăn 4.3x gross.

    Điều kiện:
      viable_maker: TP1_est >= maker_in + taker_out + fee_edge_min_pct
      viable_taker: TP1_est >= (taker_in + taker_out) × taker_rt_mult + fee_edge_min_pct

    v3.2 FEE: taker round-trip phải chỉ là PHẦN NHỎ của TP1 (mặc định ×2 dư dả),
    không phải "vừa đủ trả phí". Live 02/07: 3 lệnh BTC taker entry với TP1 est
    0.150% vs required 0.130% (biên 0.02%!) — tất cả time_stop thua, phí taker
    2 chiều $0.90/lệnh trên notional $1000-2000.

    - Không viable_maker → BỎ setup hoàn toàn (không có cách nào dương EV)
    - viable_maker nhưng không viable_taker → chỉ được vào maker,
      kể cả khi score đạt ngưỡng taker
    """
    taker = cfg.get("taker_fee_pct", 0.045)
    maker = cfg.get("maker_fee_pct", 0.015)
    min_edge = cfg.get("fee_edge_min_pct", 0.04)
    taker_rt_mult = cfg.get("taker_rt_mult", 2.0)

    atr_pct = (atr_5m / price * 100) if (price > 0 and atr_5m > 0) else 0.0
    if cfg.get("use_dynamic_tp_sl", False) and atr_pct > 0:
        tp1_pct_est = min(atr_pct * cfg.get("tp1_atr_mult", 1.5),
                          cfg.get("tp1_pct_max", 0.30))
    else:
        tp1_pct_est = cfg.get("tp1_pct", 0.10)

    required_maker = maker + taker + min_edge
    # Taker path: vào taker nhưng TP1 thoát bằng Alo MAKER (use_maker_for_tp)
    # → chi phí người THẮNG = taker_in + maker_out (0.06%), không phải taker×2.
    # Vế thua (SL taker) đã được che bằng hệ số ×taker_rt_mult.
    # Bản cũ tính taker×2×2 = 0.24% → cấm tuyệt đối taker kể cả khi lời rõ.
    required_taker = (taker + maker) * taker_rt_mult + min_edge

    return {
        "tp1_pct_est": tp1_pct_est,
        "atr_pct": atr_pct,
        "required_maker_pct": required_maker,
        "required_taker_pct": required_taker,
        "viable_maker": tp1_pct_est >= required_maker,
        "viable_taker": tp1_pct_est >= required_taker,
    }


def momentum_strength(df5: pd.DataFrame, i5: int, lookback: int = 5) -> float | None:
    """
    Calculate ATR-normalized momentum on 5m timeframe.

    Returns momentum as multiple of ATR (e.g., 1.5 means price moved 1.5x ATR
    in the last `lookback` bars). Returns None if insufficient data.
    """
    if i5 < lookback or i5 >= len(df5):
        return None

    row = df5.iloc[i5]
    atr = row.get("atr")
    if pd.isna(atr) or atr <= 0:
        return None

    close_now = row["close"]
    close_prev = df5.iloc[i5 - lookback]["close"]
    raw_momentum = close_now - close_prev

    return raw_momentum / atr


def price_position_vs_ema(df1: pd.DataFrame, i1: int) -> dict:
    """
    Analyze price position relative to short-term EMAs on 1m timeframe.

    Returns dict with:
        'dist_ema9_pct':  % distance from EMA9
        'dist_ema21_pct': % distance from EMA21
        'price_above_ema9': bool
        'price_above_ema21': bool
        'ema9_slope': positive = rising, negative = falling
    """
    if i1 < 1 or i1 >= len(df1):
        return {}

    row = df1.iloc[i1]
    prev = df1.iloc[i1 - 1]
    close = row["close"]
    ema9 = row.get("ema9")
    ema21 = row.get("ema21")
    ema9_prev = prev.get("ema9")

    result = {}

    if not pd.isna(ema9) and close > 0:
        result["dist_ema9_pct"] = (close - ema9) / close * 100
        result["price_above_ema9"] = close > ema9
        if not pd.isna(ema9_prev):
            result["ema9_slope"] = ema9 - ema9_prev

    if not pd.isna(ema21) and close > 0:
        result["dist_ema21_pct"] = (close - ema21) / close * 100
        result["price_above_ema21"] = close > ema21

    return result


def compute_dynamic_levels(
    entry_price: float,
    direction: str,
    atr_1m: float,
    tp1_atr_mult: float = 1.5,
    tp2_atr_mult: float = 3.0,
    sl_atr_mult: float = 1.2,
    tp1_pct_max: float = 0.30,
    tp2_pct_max: float = 0.60,
    sl_pct_max: float = 0.40,
    sl_pct_min: float = 0.0,
) -> dict:
    """
    Compute ATR-based dynamic TP/SL levels with max caps.

    Instead of fixed percentage TP/SL, these adapt to current volatility:
    - Low volatility → tighter targets (easier to hit)
    - High volatility → wider targets (captures larger moves)
    - Max caps → ensure targets don't become unrealistically wide

    Returns dict with tp1_price, tp2_price, sl_price.
    """
    atr_pct = (atr_1m / entry_price * 100) if entry_price > 0 and atr_1m > 0 else 0

    tp1_pct_calc = atr_pct * tp1_atr_mult
    tp2_pct_calc = atr_pct * tp2_atr_mult
    sl_pct_calc = atr_pct * sl_atr_mult

    tp1_pct = min(tp1_pct_calc, tp1_pct_max)
    tp2_pct = min(tp2_pct_calc, tp2_pct_max)
    sl_pct = min(max(sl_pct_calc, sl_pct_min), sl_pct_max)

    if direction == "long":
        tp1_price = entry_price * (1 + tp1_pct / 100)
        tp2_price = entry_price * (1 + tp2_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)
    else:
        tp1_price = entry_price * (1 - tp1_pct / 100)
        tp2_price = entry_price * (1 - tp2_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)

    return {
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "sl_price": sl_price,
        "tp1_pct": tp1_pct,
        "tp2_pct": tp2_pct,
        "sl_pct": sl_pct,
    }
