"""
indicators.py
=============
Calculate technical indicators and scoring engine (Module 1-5) applied
on REAL data - no numbers in this file are "assumed",
all scores are calculated directly from historical price/volume.

IMPORTANT: Order Book Imbalance in the original module needs real-time order book data
which historical backtest DOES NOT HAVE. This module removes that part from
scoring and clearly notes in the final report that it is unverified.
"""

import pandas as pd
import numpy as np
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns to OHLCV dataframe. Do not modify original columns."""
    df = df.copy()

    df["ema9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    stoch = ta.momentum.StochRSIIndicator(df["close"])
    df["stoch_k"] = stoch.stochrsi_k() * 100
    df["stoch_d"] = stoch.stochrsi_d() * 100

    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width_pct"] = (df["bb_upper"] - df["bb_lower"]) / df["close"] * 100

    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr_pct"] = df["atr"] / df["close"] * 100

    # VWAP (rolling by day - approximation since data is not session separated)
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    df["vol_avg20"] = df["volume"].rolling(20).mean()

    return df


def detect_candle_pattern(row, prev_row) -> str:
    """
    Detect simplified candlestick patterns (engulfing, hammer, shooting star)
    based on candle body and wicks. Returns pattern name or 'none'.
    """
    body = abs(row["close"] - row["open"])
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return "none"

    upper_wick = row["high"] - max(row["close"], row["open"])
    lower_wick = min(row["close"], row["open"]) - row["low"]

    is_bullish = row["close"] > row["open"]
    is_bearish = row["close"] < row["open"]

    prev_body_top = max(prev_row["close"], prev_row["open"])
    prev_body_bottom = min(prev_row["close"], prev_row["open"])

    if is_bullish and prev_row["close"] < prev_row["open"]:
        if row["close"] > prev_body_top and row["open"] < prev_body_bottom:
            return "bullish_engulfing"

    if is_bearish and prev_row["close"] > prev_row["open"]:
        if row["close"] < prev_body_bottom and row["open"] > prev_body_top:
            return "bearish_engulfing"

    if lower_wick >= 2 * body and upper_wick < body and body > 0:
        return "hammer"

    if upper_wick >= 2 * body and lower_wick < body and body > 0:
        return "shooting_star"

    if is_bullish and body / candle_range > 0.7:
        return "strong_bullish_close"
    if is_bearish and body / candle_range > 0.7:
        return "strong_bearish_close"

    if body / candle_range < 0.1:
        return "doji"

    return "none"


def _atr_pct_at(df5: pd.DataFrame, i5: int):
    val = df5.iloc[i5]["atr_pct"]
    return None if pd.isna(val) else val


def score_setup(i15: int, df15: pd.DataFrame,
                 i5: int, df5: pd.DataFrame,
                 i1: int, df1: pd.DataFrame,
                 consecutive_losses: int = 0,
                 hour_utc: int = 12) -> dict:
    """
    Apply Module 1-5 (excluding Order Book Imbalance - no historical data)
    at candle indices i15, i5, i1 respectively across 3 timeframes.

    Returns dict: {
        'direction': 'long' | 'short' | None,
        'score': int,
        'hard_block': bool,
        'block_reasons': [...]
    }
    """
    result = {"direction": None, "score": 0, "hard_block": False, "block_reasons": []}

    if i15 < 200 or i5 < 30 or i1 < 25:
        return result

    row15 = df15.iloc[i15]
    row5 = df5.iloc[i5]
    row5_prev = df5.iloc[i5 - 1]
    row1 = df1.iloc[i1]
    row1_prev = df1.iloc[i1 - 1]

    if pd.isna(row15["ema200"]) or pd.isna(row5["macd"]) or pd.isna(row1["bb_width_pct"]):
        return result

    score = 0

    # --------------------------------------------------
    # MODULE 1 — TREND (15m), max 30 pts
    # --------------------------------------------------
    price15 = row15["close"]
    ema50_15 = row15["ema50"]
    ema200_15 = row15["ema200"]

    if pd.isna(ema50_15) or pd.isna(ema200_15):
        return result

    ema_gap_pct = abs(ema50_15 - ema200_15) / price15 * 100

    if price15 > ema50_15 > ema200_15:
        direction = "long"
        score += 15 if ema_gap_pct > 0.15 else 5
    elif price15 < ema50_15 < ema200_15:
        direction = "short"
        score += 15 if ema_gap_pct > 0.15 else 5
    else:
        result["block_reasons"].append("EMA50/EMA200 crossover or price in consolidation zone (15m)")
        result["hard_block"] = True
        return result

    ema21_5 = row5["ema21"]
    ema21_5_prev = row5_prev["ema21"]
    price5 = row5["close"]
    if pd.isna(ema21_5) or pd.isna(ema21_5_prev):
        return result

    if direction == "long":
        if price5 > ema21_5 and ema21_5 > ema21_5_prev:
            score += 10
        elif price5 > ema21_5:
            score += 5
    else:
        if price5 < ema21_5 and ema21_5 < ema21_5_prev:
            score += 10
        elif price5 < ema21_5:
            score += 5

    window = df15.iloc[max(0, i15 - 10):i15 + 1]
    highs = window["high"].values
    lows = window["low"].values
    if direction == "long" and len(highs) >= 4 and highs[-1] > highs[-4]:
        score += 5
    elif direction == "short" and len(lows) >= 4 and lows[-1] < lows[-4]:
        score += 5

    # --------------------------------------------------
    # MODULE 2 — MOMENTUM (5m), max 25 pts
    # --------------------------------------------------
    rsi5 = row5["rsi"]
    macd5 = row5["macd"]
    sig5 = row5["macd_signal"]
    macd5_prev = row5_prev["macd"]
    sig5_prev = row5_prev["macd_signal"]
    hist5 = row5["macd_hist"]
    hist5_prev = row5_prev["macd_hist"]

    if pd.isna(rsi5) or pd.isna(macd5) or pd.isna(sig5) or i5 < 3:
        return result

    rsi5_3back = df5.iloc[i5 - 3]["rsi"]

    if direction == "long":
        if 45 <= rsi5 <= 60 and rsi5 > rsi5_3back:
            score += 10
        elif 40 <= rsi5 <= 65 and rsi5 > row5_prev["rsi"]:
            score += 6
        elif rsi5 < 35:
            score += 3

        crossover_up = macd5_prev < sig5_prev and macd5 > sig5
        if crossover_up and hist5 > hist5_prev:
            score += 10
        elif macd5 > sig5 and hist5 > 0:
            score += 5
        elif macd5 < sig5 and hist5 < hist5_prev:
            result["block_reasons"].append("MACD bearish divergence (5m)")
            result["hard_block"] = True
    else:
        if 40 <= rsi5 <= 55 and rsi5 < rsi5_3back:
            score += 10
        elif 35 <= rsi5 <= 70 and rsi5 < row5_prev["rsi"]:
            score += 6
        elif rsi5 > 65:
            score += 3

        crossover_down = macd5_prev > sig5_prev and macd5 < sig5
        if crossover_down and hist5 < hist5_prev:
            score += 10
        elif macd5 < sig5 and hist5 < 0:
            score += 5
        elif macd5 > sig5 and hist5 > hist5_prev:
            result["block_reasons"].append("MACD bullish divergence (5m)")
            result["hard_block"] = True

    if result["hard_block"]:
        return result

    stoch_k = row5["stoch_k"]
    stoch_k_prev = row5_prev["stoch_k"]
    stoch_d = row5["stoch_d"]
    if not pd.isna(stoch_k) and not pd.isna(stoch_k_prev):
        if direction == "long":
            if stoch_k_prev < 20 and stoch_k > stoch_d:
                score += 5
            elif 20 <= stoch_k <= 50 and stoch_k > stoch_k_prev:
                score += 3
            elif stoch_k > 85:
                result["block_reasons"].append("Stoch RSI too overbought for long (5m)")
                result["hard_block"] = True
        else:
            if stoch_k_prev > 80 and stoch_k < stoch_d:
                score += 5
            elif 50 <= stoch_k <= 80 and stoch_k < stoch_k_prev:
                score += 3
            elif stoch_k < 15:
                result["block_reasons"].append("Stoch RSI too oversold for short (5m)")
                result["hard_block"] = True

    if result["hard_block"]:
        return result

    # --------------------------------------------------
    # MODULE 3 — PRICE ACTION (1m), max 25 pts
    # --------------------------------------------------
    pattern = detect_candle_pattern(row1, row1_prev)

    if pattern == "doji":
        result["block_reasons"].append("Doji candle at entry (1m) - indecision")
        result["hard_block"] = True
        return result

    if direction == "long":
        if pattern in ("bearish_engulfing", "shooting_star"):
            result["block_reasons"].append("Bearish candle pattern at long entry (1m)")
            result["hard_block"] = True
            return result
        if pattern == "bullish_engulfing":
            score += 10
        elif pattern == "hammer":
            score += 8
        elif pattern == "strong_bullish_close":
            score += 5
    else:
        if pattern in ("bullish_engulfing", "hammer"):
            result["block_reasons"].append("Bullish candle pattern at short entry (1m)")
            result["hard_block"] = True
            return result
        if pattern == "bearish_engulfing":
            score += 10
        elif pattern == "shooting_star":
            score += 8
        elif pattern == "strong_bearish_close":
            score += 5

    if i1 >= 30:
        lookback = df1.iloc[i1 - 30:i1]
        price1 = row1["close"]
        touch_zone = lookback[(abs(lookback["low"] - price1) / price1 < 0.0005) |
                               (abs(lookback["high"] - price1) / price1 < 0.0005)]
        touches = len(touch_zone)
        if touches >= 2:
            score += 8
        elif touches == 1:
            score += 4

    vwap1 = row1["vwap"]
    if not pd.isna(vwap1):
        if direction == "long":
            if row1_prev["close"] < vwap1 and row1["close"] > vwap1:
                score += 7
            elif row1["close"] > vwap1:
                score += 5
        else:
            if row1_prev["close"] > vwap1 and row1["close"] < vwap1:
                score += 7
            elif row1["close"] < vwap1:
                score += 5

    if i1 >= 2:
        last3 = df1.iloc[i1 - 2:i1 + 1]
        if direction == "long" and (last3["close"] < last3["open"]).all():
            result["block_reasons"].append("3 consecutive red 1m candles at long entry")
            result["hard_block"] = True
            return result
        if direction == "short" and (last3["close"] > last3["open"]).all():
            result["block_reasons"].append("3 consecutive green 1m candles at short entry")
            result["hard_block"] = True
            return result

    # --------------------------------------------------
    # MODULE 4 — VOLUME, max 20 pts
    # --------------------------------------------------
    vol_avg = row1["vol_avg20"]
    vol_now = row1["volume"]
    if pd.isna(vol_avg) or vol_avg == 0:
        return result

    vol_ratio = vol_now / vol_avg

    if vol_ratio < 1.0:
        result["block_reasons"].append("Entry candle volume lower than average (1m)")
        result["hard_block"] = True
        return result
    elif vol_ratio > 5.0:
        result["block_reasons"].append("Abnormal volume spike >500%")
        result["hard_block"] = True
        return result
    elif vol_ratio >= 2.0:
        score += 8
    elif vol_ratio >= 1.5:
        score += 5
    else:
        score += 2

    if i1 >= 2:
        last3vol = df1.iloc[i1 - 2:i1 + 1]["volume"].values
        if len(last3vol) == 3 and last3vol[0] < last3vol[1] < last3vol[2]:
            score += 6
        elif len(last3vol) == 3 and last3vol[1] < last3vol[2]:
            score += 3

    # NOTE: Order Book Imbalance is removed from scoring because historical backtests
    # do not have reliable order book data. Needs separate verification when live.

    # --------------------------------------------------
    # MODULE 5 — ENVIRONMENT FILTER (bonus/penalty)
    # --------------------------------------------------
    if 13 <= hour_utc < 17:
        score += 5
    elif 8 <= hour_utc < 13:
        score += 2
    elif 2 <= hour_utc < 8:
        score -= 5
    else:
        score -= 3

    atr5 = _atr_pct_at(df5, i5)
    if atr5 is not None:
        if 0.05 <= atr5 <= 0.20:
            score += 5
        elif atr5 > 0.20:
            score -= 5
        elif atr5 < 0.05:
            result["block_reasons"].append("ATR(5m) too low - insufficient volatility")
            result["hard_block"] = True
            return result

    if consecutive_losses == 1:
        score -= 5
    elif consecutive_losses == 2:
        score -= 10
    elif consecutive_losses >= 3:
        result["block_reasons"].append("3 consecutive losses - cooldown")
        result["hard_block"] = True
        return result

    bb_width = row1["bb_width_pct"]
    if not pd.isna(bb_width) and bb_width < 0.1:
        result["block_reasons"].append("Bollinger Band squeeze (1m) - waiting for breakout")
        result["hard_block"] = True
        return result

    result["direction"] = direction
    result["score"] = score
    return result
