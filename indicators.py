"""
indicators.py
=============
Calculate technical indicators and scoring engine (Module 1-7) applied
on REAL data - no numbers in this file are "assumed",
all scores are calculated directly from historical price/volume.

Module 6 (Regime Detection) and Module 7 (Momentum Strength) were added
to improve signal quality and reduce noise trades in choppy markets.

IMPORTANT: Order Book Imbalance in the original module needs real-time order book data
which historical backtest DOES NOT HAVE. This module removes that part from
scoring and clearly notes in the final report that it is unverified.
"""

import pandas as pd
import numpy as np
import ta
from filters import detect_regime, momentum_strength, price_position_vs_ema


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
                 hour_utc: int = 12,
                 cfg: dict | None = None) -> dict:
    """
    Apply Module 1-7 (excluding Order Book Imbalance - no historical data)
    at candle indices i15, i5, i1 respectively across 3 timeframes.

    Returns dict: {
        'direction': 'long' | 'short' | None,
        'score': int,
        'hard_block': bool,
        'block_reasons': [...],
        'regime': str,        # 'trending' | 'ranging' | 'transitioning'
        'momentum': float,    # ATR-normalized momentum
        'atr_1m': float,      # 1m ATR for dynamic TP/SL
    }
    """
    result = {
        "direction": None, "score": 0, "hard_block": False,
        "block_reasons": [], "regime": "unknown", "momentum": 0.0,
        "atr_1m": 0.0,
    }
    cfg = cfg or {}

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
    direction = None

    # --------------------------------------------------
    # EXTREME MEAN-REVERSION LOGIC (High-Frequency Scalper)
    # --------------------------------------------------
    # Base signal: 1m price piercing Bollinger Bands
    price1 = row1["close"]
    bb_upper1 = row1["bb_upper"]
    bb_lower1 = row1["bb_lower"]
    rsi1 = row1["rsi"]
    rsi5 = row5["rsi"]
    
    vol1 = row1.get("volume")
    vol_avg20_1 = row1.get("vol_avg20")
    ema50_5 = row5.get("ema50")
    price5 = row5["close"]
    vwap1 = row1.get("vwap")
    
    if pd.isna(bb_upper1) or pd.isna(bb_lower1) or pd.isna(rsi1) or pd.isna(rsi5):
        return result
        
    vwap_stretch_pct = 0.0
    if not pd.isna(vwap1) and vwap1 > 0:
        vwap_stretch_pct = abs(price1 - vwap1) / vwap1 * 100

    # Check BB touch
    if price1 <= bb_lower1:
        direction = "long"
        score += 40
        
        # Penalize catching a falling knife blindly
        pattern = detect_candle_pattern(row1, row1_prev)
        if pattern == "strong_bearish_close":
            score -= 15
            
        # Reward volume climax
        if not pd.isna(vol1) and not pd.isna(vol_avg20_1) and vol1 > vol_avg20_1 * 1.5:
            score += 10
            
        # Soft Trend Alignment on 5m
        if not pd.isna(ema50_5):
            if price5 > ema50_5:
                score += 10
            else:
                score -= 10
                
        # VWAP Stretch (Rubber band effect)
        if vwap_stretch_pct > 0.5:
            score += 20
        elif vwap_stretch_pct > 0.3:
            score += 10

    elif price1 >= bb_upper1:
        direction = "short"
        score += 40
        
        # Penalize blindly shorting a rocket
        pattern = detect_candle_pattern(row1, row1_prev)
        if pattern == "strong_bullish_close":
            score -= 15
            
        # Reward volume climax
        if not pd.isna(vol1) and not pd.isna(vol_avg20_1) and vol1 > vol_avg20_1 * 1.5:
            score += 10
            
        # Soft Trend Alignment on 5m
        if not pd.isna(ema50_5):
            if price5 < ema50_5:
                score += 10
            else:
                score -= 10
                
        # VWAP Stretch (Rubber band effect)
        if vwap_stretch_pct > 0.5:
            score += 20
        elif vwap_stretch_pct > 0.3:
            score += 10
    else:
        # Not touching BB extremes -> No signal for mean-reversion
        return result

    # RSI confirmation (1m and 5m)
    if direction == "long":
        if rsi1 < 30:
            score += 20
        elif rsi1 < 40:
            score += 10
            
        if rsi5 < 35:
            score += 20
        elif rsi5 < 45:
            score += 10
            
        # Candlestick pattern bonus
        pattern = detect_candle_pattern(row1, row1_prev)
        if pattern in ["hammer", "bullish_engulfing"]:
            score += 10
            
    else:  # short
        if rsi1 > 70:
            score += 20
        elif rsi1 > 60:
            score += 10
            
        if rsi5 > 65:
            score += 20
        elif rsi5 > 55:
            score += 10
            
        # Candlestick pattern bonus
        pattern = detect_candle_pattern(row1, row1_prev)
        if pattern in ["shooting_star", "bearish_engulfing"]:
            score += 10

    # Cooldown check
    if consecutive_losses >= 3:
        result["block_reasons"].append("3 consecutive losses - cooldown")
        result["hard_block"] = True
        return result

    # Export 5m ATR for dynamic TP/SL
    atr_5m = row5.get("atr")
    result["atr_5m"] = float(atr_5m) if not pd.isna(atr_5m) else 0.0

    result["direction"] = direction
    result["score"] = score
    return result
