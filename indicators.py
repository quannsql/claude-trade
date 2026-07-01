"""
indicators.py — Rule-based Engine
==================================
Calculate technical indicators and check entry conditions based on strict rules.
Stripped down to prioritize speed, liquidity (OBI), and price action.
"""

import pandas as pd
import numpy as np
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns to OHLCV dataframe. Do not modify original columns."""
    df = df.copy()

    # 1. Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    
    # %B = 0 -> exactly at lower band, %B = 1 -> exactly at upper band
    # %B < 0 -> piercing lower band, %B > 1 -> piercing upper band
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)

    # 2. Average True Range (ATR)
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # 3. Volume Average (for climax detection)
    df["vol_avg20"] = df["volume"].rolling(20).mean()

    # 4. VWAP (Daily UTC reset)
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        trade_date = df["timestamp"].dt.date
        cum_vol_price = (df["close"] * df["volume"]).groupby(trade_date).cumsum()
        cum_vol = df["volume"].groupby(trade_date).cumsum()
        df["vwap"] = cum_vol_price / cum_vol.replace(0, np.nan)
    else:
        # Fallback if no timestamp
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


def is_pinbar(row: pd.Series, direction: str) -> bool:
    """
    Check if the current candle is a pinbar (long wick piercing the band, closing inside).
    For LONG: long lower wick, closing in the upper half.
    For SHORT: long upper wick, closing in the lower half.
    """
    body_top = max(row["close"], row["open"])
    body_bottom = min(row["close"], row["open"])
    body = body_top - body_bottom
    candle_range = row["high"] - row["low"]
    
    if candle_range == 0:
        return False
        
    upper_wick = row["high"] - body_top
    lower_wick = body_bottom - row["low"]
    
    if direction == "long":
        # Long lower wick, short upper wick
        return (lower_wick >= 1.5 * body) and (upper_wick < body) and (body > 0)
    elif direction == "short":
        # Long upper wick, short lower wick
        return (upper_wick >= 1.5 * body) and (lower_wick < body) and (body > 0)
        
    return False


def check_entry_conditions(
    i15: int, df15: pd.DataFrame,
    i1: int, df1: pd.DataFrame,
    obi: float = 0.0,
    obi_delta: float = 0.0,
    cfg: dict | None = None
) -> dict:
    """
    Strict Rule-based Entry Checker.
    
    Returns dict:
    {
        'direction': 'long' | 'short' | None,
        'atr_1m': float,
        'confidence': str,
        'block_reasons': list[str]
    }
    """
    result = {
        "direction": None,
        "atr_1m": 0.0,
        "confidence": "C",
        "block_reasons": []
    }
    
    if i15 < 1 or i1 < 20:
        result["block_reasons"].append("Not enough data")
        return result
        
    row15 = df15.iloc[i15]
    row1 = df1.iloc[i1]
    
    price1 = row1["close"]
    vwap15 = row15.get("vwap")
    bb_pct_b = row1.get("bb_pct_b")
    vol = row1.get("volume")
    vol_avg = row1.get("vol_avg20")
    
    if pd.isna(vwap15) or pd.isna(bb_pct_b):
        result["block_reasons"].append("Missing indicators")
        return result
        
    atr_1m = row1.get("atr")
    result["atr_1m"] = float(atr_1m) if not pd.isna(atr_1m) else 0.0
    
    # Check 1: VWAP Trend Filter
    trend = "long" if price1 > vwap15 else "short"
    
    # Check 2: Order Book Imbalance (Hard Block)
    if trend == "long" and obi < -0.15:
        result["block_reasons"].append(f"OBI {obi:.2f} too low for LONG")
        return result
    if trend == "short" and obi > 0.15:
        result["block_reasons"].append(f"OBI {obi:.2f} too high for SHORT")
        return result
        
    # Check 3: Trigger (Price Action + BB)
    is_vol_climax = (vol is not None and vol_avg is not None and vol > vol_avg)
    
    if trend == "long":
        # Pierced lower band but closed inside/above
        if bb_pct_b < 0 and is_pinbar(row1, "long") and is_vol_climax:
            result["direction"] = "long"
            
            # Confidence boost with OBI Delta
            if obi_delta > 0:
                result["confidence"] = "A+"
            else:
                result["confidence"] = "A"
        else:
            result["block_reasons"].append("No LONG trigger (need BB pierce + Pinbar + Vol)")
            
    elif trend == "short":
        # Pierced upper band but closed inside/below
        if bb_pct_b > 1 and is_pinbar(row1, "short") and is_vol_climax:
            result["direction"] = "short"
            
            # Confidence boost with OBI Delta
            if obi_delta < 0:
                result["confidence"] = "A+"
            else:
                result["confidence"] = "A"
        else:
            result["block_reasons"].append("No SHORT trigger (need BB pierce + Pinbar + Vol)")
            
    return result