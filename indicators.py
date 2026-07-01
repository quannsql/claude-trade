"""
indicators.py — High-Frequency Simplified Engine
================================================
Tối giản hóa tính toán (BB, RSI) và mở rộng điều kiện (OR logic) 
giúp bot trade tần suất cao nhưng vẫn có độ chính xác nhờ RSI và OBI.
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
    
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)

    # 2. RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # 3. Average True Range (ATR) - Fallback
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # 4. Volume Average (for climax detection)
    df["vol_avg20"] = df["volume"].rolling(20).mean()

    # 5. VWAP (Daily UTC reset) - Optional filter
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        trade_date = df["timestamp"].dt.date
        cum_vol_price = (df["close"] * df["volume"]).groupby(trade_date).cumsum()
        cum_vol = df["volume"].groupby(trade_date).cumsum()
        df["vwap"] = cum_vol_price / cum_vol.replace(0, np.nan)
    else:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


def check_entry_conditions(
    i15: int, df15: pd.DataFrame,
    i1: int, df1: pd.DataFrame,
    obi: float = 0.0,
    obi_delta: float = 0.0,
    cfg: dict | None = None
) -> dict:
    """
    Simplified High-Frequency OR-logic Checker.
    """
    result = {
        "direction": None,
        "atr_1m": 0.0,
        "confidence": "B",
        "block_reasons": []
    }
    
    if i1 < 20:
        result["block_reasons"].append("Not enough data")
        return result
        
    row1 = df1.iloc[i1]
    
    price1 = row1["close"]
    bb_pct_b = row1.get("bb_pct_b")
    bb_lower = row1.get("bb_lower")
    bb_upper = row1.get("bb_upper")
    rsi1 = row1.get("rsi")
    vol = row1.get("volume")
    vol_avg = row1.get("vol_avg20")
    
    if pd.isna(bb_pct_b) or pd.isna(rsi1):
        result["block_reasons"].append("Missing indicators")
        return result
        
    atr_1m = row1.get("atr")
    result["atr_1m"] = float(atr_1m) if not pd.isna(atr_1m) else 0.0
    
    # ── OBI FILTER ──
    # Chặn nếu Order Book hoàn toàn bất lợi (chống xả hàng/bơm thổi mạnh)
    obi_block_threshold = 0.25
    allow_long = True
    allow_short = True
    
    if obi < -obi_block_threshold:
        allow_long = False
        result["block_reasons"].append(f"OBI {obi:.2f} quá thấp, chặn LONG")
        
    if obi > obi_block_threshold:
        allow_short = False
        result["block_reasons"].append(f"OBI {obi:.2f} quá cao, chặn SHORT")
        
    if not allow_long and not allow_short:
        return result
        
    # ── TRIGGER CONDITIONS (OR LOGIC) ──
    # LONG conditions
    is_bb_touch_long = (bb_pct_b <= 0 or price1 <= bb_lower)
    is_vol_climax = (vol is not None and vol_avg is not None and vol > vol_avg * 2.0)
    is_extreme_vol = (vol is not None and vol_avg is not None and vol > vol_avg * 3.0)
    
    setup1_long = is_bb_touch_long and (rsi1 < 40)
    setup2_long = (rsi1 < 25)
    setup3_long = is_bb_touch_long and is_vol_climax
    setup4_long = is_extreme_vol and (obi > 0.3) and (price1 > bb_upper)  # Trend following (Bơm)
    
    # SHORT conditions
    is_bb_touch_short = (bb_pct_b >= 1 or price1 >= bb_upper)
    setup1_short = is_bb_touch_short and (rsi1 > 60)
    setup2_short = (rsi1 > 75)
    setup3_short = is_bb_touch_short and is_vol_climax
    setup4_short = is_extreme_vol and (obi < -0.3) and (price1 < bb_lower) # Trend following (Xả)

    if allow_long and (setup1_long or setup2_long or setup3_long or setup4_long):
        result["direction"] = "long"
        result["confidence"] = "A" if (setup2_long or setup3_long or setup4_long) else "B"
        
    elif allow_short and (setup1_short or setup2_short or setup3_short or setup4_short):
        result["direction"] = "short"
        result["confidence"] = "A" if (setup2_short or setup3_short or setup4_short) else "B"
        
    else:
        result["block_reasons"].append("No trigger matched")
        
    return result