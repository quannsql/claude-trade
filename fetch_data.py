"""
fetch_data.py
=============
Fetch REAL historical data (OHLCV) from Binance Futures via CCXT.

Binance is used instead of Hyperliquid for fetching because Binance has
complete and stable 1-minute historical data for backtesting. The strategy logic,
once validated, can be transferred to run live on Hyperliquid or other exchanges -
this is just the RESEARCH phase, not execution.

Usage:
    python fetch_data.py
"""

import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT"]
TIMEFRAMES = ["1m", "5m", "15m"]
DAYS_BACK = 90  # 3 months
OUTPUT_DIR = "data"

exchange = ccxt.binance({
    "options": {"defaultType": "future"},  # Binance USDT-M Futures
    "enableRateLimit": True,
})


def fetch_ohlcv_full(symbol: str, timeframe: str, days_back: int) -> pd.DataFrame:
    """
    Download full OHLCV for the specified period, with automatic pagination
    since each request only returns a max of ~1000-1500 candles.
    """
    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - days_back * 24 * 60 * 60 * 1000
    all_candles = []

    print(f"  Downloading {symbol} [{timeframe}] from {days_back} days ago...")

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1500)
        except Exception as e:
            print(f"    Fetch error, retrying in 5s: {e}")
            time.sleep(5)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        since = candles[-1][0] + ms_per_candle

        # Reached present time
        if since > exchange.milliseconds():
            break

        # Respect rate limit
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df.drop_duplicates(subset="timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            df = fetch_ohlcv_full(symbol, tf, DAYS_BACK)
            fname = symbol.replace("/", "") + f"_{tf}.csv"
            path = os.path.join(OUTPUT_DIR, fname)
            df.to_csv(path, index=False)
            print(f"  ✅ Saved {len(df)} candles to {path}")
            print(f"     Time range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    print("\nData download complete. Next, run: python run_backtest.py")


if __name__ == "__main__":
    main()
