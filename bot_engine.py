import time
import os
import logging
import asyncio
import pandas as pd
from datetime import datetime, timezone
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account
from eth_account.signers.local import LocalAccount

from indicators import add_indicators, score_setup
from run_backtest import CONFIG

# Configure logger
logger = logging.getLogger("bot_engine")
logger.setLevel(logging.INFO)
# We will attach handlers in app.py for the dashboard, but for now just console:
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

PRIVATE_KEY = os.environ.get("HL_PRIVATE_KEY", "")
API_URL = constants.TESTNET_API_URL

def fetch_hyperliquid_candles(info: Info, coin: str, interval: str, lookback_bars: int) -> pd.DataFrame:
    """Fetch candles from Hyperliquid and convert to standard DataFrame"""
    end_time = int(time.time() * 1000)
    
    interval_ms = 0
    if interval == "1m":
        interval_ms = 60 * 1000
    elif interval == "5m":
        interval_ms = 5 * 60 * 1000
    elif interval == "15m":
        interval_ms = 15 * 60 * 1000
        
    start_time = end_time - (lookback_bars * interval_ms)
    
    req = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time
        }
    }
    
    res = info.post("/info", req)
    
    if not res:
        return pd.DataFrame()
        
    records = []
    for c in res:
        records.append({
            "timestamp": pd.to_datetime(c["t"], unit="ms", utc=True),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"])
        })
        
    df = pd.DataFrame(records)
    if not df.empty and len(df) > 30: 
        df = add_indicators(df)
        
    return df

def place_orders(exchange: Exchange, coin: str, direction: str, price: float, margin: float):
    """Place Entry and corresponding TP/SL Limit orders"""
    logger.info(f"🚀 PLACING {direction.upper()} ORDER FOR {coin} AT {price}...")
    
    leverage = CONFIG["leverage"]
    notional = margin * leverage
    size = notional / price
    
    size_str = f"{size:.4f}"
    price_str = f"{price:.1f}" 
    
    is_buy = True if direction == "long" else False
    
    try:
        # 1. Place Entry Limit Order
        logger.info(f"  -> Entry Limit: {'Buy' if is_buy else 'Sell'} {size_str} {coin} at {price_str}")
        entry_res = exchange.order(coin, is_buy, float(size_str), float(price_str), {"limit": {"tif": "Gtc"}})
        logger.info(f"  Entry Result: {entry_res}")
        
        # 2. Place Take Profit Limit Order (Assuming entry is filled immediately for demo)
        tp_pct = CONFIG["tp2_pct"] / 100
        sl_pct = CONFIG["sl_pct"] / 100
        
        if is_buy:
            tp_price = price * (1 + tp_pct)
            sl_price = price * (1 - sl_pct)
        else:
            tp_price = price * (1 - tp_pct)
            sl_price = price * (1 + sl_pct)
            
        tp_price_str = f"{tp_price:.1f}"
        sl_price_str = f"{sl_price:.1f}"
        
        logger.info(f"  -> TP Limit: {tp_price_str}")
        tp_res = exchange.order(coin, not is_buy, float(size_str), float(tp_price_str), {"limit": {"tif": "Gtc"}}, reduce_only=True)
        logger.info(f"  TP Result: {tp_res}")
        
    except Exception as e:
        logger.error(f"❌ Order placement error: {e}")

async def run_bot_async():
    """Async loop for running the bot inside a FastAPI background task"""
    logger.info("=" * 60)
    logger.info(" STARTING LIVE BOT ON HYPERLIQUID TESTNET")
    logger.info("=" * 60)
    
    if not PRIVATE_KEY:
        logger.error("❌ ERROR: No Private Key found. Set HL_PRIVATE_KEY environment variable.")
        return
        
    try:
        account: LocalAccount = eth_account.Account.from_key(PRIVATE_KEY)
        logger.info(f"✅ Wallet loaded: {account.address}")
    except Exception as e:
        logger.error(f"❌ Private Key parse error: {e}")
        return

    info = Info(API_URL, skip_ws=True)
    MAIN_ADDRESS = os.environ.get("HL_MAIN_ADDRESS", "").strip()
    exchange = Exchange(account, API_URL, account_address=MAIN_ADDRESS if MAIN_ADDRESS else None)
    
    coin = "BTC"
    
    logger.info(f"Waiting for data and monitoring {coin}...")
    
    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.second < 5:
                logger.info("Fetching real-time data...")
                
                # Run sync functions in executor to prevent blocking the async loop
                df15 = await asyncio.to_thread(fetch_hyperliquid_candles, info, coin, "15m", 250)
                df5 = await asyncio.to_thread(fetch_hyperliquid_candles, info, coin, "5m", 100)
                df1 = await asyncio.to_thread(fetch_hyperliquid_candles, info, coin, "1m", 50)
                
                if df15.empty or df5.empty or df1.empty:
                    logger.warning("Empty candle data, retrying later.")
                    await asyncio.sleep(10)
                    continue
                    
                i15 = len(df15) - 1
                i5 = len(df5) - 1
                i1 = len(df1) - 1
                
                setup = score_setup(i15, df15, i5, df5, i1, df1, hour_utc=now.hour)
                current_price = df1.iloc[-1]["close"]
                
                if setup["hard_block"]:
                    logger.info(f"  Block: {setup['block_reasons'][0]} (Price: {current_price})")
                else:
                    score = setup["score"]
                    logger.info(f"  ✅ {setup['direction'].upper()} Setup scored {score} points! (Price: {current_price})")
                    
                    if score >= CONFIG["min_score_half"]:
                        margin = CONFIG["margin_full"] if score >= CONFIG["min_score_full"] else CONFIG["margin_half"]
                        await asyncio.to_thread(place_orders, exchange, coin, setup["direction"], current_price, margin)
                        
                        logger.info("Order placed. Cooldown for 5 minutes...")
                        await asyncio.sleep(300)
                
                await asyncio.sleep(55)
            else:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(run_bot_async())
