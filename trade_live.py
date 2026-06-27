import time
import os
import pandas as pd
from datetime import datetime, timezone
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account
from eth_account.signers.local import LocalAccount

from indicators import add_indicators, score_setup
from run_backtest import CONFIG

# Đọc Private Key từ biến môi trường hoặc nhập trực tiếp (KHÔNG push lên github)
PRIVATE_KEY = os.environ.get("HL_PRIVATE_KEY", "")

# Sử dụng Testnet
API_URL = constants.TESTNET_API_URL

def fetch_hyperliquid_candles(info: Info, coin: str, interval: str, lookback_bars: int) -> pd.DataFrame:
    """Fetch nến từ Hyperliquid và chuyển đổi sang DataFrame chuẩn của hệ thống"""
    end_time = int(time.time() * 1000)
    
    # Tính start_time dựa trên số lượng nến cần thiết
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
        
    # Chuyển đổi format của Hyperliquid sang chuẩn DataFrame của backtest
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
    # Thêm các chỉ báo kỹ thuật
    if not df.empty and len(df) > 30: # Cần đủ nến để tính EMA200
        df = add_indicators(df)
        
    return df

def place_orders(exchange: Exchange, coin: str, direction: str, price: float, margin: float):
    """Đặt lệnh Entry và các lệnh TP/SL liên quan"""
    print(f"\n🚀 ĐẶT LỆNH {direction.upper()} {coin} tại giá {price}...")
    
    # Tính khối lượng (size)
    leverage = CONFIG["leverage"]
    notional = margin * leverage
    size = notional / price
    
    # Hyperliquid yêu cầu size phải được format chuẩn tùy theo coin (ví dụ BTC cần số thập phân nhỏ)
    # Đơn giản hóa: làm tròn size đến 4 chữ số thập phân cho BTC/ETH
    size_str = f"{size:.4f}"
    price_str = f"{price:.1f}" # Giả định step size giá của BTC/ETH
    
    is_buy = True if direction == "long" else False
    
    try:
        # 1. Đặt lệnh Entry (Limit order)
        print(f"  -> Entry Limit: Mua {size_str} {coin} ở giá {price_str}")
        entry_res = exchange.order(coin, is_buy, float(size_str), float(price_str), {"limit": {"tif": "Gtc"}})
        print(f"  Kết quả Entry: {entry_res}")
        
        # 2. Đặt lệnh Take Profit (Giả sử entry khớp ngay để đặt TP)
        # Lưu ý thực tế: Bạn nên theo dõi WebSocket xem lệnh entry khớp chưa rồi mới đặt TP/SL.
        # Ở script này minh họa đơn giản đặt luôn lệnh limit ngược chiều.
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
        
        print(f"  -> TP Limit: {tp_price_str}")
        tp_res = exchange.order(coin, not is_buy, float(size_str), float(tp_price_str), {"limit": {"tif": "Gtc"}}, reduce_only=True)
        print(f"  Kết quả TP: {tp_res}")
        
    except Exception as e:
        print(f"❌ Lỗi khi đặt lệnh: {e}")

def main():
    print("=" * 60)
    print(" BẮT ĐẦU BOT LIVE TRÊN HYPERLIQUID TESTNET")
    print("=" * 60)
    
    if not PRIVATE_KEY:
        print("❌ LỖI: Chưa có Private Key. Hãy cập nhật biến PRIVATE_KEY trong code hoặc set môi trường HL_PRIVATE_KEY.")
        return
        
    try:
        account: LocalAccount = eth_account.Account.from_key(PRIVATE_KEY)
        print(f"✅ Đã tải ví: {account.address}")
    except Exception as e:
        print(f"❌ Lỗi parse Private Key: {e}")
        return

    info = Info(API_URL, skip_ws=True)
    exchange = Exchange(account, API_URL)
    
    coin = "BTC"
    
    print(f"Đang chờ dữ liệu và theo dõi {coin}...")
    
    # Vòng lặp chính
    while True:
        try:
            now = datetime.now(timezone.utc)
            # Chờ đến khi hết phút (đóng nến 1m)
            if now.second < 5:
                print(f"[{now.strftime('%H:%M:%S')}] Fetching data...")
                
                # Fetch nến (cần ít nhất 250 nến 15m để tính EMA200)
                df15 = fetch_hyperliquid_candles(info, coin, "15m", 250)
                df5 = fetch_hyperliquid_candles(info, coin, "5m", 100)
                df1 = fetch_hyperliquid_candles(info, coin, "1m", 50)
                
                if df15.empty or df5.empty or df1.empty:
                    print("Dữ liệu nến bị trống, thử lại sau.")
                    time.sleep(10)
                    continue
                    
                i15 = len(df15) - 1
                i5 = len(df5) - 1
                i1 = len(df1) - 1
                
                # Chấm điểm setup hiện tại
                setup = score_setup(i15, df15, i5, df5, i1, df1, hour_utc=now.hour)
                
                current_price = df1.iloc[-1]["close"]
                
                if setup["hard_block"]:
                    print(f"  Block: {setup['block_reasons'][0]} (Price: {current_price})")
                else:
                    score = setup["score"]
                    print(f"  ✅ Setup {setup['direction']} đạt {score} điểm! (Price: {current_price})")
                    
                    if score >= CONFIG["min_score_half"]:
                        margin = CONFIG["margin_full"] if score >= CONFIG["min_score_full"] else CONFIG["margin_half"]
                        place_orders(exchange, coin, setup["direction"], current_price, margin)
                        
                        print("Đã đặt lệnh xong, nghỉ 5 phút để tránh nhồi lệnh liên tục...")
                        time.sleep(300)
                
                # Sleep để tránh spam API liên tục trong cùng 1 nến
                time.sleep(55)
            else:
                time.sleep(1)
                
        except Exception as e:
            print(f"Lỗi vòng lặp chính: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
