# Backtest Bot Scalping BTC/ETH — Kiểm chứng trước khi chạy live

Bộ công cụ này KIỂM TRA chiến lược scalping (scoring engine Module 1-5)
bằng dữ liệu lịch sử thật, trước khi bạn cân nhắc chạy live với tiền thật.

## ⚠️ Đọc trước khi chạy

- Đây là công cụ NGHIÊN CỨU, không phải bot giao dịch tự động.
- Kết quả backtest **lạc quan hơn** thực tế live vì không mô phỏng được
  100% slippage thật, độ trễ mạng, hay việc lệnh limit không khớp.
- Nếu kết quả cho thấy Profit Factor thấp (< 1.3) hoặc Win Rate sát 50%,
  điều đó nghĩa là chiến lược không có edge đáng kể sau phí — **đừng**
  chạy live với leverage cao trong trường hợp đó, vì đó là cách nhanh
  nhất để mất vốn.
- Dữ liệu lấy từ Binance Futures (lịch sử đầy đủ hơn), không phải
  Hyperliquid trực tiếp — nhưng biến động giá BTC/ETH giữa các sàn
  perp lớn thường rất tương quan, nên kết quả vẫn có giá trị tham khảo.

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy

```bash
# Bước 1: Tải dữ liệu lịch sử 3 tháng (BTC + ETH, khung 1m/5m/15m)
python fetch_data.py

# Bước 2: Chạy backtest, áp dụng đúng Scoring Engine Module 1-5
python run_backtest.py

# Bước 3 (tùy chọn): Vẽ biểu đồ equity curve / drawdown
python plot_results.py
```

## Output

- `data/` — file CSV dữ liệu OHLCV lịch sử đã tải
- `results/BTCUSDT_trades.csv`, `results/ETHUSDT_trades.csv` — log từng lệnh
- `results/*_chart.png` — biểu đồ equity curve, drawdown, PnL từng lệnh
- Báo cáo tổng hợp in ra terminal: win rate, profit factor, drawdown,
  số lệnh/ngày, % phí so với gross profit

## Cách đọc kết quả

| Chỉ số | Ý nghĩa | Ngưỡng cảnh báo |
|---|---|---|
| Win Rate | % lệnh thắng | Nếu < 50%, chiến lược thua nhiều hơn thắng |
| Profit Factor | Tổng lời / Tổng lỗ | < 1.3 nghĩa là edge rất mỏng, dễ âm sau slippage thật |
| Phí/Gross PnL | % lợi nhuận gộp bị phí ăn | > 30% nghĩa là phí đang ăn phần lớn lợi nhuận |
| Max Drawdown | Sụt giảm vốn tối đa | > 20% là rủi ro rất cao với vốn $100 |
| Lệnh/ngày | Số lệnh đủ tiêu chuẩn score | Nếu quá thấp (~0-2), chiến lược quá khắt khe để tạo thu nhập đáng kể |

## Hai profile cấu hình — so sánh target $1 vs $5

File `run_backtest.py` có 2 profile sẵn, đổi bằng dòng `ACTIVE_PROFILE` ở đầu file:

- **`scalp_1usd`** — bản gốc: TP nhỏ (+0.08%/+0.15%), entry bằng market/IOC
  (luôn trả phí taker 0.045%), nhiều lệnh/ngày hơn.
- **`swing_5usd`** (mặc định) — bản mới: TP lớn hơn (+0.30%/+0.55%) để
  target ~$5/lệnh thắng ở full size, entry bằng **limit order** (chờ
  khớp tối đa 1 nến, không chase giá — nếu không khớp thì bỏ qua setup
  đó), nên trả phí maker 0.015% thay vì taker. Ít lệnh/ngày hơn nhưng
  tỷ trọng phí trên lợi nhuận thấp hơn nhiều.

Chạy backtest với từng profile (đổi `ACTIVE_PROFILE`, lưu kết quả ra
tên file khác để không bị ghi đè), rồi so sánh bảng Profit Factor /
Win Rate / Phí-trên-lợi-nhuận giữa 2 bản — đó là cách trả lời thật
cho câu hỏi "tăng target lên $5 có khả quan hơn không", chứ không
phải suy luận lý thuyết.

Lưu ý: với entry limit order, một số setup đạt điểm sẽ **không khớp**
(giá chạy đi trước khi lệnh limit kịp khớp) — terminal sẽ in ra tỷ lệ
miss này. Đây là đánh đổi thật của việc ưu tiên phí thấp.


Mọi tham số (margin, leverage, TP/SL %, ngưỡng score, phí, slippage)
đều nằm trong `CONFIG` ở đầu file `run_backtest.py`. Sau khi có kết quả
backtest đầu tiên, bạn có thể thử thay đổi từng tham số một (không đổi
nhiều cùng lúc) để xem yếu tố nào ảnh hưởng mạnh nhất đến kết quả —
đây gọi là "sensitivity analysis", giúp hiểu chiến lược thật sự nhạy
với cái gì.

## Bước tiếp theo nếu kết quả backtest tốt

Ngay cả khi backtest cho kết quả khả quan, đừng chuyển thẳng sang live
với $100 và leverage cao. Thứ tự nên là:

1. Chạy thêm backtest trên dữ liệu giai đoạn KHÁC (ví dụ 3 tháng trước
   nữa) để tránh overfitting vào 1 giai đoạn thị trường cụ thể.
2. Chạy thử trên Hyperliquid **testnet** với dữ liệu/lệnh giả trước.
3. Nếu chạy live, bắt đầu với vốn nhỏ hơn $100 và leverage thấp hơn
   20x để có biên độ chịu đựng sai số giữa backtest và thực tế.
4. Theo dõi sát 2-4 tuần đầu, so sánh win rate live với win rate
   backtest — nếu lệch nhiều, dừng lại và xem lại giả định.
