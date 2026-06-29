"""
indicators.py — Scoring Engine v2
==================================
Calculate technical indicators and scoring engine applied on REAL data.

SCORING ENGINE V2 CHANGES (so với v1):
──────────────────────────────────────────────────────────────────────────────
1.  VWAP daily reset      — fix bug cumsum không reset theo ngày UTC
2.  BB %B indicator        — vị trí giá trong dải BB (0=lower, 1=upper)
3.  BB width percentile    — phát hiện BB squeeze
4.  BB Width Filter        — soft penalty khi BB quá rộng (trending)
5.  BB %B depth            — bonus/penalty dựa trên vị trí %B
6.  RSI Divergence (1m)    — bullish/bearish divergence đơn giản
7.  StochRSI Cross (5m)    — %K cắt qua %D trong vùng oversold/overbought
8.  MACD Histogram (5m)    — MACD hist đang quay đầu
9.  Candlestick 5m bonus   — hammer/engulfing trên 5m có giá trị cao hơn 1m
10. Time-of-Day filter     — soft penalty Asia session, bonus London/NY
11. BB Squeeze detector    — flag cho future breakout strategy
12. Score breakdown        — chi tiết điểm từng module
13. Confidence level       — A+/A/B/C phân loại chất lượng setup

THIẾT KẾ: Giữ tần suất cao ("trade nhiều, lãi nhỏ") bằng cách:
  - Dùng soft penalty thay vì hard block
  - Giữ nguyên min_score_half=45, min_score_full=55
  - Thêm nhiều module "bonus" để tăng cơ hội đạt threshold
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

    # ── NEW: BB %B — vị trí giá trong dải BB ──
    # %B = 0 → đúng lower band, %B = 1 → đúng upper band
    # %B < 0 → phá qua lower, %B > 1 → phá qua upper
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)

    # ── NEW: BB width percentile — phát hiện squeeze ──
    # Rank 0.0-1.0: 0.1 = BB đang ở vùng hẹp nhất 10% trong 20 nến gần nhất
    df["bb_width_rank"] = df["bb_width_pct"].rolling(20).rank(pct=True)

    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr_pct"] = df["atr"] / df["close"] * 100

    # ── FIX: VWAP reset mỗi ngày UTC ──
    # Trước đây: cumsum từ đầu dataset → VWAP tích lũy hàng tuần, mất ý nghĩa intraday
    # Bây giờ: reset theo ngày UTC để VWAP phản ánh đúng session hiện tại
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        trade_date = df["timestamp"].dt.date
        cum_vol_price = (df["close"] * df["volume"]).groupby(trade_date).cumsum()
        cum_vol = df["volume"].groupby(trade_date).cumsum()
        df["vwap"] = cum_vol_price / cum_vol.replace(0, np.nan)
    else:
        # Fallback nếu không có timestamp (edge case)
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


# ── NEW: RSI Divergence Detection ──────────────────────────────────────────
def _check_rsi_divergence(df: pd.DataFrame, idx: int, direction: str, lookback: int = 10) -> bool:
    """
    Kiểm tra RSI divergence đơn giản trên 1 timeframe.

    Bullish divergence: giá tạo lower low NHƯNG RSI tạo higher low
      → selling pressure giảm dù giá vẫn giảm → sắp bounce

    Bearish divergence: giá tạo higher high NHƯNG RSI tạo lower high
      → buying pressure giảm dù giá vẫn tăng → sắp drop

    Chỉ cần so sánh current vs lowest/highest trong lookback window.
    Không cần phức tạp — đây là approximation cho scalp.
    """
    if idx < lookback * 2:
        return False

    current_price = df.iloc[idx]["close"]
    current_rsi = df.iloc[idx]["rsi"]

    if pd.isna(current_rsi):
        return False

    # Window trước đó (lookback*2 → lookback bars trước hiện tại)
    prev_window = df.iloc[idx - lookback * 2:idx - lookback + 1]
    if len(prev_window) < 3:
        return False

    prev_prices = prev_window["close"]
    prev_rsis = prev_window["rsi"].dropna()
    if len(prev_rsis) < 3:
        return False

    if direction == "long":
        # Bullish divergence: giá lower low, RSI higher low
        prev_price_low = prev_prices.min()
        prev_rsi_at_low = prev_rsis.min()
        if current_price <= prev_price_low and current_rsi > prev_rsi_at_low:
            return True

    elif direction == "short":
        # Bearish divergence: giá higher high, RSI lower high
        prev_price_high = prev_prices.max()
        prev_rsi_at_high = prev_rsis.max()
        if current_price >= prev_price_high and current_rsi < prev_rsi_at_high:
            return True

    return False


def score_setup(i15: int, df15: pd.DataFrame,
                 i5: int, df5: pd.DataFrame,
                 i1: int, df1: pd.DataFrame,
                 consecutive_losses: int = 0,
                 hour_utc: int = 12,
                 cfg: dict | None = None,
                 obi: float = 0.0) -> dict:
    """
    Scoring Engine v2 — Apply tất cả module scoring tại candle indices
    i15, i5, i1 trên 3 timeframes.

    THIẾT KẾ GIỮ TẦN SUẤT CAO:
      - Soft penalty thay vì hard block (trừ consecutive losses)
      - Nhiều module bonus để tăng cơ hội đạt threshold
      - min_score_half/full giữ nguyên (45/55)

    Returns dict: {
        'direction': 'long' | 'short' | None,
        'score': int,
        'hard_block': bool,
        'block_reasons': [...],
        'regime': str,
        'momentum': float,
        'atr_1m': float,
        'atr_5m': float,
        'bb_width_pct': float,      # BB width hiện tại
        'bb_squeeze': bool,          # True nếu BB đang squeeze
        'confidence': str,           # A+ / A / B / C
        'score_details': dict,       # Breakdown điểm từng module
    }
    """
    result = {
        "direction": None, "score": 0, "hard_block": False,
        "block_reasons": [], "regime": "unknown", "momentum": 0.0,
        "atr_1m": 0.0, "atr_5m": 0.0,
        "bb_width_pct": 0.0, "bb_squeeze": False,
        "confidence": "C",
        "score_details": {},
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
    details = {}  # Track điểm từng module
    direction = None

    # --------------------------------------------------
    # MODULE 1: BB TOUCH — Base signal (giữ nguyên)
    # --------------------------------------------------
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

    # ── Lưu BB width cho logging ──
    bb_width = row1["bb_width_pct"]
    result["bb_width_pct"] = float(bb_width) if not pd.isna(bb_width) else 0.0

    # ── BB Squeeze detection ──
    bb_width_rank = row1.get("bb_width_rank")
    is_squeeze = (bb_width_rank is not None
                  and not pd.isna(bb_width_rank)
                  and bb_width_rank < 0.15)
    result["bb_squeeze"] = is_squeeze

    # VWAP stretch calculation
    vwap_stretch_pct = 0.0
    if not pd.isna(vwap1) and vwap1 > 0:
        vwap_stretch_pct = abs(price1 - vwap1) / vwap1 * 100

    # Check BB touch — base trigger
    if price1 <= bb_lower1:
        direction = "long"
        score += 40
        details["bb_touch"] = 40

        # Penalize catching a falling knife
        pattern_1m = detect_candle_pattern(row1, row1_prev)
        if pattern_1m == "strong_bearish_close":
            score -= 15
            details["strong_close_penalty"] = -15
        else:
            details["strong_close_penalty"] = 0

        # Reward volume climax
        if not pd.isna(vol1) and not pd.isna(vol_avg20_1) and vol1 > vol_avg20_1 * 1.5:
            score += 10
            details["volume_climax"] = 10
        else:
            details["volume_climax"] = 0

        # Soft Trend Alignment on 5m
        if not pd.isna(ema50_5):
            if price5 > ema50_5:
                score += 10
                details["ema50_trend"] = 10
            else:
                score -= 10
                details["ema50_trend"] = -10
        else:
            details["ema50_trend"] = 0

        # VWAP Stretch (Rubber band effect)
        if vwap_stretch_pct > 0.5:
            score += 20
            details["vwap_stretch"] = 20
        elif vwap_stretch_pct > 0.3:
            score += 10
            details["vwap_stretch"] = 10
        else:
            details["vwap_stretch"] = 0

    elif price1 >= bb_upper1:
        direction = "short"
        score += 40
        details["bb_touch"] = 40

        # Penalize blindly shorting a rocket
        pattern_1m = detect_candle_pattern(row1, row1_prev)
        if pattern_1m == "strong_bullish_close":
            score -= 15
            details["strong_close_penalty"] = -15
        else:
            details["strong_close_penalty"] = 0

        # Reward volume climax
        if not pd.isna(vol1) and not pd.isna(vol_avg20_1) and vol1 > vol_avg20_1 * 1.5:
            score += 10
            details["volume_climax"] = 10
        else:
            details["volume_climax"] = 0

        # Soft Trend Alignment on 5m
        if not pd.isna(ema50_5):
            if price5 < ema50_5:
                score += 10
                details["ema50_trend"] = 10
            else:
                score -= 10
                details["ema50_trend"] = -10
        else:
            details["ema50_trend"] = 0

        # VWAP Stretch (Rubber band effect)
        if vwap_stretch_pct > 0.5:
            score += 20
            details["vwap_stretch"] = 20
        elif vwap_stretch_pct > 0.3:
            score += 10
            details["vwap_stretch"] = 10
        else:
            details["vwap_stretch"] = 0

    else:
        # Base Signal 2: RSI Extreme (Bắt đáy/đỉnh khi RSI cực hạn, dù chưa chạm BB)
        if rsi1 < 25:
            direction = "long"
            score += 35
            details["rsi_extreme_touch"] = 35
        elif rsi1 > 75:
            direction = "short"
            score += 35
            details["rsi_extreme_touch"] = 35
        # Base Signal 3: EMA Reversion (Kéo ngược về EMA khi giá tăng/giảm sốc)
        else:
            ema9_1 = row1.get("ema9")
            if not pd.isna(ema9_1) and ema9_1 > 0:
                dist_pct = (price1 - ema9_1) / ema9_1 * 100
                if dist_pct < -0.25:  # Giá tụt > 0.25% so với EMA9
                    direction = "long"
                    score += 30
                    details["ema_reversion"] = 30
                elif dist_pct > 0.25: # Giá tăng quá mạnh
                    direction = "short"
                    score += 30
                    details["ema_reversion"] = 30
                else:
                    return result
            else:
                return result

    # --------------------------------------------------
    # MODULE 2: RSI CONFIRMATION (1m and 5m) — giữ nguyên
    # --------------------------------------------------
    if direction == "long":
        if rsi1 < 30:
            score += 20
            details["rsi_1m"] = 20
        elif rsi1 < 40:
            score += 10
            details["rsi_1m"] = 10
        else:
            details["rsi_1m"] = 0

        if rsi5 < 35:
            score += 20
            details["rsi_5m"] = 20
        elif rsi5 < 45:
            score += 10
            details["rsi_5m"] = 10
        else:
            details["rsi_5m"] = 0

        # Candlestick pattern bonus (1m)
        pattern_1m = detect_candle_pattern(row1, row1_prev)
        if pattern_1m in ["hammer", "bullish_engulfing"]:
            score += 10
            details["candle_1m"] = 10
        else:
            details["candle_1m"] = 0

    else:  # short
        if rsi1 > 70:
            score += 20
            details["rsi_1m"] = 20
        elif rsi1 > 60:
            score += 10
            details["rsi_1m"] = 10
        else:
            details["rsi_1m"] = 0

        if rsi5 > 65:
            score += 20
            details["rsi_5m"] = 20
        elif rsi5 > 55:
            score += 10
            details["rsi_5m"] = 10
        else:
            details["rsi_5m"] = 0

        # Candlestick pattern bonus (1m)
        pattern_1m = detect_candle_pattern(row1, row1_prev)
        if pattern_1m in ["shooting_star", "bearish_engulfing"]:
            score += 10
            details["candle_1m"] = 10
        else:
            details["candle_1m"] = 0

    # --------------------------------------------------
    # MODULE 3 (NEW): BB WIDTH FILTER — soft penalty
    # --------------------------------------------------
    # BB rộng = trending → mean-reversion rủi ro cao
    # BB hẹp = sideways → mean-reversion thuận lợi
    # KHÔNG hard block để giữ tần suất
    bb_width_max = cfg.get("bb_width_max_pct", 1.0)
    bb_width_warn = cfg.get("bb_width_warn_pct", 0.6)

    if cfg.get("use_bb_width_filter", True) and not pd.isna(bb_width):
        if bb_width > bb_width_max:
            score -= 15
            details["bb_width"] = -15
        elif bb_width > bb_width_warn:
            score -= 5
            details["bb_width"] = -5
        elif bb_width < 0.2:
            score += 5  # BB hẹp → sideways, mean-reversion thuận lợi
            details["bb_width"] = 5
        else:
            details["bb_width"] = 0
    else:
        details["bb_width"] = 0

    # --------------------------------------------------
    # MODULE 4 (NEW): BB %B DEPTH — bonus/penalty
    # --------------------------------------------------
    # %B < 0 (long) hoặc > 1 (short) = phá qua BB
    # Phá nhẹ → tốt (bounce), phá sâu → xấu (momentum tiếp tục)
    bb_pct_b = row1.get("bb_pct_b")
    bb_pct_b_deep = cfg.get("bb_pct_b_deep_threshold", 0.15)

    if cfg.get("use_bb_pct_b", True) and bb_pct_b is not None and not pd.isna(bb_pct_b):
        if direction == "long":
            if bb_pct_b < -bb_pct_b_deep:
                # Phá qua BB quá sâu → có thể tiếp tục giảm
                score -= 10
                details["bb_pct_b"] = -10
            elif bb_pct_b <= 0:
                # Vừa chạm/phá nhẹ BB lower → bounce zone tốt
                score += 10
                details["bb_pct_b"] = 10
            else:
                details["bb_pct_b"] = 0
        else:  # short
            if bb_pct_b > 1 + bb_pct_b_deep:
                # Phá qua BB upper quá sâu
                score -= 10
                details["bb_pct_b"] = -10
            elif bb_pct_b >= 1.0:
                # Vừa chạm/phá nhẹ BB upper → bounce zone tốt
                score += 10
                details["bb_pct_b"] = 10
            else:
                details["bb_pct_b"] = 0
    else:
        details["bb_pct_b"] = 0

    # --------------------------------------------------
    # MODULE 5 (NEW): RSI DIVERGENCE (1m)
    # --------------------------------------------------
    # Signal mạnh: giá tạo new low/high nhưng RSI không
    # → momentum đang yếu đi, reversal sắp xảy ra
    if cfg.get("use_rsi_divergence", True):
        if _check_rsi_divergence(df1, i1, direction):
            score += 15
            details["rsi_divergence"] = 15
        else:
            details["rsi_divergence"] = 0
    else:
        details["rsi_divergence"] = 0

    # --------------------------------------------------
    # MODULE 6 (NEW): STOCHRSI CROSS (5m)
    # --------------------------------------------------
    # %K cắt qua %D trong vùng oversold/overbought = entry signal cụ thể
    # Đã có data (stoch_k, stoch_d trong df5), chỉ cần check
    stoch_k = row5.get("stoch_k")
    stoch_d = row5.get("stoch_d")
    stoch_k_prev = row5_prev.get("stoch_k")
    stoch_d_prev = row5_prev.get("stoch_d")

    if cfg.get("use_stoch_cross", True):
        stoch_values = [stoch_k, stoch_d, stoch_k_prev, stoch_d_prev]
        if not any(v is None or (isinstance(v, float) and pd.isna(v)) for v in stoch_values):
            if direction == "long":
                # %K cắt lên qua %D trong vùng oversold (<20)
                if stoch_k_prev < stoch_d_prev and stoch_k > stoch_d and stoch_k < 20:
                    score += 15
                    details["stoch_cross"] = 15
                else:
                    details["stoch_cross"] = 0
            else:  # short
                # %K cắt xuống qua %D trong vùng overbought (>80)
                if stoch_k_prev > stoch_d_prev and stoch_k < stoch_d and stoch_k > 80:
                    score += 15
                    details["stoch_cross"] = 15
                else:
                    details["stoch_cross"] = 0
        else:
            details["stoch_cross"] = 0
    else:
        details["stoch_cross"] = 0

    # --------------------------------------------------
    # MODULE 7 (NEW): MACD HISTOGRAM MOMENTUM (5m)
    # --------------------------------------------------
    # MACD hist đang quay đầu = momentum đang chậm lại
    # Long: hist tăng từ vùng âm (sellers đang yếu)
    # Short: hist giảm từ vùng dương (buyers đang yếu)
    macd_hist = row5.get("macd_hist")
    macd_hist_prev = row5_prev.get("macd_hist")

    if cfg.get("use_macd_momentum", True):
        if not pd.isna(macd_hist) and not pd.isna(macd_hist_prev):
            if direction == "long" and macd_hist > macd_hist_prev and macd_hist < 0:
                score += 10
                details["macd_hist"] = 10
            elif direction == "short" and macd_hist < macd_hist_prev and macd_hist > 0:
                score += 10
                details["macd_hist"] = 10
            else:
                details["macd_hist"] = 0
        else:
            details["macd_hist"] = 0
    else:
        details["macd_hist"] = 0

    # --------------------------------------------------
    # MODULE 8 (NEW): CANDLESTICK PATTERN 5m BONUS
    # --------------------------------------------------
    # Hammer/engulfing trên 5m có giá trị cao hơn nhiều so với 1m
    # vì 5m candle tổng hợp nhiều price action hơn
    if cfg.get("use_candle_5m", True):
        pattern_5m = detect_candle_pattern(row5, row5_prev)
        if direction == "long" and pattern_5m in ["hammer", "bullish_engulfing"]:
            score += 10
            details["candle_5m"] = 10
        elif direction == "short" and pattern_5m in ["shooting_star", "bearish_engulfing"]:
            score += 10
            details["candle_5m"] = 10
        else:
            details["candle_5m"] = 0
    else:
        details["candle_5m"] = 0

    # --------------------------------------------------
    # MODULE 9 (NEW): TIME-OF-DAY FILTER — soft penalty
    # --------------------------------------------------
    # London (07-12 UTC) + NY (13-17 UTC) = high win rate sessions
    # Asia (01-06 UTC) = low volume, false signals nhiều hơn
    # SOFT PENALTY (-5) thay vì hard block để giữ tần suất
    if cfg.get("use_time_filter", True):
        if hour_utc in range(7, 13) or hour_utc in range(13, 18):
            score += 10  # High-probability session bonus
            details["time_filter"] = 10
        elif hour_utc in range(1, 7):
            score -= 5  # Low-probability session penalty (nhẹ)
            details["time_filter"] = -5
        else:
            # Midnight UTC (0) và 18-23: neutral
            details["time_filter"] = 0
    else:
        details["time_filter"] = 0

    # --------------------------------------------------
    # MODULE 10 (NEW): DUAL-REGIME / MOMENTUM FLIP
    # --------------------------------------------------
    # Nếu đang đánh Mean-Reversion nhưng OBI báo dòng tiền cực mạnh thuận xu hướng đột phá
    # -> Lật mặt (Flip) đánh Đu Trend!
    if direction is not None and obi != 0.0:
        flip_obi = 0.35
        block_obi = -0.25
        vol_climax = details.get("volume_climax", 0) > 0

        if direction == "long":
            # Đang định bắt đáy (Long)
            if obi <= -flip_obi and vol_climax:
                # OBI cực âm + Volume to -> Lực xả quá mạnh, lật sang Short ăn hôi
                direction = "short"
                score += 20
                details["momentum_flip"] = 20
            elif obi < block_obi:
                # OBI âm vừa phải, chặn bắt đáy
                result["block_reasons"].append(f"OBI={obi:.2f} (sellers extremely dominant)")
                result["hard_block"] = True
            elif obi > 0.3:
                # OBI ủng hộ bắt đáy
                score += 15
                details["obi_bonus"] = 15

        elif direction == "short":
            # Đang định bắt đỉnh (Short)
            if obi >= flip_obi and vol_climax:
                # OBI cực dương + Volume to -> Lực fomo quá mạnh, lật sang Long đu đỉnh
                direction = "long"
                score += 20
                details["momentum_flip"] = 20
            elif obi > -block_obi:  # Tương đương obi > 0.25
                # OBI dương vừa phải, chặn bắt đỉnh
                result["block_reasons"].append(f"OBI={obi:.2f} (buyers extremely dominant)")
                result["hard_block"] = True
            elif obi < -0.3:
                # OBI ủng hộ bắt đỉnh
                score += 15
                details["obi_bonus"] = 15

    # --------------------------------------------------
    # COOLDOWN CHECK (giữ nguyên — hard block duy nhất)
    # --------------------------------------------------
    if consecutive_losses >= 3:
        result["block_reasons"].append("3 consecutive losses - cooldown")
        result["hard_block"] = True
        result["score_details"] = details
        return result

    # --------------------------------------------------
    # EXPORT RESULTS
    # --------------------------------------------------
    # ATR values for dynamic TP/SL
    atr_5m = row5.get("atr")
    result["atr_5m"] = float(atr_5m) if not pd.isna(atr_5m) else 0.0

    atr_1m = row1.get("atr")
    result["atr_1m"] = float(atr_1m) if not pd.isna(atr_1m) else 0.0

    result["direction"] = direction
    result["score"] = score
    result["score_details"] = details

    # ── Confidence level — phân loại chất lượng setup ──
    if score >= 100:
        result["confidence"] = "A+"  # Rất nhiều confluence — full margin
    elif score >= 80:
        result["confidence"] = "A"   # Nhiều confluence — near full margin
    elif score >= 60:
        result["confidence"] = "B"   # Đủ confluence — standard margin
    else:
        result["confidence"] = "C"   # Ít confluence — half margin

    return result