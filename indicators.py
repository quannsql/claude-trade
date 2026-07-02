"""
indicators.py — Scoring Engine v3 (DUAL REGIME)
================================================
Calculate technical indicators and scoring engine applied on REAL data.

SCORING ENGINE V3 CHANGES (so với v2):
──────────────────────────────────────────────────────────────────────────────
1.  DUAL REGIME dispatcher — trend_up/trend_down → trade THUẬN trend (pullback
    về EMA21/BB mid/VWAP); ranging → fade 2 chiều như v2.
    Fix gốc rễ: v2 chỉ biết fade, sinh SHORT liên tục trong uptrend.
2.  vwap_stretch_base bị khóa — cần VWAP >= vwap_min_hour_utc giờ dữ liệu
    trong ngày UTC + nến đã quay đầu về VWAP. (Nguồn tín hiệu sai lớn nhất.)
3.  Veto mới — cả ema50(5m) VÀ trend(15m) ngược hướng fade → hard block,
    không candle pattern nào lách được.
4.  Bỏ momentum flip — bug giữ nguyên điểm của hướng cũ sau khi flip.
    Thay bằng hard block khi OBI chống mạnh.
5.  RSI divergence fix — so RSI tại đúng bar đáy/đỉnh giá (idxmin/idxmax),
    không phải min/max RSI của cả window (điều kiện cũ gần như luôn đúng).
6.  OBI nhận giá trị ĐÃ LÀM MƯỢT từ engine (rolling 60-90s), không snapshot.
7.  Asia session — bỏ penalty -5 tượng trưng; engine nâng min_score
    +asia_score_bump (mặc định +10) trong 01-06 UTC.
8.  Penalty trend trong fade path tăng -10 → -15.

THIẾT KẾ: Giữ tần suất cao ("trade nhiều, lãi nhỏ") bằng cách chuyển hướng
trade theo regime thay vì chỉ thêm filter: các khung giờ trending trước đây
bot ngồi block giờ thành cơ hội pullback thuận trend.
"""

import pandas as pd
import numpy as np
import ta
from filters import detect_regime, detect_trend_regime, momentum_strength, price_position_vs_ema


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

    # FIX v3: so RSI tại ĐÚNG bar của đáy/đỉnh giá, không phải min/max RSI
    # của cả window. Bản cũ (current_rsi > prev_rsis.min()) gần như luôn đúng
    # → cộng điểm "miễn phí" cho các lệnh fade sai.
    if direction == "long":
        # Bullish divergence: giá lower low, RSI tại đáy giá cao hơn RSI hiện tại
        low_idx = prev_prices.idxmin()
        rsi_at_price_low = df.loc[low_idx, "rsi"]
        if pd.isna(rsi_at_price_low):
            return False
        if current_price <= prev_prices.min() and current_rsi > rsi_at_price_low:
            return True

    elif direction == "short":
        # Bearish divergence: giá higher high, RSI tại đỉnh giá thấp hơn
        high_idx = prev_prices.idxmax()
        rsi_at_price_high = df.loc[high_idx, "rsi"]
        if pd.isna(rsi_at_price_high):
            return False
        if current_price >= prev_prices.max() and current_rsi < rsi_at_price_high:
            return True

    return False


def detect_regime_for_entry(df15: pd.DataFrame, i15: int, direction: str) -> tuple[bool, str]:
    """
    Kiểm tra xem thị trường đang ở regime phù hợp với mean-reversion không.

    Mean-reversion chỉ hoạt động khi:
    - Thị trường sideways (không có trend rõ ràng)
    - Hoặc đang pullback tạm thời trong trend ngược chiều

    Trả về (is_blocked, reason):
    - is_blocked=True → không nên vào lệnh theo hướng direction
    - reason → mô tả lý do block
    """
    if i15 < 10:
        return False, ""

    row = df15.iloc[i15]
    
    # --- Kiểm tra 1: EMA50 slope trên 15m ---
    if i15 >= 5:
        ema50_now = row.get("ema50")
        ema50_5ago = df15.iloc[i15 - 5].get("ema50")
        if (ema50_now is not None and ema50_5ago is not None
                and not pd.isna(ema50_now) and not pd.isna(ema50_5ago)
                and ema50_5ago > 0):
            slope_pct = (ema50_now - ema50_5ago) / ema50_5ago * 100
            if direction == "long" and slope_pct < -0.15:
                return True, f"EMA50 downslope {slope_pct:.3f}% (downtrend, skip long)"
            if direction == "short" and slope_pct > 0.15:
                return True, f"EMA50 upslope {slope_pct:.3f}% (uptrend, skip short)"

    # --- Kiểm tra 2: Chuỗi nến cùng màu liên tiếp trên 15m ---
    consecutive_same = 0
    for offset in range(1, 5):
        if i15 - offset < 0:
            break
        c = df15.iloc[i15 - offset]
        if direction == "long" and c["close"] < c["open"]:
            consecutive_same += 1
        elif direction == "short" and c["close"] > c["open"]:
            consecutive_same += 1
        else:
            break
    if consecutive_same >= 4:
        return True, f"{consecutive_same} consecutive {'bearish' if direction == 'long' else 'bullish'} 15m candles"

    # --- Kiểm tra 3: ATR expansion đột biến ---
    atr_now = row.get("atr")
    if i15 >= 10 and atr_now is not None and not pd.isna(atr_now):
        atr_avg = df15.iloc[i15-10:i15]["atr"].mean()
        if atr_avg > 0 and atr_now > atr_avg * 2.0:
            return True, f"ATR expansion {atr_now:.4f} > 2x avg {atr_avg:.4f} (volatility spike)"

    return False, ""


def _directional_volume(df: pd.DataFrame, idx: int, lookback: int = 5) -> tuple[float, float]:
    """
    Tính up-volume và down-volume trong lookback nến gần nhất.
    Up-volume: nến close > open (buying pressure)
    Down-volume: nến close < open (selling pressure)
    Trả về (up_vol_ratio, down_vol_ratio) — tổng = 1.0
    """
    if idx < lookback:
        return 0.5, 0.5
    window = df.iloc[idx - lookback:idx + 1]
    total_vol = window["volume"].sum()
    if total_vol == 0:
        return 0.5, 0.5
    up_vol = window[window["close"] > window["open"]]["volume"].sum()
    down_vol = window[window["close"] < window["open"]]["volume"].sum()
    return up_vol / total_vol, down_vol / total_vol


def _finalize_result(result: dict, direction: str | None, score: int, details: dict) -> dict:
    result["direction"] = direction
    result["score"] = score
    result["score_details"] = details

    if score >= 100:
        result["confidence"] = "A+"
    elif score >= 80:
        result["confidence"] = "A"
    elif score >= 60:
        result["confidence"] = "B"
    else:
        result["confidence"] = "C"
    return result


def _atr_spike_block(df15: pd.DataFrame, i15: int) -> bool:
    """True nếu ATR 15m hiện tại > 2x trung bình 10 nến — volatility spike."""
    if i15 < 10:
        return False
    atr_now = df15.iloc[i15].get("atr")
    if atr_now is None or pd.isna(atr_now):
        return False
    atr_avg = df15.iloc[i15 - 10:i15]["atr"].mean()
    return bool(atr_avg > 0 and atr_now > atr_avg * 2.0)


def score_setup(i15: int, df15: pd.DataFrame,
                 i5: int, df5: pd.DataFrame,
                 i1: int, df1: pd.DataFrame,
                 consecutive_losses: int = 0,
                 hour_utc: int = 12,
                 cfg: dict | None = None,
                 obi: float = 0.0,
                 ob_analysis: dict | None = None,
                 funding_rate: float = 0.0) -> dict:
    """
    Scoring Engine v3 — DUAL REGIME dispatcher.

    Quy trình:
      1. Phát hiện regime (trend_up / trend_down / ranging) từ 15m slope + 5m EMA stack
      2. TREND regime  → chỉ trade THUẬN trend: pullback về value zone (EMA21/BB mid/VWAP)
      3. RANGING regime → fade BB/RSI/VWAP như engine v2 (đã siết veto + bỏ momentum flip)

    LƯU Ý: `obi` truyền vào phải là OBI ĐÃ LÀM MƯỢT (rolling mean 60-90s),
    không phải snapshot đơn lẻ. obi=0.0 → bỏ qua toàn bộ logic OBI (backtest).

    Returns dict: direction, score, hard_block, block_reasons, regime,
    atr_1m/5m, bb_width_pct, bb_squeeze, confidence, entry_mode, score_details.
    """
    result = {
        "direction": None, "score": 0, "hard_block": False,
        "block_reasons": [], "regime": "ranging", "momentum": 0.0,
        "atr_1m": 0.0, "atr_5m": 0.0,
        "bb_width_pct": 0.0, "bb_squeeze": False,
        "confidence": "C", "entry_mode": None,
        "score_details": {},
    }
    cfg = cfg or {}

    if i15 < 200 or i5 < 30 or i1 < 25:
        return result

    row15 = df15.iloc[i15]
    row5 = df5.iloc[i5]
    row1 = df1.iloc[i1]

    if pd.isna(row15["ema200"]) or pd.isna(row5["macd"]) or pd.isna(row1["bb_width_pct"]):
        return result

    # ── Export common context ──
    bb_width = row1["bb_width_pct"]
    result["bb_width_pct"] = float(bb_width) if not pd.isna(bb_width) else 0.0

    bb_width_rank = row1.get("bb_width_rank")
    result["bb_squeeze"] = bool(bb_width_rank is not None
                                and not pd.isna(bb_width_rank)
                                and bb_width_rank < 0.15)

    atr_5m = row5.get("atr")
    result["atr_5m"] = float(atr_5m) if not pd.isna(atr_5m) else 0.0
    atr_1m = row1.get("atr")
    result["atr_1m"] = float(atr_1m) if not pd.isna(atr_1m) else 0.0

    if consecutive_losses >= 3:
        result["block_reasons"].append("3 consecutive losses - cooldown")
        result["hard_block"] = True
        return result

    # ── Regime dispatch ──
    regime = "ranging"
    if cfg.get("use_dual_regime", True):
        regime = detect_trend_regime(df15, i15, df5, i5)
    result["regime"] = regime

    if regime in ("trend_up", "trend_down"):
        return _score_trend_pullback(
            result, regime, i15, df15, i5, df5, i1, df1,
            hour_utc, cfg, obi, ob_analysis or {}, funding_rate,
        )
    return _score_range_fade(
        result, i15, df15, i5, df5, i1, df1,
        hour_utc, cfg, obi, ob_analysis or {}, funding_rate,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PATH A: TREND PULLBACK — trade THUẬN trend, entry khi giá pullback
# ═══════════════════════════════════════════════════════════════════════════
def _score_trend_pullback(result: dict, regime: str,
                          i15: int, df15: pd.DataFrame,
                          i5: int, df5: pd.DataFrame,
                          i1: int, df1: pd.DataFrame,
                          hour_utc: int, cfg: dict,
                          obi: float, ob_analysis: dict,
                          funding_rate: float) -> dict:
    """
    Trend regime: CẤM fade ngược trend hoàn toàn. Chỉ vào lệnh thuận trend
    khi giá pullback về value zone (EMA21 1m / BB mid / VWAP).
    Đây chính là các cơ hội mà engine cũ ngồi block suốt 2 giờ trong log.
    """
    direction = "long" if regime == "trend_up" else "short"
    result["entry_mode"] = "trend_pullback"
    score = 0
    details = {}

    row15 = df15.iloc[i15]
    row5 = df5.iloc[i5]
    row5_prev = df5.iloc[i5 - 1]
    row1 = df1.iloc[i1]
    row1_prev = df1.iloc[i1 - 1]

    price1 = row1["close"]
    rsi1 = row1["rsi"]
    rsi5 = row5["rsi"]
    ema21_1 = row1.get("ema21")
    ema50_5 = row5.get("ema50")
    price5 = row5["close"]
    bb_upper1 = row1["bb_upper"]
    bb_lower1 = row1["bb_lower"]
    vwap1 = row1.get("vwap")

    if pd.isna(bb_upper1) or pd.isna(bb_lower1) or pd.isna(rsi1) or pd.isna(rsi5):
        return result

    # ── Guard: volatility spike → đứng ngoài ──
    if _atr_spike_block(df15, i15):
        result["block_reasons"].append("ATR spike 15m > 2x avg (volatility burst)")
        result["hard_block"] = True
        return result

    # ── Guard: trend còn nguyên vẹn trên 5m ──
    if not pd.isna(ema50_5):
        if direction == "long" and price5 < ema50_5:
            return result  # pullback đã phá gãy trend → không phải pullback
        if direction == "short" and price5 > ema50_5:
            return result

    bb_mid1 = (bb_upper1 + bb_lower1) / 2
    zone_tolerance = 0.0004  # 0.04%

    # ── Trend Age (Late-Trend Filter) ──
    is_late_trend = False
    if not pd.isna(ema50_5) and ema50_5 > 0:
        if abs(price1 - ema50_5) / ema50_5 > 0.008:
            is_late_trend = True

    # ── Base trigger: giá đã pullback vào value zone chưa? ──
    in_zone = False
    pattern_1m = detect_candle_pattern(row1, row1_prev)
    
    if direction == "long":
        if is_late_trend:
            zone_edge = ema50_5 if not pd.isna(ema50_5) else bb_mid1
        else:
            zone_edge = max(
                v for v in (ema21_1, bb_mid1, vwap1)
                if v is not None and not pd.isna(v)
            )
        in_zone = price1 <= zone_edge * (1 + zone_tolerance)
        # Pullback quá sâu = breakdown, không phải pullback
        if price1 < bb_lower1 and pattern_1m == "strong_bearish_close":
            return result
        # Hard condition: bắt buộc nến đảo chiều 1m
        if pattern_1m not in ("hammer", "bullish_engulfing", "strong_bullish_close"):
            return result
    else:
        if is_late_trend:
            zone_edge = ema50_5 if not pd.isna(ema50_5) else bb_mid1
        else:
            zone_edge = min(
                v for v in (ema21_1, bb_mid1, vwap1)
                if v is not None and not pd.isna(v)
            )
        in_zone = price1 >= zone_edge * (1 - zone_tolerance)
        if price1 > bb_upper1 and pattern_1m == "strong_bullish_close":
            return result
        # Hard condition: bắt buộc nến đảo chiều 1m
        if pattern_1m not in ("shooting_star", "bearish_engulfing", "strong_bearish_close"):
            return result

    if not in_zone:
        return result

    score += 30
    details["pullback_zone"] = 30

    # ── RSI pullback: dip lành mạnh, không phải crash ──
    if direction == "long":
        if 32 <= rsi1 <= 52:
            score += 10
            details["rsi_pullback"] = 10
        elif rsi1 < 25:
            score -= 10  # rơi tự do, không phải pullback
            details["rsi_pullback"] = -10
        else:
            details["rsi_pullback"] = 0
        if rsi5 >= 45:
            score += 5
            details["rsi_5m_intact"] = 5
        else:
            details["rsi_5m_intact"] = 0
    else:
        if 48 <= rsi1 <= 68:
            score += 10
            details["rsi_pullback"] = 10
        elif rsi1 > 75:
            score -= 10
            details["rsi_pullback"] = -10
        else:
            details["rsi_pullback"] = 0
        if rsi5 <= 55:
            score += 5
            details["rsi_5m_intact"] = 5
        else:
            details["rsi_5m_intact"] = 0

    # ── Candle reversal xác nhận pullback kết thúc (đã là điều kiện cứng) ──
    score += 10
    details["candle_1m"] = 10

    pattern_5m = detect_candle_pattern(row5, row5_prev)
    if direction == "long" and pattern_5m in ("hammer", "bullish_engulfing"):
        score += 10
        details["candle_5m"] = 10
    elif direction == "short" and pattern_5m in ("shooting_star", "bearish_engulfing"):
        score += 10
        details["candle_5m"] = 10
    else:
        details["candle_5m"] = 0

    # ── 15m context alignment ──
    ema9_15 = row15.get("ema9")
    ema21_15 = row15.get("ema21")
    price15 = row15["close"]
    if not pd.isna(ema9_15) and not pd.isna(ema21_15):
        if direction == "long" and ema9_15 > ema21_15 and price15 > ema21_15:
            score += 15
            details["trend_15m"] = 15
        elif direction == "short" and ema9_15 < ema21_15 and price15 < ema21_15:
            score += 15
            details["trend_15m"] = 15
        else:
            details["trend_15m"] = 0

    # ── Directional volume: phe thuận trend vẫn đang chiếm ưu thế ──
    up_vol_ratio, down_vol_ratio = _directional_volume(df1, i1, lookback=5)
    if direction == "long" and up_vol_ratio >= 0.55:
        score += 5
        details["directional_vol"] = 5
    elif direction == "short" and down_vol_ratio >= 0.55:
        score += 5
        details["directional_vol"] = 5
    else:
        details["directional_vol"] = 0

    # ── OBI (smoothed): sổ lệnh phải KHÔNG chống lại trend ──
    obi_05 = ob_analysis.get("obi_05", 0.0) if ob_analysis else 0.0
    if obi != 0.0 or obi_05 != 0.0:
        if direction == "long":
            if obi <= -0.30 or obi_05 <= -0.30:
                result["block_reasons"].append(f"OBI={obi:.2f}/OBI_05={obi_05:.2f} against uptrend pullback")
                result["hard_block"] = True
            elif obi >= 0.15 and obi_05 >= 0.15:
                score += 15
                details["obi_bonus"] = 15
            elif obi <= -0.15 or obi_05 <= -0.15:
                score -= 10
                details["obi_penalty"] = -10
        else:
            if obi >= 0.30 or obi_05 >= 0.30:
                result["block_reasons"].append(f"OBI={obi:.2f}/OBI_05={obi_05:.2f} against downtrend pullback")
                result["hard_block"] = True
            elif obi <= -0.15 and obi_05 <= -0.15:
                score += 15
                details["obi_bonus"] = 15
            elif obi >= 0.15 or obi_05 >= 0.15:
                score -= 10
                details["obi_penalty"] = -10

    # ── Wall support / spread ──
    if ob_analysis:
        cvd = ob_analysis.get("cvd", 0.0)
        if direction == "long" and cvd < 0:
            result["block_reasons"].append(f"CVD={cvd:.4f} against long")
            result["hard_block"] = True
        elif direction == "short" and cvd > 0:
            result["block_reasons"].append(f"CVD={cvd:.4f} against short")
            result["hard_block"] = True

        if direction == "long" and ob_analysis.get("bid_wall"):
            score += 10
            details["bid_wall"] = 10
        elif direction == "short" and ob_analysis.get("ask_wall"):
            score += 10
            details["ask_wall"] = 10
        if ob_analysis.get("spread_pct", 0) > 0.02:
            score -= 10
            details["spread_penalty"] = -10

    # ── Funding bias ──
    if cfg.get("use_funding_rate", True) and abs(funding_rate) > 0.0001:
        if direction == "long" and funding_rate < -0.0001:
            score += 5
            details["funding_bias"] = 5
        elif direction == "short" and funding_rate > 0.0001:
            score += 5
            details["funding_bias"] = 5

    # ── Session bonus (Asia penalty được xử lý bằng threshold bump ở engine) ──
    if cfg.get("use_time_filter", True) and 7 <= hour_utc <= 17:
        score += 10
        details["time_filter"] = 10

    return _finalize_result(result, direction, score, details)


# ═══════════════════════════════════════════════════════════════════════════
# PATH B: RANGE FADE — mean-reversion 2 chiều, CHỈ trong ranging regime
# ═══════════════════════════════════════════════════════════════════════════
def _score_range_fade(result: dict,
                      i15: int, df15: pd.DataFrame,
                      i5: int, df5: pd.DataFrame,
                      i1: int, df1: pd.DataFrame,
                      hour_utc: int, cfg: dict,
                      obi: float, ob_analysis: dict,
                      funding_rate: float) -> dict:
    """
    Engine v2 fade logic với các fix v3:
      - vwap_stretch_base bị khóa chặt (cần VWAP đủ dữ liệu + nến đảo chiều)
      - Bỏ momentum flip (bug giữ điểm của hướng cũ) → thay bằng hard block
      - Veto mới: cả 5m VÀ 15m trend ngược → hard block bất kể candle
      - Penalty trend tăng -10 → -15
      - Bỏ Asia -5 (engine nâng min_score thay thế)
    """
    result["entry_mode"] = "range_fade"
    score = 0
    details = {}
    direction = None

    row15 = df15.iloc[i15]
    row5 = df5.iloc[i5]
    row5_prev = df5.iloc[i5 - 1]
    row1 = df1.iloc[i1]
    row1_prev = df1.iloc[i1 - 1]

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

    up_vol_ratio, down_vol_ratio = _directional_volume(df1, i1, lookback=5)
    vol_climax = (not pd.isna(vol1) and not pd.isna(vol_avg20_1) and vol1 > vol_avg20_1 * 1.5)

    vwap_stretch_pct = 0.0
    if not pd.isna(vwap1) and vwap1 > 0:
        vwap_stretch_pct = abs(price1 - vwap1) / vwap1 * 100

    is_bullish_candle = row1["close"] > row1["open"]
    is_bearish_candle = row1["close"] < row1["open"]

    # --------------------------------------------------
    # MODULE 1: BASE SIGNALS
    # --------------------------------------------------
    if price1 <= bb_lower1:
        direction = "long"
        score += 25
        details["bb_touch"] = 25

        pattern_1m = detect_candle_pattern(row1, row1_prev)
        if pattern_1m == "strong_bearish_close":
            score -= 15
            details["strong_close_penalty"] = -15
        else:
            details["strong_close_penalty"] = 0

        if vol_climax and up_vol_ratio > 0.6:
            score += 15
            details["directional_vol"] = 15
        elif vol_climax and down_vol_ratio > 0.6:
            score += 5
            details["directional_vol"] = 5
        elif vol_climax:
            score += 10
            details["directional_vol"] = 10
        else:
            details["directional_vol"] = 0

        if not pd.isna(ema50_5):
            if price5 > ema50_5:
                score += 10
                details["ema50_trend"] = 10
            else:
                score -= 15
                details["ema50_trend"] = -15
        else:
            details["ema50_trend"] = 0

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
        score += 25
        details["bb_touch"] = 25

        pattern_1m = detect_candle_pattern(row1, row1_prev)
        if pattern_1m == "strong_bullish_close":
            score -= 15
            details["strong_close_penalty"] = -15
        else:
            details["strong_close_penalty"] = 0

        if vol_climax and down_vol_ratio > 0.6:
            score += 15
            details["directional_vol"] = 15
        elif vol_climax and up_vol_ratio > 0.6:
            score += 5
            details["directional_vol"] = 5
        elif vol_climax:
            score += 10
            details["directional_vol"] = 10
        else:
            details["directional_vol"] = 0

        if not pd.isna(ema50_5):
            if price5 < ema50_5:
                score += 10
                details["ema50_trend"] = 10
            else:
                score -= 15
                details["ema50_trend"] = -15
        else:
            details["ema50_trend"] = 0

        if vwap_stretch_pct > 0.5:
            score += 20
            details["vwap_stretch"] = 20
        elif vwap_stretch_pct > 0.3:
            score += 10
            details["vwap_stretch"] = 10
        else:
            details["vwap_stretch"] = 0

    else:
        # Base Signal 2: RSI Extreme
        if rsi1 < 25:
            direction = "long"
            score += 35
            details["rsi_extreme_touch"] = 35
        elif rsi1 > 75:
            direction = "short"
            score += 35
            details["rsi_extreme_touch"] = 35
        else:
            # Base Signal 3: EMA Reversion
            ema9_1 = row1.get("ema9")
            dist_pct = 0.0
            if not pd.isna(ema9_1) and ema9_1 > 0:
                dist_pct = (price1 - ema9_1) / ema9_1 * 100

            if dist_pct < -0.40 and rsi1 < 45:
                direction = "long"
                score += 25
                details["ema_reversion"] = 25
            elif dist_pct > 0.40 and rsi1 > 55:
                direction = "short"
                score += 25
                details["ema_reversion"] = 25
            # Base Signal 4: VWAP Stretch — FIX v3: khóa chặt
            # Yêu cầu: (1) VWAP đã có >= vwap_min_hour_utc giờ dữ liệu trong ngày
            # (2) nến hiện tại ĐÃ quay đầu về phía VWAP (reversal confirm)
            # Đây là nguồn tín hiệu counter-trend sai lớn nhất trong log cũ.
            elif (vwap_stretch_pct > 0.45
                  and hour_utc >= cfg.get("vwap_min_hour_utc", 3)):
                if price1 < vwap1 and is_bullish_candle:
                    direction = "long"
                    score += 25
                    details["vwap_stretch_base"] = 25
                elif price1 > vwap1 and is_bearish_candle:
                    direction = "short"
                    score += 25
                    details["vwap_stretch_base"] = 25
                else:
                    return result
            else:
                return result

    # --------------------------------------------------
    # REGIME FILTER phụ (ATR spike + chuỗi nến + EMA slope cục bộ)
    # --------------------------------------------------
    if direction is not None and cfg.get("use_regime_filter", False):
        is_regime_blocked, regime_reason = detect_regime_for_entry(df15, i15, direction)
        if is_regime_blocked:
            result["block_reasons"].append(f"regime_filter: {regime_reason}")
            result["hard_block"] = True
            result["score_details"] = details
            return result
        else:
            score += 5
            details["regime_ok"] = 5

    # --------------------------------------------------
    # MODULE 2: RSI CONFIRMATION (1m and 5m)
    # --------------------------------------------------
    if direction == "long":
        if "rsi_extreme_touch" not in details:
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
        else:
            details["rsi_1m"] = 0
            details["rsi_5m"] = 0

        pattern_1m = detect_candle_pattern(row1, row1_prev)
        if pattern_1m in ["hammer", "bullish_engulfing"]:
            score += 10
            details["candle_1m"] = 10
        else:
            details["candle_1m"] = 0

    else:  # short
        if "rsi_extreme_touch" not in details:
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
        else:
            details["rsi_1m"] = 0
            details["rsi_5m"] = 0

        pattern_1m = detect_candle_pattern(row1, row1_prev)
        if pattern_1m in ["shooting_star", "bearish_engulfing"]:
            score += 10
            details["candle_1m"] = 10
        else:
            details["candle_1m"] = 0

    # --------------------------------------------------
    # MODULE 2.5: 15M TREND CONTEXT — penalty tăng lên -15
    # --------------------------------------------------
    ema9_15 = row15.get("ema9")
    ema21_15 = row15.get("ema21")
    price15 = row15["close"]

    if cfg.get("use_15m_context", True):
        if not pd.isna(ema9_15) and not pd.isna(ema21_15):
            if direction == "long":
                if ema9_15 > ema21_15 and price15 > ema21_15:
                    score += 15
                    details["trend_15m"] = 15
                elif ema9_15 < ema21_15 and price15 < ema21_15:
                    score -= 15
                    details["trend_15m"] = -15
                else:
                    details["trend_15m"] = 0
            else:  # short
                if ema9_15 < ema21_15 and price15 < ema21_15:
                    score += 15
                    details["trend_15m"] = 15
                elif ema9_15 > ema21_15 and price15 > ema21_15:
                    score -= 15
                    details["trend_15m"] = -15
                else:
                    details["trend_15m"] = 0
        else:
            details["trend_15m"] = 0
    else:
        details["trend_15m"] = 0

    # --------------------------------------------------
    # MODULE 3: BB WIDTH FILTER — soft penalty
    # --------------------------------------------------
    bb_width = row1["bb_width_pct"]
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
            score += 5
            details["bb_width"] = 5
        else:
            details["bb_width"] = 0
    else:
        details["bb_width"] = 0

    # --------------------------------------------------
    # MODULE 4: BB %B DEPTH — bonus/penalty
    # --------------------------------------------------
    bb_pct_b = row1.get("bb_pct_b")
    bb_pct_b_deep = cfg.get("bb_pct_b_deep_threshold", 0.15)

    if cfg.get("use_bb_pct_b", True) and bb_pct_b is not None and not pd.isna(bb_pct_b):
        if direction == "long":
            if bb_pct_b < -bb_pct_b_deep:
                score -= 10
                details["bb_pct_b"] = -10
            elif bb_pct_b <= 0:
                score += 10
                details["bb_pct_b"] = 10
            else:
                details["bb_pct_b"] = 0
        else:  # short
            if bb_pct_b > 1 + bb_pct_b_deep:
                score -= 10
                details["bb_pct_b"] = -10
            elif bb_pct_b >= 1.0:
                score += 10
                details["bb_pct_b"] = 10
            else:
                details["bb_pct_b"] = 0
    else:
        details["bb_pct_b"] = 0

    # --------------------------------------------------
    # MODULE 5-7: REVERSAL CONFIRMATION (divergence đã fix v3)
    # --------------------------------------------------
    reversal_confirmations = 0

    if cfg.get("use_rsi_divergence", True):
        if _check_rsi_divergence(df1, i1, direction):
            reversal_confirmations += 1

    stoch_k = row5.get("stoch_k")
    stoch_d = row5.get("stoch_d")
    stoch_k_prev = row5_prev.get("stoch_k")
    stoch_d_prev = row5_prev.get("stoch_d")

    if cfg.get("use_stoch_cross", True):
        stoch_values = [stoch_k, stoch_d, stoch_k_prev, stoch_d_prev]
        if not any(v is None or (isinstance(v, float) and pd.isna(v)) for v in stoch_values):
            if direction == "long" and stoch_k_prev < stoch_d_prev and stoch_k > stoch_d and stoch_k < 20:
                reversal_confirmations += 1
            elif direction == "short" and stoch_k_prev > stoch_d_prev and stoch_k < stoch_d and stoch_k > 80:
                reversal_confirmations += 1

    macd_hist = row5.get("macd_hist")
    macd_hist_prev = row5_prev.get("macd_hist")

    if cfg.get("use_macd_momentum", True):
        if not pd.isna(macd_hist) and not pd.isna(macd_hist_prev):
            if direction == "long" and macd_hist > macd_hist_prev and macd_hist < 0:
                reversal_confirmations += 1
            elif direction == "short" and macd_hist < macd_hist_prev and macd_hist > 0:
                reversal_confirmations += 1

    if reversal_confirmations == 1:
        score += 8
        details["reversal_confirm"] = 8
    elif reversal_confirmations == 2:
        score += 15
        details["reversal_confirm"] = 15
    elif reversal_confirmations == 3:
        score += 20
        details["reversal_confirm"] = 20
    else:
        details["reversal_confirm"] = 0

    # --------------------------------------------------
    # MODULE 8: CANDLESTICK PATTERN 5m BONUS
    # --------------------------------------------------
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
    # MODULE 9: TIME-OF-DAY — chỉ còn bonus London/NY
    # (Asia không còn -5: engine nâng min_score +asia_score_bump thay thế,
    #  mạnh hơn nhiều so với penalty tượng trưng)
    # --------------------------------------------------
    if cfg.get("use_time_filter", True) and 7 <= hour_utc <= 17:
        score += 10
        details["time_filter"] = 10
    else:
        details["time_filter"] = 0

    # --------------------------------------------------
    # MODULE 10: OBI (smoothed) — FIX v3: bỏ momentum flip
    # Flip cũ giữ nguyên điểm đã chấm cho hướng ngược lại (bug).
    # Giờ: sổ lệnh chống mạnh → hard block, chống nhẹ → penalty.
    # --------------------------------------------------
    obi_05 = ob_analysis.get("obi_05", 0.0) if ob_analysis else 0.0
    if direction is not None and (obi != 0.0 or obi_05 != 0.0):
        if direction == "long":
            if obi < -0.25 or obi_05 < -0.25:
                result["block_reasons"].append(f"OBI={obi:.2f}/OBI_05={obi_05:.2f} (sellers dominant, smoothed)")
                result["hard_block"] = True
            elif obi > 0.30 and obi_05 > 0.30:
                score += 15
                details["obi_bonus"] = 15
            elif obi < -0.15 or obi_05 < -0.15:
                score -= 10
                details["obi_penalty"] = -10
        else:  # short
            if obi > 0.25 or obi_05 > 0.25:
                result["block_reasons"].append(f"OBI={obi:.2f}/OBI_05={obi_05:.2f} (buyers dominant, smoothed)")
                result["hard_block"] = True
            elif obi < -0.30 and obi_05 < -0.30:
                score += 15
                details["obi_bonus"] = 15
            elif obi > 0.15 or obi_05 > 0.15:
                score -= 10
                details["obi_penalty"] = -10

    if ob_analysis:
        cvd = ob_analysis.get("cvd", 0.0)
        if direction == "long" and cvd < 0:
            result["block_reasons"].append(f"CVD={cvd:.4f} against long")
            result["hard_block"] = True
        elif direction == "short" and cvd > 0:
            result["block_reasons"].append(f"CVD={cvd:.4f} against short")
            result["hard_block"] = True

        if direction == "long" and ob_analysis.get("bid_wall"):
            score += 15
            details["bid_wall"] = 15
        elif direction == "short" and ob_analysis.get("ask_wall"):
            score += 15
            details["ask_wall"] = 15

        spread_pct = ob_analysis.get("spread_pct", 0)
        if spread_pct > 0.02:
            score -= 10
            details["spread_penalty"] = -10

    # --------------------------------------------------
    # MODULE 11: FUNDING RATE BIAS
    # --------------------------------------------------
    if cfg.get("use_funding_rate", True) and abs(funding_rate) > 0.0001:
        if direction == "long" and funding_rate < -0.0001:
            score += 10
            details["funding_bias"] = 10
        elif direction == "short" and funding_rate > 0.0001:
            score += 10
            details["funding_bias"] = 10
        elif direction == "long" and funding_rate > 0.0005:
            score -= 10
            details["funding_bias"] = -10
        elif direction == "short" and funding_rate < -0.0005:
            score -= 10
            details["funding_bias"] = -10
        else:
            details["funding_bias"] = 0

    # --------------------------------------------------
    # VETO COMBOS — FIX v3: thêm veto "cả 2 khung trend ngược"
    # --------------------------------------------------
    # NEW: 5m VÀ 15m đều ngược hướng fade → block bất kể candle pattern.
    # (Lệnh SHORT 03:29 score=75 trong log lọt qua vì candle_1m=+10 lách veto cũ.)
    if details.get("ema50_trend", 0) < 0 and details.get("trend_15m", 0) < 0:
        result["block_reasons"].append("5m_and_15m_trend_against: counter-trend fade blocked")
        result["hard_block"] = True

    if details.get("ema50_trend", 0) < 0 and details.get("bb_pct_b", 0) < 0:
        result["block_reasons"].append("trend_against + deep_bb_break: high risk fade")
        result["hard_block"] = True

    if details.get("trend_15m", 0) < 0 and details.get("strong_close_penalty", 0) < 0:
        result["block_reasons"].append("15m_trend_against + strong_close: high risk")
        result["hard_block"] = True

    if details.get("trend_15m", 0) < 0 and details.get("candle_5m", 0) == 0 and details.get("candle_1m", 0) == 0:
        result["block_reasons"].append("trend_against without candle reversal: high risk fade")
        result["hard_block"] = True

    return _finalize_result(result, direction, score, details)