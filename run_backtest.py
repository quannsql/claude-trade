"""
run_backtest.py
================
MAIN backtest engine. Simulates placing orders according to the scoring engine
(Modules 1-5), calculates actual fees + slippage, applies TP/SL/time-stop,
and outputs an honest report: win rate, profit factor, drawdown, trades/day.

NO data is assumed in the final report - everything is calculated directly
from the simulated execution on the downloaded historical data.

Usage:
    python fetch_data.py     # run once to download data
    python run_backtest.py   # run backtest, print report, export CSV trade log
"""

import pandas as pd
import numpy as np
import os
from tabulate import tabulate

from indicators import add_indicators, score_setup
from filters import compute_dynamic_levels

# ---------------------------------------------------------
# BACKTEST CONFIGURATION — BTC ONLY (ETH đã bị loại bỏ)
# ---------------------------------------------------------
# ETH bị loại vì downtrend loss quá lớn (-$32 trong 2 ngày 28–29/6).
# Sẽ thêm lại ETH sau khi regime filter được implement và verify.
#
# Thay đổi so với bản cũ:
#   - ACTIVE_PROFILE = "btc_high_freq" (chỉ BTC, trade nhiều hơn)
#   - symbols = ["BTCUSDT"] trong BASE_CONFIG
#   - Thêm max_loss_per_trade_usd (hard dollar SL)
#   - Tăng max_trades_per_day lên 40 để bù lượng thiếu từ việc bỏ ETH
#   - Giảm daily_loss_limit_usd xuống 8
# ---------------------------------------------------------

ACTIVE_PROFILE = "btc_high_freq"

BASE_CONFIG = {
    # ETH đã bị loại bỏ hoàn toàn
    "symbols": ["BTCUSDT"],
    "leverage": 20,
    "min_score_half": 50,
    "min_score_full": 65,

    "taker_fee_pct": 0.045,
    "maker_fee_pct": 0.015,

    "max_consecutive_losses": 3,
    "cooldown_minutes": 30,
    "max_trades_per_day": 20,
    "min_equity_usd": 80.0,
    "starting_equity": 100.0,

    "allowed_hours_utc": None,
    "max_entry_volume_ratio": 5.0,
    "max_entry_bb_width_pct": None,
    "max_5m_atr_pct": None,
    "max_5m_ema21_distance_pct": None,
    "move_sl_to_breakeven_after_tp1": False,

    # ── Scoring Engine v2 config ──
    "bb_width_max_pct": 1.0,         # BB width > này → soft penalty nặng (-15)
    "bb_width_warn_pct": 0.6,        # BB width > này → soft penalty nhẹ (-5)
    "bb_pct_b_deep_threshold": 0.15, # %B phá qua sâu hơn threshold → penalty
    "use_time_filter": True,         # Soft penalty Asia session, bonus London/NY
    "use_rsi_divergence": True,      # RSI divergence check (+15 nếu phát hiện)
    "use_stoch_cross": True,         # StochRSI %K/%D cross check (+15)
    "use_macd_momentum": True,       # MACD histogram momentum check (+10)
    "use_candle_5m": True,           # 5m candlestick pattern bonus (+10)
    "use_bb_width_filter": True,     # BB width soft penalty (không hard block)
    "use_bb_pct_b": True,            # BB %B depth bonus/penalty

    # Hard dollar stop-loss: đóng ngay nếu unrealized loss vượt ngưỡng này
    # Bảo vệ khỏi loss lớn không chạm % SL (ví dụ ETH -$7)
    "max_loss_per_trade_usd": 3.0,

    "data_dir": "data",
    "output_dir": "results",
}

PROFILES = {
    # ── BTC ONLY HIGH FREQUENCY (Profile chính) ──────────────────────────────
    # Phương châm: trade nhiều lệnh BTC nhỏ, lãi ít mỗi lệnh nhưng tổng dương.
    # ETH bị loại. Bù đắp bằng max_trades_per_day=40 và min_score_half=45.
    "btc_high_freq": {
        "symbols": ["BTCUSDT"],
        "leverage": 20,
        "margin_full": 100.0,
        "margin_half": 50.0,
        "min_score_half": 40,
        "min_score_full": 60,

        "tp1_pct": 0.15,
        "tp2_pct": 0.30,
        "sl_pct": 0.20,
        "time_stop_minutes": 30,

        # Hard dollar SL override: thoát ngay nếu lỗ quá $3 (không phụ thuộc %)
        "max_loss_per_trade_usd": 3.0,

        "move_sl_to_breakeven_after_tp1": True,
        "use_dynamic_tp_sl": False,

        "entry_order_type": "limit",
        "use_maker_for_entry": True,
        "use_maker_for_tp": True,
        "limit_offset_pct": 0.0,
        "limit_timeout_bars": 1,
        "slippage_pct": 0.0,

        "allowed_hours_utc": list(range(0, 24)),
        "max_trades_per_day": 60,
        "daily_loss_limit_usd": 8.0,
        "cooldown_minutes": 5,
        "max_consecutive_losses": 3,
        "min_equity_usd": 50.0,

        "use_regime_filter": False,
        "use_momentum_filter": False,
    },

    # ── Scalp nhỏ (giữ để tham khảo) ────────────────────────────────────────
    "scalp_1usd": {
        "margin_full": 50.0,
        "margin_half": 25.0,
        "tp1_pct": 0.08,
        "tp2_pct": 0.15,
        "sl_pct": 0.25,
        "time_stop_minutes": 5,
        "daily_loss_limit_usd": 3.0,
        "entry_order_type": "market",
        "use_maker_for_entry": False,
        "use_maker_for_tp": True,
        "limit_offset_pct": 0.0,
        "limit_timeout_bars": 0,
        "slippage_pct": 0.02,
    },

    # ── Swing target $5/trade ────────────────────────────────────────────────
    "swing_5usd": {
        "margin_full": 50.0,
        "margin_half": 25.0,
        "tp1_pct": 0.30,
        "tp2_pct": 0.55,
        "sl_pct": 0.40,
        "time_stop_minutes": 30,
        "entry_order_type": "limit",
        "use_maker_for_entry": True,
        "use_maker_for_tp": True,
        "limit_offset_pct": 0.0,
        "limit_timeout_bars": 1,
        "slippage_pct": 0.0,
        "daily_loss_limit_usd": 8.0,
    },

    # ── Conservative testnet ─────────────────────────────────────────────────
    "conservative_testnet": {
        "symbols": ["BTCUSDT"],
        "leverage": 10,
        "margin_full": 10.0,
        "margin_half": 5.0,
        "min_score_half": 65,
        "min_score_full": 75,
        "tp1_pct": 0.30,
        "tp2_pct": 0.55,
        "sl_pct": 0.40,
        "time_stop_minutes": 30,
        "entry_order_type": "limit",
        "use_maker_for_entry": True,
        "use_maker_for_tp": True,
        "limit_offset_pct": 0.0,
        "limit_timeout_bars": 1,
        "slippage_pct": 0.0,
        "allowed_hours_utc": list(range(8, 22)),
        "max_entry_volume_ratio": 2.0,
        "max_entry_bb_width_pct": 0.45,
        "move_sl_to_breakeven_after_tp1": False,
        "max_trades_per_day": 6,
        "daily_loss_limit_usd": 2.0,
        "cooldown_minutes": 60,
        "min_equity_usd": 50.0,
        "use_regime_filter": True,
        "use_momentum_filter": True,
    },

    # ── High freq cũ (có ETH — giữ để tham khảo, không dùng) ────────────────
    "high_freq_scalp": {
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "leverage": 20,
        "margin_full": 100.0,
        "margin_half": 50.0,
        "min_score_half": 45,
        "min_score_full": 55,
        "tp1_pct": 0.20,
        "tp2_pct": 0.40,
        "sl_pct": 0.25,
        "time_stop_minutes": 30,
        "use_dynamic_tp_sl": False,
        "entry_order_type": "limit",
        "use_maker_for_entry": True,
        "use_maker_for_tp": True,
        "limit_offset_pct": 0.0,
        "limit_timeout_bars": 2,
        "slippage_pct": 0.0,
        "allowed_hours_utc": list(range(0, 24)),
        "max_trades_per_day": 100,
        "daily_loss_limit_usd": 15.0,
        "cooldown_minutes": 5,
        "min_equity_usd": 50.0,
        "use_regime_filter": False,
        "use_momentum_filter": False,
    },

    # ── Smart / ATR-based ────────────────────────────────────────────────────
    "smart": {
        "symbols": ["BTCUSDT"],
        "leverage": 10,
        "margin_full": 10.0,
        "margin_half": 5.0,
        "min_score_half": 50,
        "min_score_full": 60,
        "tp1_pct": 0.35,
        "tp2_pct": 0.55,
        "sl_pct": 0.30,
        "time_stop_minutes": 120,
        "use_dynamic_tp_sl": True,
        "tp1_atr_mult": 2.0,
        "tp2_atr_mult": 4.0,
        "sl_atr_mult": 2.0,
        "use_trailing_stop": False,
        "entry_order_type": "limit",
        "use_maker_for_entry": True,
        "use_maker_for_tp": True,
        "limit_offset_pct": 0.0,
        "limit_timeout_bars": 1,
        "slippage_pct": 0.0,
        "allowed_hours_utc": list(range(0, 24)),
        "max_trades_per_day": 20,
        "daily_loss_limit_usd": 5.0,
        "cooldown_minutes": 15,
        "min_equity_usd": 50.0,
        "use_regime_filter": False,
        "use_momentum_filter": False,
    },
}

CONFIG = {**BASE_CONFIG, **PROFILES[ACTIVE_PROFILE]}
CONFIG["profile_name"] = ACTIVE_PROFILE

def load_data(symbol: str, timeframe: str, data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{symbol}_{timeframe}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run 'python fetch_data.py' first."
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def align_index(target_ts, df: pd.DataFrame, last_idx: int, timeframe_minutes: int = 1) -> int:
    """
    Finds the most recent fully closed candle index for the 1m signal time.
    Backtest evaluates a 1m candle after it closes, so higher-timeframe candles
    are available only when their open timestamp + timeframe <= target_ts + 1m.
    """
    closed_cutoff = target_ts + pd.Timedelta(minutes=1 - timeframe_minutes)
    n = len(df)
    idx = last_idx
    while idx + 1 < n and df.iloc[idx + 1]["timestamp"] <= closed_cutoff:
        idx += 1
    return idx


def try_fill_limit_entry(direction: str, limit_price: float, df1: pd.DataFrame,
                          signal_idx: int, timeout_bars: int):
    """
    Simulates placing a LIMIT order at limit_price right after a signal,
    and waiting max timeout_bars 1m candles for a fill.

    Long: filled if low of subsequent candle touches <= limit_price.
    Short: filled if high touches >= limit_price.

    Returns (fill_idx, fill_price) if filled, or (None, None) if timeout
    (DO NOT chase price - skip setup).
    """
    n = len(df1)
    for offset in range(1, timeout_bars + 1):
        idx = signal_idx + offset
        if idx >= n:
            return None, None
        bar = df1.iloc[idx]
        if direction == "long" and bar["low"] <= limit_price:
            return idx, limit_price
        if direction == "short" and bar["high"] >= limit_price:
            return idx, limit_price
    return None, None


def simulate_trade(direction: str, entry_price: float, entry_time,
                    df1: pd.DataFrame, entry_idx: int, margin: float,
                    cfg: dict, atr_5m: float = 0.0) -> dict:
    """
    Simulates 1 trade from entry to close (TP1+TP2, SL, trailing-stop, or
    time-stop), iterating through 1m candles after entry_idx.

    Supports:
    - Fixed % TP/SL (original behavior)
    - Dynamic ATR-based TP/SL (when use_dynamic_tp_sl=True and atr_1m > 0)
    - Trailing stop (when use_trailing_stop=True)

    Returns dict of trade result, or None if limit order not filled.
    """
    leverage = cfg["leverage"]
    notional = margin * leverage

    entry_order_type = cfg.get("entry_order_type", "market")

    if entry_order_type == "limit":
        offset_pct = cfg.get("limit_offset_pct", 0.0) / 100
        if direction == "long":
            limit_price = entry_price * (1 - offset_pct)
        else:
            limit_price = entry_price * (1 + offset_pct)

        fill_idx, fill_price = try_fill_limit_entry(
            direction, limit_price, df1, entry_idx, cfg.get("limit_timeout_bars", 1)
        )
        if fill_idx is None:
            return None  # NOT filled within wait time -> skip, do not chase price
        entry_idx = fill_idx  # TP/SL/time-stop timer starts from actual fill
    else:
        slippage = cfg["slippage_pct"] / 100
        if direction == "long":
            fill_price = entry_price * (1 + slippage)
        else:
            fill_price = entry_price * (1 - slippage)

    entry_fee_pct = cfg["maker_fee_pct"] if cfg["use_maker_for_entry"] else cfg["taker_fee_pct"]
    entry_fee = notional * (entry_fee_pct / 100)

    # ------- Dynamic ATR-based TP/SL or fixed % -------
    if cfg.get("use_dynamic_tp_sl", False) and atr_5m > 0:
        levels = compute_dynamic_levels(
            fill_price, direction, atr_5m,
            tp1_atr_mult=cfg.get("tp1_atr_mult", 1.5),
            tp2_atr_mult=cfg.get("tp2_atr_mult", 3.0),
            sl_atr_mult=cfg.get("sl_atr_mult", 1.2),
        )
        tp1_price = levels["tp1_price"]
        tp2_price = levels["tp2_price"]
        sl_price = levels["sl_price"]
        tp1_pct = levels["tp1_pct"] / 100
        tp2_pct = levels["tp2_pct"] / 100
        sl_pct = levels["sl_pct"] / 100
    else:
        tp1_pct = cfg["tp1_pct"] / 100
        tp2_pct = cfg["tp2_pct"] / 100
        sl_pct = cfg["sl_pct"] / 100

        if direction == "long":
            tp1_price = fill_price * (1 + tp1_pct)
            tp2_price = fill_price * (1 + tp2_pct)
            sl_price = fill_price * (1 - sl_pct)
        else:
            tp1_price = fill_price * (1 - tp1_pct)
            tp2_price = fill_price * (1 - tp2_pct)
            sl_price = fill_price * (1 + sl_pct)

    # ------- Trailing stop config -------
    use_trailing = cfg.get("use_trailing_stop", False)
    trailing_activate_pct = cfg.get("trailing_activate_pct", 0.15) / 100
    trailing_lock_pct = cfg.get("trailing_lock_pct", 0.50)
    max_favorable_move = 0.0  # Track best price reached

    remaining_notional = notional
    realized_pnl = 0.0
    fees_paid = entry_fee
    tp1_hit = False
    exit_reason = None
    exit_price = None
    exit_time = None
    exit_idx = None

    max_bars = cfg["time_stop_minutes"]
    n = len(df1)

    for offset in range(1, max_bars + 1):
        idx = entry_idx + offset
        if idx >= n:
            exit_reason = "data_end"
            exit_price = df1.iloc[-1]["close"]
            exit_time = df1.iloc[-1]["timestamp"]
            exit_idx = n - 1
            break

        bar = df1.iloc[idx]
        high, low, close = bar["high"], bar["low"], bar["close"]

        # ------- Update trailing stop -------
        if use_trailing:
            if direction == "long":
                favorable = (high - fill_price) / fill_price
            else:
                favorable = (fill_price - low) / fill_price

            max_favorable_move = max(max_favorable_move, favorable)

            if max_favorable_move >= trailing_activate_pct:
                # Trail: move SL to lock a portion of the best move
                lock_amount = max_favorable_move * trailing_lock_pct
                if direction == "long":
                    trailing_sl = fill_price * (1 + lock_amount)
                    sl_price = max(sl_price, trailing_sl)
                else:
                    trailing_sl = fill_price * (1 - lock_amount)
                    sl_price = min(sl_price, trailing_sl)

        if direction == "long":
            hit_sl = low <= sl_price
            hit_tp1 = (not tp1_hit) and high >= tp1_price
            hit_tp2 = tp1_hit and high >= tp2_price
        else:
            hit_sl = high >= sl_price
            hit_tp1 = (not tp1_hit) and low <= tp1_price
            hit_tp2 = tp1_hit and low <= tp2_price

        # Hard dollar stop-loss: thoát khẩn cấp nếu unrealized loss vượt ngưỡng USD
        # Kiểm tra dựa trên close của nến hiện tại (conservative estimate)
        # Bảo vệ khỏi trường hợp giá trượt dài nhưng chưa chạm % SL
        max_loss_usd = cfg.get("max_loss_per_trade_usd", None)
        if max_loss_usd is not None and not hit_sl:
            if direction == "long":
                unrealized_pct = (close - fill_price) / fill_price
            else:
                unrealized_pct = (fill_price - close) / fill_price
            unrealized_pnl = remaining_notional * unrealized_pct
            if unrealized_pnl < -max_loss_usd:
                hit_sl = True  # Kích hoạt thoát — exit_reason sẽ là "hard_dollar_sl"

        # SL is checked first (pessimistic assumption: if both SL and TP are hit in same candle, SL triggers)
        if hit_sl:
            # Xác định lý do thoát
            if max_loss_usd is not None:
                if direction == "long":
                    check_pnl = (close - fill_price) / fill_price
                else:
                    check_pnl = (fill_price - close) / fill_price
                is_hard_dollar = (remaining_notional * check_pnl) < -max_loss_usd
            else:
                is_hard_dollar = False

            # Determine if this is a trailing stop or original SL
            if use_trailing and max_favorable_move >= trailing_activate_pct and not is_hard_dollar:
                # Trailing stop: PnL based on actual SL level
                if direction == "long":
                    actual_pnl_pct = (sl_price - fill_price) / fill_price
                else:
                    actual_pnl_pct = (fill_price - sl_price) / fill_price
                pnl_remaining = remaining_notional * actual_pnl_pct
                exit_reason = "trailing_stop"
            elif is_hard_dollar:
                # Hard dollar SL: exit tại close của nến hiện tại
                if direction == "long":
                    pnl_remaining = remaining_notional * (close - fill_price) / fill_price
                else:
                    pnl_remaining = remaining_notional * (fill_price - close) / fill_price
                exit_reason = "hard_dollar_sl"
                exit_price = close  # Thoát tại close, không phải sl_price
                exit_time = bar["timestamp"]
                exit_idx = idx
                exit_fee = remaining_notional * (cfg["taker_fee_pct"] / 100)
                realized_pnl += pnl_remaining
                fees_paid += exit_fee
                break
            else:
                pnl_remaining = -remaining_notional * sl_pct
                exit_reason = "SL"
            exit_fee = remaining_notional * (cfg["taker_fee_pct"] / 100)  # SL is always market = taker
            realized_pnl += pnl_remaining
            fees_paid += exit_fee
            exit_price = sl_price
            exit_time = bar["timestamp"]
            exit_idx = idx
            break

        if hit_tp1:
            half_notional = remaining_notional / 2
            pnl_half = half_notional * tp1_pct
            tp_fee_pct = cfg["maker_fee_pct"] if cfg["use_maker_for_tp"] else cfg["taker_fee_pct"]
            fee_half = half_notional * (tp_fee_pct / 100)
            realized_pnl += pnl_half
            fees_paid += fee_half
            remaining_notional = half_notional
            tp1_hit = True
            if cfg.get("move_sl_to_breakeven_after_tp1", False):
                sl_price = fill_price
            continue

        if hit_tp2:
            pnl_rest = remaining_notional * tp2_pct
            tp_fee_pct = cfg["maker_fee_pct"] if cfg["use_maker_for_tp"] else cfg["taker_fee_pct"]
            fee_rest = remaining_notional * (tp_fee_pct / 100)
            realized_pnl += pnl_rest
            fees_paid += fee_rest
            exit_reason = "TP2"
            exit_price = tp2_price
            exit_time = bar["timestamp"]
            exit_idx = idx
            remaining_notional = 0
            break

    if exit_reason is None:
        idx = min(entry_idx + max_bars, n - 1)
        bar = df1.iloc[idx]
        close_price = bar["close"]
        if direction == "long":
            move_pct = (close_price - fill_price) / fill_price
        else:
            move_pct = (fill_price - close_price) / fill_price
        pnl_rest = remaining_notional * move_pct
        exit_fee = remaining_notional * (cfg["taker_fee_pct"] / 100)
        realized_pnl += pnl_rest
        fees_paid += exit_fee
        exit_reason = "time_stop"
        exit_price = close_price
        exit_time = bar["timestamp"]
        exit_idx = idx

    net_pnl = realized_pnl - fees_paid

    return {
        "direction": direction,
        "entry_time": entry_time,
        "entry_price": fill_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "margin": margin,
        "notional": notional,
        "gross_pnl": realized_pnl,
        "fees": fees_paid,
        "net_pnl": net_pnl,
        "win": net_pnl > 0,
        "exit_idx": exit_idx,
    }


def run_symbol_backtest(symbol: str, cfg: dict) -> pd.DataFrame:
    print(f"\n{'='*60}\nBacktest {symbol}\n{'='*60}")

    df1 = add_indicators(load_data(symbol, "1m", cfg["data_dir"]))
    df5 = add_indicators(load_data(symbol, "5m", cfg["data_dir"]))
    df15 = add_indicators(load_data(symbol, "15m", cfg["data_dir"]))

    print(f"  1m Candles: {len(df1)} | 5m: {len(df5)} | 15m: {len(df15)}")

    trades = []
    equity = cfg["starting_equity"]
    consecutive_losses = 0
    cooldown_until = None
    last_trade_exit_idx = -10**9
    trades_today = {}
    missed_limit_entries = 0

    idx5, idx15 = 0, 0
    min_start = 250

    for i1 in range(min_start, len(df1) - 10):
        ts = df1.iloc[i1]["timestamp"]

        if i1 - last_trade_exit_idx < 2:
            continue

        if cooldown_until is not None and ts < cooldown_until:
            continue

        if equity < cfg["min_equity_usd"]:
            break

        day_key = ts.date()
        if trades_today.get(day_key, 0) >= cfg["max_trades_per_day"]:
            continue

        idx5 = align_index(ts, df5, idx5, timeframe_minutes=5)
        idx15 = align_index(ts, df15, idx15, timeframe_minutes=15)

        setup = score_setup(
            idx15, df15, idx5, df5, i1, df1,
            consecutive_losses=consecutive_losses,
            hour_utc=ts.hour,
            cfg=cfg,
        )

        if setup["hard_block"] or setup["direction"] is None:
            continue

        score = setup["score"]
        if score < cfg["min_score_half"]:
            continue

        margin = cfg["margin_full"] if score >= cfg["min_score_full"] else cfg["margin_half"]

        entry_price = df1.iloc[i1]["close"]
        atr_5m = setup.get("atr_5m", 0.0)
        trade = simulate_trade(setup["direction"], entry_price, ts, df1, i1, margin, cfg, atr_5m=atr_5m)

        if trade is None:
            # Limit order not filled within wait time - according to rule
            # "no chasing price", skip this setup entirely (not a loss,
            # simply no trade was opened).
            missed_limit_entries += 1
            last_trade_exit_idx = i1  # still apply minimum 2 min gap
            continue

        trade["score"] = score
        trade["symbol"] = symbol
        trades.append(trade)

        equity += trade["net_pnl"]
        trade["equity_after"] = equity

        trades_today[day_key] = trades_today.get(day_key, 0) + 1

        if trade["win"]:
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            if consecutive_losses >= cfg["max_consecutive_losses"]:
                cooldown_until = trade["exit_time"] + pd.Timedelta(minutes=cfg["cooldown_minutes"])
                consecutive_losses = 0

        day_trades = [t for t in trades if t["entry_time"].date() == day_key]
        day_pnl = sum(t["net_pnl"] for t in day_trades)
        if day_pnl <= -cfg["daily_loss_limit_usd"]:
            cooldown_until = pd.Timestamp(day_key, tz="UTC") + pd.Timedelta(days=1)

        last_trade_exit_idx = int(trade.get("exit_idx", i1))

    if cfg.get("entry_order_type") == "limit":
        total_signals = len(trades) + missed_limit_entries
        miss_rate = (missed_limit_entries / total_signals * 100) if total_signals else 0
        print(f"  Scored Setups: {total_signals} | Limit Filled: {len(trades)} | "
              f"Missed (Not Filled): {missed_limit_entries} ({miss_rate:.1f}%)")

    return pd.DataFrame(trades)


def build_report(trades_df: pd.DataFrame, symbol: str, cfg: dict) -> dict:
    if trades_df.empty:
        return {
            "symbol": symbol, "total_trades": 0,
            "note": "No trades qualified for scoring."
        }

    wins = trades_df[trades_df["win"]]
    losses = trades_df[~trades_df["win"]]

    gross_profit = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity_curve = cfg["starting_equity"] + trades_df["net_pnl"].cumsum()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100
    max_drawdown_pct = drawdown.min()

    days_span = (trades_df["entry_time"].max() - trades_df["entry_time"].min()).days or 1
    trades_per_day = len(trades_df) / days_span

    total_fees = trades_df["fees"].sum()
    # Use gross_profit (total profit before fees, always >= 0) as denominator
    # so the metric "fees as % of gross profit" remains clearly meaningful
    # even when total gross_pnl (profit - loss before fees) is negative or near 0.
    fee_pct_of_gross = (total_fees / gross_profit * 100) if gross_profit > 0 else float("inf")

    return {
        "symbol": symbol,
        "total_trades": len(trades_df),
        "days_span": days_span,
        "trades_per_day": round(trades_per_day, 2),
        "win_rate_pct": round(len(wins) / len(trades_df) * 100, 2),
        "avg_win_usd": round(wins["net_pnl"].mean(), 4) if len(wins) else 0,
        "avg_loss_usd": round(losses["net_pnl"].mean(), 4) if len(losses) else 0,
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "total_fees_usd": round(total_fees, 2),
        "fee_pct_of_gross_pnl": round(fee_pct_of_gross, 1),
        "net_pnl_usd": round(trades_df["net_pnl"].sum(), 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "final_equity_usd": round(cfg["starting_equity"] + trades_df["net_pnl"].sum(), 2),
        "exit_reason_counts": trades_df["exit_reason"].value_counts().to_dict(),
    }


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    all_reports = []

    for symbol in CONFIG["symbols"]:
        trades_df = run_symbol_backtest(symbol, CONFIG)
        if not trades_df.empty:
            out_path = os.path.join(CONFIG["output_dir"], f"{symbol}_trades.csv")
            trades_df.to_csv(out_path, index=False)
            print(f"  Saved trade log: {out_path}")

        report = build_report(trades_df, symbol, CONFIG)
        all_reports.append(report)

    print("\n\n" + "=" * 70)
    print(f" CONSOLIDATED REPORT — Profile: {CONFIG['profile_name']}")
    print(f" TP1: +{CONFIG['tp1_pct']}% | TP2: +{CONFIG['tp2_pct']}% | SL: -{CONFIG['sl_pct']}% | "
          f"Entry: {CONFIG['entry_order_type']}")
    print("=" * 70)

    table_rows = []
    for r in all_reports:
        if r.get("total_trades", 0) == 0:
            table_rows.append([r["symbol"], 0, "-", "-", "-", "-", "-", "-"])
            continue
        table_rows.append([
            r["symbol"], r["total_trades"], r["trades_per_day"],
            f"{r['win_rate_pct']}%", r["profit_factor"],
            f"${r['net_pnl_usd']}", f"{r['max_drawdown_pct']}%",
            f"{r['fee_pct_of_gross_pnl']}%",
        ])

    headers = ["Symbol", "Total Trades", "Trades/Day", "Win Rate", "Profit Factor",
               "Net PnL", "Max Drawdown", "Fee/Winning Profit"]
    print(tabulate(table_rows, headers=headers, tablefmt="grid"))

    print("\nIMPORTANT NOTES:")
    print("  - These are REAL metrics simulated on Binance Futures historical data,")
    print("    not estimated or assumed figures.")
    print("  - Backtests using historical price DO NOT simulate real slippage/latency")
    print("    100% accurately on Hyperliquid - live results are often WORSE than backtest.")
    print("  - Order Book Imbalance module is excluded from scoring due to missing")
    print("    reliable historical data - must be tested separately when running live.")
    print("  - If Profit Factor < 1.3 or Win Rate is near 50%, the strategy LACKS")
    print("    a significant post-fee edge - do not run live with high leverage.")

    for r in all_reports:
        if r.get("total_trades", 0) > 0:
            print(f"\n--- Details for {r['symbol']} ---")
            for k, v in r.items():
                if k != "symbol":
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()