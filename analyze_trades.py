"""Analyze trade_history.csv (live testnet) vs backtest results."""
import pandas as pd
import numpy as np

# === TRADE HISTORY (Live Testnet) ===
print("=" * 80)
print("PHAN TICH TRADE HISTORY THUC TE (Hyperliquid Testnet)")
print("=" * 80)

df = pd.read_csv(r"d:\backtest\trade_history.csv")
print(f"\nTong fills: {len(df)}")
print(f"Coins: {df['coin'].unique()}")
print(f"\nHuong lenh:")
print(df["dir"].value_counts().to_string())

# Phan tich BTC
btc = df[df["coin"] == "BTC"].copy()
hype = df[df["coin"] == "HYPE"].copy()

print(f"\n--- BTC ---")
print(f"Tong fills BTC: {len(btc)}")
btc_open = btc[btc["dir"].str.contains("Open")]
btc_close = btc[btc["dir"].str.contains("Close")]
print(f"Open fills: {len(btc_open)}")
print(f"Close fills: {len(btc_close)}")
print(f"Tong fee BTC: {btc['fee'].sum():.6f}")
print(f"Tong closedPnl BTC: {btc['closedPnl'].sum():.6f}")
print(f"Tong notional BTC: {btc['ntl'].sum():.2f}")

print(f"\n--- HYPE ---")
print(f"Tong fills HYPE: {len(hype)}")
hype_open = hype[hype["dir"].str.contains("Open")]
hype_close = hype[hype["dir"].str.contains("Close")]
print(f"Open fills: {len(hype_open)}")
print(f"Close fills: {len(hype_close)}")
print(f"Tong fee HYPE: {hype['fee'].sum():.6f}")
print(f"Tong closedPnl HYPE: {hype['closedPnl'].sum():.6f}")

# Phan tich chi tiet
print("\n" + "=" * 80)
print("PHAN TICH CHI TIET CAC VI THE")
print("=" * 80)

print("\n--- BTC Open Long fills ---")
for _, r in btc_open.iterrows():
    print(f"  {r['time']} | px={r['px']} | sz={r['sz']} | ntl={r['ntl']:.2f} | fee={r['fee']:.6f}")

print(f"\n--- BTC Close fills ---")
for _, r in btc_close.iterrows():
    print(f"  {r['time']} | px={r['px']} | sz={r['sz']} | ntl={r['ntl']:.2f} | fee={r['fee']:.6f} | pnl={r['closedPnl']:.6f}")

total_open_size = btc_open["sz"].sum()
total_close_size = btc_close["sz"].sum()
print(f"\nBTC: Tong size Open={total_open_size:.8f} | Tong size Close={total_close_size:.8f}")
print(f"BTC: Chenh lech (vi the hien tai)={total_open_size - total_close_size:.8f}")
print(f"BTC: Tong open notional={btc_open['ntl'].sum():.2f}")

# Phan tich entry fragmentation
print("\n--- PHAN TICH ENTRY FRAGMENTATION ---")
btc_open_prices = btc_open["px"].unique()
print(f"So gia entry khac nhau: {len(btc_open_prices)}")
for px in btc_open_prices:
    fills_at_px = btc_open[btc_open["px"] == px]
    print(f"  px={px}: {len(fills_at_px)} fills, tong sz={fills_at_px['sz'].sum():.8f}, tong ntl={fills_at_px['ntl'].sum():.2f}")

# Phan tich thoi gian giua cac fills
print("\n--- PHAN TICH THOI GIAN GIUA CAC FILLS ---")
btc["time_parsed"] = pd.to_datetime(btc["time"], format="%H:%M:%S %d/%m/%Y")
btc_sorted = btc.sort_values("time_parsed")
for i in range(1, len(btc_sorted)):
    delta = (btc_sorted.iloc[i]["time_parsed"] - btc_sorted.iloc[i - 1]["time_parsed"]).total_seconds()
    print(f"  {btc_sorted.iloc[i-1]['dir']} -> {btc_sorted.iloc[i]['dir']}: {delta:.0f}s ({delta/60:.1f}min)")

print()
print("=" * 80)
print("PHAN TICH BACKTEST RESULTS")
print("=" * 80)

bt = pd.read_csv(r"d:\backtest\results\BTCUSDT_trades.csv")
print(f"\nTong trades: {len(bt)}")
print(f"Win rate: {bt['win'].sum()}/{len(bt)} = {bt['win'].mean()*100:.1f}%")
print(f"Net PnL: ${bt['net_pnl'].sum():.4f}")
print(f"Tong fees: ${bt['fees'].sum():.4f}")
print(f"Avg net_pnl per trade: ${bt['net_pnl'].mean():.6f}")

# Profit factor
wins_pnl = bt[bt["win"]]["net_pnl"].sum()
losses_pnl = -bt[~bt["win"]]["net_pnl"].sum()
pf = wins_pnl / losses_pnl if losses_pnl > 0 else float("inf")
print(f"Profit factor: {pf:.4f}")

print(f"\nExit reasons:")
print(bt["exit_reason"].value_counts().to_string())

print(f"\nMargin distribution:")
print(bt["margin"].value_counts().to_string())

# Win rate by exit reason
print(f"\nWin rate by exit reason:")
for reason in bt["exit_reason"].unique():
    subset = bt[bt["exit_reason"] == reason]
    wr = subset["win"].mean() * 100
    avg = subset["net_pnl"].mean()
    print(f"  {reason}: {subset['win'].sum()}/{len(subset)} = {wr:.1f}%, avg_pnl=${avg:.6f}")

# Time-stop analysis
ts = bt[bt["exit_reason"] == "time_stop"]
print(f"\nTime-stop trades analysis:")
print(f"  Count: {len(ts)}/{len(bt)} ({len(ts)/len(bt)*100:.1f}%)")
print(f"  Win: {ts['win'].sum()}/{len(ts)} ({ts['win'].mean()*100:.1f}%)")
print(f"  Avg PnL: ${ts['net_pnl'].mean():.6f}")
print(f"  Net PnL total: ${ts['net_pnl'].sum():.4f}")
print(f"  Avg gross_pnl: ${ts['gross_pnl'].mean():.6f}")
print(f"  Avg fees: ${ts['fees'].mean():.6f}")

ts_win = ts[ts["win"]]
ts_lose = ts[~ts["win"]]
print(f"  Winning time-stops: {len(ts_win)}, avg_net_pnl=${ts_win['net_pnl'].mean():.6f}")
print(f"  Losing time-stops: {len(ts_lose)}, avg_net_pnl=${ts_lose['net_pnl'].mean():.6f}")

# Score distribution
print(f"\nScore distribution and win rate:")
for lo, hi in [(65, 69), (70, 74), (75, 79), (80, 84), (85, 100)]:
    subset = bt[(bt["score"] >= lo) & (bt["score"] <= hi)]
    if len(subset) > 0:
        wr = subset["win"].mean() * 100
        avg = subset["net_pnl"].mean()
        print(f"  Score {lo}-{hi}: {len(subset)} trades, win={wr:.1f}%, avg_pnl=${avg:.6f}")

# Fee impact
print(f"\n--- FEE IMPACT ANALYSIS ---")
print(f"Total gross PnL: ${bt['gross_pnl'].sum():.4f}")
print(f"Total fees: ${bt['fees'].sum():.4f}")
print(f"Total net PnL: ${bt['net_pnl'].sum():.4f}")
print(f"Fees as % of |gross|: {bt['fees'].sum() / abs(bt['gross_pnl'].sum()) * 100:.1f}%")
print(f"Trades that would be profitable without fees but lose with fees:")
marginal = bt[(bt["gross_pnl"] > 0) & (bt["net_pnl"] < 0)]
print(f"  Count: {len(marginal)} ({len(marginal)/len(bt)*100:.1f}% of all trades)")
print(f"  Avg gross_pnl: ${marginal['gross_pnl'].mean():.6f}")
print(f"  Avg fees: ${marginal['fees'].mean():.6f}")

# Direction analysis
print(f"\n--- DIRECTION ANALYSIS ---")
for d in ["long", "short"]:
    subset = bt[bt["direction"] == d]
    if len(subset) > 0:
        wr = subset["win"].mean() * 100
        avg = subset["net_pnl"].mean()
        tot = subset["net_pnl"].sum()
        print(f"  {d}: {len(subset)} trades, win={wr:.1f}%, avg_pnl=${avg:.6f}, total=${tot:.4f}")
