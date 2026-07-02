"""
plot_results.py
================
Plot equity curve and PnL distribution from the trade log exported in results/.
Run after executing run_backtest.py.

Usage:
    python plot_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def plot_symbol(symbol: str):
    path = os.path.join(RESULTS_DIR, f"{symbol}_trades.csv")
    if not os.path.exists(path):
        print(f"Skipping {symbol}: file {path} not found")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"Skipping {symbol}: no trades found.")
        return

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["equity"] = 100.0 + df["net_pnl"].cumsum()
    running_max = df["equity"].cummax()
    df["drawdown_pct"] = (df["equity"] - running_max) / running_max * 100

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    axes[0].plot(df["entry_time"], df["equity"], color="#2563eb", linewidth=1.5)
    axes[0].axhline(100, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"{symbol} — Equity Curve (starting balance $100)")
    axes[0].set_ylabel("Equity (USD)")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(df["entry_time"], df["drawdown_pct"], 0, color="#dc2626", alpha=0.4)
    axes[1].set_title(f"{symbol} — Drawdown (%)")
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(alpha=0.3)

    colors = ["#16a34a" if w else "#dc2626" for w in df["win"]]
    axes[2].bar(range(len(df)), df["net_pnl"], color=colors)
    axes[2].set_title(f"{symbol} — Net PnL per Trade (green=win, red=loss)")
    axes[2].set_ylabel("Net PnL (USD)")
    axes[2].set_xlabel("Trade Number")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{symbol}_chart.png")
    plt.savefig(out_path, dpi=130)
    print(f"Saved chart: {out_path}")
    plt.close()


def main():
    for symbol in SYMBOLS:
        plot_symbol(symbol)


if __name__ == "__main__":
    main()