
from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt


def plot_equity_curve(equity: pd.DataFrame, title: str = "Equity Curve", out_path: Optional[str] = None):
    if equity is None or equity.empty:
        raise ValueError("Equity curve is empty.")
    fig = plt.figure()
    plt.plot(equity.index, equity["equity"])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")
    plt.xticks(rotation=25)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_drawdown(equity: pd.DataFrame, title: str = "Drawdown", out_path: Optional[str] = None):
    if equity is None or equity.empty:
        raise ValueError("Equity curve is empty.")
    eq = equity["equity"].astype(float)
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max.replace(0, pd.NA)
    fig = plt.figure()
    plt.plot(equity.index, dd * 100.0)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Drawdown (%)")
    plt.xticks(rotation=25)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_trade_pnl_hist(trades: pd.DataFrame, title: str = "Trade PnL Distribution", out_path: Optional[str] = None):
    if trades is None or trades.empty:
        raise ValueError("Trades dataframe is empty.")
    fig = plt.figure()
    plt.hist(trades["pnl_usd"].astype(float), bins=40)
    plt.title(title)
    plt.xlabel("PnL (USD)")
    plt.ylabel("Count")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_signal_diagnostics(cycle_log: pd.DataFrame, title: str = "Score & Confidence", out_path: Optional[str] = None):
    if cycle_log is None or cycle_log.empty:
        raise ValueError("cycle_log is empty.")
    fig = plt.figure()
    plt.plot(cycle_log.index, cycle_log["score"].astype(float), label="score")
    plt.plot(cycle_log.index, cycle_log["confidence"].astype(float), label="confidence")
    plt.axhline(50, linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig
