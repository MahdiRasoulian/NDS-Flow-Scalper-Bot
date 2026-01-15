from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable

import pandas as pd


@dataclass
class Diagnostics:
    entry_type_counts: Dict[str, int]
    tier_counts: Dict[str, int]
    entry_model_counts: Dict[str, int]
    zone_rejections: Dict[str, int]
    retest_rejections: Dict[str, int]
    tier_ratios: Dict[str, float]
    entry_model_ratios: Dict[str, float]


def _safe_count(series: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in series:
        key = str(value) if value is not None else "NONE"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _merge_reason_counts(existing: Dict[str, int], payload: Any) -> Dict[str, int]:
    if not payload:
        return existing
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return existing
        else:
            return existing
    if isinstance(payload, dict):
        for key, value in payload.items():
            try:
                count_val = int(value)
            except (TypeError, ValueError):
                count_val = 0
            existing[key] = existing.get(key, 0) + count_val
    return existing


def summarize_cycle_log(cycle_log: pd.DataFrame) -> Diagnostics:
    if cycle_log is None or cycle_log.empty:
        return Diagnostics(
            entry_type_counts={},
            tier_counts={},
            entry_model_counts={},
            zone_rejections={},
            retest_rejections={},
            tier_ratios={},
            entry_model_ratios={},
        )

    entry_type_counts = _safe_count(cycle_log.get("entry_type", []))
    tier_counts = _safe_count(cycle_log.get("tier", []))
    entry_model_counts = _safe_count(cycle_log.get("entry_model", []))

    zone_rejections: Dict[str, int] = {}
    retest_rejections: Dict[str, int] = {}

    if "zone_rejections" in cycle_log.columns:
        for payload in cycle_log["zone_rejections"]:
            zone_rejections = _merge_reason_counts(zone_rejections, payload)
    if "retest_rejections" in cycle_log.columns:
        for payload in cycle_log["retest_rejections"]:
            retest_rejections = _merge_reason_counts(retest_rejections, payload)

    total_tiered = sum(tier_counts.get(tier, 0) for tier in ("A", "B", "C", "D"))
    total_entries = sum(entry_type_counts.values())
    total_entry_models = sum(entry_model_counts.values())

    tier_ratios = {
        "A": tier_counts.get("A", 0) / total_tiered if total_tiered else 0.0,
        "B": tier_counts.get("B", 0) / total_tiered if total_tiered else 0.0,
        "C": tier_counts.get("C", 0) / total_tiered if total_tiered else 0.0,
        "D": tier_counts.get("D", 0) / total_tiered if total_tiered else 0.0,
        "A+B": (
            (tier_counts.get("A", 0) + tier_counts.get("B", 0)) / total_tiered
            if total_tiered
            else 0.0
        ),
    }

    entry_model_ratios = {
        "MARKET": entry_model_counts.get("MARKET", 0) / total_entry_models if total_entry_models else 0.0,
        "STOP": entry_model_counts.get("STOP", 0) / total_entry_models if total_entry_models else 0.0,
        "LIMIT": entry_model_counts.get("LIMIT", 0) / total_entry_models if total_entry_models else 0.0,
    }

    return Diagnostics(
        entry_type_counts=entry_type_counts,
        tier_counts=tier_counts,
        entry_model_counts=entry_model_counts,
        zone_rejections=zone_rejections,
        retest_rejections=retest_rejections,
        tier_ratios=tier_ratios,
        entry_model_ratios=entry_model_ratios,
    )


def compute_trade_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    starting_equity: float,
    bars_per_day: int,
) -> Dict[str, Any]:
    net_pnl = float(trades["pnl_usd"].sum()) if trades is not None and not trades.empty else 0.0
    gross_profit = float(trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].sum()) if trades is not None and not trades.empty else 0.0
    gross_loss = float(-trades.loc[trades["pnl_usd"] < 0, "pnl_usd"].sum()) if trades is not None and not trades.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    wins = int((trades["pnl_usd"] > 0).sum()) if trades is not None and not trades.empty else 0
    losses = int((trades["pnl_usd"] < 0).sum()) if trades is not None and not trades.empty else 0
    total_trades = int(len(trades)) if trades is not None else 0
    win_rate = wins / total_trades if total_trades else 0.0

    avg_win = float(trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].mean()) if wins else 0.0
    avg_loss = float(trades.loc[trades["pnl_usd"] < 0, "pnl_usd"].mean()) if losses else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    max_dd = 0.0
    if equity_curve is not None and not equity_curve.empty:
        eq = equity_curve["equity"].astype(float)
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max.replace(0, pd.NA)
        max_dd = float(dd.min() * 100.0) if not dd.empty else 0.0

    trades_per_day = 0.0
    if trades is not None and not trades.empty:
        time_col = trades["close_time"] if "close_time" in trades.columns else trades.get("open_time")
        if time_col is not None:
            dates = pd.to_datetime(time_col).dt.date
            unique_days = len(set(dates))
            trades_per_day = total_trades / unique_days if unique_days else 0.0

    avg_trade_duration = float(trades["duration_bars"].mean()) if trades is not None and not trades.empty else 0.0

    return {
        "starting_equity": starting_equity,
        "ending_equity": float(equity_curve["equity"].iloc[-1]) if equity_curve is not None and not equity_curve.empty else starting_equity,
        "net_pnl": net_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "total_trades": total_trades,
        "expectancy": expectancy,
        "max_drawdown_pct": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "trades_per_day": trades_per_day,
        "avg_trade_duration_bars": avg_trade_duration,
        "bars_per_day": bars_per_day,
    }


def format_summary_text(metrics: Dict[str, Any], diagnostics: Diagnostics) -> str:
    lines = []
    lines.append("NDS Flow Scalper Backtest Summary")
    lines.append("=" * 44)
    lines.append("")
    lines.append("Performance")
    lines.append("-" * 20)
    for key in (
        "starting_equity",
        "ending_equity",
        "net_pnl",
        "profit_factor",
        "win_rate",
        "max_drawdown_pct",
        "expectancy",
        "total_trades",
        "trades_per_day",
    ):
        value = metrics.get(key)
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Entry Types")
    lines.append("-" * 20)
    for key, value in diagnostics.entry_type_counts.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Tier Counts")
    lines.append("-" * 20)
    for key, value in diagnostics.tier_counts.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Entry Models")
    lines.append("-" * 20)
    for key, value in diagnostics.entry_model_counts.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Zone Rejections")
    lines.append("-" * 20)
    for key, value in diagnostics.zone_rejections.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Retest Rejections")
    lines.append("-" * 20)
    for key, value in diagnostics.retest_rejections.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Tier Ratios")
    lines.append("-" * 20)
    for key, value in diagnostics.tier_ratios.items():
        lines.append(f"{key}: {value:.3f}")

    lines.append("")
    lines.append("Entry Model Ratios")
    lines.append("-" * 20)
    for key, value in diagnostics.entry_model_ratios.items():
        lines.append(f"{key}: {value:.3f}")

    return "\n".join(lines)
