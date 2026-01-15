from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from .metrics import Diagnostics, summarize_cycle_log


@dataclass
class ParityReport:
    backtest: Diagnostics
    live: Diagnostics
    deltas: Dict[str, Dict[str, float]]


def _ratio_delta(back: Dict[str, float], live: Dict[str, float]) -> Dict[str, float]:
    keys = set(back.keys()) | set(live.keys())
    return {key: float(back.get(key, 0.0)) - float(live.get(key, 0.0)) for key in keys}


def compare_cycle_logs(backtest_log: pd.DataFrame, live_log: pd.DataFrame) -> ParityReport:
    back_diag = summarize_cycle_log(backtest_log)
    live_diag = summarize_cycle_log(live_log)

    deltas = {
        "tier_ratios": _ratio_delta(back_diag.tier_ratios, live_diag.tier_ratios),
        "entry_model_ratios": _ratio_delta(back_diag.entry_model_ratios, live_diag.entry_model_ratios),
    }
    return ParityReport(backtest=back_diag, live=live_diag, deltas=deltas)


def format_parity_report(report: ParityReport) -> str:
    lines = []
    lines.append("Live Parity Comparison")
    lines.append("=" * 32)

    lines.append("\nTier Ratio Delta (backtest - live)")
    for key, value in report.deltas.get("tier_ratios", {}).items():
        lines.append(f"{key}: {value:.3f}")

    lines.append("\nEntry Model Ratio Delta (backtest - live)")
    for key, value in report.deltas.get("entry_model_ratios", {}).items():
        lines.append(f"{key}: {value:.3f}")

    lines.append("\nZone Rejections (backtest)")
    for key, value in report.backtest.zone_rejections.items():
        lines.append(f"{key}: {value}")

    lines.append("\nZone Rejections (live)")
    for key, value in report.live.zone_rejections.items():
        lines.append(f"{key}: {value}")

    lines.append("\nRetest Rejections (backtest)")
    for key, value in report.backtest.retest_rejections.items():
        lines.append(f"{key}: {value}")

    lines.append("\nRetest Rejections (live)")
    for key, value in report.live.retest_rejections.items():
        lines.append(f"{key}: {value}")

    return "\n".join(lines)
