from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .metrics import Diagnostics


@dataclass
class ObjectiveConfig:
    name: str
    lambda_dd: float = 0.0
    lambda_pf: float = 0.0
    lambda_ab: float = 0.0
    lambda_overtrade: float = 0.0


OBJECTIVE_PRESETS = {
    "net_pnl": ObjectiveConfig(name="net_pnl"),
    "balanced": ObjectiveConfig(name="balanced", lambda_dd=0.5, lambda_pf=50.0, lambda_ab=100.0, lambda_overtrade=10.0),
    "pf_focus": ObjectiveConfig(name="pf_focus", lambda_dd=0.8, lambda_pf=80.0, lambda_ab=50.0, lambda_overtrade=5.0),
}


def objective_from_payload(payload: Dict[str, Any]) -> ObjectiveConfig:
    return ObjectiveConfig(
        name=payload.get("name", "custom"),
        lambda_dd=float(payload.get("lambda_dd", 0.0)),
        lambda_pf=float(payload.get("lambda_pf", 0.0)),
        lambda_ab=float(payload.get("lambda_ab", 0.0)),
        lambda_overtrade=float(payload.get("lambda_overtrade", 0.0)),
    )


def get_objective_config(name: str) -> ObjectiveConfig:
    if name in OBJECTIVE_PRESETS:
        return OBJECTIVE_PRESETS[name]
    return ObjectiveConfig(name=name)


def score_objective(metrics: Dict[str, Any], diagnostics: Diagnostics, config: ObjectiveConfig) -> float:
    net_pnl = float(metrics.get("net_pnl", 0.0))
    max_dd = abs(float(metrics.get("max_drawdown_pct", 0.0)))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    ab_ratio = float(diagnostics.tier_ratios.get("A+B", 0.0))
    trades_per_day = float(metrics.get("trades_per_day", 0.0))

    score = net_pnl
    score -= config.lambda_dd * max_dd
    score += config.lambda_pf * profit_factor
    score += config.lambda_ab * ab_ratio
    score -= config.lambda_overtrade * trades_per_day
    return score
