"""
Centralized distance and pip/point conversion utilities for NDS Flow Scalper.

Assumptions (project-wide):
    point_size = 0.001
    100 points = 10 pips = 1 USD (for this setup)
"""
from __future__ import annotations

from typing import Dict, Optional

DEFAULT_POINT_SIZE = 0.001


def _safe_point_size(point_size: Optional[float]) -> float:
    """Return a valid point size with a conservative default."""
    try:
        value = float(point_size) if point_size is not None else DEFAULT_POINT_SIZE
    except (TypeError, ValueError):
        value = DEFAULT_POINT_SIZE
    if value <= 0:
        return DEFAULT_POINT_SIZE
    return value


def price_to_points(price_distance: float, point_size: Optional[float] = None) -> float:
    """Convert price distance to points."""
    pt = _safe_point_size(point_size)
    return float(price_distance) / pt


def points_to_pips(points: float) -> float:
    """Convert points to pips (100 points = 10 pips)."""
    return float(points) / 100.0


def pips_to_points(pips: float) -> float:
    """Convert pips to points."""
    return float(pips) * 100.0


def pips_to_price(pips: float, point_size: Optional[float] = None) -> float:
    """Convert pips to price distance."""
    pt = _safe_point_size(point_size)
    return pips_to_points(pips) * pt


def calculate_distance_metrics(
    entry_price: float,
    current_price: float,
    point_size: Optional[float] = None,
    atr_value: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Calculate standardized distance metrics used across analyzer and risk manager."""
    metrics: Dict[str, Optional[float]] = {
        "dist_price": None,
        "point_size": None,
        "dist_points": None,
        "dist_pips": None,
        "dist_usd": None,
        "dist_atr": None,
    }

    try:
        entry_f = float(entry_price)
        price_f = float(current_price)
    except (TypeError, ValueError):
        return metrics

    pt = _safe_point_size(point_size)
    dist_price = abs(entry_f - price_f)
    dist_points = price_to_points(dist_price, pt)
    dist_pips = points_to_pips(dist_points)

    dist_atr = None
    if atr_value is not None:
        try:
            atr_val = float(atr_value)
            if atr_val > 0:
                dist_atr = dist_price / atr_val
        except (TypeError, ValueError):
            dist_atr = None

    metrics.update(
        {
            "dist_price": dist_price,
            "point_size": pt,
            "dist_points": dist_points,
            "dist_pips": dist_pips,
            "dist_usd": dist_price,
            "dist_atr": dist_atr,
        }
    )
    return metrics
