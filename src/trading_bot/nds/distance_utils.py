"""
Centralized distance and pip/point conversion utilities for NDS Flow Scalper.

Assumptions (project-wide, XAUUSD in this project):
    point_size = 0.01  (based on broker quote precision and historical CSV samples)
    1 point = point_size in price units
    pips are a "project-defined" unit: 100 points = 10 pips  => 1 pip = 10 points

Notes:
- Do NOT hardcode point_size in multiple modules. Always resolve it through this file.
- If your broker symbol precision changes (e.g., 0.001), update POINT_SIZE in config,
  or use infer_point_size_from_prices(...) for sanity checks.
"""
from __future__ import annotations

from typing import Dict, Optional, Iterable, Any

DEFAULT_POINT_SIZE = 0.01


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


def resolve_point_size_from_config(
    config_payload: Optional[Dict[str, Any]],
    default: Optional[float] = None,
) -> float:
    """
    Resolve point size from a typical project config payload.

    Supported keys (checked in this order):
      trading_settings.GOLD_SPECIFICATIONS.point / POINT / point_size / POINT_SIZE
      trading_settings.POINT_SIZE / POINT / point
      top-level POINT_SIZE / POINT / point

    Returns a safe float > 0, otherwise falls back to DEFAULT_POINT_SIZE (or `default` if provided).
    """
    fallback = _safe_point_size(default)

    if not isinstance(config_payload, dict):
        return fallback

    trading_settings = config_payload.get("trading_settings", {})
    if not isinstance(trading_settings, dict):
        trading_settings = {}

    gold_specs = trading_settings.get("GOLD_SPECIFICATIONS", {})
    if not isinstance(gold_specs, dict):
        gold_specs = {}

    candidates = [
        gold_specs.get("POINT_SIZE"),
        gold_specs.get("point_size"),
        gold_specs.get("point"),
        gold_specs.get("POINT"),
        trading_settings.get("POINT_SIZE"),
        trading_settings.get("point_size"),
        trading_settings.get("POINT"),
        trading_settings.get("point"),
        config_payload.get("POINT_SIZE"),
        config_payload.get("point_size"),
        config_payload.get("POINT"),
        config_payload.get("point"),
    ]

    for v in candidates:
        try:
            if v is None:
                continue
            vv = float(v)
            if vv > 0:
                return vv
        except (TypeError, ValueError):
            continue

    return fallback


def infer_point_size_from_prices(prices: Iterable[float], default: Optional[float] = None) -> float:
    """
    Infer point size from a sequence of prices by inspecting the smallest positive increment.

    This is intended as a sanity-check helper (e.g., in smoke tests), not as a trading decision input.

    - Ignores zero and negative diffs.
    - Uses a conservative fallback if cannot infer.

    Typical expected outputs:
      0.01  for two-decimal XAUUSD datasets
      0.001 for three-decimal datasets
    """
    fallback = _safe_point_size(default)

    # Collect diffs
    last = None
    diffs = []
    for p in prices:
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if last is not None:
            d = abs(pf - last)
            if d > 0:
                diffs.append(d)
        last = pf

    if not diffs:
        return fallback

    min_diff = min(diffs)

    # Snap to common quote precisions to avoid weird floating-point residue
    common = [0.1, 0.01, 0.001, 0.0001]
    for c in common:
        if abs(min_diff - c) <= (c * 0.25):
            return c

    # Otherwise return the raw min_diff but still safe-guarded
    return _safe_point_size(min_diff)
