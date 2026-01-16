#!/usr/bin/env python3
"""
Quick sanity check for spread normalization and pip/point conventions.
"""
from __future__ import annotations

import math

import sys
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISTANCE_UTILS = PROJECT_ROOT / "src" / "trading_bot" / "nds" / "distance_utils.py"

spec = spec_from_file_location("distance_utils", DISTANCE_UTILS)
if spec is None or spec.loader is None:
    raise ImportError("Unable to load distance_utils module")
distance_utils = module_from_spec(spec)
sys.modules["distance_utils"] = distance_utils
spec.loader.exec_module(distance_utils)

normalize_spread = distance_utils.normalize_spread


def _is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def main() -> int:
    bid = 4600.62
    ask = 4600.81
    spread_price = ask - bid  # 0.19
    point_size = 0.01
    pip_size = 0.1  # 10 points

    normalized = normalize_spread(spread_price, point_size=point_size, pip_size=pip_size)
    spread_points = float(normalized.get("spread_points") or 0.0)
    spread_pips = float(normalized.get("spread_pips") or 0.0)

    normalized_points = normalize_spread(19, point_size=point_size, pip_size=pip_size, raw_unit="points")
    legacy_price = float(normalized_points.get("spread_price") or 0.0)
    legacy_pips = float(normalized_points.get("spread_pips") or 0.0)

    if not _is_close(spread_price, 0.19):
        raise AssertionError(f"Expected spread_price 0.19, got {spread_price}")
    if not _is_close(spread_points, 19.0):
        raise AssertionError(f"Expected spread_points 19.0, got {spread_points}")
    if not _is_close(spread_pips, 1.9):
        raise AssertionError(f"Expected spread_pips 1.9, got {spread_pips}")
    if not _is_close(legacy_price, 0.19):
        raise AssertionError(f"Expected legacy spread_price 0.19, got {legacy_price}")
    if not _is_close(legacy_pips, 1.9):
        raise AssertionError(f"Expected legacy spread_pips 1.9, got {legacy_pips}")

    max_spread_pips = 2.5
    if spread_pips > max_spread_pips:
        raise AssertionError(
            f"Spread should be tradable for max {max_spread_pips} pips; got {spread_pips}"
        )

    print(
        "✅ Spread normalization OK:",
        f"spread_price={spread_price:.2f}",
        f"spread_points={spread_points:.1f}",
        f"spread_pips={spread_pips:.1f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
