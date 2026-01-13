#!/usr/bin/env python3
"""
Smoke test for NDS Flow Scalper architecture.

Usage:
  python scripts/smoke_test_flow_scalper.py --csv path/to/file.csv --limit 2000
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import pandas as pd

from src.trading_bot.nds.analyzer import GoldNDSAnalyzer
from src.trading_bot.nds.distance_utils import price_to_points, points_to_pips


def _distance_pips(price_distance: float, point_size: float) -> float:
    return points_to_pips(price_to_points(price_distance, point_size))


def run(csv_path: str, limit: int | None) -> int:
    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty.")
        return 1

    if limit:
        df = df.tail(limit).reset_index(drop=True)

    entry_counts: Dict[str, int] = {"BREAKER": 0, "IFVG": 0, "MOMENTUM": 0, "NONE": 0}
    sl_pips_list: List[float] = []
    tp_pips_list: List[float] = []

    start_idx = min(len(df) - 1, 60)
    for i in range(start_idx, len(df)):
        window = df.iloc[: i + 1].copy()
        analyzer = GoldNDSAnalyzer(window, config=None)
        result = analyzer.generate_trading_signal(timeframe="M15", scalping_mode=True)

        entry_type = "NONE"
        if result.context:
            entry_type = result.context.get("entry_type", "NONE") or "NONE"
        entry_counts.setdefault(entry_type, 0)
        entry_counts[entry_type] += 1

        if result.signal in {"BUY", "SELL"} and result.entry_price and result.stop_loss and result.take_profit:
            point_size = 0.001
            sl_pips = _distance_pips(abs(result.entry_price - result.stop_loss), point_size)
            tp_pips = _distance_pips(abs(result.take_profit - result.entry_price), point_size)
            sl_pips_list.append(sl_pips)
            tp_pips_list.append(tp_pips)

    sl_avg = sum(sl_pips_list) / len(sl_pips_list) if sl_pips_list else 0.0
    tp_avg = sum(tp_pips_list) / len(tp_pips_list) if tp_pips_list else 0.0

    print("Entry type counts:")
    for key in ["BREAKER", "IFVG", "MOMENTUM", "NONE"]:
        print(f"  {key}: {entry_counts.get(key, 0)}")
    print(f"Average SL pips: {sl_avg:.2f}")
    print(f"Average TP pips: {tp_avg:.2f}")

    failed = False
    diagnostics = []
    if sl_pips_list and max(sl_pips_list) > 60:
        failed = True
        diagnostics.append(f"SL pips exceeded 60 (max={max(sl_pips_list):.2f})")
    if tp_pips_list and max(tp_pips_list) > 100:
        failed = True
        diagnostics.append(f"TP pips exceeded 100 (max={max(tp_pips_list):.2f})")

    if failed:
        print("Sanity check failures:")
        for line in diagnostics:
            print(f"  - {line}")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test NDS Flow Scalper.")
    parser.add_argument("--csv", required=True, help="Path to CSV file with OHLCV data.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    args = parser.parse_args()

    exit_code = run(args.csv, args.limit)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
