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
from src.trading_bot.risk_manager import ScalpingRiskManager


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
    entry_model_counts: Dict[str, int] = {"MARKET": 0, "STOP": 0}
    sl_pips_list: List[float] = []
    tp_pips_list: List[float] = []
    rejection_reasons: Dict[str, int] = {}
    non_fresh_zones = 0
    strong_trend_cycles = 0

    risk_manager = ScalpingRiskManager()

    start_idx = min(len(df) - 1, 60)
    for i in range(start_idx, len(df)):
        window = df.iloc[: i + 1].copy()
        analyzer = GoldNDSAnalyzer(window, config=None)
        result = analyzer.generate_trading_signal(timeframe="M15", scalping_mode=True)

        entry_type = "NONE"
        entry_model = "MARKET"
        entry_level = None
        entry_context = {}
        signal_context = {}
        flow_zones = {}

        if result.context:
            entry_type = result.context.get("entry_type", "NONE") or "NONE"
            entry_model = result.context.get("entry_model", "MARKET") or "MARKET"
            entry_idea = result.context.get("entry_idea", {}) or {}
            entry_level = entry_idea.get("entry_level") or result.entry_price
            entry_context = result.context.get("entry_context", {}) or {}
            signal_context = result.context.get("signal_context", {}) or {}
            flow_zones = result.context.get("flow_zones", {}) or {}

        if signal_context.get("strong_trend"):
            strong_trend_cycles += 1

        entry_counts.setdefault(entry_type, 0)
        entry_counts[entry_type] += 1
        entry_model_counts.setdefault(entry_model, 0)
        entry_model_counts[entry_model] += 1

        for zone in (flow_zones.get("breakers") or []) + (flow_zones.get("inversion_fvgs") or []):
            if zone and not zone.get("fresh", True):
                non_fresh_zones += 1

        if result.signal in {"BUY", "SELL"} and entry_level:
            atr_value = None
            if result.context:
                market_metrics = result.context.get("market_metrics", {}) or {}
                atr_value = market_metrics.get("atr_short") or market_metrics.get("atr")
            sltp = risk_manager._compute_scalping_sl_tp(
                signal=result.signal,
                entry_price=float(entry_level),
                atr_value=atr_value,
                recent_low=entry_context.get("recent_low"),
                recent_high=entry_context.get("recent_high"),
                config_payload=risk_manager.settings,
            )
            sl_pips = _distance_pips(abs(float(entry_level) - float(sltp["stop_loss"])), 0.001)
            tp_pips = _distance_pips(abs(float(sltp["take_profit"]) - float(entry_level)), 0.001)
            sl_pips_list.append(sl_pips)
            tp_pips_list.append(tp_pips)
        elif result.signal == "NONE" and result.reasons:
            reason_key = result.reasons[-1]
            rejection_reasons[reason_key] = rejection_reasons.get(reason_key, 0) + 1

    sl_avg = sum(sl_pips_list) / len(sl_pips_list) if sl_pips_list else 0.0
    tp_avg = sum(tp_pips_list) / len(tp_pips_list) if tp_pips_list else 0.0

    print("Entry type counts:")
    for key in ["BREAKER", "IFVG", "MOMENTUM", "NONE"]:
        print(f"  {key}: {entry_counts.get(key, 0)}")
    print("Entry model counts:")
    for key in ["MARKET", "STOP"]:
        print(f"  {key}: {entry_model_counts.get(key, 0)}")
    print(f"Non-fresh zones observed: {non_fresh_zones}")
    print(f"Average SL pips: {sl_avg:.2f}")
    print(f"Average TP pips: {tp_avg:.2f}")
    if rejection_reasons:
        print("Rejected trades by reason:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")

    failed = False
    diagnostics = []
    total_trades = sum(entry_counts.values()) or 1
    tier_c_ratio = entry_counts.get("MOMENTUM", 0) / total_trades
    print(f"Tier ratios: A={entry_counts.get('BREAKER', 0)/total_trades:.2%} "
          f"B={entry_counts.get('IFVG', 0)/total_trades:.2%} "
          f"C={tier_c_ratio:.2%}")
    if sl_pips_list and max(sl_pips_list) > 60:
        failed = True
        diagnostics.append(f"SL pips exceeded 60 (max={max(sl_pips_list):.2f})")
    if strong_trend_cycles > 0 and entry_counts.get("MOMENTUM", 0) == 0:
        failed = True
        diagnostics.append("Tier C momentum trades missing during strong trend cycles.")
    if entry_model_counts.get("MARKET", 0) == 0 or entry_model_counts.get("STOP", 0) == 0:
        failed = True
        diagnostics.append("Entry model is not diversified (missing MARKET or STOP).")
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
