#!/usr/bin/env python3
"""
Smoke test for NDS Flow Scalper architecture.

Usage:
  python scripts/smoke_test_flow_scalper.py --csv path/to/file.csv --limit 2000
Recommended:
  python scripts/smoke_test_flow_scalper.py --csv ... --limit 3000 --window 800 --step 2 --progress 50
"""
from __future__ import annotations

import argparse
import sys
import time
import logging
from typing import Dict, List
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../NDS-Flow-Scalper-Bot
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trading_bot.nds.analyzer import GoldNDSAnalyzer
from src.trading_bot.nds.distance_utils import (
    calculate_distance_metrics,
    infer_point_size_from_prices,
    resolve_point_size_from_config,
)
from src.trading_bot.risk_manager import ScalpingRiskManager
from src.trading_bot.config_utils import log_active_settings
from config.settings import config


def _distance_pips(price_distance: float, point_size: float) -> float:
    metrics = calculate_distance_metrics(
        entry_price=0.0,
        current_price=price_distance,
        point_size=point_size,
    )
    return float(metrics.get("dist_pips") or 0.0)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CSV columns so analyzer expects 'open/high/low/close/time/volume/spread' consistently.
    Does NOT drop anything; only adds/fills standard names if possible.
    """
    df = df.copy()
    # strip column names
    df.columns = [str(c).strip() for c in df.columns]

    # common aliases mapping
    alias = {
        "Open": "open", "OPEN": "open",
        "High": "high", "HIGH": "high",
        "Low": "low", "LOW": "low",
        "Close": "close", "CLOSE": "close",
        "Time": "time", "TIME": "time",
        "Datetime": "time", "datetime": "time", "Date": "time", "date": "time",
        "Volume": "volume", "VOL": "volume", "tick_volume": "volume",
        "Spread": "spread", "SPREAD": "spread",
    }
    for src, dst in alias.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # ensure required OHLC exist
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in CSV. Available: {list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # parse time once (huge speedup versus repeated parsing in loop)
    if "time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # volume/spread optional
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")

    # forward-fill numeric gaps
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()

    return df


def run(csv_path: str, limit: int | None, window: int, step: int, progress: int) -> int:
    logging.basicConfig(level=logging.INFO)
    log_active_settings(config, logging.getLogger(__name__))
    t0 = time.time()

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty.")
        return 1

    df = _normalize_columns(df)

    if limit:
        df = df.tail(limit).reset_index(drop=True)

    if len(df) < max(100, window):
        print(f"Not enough rows after limit. rows={len(df)} window={window}")
        return 1

    entry_counts: Dict[str, int] = {"BREAKER": 0, "IFVG": 0, "MOMENTUM": 0, "LEGACY": 0, "NONE": 0}
    tier_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "NONE": 0}
    entry_model_counts: Dict[str, int] = {"MARKET": 0, "STOP": 0}
    sl_pips_list: List[float] = []
    tp_pips_list: List[float] = []
    rejection_reasons: Dict[str, int] = {}
    zone_rejections: Dict[str, int] = {"too_far": 0, "too_old": 0, "too_many_touches": 0, "stale": 0}
    retest_rejections: Dict[str, int] = {
        "TOO_MANY_TOUCHES": 0,
        "FIRST_TOUCH_UNCONFIRMED": 0,
        "NO_CONFIRMED_TOUCH": 0,
    }
    momentum_rejections: Dict[str, int] = {"time_outside_window": 0, "time_parse_failed": 0}
    spread_rejections: Dict[str, int] = {"spread_too_high": 0}
    sl_pips_by_tier: Dict[str, List[float]] = {}
    tp_pips_by_tier: Dict[str, List[float]] = {}
    non_fresh_zones = 0
    strong_trend_cycles = 0
    ifvg_zone_count = 0

    risk_manager = ScalpingRiskManager()
    config_payload = config.get_full_config()
    inferred_point_size = infer_point_size_from_prices(df["close"].tail(window))
    resolved_point_size = resolve_point_size_from_config(risk_manager.settings, default=None)
    print(
        f"[SMOKE] inferred_point_size={inferred_point_size:.4f} "
        f"resolved_point_size={resolved_point_size:.4f}"
    )

    # Use rolling fixed window to avoid O(n^2)
    start_idx = window - 1
    total_iters = ((len(df) - start_idx - 1) // step) + 1

    print(f"[SMOKE] rows={len(df)} window={window} step={step} iters~={total_iters}")
    print("[SMOKE] running... (progress prints enabled)")

    iter_no = 0
    for i in range(start_idx, len(df), step):
        iter_no += 1
        if progress and (iter_no % progress == 0):
            elapsed = time.time() - t0
            last_price = float(df["close"].iloc[i])
            print(f"[SMOKE][{iter_no}/{total_iters}] i={i} close={last_price:.3f} elapsed={elapsed:.1f}s")

        window_df = df.iloc[i - window + 1: i + 1]  # view, not copy
        analyzer = GoldNDSAnalyzer(window_df, config=config_payload)

        result = analyzer.generate_trading_signal(timeframe="M15", scalping_mode=True)

        entry_type = "NONE"
        entry_model = "MARKET"
        entry_tier = "NONE"
        entry_level = None
        entry_context = {}
        signal_context = {}
        flow_zones = {}

        if result.context:
            entry_idea = result.context.get("entry_idea", {}) or {}
            entry_type = entry_idea.get("entry_type", "NONE") or "NONE"
            entry_model = entry_idea.get("entry_model", "MARKET") or "MARKET"
            entry_tier = entry_idea.get("tier", "NONE") or "NONE"
            entry_level = entry_idea.get("entry_level") or result.entry_price
            entry_context = entry_idea.get("metrics", {}) or result.context.get("entry_context", {}) or {}
            signal_context = result.context.get("analysis_signal_context", {}) or {}
            analysis_data = result.context.get("analysis_data", {}) or {}
            flow_zones = analysis_data.get("flow_zones", {}) or {}
            rejections = entry_context.get("zone_rejections", {}) or {}
            for key in zone_rejections.keys():
                zone_rejections[key] += int(rejections.get(key, 0) or 0)
            momentum_block = entry_context.get("momentum_block_reason")
            if momentum_block in momentum_rejections:
                momentum_rejections[momentum_block] += 1

        if signal_context.get("strong_trend"):
            strong_trend_cycles += 1

        entry_counts.setdefault(entry_type, 0)
        entry_counts[entry_type] += 1
        tier_counts.setdefault(entry_tier, 0)
        tier_counts[entry_tier] += 1
        entry_model_counts.setdefault(entry_model, 0)
        entry_model_counts[entry_model] += 1

        breakers = flow_zones.get("breakers") or []
        inversion_fvgs = flow_zones.get("inversion_fvgs") or []
        ifvg_zone_count += len(inversion_fvgs)
        for zone in breakers + inversion_fvgs:
            if zone and not zone.get("fresh", True):
                non_fresh_zones += 1
            if zone and not zone.get("eligible", True):
                retest_reason = str(zone.get("retest_reason") or "").upper()
                if retest_reason in retest_rejections:
                    retest_rejections[retest_reason] += 1

        if result.signal in {"BUY", "SELL"} and entry_level:
            atr_value = None
            if result.context:
                market_metrics = result.context.get("market_metrics", {}) or {}
                atr_value = market_metrics.get("atr_short") or market_metrics.get("atr")

            sl_price = result.stop_loss
            tp_price = result.take_profit
            if sl_price is None or tp_price is None:
                sltp = risk_manager._compute_scalping_sl_tp(
                    signal=result.signal,
                    entry_price=float(entry_level),
                    atr_value=atr_value,
                    recent_low=entry_context.get("recent_low"),
                    recent_high=entry_context.get("recent_high"),
                    config_payload=risk_manager.settings,
                )
                sl_price = sltp.get("stop_loss")
                tp_price = sltp.get("take_profit")

            if sl_price is not None and tp_price is not None:
                sl_pips = _distance_pips(abs(float(entry_level) - float(sl_price)), resolved_point_size)
                tp_pips = _distance_pips(abs(float(tp_price) - float(entry_level)), resolved_point_size)
                sl_pips_list.append(sl_pips)
                tp_pips_list.append(tp_pips)
                sl_pips_by_tier.setdefault(entry_tier, []).append(sl_pips)
                tp_pips_by_tier.setdefault(entry_tier, []).append(tp_pips)
        elif result.signal == "NONE" and result.reasons:
            reason_key = result.reasons[-1]
            rejection_reasons[reason_key] = rejection_reasons.get(reason_key, 0) + 1

        if "spread" in df.columns and not df["spread"].isna().all():
            spread_value = float(df["spread"].iloc[i] or 0.0)
            spread_pips = _distance_pips(spread_value, resolved_point_size)
            spread_max = float(risk_manager.settings.get("SPREAD_MAX_PIPS", 2.5))
            if spread_pips > spread_max:
                spread_rejections["spread_too_high"] += 1

    # ---- Summary prints ----
    sl_avg = sum(sl_pips_list) / len(sl_pips_list) if sl_pips_list else 0.0
    tp_avg = sum(tp_pips_list) / len(tp_pips_list) if tp_pips_list else 0.0

    print("\nEntry type counts:")
    for key in ["BREAKER", "IFVG", "MOMENTUM", "LEGACY", "NONE"]:
        print(f"  {key}: {entry_counts.get(key, 0)}")

    print("Tier counts:")
    for key in ["A", "B", "C", "D", "NONE"]:
        print(f"  {key}: {tier_counts.get(key, 0)}")

    print("Entry model counts:")
    for key in ["MARKET", "STOP"]:
        print(f"  {key}: {entry_model_counts.get(key, 0)}")

    print("Zone rejections:")
    for key in ["too_many_touches", "stale", "too_far", "too_old"]:
        print(f"  {key}: {zone_rejections.get(key, 0)}")

    print("Retest rejections:")
    for key in ["TOO_MANY_TOUCHES", "FIRST_TOUCH_UNCONFIRMED", "NO_CONFIRMED_TOUCH"]:
        print(f"  {key}: {retest_rejections.get(key, 0)}")

    print("Momentum rejections:")
    for key in ["time_outside_window"]:
        print(f"  {key}: {momentum_rejections.get(key, 0)}")

    print("Spread rejections:")
    print(f"  spread_too_high: {spread_rejections.get('spread_too_high', 0)}")

    print(f"Detected IFVG zones: {ifvg_zone_count}")
    print(f"Non-fresh zones observed: {non_fresh_zones}")
    print(f"Average SL pips: {sl_avg:.2f}")
    print(f"Average TP pips: {tp_avg:.2f}")

    print("Average SL/TP by tier:")
    for tier_key in ["A", "B", "C", "D", "NONE"]:
        tier_sl = sl_pips_by_tier.get(tier_key, [])
        tier_tp = tp_pips_by_tier.get(tier_key, [])
        tier_sl_avg = sum(tier_sl) / len(tier_sl) if tier_sl else 0.0
        tier_tp_avg = sum(tier_tp) / len(tier_tp) if tier_tp else 0.0
        print(f"  {tier_key}: sl_avg={tier_sl_avg:.2f} tp_avg={tier_tp_avg:.2f}")

    if rejection_reasons:
        print("Rejected trades by reason:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")

    failed = False
    diagnostics = []
    total_cycles = sum(entry_counts.values()) or 1
    tier_c_ratio = entry_counts.get("MOMENTUM", 0) / total_cycles

    print(
        f"Tier ratios: A={entry_counts.get('BREAKER', 0)/total_cycles:.2%} "
        f"B={entry_counts.get('IFVG', 0)/total_cycles:.2%} "
        f"C={tier_c_ratio:.2%}"
    )

    if sl_pips_list and max(sl_pips_list) > 60:
        failed = True
        diagnostics.append(f"SL pips exceeded 60 (max={max(sl_pips_list):.2f})")
    if strong_trend_cycles > 0 and entry_counts.get("MOMENTUM", 0) == 0:
        failed = True
        diagnostics.append("Tier C momentum trades missing during strong trend cycles.")
    if entry_model_counts.get("MARKET", 0) == 0:
        failed = True
        diagnostics.append("MARKET entries missing.")
    if ifvg_zone_count == 0:
        failed = True
        diagnostics.append("IFVG detection is zero (possible inversion FVG bug).")
    if tp_pips_list and max(tp_pips_list) > 100:
        failed = True
        diagnostics.append(f"TP pips exceeded 100 (max={max(tp_pips_list):.2f})")

    elapsed = time.time() - t0
    print(f"\n[SMOKE] done. elapsed={elapsed:.1f}s")

    if failed:
        print("Sanity check failures:")
        for line in diagnostics:
            print(f"  - {line}")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test NDS Flow Scalper.")
    parser.add_argument("--csv", required=True, help="Path to CSV file with OHLCV data.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit (tail).")
    parser.add_argument("--window", type=int, default=800, help="Rolling window size (bars) per evaluation.")
    parser.add_argument("--step", type=int, default=1, help="Evaluate every N bars (speed-up).")
    parser.add_argument("--progress", type=int, default=50, help="Print progress every N iterations (0 to disable).")
    args = parser.parse_args()

    exit_code = run(args.csv, args.limit, args.window, args.step, args.progress)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
