from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .engine import BacktestEngine, BacktestConfig
from .io import load_config, load_ohlcv, apply_overrides, slice_ohlcv
from .metrics import summarize_cycle_log
from .objectives import get_objective_config, objective_from_payload, score_objective
from .plots import plot_drawdown, plot_equity_curve, plot_signal_diagnostics, plot_trade_pnl_hist


def _nested_override(path: str, value: Any) -> Dict[str, Any]:
    parts = path.split(".")
    out: Dict[str, Any] = {}
    cursor = out
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return out


def _grid_combinations(grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    combos = []
    for combo in itertools.product(*values):
        overrides: Dict[str, Any] = {}
        for key, val in zip(keys, combo):
            overrides = apply_overrides(overrides, [_nested_override(key, val)])
        combos.append(overrides)
    return combos


def _flatten_grid(overrides: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in overrides.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_grid(value, path))
        else:
            flat[path] = value
    return flat


def _passes_failsafe(metrics: Dict[str, Any], diagnostics, failsafe: Dict[str, Any]) -> Tuple[bool, str]:
    min_ab_ratio = float(failsafe.get("min_ab_ratio", 0.0) or 0.0)
    max_touches = int(failsafe.get("max_too_many_touches", 10_000) or 10_000)
    min_pf = float(failsafe.get("min_profit_factor", 0.0) or 0.0)

    ab_ratio = float(diagnostics.tier_ratios.get("A+B", 0.0))
    if ab_ratio < min_ab_ratio:
        return False, f"AB_RATIO<{min_ab_ratio}"

    too_many = 0
    for key in ("TOO_MANY_TOUCHES", "too_many_touches"):
        too_many += diagnostics.zone_rejections.get(key, 0)
        too_many += diagnostics.retest_rejections.get(key, 0)
    if too_many > max_touches:
        return False, f"TOO_MANY_TOUCHES>{max_touches}"

    pf = float(metrics.get("profit_factor", 0.0))
    if pf < min_pf:
        return False, f"PF<{min_pf}"

    return True, "OK"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV/XLSX with OHLCV")
    ap.add_argument("--config", required=True, help="bot_config.json")
    ap.add_argument("--grid", required=True, help="grid JSON")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--spread", type=float, default=None)
    ap.add_argument("--slippage", type=float, default=None)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--objective", type=str, default="net_pnl")
    args = ap.parse_args()

    df = load_ohlcv(args.data)
    df = slice_ohlcv(df, rows=args.rows, days=args.days)

    config = load_config(args.config)
    grid_payload = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    grid = grid_payload.get("grid", {})
    objective_payload = grid_payload.get("objective")
    failsafe = grid_payload.get("failsafe", {})

    if objective_payload:
        objective = objective_from_payload(objective_payload)
    else:
        objective = get_objective_config(args.objective)

    combinations = _grid_combinations(grid)
    results: List[Dict[str, Any]] = []

    for idx, override in enumerate(combinations, start=1):
        merged_config = apply_overrides(config, [override])
        bt_config = BacktestConfig.from_bot_config(
            merged_config,
            warmup=args.warmup,
            spread=args.spread,
            slippage=args.slippage,
        )
        engine = BacktestEngine(config=merged_config, bt_config=bt_config, overrides=override)
        result = engine.run(df)
        diagnostics = summarize_cycle_log(result.cycle_log)
        score = score_objective(result.metrics, diagnostics, objective)
        passed, reason = _passes_failsafe(result.metrics, diagnostics, failsafe)
        flat = _flatten_grid(override)
        results.append(
            {
                "grid_id": idx,
                **flat,
                "score": score if passed else float("-inf"),
                "passed": passed,
                "failsafe_reason": reason,
                "net_pnl": result.metrics.get("net_pnl"),
                "profit_factor": result.metrics.get("profit_factor"),
                "max_drawdown_pct": result.metrics.get("max_drawdown_pct"),
                "trades": result.metrics.get("total_trades"),
                "ab_ratio": diagnostics.tier_ratios.get("A+B", 0.0),
            }
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)
    results_df.to_csv(out_dir / "grid_results.csv", index=False, encoding="utf-8-sig")

    best_row = results_df[results_df["passed"]].head(1)
    if best_row.empty:
        print("No configuration passed failsafe filters.")
        return

    best_index = int(best_row["grid_id"].iloc[0]) - 1
    best_override = combinations[best_index]
    (out_dir / "best_overrides.json").write_text(json.dumps(best_override, indent=2, ensure_ascii=False), encoding="utf-8")

    best_config = apply_overrides(config, [best_override])
    best_bt_config = BacktestConfig.from_bot_config(
        best_config,
        warmup=args.warmup,
        spread=args.spread,
        slippage=args.slippage,
    )

    best_engine = BacktestEngine(config=best_config, bt_config=best_bt_config, overrides=best_override)
    best_result = best_engine.run(df)

    (out_dir / "best_metrics.json").write_text(json.dumps(best_result.metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    best_result.trades.to_csv(out_dir / "best_trades.csv", index=False, encoding="utf-8-sig")
    best_result.equity_curve.to_csv(out_dir / "best_equity_curve.csv", encoding="utf-8-sig")
    if not best_result.cycle_log.empty:
        best_result.cycle_log.to_csv(out_dir / "best_cycle_log.csv", encoding="utf-8-sig")

    plot_equity_curve(best_result.equity_curve, out_path=str(out_dir / "best_equity_curve.png"))
    plot_drawdown(best_result.equity_curve, out_path=str(out_dir / "best_drawdown.png"))
    if not best_result.trades.empty:
        plot_trade_pnl_hist(best_result.trades, out_path=str(out_dir / "best_pnl_hist.png"))
    if not best_result.cycle_log.empty:
        plot_signal_diagnostics(best_result.cycle_log, out_path=str(out_dir / "best_score_conf.png"))

    print("Optimization complete.")


if __name__ == "__main__":
    main()
