from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .engine import BacktestEngine, BacktestConfig
from .io import load_config, load_ohlcv, apply_overrides, slice_ohlcv
from .metrics import summarize_cycle_log
from .objectives import get_objective_config, objective_from_payload, score_objective
from .plots import plot_drawdown, plot_equity_curve, plot_signal_diagnostics, plot_trade_pnl_hist


# -----------------------------
# Logging helpers
# -----------------------------
def _setup_logging(out_dir: Path, level: str = "INFO") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "optimize.log"

    logger = logging.getLogger("backtest.optimize")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # avoid double logs if root configured elsewhere

    # Clear existing handlers (important if rerun in same process)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("✅ Logging initialized | level=%s | file=%s", level.upper(), str(log_path))
    return logger


def _fmt_sec(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h > 0:
        return f"{h}h {m}m {ss}s"
    if m > 0:
        return f"{m}m {ss}s"
    return f"{ss}s"


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


# -----------------------------
# NEW: diagnostics helpers
# -----------------------------
def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _summarize_cycle_log_for_debug(logger: logging.Logger, cycle_log: pd.DataFrame) -> None:
    if cycle_log is None or cycle_log.empty:
        logger.warning("⚠️ cycle_log is empty -> strategy produced no cycles or cycles were not recorded.")
        return

    logger.info("🧾 cycle_log present | rows=%d cols=%d", len(cycle_log), len(cycle_log.columns))
    logger.debug("cycle_log columns=%s", list(cycle_log.columns))

    # Heuristic columns
    cols = set([c.lower() for c in cycle_log.columns])
    tier_col = None
    for cand in ("tier", "signal_tier", "grade"):
        if cand in cols:
            # map back to original case
            tier_col = next(c for c in cycle_log.columns if c.lower() == cand)
            break

    signal_col = None
    for cand in ("signal", "final_signal", "trade_signal"):
        if cand in cols:
            signal_col = next(c for c in cycle_log.columns if c.lower() == cand)
            break

    reject_col = None
    for cand in ("reject_reason", "rejection_reason", "reject", "reason"):
        if cand in cols:
            reject_col = next(c for c in cycle_log.columns if c.lower() == cand)
            break

    # Tier distribution
    if tier_col:
        vc = cycle_log[tier_col].astype(str).value_counts().head(10)
        logger.info("📌 Tier distribution (top10): %s", vc.to_dict())
    else:
        logger.warning("⚠️ No tier column found in cycle_log (expected one of tier/signal_tier/grade). AB_RATIO may be invalid.")

    # Signal distribution
    if signal_col:
        vc = cycle_log[signal_col].astype(str).value_counts().head(10)
        logger.info("📌 Signal distribution (top10): %s", vc.to_dict())
    else:
        logger.warning("⚠️ No signal column found in cycle_log (expected one of signal/final_signal/trade_signal).")

    # Rejection reasons
    if reject_col:
        vc = cycle_log[reject_col].astype(str).value_counts().head(15)
        logger.info("📌 Reject reasons (top15): %s", vc.to_dict())
    else:
        logger.info("ℹ️ No reject-reason column detected in cycle_log.")


def _export_debug_artifacts(
    logger: logging.Logger,
    out_dir: Path,
    tag: str,
    override: Dict[str, Any],
    result,
) -> None:
    """
    Export artifacts even when failsafe fails, so we can diagnose why trades are low/zero.
    """
    try:
        (out_dir / f"{tag}_overrides.json").write_text(
            json.dumps(override, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.error("Failed to write %s_overrides.json: %s", tag, e)

    try:
        (out_dir / f"{tag}_metrics.json").write_text(
            json.dumps(result.metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.error("Failed to write %s_metrics.json: %s", tag, e)

    try:
        if hasattr(result, "trades") and result.trades is not None:
            result.trades.to_csv(out_dir / f"{tag}_trades.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        logger.error("Failed to write %s_trades.csv: %s", tag, e)

    try:
        if hasattr(result, "equity_curve") and result.equity_curve is not None:
            result.equity_curve.to_csv(out_dir / f"{tag}_equity_curve.csv", encoding="utf-8-sig")
    except Exception as e:
        logger.error("Failed to write %s_equity_curve.csv: %s", tag, e)

    try:
        if hasattr(result, "cycle_log") and result.cycle_log is not None and not result.cycle_log.empty:
            result.cycle_log.to_csv(out_dir / f"{tag}_cycle_log.csv", encoding="utf-8-sig")
    except Exception as e:
        logger.error("Failed to write %s_cycle_log.csv: %s", tag, e)

    logger.info("✅ Debug artifacts saved with tag=%s (files: %s_*.csv/json)", tag, tag)


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
    ap.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    # NEW: always export debug artifacts for each combo (recommended for diagnosis)
    ap.add_argument("--export-debug", action="store_true", help="Export debug_* artifacts for each combo")
    args = ap.parse_args()

    out_dir = Path(args.out)
    logger = _setup_logging(out_dir, level=args.log_level)

    t0 = time.perf_counter()
    logger.info("==================================================")
    logger.info("🚀 Backtest Optimization Started")
    logger.info("data=%s", args.data)
    logger.info("config=%s", args.config)
    logger.info("grid=%s", args.grid)
    logger.info("out=%s", str(out_dir))
    logger.info("warmup=%s spread=%s slippage=%s rows=%s days=%s objective=%s export_debug=%s",
                args.warmup, args.spread, args.slippage, args.rows, args.days, args.objective, args.export_debug)
    logger.info("==================================================")

    # Load data
    logger.info("📥 Loading OHLCV: %s", args.data)
    df = load_ohlcv(args.data)
    logger.info("✅ OHLCV loaded | rows=%d | cols=%s", len(df), list(df.columns))

    # Slice
    logger.info("✂️ Slicing OHLCV | rows=%s days=%s", args.rows, args.days)
    df = slice_ohlcv(df, rows=args.rows, days=args.days)
    logger.info("✅ Slice complete | rows=%d | from=%s | to=%s",
                len(df), str(df["time"].iloc[0]), str(df["time"].iloc[-1]))

    # Warmup sanity
    if args.warmup >= len(df):
        logger.warning("⚠️ warmup=%d >= df_rows=%d -> engine may never trade. Consider lowering warmup.",
                       args.warmup, len(df))
    else:
        effective = len(df) - args.warmup
        logger.info("🧪 Warmup sanity | warmup=%d -> effective bars after warmup=%d", args.warmup, effective)

    # Data sanity
    try:
        if "close" in df.columns:
            logger.info("📈 Price snapshot | close[min]=%.2f close[max]=%.2f close[last]=%.2f",
                        float(df["close"].min()), float(df["close"].max()), float(df["close"].iloc[-1]))
    except Exception:
        pass

    # Load configs
    logger.info("📦 Loading config: %s", args.config)
    config = load_config(args.config)
    logger.info("✅ Config loaded | top_keys=%s", list(config.keys()))

    logger.info("📦 Loading grid: %s", args.grid)
    grid_payload = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    grid = grid_payload.get("grid", {})
    objective_payload = grid_payload.get("objective")
    failsafe = grid_payload.get("failsafe", {})
    logger.info("✅ Grid loaded | grid_keys=%s | failsafe=%s",
                list(grid.keys()) if isinstance(grid, dict) else type(grid),
                failsafe if isinstance(failsafe, dict) else type(failsafe))

    # Objective
    if objective_payload:
        objective = objective_from_payload(objective_payload)
        logger.info("🎯 Objective: from grid payload")
    else:
        objective = get_objective_config(args.objective)
        logger.info("🎯 Objective: %s", args.objective)
    logger.info("🎯 Objective config: %s", objective)

    # Build combos
    logger.info("🧮 Building grid combinations ...")
    combinations = _grid_combinations(grid)
    total = len(combinations)
    logger.info("✅ Combinations generated: %d", total)

    results: List[Dict[str, Any]] = []
    combo_times: List[float] = []

    # Run grid
    for idx, override in enumerate(combinations, start=1):
        combo_t0 = time.perf_counter()
        flat = _flatten_grid(override)

        logger.info("--------------------------------------------------")
        logger.info("🔁 Running combo %d/%d | overrides=%s", idx, total, flat)

        try:
            merged_config = apply_overrides(config, [override])

            bt_config = BacktestConfig.from_bot_config(
                merged_config,
                warmup=args.warmup,
                spread=args.spread,
                slippage=args.slippage,
            )

            # In case BacktestConfig is a dataclass
            try:
                logger.debug("BacktestConfig: %s", asdict(bt_config))  # type: ignore[arg-type]
            except Exception:
                logger.debug("BacktestConfig: %s", bt_config)

            # Key config sanity (log the critical knobs that often block trading)
            try:
                ts = merged_config.get("technical_settings", {}) or {}
                rs = merged_config.get("risk_settings", {}) or {}
                fr = merged_config.get("flow_settings", {}) or {}
                tr = merged_config.get("trading_rules", {}) or {}

                logger.info(
                    "🔧 Critical knobs | MIN_CONF=%s ENTRY_FACTOR=%s MAX_DEV_PIPS=%s FLOW_MAX_TOUCHES=%s FLOW_MIN_SEP=%s MIN_CANDLES_BETWEEN=%s MAX_POS=%s",
                    ts.get("SCALPING_MIN_CONFIDENCE"),
                    ts.get("ENTRY_FACTOR"),
                    rs.get("MAX_PRICE_DEVIATION_PIPS"),
                    fr.get("FLOW_MAX_TOUCHES"),
                    fr.get("FLOW_TOUCH_MIN_SEPARATION_BARS"),
                    tr.get("MIN_CANDLES_BETWEEN_TRADES"),
                    tr.get("MAX_POSITIONS"),
                )
            except Exception:
                pass

            engine = BacktestEngine(config=merged_config, bt_config=bt_config, overrides=override)

            logger.info("▶️ engine.run started | df_rows=%d", len(df))
            result = engine.run(df)
            logger.info("✅ engine.run finished")

            # Export debug artifacts (optional, but recommended while diagnosing)
            if args.export_debug:
                _export_debug_artifacts(logger, out_dir, tag=f"debug_combo_{idx:03d}", override=override, result=result)

            # Summaries
            diagnostics = summarize_cycle_log(result.cycle_log)
            score = score_objective(result.metrics, diagnostics, objective)
            passed, reason = _passes_failsafe(result.metrics, diagnostics, failsafe)

            net_pnl = result.metrics.get("net_pnl")
            pf = result.metrics.get("profit_factor")
            mdd = result.metrics.get("max_drawdown_pct")
            trades = result.metrics.get("total_trades")
            ab_ratio = diagnostics.tier_ratios.get("A+B", 0.0)

            logger.info(
                "📊 Combo metrics | passed=%s reason=%s score=%s net_pnl=%s pf=%s mdd=%s trades=%s ab_ratio=%s",
                passed, reason, score if passed else float("-inf"),
                net_pnl, pf, mdd, trades, ab_ratio
            )

            # NEW: deep cycle_log diagnostic summary (key for “no trades” cases)
            _summarize_cycle_log_for_debug(logger, result.cycle_log)

            results.append(
                {
                    "grid_id": idx,
                    **flat,
                    "score": score if passed else float("-inf"),
                    "passed": passed,
                    "failsafe_reason": reason,
                    "net_pnl": net_pnl,
                    "profit_factor": pf,
                    "max_drawdown_pct": mdd,
                    "trades": trades,
                    "ab_ratio": ab_ratio,
                }
            )

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("❌ Combo %d failed: %s", idx, str(e))
            logger.error("Traceback:\n%s", tb)

            # Still record a row so optimization continues
            results.append(
                {
                    "grid_id": idx,
                    **flat,
                    "score": float("-inf"),
                    "passed": False,
                    "failsafe_reason": f"EXCEPTION: {type(e).__name__}",
                    "net_pnl": None,
                    "profit_factor": None,
                    "max_drawdown_pct": None,
                    "trades": None,
                    "ab_ratio": None,
                }
            )

        combo_dt = time.perf_counter() - combo_t0
        combo_times.append(combo_dt)
        avg = sum(combo_times) / len(combo_times)
        remaining = (total - idx) * avg
        logger.info("⏱️ Combo time=%s | avg=%s | ETA=%s",
                    _fmt_sec(combo_dt), _fmt_sec(avg), _fmt_sec(remaining))

    # Save overall results
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("score", ascending=False)

    results_path = out_dir / "grid_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    logger.info("💾 Saved: %s (rows=%d)", str(results_path), len(results_df))

    # Find best passed
    best_row = results_df[results_df["passed"] == True].head(1)  # noqa: E712
    if best_row.empty:
        logger.warning("⚠️ No configuration passed failsafe filters.")
        logger.warning("➡️ Tip: rerun with --export-debug to generate debug_combo_*.csv for diagnosis.")
        print("No configuration passed failsafe filters.")
        logger.info("Total elapsed: %s", _fmt_sec(time.perf_counter() - t0))
        return

    best_grid_id = int(best_row["grid_id"].iloc[0])
    best_index = best_grid_id - 1
    best_override = combinations[best_index]

    overrides_path = out_dir / "best_overrides.json"
    overrides_path.write_text(json.dumps(best_override, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("🏆 Best grid_id=%d saved overrides: %s", best_grid_id, str(overrides_path))

    # Run best again (full artifacts)
    logger.info("🔁 Re-running best configuration to export artifacts ...")
    best_config = apply_overrides(config, [best_override])
    best_bt_config = BacktestConfig.from_bot_config(
        best_config,
        warmup=args.warmup,
        spread=args.spread,
        slippage=args.slippage,
    )

    best_engine = BacktestEngine(config=best_config, bt_config=best_bt_config, overrides=best_override)
    logger.info("▶️ best_engine.run started | df_rows=%d", len(df))
    best_result = best_engine.run(df)
    logger.info("✅ best_engine.run finished")

    metrics_path = out_dir / "best_metrics.json"
    metrics_path.write_text(json.dumps(best_result.metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("💾 Saved: %s", str(metrics_path))

    trades_path = out_dir / "best_trades.csv"
    best_result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    logger.info("💾 Saved: %s (rows=%d)", str(trades_path), len(best_result.trades))

    eq_path = out_dir / "best_equity_curve.csv"
    best_result.equity_curve.to_csv(eq_path, encoding="utf-8-sig")
    logger.info("💾 Saved: %s (rows=%d)", str(eq_path), len(best_result.equity_curve))

    if not best_result.cycle_log.empty:
        cycle_path = out_dir / "best_cycle_log.csv"
        best_result.cycle_log.to_csv(cycle_path, encoding="utf-8-sig")
        logger.info("💾 Saved: %s (rows=%d)", str(cycle_path), len(best_result.cycle_log))
    else:
        logger.info("ℹ️ best_result.cycle_log is empty; skipped saving best_cycle_log.csv")

    # Plots
    logger.info("🖼️ Generating plots ...")
    try:
        plot_equity_curve(best_result.equity_curve, out_path=str(out_dir / "best_equity_curve.png"))
        logger.info("✅ Plot saved: best_equity_curve.png")
    except Exception as e:
        logger.error("Plot equity_curve failed: %s", e)

    try:
        plot_drawdown(best_result.equity_curve, out_path=str(out_dir / "best_drawdown.png"))
        logger.info("✅ Plot saved: best_drawdown.png")
    except Exception as e:
        logger.error("Plot drawdown failed: %s", e)

    if not best_result.trades.empty:
        try:
            plot_trade_pnl_hist(best_result.trades, out_path=str(out_dir / "best_pnl_hist.png"))
            logger.info("✅ Plot saved: best_pnl_hist.png")
        except Exception as e:
            logger.error("Plot pnl_hist failed: %s", e)
    else:
        logger.info("ℹ️ No trades; skipped best_pnl_hist.png")

    if not best_result.cycle_log.empty:
        try:
            plot_signal_diagnostics(best_result.cycle_log, out_path=str(out_dir / "best_score_conf.png"))
            logger.info("✅ Plot saved: best_score_conf.png")
        except Exception as e:
            logger.error("Plot signal_diagnostics failed: %s", e)
    else:
        logger.info("ℹ️ No cycle_log; skipped best_score_conf.png")

    total_dt = time.perf_counter() - t0
    logger.info("✅ Optimization complete | elapsed=%s", _fmt_sec(total_dt))
    print("Optimization complete.")


if __name__ == "__main__":
    main()
