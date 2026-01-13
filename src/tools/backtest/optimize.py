from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Dict, Any, List, Iterable, Optional
from dataclasses import asdict

import pandas as pd

# Force non-interactive backend (safe for .bat double-click runs)
import matplotlib
matplotlib.use("Agg")  # noqa: E402

from .engine import NDSBacktester, BacktestConfig
from .io import load_ohlcv

# Plot helpers
from .plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_trade_pnl_hist,
    plot_signal_diagnostics,
)

# ------------------------------------------------------------
# ✅ ADD: Cycle-log enrichment helpers (extract entry context fields)
# ------------------------------------------------------------
def _safe_json_loads(x: Any) -> Optional[dict]:
    """Parse JSON string into dict safely; return None if not parseable."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _extract_context_dict_from_row(row: Any) -> dict:
    """
    Extract context dict from a cycle_log row.
    - row can be dict-like (Series)
    - context may already be dict OR JSON string OR missing
    """
    try:
        ctx = row.get("context", None)
    except Exception:
        ctx = None
    ctx_dict = _safe_json_loads(ctx)
    return ctx_dict or {}


def _extract_entry_fields_from_context(ctx: dict) -> Dict[str, Any]:
    """
    Extract entry_source / entry_distance_* / zone_meta from context.
    Supports both:
      ctx["entry_source"]...
    and nested:
      ctx["context"]["entry_source"]... (if someone nested it accidentally)
    """
    if not isinstance(ctx, dict):
        return {
            "entry_source": None,
            "entry_distance_pips": None,
            "entry_distance_atr": None,
            "zone_meta": None,
        }

    # direct
    entry_source = ctx.get("entry_source")
    entry_distance_pips = ctx.get("entry_distance_pips")
    entry_distance_atr = ctx.get("entry_distance_atr")
    zone_meta = ctx.get("zone_meta")

    # nested fallback
    nested = ctx.get("context")
    nested = _safe_json_loads(nested) or (nested if isinstance(nested, dict) else None)
    if isinstance(nested, dict):
        entry_source = entry_source if entry_source is not None else nested.get("entry_source")
        entry_distance_pips = entry_distance_pips if entry_distance_pips is not None else nested.get("entry_distance_pips")
        entry_distance_atr = entry_distance_atr if entry_distance_atr is not None else nested.get("entry_distance_atr")
        zone_meta = zone_meta if zone_meta is not None else nested.get("zone_meta")

    return {
        "entry_source": entry_source,
        "entry_distance_pips": entry_distance_pips,
        "entry_distance_atr": entry_distance_atr,
        "zone_meta": zone_meta,
    }


def enrich_cycle_log_with_entry_context(cycle_log: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns to cycle_log:
      - entry_source
      - entry_distance_pips
      - entry_distance_atr
      - zone_meta_json

    Leaves original data intact (adds new columns only).
    Safe for empty/missing columns.
    """
    if cycle_log is None or cycle_log.empty:
        return cycle_log

    x = cycle_log.copy()

    # Ensure 'context' exists (if not, create empty)
    if "context" not in x.columns:
        x["context"] = None

    entry_sources: List[Any] = []
    entry_dist_pips: List[Any] = []
    entry_dist_atr: List[Any] = []
    zone_meta_json: List[Any] = []

    # Row-wise extraction (robust)
    for _, row in x.iterrows():
        ctx = _extract_context_dict_from_row(row)
        fields = _extract_entry_fields_from_context(ctx)

        entry_sources.append(fields.get("entry_source"))
        entry_dist_pips.append(fields.get("entry_distance_pips"))
        entry_dist_atr.append(fields.get("entry_distance_atr"))

        zm = fields.get("zone_meta")
        if zm is None:
            zone_meta_json.append(None)
        else:
            # Keep JSON string for CSV friendliness
            try:
                zone_meta_json.append(json.dumps(zm, ensure_ascii=False))
            except Exception:
                zone_meta_json.append(str(zm))

    # Add columns (only additions; no overwrites of existing columns unless same name exists)
    if "entry_source" not in x.columns:
        x["entry_source"] = entry_sources
    else:
        # if column exists, still keep it; but do not overwrite by default
        # (we just add an alternate column to avoid altering existing behavior)
        x["entry_source_extracted"] = entry_sources

    if "entry_distance_pips" not in x.columns:
        x["entry_distance_pips"] = entry_dist_pips
    else:
        x["entry_distance_pips_extracted"] = entry_dist_pips

    if "entry_distance_atr" not in x.columns:
        x["entry_distance_atr"] = entry_dist_atr
    else:
        x["entry_distance_atr_extracted"] = entry_dist_atr

    if "zone_meta_json" not in x.columns:
        x["zone_meta_json"] = zone_meta_json
    else:
        x["zone_meta_json_extracted"] = zone_meta_json

    return x


def export_entry_source_diagnostics(out_dir: str, cycle_log: pd.DataFrame):
    """
    Produce:
      - entry_source_summary.csv (counts by entry_source)
      - reject_by_entry_source.csv (counts by entry_source x reject_reason)
    """
    if cycle_log is None or cycle_log.empty:
        return

    x = cycle_log.copy()

    # Ensure needed columns exist
    if "entry_source" not in x.columns:
        if "entry_source_extracted" in x.columns:
            x["entry_source"] = x["entry_source_extracted"]
        else:
            x["entry_source"] = None

    if "reject_reason" not in x.columns:
        x["reject_reason"] = None

    # Summary by source
    src_summary = (
        x["entry_source"]
        .fillna("NA")
        .astype(str)
        .value_counts()
        .rename_axis("entry_source")
        .reset_index(name="count")
    )
    src_summary.to_csv(os.path.join(out_dir, "entry_source_summary.csv"), index=False, encoding="utf-8-sig")

    # Reject by source
    rej = x["reject_reason"].fillna("OK").astype(str)
    src = x["entry_source"].fillna("NA").astype(str)
    pivot = (
        pd.DataFrame({"entry_source": src, "reject_reason": rej})
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["entry_source", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    pivot.to_csv(os.path.join(out_dir, "reject_by_entry_source.csv"), index=False, encoding="utf-8-sig")


# ------------------------------------------------------------
# Timeframe helpers (anti-lookahead resample)
# ------------------------------------------------------------
_TF_TO_PANDAS_RULE = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1H",
    "H4": "4H",
    "D1": "1D",
}


def _normalize_tf(tf: str) -> str:
    tf = (tf or "").upper().strip()
    tf = tf.replace("MIN", "M")
    return tf


def maybe_resample_to_timeframe(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample OHLCV to target timeframe using strictly past data inside each bar.
    This is anti-lookahead if you later simulate bar-by-bar on the resampled bars.

    Aggregation:
      open = first
      high = max
      low  = min
      close= last
      volume = sum
    """
    if df is None or df.empty:
        return df

    target_tf = _normalize_tf(target_tf)
    rule = _TF_TO_PANDAS_RULE.get(target_tf)
    if not rule:
        return df

    if "time" not in df.columns:
        raise ValueError("Resample expects a dataframe with a 'time' column.")

    x = df.copy()
    x["time"] = pd.to_datetime(x["time"], errors="coerce")
    x = x.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Infer current bar spacing in minutes (approx)
    if len(x) < 3:
        return x
    deltas = x["time"].diff().dropna()
    median_delta = deltas.median()
    if pd.isna(median_delta):
        return x

    # If already close to target timeframe, skip resample
    # Example: M15 target = 15min, data is already 15min
    target_td = pd.to_timedelta(rule)
    # pandas '1H' etc: convert
    if "H" in rule or "D" in rule:
        target_td = pd.to_timedelta(rule.lower())

    # Heuristic: if median delta >= 0.9*target, treat as already aggregated
    if median_delta >= target_td * 0.9:
        return x

    # Perform resample
    x = x.set_index("time")
    ohlcv = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # label/closed: right is typical for trading bars; avoids using future bars
    y = (
        x.resample(rule, label="right", closed="right")
        .agg(ohlcv)
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return y


# ------------------------------------------------------------
# Grid helpers
# ------------------------------------------------------------
def iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def dotted_to_nested(overrides: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (overrides or {}).items():
        parts = k.split(".")
        cur = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return out


def run_grid_search(
    df: pd.DataFrame,
    base_bot_config: Dict[str, Any],
    grid: Dict[str, List[Any]],
    bt_cfg: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for flat in iter_grid(grid):
        nested = dotted_to_nested(flat)
        bt = NDSBacktester(base_bot_config, bt_cfg=bt_cfg, overrides=nested)

        try:
            res = bt.run(df)
            m = dict(res.metrics)
            m["overrides_flat"] = json.dumps(flat, ensure_ascii=False)
            m["error"] = ""
        except Exception as e:
            m = {
                "starting_equity": float(bt_cfg.starting_equity if bt_cfg else 0.0),
                "ending_equity": float(bt_cfg.starting_equity if bt_cfg else 0.0),
                "net_pnl": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "avg_trade_pnl": 0.0,
                "median_trade_pnl": 0.0,
                "expectancy": 0.0,
                "overrides_flat": json.dumps(flat, ensure_ascii=False),
                "error": str(e),
            }
        rows.append(m)

    out = pd.DataFrame(rows)

    # Prefer successful rows on top
    if "error" in out.columns:
        ok = out[out["error"].fillna("") == ""].copy()
        err = out[out["error"].fillna("") != ""].copy()
        if not ok.empty and "net_pnl" in ok.columns and "max_drawdown" in ok.columns:
            ok = ok.sort_values(["net_pnl", "max_drawdown"], ascending=[False, True])
        elif not ok.empty and "net_pnl" in ok.columns:
            ok = ok.sort_values(["net_pnl"], ascending=[False])
        out = pd.concat([ok, err], ignore_index=True)
        return out

    if "net_pnl" in out.columns and "max_drawdown" in out.columns:
        out = out.sort_values(["net_pnl", "max_drawdown"], ascending=[False, True])
    elif "net_pnl" in out.columns:
        out = out.sort_values(["net_pnl"], ascending=[False])
    return out


# ------------------------------------------------------------
# Data slicing / range control
# ------------------------------------------------------------
def slice_df_by_range(
    df: pd.DataFrame,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    days: Optional[int] = None,
    rows: Optional[int] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if "time" not in df.columns:
        raise ValueError("slice_df_by_range expects a dataframe with a 'time' column.")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    if date_from or date_to:
        start = pd.to_datetime(date_from) if date_from else df["time"].iloc[0]
        end = pd.to_datetime(date_to) if date_to else df["time"].iloc[-1]
        mask = (df["time"] >= start) & (df["time"] <= end)
        df = df.loc[mask].reset_index(drop=True)
        return df

    if days is not None and days > 0:
        last_time = df["time"].iloc[-1]
        start = last_time - pd.Timedelta(days=int(days))
        df = df[df["time"] >= start].reset_index(drop=True)
        return df

    if rows is not None and rows > 0:
        df = df.tail(int(rows)).reset_index(drop=True)
        return df

    return df


# ------------------------------------------------------------
# File I/O helpers
# ------------------------------------------------------------
def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _parse_grid_from_json(path: str) -> Dict[str, List[Any]]:
    obj = _read_json(path)
    grid = obj.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("Grid JSON must contain top-level key 'grid' as a dict.")
    for k, v in grid.items():
        if not isinstance(v, list):
            raise ValueError(f"Grid value must be a list for key={k}. Got {type(v)}")
        if len(v) == 0:
            raise ValueError(f"Grid list is empty for key={k}.")
    return grid


# ------------------------------------------------------------
# Plot export
# ------------------------------------------------------------
def export_plots(
    *,
    out_dir: str,
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    cycle_log: pd.DataFrame,
    prefix: str = "best",
):
    plots_dir = _ensure_dir(os.path.join(out_dir, "plots"))

    try:
        fig = plot_equity_curve(
            equity_curve,
            title=f"{prefix.upper()} - Equity Curve",
            out_path=os.path.join(plots_dir, f"{prefix}_equity_curve.png"),
        )
        fig.clf()
    except Exception:
        pass

    try:
        fig = plot_drawdown(
            equity_curve,
            title=f"{prefix.upper()} - Drawdown",
            out_path=os.path.join(plots_dir, f"{prefix}_drawdown.png"),
        )
        fig.clf()
    except Exception:
        pass

    try:
        fig = plot_trade_pnl_hist(
            trades,
            title=f"{prefix.upper()} - Trade PnL Distribution",
            out_path=os.path.join(plots_dir, f"{prefix}_pnl_hist.png"),
        )
        fig.clf()
    except Exception:
        pass

    try:
        fig = plot_signal_diagnostics(
            cycle_log,
            title=f"{prefix.upper()} - Score & Confidence",
            out_path=os.path.join(plots_dir, f"{prefix}_signal_diagnostics.png"),
        )
        fig.clf()
    except Exception:
        pass


# ------------------------------------------------------------
# Diagnostics exports
# ------------------------------------------------------------
def export_cycle_diagnostics(out_dir: str, cycle_log: pd.DataFrame):
    if cycle_log is None or cycle_log.empty:
        return

    x = cycle_log.copy()
    # ensure columns exist
    for col in ["analyzer_signal", "final_signal", "reject_reason", "is_trade_allowed"]:
        if col not in x.columns:
            x[col] = None

    # reject summary
    rej = (
        x["reject_reason"]
        .fillna("OK")
        .value_counts()
        .rename_axis("reject_reason")
        .reset_index(name="count")
    )
    rej.to_csv(os.path.join(out_dir, "cycle_reject_summary.csv"), index=False, encoding="utf-8-sig")

    # signal summary
    sig = (
        x["analyzer_signal"]
        .fillna("NA")
        .value_counts()
        .rename_axis("analyzer_signal")
        .reset_index(name="count")
    )
    sig.to_csv(os.path.join(out_dir, "signal_summary.csv"), index=False, encoding="utf-8-sig")

    # allowed summary
    if "is_trade_allowed" in x.columns:
        allowed = x["is_trade_allowed"].infer_objects(copy=False).fillna(False).astype(bool)
        allowed_rate = float(allowed.mean() * 100.0)
        df = pd.DataFrame(
            [{"allowed_rate_percent": allowed_rate, "allowed_true": int(allowed.sum()), "rows": int(len(x))}]
        )
        df.to_csv(os.path.join(out_dir, "allowed_summary.csv"), index=False, encoding="utf-8-sig")


# ------------------------------------------------------------
# Warmup adjust
# ------------------------------------------------------------
def _auto_adjust_warmup(bt_cfg: BacktestConfig, df: pd.DataFrame) -> None:
    n = int(len(df)) if df is not None else 0
    if n <= 0:
        return
    recommended = max(50, min(300, int(n * 0.2)))
    if bt_cfg.warmup_bars >= n:
        bt_cfg.warmup_bars = max(10, min(recommended, n - 1))
    if bt_cfg.warmup_bars > recommended:
        bt_cfg.warmup_bars = recommended


def _print_data_summary(tag: str, df: pd.DataFrame):
    if df is None or df.empty:
        print(f"[{tag}] df is EMPTY")
        return
    cols = list(df.columns)
    print(f"[{tag}] rows={len(df)} cols={cols}")
    if "time" in cols:
        t = pd.to_datetime(df["time"], errors="coerce")
        print(f"[{tag}] time range: {t.min()} -> {t.max()}")


# ------------------------------------------------------------
# One-shot Optimization Runner (CLI)
# ------------------------------------------------------------
def run_optimization_job(
    *,
    data_path: str,
    bot_config_path: str,
    out_dir: str,
    grid: Dict[str, List[Any]],
    dayfirst: bool = False,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    days: Optional[int] = None,
    rows: Optional[int] = None,
    warmup_bars: Optional[int] = None,
    spread: Optional[float] = None,
    slippage: Optional[float] = None,
    make_plots: bool = True,
    enable_resample: bool = True,
) -> pd.DataFrame:
    out_dir = _ensure_dir(out_dir)
    base_cfg = _read_json(bot_config_path)

    target_tf = str(base_cfg.get("trading_settings", {}).get("TIMEFRAME", "M15"))
    target_tf = _normalize_tf(target_tf)

    df = load_ohlcv(data_path, dayfirst=dayfirst)
    _print_data_summary("LOAD", df)

    df = slice_df_by_range(df, date_from=date_from, date_to=date_to, days=days, rows=rows)
    _print_data_summary("SLICE", df)

    # Resample if needed (M1 file with M15 config is your exact case)
    if enable_resample:
        before = len(df)
        df = maybe_resample_to_timeframe(df, target_tf=target_tf)
        after = len(df)
        if after != before:
            _print_data_summary(f"RESAMPLE->{target_tf}", df)

    # preview
    try:
        df.head(5000).to_csv(os.path.join(out_dir, "filtered_data_preview.csv"), index=False, encoding="utf-8-sig")
    except Exception:
        pass

    if df.empty:
        raise ValueError("After applying range filters/resample, dataframe is empty.")

    # Backtest runtime config
    bt_cfg = BacktestConfig(
        symbol=str(base_cfg.get("trading_settings", {}).get("SYMBOL", "XAUUSD!")),
        timeframe=target_tf,
        bars_to_fetch=int(base_cfg.get("trading_settings", {}).get("BARS_TO_FETCH", 3500) or 3500),
        starting_equity=float(base_cfg.get("ACCOUNT_BALANCE", 1000.0) or 1000.0),
        max_positions=int(base_cfg.get("trading_rules", {}).get("MAX_POSITIONS", 5) or 5),
        allow_multiple_positions=bool(base_cfg.get("trading_rules", {}).get("ALLOW_MULTIPLE_POSITIONS", True)),
        min_candles_between_trades=int(base_cfg.get("trading_rules", {}).get("MIN_CANDLES_BETWEEN_TRADES", 10) or 10),
        min_time_between_trades_minutes=int(base_cfg.get("trading_rules", {}).get("MIN_TIME_BETWEEN_TRADES_MINUTES", 150) or 150),
        daily_max_trades=int(base_cfg.get("trading_rules", {}).get("DAILY_MAX_TRADES", 40) or 40),
        max_daily_risk_percent=float(base_cfg.get("risk_settings", {}).get("MAX_DAILY_RISK_PERCENT", 6.0) or 6.0),
        spread=float(base_cfg.get("trading_settings", {}).get("GOLD_SPECIFICATIONS", {}).get("TYPICAL_SPREAD", 0.25) or 0.25),
        slippage=float(base_cfg.get("trading_settings", {}).get("GOLD_SPECIFICATIONS", {}).get("TYPICAL_SLIPPAGE", 0.10) or 0.10),
    )

    if warmup_bars is not None:
        bt_cfg.warmup_bars = int(warmup_bars)
    else:
        _auto_adjust_warmup(bt_cfg, df)

    if spread is not None:
        bt_cfg.spread = float(spread)
    if slippage is not None:
        bt_cfg.slippage = float(slippage)

    print(f"[BT] warmup_bars={bt_cfg.warmup_bars} bars_to_fetch={bt_cfg.bars_to_fetch} timeframe={bt_cfg.timeframe}")

    # Run grid
    results = run_grid_search(df, base_cfg, grid, bt_cfg=bt_cfg)
    results.to_csv(os.path.join(out_dir, "grid_results.csv"), index=False, encoding="utf-8-sig")

    if results.empty:
        raise ValueError("Grid search returned no rows.")
    if "error" in results.columns and (results["error"].fillna("") != "").all():
        raise RuntimeError("All grid runs failed. See grid_results.csv 'error' column for details.")

    # Best row
    if "error" in results.columns:
        ok = results[results["error"].fillna("") == ""]
        if ok.empty:
            raise RuntimeError("All grid runs failed (no successful rows). See grid_results.csv.")
        best_row = ok.iloc[0]
    else:
        best_row = results.iloc[0]

    best_overrides_flat = json.loads(best_row["overrides_flat"])
    best_overrides_nested = dotted_to_nested(best_overrides_flat)

    _save_json(os.path.join(out_dir, "best_overrides.json"), best_overrides_flat)
    _save_json(os.path.join(out_dir, "best_metrics.json"), best_row.to_dict())

    # Best run
    bt_best = NDSBacktester(base_cfg, bt_cfg=bt_cfg, overrides=best_overrides_nested)
    best_res = bt_best.run(df)

    best_res.trades.to_csv(os.path.join(out_dir, "best_trades.csv"), index=False, encoding="utf-8-sig")
    best_res.equity_curve.to_csv(os.path.join(out_dir, "best_equity_curve.csv"), encoding="utf-8-sig")
    best_res.cycle_log.to_csv(os.path.join(out_dir, "best_cycle_log.csv"), encoding="utf-8-sig")

    # ------------------------------------------------------------
    # ✅ ADD: Enrich cycle_log with entry context columns and export
    # (No changes to existing export line above; we only add enriched export.)
    # ------------------------------------------------------------
    try:
        enriched_cycles = enrich_cycle_log_with_entry_context(best_res.cycle_log)
        # Save additional enriched file for clarity (and to avoid surprises)
        enriched_cycles.to_csv(os.path.join(out_dir, "best_cycle_log_enriched.csv"), encoding="utf-8-sig")
        # Optionally overwrite the main best_cycle_log.csv as well (still no deletion; just write again)
        enriched_cycles.to_csv(os.path.join(out_dir, "best_cycle_log.csv"), encoding="utf-8-sig")

        # Export source-based diagnostics (helps answer: which branch causes rejects/deviation)
        export_entry_source_diagnostics(out_dir, enriched_cycles)

        # Also re-run cycle diagnostics on enriched cycles (adds no risk; overwrites summaries)
        export_cycle_diagnostics(out_dir, enriched_cycles)
    except Exception:
        pass
    # ------------------------------------------------------------

    # -------------------------------
    # ✅ Score distribution (diagnostic)
    # -------------------------------
    cycles = best_res.cycle_log  # <-- FIX: this is the actual cycle dataframe

    if cycles is not None and (not cycles.empty) and ("score" in cycles.columns):
        s = pd.to_numeric(cycles["score"], errors="coerce").dropna()
        if len(s):
            print("[SCORE] min=", float(s.min()), "max=", float(s.max()))
            print(
                "[SCORE] p50=", float(s.quantile(0.50)),
                "p90=", float(s.quantile(0.90)),
                "p95=", float(s.quantile(0.95)),
                "p99=", float(s.quantile(0.99)),
            )
            # optional: also write to file for review
            try:
                score_stats = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "p50": float(s.quantile(0.50)),
                    "p90": float(s.quantile(0.90)),
                    "p95": float(s.quantile(0.95)),
                    "p99": float(s.quantile(0.99)),
                    "count": int(len(s)),
                }
                _save_json(os.path.join(out_dir, "score_distribution.json"), score_stats)
            except Exception:
                pass

    _save_json(os.path.join(out_dir, "best_backtest_metrics.json"), best_res.metrics)
    _save_json(os.path.join(out_dir, "best_backtest_config.json"), {"bt_cfg": asdict(bt_cfg), "bot_overrides": best_overrides_flat})

    # Diagnostics summary to explain "why 0 trades"
    export_cycle_diagnostics(out_dir, best_res.cycle_log)

    print(f"[BEST] trades={len(best_res.trades)} equity_rows={len(best_res.equity_curve)} cycles={len(best_res.cycle_log)}")

    # Plots
    if make_plots:
        export_plots(
            out_dir=out_dir,
            equity_curve=best_res.equity_curve,
            trades=best_res.trades,
            cycle_log=best_res.cycle_log,
            prefix="best",
        )

        # ------------------------------------------------------------
        # ✅ ADD: if enriched cycles exist, also plot diagnostics on enriched file
        # (does not change original plot behavior; just adds extra artifact)
        # ------------------------------------------------------------
        try:
            enriched_path = os.path.join(out_dir, "best_cycle_log_enriched.csv")
            if os.path.exists(enriched_path):
                enriched_cycles_for_plot = pd.read_csv(enriched_path)
                export_plots(
                    out_dir=out_dir,
                    equity_curve=best_res.equity_curve,
                    trades=best_res.trades,
                    cycle_log=enriched_cycles_for_plot,
                    prefix="best_enriched",
                )
        except Exception:
            pass
        # ------------------------------------------------------------

    return results



def main():
    p = argparse.ArgumentParser(description="NDS Backtest Optimization (Grid Search) - Excel/CSV input, anti-lookahead + plots")
    p.add_argument("--data", required=True, help="Path to OHLCV Excel/CSV exported from MT5")
    p.add_argument("--config", required=True, help="Path to bot_config.json")
    p.add_argument("--out", required=True, help="Output directory")

    # date/range controls
    p.add_argument("--from", dest="date_from", default=None, help="Start datetime (e.g. 2026-01-09 00:00)")
    p.add_argument("--to", dest="date_to", default=None, help="End datetime (e.g. 2026-01-12 23:59)")
    p.add_argument("--days", type=int, default=None, help="Use last N days")
    p.add_argument("--rows", type=int, default=None, help="Use last N rows")
    p.add_argument("--dayfirst", action="store_true", help="Parse dates as day-first (for ambiguous formats)")

    # runtime knobs
    p.add_argument("--warmup", type=int, default=None, help="Warmup bars before trading")
    p.add_argument("--spread", type=float, default=None, help="Override spread ($)")
    p.add_argument("--slippage", type=float, default=None, help="Override slippage ($)")

    # grid source
    p.add_argument("--grid", default=None, help="Path to JSON file containing {'grid': {...}}")

    # plots
    p.add_argument("--no-plots", action="store_true", help="Disable saving plots")

    # resample
    p.add_argument("--no-resample", action="store_true", help="Disable auto-resampling to config timeframe")

    args = p.parse_args()

    if args.grid:
        grid = _parse_grid_from_json(args.grid)
    else:
        grid = {
            "technical_settings.ENTRY_FACTOR": [0.15, 0.20, 0.25, 0.30],
            "technical_settings.ATR_SL_MULTIPLIER": [2.0, 2.4, 2.8],
            "technical_settings.SCALPING_MIN_CONFIDENCE": [32, 35, 38],
            "risk_settings.RISK_AMOUNT_USD": [10, 15, 20, 25],
        }

    run_optimization_job(
        data_path=args.data,
        bot_config_path=args.config,
        out_dir=args.out,
        grid=grid,
        dayfirst=bool(args.dayfirst),
        date_from=args.date_from,
        date_to=args.date_to,
        days=args.days,
        rows=args.rows,
        warmup_bars=args.warmup,
        spread=args.spread,
        slippage=args.slippage,
        make_plots=(not args.no_plots),
        enable_resample=(not args.no_resample),
    )


if __name__ == "__main__":
    main()
