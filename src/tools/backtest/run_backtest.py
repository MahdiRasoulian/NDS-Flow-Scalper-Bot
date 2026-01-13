
"""
CLI runner

Example:
  python -m tools.backtest.run_backtest --data data/XAUUSD_M15.csv --config config/bot_config.json --out out_bt

Optional overrides:
  --override technical_settings.ENTRY_FACTOR=0.22
  --override technical_settings.ATR_WINDOW=14
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .engine import NDSBacktester, BacktestConfig
from .plots import plot_equity_curve, plot_drawdown, plot_trade_pnl_hist, plot_signal_diagnostics


def parse_override(s: str) -> Dict[str, Any]:
    # key=value with type inference
    if "=" not in s:
        raise ValueError(f"Invalid override: {s}")
    key, val = s.split("=", 1)
    v: Any = val
    # bool
    if val.lower() in ("true", "false"):
        v = val.lower() == "true"
    else:
        # int/float
        try:
            if "." in val:
                v = float(val)
            else:
                v = int(val)
        except Exception:
            v = val
    # dotted to nested
    parts = key.split(".")
    out: Dict[str, Any] = {}
    cur = out
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = v
    return out


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with OHLCV")
    ap.add_argument("--config", required=True, help="bot_config.json")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--spread", type=float, default=None)
    ap.add_argument("--slippage", type=float, default=None)
    ap.add_argument("--override", action="append", default=[], help="dotted.path=value (repeatable)")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    overrides: Dict[str, Any] = {}
    for o in args.override or []:
        overrides = deep_update(overrides, parse_override(o))

    bt_cfg = BacktestConfig(
        symbol=cfg.get("trading_settings", {}).get("SYMBOL", "XAUUSD!"),
        timeframe=cfg.get("trading_settings", {}).get("TIMEFRAME", "M15"),
        bars_to_fetch=int(cfg.get("trading_settings", {}).get("BARS_TO_FETCH", 3500) or 3500),
        warmup_bars=int(args.warmup),
        starting_equity=float(cfg.get("ACCOUNT_BALANCE", 1000.0) or 1000.0),
    )
    if args.spread is not None:
        bt_cfg.spread = float(args.spread)
    if args.slippage is not None:
        bt_cfg.slippage = float(args.slippage)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bt = NDSBacktester(cfg, bt_cfg=bt_cfg, overrides=overrides)
    res = bt.run(df)

    # save artifacts
    res.trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    res.equity_curve.to_csv(out_dir / "equity_curve.csv", encoding="utf-8-sig")
    if not res.cycle_log.empty:
        res.cycle_log.to_csv(out_dir / "cycle_log.csv", encoding="utf-8-sig")
    (out_dir / "metrics.json").write_text(json.dumps(res.metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # plots
    plot_equity_curve(res.equity_curve, out_path=str(out_dir / "equity_curve.png"))
    plot_drawdown(res.equity_curve, out_path=str(out_dir / "drawdown.png"))
    if not res.trades.empty:
        plot_trade_pnl_hist(res.trades, out_path=str(out_dir / "trade_pnl_hist.png"))
    if not res.cycle_log.empty:
        plot_signal_diagnostics(res.cycle_log, out_path=str(out_dir / "score_conf.png"))

    print("Done.")
    print(json.dumps(res.metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
