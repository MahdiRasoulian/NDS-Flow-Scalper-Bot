from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .engine import BacktestEngine, BacktestConfig
from .io import load_config, load_ohlcv, parse_override, apply_overrides, slice_ohlcv
from .plots import plot_equity_curve, plot_drawdown, plot_trade_pnl_hist, plot_signal_diagnostics
from .parity import compare_cycle_logs, format_parity_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV/XLSX with OHLCV")
    ap.add_argument("--config", required=True, help="bot_config.json")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--spread", type=float, default=None)
    ap.add_argument("--slippage", type=float, default=None)
    ap.add_argument("--override", action="append", default=[], help="dotted.path=value (repeatable)")
    ap.add_argument("--date-from", dest="date_from", default=None)
    ap.add_argument("--date-to", dest="date_to", default=None)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--parity-live-cycle-log", dest="parity_live", default=None)
    args = ap.parse_args()

    df = load_ohlcv(args.data)
    df = slice_ohlcv(df, date_from=args.date_from, date_to=args.date_to, rows=args.rows, days=args.days)

    config = load_config(args.config)

    overrides = []
    for override in args.override or []:
        overrides.append(parse_override(override))

    config = apply_overrides(config, overrides)

    bt_config = BacktestConfig.from_bot_config(
        config,
        warmup=args.warmup,
        spread=args.spread,
        slippage=args.slippage,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = BacktestEngine(config=config, bt_config=bt_config, overrides=apply_overrides({}, overrides))
    result = engine.run(df)

    result.trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    result.equity_curve.to_csv(out_dir / "equity_curve.csv", encoding="utf-8-sig")
    if not result.cycle_log.empty:
        result.cycle_log.to_csv(out_dir / "cycle_log.csv", encoding="utf-8-sig")
    (out_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.txt").write_text(result.summary_text, encoding="utf-8")

    plot_equity_curve(result.equity_curve, out_path=str(out_dir / "equity_curve.png"))
    plot_drawdown(result.equity_curve, out_path=str(out_dir / "drawdown.png"))
    if not result.trades.empty:
        plot_trade_pnl_hist(result.trades, out_path=str(out_dir / "pnl_hist.png"))
    if not result.cycle_log.empty:
        plot_signal_diagnostics(result.cycle_log, out_path=str(out_dir / "score_conf.png"))

    if args.parity_live:
        live_log = pd.read_csv(args.parity_live)
        backtest_log = result.cycle_log.reset_index()
        report = compare_cycle_logs(backtest_log, live_log)
        report_text = format_parity_report(report)
        (out_dir / "parity_report.txt").write_text(report_text, encoding="utf-8")
        print(report_text)

    print("Done.")
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
