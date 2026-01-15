# NDS Flow Scalper Backtest v2

This package rebuilds the legacy backtesting/optimization toolkit to align with the **current** NDS Flow Scalper pipeline (XAUUSD M15), including touch-debounce logic, the A/B/C tier pipeline, and live parity diagnostics.

## Features

- Walk-forward candle-by-candle simulation (anti-lookahead)
- MARKET + STOP/LIMIT pending order execution
- Spread + slippage + conservative intrabar SL/TP handling
- Analyzer + RiskManager integration (same as bot)
- Cycle log with tier/entry context + rejection reasons
- Grid search optimization with configurable objective
- Live parity comparison against live cycle logs

## Quick Start

```bash
pip install -r requirements.txt

python -m tools.backtest_v2.cli \
  --data data/XAUUSD_M15.csv \
  --config config/bot_config.json \
  --out out_bt
```

Optional overrides:

```bash
python -m tools.backtest_v2.cli \
  --data data/XAUUSD_M15.csv \
  --config config/bot_config.json \
  --out out_bt \
  --override technical_settings.ENTRY_FACTOR=0.25 \
  --override flow_settings.FLOW_MAX_TOUCHES=5
```

## Optimization

```bash
python -m tools.backtest_v2.optimize \
  --data data/XAUUSD_M15.csv \
  --config config/bot_config.json \
  --grid src/tools/backtest_v2/grid.example.json \
  --out out_opt
```

The optimizer produces:

- `grid_results.csv`
- `best_overrides.json`
- `best_metrics.json`
- `best_trades.csv`, `best_equity_curve.csv`, `best_cycle_log.csv`
- `best_equity_curve.png`, `best_drawdown.png`, `best_pnl_hist.png`, `best_score_conf.png`

## Live Parity Checks

Use `--parity-live-cycle-log` to compare distributions against live cycle logs:

```bash
python -m tools.backtest_v2.cli \
  --data data/XAUUSD_M15.csv \
  --config config/bot_config.json \
  --out out_bt \
  --parity-live-cycle-log path/to/live_cycle_log.csv
```

The comparison writes `parity_report.txt` alongside the backtest output.

## Output Artifacts

- `trades.csv` – trade ledger
- `equity_curve.csv` – equity/balance series
- `cycle_log.csv` – per-cycle diagnostics
- `metrics.json` – KPIs (PnL, win rate, PF, max DD, expectancy, etc.)
- `summary.txt` – smoke-test-like distribution summary
- Plots: `equity_curve.png`, `drawdown.png`, `pnl_hist.png`, `score_conf.png`

## Reproducibility

- The backtest engine is deterministic (no randomness).
- Analyzer and RiskManager are invoked using the same config keys as the live bot.
- Slippage and spread are deterministic unless overridden in CLI.

## Notes

- STOP/LIMIT orders are simulated using high/low triggers.
- If both SL and TP are hit in the same bar, SL is applied (worst-case).
- Missing columns (e.g., `tick_volume` vs `volume`) are handled by the loader.
