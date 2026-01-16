# NDS Signal Drop Diagnosis & Upgrade Report

## Decision DAG (current)
1. Indicators & structure → score → confidence → base signal (thresholds).
2. Final filters (spread/session/liquidity/trend gates).
3. Flow entry selector (breaker/IFVG/momentum) only if base signal is BUY/SELL.
4. Retest touch counting & zone eligibility (TOO_MANY_TOUCHES, NO_TOUCH, etc.).
5. Risk manager + execution gates.

## Log evidence (debug.log)
- cycles analyzed: 550
- summary signals: {'NONE': 550}
- pre-flow result signals: {'NONE': 522, 'SELL': 28}
- score stats: avg=49.37, min=39.2, max=54.0
- gate fails: {'trend_ok': 322, 'liquidity_ok': 104}

Top rejection reasons (summary):
- no_base_signal: 546
- ✅ BULLISH_SWEEP (Penetration: $3.97): 296
- gate:trend: 222
- ✅ BULLISH_SWEEP (Penetration: $2.66): 222
- ✅ BULLISH_SWEEP (Penetration: $2.97): 204
- ✅ BULLISH_SWEEP (Penetration: $2.62): 166
- 🔻 BEARISH_SWEEP (Penetration: $7.52): 158
- 🔻 BEARISH_SWEEP (Penetration: $4.39): 158

Retest rejection reasons:
- TOO_MANY_TOUCHES: 3344


### Touch-count inflation check
Across 11 unique TOO_MANY_TOUCHES zones in the M5 CSV window, a 200-bar window yields 0/11 zones with <=6 touches, while a 60-bar window yields 8/11 zones with <=6 touches (min separation=6 bars).
## Price validation for TOO_MANY_TOUCHES (XAUUSD M5)
Touch count recomputed using FLOW_TOUCH_PENETRATION_ATR=0.02 and min separation=6 bars.

| Zone Type | Break Time | Top | Bottom | Touches (200 bars) | Touches (60 bars) |
| --- | --- | --- | --- | --- | --- |
| BEARISH_IFVG | 2026-01-15 15:30 | 4608.00 | 4605.41 | 14 | 4 |
| BULLISH_IFVG | 2026-01-15 11:30 | 4608.62 | 4604.43 | 14 | 3 |
| BULLISH_IFVG | 2026-01-15 04:00 | 4606.71 | 4603.13 | 14 | 4 |
| BEARISH_IFVG | 2026-01-15 14:00 | 4617.13 | 4615.62 | 12 | 5 |
| BEARISH_IFVG | 2026-01-15 15:30 | 4603.86 | 4598.93 | 17 | 3 |
| BULLISH_IFVG | 2026-01-16 06:15 | 4599.05 | 4595.46 | 12 | 4 |
| BEARISH_IFVG | 2026-01-16 08:10 | 4610.95 | 4606.91 | 12 | 7 |
| BULLISH_IFVG | 2026-01-15 23:30 | 4612.64 | 4608.64 | 12 | 2 |

## Upgrade plan (high impact)
1. Two-stage signal: allow high-quality SMC flow overrides even when base score ~50.
2. Retest window: shorten touch-count horizon + rolling window on M5 to avoid inflation.
3. OB filtering: relax min size if filters drop everything; keep top-k strongest zones.
4. Threshold calibration: percentile-based thresholds from score history.
5. Validation: before/after backtest and ablation toggles per fix.