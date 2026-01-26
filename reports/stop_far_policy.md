# Stop Far-from-Market Policy (RiskManager)

## Summary
When the analyzer emits a `STOP` entry that is far away from the current market price, the RiskManager now applies a deterministic **STOP_FAR_POLICY** before RR validation. This prevents RR collapse from invalid geometry and enables a controlled response based on regime.

## Inputs Used
The policy only uses data already available to `RiskManager.finalize_order`:
- Live price snapshot (`bid`, `ask`, derived `market_entry`).
- `entry_model=STOP` and `entry_level` from analyzer.
- ATR (from `market_metrics.atr` or `atr_short` if available).
- ADX and volatility state (from `market_metrics` or fallback payload fields).
- Analyzer `confidence` (for LIMIT vs WAIT).

## Decision Flow
1. **Soft threshold** (`STOP_MAX_DEVIATION_PIPS`): policy activates when deviation exceeds this value.
2. **Hard reject** (`STOP_HARD_REJECT_PIPS`): if deviation exceeds this cap → reject with `Stop too far`.
3. **Trend continuation** (`TREND_STRENGTH_ADX_MIN`): cap entry distance using `MAX_ENTRY_CAP_PIPS` and recompute SL/TP.
4. **Mean reversion** (`MEAN_REVERSION_ADX_MAX` or low volatility): convert to **LIMIT** (or **WAIT** if confidence is below `LIMIT_ORDER_MIN_CONFIDENCE`).
5. **Fallback**: reject if regime cannot be classified.

## Parameters (bot_config.json)
Defined under `risk_manager_config`:
- `STOP_MAX_DEVIATION_PIPS`: soft threshold that triggers the policy.
- `STOP_HARD_REJECT_PIPS`: hard safety cap → immediate reject.
- `STOP_CONVERT_TO_LIMIT_PIPS`: distance for LIMIT pullback entries in mean-reversion.
- `TREND_STRENGTH_ADX_MIN`: ADX threshold to treat as continuation.
- `MEAN_REVERSION_ADX_MAX`: ADX threshold to treat as mean-reversion.
- `MAX_ENTRY_CAP_PIPS`: max allowed distance for near-executable capped STOP.

## Trade-offs
- **Pros:** Avoids false RR rejections; preserves valid breakout setups; introduces deterministic behavior.
- **Cons:** Capped entries may deviate from original structural levels; LIMIT conversions introduce pullback risk.

## Tuning Guidance
- Raise `STOP_HARD_REJECT_PIPS` only if you trust the analyzer on distant breakouts.
- Lower `MAX_ENTRY_CAP_PIPS` to reduce chase risk in fast markets.
- Adjust `TREND_STRENGTH_ADX_MIN` and `MEAN_REVERSION_ADX_MAX` to fit your preferred regime boundaries.

## Sample Log (Scenario)
```
STOP_FAR_POLICY:deviation_pips=114.5 soft=70.0 hard=120.0 adx=14.2(payload) vol=LOW
STOP_FAR_POLICY:LIMIT limit_entry=5070.71 limit_pips=25.0
```

This ensures the trade is **converted to a LIMIT pullback** (or WAIT if confidence is too low) instead of being rejected for **RR ratio below minimum** when STOP entries are too far away.
