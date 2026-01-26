# RCA: Low RR Rejection When STOP Deviation Is Small

## Issue Summary
Even when STOP entries are close to market (e.g., deviation ≈ 13 pips), trades were rejected with **"RR ratio below minimum"**. The STOP-far policy correctly skipped, but the SL/TP geometry produced a low RR.

## Evidence (Active Failure)
- Entry model: STOP
- Entry level ≈ 5097.718
- Live prices: Bid=5096.14, Ask=5096.42
- Deviation ≈ 13 pips (small)
- RiskManager: RR ≈ 0.34 → reject

## Root Cause
The SL/TP model uses a **fixed TP1 (TP1_PIPS)** but can produce a **larger SL** when ATR-based distance is large. For the scenario:
- ATR-based SL distance dominates (SCALP_ATR_SL_MULT × ATR).
- TP1 remains fixed (TP1_PIPS), so TP distance is too small relative to SL distance.
- RR = TP_distance / SL_distance falls below MIN_RR_RATIO, causing rejection.

This is not a STOP-far issue; it is a TP/SL geometry mismatch for near-market STOPs.

## Fix Applied
A **RR repair** step was added in `finalize_order` after SL/TP computation and before RR validation:
- If RR < MIN_RR_RATIO, TP is expanded to meet the minimum RR.
- Expansion is bounded by `RR_REPAIR_MAX_TP_PIPS` and `RR_REPAIR_MAX_TP_ATR_MULT`.
- If bounds are exceeded, the trade is rejected with a **specific reason** ("TP cap exceeded for RR repair.").

## Diagnostics Added
- Always logs SL/TP pips, RR before/after, SL model branch, TP source.
- STOP-far policy logs whether it applied or was skipped.
- Decision notes include RR repair details and caps.

## Tuning Notes
- Increase `RR_REPAIR_MAX_TP_PIPS` to allow larger TP adjustments in high ATR.
- Decrease `RR_REPAIR_MAX_TP_ATR_MULT` to enforce stricter ATR-based limits.
