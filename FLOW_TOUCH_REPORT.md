# Flow touch accounting + gating report

## Root causes observed
- **Zone/retest counter mismatch**: `ZONE_REJECT` in the analyzer only counted *eligible* zones that failed distance/age/touch gates, while retest rejections were tracked only in the zone payloads from SMC. This meant retest rejections (e.g., `TOO_MANY_TOUCHES`) could be huge in logs/zones but never increment `zone_rejections.too_many_touches`, causing the contradictory summary output.
- **Zone identity instability across rolling windows**: `zone_id` used window-local indices (`origin_idx`/`break_idx`), which shift each rolling iteration. This made zone IDs unstable across windows and impeded dedup/traceability across iterations.
- **Retest rejection logs lacked machine-parseable context**: `TOO_MANY_TOUCHES` in retest logs had no stable ID, touch-mode, or iteration signature, making it hard to correlate with smoke-test summaries.

## Changes applied
- **Unified retest rejection accounting**: retest rejection counters are now computed in the analyzer and attached to `entry_context.retest_rejections`. `TOO_MANY_TOUCHES` retest failures also increment `zone_rejections.too_many_touches`, so the smoke summary aligns with logs.
- **Stable zone IDs across windows**: zone IDs now use candle timestamps (when available) for `origin` and `break` anchors, ensuring stable IDs across rolling windows for the same zone geometry.
- **Event-based touch enforcement**: retest touch counting remains event-based; if a candle-based touch mode is configured, a warning is emitted with a call-stack hint and the logic still enforces event-based counting.
- **Parseable retest logs**: `TOO_MANY_TOUCHES` retest rejections now emit a key=value log line with zone ID, zone type, touches, max, touch mode, touch source, and iteration window signature.
- **Distance/age baseline updates**: defaults updated to the requested ATR/age values (IFVG 1.4/45, BRK 1.2/45), and `TOO_FAR` logs now include boundary used and reference price/time/index.
- **Smoke-test summary alignment**: smoke test retest rejection totals now use `entry_context.retest_rejections` (same source as analyzer/logs) to avoid contradictions.

## Expected new smoke-test signatures
- `retest_rejections.TOO_MANY_TOUCHES` should drop sharply and align with `zone_rejections.too_many_touches` (non-zero only when true re-entries occur).
- `zone_rejections.too_far` should reduce materially with the new max distance defaults.
- `TOO_FAR` logs now show `boundary` and `ref_price/ref_time/ref_idx` for straightforward verification.

## Phase-1 optimization guidance
- **IFVG_MAX_DIST_ATR**: explore 1.2–1.6 (keep a cap <1.8 to avoid stretched entries).
- **BRK_MAX_DIST_ATR**: explore 1.0–1.3 (breaker zones remain tighter).
- **IFVG/BRK_MAX_AGE_BARS**: explore 35–55 to trade freshness against opportunity count.
