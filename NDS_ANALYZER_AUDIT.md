# NDS Analyzer Audit (2026-01-16 03:48:38 snapshot)

## Root-cause findings
- **Zone explosion & noise**: FVG/OB detection was effectively unbounded over ~800 candles, creating a large universe of stale zones (e.g., FVG≈140, OB≈84) and amplifying retest noise. This flooded downstream scoring and retest logic, producing “TOO_MANY_TOUCHES” rejections and inconsistent selection. The lack of strong recency/size filtering allowed very old zones to remain “eligible,” which makes scalping signals unstable.
- **Touch counting too permissive**: Retest scans were allowed across wide windows with limited cooldown enforcement, causing repeated “machine-gun” touches to count as new retests. This inflated touch counts and penalized otherwise valid setups.
- **Low-liquidity regime not gated strongly enough**: In the provided logs, ADX≈14, RVOL≈0.41 (VERY_LOW), and ASIA session activity=LOW. The previous gating logic allowed signals to pass without a strict liquidity safeguard, which can lead to overtrading in weak regimes.
- **Regime weighting mismatch**: Low ADX + low RVOL conditions still allowed trend or zone signals to contribute strongly to score, which contradicts the intended scalping mechanics for quiet sessions.
- **Traceability gaps**: Final output did not consistently expose gates, top setup context, and decision notes in a single structured payload; summary logs were verbose and scattered.

## Why this caused unreliable entries
- **Noise domination**: When zone lists balloon, the probability of a “candidate” zone near price increases even if it is stale or low-quality. This is a classic false-positive driver for SMC-based scalping.
- **False retest rejection**: Excessive touch counts on long windows caused valid retests to be marked as “TOO_MANY_TOUCHES,” which paradoxically allows weaker zones to pass when counting is inconsistent.
- **Incoherent gating**: Low ADX + VERY_LOW RVOL in a low-activity session should bias toward NONE unless there is a high-quality, fresh retest. The prior pipeline could still allow BUY/SELL without a strong “exceptional setup” threshold.

## Fixes implemented
- **Zone explosion control**: Added recency, size (ATR-based), and distance filters for FVGs and OBs, with deterministic ranking and caps (top-K by relevance score). Filters automatically tighten when raw detection exceeds caps.
- **Retest cooldown**: Added a cooldown between touches to avoid machine-gun counting and over-penalization.
- **Regime-aware scoring**: Trend, FVG, and OB components are dampened in weak ADX or low RVOL regimes so low-quality market states do not generate outsized scores.
- **Low-liquidity integrity gate**: If RVOL is VERY_LOW and session activity is LOW, signals are blocked unless the setup score is exceptional; otherwise confidence is penalized.
- **Best setup selector**: Added deterministic setup scoring (retest quality, freshness, proximity, displacement, trend alignment, liquidity) with top-K runner-ups for debugging.
- **Traceability**: Added structured `analysis_trace` payload and a concise summary log that explains signal, gates, and top setup.

## New configuration keys (defaults)
- `SMC_MIN_CANDLES`: 300
- `SMC_MAX_FVG_COUNT`: 50
- `SMC_MAX_OB_COUNT`: 25
- `SMC_MAX_FVG_AGE_BARS`: 240
- `SMC_MAX_OB_AGE_BARS`: 240
- `SMC_MIN_FVG_SIZE_ATR`: 0.15
- `SMC_MIN_OB_SIZE_ATR`: 0.20
- `SMC_ZONE_MAX_DIST_ATR`: 3.0
- `SMC_ZONE_TIGHTEN_MULT`: 1.2
- `SMC_FVG_RANK_WEIGHTS`: `{size:0.35,strength:0.35,recency:0.2,proximity:0.1}`
- `SMC_OB_RANK_WEIGHTS`: `{strength:0.45,recency:0.25,proximity:0.2,size:0.1}`
- `INTEGRITY_LOW_LIQUIDITY_RVOL_MAX`: 0.55
- `INTEGRITY_LOW_LIQUIDITY_SESSION_ACTIVITY`: `LOW`
- `INTEGRITY_LOW_LIQUIDITY_FORCE_NONE`: true
- `INTEGRITY_EXCEPTIONAL_SETUP_SCORE`: 0.78
- `REGIME_ADX_WEAK_MAX`: 18.0
- `REGIME_TREND_WEIGHT_MULT_LOW_ADX`: 0.6
- `REGIME_FVG_WEIGHT_MULT_LOW_RVOL`: 0.7
- `REGIME_OB_WEIGHT_MULT_LOW_RVOL`: 0.7
- `FLOW_TOUCH_COOLDOWN_BARS`: 3
- `FLOW_SETUP_DISPLACEMENT_ATR_TARGET`: 1.0
- `FLOW_SETUP_TOP_K`: 3
- `FLOW_SETUP_WEIGHTS`: `{retest_quality:0.25,freshness:0.2,proximity:0.2,displacement:0.15,trend_alignment:0.1,liquidity:0.1}`
