# FULL ARCHITECTURE RE-AUDIT (Production Capital Preservation Lens)

## Scope and evidence baseline
This re-audit is based on execution-critical code paths:
- `ScalpingRiskManager.finalize_order(...)` (single biggest decision chokepoint, ~1375 LOC).
- `NDSBot.execute_scalping_trade(...)` (execution orchestration chokepoint, ~730 LOC).
- `NDSBot._monitor_open_trades(...)` + `_emit_position_closed_event(...)` (close attribution and reconciliation).
- `PositionManager` FSM (`position_manager_state_machine.py`) for TP1/TP2 lifecycle.
- Analyzer momentum entry assignment for `STOP` generation.

---

## SECTION 1 — Architecture diagnosis

## 1) Full system map (actual runtime)

### Analyzer layer
- Analyzer emits directional signal + `entry_model` (`MARKET`/`LIMIT`/`STOP`) and entry context.
- Momentum logic explicitly creates `STOP` entries when trigger not yet crossed, and flips to `MARKET` if already crossed.

### Decision layer
- `finalize_order(...)` performs almost all hard decisioning in one function: entry type transformations, regime policy, SR/countertrend gating, SL/TP synthesis, RR mode selection, RR repair/autogen, sizing, invariants.
- This is a central monolith with intertwined policy + geometry + validation + formatting responsibility.

### Risk manager
- Risk guardrails exist in both `bot.py` and `risk_manager.py`:
  - Exposure/cooldown/intents in bot.
  - Session, limits, risk limits in `can_scalp(...)`.
  - Geometry/RR/ATR/spread/deviation in `finalize_order(...)`.
- Net effect: layered protection exists, but with non-trivial overlap and rejection bias.

### Execution layer
- Bot decides final send path.
- STOP orders are staged through breakout-watch; if staging fails, forced fallback to market.
- Even if STOP reaches send branch, it is force-converted to market with warning.

### FSM state management
- Position lifecycle states: OPEN → WAIT_TP1 → WAIT_TP2, terminal close inferred by broker disappearance/history reconciliation.
- TP2 crossing in FSM is informational; terminalization is external (broker close detect), not internal state event.

### Broker reconciliation
- Missing open ticket enters pending-close queue.
- Repeated history lookup with backoff.
- Timeout promotes event to `CLOSE_UNKNOWN`.

### Reporting layer
- `_emit_position_closed_event(...)` merges open snapshot + history and writes report/notifications.
- Close attribution (`TP1/TP2/SL/UNKNOWN`) is inferred from reason strings + price proximity + partial flags.

---

## 2) Structural defects

## Hidden coupling
1. Analyzer `entry_model=STOP` materially changes downstream risk + execution path, but execution layer can silently convert STOP to market; this breaks contract between analyzer intent and realized execution mode.
2. Risk settings and bot-side flow settings both influence TP execution semantics and RR validation context; ownership is split.
3. FSM plan reconstruction depends on metadata integrity from tracker/report pipeline; if metadata is absent, fallback TPs are synthesized from config, not original intent.

## Responsibility leakage
1. `finalize_order` does market microstructure policy, regime policy, geometry building, RR math, position sizing, and result DTO shaping in one method.
2. Bot does additional geometry validation after `finalize_order`, indicating risk manager output is not treated as authoritative.
3. Close-state truth is spread across MT5 open list, history calls, in-memory tracker, and JSON state store.

## Atomic-logic violations
1. Entry atomicity violation: planned order type can mutate multiple times (STOP→LIMIT/WAIT/NONE in risk, then STOP→MARKET in execution fallback).
2. Exit atomicity violation: TP2 hit is not a terminal internal FSM event; closure is later inferred by polling/reconciliation.
3. Risk atomicity violation: RR is pre-gated then post-repaired/re-gated under different contexts, creating branch-dependent math semantics.

## Redundant validations / overlapping filters
1. Exposure + pending intent checked pre-submit and again around submit.
2. Session eligibility appears in analyzer context, `can_scalp`, and bot flow gating.
3. RR checked at pre-approval and final gate; TP1/TP2 mode switching creates multiple rejection paths for same underlying geometry.

## Side effects impacting trade geometry consistency
1. STOP far CAP_ENTRY rewrites entry while SL/TP are later recomputed from rewritten value; execution can still become market, changing practical fill geometry.
2. LIMIT conversion branch in STOP far path may produce geometry that is never executed as intended when later normalization/fallback happens.
3. TP1 virtual trigger branch (min lot constraints) alters lifecycle behavior while preserving two-target semantics in metadata, increasing attribution ambiguity.

---

## 3) Complexity and decomposition hotspots

1. `ScalpingRiskManager.finalize_order`: cc ~253, ~1375 LOC. Critical refactor target.
2. `NDSBot.execute_scalping_trade`: cc ~142, ~730 LOC. Second critical refactor target.
3. `_monitor_open_trades`: cc ~57 and combines reconcile + attribution + notifications.
4. `_build_plan` in FSM: cc ~32 with metadata fallback synthesis.

Recommended decomposition boundaries:
- `finalize_order` → 8 pure steps: `resolve_entry_policy`, `compute_geometry`, `validate_geometry`, `resolve_rr_mode`, `apply_rr_policy`, `size_position`, `enforce_invariants`, `build_decision`.
- `execute_scalping_trade` → preflight / finalize / broker_send / persist_open_event.
- `_monitor_open_trades` → detect_closed / reconcile_history / emit_terminal / handle_partial.

---

## 4) Legacy / dead / low-value branches

1. STOP send branch in execution is effectively legacy: branch logs warning then forces market.
2. STOP conversion-to-LIMIT policy is low-value if STOP flow is not consistently executed as pending intent.
3. `CLOSE_UNKNOWN` path can become overused under transient broker history gaps, weakening analytics quality.
4. Defensive fallback TP synthesis in FSM (`tp1/tp2` from config when metadata missing) hides earlier pipeline defects and may distort close attribution.

---

## SECTION 2 — STOP entry model critical review

## 1) Structural flaws (near/far STOP)
1. STOP near/far is built on distance-to-market thresholds, but realized execution path is not guaranteed to stay STOP.
2. Far STOP policy mixes regime inference + entry-type mutation + confidence logic in a single gate.
3. CAP_ENTRY for trend continuation artificially drags entry toward market while still calling it STOP, reducing breakout confirmation quality.

## 2) Regime misclassification risks
1. Regime routing mostly relies on ADX + volatility tags; this is weak for microstructure transitions and SMC trap conditions.
2. Mean-reversion branch converts to LIMIT based on confidence threshold, which is orthogonal to order-book sweep behavior.
3. Borderline ADX zones create unstable branch switching (trend/mean-reversion/reject) across adjacent candles.

## 3) Entry geometry distortions
1. STOP far CAP_ENTRY rewrites trigger geometry independent of original structure invalidation distance.
2. STOP→LIMIT conversion changes directional logic from continuation to pullback fill without re-deriving structural SL thesis.
3. Downstream market fallback can erase pending semantics after RR and SL/TP were computed for pending entry assumptions.

## 4) RR illusion
1. RR can look valid on planned entry, but real fill (market fallback/slippage) degrades realized RR below gate threshold.
2. TP2-only validation can pass while TP1 path remains weak; realized distribution can still be negative under high spread/slippage.

## 5) Slippage amplification
1. STOP entries in breakout zones are most slippage-sensitive; far STOP increases adverse selection risk.
2. Converting far STOP into urgent market fallback compounds slippage exactly where volatility expansion is highest.

## 6) False breakout vulnerability (SMC view)
1. STOP breakout logic around previous highs/lows is vulnerable to liquidity sweeps (inducement grabs).
2. Without dedicated sweep-then-reclaim confirmation, STOP entries are exposed to engineered wick expansions.
3. CAP_ENTRY and STOP revalidation do not fully model displacement quality vs liquidity void fill behavior.

## 7) Atomic entry logic verdict
- Current STOP model violates atomic entry logic: one signal can traverse incompatible intent classes (`STOP`, `LIMIT`, `WAIT`, `MARKET`) before execution.

## STOP model decision
## **C) Remove entirely**
Given observed loss behavior + architecture inconsistency + execution-path non-determinism, STOP should be removed from live execution path.

## Clean architecture without STOP
1. Analyzer emits only `MARKET` or `LIMIT` intents (`STOP` removed at source).
2. Risk manager no longer contains STOP-far policy, STOP revalidation, STOP→LIMIT conversion.
3. Execution layer has deterministic send path:
   - MARKET intent → market send.
   - LIMIT intent → limit send.
   - No fallback type mutation except explicit reject.
4. Regime-specific behavior affects *allow/reject + sizing*, not order-type mutation.

---

## SECTION 3 — Risk engine validation

## Findings
1. RR definition mostly uses `distance(tp)/distance(sl)` and matches formula, but branch-specific variable selection (`TP1` vs `TP2_ONLY`) causes policy inconsistency.
2. Pre-approval RR gate + final RR gate can reject the same idea twice under slightly different references.
3. RR repair and TP2 autogen can create optimization drift: rejected setup is transformed into acceptable math without renewed structural validation.
4. Countertrend TP1 override + TP2 autogen may overfit RR numerically while weakening execution realism.
5. ATR caps are applied in multiple contexts (entry deviation, RR repair caps), increasing hidden rejection bias.
6. Time-based exits rely on later reconciliation for attribution; timeout closes can still end in `CLOSE_UNKNOWN` on history misses.

## Deterministic RR policy proposal
1. Single RR mode for live: **TP1-anchored only**.
2. Single gate moment: after final entry/sl/tp geometry, before sizing.
3. No RR repair in live path. If RR < min → reject (or downsize not retarget).
4. TP2 optional runner logic must not participate in gate acceptance.
5. Formula fixed globally:
   - `sl_pips = abs(entry - sl) / point`
   - `tp1_pips = abs(tp1 - entry) / point`
   - `rr = tp1_pips / sl_pips`
   - accept iff `rr >= MIN_RR + epsilon` and all distances positive.

---

## SECTION 4 — FSM and position management

## Risks
1. TP2 “reached” is logged but not terminalized internally; closure depends on broker disappearance polling.
2. Broker-driven closure + delayed history can force `CLOSE_UNKNOWN`, reducing auditability.
3. Potential race: timeout force-close and normal broker-close detection can overlap near same cycle.
4. TP2 attribution depends on heuristic resolution from reason text/price tolerance + partial metadata; can be wrong under partial fills/slippage.

## Event-driven FSM redesign
1. Internal events: `OnTp1Hit`, `OnPartialCloseConfirmed`, `OnBeMoveConfirmed`, `OnTp2Placed`, `OnPositionClosed`.
2. Terminal state must be triggered by explicit close confirmation event, not inferred solely by polling absence.
3. Reconciliation loop becomes event-source, not state owner.
4. `CLOSE_UNKNOWN` only after deterministic retries with explicit reason code taxonomy.

---

## SECTION 5 — Performance and file efficiency

## Hotspots and 10x symbol-load concerns
1. `finalize_order` and `execute_scalping_trade` dominate CPU and branch depth.
2. Repeated distance metric recomputation for same (entry/sl/tp) tuples is expensive and error-prone.
3. Logging is verbose in hot loops; at 10x symbols this becomes I/O-bound quickly.
4. `_monitor_open_trades` does multiple history queries with retries; scales poorly without batching or async queueing.

## Efficiency actions
1. Introduce immutable `Geometry` object (entry/sl/tp/tp2/distances/rr precomputed once).
2. Cache point-size and symbol specs per cycle.
3. Collapse repetitive logger statements into structured single-line events with IDs.
4. Move history reconciliation to bounded worker queue with rate limits.

---

## SECTION 6 — Required outputs

## 1) Structured Risk Map (highest first)
1. **Non-deterministic entry intent mutation** (STOP/LIMIT/MARKET cross-mutation).
2. **Monolithic critical functions** causing hidden regressions and test blind spots.
3. **Close attribution fragility** (`CLOSE_UNKNOWN`, TP-level inference ambiguity).
4. **RR policy branch explosion** (pre-gate + repair + mode switching).
5. **Polling-centric lifecycle** prone to timing races and delayed truth.

## 2) Refactoring roadmap (phased)

### Phase 0 (Safety freeze)
- Freeze new strategy features.
- Add golden-path deterministic regression tests for accepted/rejected order outcomes.

### Phase 1 (Execution determinism)
- Remove STOP from analyzer output and risk manager.
- Remove STOP staging/fallback logic in bot.
- Enforce intent immutability: order type cannot change after risk acceptance.

### Phase 2 (Risk core extraction)
- Split `finalize_order` into pure modules.
- Remove live RR repair/autogen; keep optional offline advisory mode only.

### Phase 3 (FSM eventification)
- Convert FSM to event-driven transitions with explicit terminal events.
- Reduce `CLOSE_UNKNOWN` by introducing typed close-resolution workflow.

### Phase 4 (Scale hardening)
- Add reconcile worker queue, bounded retries, and telemetry counters.
- Optimize logging volume for multi-symbol throughput.

## 3) STOP model decision
- **Remove entirely (C)** for live trading architecture.

## 4) Simplified architecture proposal
- Analyzer: structure + regime + deterministic entry intent (`MARKET|LIMIT`).
- Risk Engine (pure): geometry, deterministic RR, sizing, invariants.
- Execution Router: sends exactly accepted intent, no hidden conversion.
- Position FSM: event-driven TP1/TP2/close transitions.
- Reconciler: broker/history event producer.
- Reporter: consumes terminal events only.

## 5) Clean flowchart
```text
[Analyzer]
   -> emits Signal + Intent(MARKET|LIMIT) + Context
      -> [Risk Engine (pure)]
            1) Validate signal
            2) Build Geometry(entry,sl,tp1,tp2?)
            3) RR Gate (single deterministic)
            4) Size + Invariants
            => ACCEPT or REJECT
      -> [Execution Router]
            if ACCEPT: send exact intent
            if REJECT: log typed reason
      -> [Position FSM (event-driven)]
            OnFill -> WAIT_TP1
            OnTP1Confirmed -> WAIT_TP2
            OnCloseConfirmed -> CLOSED
      <- [Broker Reconciler emits events]
      -> [Reporting]
            terminal event -> report + notification
```

## 6) Legacy removal plan
1. Delete `_apply_stop_far_from_market_policy` and all STOP revalidation settings usage.
2. Remove STOP-related config keys:
   - `STOP_MAX_DEVIATION_PIPS`
   - `STOP_HARD_REJECT_PIPS`
   - `STOP_CONVERT_TO_LIMIT_PIPS`
   - `STOP_REVALIDATE_PIPS`
3. Remove execution STOP fallback branches and breakout-watch STOP staging dependency.
4. Remove RR live repair/autogen toggles from execution path; keep as research diagnostics only.
5. Tighten close statuses to deterministic typed set; keep `CLOSE_UNKNOWN` as strict exception state.
