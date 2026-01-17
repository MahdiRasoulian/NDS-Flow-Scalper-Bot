# NDS Flow Scalper MT5 Parity Checklist

## Phase 0 — Scaffolding & Data Layer
- [x] EA skeleton and OOP structure in `NDS_Flow_Scalper_Parity.mq5`
- [x] `CBarSeries` for rate caching with bounded `CopyRates`
- [x] `CIndicatorCache` for ATR handle and on-bar updates
- [x] `CLogger` structured output
- [x] `IsNewBar()` gating for M15 bar close

## Phase 1 — Signal Parity (No Trades)
- [x] SMC module scaffolding with breaker + IFVG detection
- [x] Momentum module with ADX handle
- [x] Filters: spread + session cooldown
- [x] Single `SignalResult` struct with zone metadata
- [x] Chart visualization: rectangles + arrows

## Phase 2 — Execution (Controlled)
- [ ] `CTradeExecutor` with slippage/deviation + magic number tagging
- [ ] Order send with hedging/netting support
- [ ] Execute only after bar-close signal

## Phase 3 — Risk Manager Parity
- [ ] SL/TP computation parity with Python clamps
- [ ] TP1 partial close (50%) + SL to BE
- [ ] ATR trailing for runner
- [ ] Lot-step handling + explicit logging

## Parity Validation Steps
- [ ] Compare `LOG_SCHEMA.md` outputs with Python logs line-by-line
- [ ] Verify signals match for identical historical data
- [ ] Confirm no trades in Phase 1
- [ ] Enable execution only after signal parity confirmed
