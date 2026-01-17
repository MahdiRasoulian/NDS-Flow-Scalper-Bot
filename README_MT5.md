# MT5 Parity Port (NDS Flow Scalper)

## Compile
1. Open MetaEditor.
2. Add the repo root as a project folder.
3. Compile: `mql5/Experts/NDS_Flow_Scalper_Parity.mq5`.

## Attach in Strategy Tester
1. Open MT5 Strategy Tester.
2. Select the EA: `NDS_Flow_Scalper_Parity`.
3. Use M15 timeframe and the desired symbol.
4. Enable visual mode for chart objects.

## Phase Controls
- Phase 0/1 is analysis-only (no trading).
- Follow `docs/mt5_port/PARITY_CHECKLIST.md` for next steps.

## Logs
See `docs/mt5_port/LOG_SCHEMA.md` for structured log tags used in Strategy Tester.
