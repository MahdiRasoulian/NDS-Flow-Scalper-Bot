# MT5 Parity Port — Root Cause Analysis & Unified Contracts

## Step 1 — Root Cause Analysis (No Code Yet)

### Why the current EA can compile but still open zero trades
1. **Filters can silently block**
   - Spread/ATR spike/NY cooldown filters can return `false` without any logging, so the EA exits early on every bar.
   - Without explicit logs, it appears to “do nothing” even though the logic is actively rejecting trades.

2. **Signals can be starved by freshness/age rules**
   - IFVG zones can expire quickly if `bars_alive` is incremented without a matching reset or dedup rule.
   - Breaker/IFVG detection can add zones repeatedly, exhausting the fixed array and making new signals unavailable.

3. **Order placement can be rejected silently**
   - Market orders can fail due to invalid volume or stops (e.g., minimum lot or stop level constraints), but no retcode is logged.
   - MQL5 requires `CTrade::ResultRetcode()` and `ResultRetcodeDescription()` logging to see rejections.

4. **Indicator handles can return empty data**
   - ATR/ADX handles can be invalid or the buffer can return 0.0 on insufficient data, leading to zero momentum or displacement checks.

5. **Strategy tester setup mismatch**
   - If the tester uses a different symbol or timeframe without a compatible spread/point size, filters can block.

### File-level contradictions that caused compilation failures
- **Inconsistent method signatures**: `Analyze()` returning a struct in some places but `void` with out-parameters in others.
- **Filters without consistent parameters**: `SpreadOk()` and `SessionOk()` expecting a logger in some declarations but not others.
- **Logger ownership mismatch**: Some classes expecting a reference `CLogger&`, others a pointer `CLogger*`.

### “Modern C++” patterns that are not MQL5-safe
- Use of unsupported STL containers or patterns.
- Overreliance on references/pointers, which can lead to parsing errors in MetaEditor.
- C++ style headers (`#pragma once`) and templates not supported consistently.

**Replacement plan**
- Use MQL5 arrays and simple structs.
- Use include guards where headers are required.
- Prefer plain classes with explicit `Init/Update` methods.

## Step 2 — Unified Contract (One API for All Modules)

### Logger Design (single approach)
- **Store a logger pointer inside each class** and set via `SetLogger(CLogger *logger)`.

### Contracts

#### CLogger
- `void Info(string msg);`
- `void Warn(string msg);`
- `void Error(string msg);`

#### CFilters
- `void SetLogger(CLogger *logger);`
- `bool SpreadOk(double max_spread_pips);`
- `bool SessionOk(int ny_open_hour, int ny_open_minute, int cooldown_minutes);`
- `bool AtrSpikeOk(double atr, double atr_avg, double spike_mult);`

#### CSMC_Analyzer (Signal Engine)
- `void SetLogger(CLogger *logger);`
- `void SetSymbol(string symbol, ENUM_TIMEFRAMES tf);`
- `void UpdateIndicators();`
- `void UpdateZones();`
- `bool GetSignal(SSignal &signal_out);`

#### CRiskManager
- `void SetLogger(CLogger *logger);`
- `double PipSize();`
- `double CalcVolume(double sl_pips, double risk_percent);`
- `void BuildLevels(ENDSDirection dir, double entry, double sl_pips, double tp1_pips, double &sl_out, double &tp1_out);`

#### CTradeExecutor
- `void SetLogger(CLogger *logger);`
- `bool OpenPosition(const SSignal &signal, double sl, double tp1, double volume);`
- `void ManagePosition(double tp1_price, double trailing_atr_mult);`
- `bool HasPosition();`

---

**Next step**: Implement logging and rewire EA to use these contracts consistently.
