# NDS Flow Scalper – MT5 Strategy Tester Bridge (Phase 1–4)

## Phase 1 – Architecture Decision

### Selected Option: **Memory-Mapped File (MMF) + Event Signaling**

**Justification**
- **Strategy Tester compatible**: Uses WinAPI (`CreateFileMapping`, `MapViewOfFile`, `CreateEvent`) available from MQL5 without external DLLs, avoiding Tester restrictions around socket access and external processes.
- **Deterministic latency**: Shared memory avoids kernel socket overhead and keeps round-trip latency bounded by memory copy + event signaling.
- **Low operational risk**: No dependency on Python embedding inside MT5 or third-party backtest frameworks.
- **Cross-process separation**: Python engine remains unchanged; only an adapter reads MMF requests and returns commands.

### Latency Expectations
- **In-process MT5 → MMF write**: ~5–20 μs
- **Event signal + Python wake-up**: ~50–200 μs typical (depends on OS scheduling)
- **Python decision time**: dominated by your strategy logic
- **Target end-to-end** (per tick): **<1 ms** excluding strategy compute time.

### Failure Modes & Recovery
| Failure Mode | Symptom | Mitigation |
| --- | --- | --- |
| Python process not running | MT5 timeouts waiting for response | Timeout + log warning; EA continues without trading until reconnection. |
| Stale response (sequence mismatch) | Ignored command | Sequence ID validation; EA logs mismatch and waits for next tick. |
| Memory corruption | Invalid header | Python ignores invalid packets; EA re-sends next tick. |
| Trade execution failure | MT5 retcode != DONE | Log to CSV and MT5 journal; decision loop continues. |

### Synchronization Model
- **Blocking, event-driven**
  - EA writes request snapshot → signals `REQ` event → waits for `RESP` event.
  - Python adapter blocks waiting for request sequence; writes response and signals `RESP`.
- **Deadlock prevention**
  - EA waits with `DecisionTimeoutMs` (default 500 ms). Timeouts are logged, and trading resumes on next tick.

## Phase 2 – MQL5 Bridge Agent (NDS_Bridge_Agent.mq5)

### Responsibilities
- Runs inside Strategy Tester (Agent/Visual mode compatible).
- Collects tick/bar market state on each trigger.
- Sends serialized snapshot to MMF.
- Polls for execution command and executes using standard MT5 trade APIs.
- Logs execution results to `Common` CSV for audit.

### Key Notes
- Symbol fixed to `XAUUSD` by default (input override if needed).
- Uses pip/point math for XAUUSD (0.1 pip, 0.01 point). Spread is passed in raw price units.
- Handles **Market**, **Limit**, and **Stop** order types.

## Phase 3 – Python Connector (Engine Adapter)

### Adapter Contract
The Python adapter reads `MarketSnapshot` and must return `ExecutionCommand`:

```json
{
  "signal": "BUY | SELL | NONE",
  "entry": 0.0,
  "sl": 0.0,
  "tp": 0.0,
  "volume": 0.0,
  "confidence": 0.0,
  "reason_codes": []
}
```

### Blocking Synchronization
The MMF bridge loop blocks until a valid request sequence appears, then synchronously returns a command. MT5 waits on each tick (or bar) for the decision, yielding deterministic replay behavior.

## Phase 4 – Integration & Validation

### Mandatory Tests
1. **Tick-by-tick stress test**: Run Strategy Tester in "Every Tick" mode with high logging.
2. **High-volatility spikes**: Replay historical spikes; validate no missed commands.
3. **Failure simulation**: Stop Python bridge mid-test; confirm EA logs timeout but remains alive.
4. **Mode coverage**: Run Every Tick, 1-Minute OHLC, and Visual Mode.

### Acceptance Criteria
- Zero missed trade commands (sequence monotonic, response matching).
- No Strategy Tester freeze or deadlock (timeouts are respected).
- Execution prices align with tester execution model.
- Logs reproducible and auditable.

## Integration Guide

### 1) Place the EA
- Copy `scripts/bridge/NDS_Bridge_Agent.mq5` into:
  - `MQL5/Experts/NDS/` (recommended)
- Compile in MetaEditor.

### 2) Start the Python Bridge
```bash
python scripts/bridge/run_mmf_bridge.py
```
Replace `dummy_decision` with your engine adapter call. The adapter must return an `ExecutionCommand`.

### 3) Wire the Existing Python Engine
Example skeleton:

```python
from src.bridge.mmf_protocol import ExecutionCommand


def decision_adapter(snapshot):
    # Call your existing engine (black box)
    engine_result = engine.evaluate(snapshot)
    return ExecutionCommand(
        action=engine_result["signal"],
        entry=engine_result["entry"],
        sl=engine_result["sl"],
        tp=engine_result["tp"],
        volume=engine_result["volume"],
        confidence=engine_result.get("confidence", 0.0),
        reason_codes=engine_result.get("reason_codes", []),
    )
```

### 4) Strategy Tester Settings
- Symbol: **XAUUSD**
- Timeframes: **M5 / M15**
- Model: **Every Tick** (preferred) or 1-Minute OHLC
- Visual Mode: optional; should behave identically

### 5) Logs & Audit
- MT5 CSV log: `Common/Files/nds_bridge_exec_log.csv`
- Python log: stdout JSON snapshot + decision

---

## Delivered Components
- **MQL5 EA**: `scripts/bridge/NDS_Bridge_Agent.mq5`
- **Python bridge**: `src/bridge/mmf_bridge.py`, `src/bridge/mmf_protocol.py`
- **Demo runner**: `scripts/bridge/run_mmf_bridge.py`
