# NDS Flow Scalper MT5 Log Schema

Use these structured tags to align MT5 Strategy Tester logs with Python debug logs.

## Bar Context
```
[NDS][BAR] time=YYYY.MM.DD HH:MM close=...
```

## SMC Analysis
```
[NDS][SMC] breaker_found=true|false ifvg_found=true|false
```

## Filters
```
[NDS][FILTER] spread_ok=true|false spread=... max=...
[NDS][FILTER] session_ok=true|false cooldown_minutes=...
```

## Signal
```
[NDS][SIGNAL] signal=BUY|SELL|NONE reasons=... zone=...
```

## Execution (Phase 2)
```
[NDS][EXEC] order=... price=... sl=... tp=... deviation=...
```

## Risk Manager (Phase 3)
```
[NDS][RM] tp1_hit=true|false partial_closed=... sl_to_be=... atr_trail=...
```
