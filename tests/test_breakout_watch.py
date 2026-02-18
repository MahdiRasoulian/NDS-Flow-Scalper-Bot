from __future__ import annotations

from src.trading_bot.breakout_watch import (
    BreakoutWatch,
    BreakoutValidationConfig,
    BreakoutWatchManager,
    BreakoutWatchState,
    revalidate_breakout,
)


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None


def _watch(direction: str = "BUY", trigger: float = 2000.0) -> BreakoutWatch:
    return BreakoutWatch(
        direction=direction,
        trigger_price=trigger,
        expiration_candles=3,
        structural_reference={"level": trigger},
        original_score=87.0,
        finalized_payload={"order_type": "stop"},
        signal_snapshot={"signal": direction},
    )


def test_revalidate_breakout_accepts_strong_buy_breakout():
    watch = _watch("BUY", 2000.0)
    state = {
        "rsi": 61,
        "ema_slope": 0.8,
        "atr": 2.1,
        "atr_baseline": 2.0,
        "inside_sr": False,
        "liquidity_sweep": True,
        "breakout_candle": {"open": 1999.5, "close": 2001.4, "high": 2001.6, "low": 1999.4},
    }
    result = revalidate_breakout(watch, state)
    assert result.passed is True
    assert result.reasons == []


def test_revalidate_breakout_rejects_weak_reversal_context():
    watch = _watch("BUY", 2000.0)
    state = {
        "rsi": 44,
        "ema_slope": -0.2,
        "atr": 1.3,
        "atr_baseline": 2.0,
        "inside_sr": True,
        "liquidity_sweep": False,
        "breakout_candle": {"open": 1999.8, "close": 2000.01, "high": 2001.2, "low": 1998.9},
    }
    result = revalidate_breakout(watch, state, cfg=BreakoutValidationConfig(require_liquidity_sweep=True))
    assert result.passed is False
    assert "momentum_not_aligned" in result.reasons
    assert "inside_sr_zone" in result.reasons


def test_watch_manager_expiration_transition():
    mgr = BreakoutWatchManager(_DummyLogger())
    watch = _watch("SELL", 1995.0)
    watch.expiration_candles = 1
    mgr.add(watch)

    mgr.on_new_candle()

    assert watch.state == BreakoutWatchState.EXPIRED
    assert watch.cancelled_reason == "expiration"
