from datetime import datetime, timezone

from src.trading_bot.cooldown import evaluate_cooldown
from src.trading_bot.state import BotState


class _FakeSeries:
    def __init__(self, values):
        self._values = list(values)

    def __gt__(self, other):
        return _FakeSeries([value > other for value in self._values])

    def sum(self):
        return sum(1 for value in self._values if value)

    @property
    def iloc(self):
        return self

    def __getitem__(self, idx):
        return self._values[idx]


class _FakeDataFrame:
    def __init__(self, times):
        self._series = _FakeSeries(times)
        self.empty = len(times) == 0

    def __getitem__(self, key):
        if key != "time":
            raise KeyError(key)
        return self._series


def test_cooldown_handles_naive_candles_and_aware_last_trade():
    # Naive candle times (legacy/backtest shape)
    candles = [
        datetime(2026, 2, 16, 7, 20, 0),
        datetime(2026, 2, 16, 7, 25, 0),
        datetime(2026, 2, 16, 7, 30, 0),
        datetime(2026, 2, 16, 7, 35, 0),
    ]
    df = _FakeDataFrame(candles)

    # Aware last-trade time (live UTC path)
    open_positions = [
        {
            "side": "SELL",
            "open_time": datetime(2026, 2, 16, 7, 25, 0, tzinfo=timezone.utc),
            "position_ticket": 89361790,
        }
    ]

    decision = evaluate_cooldown(
        signal="SELL",
        min_candles_between=2,
        df=df,
        open_positions=open_positions,
        last_trade_candle_time=None,
        last_trade_direction=None,
    )

    assert decision.allowed
    assert decision.reason == "COOLDOWN_OK"


def test_bot_state_normalizes_times_to_utc():
    state = BotState()
    state.set_last_analysis("2026-02-16T07:25:11Z")
    state.set_last_trade_times(
        wall_time=datetime(2026, 2, 16, 9, 25, 11),
        candle_time="2026-02-16T07:25:00+00:00",
    )

    assert state.last_analysis is not None
    assert state.last_analysis.utcoffset().total_seconds() == 0
    assert state.last_trade_wall_time is not None
    assert state.last_trade_wall_time.utcoffset().total_seconds() == 0
    assert state.last_trade_candle_time is not None
    assert state.last_trade_candle_time.utcoffset().total_seconds() == 0
