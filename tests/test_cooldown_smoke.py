from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading_bot.cooldown import evaluate_cooldown


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


def _build_df(start: datetime, bars: int = 10) -> _FakeDataFrame:
    times = [start + timedelta(minutes=5 * i) for i in range(bars)]
    return _FakeDataFrame(times)


def run_smoke() -> None:
    df = _build_df(datetime(2024, 1, 1, 0, 0, 0), bars=10)
    last_trade_time = df["time"].iloc[-2]

    decision = evaluate_cooldown(
        signal="BUY",
        min_candles_between=4,
        df=df,
        open_positions=[],
        last_trade_candle_time=last_trade_time,
        last_trade_direction="BUY",
    )
    assert not decision.allowed and decision.reason == "COOLDOWN_BLOCKED"

    decision = evaluate_cooldown(
        signal="SELL",
        min_candles_between=4,
        df=df,
        open_positions=[],
        last_trade_candle_time=last_trade_time,
        last_trade_direction="BUY",
    )
    assert decision.allowed

    open_positions = [
        {"side": "BUY", "open_time": last_trade_time, "position_ticket": 101},
        {"side": "BUY", "open_time": last_trade_time, "position_ticket": 102},
    ]
    decision = evaluate_cooldown(
        signal="BUY",
        min_candles_between=4,
        df=df,
        open_positions=open_positions,
        last_trade_candle_time=None,
        last_trade_direction=None,
    )
    assert not decision.allowed and decision.reason == "COOLDOWN_BLOCKED"

    decision = evaluate_cooldown(
        signal="SELL",
        min_candles_between=4,
        df=df,
        open_positions=open_positions,
        last_trade_candle_time=None,
        last_trade_direction=None,
    )
    assert decision.allowed

    mixed_positions = [
        {"side": "BUY", "open_time": last_trade_time, "position_ticket": 201},
        {"side": "SELL", "open_time": last_trade_time, "position_ticket": 202},
    ]
    decision = evaluate_cooldown(
        signal="BUY",
        min_candles_between=4,
        df=df,
        open_positions=mixed_positions,
        last_trade_candle_time=None,
        last_trade_direction=None,
    )
    assert not decision.allowed and decision.reason == "MIXED_EXPOSURE"

    print("✅ Cooldown smoke tests passed.")


if __name__ == "__main__":
    run_smoke()
