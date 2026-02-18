from datetime import datetime

from src.trading_bot.bot import NDSBot


class DummyMT5:
    connected = False

    def __init__(self, *args, **kwargs):
        pass


class _TrackerOK:
    def reconcile_with_pending_orders(self, _orders):
        return True, {"orphan_intents": [], "missing_intents": []}


class _TrackerMismatch:
    def reconcile_with_pending_orders(self, _orders):
        return False, {"orphan_intents": [11], "missing_intents": [22]}


def _make_bot():
    bot = NDSBot(DummyMT5)
    bot.trade_tracker = _TrackerOK()
    bot.bot_state.last_trade_candle_time = datetime(2026, 1, 1, 0, 0, 0)
    bot.bot_state.last_trade_direction = "BUY"
    return bot


def test_blocks_when_open_and_pending_same_direction(monkeypatch):
    bot = _make_bot()
    monkeypatch.setattr(
        bot,
        "get_open_positions_info",
        lambda: [{"side": "BUY", "position_ticket": 1, "open_time": datetime(2026, 1, 1, 0, 0, 0)}],
    )
    monkeypatch.setattr(
        bot,
        "get_pending_orders_info",
        lambda: [
            {"ticket": 10, "type": "BUY_STOP"},
            {"ticket": 11, "type": "BUY_LIMIT"},
            {"ticket": 12, "type": "BUY_STOP"},
        ],
    )

    allowed, reason = bot._can_execute_trade(direction="BUY", symbol="XAUUSD", df=None)
    assert not allowed
    assert reason == "OPEN_POSITION"


def test_blocks_when_only_pending_same_direction(monkeypatch):
    bot = _make_bot()
    monkeypatch.setattr(bot, "get_open_positions_info", lambda: [])
    monkeypatch.setattr(bot, "get_pending_orders_info", lambda: [{"ticket": 99, "type": "BUY_STOP"}])

    allowed, reason = bot._can_execute_trade(direction="BUY", symbol="XAUUSD", df=None)
    assert not allowed
    assert reason == "PENDING_ORDER"


def test_blocks_when_cooldown_not_passed_without_exposure(monkeypatch):
    bot = _make_bot()
    times = [datetime(2026, 1, 1, 0, i * 5, 0) for i in range(4)]

    class _Series:
        def __init__(self, values):
            self._values = values

        def __gt__(self, other):
            return _Series([v > other for v in self._values])

        def sum(self):
            return sum(1 for v in self._values if v)

        @property
        def iloc(self):
            return self

        def __getitem__(self, idx):
            return self._values[idx]

    class _DF:
        empty = False

        def __getitem__(self, key):
            assert key == "time"
            return _Series(times)

    monkeypatch.setattr(bot, "get_open_positions_info", lambda: [])
    monkeypatch.setattr(bot, "get_pending_orders_info", lambda: [])
    bot.config.update_setting("trading_rules.MIN_CANDLES_BETWEEN_TRADES", 3)
    bot.bot_state.last_trade_candle_time = times[-2]
    bot.bot_state.last_trade_direction = "BUY"

    allowed, reason = bot._can_execute_trade(direction="BUY", symbol="XAUUSD", df=_DF())
    assert not allowed
    assert reason == "COOLDOWN_BLOCKED"


def test_allows_exactly_one_when_no_exposure_and_cooldown_passed(monkeypatch):
    bot = _make_bot()
    times = [datetime(2026, 1, 1, 0, i * 5, 0) for i in range(10)]

    class _Series:
        def __init__(self, values):
            self._values = values

        def __gt__(self, other):
            return _Series([v > other for v in self._values])

        def sum(self):
            return sum(1 for v in self._values if v)

        @property
        def iloc(self):
            return self

        def __getitem__(self, idx):
            return self._values[idx]

    class _DF:
        empty = False

        def __getitem__(self, key):
            assert key == "time"
            return _Series(times)

    monkeypatch.setattr(bot, "get_open_positions_info", lambda: [])
    monkeypatch.setattr(bot, "get_pending_orders_info", lambda: [])
    bot.config.update_setting("trading_rules.MIN_CANDLES_BETWEEN_TRADES", 3)
    bot.bot_state.last_trade_candle_time = times[1]
    bot.bot_state.last_trade_direction = "BUY"

    allowed, reason = bot._can_execute_trade(direction="BUY", symbol="XAUUSD", df=_DF())
    assert allowed
    assert reason == "ALLOWED"


def test_blocks_on_intent_mismatch(monkeypatch):
    bot = _make_bot()
    bot.trade_tracker = _TrackerMismatch()
    monkeypatch.setattr(bot, "get_open_positions_info", lambda: [])
    monkeypatch.setattr(bot, "get_pending_orders_info", lambda: [{"ticket": 22, "type": "SELL_STOP"}])

    allowed, reason = bot._can_execute_trade(direction="BUY", symbol="XAUUSD", df=None)
    assert not allowed
    assert reason == "INTENT_MISMATCH"
