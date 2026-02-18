from datetime import datetime, timedelta

from src.trading_bot.bot import NDSBot


class DummyConfig(dict):
    def get(self, key, default=None):
        if key == "trading_settings.TIMEFRAME":
            return "M5"
        return super().get(key, default)


class DummyMT5:
    def __init__(self):
        self.calls = []

    def close_position(self, ticket: int, volume: float = None):
        self.calls.append({"ticket": ticket, "volume": volume})
        return {"success": True, "ticket": ticket, "volume": volume}


class DummyTracker:
    def __init__(self):
        self.active_trades = {
            1001: {
                "trade_identity": {"position_ticket": 1001, "opened_at": datetime.utcnow() - timedelta(minutes=90)},
                "open_event": {"metadata": {}},
                "last_update_event": {},
                "close_event": {},
                "status": "OPEN",
            }
        }
        self.pending = []

    def normalize_trade_record(self, record):
        return record

    def register_pending_close(self, position_ticket, record, detected_time):
        self.pending.append((position_ticket, detected_time))
        return True


def test_enforce_time_based_exits_for_m5_forces_close_after_60_min():
    bot = NDSBot.__new__(NDSBot)
    bot.config = DummyConfig({"risk_settings": {"POSITION_TIMEOUT_MINUTES": 60}})
    bot.mt5_client = DummyMT5()
    bot.trade_tracker = DummyTracker()

    now = datetime.utcnow()
    open_positions = [
        {
            "position_ticket": 1001,
            "symbol": "XAUUSD",
            "side": "BUY",
            "volume": 0.5,
            "open_time": now - timedelta(minutes=75),
        }
    ]

    bot._enforce_time_based_exits(open_positions, now)

    assert bot.mt5_client.calls, "expected timeout enforcement to force close overdue M5 trade"
    assert bot.mt5_client.calls[0]["ticket"] == 1001
    assert bot.trade_tracker.pending and bot.trade_tracker.pending[0][0] == 1001
