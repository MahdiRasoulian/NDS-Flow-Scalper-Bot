from src.trading_bot.bot import NDSBot


class DummyMT5Client:
    def __init__(self):
        self.connected = True
        self.disconnect_calls = 0

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False


def test_cleanup_idempotent_and_single_disconnect():
    bot = NDSBot(mt5_client_cls=DummyMT5Client)
    bot.mt5_client = DummyMT5Client()
    monitor_calls = {"count": 0}

    def _fake_monitor(force=False):
        monitor_calls["count"] += 1

    bot._maybe_monitor_trades = _fake_monitor

    bot.cleanup()
    bot.cleanup()

    assert bot.mt5_client.disconnect_calls == 1
    assert monitor_calls["count"] == 1
    assert bot._cleanup_done is True
