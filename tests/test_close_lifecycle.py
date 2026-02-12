from datetime import datetime, timedelta
from pathlib import Path

from src.trading_bot.bot import NDSBot
from src.trading_bot.execution_reporting import generate_execution_report


class DummyRiskManager:
    def _get_gold_spec(self, _key):
        return 0.01


class DummyTracker:
    def __init__(self):
        self.closed_events = []
        self.unknown_events = []

    def close_trade_event(self, event):
        self.closed_events.append(event)

    def finalize_unknown_close(self, position_ticket, event):
        self.unknown_events.append((position_ticket, event))


class DummyNotifier:
    def __init__(self):
        self.close_calls = []

    def send_trade_close_notification(self, **kwargs):
        self.close_calls.append(kwargs)


def _build_bot(monkeypatch):
    bot = NDSBot.__new__(NDSBot)
    bot.config = {}
    bot.risk_manager = DummyRiskManager()
    bot.trade_tracker = DummyTracker()
    bot.notifier = DummyNotifier()
    events = []

    def _capture_report(*, logger, event, df=None):
        events.append(event)

    monkeypatch.setattr("src.trading_bot.bot.generate_execution_report", _capture_report)
    return bot, events


def _sample_record():
    opened_at = datetime.utcnow() - timedelta(minutes=10)
    return {
        "trade_identity": {
            "order_ticket": 501,
            "position_ticket": 9001,
            "symbol": "XAUUSD",
            "opened_at": opened_at,
        },
        "open_event": {
            "side": "BUY",
            "volume": 1.0,
            "entry_price": 2000.0,
            "sl": 1990.0,
            "tp": 2010.0,
            "metadata": {
                "tp1_price": 2005.0,
                "tp2_price": 2010.0,
                "tp_execution_mode": "TP1_PARTIAL_MANAGED",
                "partial_close_count": 1,
                "remaining_volume_after_tp1": 0.5,
            },
        },
        "last_update_event": {"metadata": {"current_price": 2009.0}},
    }


def test_emit_position_closed_event_close_path(monkeypatch):
    bot, reported_events = _build_bot(monkeypatch)

    event = bot._emit_position_closed_event(
        position_ticket=9001,
        record=_sample_record(),
        history={
            "exit_price": 2010.0,
            "total_profit": 100.0,
            "close_time": datetime.utcnow(),
            "reason": "TP",
        },
        now=datetime.utcnow(),
        symbol_fallback="XAUUSD",
        close_status="CLOSE",
    )

    assert event["event_type"] == "CLOSE"
    assert bot.trade_tracker.closed_events
    assert not bot.trade_tracker.unknown_events
    assert reported_events and reported_events[0]["event_type"] == "CLOSE"
    assert bot.notifier.close_calls and bot.notifier.close_calls[0]["reason"] == "TP"


def test_emit_position_closed_event_timeout_fallback(monkeypatch):
    bot, reported_events = _build_bot(monkeypatch)

    event = bot._emit_position_closed_event(
        position_ticket=9001,
        record=_sample_record(),
        history={"reason": "HistoryTimeout/Unknown"},
        now=datetime.utcnow(),
        symbol_fallback="XAUUSD",
        close_status="CLOSE_UNKNOWN",
        close_reason="HistoryTimeout/Unknown",
    )

    assert event["event_type"] == "CLOSE_UNKNOWN"
    assert not bot.trade_tracker.closed_events
    assert bot.trade_tracker.unknown_events and bot.trade_tracker.unknown_events[0][0] == 9001
    assert reported_events and reported_events[0]["event_type"] == "CLOSE_UNKNOWN"
    assert bot.notifier.close_calls and bot.notifier.close_calls[0]["reason"] == "HistoryTimeout/Unknown"


def test_generate_execution_report_for_close_unknown_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class DummyReportGenerator:
        def __init__(self, output_dir="trade_reports/scalping_reports"):
            self.output_dir = Path(output_dir)

        def generate_close_report(self, event, base_filename=None):
            target = self.output_dir / "summary" / f"{base_filename}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("close", encoding="utf-8")
            return str(target)

        def update_daily_summary(self, event):
            day = (event.get("event_time")).strftime("%Y-%m-%d")
            target = self.output_dir / "daily" / f"daily_{day}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("src.reporting.report_generator.ReportGenerator", DummyReportGenerator)

    event = {
        "event_type": "CLOSE_UNKNOWN",
        "event_time": datetime(2026, 1, 2, 10, 30, 0),
        "symbol": "XAUUSD",
        "order_ticket": 501,
        "position_ticket": 9001,
        "side": "BUY",
        "volume": 1.0,
        "entry_price": 2000.0,
        "exit_price": 1998.0,
        "sl": 1990.0,
        "tp": 2010.0,
        "profit": -20.0,
        "pips": -20.0,
        "pips_abs": 20.0,
        "reason": "HistoryTimeout/Unknown",
        "metadata": {"tp_execution_mode": "TP1_PARTIAL_MANAGED"},
    }

    class L:
        def warning(self, *_a, **_k):
            return None

        def info(self, *_a, **_k):
            return None

        def error(self, *_a, **_k):
            return None

        def debug(self, *_a, **_k):
            return None

    generate_execution_report(logger=L(), event=event)

    summary_file = Path("reports/2026-01-02/trades/9001/summary.json")
    close_report = list(Path("trade_reports/scalping_reports/summary").glob("XAUUSD_CLOSE_9001_*.txt"))
    daily_json = Path("trade_reports/scalping_reports/daily/daily_2026-01-02.json")

    assert summary_file.exists()
    assert close_report, "expected close summary in scalping_reports/summary"
    assert daily_json.exists()

