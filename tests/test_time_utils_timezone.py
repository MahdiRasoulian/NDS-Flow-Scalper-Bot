from datetime import datetime, timezone

from src.trading_bot.time_utils import parse_timestamp, to_broker_time, to_utc_time
from src.trading_bot.session_policy import evaluate_session


def test_parse_timestamp_iso_z_is_utc_aware():
    ts = parse_timestamp("2026-01-15T10:30:00Z")
    assert ts is not None
    assert ts.tzinfo is not None
    assert ts.utcoffset().total_seconds() == 0


def test_broker_utc_roundtrip_consistency():
    ts_utc = datetime(2026, 1, 15, 8, 0, tzinfo=timezone.utc)
    ts_broker = to_broker_time(ts_utc, offset_hours=2, time_mode="UTC")
    assert ts_broker.hour == 10
    assert ts_broker.utcoffset().total_seconds() == 7200

    roundtrip = to_utc_time(ts_broker, offset_hours=2, time_mode="BROKER")
    assert roundtrip == ts_utc


def test_session_policy_handles_utc_input_with_offset():
    cfg = {
        "trading_settings": {
            "SESSION_POLICY_MODE": "WEIGHT_ONLY",
            "SESSION_TIME_MODE": "UTC",
            "BROKER_UTC_OFFSET_HOURS": 2,
            "SESSION_REQUIRE_BROKER_TIME": False,
            "SESSION_DEFINITIONS": {
                "LONDON": {"start": "10:00", "end": "19:00", "weight": 1.2, "allow_momentum": True},
                "OTHER": {"start": "19:00", "end": "10:00", "weight": 0.4, "allow_momentum": False},
            },
        }
    }
    ts_utc = datetime(2026, 1, 15, 8, 30, tzinfo=timezone.utc)
    decision = evaluate_session(ts_utc, cfg)

    assert decision.session_name == "LONDON"
    assert decision.ts_broker is not None
    assert decision.ts_broker.hour == 10
