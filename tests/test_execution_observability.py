from datetime import datetime, timedelta

from src.trading_bot.bot import NDSBot
from src.trading_bot.trade_tracker import TradeTracker
from src.trading_bot.contracts import PositionContract


def test_tp_pips_sanity():
    pips = NDSBot._compute_tp_pips(entry_price=2000.0, tp_price=2005.0, point_size=0.01)
    assert pips == 50.0
    assert pips < 1000.0


def test_close_metadata_propagates_from_entry_snapshot():
    entry_snapshot = {
        "risk": {
            "tp_execution_mode": "TP1_PARTIAL_MANAGED",
            "tp_sent_to_broker": 0.0,
            "tp1": 2005.0,
            "tp2": 2010.0,
            "rr_validate_mode": "TP2_ONLY",
            "rr_checked": "TP2",
            "min_rr_source": "scalp.SCALP_TP1_ONLY_MIN_RR",
        }
    }
    close_metadata = NDSBot._build_close_metadata(
        open_metadata={},
        entry_snapshot=entry_snapshot,
        tp_level_hit="TP1",
    )

    assert close_metadata["tp_execution_mode"] == "TP1_PARTIAL_MANAGED"
    assert close_metadata["tp_sent_to_broker"] == 0.0
    assert close_metadata["tp1_price"] == 2005.0
    assert close_metadata["tp2_price"] == 2010.0
    assert close_metadata["rr_validate_mode"] == "TP2_ONLY"
    assert close_metadata["rr_checked"] == "TP2"
    assert close_metadata["min_rr_source"] == "scalp.SCALP_TP1_ONLY_MIN_RR"


def test_trade_tracker_fallback_maps_pending_trade_metadata():
    tracker = TradeTracker()
    opened_at = datetime.utcnow()
    tracker.add_trade_open(
        {
            "event_type": "OPEN",
            "event_time": opened_at,
            "symbol": "XAUUSD",
            "order_ticket": 101,
            "position_ticket": None,
            "side": "BUY",
            "volume": 1.0,
            "entry_price": 2000.0,
            "exit_price": None,
            "sl": 1995.0,
            "tp": 0.0,
            "profit": None,
            "pips": None,
            "pips_abs": None,
            "reason": None,
            "metadata": {"tp1_price": 2005.0},
        }
    )

    open_positions: list[PositionContract] = [
        {
            "position_ticket": 555,
            "symbol": "XAUUSD",
            "side": "BUY",
            "volume": 1.0,
            "entry_price": 2000.0,
            "current_price": 2000.0,
            "sl": 1995.0,
            "tp": 0.0,
            "profit": 0.0,
            "magic": 0,
            "comment": "broker_comment",
            "open_time": opened_at + timedelta(seconds=5),
            "update_time": opened_at + timedelta(seconds=5),
        }
    ]

    tracker.reconcile_with_open_positions(open_positions, reconcile_time=opened_at + timedelta(seconds=10))
    assert 555 in tracker.active_trades
    metadata = tracker.active_trades[555]["open_event"]["metadata"]
    assert metadata.get("tp1_price") == 2005.0
