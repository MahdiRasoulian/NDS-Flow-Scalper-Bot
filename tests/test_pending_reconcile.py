from datetime import datetime, timedelta

from src.trading_bot.position_manager import PositionManager
from src.trading_bot.trade_tracker import TradeTracker


class DummyMT5:
    def __init__(self):
        self.modified = []
        self.closed = []

    def modify_position(self, ticket: int, new_sl: float = None, new_tp: float = None):
        payload = {"ticket": ticket, "new_sl": new_sl, "new_tp": new_tp}
        self.modified.append(payload)
        return {"success": True, **payload}

    def close_position(self, ticket: int, volume: float = None, comment: str = ""):
        payload = {"ticket": ticket, "volume": volume, "comment": comment, "price": 2005.0}
        self.closed.append(payload)
        return {"success": True, **payload}


def test_pending_trigger_reconciles_metadata_and_manages_tp1():
    tracker = TradeTracker()
    opened_at = datetime.utcnow() - timedelta(minutes=1)
    open_event = {
        "event_type": "OPEN",
        "event_time": opened_at,
        "symbol": "XAUUSD",
        "order_ticket": 501,
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
        "metadata": {
            "tp1_price": 2005.0,
            "tp2_price": 2010.0,
            "tp_execution_mode": "TP1_PARTIAL_MANAGED",
            "tp_sent_to_broker": 0.0,
            "request_comment": "NDS Scalping - TEST",
        },
    }
    tracker.add_trade_open(open_event)

    open_positions = [
        {
            "position_ticket": 9001,
            "symbol": "XAUUSD",
            "side": "BUY",
            "volume": 1.0,
            "entry_price": 2000.0,
            "current_price": 2005.0,
            "sl": 1995.0,
            "tp": 0.0,
            "profit": 0.0,
            "magic": 202402,
            "comment": "NDS Scalping - TEST",
            "open_time": datetime.utcnow(),
            "update_time": datetime.utcnow(),
        }
    ]

    tracker.reconcile_with_open_positions(open_positions)
    assert 9001 in tracker.active_trades

    config = {
        "risk_settings": {"TP2_ENABLED": True},
        "flow_settings": {
            "FLOW_TP1_PARTIAL_CLOSE_PCT": 0.5,
            "FLOW_TP1_MOVE_SL_TO_BE": True,
            "FLOW_TRAIL_AFTER_TP1": False,
            "FLOW_TRAIL_ATR_MULT": 2.0,
        },
        "trading_settings": {"GOLD_SPECIFICATIONS": {"MIN_LOT": 0.01, "LOT_STEP": 0.01}},
    }
    mt5 = DummyMT5()
    manager = PositionManager(config, mt5, trade_tracker=tracker)
    manager.manage_positions(open_positions)

    assert mt5.closed, "Expected TP1 partial close after reconcile"
    assert mt5.closed[0]["volume"] == 0.5
