from datetime import datetime, timedelta

from src.trading_bot.position_manager import PositionManager
from src.trading_bot.trade_tracker import TradeTracker


class DummyMT5:
    def __init__(self):
        self.closed = []

    def close_position(self, ticket: int, volume: float = None, comment: str = ""):
        payload = {"ticket": ticket, "volume": volume, "comment": comment, "price": 2005.0}
        self.closed.append(payload)
        return {"success": True, **payload}

    def modify_position(self, ticket: int, new_sl: float = None, new_tp: float = None):
        return {"success": True, "ticket": ticket, "new_sl": new_sl, "new_tp": new_tp}


def test_market_open_reconcile_updates_position_ticket_before_manage(caplog):
    tracker = TradeTracker()
    opened_at = datetime.utcnow() - timedelta(minutes=1)
    tracker.add_trade_open(
        {
            "event_type": "OPEN",
            "event_time": opened_at,
            "symbol": "XAUUSD",
            "order_ticket": 701,
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
                "request_comment": "NDS Scalping - MARKET",
                "magic": 202401,
                "order_type": "market",
            },
        }
    )

    open_positions = [
        {
            "position_ticket": 8801,
            "symbol": "XAUUSD",
            "side": "BUY",
            "volume": 1.0,
            "entry_price": 2000.0,
            "current_price": 2005.0,
            "sl": 1995.0,
            "tp": 0.0,
            "profit": 0.0,
            "magic": 202401,
            "comment": "NDS Scalping - MARKET",
            "open_time": datetime.utcnow(),
            "update_time": datetime.utcnow(),
        }
    ]

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

    with caplog.at_level("INFO"):
        tracker.reconcile_with_open_positions(open_positions)
        manager.manage_positions(open_positions)

    assert 8801 in tracker.active_trades
    assert tracker.active_trades[8801]["open_event"]["position_ticket"] == 8801
    assert tracker.active_trades[8801]["open_event"]["metadata"]["tp1_price"] == 2005.0
    assert mt5.closed, "Expected TP1 partial close after reconcile"

    messages = [record.message for record in caplog.records]
    resolved_idx = next(
        idx for idx, msg in enumerate(messages) if "[TRADE][OPEN_RESOLVED_POSITION]" in msg
    )
    plan_meta_idx = next(idx for idx, msg in enumerate(messages) if "[PM][PLAN_META]" in msg)
    manage_idx = next(idx for idx, msg in enumerate(messages) if "[PM][MANAGE]" in msg)
    assert resolved_idx < plan_meta_idx < manage_idx
