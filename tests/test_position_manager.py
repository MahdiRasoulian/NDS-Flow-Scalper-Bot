from datetime import datetime

from src.trading_bot.position_manager import PositionManager


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


class DummyTradeTracker:
    def __init__(self, metadata):
        self.active_trades = {
            101: {"open_event": {"metadata": metadata}},
        }


def test_tp1_partial_close_and_tp2_set():
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
    metadata = {
        "tp2_price": 2010.0,
        "analysis_snapshot": {
            "entry_context": {"counter_trend": False},
        },
    }
    mt5 = DummyMT5()
    manager = PositionManager(config, mt5, trade_tracker=DummyTradeTracker(metadata))
    open_positions = [
        {
            "position_ticket": 101,
            "symbol": "XAUUSD",
            "side": "BUY",
            "volume": 1.0,
            "entry_price": 2000.0,
            "current_price": 2005.0,
            "sl": 1995.0,
            "tp": 2005.0,
            "profit": 0.0,
            "magic": 0,
            "comment": "",
            "open_time": datetime.utcnow(),
            "update_time": datetime.utcnow(),
        }
    ]

    manager.manage_positions(open_positions)

    assert mt5.closed, "Expected TP1 partial close"
    assert mt5.closed[0]["volume"] == 0.5
    assert any(call["new_tp"] == 2010.0 for call in mt5.modified)
    assert any(call["new_sl"] == 2000.0 for call in mt5.modified)
