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
        self.partial_calls = []

    def register_partial_close(self, **kwargs):
        self.partial_calls.append(kwargs)
        return None


class FailingMT5:
    def __init__(self):
        self.modified = []
        self.closed = []

    def modify_position(self, ticket: int, new_sl: float = None, new_tp: float = None):
        payload = {"ticket": ticket, "new_sl": new_sl, "new_tp": new_tp, "retcode": "REJECT"}
        self.modified.append(payload)
        return {"success": False, **payload}

    def close_position(self, ticket: int, volume: float = None, comment: str = ""):
        payload = {"ticket": ticket, "volume": volume, "comment": comment, "retcode": "REJECT"}
        self.closed.append(payload)
        return {"success": False, **payload}


class DummyTradeTrackerNoPartial:
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
        "tp1_price": 2005.0,
        "tp2_price": 2010.0,
        "analysis_snapshot": {
            "entry_context": {"counter_trend": False},
        },
    }
    mt5 = DummyMT5()
    tracker = DummyTradeTracker(metadata)
    manager = PositionManager(config, mt5, trade_tracker=tracker)
    open_positions = [
        {
            "position_ticket": 101,
            "symbol": "XAUUSD",
            "side": "BUY",
            "volume": 1.0,
            "entry_price": 2000.0,
            "current_price": 2005.0,
            "sl": 1995.0,
            "tp": 0.0,
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
    assert tracker.partial_calls and tracker.partial_calls[0]["position_ticket"] == 101


def test_tp1_partial_close_only_once():
    config = {
        "risk_settings": {"TP2_ENABLED": True},
        "flow_settings": {
            "FLOW_TP1_PARTIAL_CLOSE_PCT": 0.5,
            "FLOW_TP1_MOVE_SL_TO_BE": True,
            "FLOW_TRAIL_AFTER_TP1": True,
        },
        "trading_settings": {"GOLD_SPECIFICATIONS": {"MIN_LOT": 0.01, "LOT_STEP": 0.01}},
    }
    metadata = {
        "tp1_price": 2005.0,
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
            "tp": 0.0,
            "profit": 0.0,
            "magic": 0,
            "comment": "",
            "open_time": datetime.utcnow(),
            "update_time": datetime.utcnow(),
        }
    ]

    manager.manage_positions(open_positions)
    manager.manage_positions(open_positions)

    assert len(mt5.closed) == 1


def test_tp1_failure_logs_and_keeps_retryable_state(caplog):
    config = {
        "risk_settings": {"TP2_ENABLED": True},
        "flow_settings": {
            "FLOW_TP1_PARTIAL_CLOSE_PCT": 0.5,
            "FLOW_TP1_MOVE_SL_TO_BE": True,
            "FLOW_TRAIL_AFTER_TP1": False,
        },
        "trading_settings": {"GOLD_SPECIFICATIONS": {"MIN_LOT": 0.01, "LOT_STEP": 0.01}},
    }
    metadata = {
        "tp1_price": 2005.0,
        "tp2_price": 2010.0,
        "analysis_snapshot": {
            "entry_context": {"counter_trend": False},
        },
    }
    mt5 = FailingMT5()
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
            "tp": 0.0,
            "profit": 0.0,
            "magic": 0,
            "comment": "",
            "open_time": datetime.utcnow(),
            "update_time": datetime.utcnow(),
        }
    ]

    with caplog.at_level("WARNING"):
        manager.manage_positions(open_positions)

    assert mt5.closed, "Expected partial close attempt even on failure"
    assert mt5.modified, "Expected modify attempt for TP2 or SL move"
    assert any("[NDS][TP1_PARTIAL_FAIL]" in record.message for record in caplog.records)
    assert any("[NDS][SL_BE_FAIL]" in record.message for record in caplog.records)


def test_fallback_plan_synthesizes_tp_targets():
    config = {
        "risk_settings": {"TP2_ENABLED": True, "TP1_PIPS": 35.0, "TP2_PIPS": 70.0},
        "flow_settings": {
            "FLOW_TP1_PARTIAL_CLOSE_PCT": 0.5,
            "FLOW_TP1_MOVE_SL_TO_BE": True,
            "FLOW_TRAIL_AFTER_TP1": False,
        },
        "trading_settings": {"GOLD_SPECIFICATIONS": {"MIN_LOT": 0.01, "LOT_STEP": 0.01}},
    }
    mt5 = DummyMT5()
    manager = PositionManager(config, mt5, trade_tracker=DummyTradeTracker({}))
    open_positions = [
        {
            "position_ticket": 101,
            "symbol": "XAUUSD",
            "side": "BUY",
            "volume": 1.0,
            "entry_price": 2000.0,
            "current_price": 2003.5,
            "sl": 1995.0,
            "tp": 0.0,
            "profit": 0.0,
            "magic": 0,
            "comment": "",
            "open_time": datetime.utcnow(),
            "update_time": datetime.utcnow(),
        }
    ]

    manager.manage_positions(open_positions)

    assert mt5.closed, "Expected TP1 partial close even with synthesized plan"


def test_partial_close_metadata_matches_full_close_fallback():
    config = {
        "risk_settings": {"TP2_ENABLED": True},
        "flow_settings": {
            "FLOW_TP1_PARTIAL_CLOSE_PCT": 0.5,
            "FLOW_TP1_MOVE_SL_TO_BE": True,
            "FLOW_TRAIL_AFTER_TP1": False,
        },
        "trading_settings": {"GOLD_SPECIFICATIONS": {"MIN_LOT": 0.02, "LOT_STEP": 0.01}},
    }
    metadata = {"tp1_price": 2005.0, "tp2_price": 2010.0}
    mt5 = DummyMT5()
    tracker = DummyTradeTracker(metadata)
    manager = PositionManager(config, mt5, trade_tracker=tracker)
    open_positions = [{
        "position_ticket": 101, "symbol": "XAUUSD", "side": "BUY", "volume": 0.02,
        "entry_price": 2000.0, "current_price": 2005.0, "sl": 1995.0, "tp": 0.0,
        "profit": 0.0, "magic": 0, "comment": "", "open_time": datetime.utcnow(), "update_time": datetime.utcnow(),
    }]

    manager.manage_positions(open_positions)

    assert mt5.closed and mt5.closed[0]["volume"] == 0.02
    assert tracker.partial_calls and tracker.partial_calls[0]["remaining_volume"] == 0.0


def test_tp1_flow_without_register_partial_close_hook():
    config = {
        "risk_settings": {"TP2_ENABLED": True},
        "flow_settings": {
            "FLOW_TP1_PARTIAL_CLOSE_PCT": 0.3,
            "FLOW_TP1_MOVE_SL_TO_BE": True,
            "FLOW_TRAIL_AFTER_TP1": False,
        },
        "trading_settings": {"GOLD_SPECIFICATIONS": {"MIN_LOT": 0.01, "LOT_STEP": 0.01}},
    }
    metadata = {"tp1_price": 2005.0, "tp2_price": 2010.0}
    mt5 = DummyMT5()
    manager = PositionManager(config, mt5, trade_tracker=DummyTradeTrackerNoPartial(metadata))
    open_positions = [{
        "position_ticket": 101, "symbol": "XAUUSD", "side": "BUY", "volume": 1.0,
        "entry_price": 2000.0, "current_price": 2005.0, "sl": 1995.0, "tp": 0.0,
        "profit": 0.0, "magic": 0, "comment": "", "open_time": datetime.utcnow(), "update_time": datetime.utcnow(),
    }]

    manager.manage_positions(open_positions)

    assert mt5.closed and mt5.closed[0]["volume"] == 0.3
