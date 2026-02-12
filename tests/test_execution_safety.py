from __future__ import annotations

import types

from src.trading_bot.bot import NDSBot
from src.trading_bot.position_manager import PositionManager
from src.trading_bot.mt5_client import MT5Client


class _DummyRisk:
    def __init__(self, min_lot: float = 0.02, lot_step: float = 0.01):
        self.values = {"MIN_LOT": min_lot, "LOT_STEP": lot_step}

    def _get_gold_spec(self, key: str, default=None):
        return self.values.get(key, default)


class _DummyClient:
    def __init__(self, volume_min: float = 0.01):
        self._info = types.SimpleNamespace(volume_min=volume_min)

    def _get_symbol_info(self, _symbol: str):
        return self._info


class _DummyMT5ForPM:
    def __init__(self):
        self.closed = []

    def modify_position(self, ticket: int, new_sl: float = None, new_tp: float = None):
        return {"success": True, "ticket": ticket, "new_sl": new_sl, "new_tp": new_tp}

    def close_position(self, ticket: int, volume: float = None, comment: str = ""):
        self.closed.append((ticket, volume, comment))
        return {"success": True, "ticket": ticket, "closed_volume": volume}


class _FakeResult:
    def __init__(self, retcode: int, comment: str = "ok", order: int = 11, price: float = 2000.0, volume: float = 0.02):
        self.retcode = retcode
        self.comment = comment
        self.order = order
        self.deal = 22
        self.price = price
        self.volume = volume


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _make_bot(min_lot=0.02, lot_step=0.01, broker_min=0.01):
    bot = NDSBot.__new__(NDSBot)
    bot.risk_manager = _DummyRisk(min_lot=min_lot, lot_step=lot_step)
    bot.mt5_client = _DummyClient(volume_min=broker_min)
    return bot


def test_volume_enforcement_rejects_below_strategy_min_lot():
    bot = _make_bot(min_lot=0.02, lot_step=0.01, broker_min=0.01)
    assert bot._enforce_execution_volume(0.01, "XAUUSD") is None


def test_volume_enforcement_rejects_illegal_001_open():
    bot = _make_bot(min_lot=0.02, lot_step=0.01, broker_min=0.01)
    assert bot._enforce_execution_volume(0.019, "XAUUSD") is None


def test_partial_close_small_position_closes_full_when_remainder_illegal():
    config = {
        "flow_settings": {"FLOW_TP1_MOVE_SL_TO_BE_PLUS_PIPS": 0.0},
        "trading_settings": {"GOLD_SPECIFICATIONS": {"MIN_LOT": 0.02, "LOT_STEP": 0.01}},
    }
    pm_client = _DummyMT5ForPM()
    manager = PositionManager(config, pm_client)
    plan = types.SimpleNamespace(ticket=777, symbol="XAUUSD", volume=0.02)

    ok = manager._partial_close(plan)

    assert ok is True
    assert pm_client.closed
    _, volume, _ = pm_client.closed[0]
    assert volume == 0.02


def test_multiple_positions_have_independent_tp1_evaluations():
    config = {
        "flow_settings": {"FLOW_TP1_MOVE_SL_TO_BE_PLUS_PIPS": 0.0},
        "trading_settings": {"GOLD_SPECIFICATIONS": {"MIN_LOT": 0.01, "LOT_STEP": 0.01}},
    }
    pm_client = _DummyMT5ForPM()
    manager = PositionManager(config, pm_client)

    p1 = types.SimpleNamespace(ticket=1, direction="BUY", tp1_price=2001.0, symbol="XAUUSD")
    p2 = types.SimpleNamespace(ticket=2, direction="BUY", tp1_price=2005.0, symbol="XAUUSD")

    assert manager._crossed_tp1(p1, 2002.0) is True
    assert manager._crossed_tp1(p2, 2002.0) is False


def test_mt5_client_failure_return_structure(monkeypatch):
    import src.trading_bot.mt5_client as mt5_mod

    monkeypatch.setattr(mt5_mod.mt5, "SYMBOL_FILLING_FOK", 1, raising=False)
    monkeypatch.setattr(mt5_mod.mt5, "SYMBOL_FILLING_IOC", 2, raising=False)
    monkeypatch.setattr(mt5_mod.mt5, "ORDER_FILLING_FOK", 11, raising=False)
    monkeypatch.setattr(mt5_mod.mt5, "ORDER_FILLING_IOC", 12, raising=False)
    monkeypatch.setattr(mt5_mod.mt5, "TRADE_RETCODE_DONE", 10009, raising=False)
    monkeypatch.setattr(mt5_mod.mt5, "TRADE_RETCODE_REQUOTE", 10004, raising=False)
    monkeypatch.setattr(mt5_mod.mt5, "TRADE_RETCODE_PRICE_OFF", 10021, raising=False)

    monkeypatch.setattr(mt5_mod.mt5, "symbol_info", lambda _symbol: types.SimpleNamespace(filling_mode=2), raising=False)
    monkeypatch.setattr(mt5_mod.mt5, "order_send", lambda _req: _FakeResult(10014, "Invalid volume"), raising=False)

    client = MT5Client.__new__(MT5Client)
    client._logger = _DummyLogger()

    result = client._order_send_with_retry(
        {"type": 0, "price": 2000.0, "comment": "t"},
        "XAUUSD",
        "close_position",
    )

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["retcode"] == 10014
    assert "comment" in result
    assert "raw" in result
