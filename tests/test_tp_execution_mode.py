from src.trading_bot.bot import NDSBot
from src.trading_bot.nds.models import FinalizedOrderParams


def _finalized_with_tp(tp1: float, tp2: float | None, mode: str) -> FinalizedOrderParams:
    return FinalizedOrderParams(
        signal="BUY",
        order_type="market",
        symbol="XAUUSD",
        entry_price=2000.0,
        stop_loss=1995.0,
        take_profit=tp1,
        lot_size=0.5,
        risk_amount_usd=10.0,
        rr_ratio=1.0,
        deviation_pips=0.0,
        decision_notes=[],
        is_trade_allowed=True,
        take_profit2=tp2,
        tp2=tp2,
        tp_execution_mode=mode,
    )


def test_tp1_partial_mode_avoids_broker_tp1():
    finalized = _finalized_with_tp(tp1=2005.0, tp2=2010.0, mode="TP1_PARTIAL_MANAGED")
    flow_settings = {"FLOW_TRAIL_AFTER_TP1": False, "FLOW_TP1_PARTIAL_CLOSE_PCT": 0.5}
    risk_settings = {"TP2_ENABLED": True}

    tp_sent, reason = NDSBot._resolve_broker_tp(finalized, flow_settings, risk_settings)

    assert tp_sent != finalized.take_profit
    assert tp_sent == 2010.0
    assert reason == "tp2_runner"


def test_tp1_partial_without_tp2_sends_no_tp():
    finalized = _finalized_with_tp(tp1=2005.0, tp2=None, mode="TP1_PARTIAL_MANAGED")
    flow_settings = {"FLOW_TRAIL_AFTER_TP1": True, "FLOW_TP1_PARTIAL_CLOSE_PCT": 0.5}
    risk_settings = {"TP2_ENABLED": False}

    tp_sent, reason = NDSBot._resolve_broker_tp(finalized, flow_settings, risk_settings)

    assert tp_sent == 0.0
    assert reason == "sl_only_fsm_managed"


def test_single_tp_mode_sends_no_tp_to_broker():
    finalized = _finalized_with_tp(tp1=2005.0, tp2=None, mode="SINGLE_TP")
    flow_settings = {"FLOW_TRAIL_AFTER_TP1": False, "FLOW_TP1_PARTIAL_CLOSE_PCT": 0.0}
    risk_settings = {"TP2_ENABLED": False}

    tp_sent, reason = NDSBot._resolve_broker_tp(finalized, flow_settings, risk_settings)

    assert tp_sent == 0.0
    assert reason == "sl_only_internal_tp"
