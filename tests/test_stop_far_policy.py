from config.settings import config
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.risk_manager import create_scalping_risk_manager


def _build_risk_manager():
    return create_scalping_risk_manager(
        overrides={
            "risk_settings": {
                "MIN_RISK_REWARD": 0.1,
            }
        }
    )


def _build_analysis_payload(
    *,
    signal: str,
    entry_level: float,
    entry_model: str = "STOP",
    confidence: float = 80.0,
    adx: float = 20.0,
    volatility_state: str = "LOW",
) -> dict:
    return {
        "signal": signal,
        "confidence": confidence,
        "entry_level": entry_level,
        "entry_model": entry_model,
        "entry_idea": {
            "entry_level": entry_level,
            "entry_model": entry_model,
            "entry_type": "FLOW",
        },
        "market_metrics": {
            "atr": 5.0,
            "adx": adx,
            "volatility_state": volatility_state,
        },
        "context": {
            "entry_idea": {
                "entry_level": entry_level,
                "entry_model": entry_model,
            },
        },
    }


def _build_config_payload(overrides: dict | None = None) -> dict:
    cfg = config.get_full_config()
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("risk_settings", {})
    cfg["risk_manager_config"].update(
        {
            "MIN_RR_RATIO": 0.1,
            "STOP_MAX_DEVIATION_PIPS": 50.0,
            "STOP_HARD_REJECT_PIPS": 150.0,
            "STOP_CONVERT_TO_LIMIT_PIPS": 20.0,
            "TREND_STRENGTH_ADX_MIN": 25.0,
            "MEAN_REVERSION_ADX_MAX": 18.0,
            "MAX_ENTRY_CAP_PIPS": 30.0,
        }
    )
    cfg["risk_settings"].setdefault("RISK_AMOUNT_USD", 25.0)
    cfg["risk_settings"]["MIN_RISK_REWARD"] = 0.1
    if overrides:
        for section, values in overrides.items():
            cfg.setdefault(section, {})
            cfg[section].update(values)
    return cfg


def test_stop_far_policy_rejects_hard_cap():
    risk_manager = _build_risk_manager()
    cfg = _build_config_payload(
        {
            "risk_manager_config": {"STOP_HARD_REJECT_PIPS": 110.0},
        }
    )
    analysis = _build_analysis_payload(signal="BUY", entry_level=5084.94, adx=30.0)
    live = LivePriceSnapshot(bid=5073.21, ask=5073.31)

    finalized = risk_manager.finalize_order(
        analysis=analysis,
        live=live,
        symbol="XAUUSD",
        config=cfg,
    )

    assert not finalized.is_trade_allowed
    assert finalized.reject_reason == "Stop too far."
    assert finalized.order_type == "NONE"
    assert any("STOP_FAR_POLICY:REJECT_HARD" in note for note in finalized.decision_notes)


def test_stop_far_policy_converts_to_limit_on_mean_reversion():
    risk_manager = _build_risk_manager()
    cfg = _build_config_payload()
    analysis = _build_analysis_payload(
        signal="BUY",
        entry_level=5084.94,
        adx=12.0,
        volatility_state="LOW",
    )
    live = LivePriceSnapshot(bid=5073.21, ask=5073.31)

    finalized = risk_manager.finalize_order(
        analysis=analysis,
        live=live,
        symbol="XAUUSD",
        config=cfg,
    )

    assert finalized.is_trade_allowed
    assert finalized.order_type == "LIMIT"
    assert finalized.entry_price < live.ask
    assert any("STOP_FAR_POLICY:LIMIT" in note for note in finalized.decision_notes)


def test_stop_far_policy_caps_entry_on_trend_continuation_buy_sell():
    risk_manager = _build_risk_manager()
    cfg = _build_config_payload()

    buy_analysis = _build_analysis_payload(signal="BUY", entry_level=5084.94, adx=40.0)
    buy_live = LivePriceSnapshot(bid=5073.21, ask=5073.31)
    buy_finalized = risk_manager.finalize_order(
        analysis=buy_analysis,
        live=buy_live,
        symbol="XAUUSD",
        config=cfg,
    )
    assert buy_finalized.is_trade_allowed
    assert buy_finalized.order_type in {"STOP", "MARKET"}
    assert buy_finalized.entry_price <= buy_analysis["entry_level"]
    assert any("STOP_FAR_POLICY:CAP_ENTRY" in note for note in buy_finalized.decision_notes)

    sell_analysis = _build_analysis_payload(signal="SELL", entry_level=5060.0, adx=40.0)
    sell_live = LivePriceSnapshot(bid=5073.21, ask=5073.31)
    sell_finalized = risk_manager.finalize_order(
        analysis=sell_analysis,
        live=sell_live,
        symbol="XAUUSD",
        config=cfg,
    )
    assert sell_finalized.is_trade_allowed
    assert sell_finalized.order_type in {"STOP", "MARKET"}
    assert sell_finalized.entry_price >= sell_analysis["entry_level"]
    assert sell_finalized.entry_price <= sell_live.bid
    assert any("STOP_FAR_POLICY:CAP_ENTRY" in note for note in sell_finalized.decision_notes)


def test_stop_far_policy_threshold_below_soft_does_not_trigger():
    risk_manager = _build_risk_manager()
    cfg = _build_config_payload({"risk_manager_config": {"STOP_MAX_DEVIATION_PIPS": 20.0}})
    analysis = _build_analysis_payload(signal="BUY", entry_level=5074.0, adx=30.0)
    live = LivePriceSnapshot(bid=5073.21, ask=5073.31)

    finalized = risk_manager.finalize_order(
        analysis=analysis,
        live=live,
        symbol="XAUUSD",
        config=cfg,
    )

    assert finalized.is_trade_allowed
    assert any("STOP_FAR_POLICY:SKIP" in note for note in finalized.decision_notes)
    assert not any(
        token in note
        for note in finalized.decision_notes
        for token in ("STOP_FAR_POLICY:LIMIT", "STOP_FAR_POLICY:CAP_ENTRY", "STOP_FAR_POLICY:REJECT")
    )
