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


def _build_config_payload(overrides: dict | None = None) -> dict:
    cfg = config.get_full_config()
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("risk_settings", {})
    cfg["risk_manager_config"].update(
        {
            "MIN_RR_RATIO": 0.9,
            "RR_REPAIR_ENABLED": True,
            "RR_REPAIR_MAX_TP_PIPS": 120.0,
            "RR_REPAIR_MAX_TP_ATR_MULT": 2.0,
            "STOP_MAX_DEVIATION_PIPS": 70.0,
            "STOP_HARD_REJECT_PIPS": 120.0,
        }
    )
    cfg["risk_settings"].setdefault("RISK_AMOUNT_USD", 25.0)
    cfg["risk_settings"]["MIN_RISK_REWARD"] = 0.1
    if overrides:
        for section, values in overrides.items():
            cfg.setdefault(section, {})
            cfg[section].update(values)
    return cfg


def _build_analysis_payload(*, signal: str, entry_level: float, atr: float) -> dict:
    return {
        "signal": signal,
        "confidence": 78.0,
        "entry_level": entry_level,
        "entry_model": "STOP",
        "entry_idea": {
            "entry_level": entry_level,
            "entry_model": "STOP",
        },
        "market_metrics": {
            "atr": atr,
            "adx": 22.0,
            "volatility_state": "MODERATE",
        },
        "context": {
            "entry_idea": {
                "entry_level": entry_level,
                "entry_model": "STOP",
            },
        },
    }


def test_rr_repair_adjusts_tp_for_buy_stop_near_market():
    risk_manager = _build_risk_manager()
    cfg = _build_config_payload()
    analysis = _build_analysis_payload(signal="BUY", entry_level=5097.718, atr=6.5)
    live = LivePriceSnapshot(bid=5096.14, ask=5096.42)

    finalized = risk_manager.finalize_order(
        analysis=analysis,
        live=live,
        symbol="XAUUSD",
        config=cfg,
    )

    assert finalized.is_trade_allowed
    assert finalized.rr_ratio + 1e-6 >= cfg["risk_manager_config"]["MIN_RR_RATIO"]
    assert any("RR_REPAIR_TP" in note for note in finalized.decision_notes)
    assert any("RR_POSTREPAIR" in note for note in finalized.decision_notes)


def test_rr_repair_rejects_when_tp_cap_exceeded():
    risk_manager = _build_risk_manager()
    cfg = _build_config_payload(
        {
            "risk_manager_config": {
                "RR_REPAIR_MAX_TP_PIPS": 40.0,
            }
        }
    )
    analysis = _build_analysis_payload(signal="BUY", entry_level=5097.718, atr=6.5)
    live = LivePriceSnapshot(bid=5096.14, ask=5096.42)

    finalized = risk_manager.finalize_order(
        analysis=analysis,
        live=live,
        symbol="XAUUSD",
        config=cfg,
    )

    assert not finalized.is_trade_allowed
    assert finalized.reject_reason == "TP cap exceeded for RR repair."
    assert any("RR_REPAIR_REJECT" in note for note in finalized.decision_notes)


def test_rr_repair_sell_direction_keeps_tp_below_entry():
    risk_manager = _build_risk_manager()
    cfg = _build_config_payload()
    analysis = _build_analysis_payload(signal="SELL", entry_level=5094.0, atr=6.5)
    live = LivePriceSnapshot(bid=5096.14, ask=5096.42)

    finalized = risk_manager.finalize_order(
        analysis=analysis,
        live=live,
        symbol="XAUUSD",
        config=cfg,
    )

    assert finalized.is_trade_allowed
    assert finalized.take_profit < finalized.entry_price
    assert any("RR_REPAIR_TP" in note for note in finalized.decision_notes)
