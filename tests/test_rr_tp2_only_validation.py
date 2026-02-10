from config.settings import config
from src.trading_bot.nds.distance_utils import calculate_distance_metrics
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.risk_manager import create_scalping_risk_manager


def _base_config():
    cfg = config.get_full_config()
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("risk_settings", {})
    cfg.setdefault("flow_settings", {})
    cfg["risk_manager_config"].update(
        {
            "MIN_RR_RATIO": 0.9,
            "SCALP_RR_MODE": "TP2_ONLY",
            "SCALP_TP1_ONLY_MIN_RR": 0.4,
            "RR_TP2_AUTOGEN_ENABLED": True,
            "RR_TP2_AUTOGEN_MAX_PIPS": 120.0,
            "RR_TP2_AUTOGEN_MAX_ATR_MULT": 2.0,
            "RR_REPAIR_ENABLED": True,
            "RR_REPAIR_MODE": "TP2_ONLY",
            "RR_REPAIR_MAX_TP_PIPS": 120.0,
            "RR_REPAIR_MAX_TP_ATR_MULT": 2.0,
            "RR_EPSILON": 1e-6,
        }
    )
    cfg["risk_settings"].update(
        {
            "RISK_AMOUNT_USD": 25.0,
            "MIN_RISK_REWARD": 0.1,
            "TP1_PIPS": 35.0,
            "TP2_ENABLED": False,
            "TP2_PIPS": 0.0,
            "SL_MIN_PIPS": 45.0,
            "MIN_SL_PIPS": 45.0,
            "SL_MAX_PIPS": 300.0,
            "SL_MAX_PIPS_SCALP": 300.0,
        }
    )
    cfg["flow_settings"].update(
        {
            "SCALP_PRESERVE_TP1": True,
            "FLOW_TRAIL_AFTER_TP1": True,
        }
    )
    return cfg


def test_tp2_only_uses_tp2_rr_for_validation_and_allows_trade():
    cfg = _base_config()
    risk_manager = create_scalping_risk_manager(overrides=cfg)
    analysis_payload = {
        "signal": "BUY",
        "confidence": 80.0,
        "entry_level": 2000.0,
        "entry_model": "MARKET",
        "market_metrics": {"atr": 0.0},
        "entry_context": {
            "recent_low": 1999.8,
            "recent_high": 2003.0,
        },
        "scalping_mode": True,
    }
    live_snapshot = LivePriceSnapshot(bid=2000.0, ask=2000.05, timestamp="2026-02-02T10:07:57")
    finalized = risk_manager.finalize_order(
        analysis=analysis_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )

    assert finalized.is_trade_allowed
    assert finalized.take_profit2 is not None
    assert any("RR_TP2_AUTOGEN" in note for note in finalized.decision_notes)

    tp2_metrics = calculate_distance_metrics(
        entry_price=finalized.entry_price,
        current_price=finalized.take_profit2,
        point_size=risk_manager._get_gold_spec("point", 0.01),
    )
    sl_metrics = calculate_distance_metrics(
        entry_price=finalized.entry_price,
        current_price=finalized.stop_loss,
        point_size=risk_manager._get_gold_spec("point", 0.01),
    )
    tp2_pips = float(tp2_metrics.get("dist_pips") or 0.0)
    sl_pips = float(sl_metrics.get("dist_pips") or 0.0)
    assert tp2_pips + cfg["risk_manager_config"]["RR_EPSILON"] >= sl_pips * cfg["risk_manager_config"]["SCALP_TP1_ONLY_MIN_RR"]


def test_tp2_only_rejects_when_repair_caps_block_tp2():
    cfg = _base_config()
    cfg["risk_settings"]["TP2_ENABLED"] = True
    cfg["risk_settings"]["TP2_PIPS"] = 5.0
    cfg["flow_settings"]["FLOW_TRAIL_AFTER_TP1"] = False
    cfg["risk_manager_config"]["RR_TP2_AUTOGEN_ENABLED"] = False
    cfg["risk_manager_config"]["RR_REPAIR_MAX_TP_PIPS"] = 10.0

    risk_manager = create_scalping_risk_manager(overrides=cfg)
    analysis_payload = {
        "signal": "BUY",
        "confidence": 80.0,
        "entry_level": 2000.0,
        "entry_model": "MARKET",
        "market_metrics": {"atr": 0.0},
        "entry_context": {
            "recent_low": 1999.8,
            "recent_high": 2003.0,
        },
        "scalping_mode": True,
    }
    live_snapshot = LivePriceSnapshot(bid=2000.0, ask=2000.05, timestamp="2026-02-02T10:07:57")
    finalized = risk_manager.finalize_order(
        analysis=analysis_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )

    assert not finalized.is_trade_allowed
    assert finalized.reject_reason == "INVALID_SETUP: TP2 gap repair exceeds RR repair caps."


def test_tp2_only_allows_large_sl_with_tp2_autogen():
    cfg = _base_config()
    cfg["risk_manager_config"]["SCALP_TP1_ONLY_MIN_RR"] = 0.4
    cfg["risk_settings"]["SL_MIN_PIPS"] = 10.0
    cfg["risk_settings"]["MIN_SL_PIPS"] = 10.0
    cfg["risk_settings"]["SL_MAX_PIPS"] = 300.0
    cfg["risk_settings"]["SL_MAX_PIPS_SCALP"] = 300.0
    cfg["risk_settings"]["MIN_RISK_REWARD"] = 0.9

    risk_manager = create_scalping_risk_manager(overrides=cfg)
    analysis_payload = {
        "signal": "BUY",
        "confidence": 80.0,
        "entry_level": 2000.0,
        "entry_model": "MARKET",
        "market_metrics": {"atr": 0.0},
        "entry_context": {
            "recent_low": 1997.055,
            "recent_high": 2003.0,
        },
        "scalping_mode": True,
    }
    live_snapshot = LivePriceSnapshot(bid=2000.0, ask=2000.05, timestamp="2026-02-02T10:07:57")
    finalized = risk_manager.finalize_order(
        analysis=analysis_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )

    assert finalized.is_trade_allowed
    assert any(
        "RR_VALIDATE scope=TP2_ONLY rr_checked=TP2" in note for note in finalized.decision_notes
    )
    assert any(
        "source=min_rr:scalp.SCALP_TP1_ONLY_MIN_RR" in note for note in finalized.decision_notes
    )
    assert not any("0.90" in note for note in finalized.decision_notes)
