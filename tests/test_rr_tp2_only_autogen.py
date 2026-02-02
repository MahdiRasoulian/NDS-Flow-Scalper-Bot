from config.settings import config
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.nds.distance_utils import calculate_distance_metrics
from src.trading_bot.risk_manager import create_scalping_risk_manager


def test_tp2_only_autogen_allows_trade_when_tp2_missing():
    cfg = config.get_full_config()
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("risk_settings", {})
    cfg.setdefault("flow_settings", {})

    cfg["risk_manager_config"].update(
        {
            "MIN_RR_RATIO": 0.3,
            "SCALP_RR_MODE": "TP2_ONLY",
            "SCALP_TP1_ONLY_MIN_RR": 0.4,
            "RR_TP2_AUTOGEN_ENABLED": True,
            "RR_TP2_AUTOGEN_MAX_PIPS": 150.0,
            "RR_TP2_AUTOGEN_MAX_ATR_MULT": 2.0,
        }
    )
    cfg["risk_settings"].update(
        {
            "RISK_AMOUNT_USD": 25.0,
            "MIN_RISK_REWARD": 0.1,
            "TP1_PIPS": 35.0,
            "TP2_ENABLED": False,
            "TP2_PIPS": 0.0,
            "SL_MIN_PIPS": 300.0,
            "MIN_SL_PIPS": 300.0,
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

    risk_manager = create_scalping_risk_manager(overrides=cfg)
    analysis_payload = {
        "signal": "BUY",
        "confidence": 80.0,
        "entry_level": 2000.0,
        "entry_model": "MARKET",
        "market_metrics": {"atr": 10.0},
        "entry_context": {
            "recent_low": 1999.2,
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

    assert tp2_pips >= sl_pips * cfg["risk_manager_config"]["SCALP_TP1_ONLY_MIN_RR"]
