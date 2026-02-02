from config.settings import config
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.nds.distance_utils import calculate_distance_metrics
from src.trading_bot.risk_manager import create_scalping_risk_manager


def _build_analysis_payload(signal: str, entry_level: float) -> dict:
    return {
        "signal": signal,
        "confidence": 80.0,
        "entry_level": entry_level,
        "entry_model": "MARKET",
        "entry_idea": {
            "entry_level": entry_level,
            "entry_model": "MARKET",
            "entry_type": "FLOW",
        },
        "market_metrics": {
            "atr": 10.0,
        },
        "entry_context": {
            "recent_low": entry_level - 0.8,
            "recent_high": entry_level + 0.8,
        },
    }


def test_rr_repair_tp2_preserves_tp1():
    cfg = config.get_full_config()
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("risk_settings", {})
    cfg.setdefault("flow_settings", {})

    cfg["risk_manager_config"].update(
        {
            "MIN_RR_RATIO": 0.9,
            "RR_REPAIR_ENABLED": True,
            "RR_REPAIR_MODE": "TP2_ONLY",
            "RR_REPAIR_MAX_TP_PIPS": 200.0,
        }
    )
    cfg["risk_settings"].update(
        {
            "RISK_AMOUNT_USD": 25.0,
            "MIN_RISK_REWARD": 0.1,
            "TP1_PIPS": 35.0,
            "TP2_ENABLED": True,
            "TP2_PIPS": 60.0,
            "SL_MIN_PIPS": 80.0,
            "MIN_SL_PIPS": 80.0,
            "SL_MAX_PIPS": 300.0,
        }
    )
    cfg["flow_settings"].update(
        {
            "SCALP_PRESERVE_TP1": True,
            "FLOW_TRAIL_AFTER_TP1": False,
        }
    )

    risk_manager = create_scalping_risk_manager(overrides=cfg)
    analysis_payload = _build_analysis_payload("BUY", 2000.0)
    live_snapshot = LivePriceSnapshot(bid=2000.0, ask=2000.05, timestamp="2026-01-15T01:00:00")
    finalized = risk_manager.finalize_order(
        analysis=analysis_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )

    assert finalized.is_trade_allowed
    assert finalized.take_profit2 is not None

    tp1_metrics = calculate_distance_metrics(
        entry_price=finalized.entry_price,
        current_price=finalized.take_profit,
        point_size=risk_manager._get_gold_spec("point", 0.01),
    )
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

    rr_tp1 = float(tp1_metrics.get("dist_price") or 0.0) / float(sl_metrics.get("dist_price") or 1.0)
    rr_tp2 = float(tp2_metrics.get("dist_price") or 0.0) / float(sl_metrics.get("dist_price") or 1.0)

    assert rr_tp1 < cfg["risk_manager_config"]["MIN_RR_RATIO"]
    assert rr_tp2 + 1e-6 >= cfg["risk_manager_config"]["MIN_RR_RATIO"]
