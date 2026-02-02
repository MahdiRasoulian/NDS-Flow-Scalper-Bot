from datetime import datetime

from config.settings import config
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.risk_manager import create_scalping_risk_manager


def _build_config_payload() -> dict:
    cfg = config.get_full_config()
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("risk_settings", {})
    cfg["risk_manager_config"]["MIN_RR_RATIO"] = 0.1
    cfg["risk_settings"].setdefault("RISK_AMOUNT_USD", 25.0)
    cfg.setdefault("ACCOUNT_BALANCE", 10_000.0)
    return cfg


def test_countertrend_sell_rejected_without_reversal_confirmation():
    risk_manager = create_scalping_risk_manager()
    analysis_payload = {
        "signal": "SELL",
        "confidence": 80.0,
        "entry_level": 2000.0,
        "entry_model": "MARKET",
        "entry_idea": {
            "entry_level": 2000.0,
            "entry_model": "MARKET",
            "entry_type": "FLOW",
        },
        "market_metrics": {"atr": 5.0},
        "analysis_signal_context": {
            "bias": "BULLISH",
            "directional_bias": "UPTREND",
            "choch": "NONE",
            "bos": "NONE",
            "sweeps_count": 0,
        },
        "entry_context": {
            "counter_trend": True,
            "reversal_ok": False,
            "disp_atr": 0.1,
            "liquidity_ok": True,
            "trend_ok": True,
        },
    }
    cfg = _build_config_payload()
    live_snapshot = LivePriceSnapshot(bid=2000.0, ask=2000.01, timestamp=datetime.utcnow().isoformat())
    finalized = risk_manager.finalize_order(
        analysis=analysis_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )

    assert not finalized.is_trade_allowed
    assert finalized.reject_reason == "Counter-trend reversal confirmation missing."


def test_static_resistance_blocks_buy_without_break_confirmation():
    risk_manager = create_scalping_risk_manager()
    analysis_payload = {
        "signal": "BUY",
        "confidence": 80.0,
        "entry_level": 100.0,
        "entry_model": "MARKET",
        "entry_idea": {
            "entry_level": 100.0,
            "entry_model": "MARKET",
            "entry_type": "FLOW",
        },
        "market_metrics": {"atr": 2.0},
        "analysis_signal_context": {
            "bias": "BULLISH",
            "directional_bias": "UPTREND",
            "choch": "NONE",
            "bos": "NONE",
            "sweeps_count": 0,
        },
        "entry_context": {
            "counter_trend": False,
            "reversal_ok": False,
            "disp_atr": 0.1,
            "liquidity_ok": True,
            "trend_ok": True,
        },
        "static_sr": {
            "nearest_resistance": {
                "price": 100.1,
                "band_bottom": 99.5,
                "band_top": 100.5,
                "dist_pips": 4.0,
                "dist_atr": 0.05,
            },
            "nearest_support": {
                "price": 98.0,
                "band_bottom": 97.0,
                "band_top": 99.0,
                "dist_pips": 20.0,
                "dist_atr": 1.0,
            },
        },
    }
    cfg = _build_config_payload()
    live_snapshot = LivePriceSnapshot(bid=100.0, ask=100.01, timestamp=datetime.utcnow().isoformat())
    finalized = risk_manager.finalize_order(
        analysis=analysis_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )

    assert not finalized.is_trade_allowed
    assert finalized.reject_reason == "Entry inside static resistance without confirmation."
