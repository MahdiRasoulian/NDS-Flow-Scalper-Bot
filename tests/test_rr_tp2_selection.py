from datetime import datetime

from config.settings import config
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.risk_manager import create_scalping_risk_manager


def _cfg(tp2_enabled: bool, tp1_pips: float = 4.0, tp2_pips: float = 11.0, min_rr: float = 0.8):
    cfg = config.get_full_config()
    cfg.setdefault("risk_settings", {})
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("ACCOUNT_BALANCE", 10_000.0)
    cfg["risk_settings"].update(
        {
            "RISK_AMOUNT_USD": 25.0,
            "SL_MIN_PIPS": 8.0,
            "MIN_SL_PIPS": 8.0,
            "SL_MAX_PIPS": 100.0,
            "SL_MAX_PIPS_SCALP": 100.0,
            "SCALP_ATR_SL_MULT": 1.0,
            "TP1_PIPS": tp1_pips,
            "TP2_ENABLED": tp2_enabled,
            "TP2_PIPS": tp2_pips,
            "MIN_RR_RATIO": min_rr,
            "MIN_RISK_REWARD": min_rr,
        }
    )
    cfg["risk_manager_config"]["MIN_RR_RATIO"] = min_rr
    return cfg


def _analysis():
    return {
        "signal": "BUY",
        "confidence": 75.0,
        "entry_level": 2000.0,
        "entry_model": "MARKET",
        "market_metrics": {"atr": 0.0},
    }


def test_rr_uses_tp2_when_enabled():
    cfg = _cfg(tp2_enabled=True)
    rm = create_scalping_risk_manager(overrides=cfg)
    finalized = rm.finalize_order(
        analysis=_analysis(),
        live=LivePriceSnapshot(bid=2000.0, ask=2000.01, timestamp=datetime.utcnow().isoformat()),
        symbol="XAUUSD",
        config=cfg,
    )

    assert finalized.is_trade_allowed
    assert finalized.rr_checked == "TP2"
    assert finalized.rr_validate_mode == "TP2_ENABLED"
    assert finalized.rr_tp1 < cfg["risk_manager_config"]["MIN_RR_RATIO"]
    assert finalized.rr_ratio == finalized.rr_tp2
    assert finalized.rr_ratio >= cfg["risk_manager_config"]["MIN_RR_RATIO"]


def test_rr_falls_back_to_tp1_when_tp2_disabled():
    cfg = _cfg(tp2_enabled=False)
    rm = create_scalping_risk_manager(overrides=cfg)
    finalized = rm.finalize_order(
        analysis=_analysis(),
        live=LivePriceSnapshot(bid=2000.0, ask=2000.01, timestamp=datetime.utcnow().isoformat()),
        symbol="XAUUSD",
        config=cfg,
    )

    assert not finalized.is_trade_allowed
    assert finalized.rr_checked == "TP1"
    assert finalized.rr_validate_mode == "TP1_ONLY"
    assert finalized.reject_reason.startswith("RR ratio below minimum")


def test_rejects_when_tp2_is_not_further_than_tp1():
    cfg = _cfg(tp2_enabled=True, tp1_pips=40.0, tp2_pips=20.0, min_rr=0.1)
    rm = create_scalping_risk_manager(overrides=cfg)
    finalized = rm.finalize_order(
        analysis=_analysis(),
        live=LivePriceSnapshot(bid=2000.0, ask=2000.01, timestamp=datetime.utcnow().isoformat()),
        symbol="XAUUSD",
        config=cfg,
    )

    assert not finalized.is_trade_allowed
    assert finalized.reject_reason == "TP2 must be further than TP1."
