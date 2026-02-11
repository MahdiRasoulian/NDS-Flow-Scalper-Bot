from datetime import datetime, timedelta
import logging

import pytest

from config.settings import config
from src.trading_bot.nds.analyzer import GoldNDSAnalyzer
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.risk_manager import create_scalping_risk_manager

pd = pytest.importorskip("pandas")


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


def test_ctgate_blocks_before_sr_gate(caplog):
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
        "static_sr": {
            "nearest_resistance": {
                "price": 2005.0,
                "band_bottom": 1998.0,
                "band_top": 2006.0,
                "dist_pips": 10.0,
                "dist_atr": 0.2,
            },
            "nearest_support": {
                "price": 1980.0,
                "band_bottom": 1975.0,
                "band_top": 1985.0,
                "dist_pips": 20.0,
                "dist_atr": 0.5,
            },
        },
    }
    cfg = _build_config_payload()
    live_snapshot = LivePriceSnapshot(bid=2000.0, ask=2000.01, timestamp=datetime.utcnow().isoformat())

    with caplog.at_level(logging.INFO):
        finalized = risk_manager.finalize_order(
            analysis=analysis_payload,
            live=live_snapshot,
            symbol="XAUUSD",
            config=cfg,
        )

    assert not finalized.is_trade_allowed
    assert "SR_GATE" not in caplog.text


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
    assert finalized.reject_reason == "Entry near static resistance without confirmation."


def test_support_must_be_below_price():
    start = datetime(2025, 1, 1, 0, 0, 0)
    times = [start + timedelta(minutes=5 * idx) for idx in range(6)]
    data = {
        "time": times,
        "open": [10.0, 11.0, 10.5, 12.0, 11.5, 11.0],
        "high": [10.5, 12.5, 11.0, 13.0, 12.0, 11.2],
        "low": [9.5, 10.0, 9.8, 11.0, 10.8, 10.5],
        "close": [10.2, 12.0, 10.7, 12.5, 11.2, 11.0],
        "volume": [1, 1, 1, 1, 1, 1],
    }
    cfg = {
        "technical_settings": {
            "STATIC_SR_LOOKBACK": 6,
            "STATIC_SR_SWING_WINDOW": 1,
            "STATIC_SR_CLUSTER_PIPS": 1.0,
            "STATIC_SR_MAX_LEVELS": 5,
            "STATIC_SR_BAND_ATR": 0.1,
            "STATIC_SR_MIN_BAND_PIPS": 1.0,
        },
        "trading_settings": {"POINT_SIZE": 0.01},
    }
    try:
        df = pd.DataFrame(data)
    except TypeError:
        pytest.skip("pandas DataFrame constructor unavailable in this environment")
    analyzer = GoldNDSAnalyzer(df, config=cfg)
    ref_price = 11.0
    sr_context = analyzer._compute_static_sr_context(ref_price, atr_value=1.0)
    nearest_support = sr_context.get("nearest_support")
    nearest_resistance = sr_context.get("nearest_resistance")
    if nearest_support:
        assert nearest_support["price"] <= ref_price
    if nearest_resistance:
        assert nearest_resistance["price"] >= ref_price


def test_sr_gate_applies_to_ifvg_and_breaker():
    risk_manager = create_scalping_risk_manager()
    base_payload = {
        "signal": "BUY",
        "confidence": 80.0,
        "entry_level": 100.0,
        "entry_model": "MARKET",
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

    ifvg_payload = dict(base_payload, entry_idea={"entry_level": 100.0, "entry_model": "MARKET", "entry_type": "IFVG"})
    breaker_payload = dict(base_payload, entry_idea={"entry_level": 100.0, "entry_model": "MARKET", "entry_type": "BREAKER"})

    ifvg_result = risk_manager.finalize_order(
        analysis=ifvg_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )
    breaker_result = risk_manager.finalize_order(
        analysis=breaker_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )

    assert not ifvg_result.is_trade_allowed
    assert not breaker_result.is_trade_allowed


def test_analyzer_sr_permission_gate_blocks_buy_near_strong_resistance():
    data = {
        'time': [datetime(2025, 1, 1, 0, 0, 0), datetime(2025, 1, 1, 0, 5, 0)],
        'open': [100.0, 100.1],
        'high': [100.3, 100.4],
        'low': [99.8, 99.9],
        'close': [100.0, 100.2],
        'volume': [1, 1],
    }
    try:
        df = pd.DataFrame(data)
    except TypeError:
        pytest.skip("pandas DataFrame constructor unavailable in this environment")
    analyzer = GoldNDSAnalyzer(df)
    analyzer.GOLD_SETTINGS.update(
        {
            "STATIC_SR_PROXIMITY_BLOCK_ATR": 0.25,
            "STATIC_SR_STRONG_TOUCHES_MIN": 3,
        }
    )
    payload = {
        "signal": "BUY",
        "reasons": [],
        "market_metrics": {"current_rvol": 1.2, "adx": 25.0},
        "session_analysis": {"weight": 1.0},
        "confidence": 80.0,
        "structure": {"structure_score": 70.0, "bos": "NONE", "choch": "NONE"},
        "context": {
            "analysis_signal_context": {
                "bias": "BULLISH",
                "strong_trend": False,
                "reversal_ok": False,
                "choch": "NONE",
                "bos": "NONE",
            },
            "entry_context": {
                "reversal_ok": False,
                "disp_atr": 0.1,
            },
            "static_sr": {
                "nearest_resistance": {"dist_atr": 0.12, "touches": 5},
            },
        },
        "static_sr": {
            "nearest_resistance": {"dist_atr": 0.12, "touches": 5},
        },
    }

    out = analyzer._apply_final_filters(payload, scalping_mode=True)
    assert out["signal"] == "NONE"
    assert any("SR permission gate" in r for r in out["reasons"])


def test_structural_sl_anchor_uses_nearest_support_for_buy():
    risk_manager = create_scalping_risk_manager()
    sltp = risk_manager._compute_scalping_sl_tp(
        signal="BUY",
        entry_price=5060.06,
        atr_value=6.0,
        recent_low=5057.0,
        recent_high=5062.0,
        config_payload=_build_config_payload(),
        point_size=0.01,
        tp1_target_price=None,
        counter_trend=False,
        reversal_ok=False,
        nearest_support={"price": 5053.0},
        nearest_resistance={"price": 5058.7},
    )
    assert sltp["sl_source"].startswith("structural_anchor")
    assert sltp["stop_loss"] < 5053.0


def test_static_resistance_proximity_blocks_buy_without_confirmation():
    risk_manager = create_scalping_risk_manager()
    analysis_payload = {
        "signal": "BUY",
        "confidence": 80.0,
        "entry_level": 100.7,
        "entry_model": "MARKET",
        "entry_idea": {
            "entry_level": 100.7,
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
                "price": 100.5,
                "band_bottom": 99.0,
                "band_top": 100.6,
                "dist_pips": 2.0,
                "dist_atr": 0.12,
                "touches": 5,
            },
            "nearest_support": {
                "price": 98.0,
                "band_bottom": 97.0,
                "band_top": 99.0,
                "dist_pips": 20.0,
                "dist_atr": 1.0,
                "touches": 2,
            },
        },
    }
    cfg = _build_config_payload()
    live_snapshot = LivePriceSnapshot(bid=100.7, ask=100.71, timestamp=datetime.utcnow().isoformat())
    finalized = risk_manager.finalize_order(
        analysis=analysis_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )

    assert not finalized.is_trade_allowed
    assert finalized.reject_reason == "Entry near static resistance without confirmation."
