from datetime import datetime, timedelta

import pytest

pd = pytest.importorskip("pandas")

from config.settings import config
from src.trading_bot.nds.analyzer import GoldNDSAnalyzer
from src.trading_bot.nds.distance_utils import pips_to_price
from src.trading_bot.nds.models import FVG, FVGType
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.risk_manager import create_scalping_risk_manager


def _make_df() -> pd.DataFrame:
    start = datetime(2025, 1, 1, 9, 0, 0)
    times = [start, start + timedelta(minutes=5)]
    return pd.DataFrame(
        {
            "time": times,
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1.0, 1.0],
        }
    )


def _build_config_payload() -> dict:
    cfg = config.get_full_config()
    cfg.setdefault("risk_manager_config", {})
    cfg.setdefault("risk_settings", {})
    cfg["risk_manager_config"]["MIN_RR_RATIO"] = 0.1
    cfg["risk_settings"].setdefault("RISK_AMOUNT_USD", 25.0)
    cfg.setdefault("ACCOUNT_BALANCE", 10_000.0)
    return cfg


def test_opposing_ifvg_blocked_for_countertrend_without_reversal():
    analyzer = GoldNDSAnalyzer(_make_df())
    fvg = FVG(
        type=FVGType.BEARISH,
        top=110.0,
        bottom=105.0,
        mid=107.5,
        time=datetime(2025, 1, 1, 9, 0, 0),
        index=1,
    )
    result = analyzer._resolve_opposing_structure_target(
        signal="BUY",
        entry_price=100.0,
        fvgs=[fvg],
        order_blocks=[],
        signal_context={"bias": "BEARISH", "trend": "DOWNTREND", "reversal_ok": False},
    )
    assert result["price"] is None
    assert result["reason"] == "aligned_ifvg_blocked_no_reversal"


def test_opposing_ifvg_allows_with_reversal_confirmation():
    analyzer = GoldNDSAnalyzer(_make_df())
    fvg = FVG(
        type=FVGType.BEARISH,
        top=110.0,
        bottom=105.0,
        mid=107.5,
        time=datetime(2025, 1, 1, 9, 0, 0),
        index=1,
    )
    result = analyzer._resolve_opposing_structure_target(
        signal="BUY",
        entry_price=100.0,
        fvgs=[fvg],
        order_blocks=[],
        signal_context={"bias": "BEARISH", "trend": "DOWNTREND", "reversal_ok": True},
    )
    assert result["price"] == 105.0
    assert result["zone_type"] == "IFVG"
    assert result["source"] == "BEARISH_FVG"


def test_tp1_policy_switches_countertrend_ifvg_to_fixed_pips():
    risk_manager = create_scalping_risk_manager(
        overrides={"FLOW_TP1_COUNTERTREND_IFVG_POLICY": "fixed_pips"}
    )
    analysis_payload = {
        "signal": "BUY",
        "confidence": 80.0,
        "entry_level": 2000.0,
        "entry_model": "MARKET",
        "entry_idea": {
            "entry_level": 2000.0,
            "entry_model": "MARKET",
            "entry_type": "FLOW",
        },
        "market_metrics": {"atr": 5.0},
        "analysis_signal_context": {"bias": "BEARISH", "trend": "DOWNTREND", "reversal_ok": False},
        "entry_context": {
            "tp1_target_price": 2010.0,
            "tp1_target_source": "BEARISH_FVG",
            "tp1_target_zone_type": "IFVG",
            "tp1_target_zone_direction": "BEARISH",
            "counter_trend": True,
            "reversal_ok": False,
            "liquidity_ok": False,
            "trend_ok": False,
        },
        "context": {
            "analysis_signal_context": {"bias": "BEARISH", "trend": "DOWNTREND", "reversal_ok": False},
        },
    }
    cfg = _build_config_payload()
    live_snapshot = LivePriceSnapshot(bid=2000.0, ask=2000.01, timestamp="2026-01-15T01:00:00")
    finalized = risk_manager.finalize_order(
        analysis=analysis_payload,
        live=live_snapshot,
        symbol="XAUUSD",
        config=cfg,
    )

    assert finalized.is_trade_allowed
    expected_tp = 2000.0 + pips_to_price(35.0, risk_manager._get_gold_spec("point", 0.01))
    assert abs(finalized.take_profit - expected_tp) < 1e-6
    assert any("TP1 policy: counter-trend aligned IFVG -> fixed_pips" in note for note in finalized.decision_notes)
