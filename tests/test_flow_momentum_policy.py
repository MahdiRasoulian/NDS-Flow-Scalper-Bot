from datetime import datetime, timedelta

import pytest

pd = pytest.importorskip("pandas")

from src.trading_bot.nds.analyzer import GoldNDSAnalyzer
from src.trading_bot.nds.models import MarketStructure, MarketTrend, SessionAnalysis


def _make_df() -> pd.DataFrame:
    start = datetime(2025, 1, 1, 10, 0, 0)
    times = [start, start + timedelta(minutes=5)]
    try:
        return pd.DataFrame(
            {
                "time": times,
                "open": [100.0, 100.5],
                "high": [101.0, 101.2],
                "low": [99.0, 99.5],
                "close": [100.2, 100.0],
                "volume": [1.0, 1.0],
            }
        )
    except TypeError:
        pytest.skip("pandas DataFrame constructor unavailable in this environment")


def _make_structure(current_price: float) -> MarketStructure:
    return MarketStructure(
        trend=MarketTrend.DOWNTREND,
        bos="BEARISH_BOS",
        choch="NONE",
        last_high=None,
        last_low=None,
        current_price=current_price,
        structure_score=80.0,
    )


def _make_session() -> SessionAnalysis:
    return SessionAnalysis(
        current_session="LONDON",
        session_weight=1.2,
        weight=1.2,
        gmt_hour=10,
        is_active_session=True,
        is_overlap=False,
        session_activity="HIGH",
        optimal_trading=True,
        is_tradable=True,
    )


def test_momentum_proximity_gate_rejects_far_entries():
    analyzer = GoldNDSAnalyzer(_make_df())
    analyzer._point_size = 0.01
    analyzer.GOLD_SETTINGS.update(
        {
            "MOMO_ADX_MIN": 10.0,
            "MOMO_TIME_START": "00:00",
            "MOMO_TIME_END": "23:59",
            "MOMO_SESSION_ONLY": False,
            "FLOW_MOMENTUM_MAX_DIST_ATR": 0.35,
            "FLOW_MOMENTUM_MAX_DIST_PIPS": 60.0,
        }
    )

    entry = analyzer._select_flow_entry(
        signal="SELL",
        structure=_make_structure(current_price=115.0),
        current_price=115.0,
        atr_value=1.0,
        adx_value=40.0,
        session_analysis=_make_session(),
        volume_analysis={"rvol": 1.0, "market_status": "OPEN"},
        scalping_mode=True,
        signal_context={"bias": "BEARISH", "trend": "DOWNTREND", "adx": 40.0},
        log_decisions=False,
    )

    assert entry["signal"] == "NONE"
    assert entry["reject_reason"] == "MOMO_ENTRY_TOO_FAR"


def test_strong_trend_sell_promotes_momentum_without_base_signal():
    analyzer = GoldNDSAnalyzer(_make_df())
    analyzer._point_size = 0.01
    analyzer.GOLD_SETTINGS.update(
        {
            "STRICT_QUALITY_MODE": True,
            "FLOW_REQUIRES_BASE_SIGNAL": True,
            "FLOW_ALLOW_MOMO_WITHOUT_BASE_SIGNAL": True,
            "ADX_STRONG_TREND_MIN": 30.0,
            "MOMO_ADX_MIN": 10.0,
            "MOMO_TIME_START": "00:00",
            "MOMO_TIME_END": "23:59",
            "MOMO_SESSION_ONLY": False,
            "FLOW_MOMENTUM_MAX_DIST_ATR": 1.5,
            "FLOW_MOMENTUM_MAX_DIST_PIPS": 200.0,
        }
    )

    structure = _make_structure(current_price=100.0)
    session = _make_session()
    market_metrics = {
        "signal": "NONE",
        "current_price": 100.0,
        "atr": 1.0,
        "adx": 35.0,
    }
    signal_context = {
        "bias": "BEARISH",
        "trend": "DOWNTREND",
        "adx": 35.0,
        "strong_trend": True,
        "reversal_ok": False,
    }
    volume_analysis = {"rvol": 1.0, "market_status": "OPEN"}

    entry = analyzer.select_entry_idea(
        df=analyzer.df,
        structure=structure,
        market_metrics=market_metrics,
        session_analysis=session,
        signal_context=signal_context,
        volume_analysis=volume_analysis,
        scalping_mode=True,
        entry_factor=1.0,
        fvgs=[],
        order_blocks=[],
    )

    assert entry["signal"] == "SELL"
    assert entry["entry_type"] == "MOMENTUM"


def test_flow_setup_touch_parabola_and_strength_gate():
    analyzer = GoldNDSAnalyzer(_make_df())
    analyzer.GOLD_SETTINGS.update(
        {
            "FLOW_PROXIMITY_GAUSS_SIGMA": 0.35,
            "FLOW_SETUP_WEIGHTS": {
                "retest_quality": 0.25,
                "freshness": 0.2,
                "proximity": 0.2,
                "displacement": 0.15,
                "trend_alignment": 0.1,
                "liquidity": 0.1,
            },
        }
    )

    base_zone = {"retest_reason": "CLOSE_RECLAIM", "disp_atr": 1.0}
    score_touch_2 = analyzer._score_flow_setup(
        zone={**base_zone, "touch_count": 2},
        dist_atr=0.1,
        max_dist_atr=1.0,
        signal="BUY",
        session_analysis=_make_session(),
        volume_analysis={"rvol": 1.0},
        signal_context={"bias": "BULLISH"},
    )
    score_touch_4 = analyzer._score_flow_setup(
        zone={**base_zone, "touch_count": 4},
        dist_atr=0.1,
        max_dist_atr=1.0,
        signal="BUY",
        session_analysis=_make_session(),
        volume_analysis={"rvol": 1.0},
        signal_context={"bias": "BULLISH"},
    )

    assert score_touch_2["freshness"] == 1.0
    assert score_touch_4["freshness"] == 0.2
    assert score_touch_4["setup_score"] < score_touch_2["setup_score"]
    assert score_touch_4["setup_score"] == pytest.approx(
        score_touch_4["additive_score"] * score_touch_4["strength_gate"]
    )


def test_flow_setup_gaussian_proximity_rewards_edge_location():
    analyzer = GoldNDSAnalyzer(_make_df())
    analyzer.GOLD_SETTINGS.update({"FLOW_PROXIMITY_GAUSS_SIGMA": 0.35})

    near_edge = analyzer._score_flow_setup(
        zone={"touch_count": 1, "retest_reason": "CLOSE_RECLAIM", "disp_atr": 1.0},
        dist_atr=0.1,
        max_dist_atr=1.0,
        signal="BUY",
        session_analysis=_make_session(),
        volume_analysis={"rvol": 1.0},
        signal_context={"bias": "BULLISH"},
    )
    center_like = analyzer._score_flow_setup(
        zone={"touch_count": 1, "retest_reason": "CLOSE_RECLAIM", "disp_atr": 1.0},
        dist_atr=1.0,
        max_dist_atr=1.0,
        signal="BUY",
        session_analysis=_make_session(),
        volume_analysis={"rvol": 1.0},
        signal_context={"bias": "BULLISH"},
    )

    assert near_edge["proximity"] > center_like["proximity"]
