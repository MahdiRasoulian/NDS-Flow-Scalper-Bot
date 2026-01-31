from datetime import datetime, timedelta

import pytest

pd = pytest.importorskip("pandas")

from src.trading_bot.nds.analyzer import GoldNDSAnalyzer


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


def test_strict_quality_blocks_countertrend_low_rvol():
    analyzer = GoldNDSAnalyzer(_make_df())
    analyzer.GOLD_SETTINGS.update(
        {
            "STRICT_QUALITY_MODE": True,
            "COUNTERTREND_VERY_LOW_RVOL_MAX": 0.3,
        }
    )
    analysis_result = {
        "signal": "BUY",
        "confidence": 80.0,
        "reasons": [],
        "market_metrics": {"current_rvol": 0.2, "adx": 25.0},
        "structure": {"structure_score": 70.0, "bos": "BEARISH_BOS", "choch": "NONE"},
        "context": {
            "analysis_signal_context": {
                "bias": "BEARISH",
                "strong_trend": True,
                "reversal_ok": False,
            }
        },
    }

    filtered = analyzer._apply_final_filters(analysis_result, scalping_mode=True)
    assert filtered["signal"] == "NONE"
    assert "strict_quality_block:countertrend_low_rvol" in filtered["reasons"]


def test_soft_quality_allows_exceptional_countertrend_low_rvol():
    analyzer = GoldNDSAnalyzer(_make_df())
    analyzer.GOLD_SETTINGS.update(
        {
            "STRICT_QUALITY_MODE": False,
            "COUNTERTREND_VERY_LOW_RVOL_MAX": 0.3,
            "COUNTERTREND_LOW_RVOL_EXCEPTIONAL_STRUCTURE_SCORE": 85.0,
            "COUNTERTREND_LOW_RVOL_CONF_PENALTY": 0.8,
        }
    )
    analysis_result = {
        "signal": "BUY",
        "confidence": 80.0,
        "reasons": [],
        "market_metrics": {"current_rvol": 0.2, "adx": 25.0},
        "structure": {"structure_score": 90.0, "bos": "BEARISH_BOS", "choch": "NONE"},
        "context": {
            "analysis_signal_context": {
                "bias": "BEARISH",
                "strong_trend": True,
                "reversal_ok": False,
            }
        },
    }

    filtered = analyzer._apply_final_filters(analysis_result, scalping_mode=True)
    assert filtered["signal"] == "BUY"
    assert "countertrend_soft_allowed" in filtered["reasons"]
    assert filtered["confidence"] < 80.0
