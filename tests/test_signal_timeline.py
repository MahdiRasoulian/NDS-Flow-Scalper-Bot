from datetime import datetime, timedelta

import pytest

from src.trading_bot.nds.analyzer import GoldNDSAnalyzer

pd = pytest.importorskip("pandas")


def _build_analyzer() -> GoldNDSAnalyzer:
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
    try:
        df = pd.DataFrame(data)
    except TypeError:
        pytest.skip("pandas DataFrame constructor unavailable in this environment")
    return GoldNDSAnalyzer(df, config={})


def test_signal_timeline_flow_override_flip():
    analyzer = _build_analyzer()
    entry_idea = {
        "signal": "SELL",
        "reason": "flow_override",
        "metrics": {
            "override": True,
            "override_reason": "flow_override",
            "override_bypassed_gates": ["liquidity_ok"],
        },
    }
    signal_context = {"bias": "BULLISH", "strong_trend": True, "reversal_ok": False}
    timeline = analyzer._build_signal_timeline(
        pre_signal="BUY",
        post_filter_signal="NONE",
        entry_idea=entry_idea,
        signal_context=signal_context,
    )

    assert timeline["pre_signal"] == "BUY"
    assert timeline["post_filter_signal"] == "NONE"
    assert timeline["override_signal"] == "SELL"
    assert timeline["final_signal"] == "SELL"
    assert timeline["override_reason"] == "flow_override"
    assert timeline["bypassed_gates"] == ["liquidity_ok"]


def test_signal_timeline_countertrend_requires_reversal():
    analyzer = _build_analyzer()
    entry_idea = {"signal": "SELL", "metrics": {}}
    signal_context = {"bias": "BULLISH", "strong_trend": True, "reversal_ok": False}
    timeline = analyzer._build_signal_timeline(
        pre_signal="BUY",
        post_filter_signal="SELL",
        entry_idea=entry_idea,
        signal_context=signal_context,
    )

    assert timeline["counter_trend_flag"] is True
    assert timeline["required_reversal_confirmation"] is True
    assert timeline["reversal_ok"] is False
