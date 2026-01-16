from datetime import datetime

import pandas as pd
import pytest

from config.settings import config
from src.trading_bot.nds.analyzer import GoldNDSAnalyzer
from src.trading_bot.risk_manager import create_scalping_risk_manager
from src.trading_bot.session_policy import evaluate_session


def _build_df(end_ts: datetime, bars: int = 200) -> pd.DataFrame:
    times = pd.date_range(end=end_ts, periods=bars, freq="5min")
    base = 4600.0
    data = {
        "time": times,
        "open": [base] * bars,
        "high": [base + 1.0] * bars,
        "low": [base - 1.0] * bars,
        "close": [base + 0.2] * bars,
        "volume": [1000] * bars,
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "ts",
    [
        datetime(2026, 1, 15, 9, 55),
        datetime(2026, 1, 15, 10, 0),
        datetime(2026, 1, 15, 15, 0),
        datetime(2026, 1, 15, 22, 30),
    ],
)
def test_session_policy_consistency(ts: datetime):
    cfg = config.get_full_config()
    session_decision = evaluate_session(ts, cfg)

    analyzer = GoldNDSAnalyzer(_build_df(ts), config=cfg)
    session_analysis = analyzer._analyze_trading_sessions({"rvol": 1.0})

    assert session_analysis.current_session == session_decision.session_name
    assert session_analysis.session_weight == session_decision.weight

    risk_manager = create_scalping_risk_manager()
    resolved, source = risk_manager._resolve_session_decision(
        {"session_decision": session_decision.to_payload()}
    )
    assert source == "payload"
    assert resolved.session_name == session_decision.session_name
    assert resolved.weight == session_decision.weight
