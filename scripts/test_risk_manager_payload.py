from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.trading_bot.risk_manager import ScalpingRiskManager


def main() -> None:
    risk_manager = ScalpingRiskManager(overrides={})
    sample_payload = {
        "signal": "SELL",
        "confidence": 61.9,
        "score": 55.0,
        "session": "ASIA",
        "session_activity": "LOW",
        "ts_broker": "2026-01-16 03:00:00",
        "time_mode": "BROKER",
        "broker_utc_offset_hours": 2,
        "adx": 16.1,
    }

    risk_manager.can_scalp(account_equity=10_000.0, signal_data=sample_payload)

    assert risk_manager.last_session == "ASIA", (
        f"Expected session ASIA, got {risk_manager.last_session}"
    )
    assert abs(risk_manager.last_adx - 16.1) < 1e-6, (
        f"Expected ADX 16.1, got {risk_manager.last_adx}"
    )

    print("✅ RiskManager payload propagation test passed.")


if __name__ == "__main__":
    main()
