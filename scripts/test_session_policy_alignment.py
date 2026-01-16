from __future__ import annotations

import copy
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import config
from src.trading_bot.risk_manager import ScalpingRiskManager
from src.trading_bot.session_policy import evaluate_session


def _config_with_policy(mode: str, allowed: list[str]) -> dict:
    base = copy.deepcopy(config.get_full_config())
    trading_settings = base.get("trading_settings", {})
    trading_settings["SESSION_POLICY_MODE"] = mode
    trading_settings["SESSION_ALLOWED"] = allowed
    trading_settings["SESSION_BLOCKED"] = []
    trading_settings["SESSION_FAILSAFE_MODE"] = "WEIGHT_ONLY"
    trading_settings["SESSION_TIME_MODE"] = "BROKER"
    trading_settings["SESSION_REQUIRE_BROKER_TIME"] = True
    trading_settings["SESSION_STRICT_ASSERT_MATCH"] = True
    base["trading_settings"] = trading_settings
    return base


def main() -> None:
    ny_ts = "2026-01-16 20:00:00"

    allow_config = _config_with_policy("ALLOWLIST", ["LONDON", "NEW_YORK"])
    decision = evaluate_session(ny_ts, allow_config)
    assert decision.is_tradable, f"Expected NEW_YORK tradable, got {decision}"

    risk_manager = ScalpingRiskManager(overrides={"trading_settings": allow_config["trading_settings"]})
    payload = {
        "signal": "BUY",
        "confidence": 80.0,
        "score": 70.0,
        "session_decision": decision.to_payload(),
        "ts_broker": ny_ts,
        "time_mode": "BROKER",
        "broker_utc_offset_hours": 2,
        "adx": 25.0,
    }
    can_trade, reason = risk_manager.can_scalp(account_equity=10_000.0, signal_data=payload)
    assert can_trade, f"Expected trade allowed, got {reason}"

    block_config = _config_with_policy("ALLOWLIST", ["LONDON"])
    blocked_decision = evaluate_session(ny_ts, block_config)
    assert not blocked_decision.is_tradable, "Expected NEW_YORK blocked in allowlist"
    assert "Non-optimal session" in (blocked_decision.block_reason or ""), blocked_decision.block_reason

    risk_manager_block = ScalpingRiskManager(overrides={"trading_settings": block_config["trading_settings"]})
    blocked_payload = dict(payload)
    blocked_payload["session_decision"] = blocked_decision.to_payload()
    can_trade_blocked, reason_blocked = risk_manager_block.can_scalp(
        account_equity=10_000.0,
        signal_data=blocked_payload,
    )
    assert not can_trade_blocked, "Expected trade blocked for NEW_YORK in allowlist"
    assert "Non-optimal session" in reason_blocked, reason_blocked

    weight_config = _config_with_policy("WEIGHT_ONLY", [])
    weight_decision = evaluate_session(ny_ts, weight_config)
    assert weight_decision.is_tradable, "WEIGHT_ONLY mode should not block sessions"

    risk_manager_weight = ScalpingRiskManager(overrides={"trading_settings": weight_config["trading_settings"]})
    weight_payload = dict(payload)
    weight_payload["session_decision"] = weight_decision.to_payload()
    can_trade_weight, reason_weight = risk_manager_weight.can_scalp(
        account_equity=10_000.0,
        signal_data=weight_payload,
    )
    assert can_trade_weight, f"Expected trade allowed in WEIGHT_ONLY, got {reason_weight}"

    print("✅ Session policy alignment test passed.")


if __name__ == "__main__":
    main()
