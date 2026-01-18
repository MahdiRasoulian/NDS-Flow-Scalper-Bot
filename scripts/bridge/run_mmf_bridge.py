"""CLI entry point for the NDS MMF bridge."""
from __future__ import annotations

import logging
from typing import Callable

from src.bridge.mmf_bridge import BridgeConfig, JsonLoggingDecisionWrapper, MMFBridgeServer
from src.bridge.mmf_protocol import ExecutionCommand, MarketSnapshot

logging.basicConfig(level=logging.INFO)


def dummy_decision(snapshot: MarketSnapshot) -> ExecutionCommand:
    return ExecutionCommand(
        action="NONE",
        entry=0.0,
        sl=0.0,
        tp=0.0,
        volume=0.0,
        confidence=0.0,
        reason_codes=["NO_SIGNAL"],
    )


def run(decision_fn: Callable[[MarketSnapshot], ExecutionCommand]) -> None:
    server = MMFBridgeServer(BridgeConfig())
    server.run(JsonLoggingDecisionWrapper(decision_fn))


if __name__ == "__main__":
    run(dummy_decision)
