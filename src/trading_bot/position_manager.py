"""Unified PositionManager entrypoint.

This module intentionally exposes only the deterministic finite-state-machine
implementation and deprecates the legacy imperative manager.
"""

from src.trading_bot.position_manager_state_machine import (  # noqa: F401
    PositionManager,
    PositionPlan,
    PositionStatus,
)
