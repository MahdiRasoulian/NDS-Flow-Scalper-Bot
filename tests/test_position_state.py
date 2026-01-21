from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trading_bot.position_state import PositionStateStore


def _build_position(ticket: int, volume: float):
    now = datetime.utcnow()
    return {
        "position_ticket": ticket,
        "symbol": "XAUUSD",
        "side": "BUY",
        "volume": volume,
        "entry_price": 2000.0,
        "current_price": 2001.0,
        "sl": 1990.0,
        "tp": 2010.0,
        "profit": 5.0,
        "magic": 202401,
        "comment": "test",
        "open_time": now - timedelta(minutes=5),
        "update_time": now,
    }


def test_position_state_reconcile_detects_close_and_partial(tmp_path: Path):
    store = PositionStateStore(tmp_path / "positions.json")
    now = datetime.utcnow()

    positions = [_build_position(1001, 0.2)]
    result = store.reconcile(positions, now)
    assert result.new_positions
    assert result.closed_positions == []

    reduced_positions = [_build_position(1001, 0.1)]
    result = store.reconcile(reduced_positions, now + timedelta(seconds=30))
    assert result.partial_positions

    result = store.reconcile([], now + timedelta(seconds=60))
    assert result.closed_positions

    store.save()
    reloaded = PositionStateStore(tmp_path / "positions.json")
    reloaded.load()
    assert 1001 in reloaded.positions
