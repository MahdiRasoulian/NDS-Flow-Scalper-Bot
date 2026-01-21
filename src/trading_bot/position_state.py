"""Persistent state store for MT5 positions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.trading_bot.contracts import PositionContract


def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


@dataclass
class ReconcileResult:
    current_tickets: List[int]
    new_positions: List[PositionContract]
    closed_positions: List[Dict[str, Any]]
    partial_positions: List[Tuple[Dict[str, Any], float]]


class PositionStateStore:
    """Persisted state for open positions across restarts."""

    def __init__(self, path: Path):
        self.path = path
        self.positions: Dict[int, Dict[str, Any]] = {}
        self.last_reconcile_at: Optional[datetime] = None

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        raw_positions = payload.get("positions", {})
        parsed: Dict[int, Dict[str, Any]] = {}
        for key, record in raw_positions.items():
            try:
                ticket = int(key)
            except Exception:
                continue
            record = dict(record)
            record["open_time"] = _parse_datetime(record.get("open_time"))
            record["last_seen"] = _parse_datetime(record.get("last_seen"))
            record["last_reconcile"] = _parse_datetime(record.get("last_reconcile"))
            record["close_time"] = _parse_datetime(record.get("close_time"))
            parsed[ticket] = record
        self.positions = parsed
        self.last_reconcile_at = _parse_datetime(payload.get("last_reconcile_at"))

    def save(self) -> None:
        payload = {
            "last_reconcile_at": _serialize_datetime(self.last_reconcile_at),
            "positions": {
                str(ticket): {
                    **record,
                    "open_time": _serialize_datetime(record.get("open_time")),
                    "last_seen": _serialize_datetime(record.get("last_seen")),
                    "last_reconcile": _serialize_datetime(record.get("last_reconcile")),
                    "close_time": _serialize_datetime(record.get("close_time")),
                }
                for ticket, record in self.positions.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _build_state_record(self, position: PositionContract, now: datetime) -> Dict[str, Any]:
        return {
            "position_ticket": int(position["position_ticket"]),
            "symbol": position["symbol"],
            "magic": position.get("magic"),
            "comment": position.get("comment"),
            "side": position.get("side"),
            "volume": float(position.get("volume") or 0.0),
            "entry_price": float(position.get("entry_price") or 0.0),
            "open_time": position.get("open_time"),
            "status": "OPEN",
            "last_seen": now,
            "last_reconcile": now,
        }

    def reconcile(self, open_positions: List[PositionContract], now: datetime) -> ReconcileResult:
        current_tickets: List[int] = []
        new_positions: List[PositionContract] = []
        closed_positions: List[Dict[str, Any]] = []
        partial_positions: List[Tuple[Dict[str, Any], float]] = []

        open_map = {int(pos["position_ticket"]): pos for pos in open_positions}
        current_tickets = list(open_map.keys())

        for ticket, position in open_map.items():
            if ticket not in self.positions:
                new_positions.append(position)
                self.positions[ticket] = self._build_state_record(position, now)
                continue
            record = self.positions[ticket]
            prev_volume = float(record.get("volume") or 0.0)
            current_volume = float(position.get("volume") or 0.0)
            if current_volume < prev_volume and current_volume > 0.0:
                partial_positions.append((record, prev_volume - current_volume))
                record["status"] = "PARTIAL"
            record.update(
                {
                    "symbol": position.get("symbol"),
                    "magic": position.get("magic"),
                    "comment": position.get("comment"),
                    "side": position.get("side"),
                    "volume": current_volume,
                    "entry_price": float(position.get("entry_price") or 0.0),
                    "open_time": position.get("open_time"),
                    "last_seen": now,
                    "last_reconcile": now,
                }
            )

        for ticket, record in list(self.positions.items()):
            status = record.get("status")
            if status in ("OPEN", "PARTIAL") and ticket not in open_map:
                closed_positions.append(record)
                record["status"] = "CLOSED"
                record["close_time"] = now
                record["last_reconcile"] = now

        self.last_reconcile_at = now
        return ReconcileResult(
            current_tickets=current_tickets,
            new_positions=new_positions,
            closed_positions=closed_positions,
            partial_positions=partial_positions,
        )

