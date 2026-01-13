"""Stable data contracts and helpers for MT5 execution flow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional, TypedDict

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover - optional import for offline tests
    mt5 = None


class PositionContract(TypedDict):
    position_ticket: int
    symbol: str
    side: Literal["BUY", "SELL"]
    volume: float
    entry_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    magic: int
    comment: str
    open_time: datetime
    update_time: Optional[datetime]


class TradeIdentity(TypedDict):
    order_ticket: Optional[int]
    position_ticket: Optional[int]
    symbol: str
    magic: Optional[int]
    comment: Optional[str]
    opened_at: datetime
    detected_by: str


class ExecutionEvent(TypedDict):
    event_type: Literal["OPEN", "UPDATE", "CLOSE", "ERROR"]
    event_time: datetime
    symbol: str
    order_ticket: Optional[int]
    position_ticket: Optional[int]
    side: Optional[str]
    volume: Optional[float]
    entry_price: Optional[float]
    exit_price: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    profit: Optional[float]
    pips: Optional[float]
    reason: Optional[str]
    metadata: Dict[str, Any]


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if hasattr(value, "timestamp"):
            return datetime.fromtimestamp(value.timestamp())
    except Exception:
        return None
    return None


def normalize_position(raw: Dict[str, Any]) -> PositionContract:
    """Normalize a raw MT5 position dict/object into the PositionContract."""
    position_ticket = raw.get("position_ticket", raw.get("ticket", 0)) or 0
    side = raw.get("side", raw.get("type", "BUY"))
    side = "BUY" if str(side).upper() == "BUY" else "SELL"

    open_time = _coerce_datetime(raw.get("open_time", raw.get("time"))) or datetime.now()
    update_time = _coerce_datetime(raw.get("update_time", raw.get("time_update")))

    return PositionContract(
        position_ticket=int(position_ticket),
        symbol=str(raw.get("symbol") or ""),
        side=side,
        volume=float(raw.get("volume", 0.0) or 0.0),
        entry_price=float(raw.get("entry_price", raw.get("price_open", 0.0)) or 0.0),
        current_price=float(raw.get("current_price", raw.get("price_current", 0.0)) or 0.0),
        sl=float(raw.get("sl", 0.0) or 0.0),
        tp=float(raw.get("tp", 0.0) or 0.0),
        profit=float(raw.get("profit", 0.0) or 0.0),
        magic=int(raw.get("magic", 0) or 0),
        comment=str(raw.get("comment") or ""),
        open_time=open_time,
        update_time=update_time,
    )


def _infer_pip_size(symbol: str) -> float:
    symbol_upper = symbol.upper()
    if "XAU" in symbol_upper or "GOLD" in symbol_upper:
        return 0.10

    if mt5 is not None:
        try:
            info = mt5.symbol_info(symbol)
            if info and info.point:
                if info.digits in (3, 5):
                    return info.point * 10
                return info.point
        except Exception:
            pass

    return 0.0001


def compute_pips(symbol: str, entry: float, exit: float) -> float:
    """Compute pips between entry and exit using symbol info when available."""
    if not entry or not exit:
        return 0.0
    pip_size = _infer_pip_size(symbol)
    if pip_size <= 0:
        return 0.0
    return abs(exit - entry) / pip_size
