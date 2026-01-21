"""Stable data contracts and helpers for MT5 execution flow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional, TypedDict

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover - optional import for offline tests
    mt5 = None

from src.trading_bot.nds.distance_utils import (
    calculate_distance_metrics,
    DEFAULT_POINT_SIZE,
    pips_to_price,
    resolve_point_size_with_source,
)


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
            return datetime.utcfromtimestamp(value)
        if hasattr(value, "timestamp"):
            return datetime.utcfromtimestamp(value.timestamp())
    except Exception:
        return None
    return None


def normalize_position(raw: Dict[str, Any]) -> PositionContract:
    """Normalize a raw MT5 position dict/object into the PositionContract."""
    position_ticket = raw.get("position_ticket", raw.get("ticket", 0)) or 0
    side = raw.get("side", raw.get("type", "BUY"))
    side = "BUY" if str(side).upper() == "BUY" else "SELL"

    # Use UTC as a safe fallback to avoid local-time skew in history queries.
    open_time = _coerce_datetime(raw.get("open_time", raw.get("time"))) or datetime.utcnow()
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


def _infer_pip_size(symbol: str, config_payload: Optional[Dict[str, Any]] = None) -> float:
    """Deprecated: use calculate_distance_metrics with resolved point_size instead."""
    point_size, source = resolve_point_size_with_source(config_payload, default=None)
    if source == "default" and mt5 is not None:
        try:
            info = mt5.symbol_info(symbol)
            if info and info.point:
                point_size = float(info.point)
        except Exception:
            pass
    if not point_size or point_size <= 0:
        point_size = DEFAULT_POINT_SIZE
    return pips_to_price(1.0, point_size)


def compute_pips(
    symbol: str,
    entry: float,
    exit: float,
    config_payload: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute pips between entry and exit using centralized distance utilities."""
    if not entry or not exit:
        return 0.0
    point_size = None
    if config_payload is not None:
        point_size, source = resolve_point_size_with_source(config_payload, default=None)
        if source == "default":
            point_size = None
    if point_size is None and mt5 is not None:
        try:
            info = mt5.symbol_info(symbol)
            if info and info.point:
                point_size = float(info.point)
        except Exception:
            point_size = None
    if not point_size or point_size <= 0:
        point_size = DEFAULT_POINT_SIZE
    metrics = calculate_distance_metrics(
        entry_price=entry,
        current_price=exit,
        point_size=point_size,
    )
    return float(metrics.get("dist_pips") or 0.0)
