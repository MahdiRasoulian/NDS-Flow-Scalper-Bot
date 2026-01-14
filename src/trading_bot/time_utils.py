"""
Unified time/session utilities for NDS Flow Scalper.

Canonical time mode: BROKER (server time, UTC+02 by default).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

DEFAULT_BROKER_OFFSET_HOURS = 2
DEFAULT_TIME_MODE = "BROKER"

DEFAULT_SESSION_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "ASIA": {"start": "02:00", "end": "10:00", "weight": 0.8, "allow_momentum": False},
    "LONDON": {"start": "10:00", "end": "19:00", "weight": 1.2, "allow_momentum": True},
    "NEW_YORK": {"start": "15:00", "end": "22:00", "weight": 1.3, "allow_momentum": True},
    "OVERLAP": {"start": "15:00", "end": "19:00", "weight": 1.5, "allow_momentum": True},
    "OTHER": {"start": "22:00", "end": "02:00", "weight": 0.4, "allow_momentum": False},
}


def parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            pass
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except (OSError, ValueError):
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
    return None


def _strip_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(tz=None).replace(tzinfo=None)


def to_broker_time(
    dt: datetime,
    offset_hours: float = DEFAULT_BROKER_OFFSET_HOURS,
    time_mode: str = DEFAULT_TIME_MODE,
) -> datetime:
    if dt is None:
        return None
    normalized = _strip_tz(dt)
    mode = str(time_mode or DEFAULT_TIME_MODE).upper()
    if mode == "UTC":
        return normalized + timedelta(hours=float(offset_hours))
    return normalized


def get_broker_now(offset_hours: float = DEFAULT_BROKER_OFFSET_HOURS) -> datetime:
    return datetime.utcnow() + timedelta(hours=float(offset_hours))


def minutes_from_midnight(dt: datetime) -> int:
    if dt is None:
        return 0
    return int(dt.hour) * 60 + int(dt.minute)


def _coerce_hhmm(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        hour = int(value)
        return f"{hour:02d}:00"
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if ":" not in raw:
            try:
                hour = int(raw)
                return f"{hour:02d}:00"
            except ValueError:
                return None
        parts = raw.split(":")
        if len(parts) >= 2:
            try:
                hour = int(parts[0])
                minute = int(parts[1])
                return f"{hour:02d}:{minute:02d}"
            except ValueError:
                return None
    return None


def in_time_window(dt: datetime, start_hhmm: str, end_hhmm: str) -> Tuple[bool, Optional[str]]:
    if dt is None:
        return False, "time_parse_failed"
    start = _coerce_hhmm(start_hhmm)
    end = _coerce_hhmm(end_hhmm)
    if not start or not end:
        return False, "time_parse_failed"
    try:
        start_h, start_m = [int(p) for p in start.split(":")]
        end_h, end_m = [int(p) for p in end.split(":")]
    except ValueError as exc:
        return False, f"time_parse_failed:{exc}"
    start_min = start_h * 60 + start_m
    end_min = end_h * 60 + end_m
    now_min = minutes_from_midnight(dt)
    if start_min == end_min:
        return False, "time_window_zero"
    if start_min < end_min:
        return start_min <= now_min < end_min, None
    return now_min >= start_min or now_min < end_min, None


def normalize_session_definitions(definitions: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(definitions, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_name, data in definitions.items():
        if not isinstance(data, dict):
            continue
        start = _coerce_hhmm(data.get("start"))
        end = _coerce_hhmm(data.get("end"))
        if not start or not end:
            continue
        name = str(raw_name).strip().upper() if raw_name else "UNKNOWN"
        normalized[name] = {
            "start": start,
            "end": end,
            "weight": float(data.get("weight", 0.5)),
            "allow_momentum": bool(data.get("allow_momentum", True)),
        }
    return normalized


def classify_session(
    dt: datetime,
    session_definitions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    definitions = normalize_session_definitions(session_definitions) if session_definitions else {}
    if not definitions:
        definitions = dict(DEFAULT_SESSION_DEFINITIONS)
    if dt is None:
        return {
            "session": "UNKNOWN",
            "weight": 0.0,
            "is_overlap": False,
            "allow_momentum": False,
        }
    matched = []
    for name, data in definitions.items():
        in_window, _ = in_time_window(dt, data.get("start"), data.get("end"))
        if in_window:
            matched.append((name, data))
    if not matched:
        fallback = definitions.get("OTHER")
        if fallback:
            return {
                "session": "OTHER",
                "weight": float(fallback.get("weight", 0.5)),
                "is_overlap": False,
                "allow_momentum": bool(fallback.get("allow_momentum", False)),
            }
        return {
            "session": "UNKNOWN",
            "weight": 0.0,
            "is_overlap": False,
            "allow_momentum": False,
        }
    if any(name == "OVERLAP" for name, _ in matched):
        overlap_data = next(data for name, data in matched if name == "OVERLAP")
        return {
            "session": "OVERLAP",
            "weight": float(overlap_data.get("weight", 1.0)),
            "is_overlap": True,
            "allow_momentum": bool(overlap_data.get("allow_momentum", True)),
        }
    if len(matched) > 1:
        name, data = max(matched, key=lambda item: float(item[1].get("weight", 0.0)))
        return {
            "session": name,
            "weight": float(data.get("weight", 0.5)),
            "is_overlap": True,
            "allow_momentum": bool(data.get("allow_momentum", True)),
        }
    name, data = matched[0]
    return {
        "session": name,
        "weight": float(data.get("weight", 0.5)),
        "is_overlap": False,
        "allow_momentum": bool(data.get("allow_momentum", True)),
    }
