"""
Canonical session policy evaluation for NDS Flow Scalper.

All session behavior must be driven by config/bot_config.json.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from src.trading_bot.config_utils import get_setting
from src.trading_bot.time_utils import (
    DEFAULT_BROKER_OFFSET_HOURS,
    DEFAULT_SESSION_DEFINITIONS,
    DEFAULT_TIME_MODE,
    classify_session,
    in_time_window,
    normalize_session_definitions,
    parse_timestamp,
    to_broker_time,
)


@dataclass
class SessionDecision:
    session_name: str
    is_tradable: bool
    block_reason: Optional[str]
    weight: float
    activity: str
    policy_mode: str
    is_overlap: bool
    ts_broker: Optional[datetime] = None
    time_mode: Optional[str] = None
    broker_utc_offset_hours: Optional[float] = None
    untradable_window: Optional[str] = None
    untradable_reason: Optional[str] = None
    policy_source: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "session_name": self.session_name,
            "is_tradable": self.is_tradable,
            "block_reason": self.block_reason,
            "weight": self.weight,
            "activity": self.activity,
            "policy_mode": self.policy_mode,
            "is_overlap": self.is_overlap,
            "ts_broker": self.ts_broker.isoformat() if self.ts_broker else None,
            "time_mode": self.time_mode,
            "broker_utc_offset_hours": self.broker_utc_offset_hours,
            "untradable_window": self.untradable_window,
            "untradable_reason": self.untradable_reason,
            "policy_source": self.policy_source,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SessionDecision":
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        ts_broker = parse_timestamp(payload.get("ts_broker"))
        return cls(
            session_name=str(payload.get("session_name", "UNKNOWN")),
            is_tradable=bool(payload.get("is_tradable", True)),
            block_reason=payload.get("block_reason"),
            weight=float(payload.get("weight", 0.0) or 0.0),
            activity=str(payload.get("activity", "NORMAL")),
            policy_mode=str(payload.get("policy_mode", "WEIGHT_ONLY")).upper(),
            is_overlap=bool(payload.get("is_overlap", False)),
            ts_broker=ts_broker,
            time_mode=payload.get("time_mode"),
            broker_utc_offset_hours=payload.get("broker_utc_offset_hours"),
            untradable_window=payload.get("untradable_window"),
            untradable_reason=payload.get("untradable_reason"),
            policy_source=payload.get("policy_source"),
        )


def _normalize_list(values: Any) -> Iterable[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values.strip().upper()]
    if isinstance(values, Iterable):
        normalized = []
        for item in values:
            if item is None:
                continue
            normalized.append(str(item).strip().upper())
        return normalized
    return []


def _resolve_policy_mode(config: Any) -> str:
    mode = str(
        get_setting(config, "trading_settings.SESSION_POLICY_MODE", "WEIGHT_ONLY")
    ).upper()
    if mode not in ("ALLOWLIST", "DENYLIST", "WEIGHT_ONLY"):
        return "WEIGHT_ONLY"
    return mode


def _resolve_session_definitions(config: Any) -> Dict[str, Dict[str, Any]]:
    definitions = get_setting(config, "trading_settings.SESSION_DEFINITIONS", None)
    if not definitions:
        definitions = get_setting(config, "sessions_config.BASE_TRADING_SESSIONS", None)
    definitions = normalize_session_definitions(definitions or {})
    return definitions or dict(DEFAULT_SESSION_DEFINITIONS)


def session_weight_from_config(session_name: str, config: Any) -> Optional[float]:
    if not session_name:
        return None
    weights = get_setting(config, "trading_settings.SESSION_WEIGHTS", None)
    if isinstance(weights, dict):
        normalized = {str(k).upper(): v for k, v in weights.items()}
        if session_name.upper() in normalized:
            try:
                return float(normalized[session_name.upper()])
            except Exception:
                return None
    definitions = _resolve_session_definitions(config)
    entry = definitions.get(session_name.upper())
    if isinstance(entry, dict) and "weight" in entry:
        try:
            return float(entry.get("weight", 0.0))
        except Exception:
            return None
    return None


def evaluate_session(ts_input: Any, config: Any) -> SessionDecision:
    policy_mode = _resolve_policy_mode(config)
    allowlist = set(_normalize_list(get_setting(config, "trading_settings.SESSION_ALLOWED", [])))
    denylist = set(_normalize_list(get_setting(config, "trading_settings.SESSION_BLOCKED", [])))
    failsafe = str(
        get_setting(config, "trading_settings.SESSION_FAILSAFE_MODE", "WEIGHT_ONLY")
    ).upper()
    time_mode = str(
        get_setting(config, "trading_settings.SESSION_TIME_MODE",
                    get_setting(config, "trading_settings.TIME_MODE", DEFAULT_TIME_MODE))
        or DEFAULT_TIME_MODE
    ).upper()
    broker_offset = float(
        get_setting(config, "trading_settings.BROKER_UTC_OFFSET_HOURS", DEFAULT_BROKER_OFFSET_HOURS)
        or DEFAULT_BROKER_OFFSET_HOURS
    )
    require_broker_time = bool(
        get_setting(config, "trading_settings.SESSION_REQUIRE_BROKER_TIME", True)
    )

    raw_ts = parse_timestamp(ts_input)
    if raw_ts is None:
        is_tradable = failsafe == "WEIGHT_ONLY"
        return SessionDecision(
            session_name="UNKNOWN",
            is_tradable=is_tradable,
            block_reason=None if is_tradable else "session_time_unavailable",
            weight=0.0,
            activity="NORMAL",
            policy_mode=policy_mode,
            is_overlap=False,
            ts_broker=None,
            time_mode=time_mode,
            broker_utc_offset_hours=broker_offset,
            policy_source="config",
        )

    ts_broker = to_broker_time(raw_ts, broker_offset, time_mode)
    if require_broker_time and time_mode != "BROKER":
        is_tradable = failsafe == "WEIGHT_ONLY"
        return SessionDecision(
            session_name="UNKNOWN",
            is_tradable=is_tradable,
            block_reason=None if is_tradable else "non_broker_time_mode",
            weight=0.0,
            activity="NORMAL",
            policy_mode=policy_mode,
            is_overlap=False,
            ts_broker=ts_broker,
            time_mode=time_mode,
            broker_utc_offset_hours=broker_offset,
            policy_source="config",
        )

    definitions = _resolve_session_definitions(config)
    session_info = classify_session(ts_broker, definitions)
    session_name = str(session_info.get("session", "UNKNOWN")).upper()
    weight = float(session_info.get("weight", 0.0) or 0.0)
    weight_override = session_weight_from_config(session_name, config)
    if weight_override is not None:
        weight = float(weight_override)

    is_overlap = bool(session_info.get("is_overlap", False))
    is_tradable = True
    block_reason = None
    untradable_window = None
    untradable_reason = None

    windows = get_setting(config, "trading_settings.SESSION_UNTRADABLE_WINDOWS", []) or []
    if isinstance(windows, list):
        for window in windows:
            if not isinstance(window, dict):
                continue
            start = window.get("start")
            end = window.get("end")
            in_window, _ = in_time_window(ts_broker, start, end)
            if in_window:
                is_tradable = False
                untradable_window = f"{start}-{end}"
                untradable_reason = str(window.get("reason", "untradable_window"))
                block_reason = untradable_reason
                break

    if session_name == "UNKNOWN":
        if failsafe != "WEIGHT_ONLY":
            is_tradable = False
            block_reason = "session_unknown"

    if policy_mode == "ALLOWLIST":
        if allowlist and session_name not in allowlist:
            is_tradable = False
            block_reason = f"Non-optimal session: {session_name}"
    elif policy_mode == "DENYLIST":
        if session_name in denylist:
            is_tradable = False
            block_reason = f"Non-optimal session: {session_name}"

    return SessionDecision(
        session_name=session_name,
        is_tradable=is_tradable,
        block_reason=block_reason,
        weight=weight,
        activity="NORMAL",
        policy_mode=policy_mode,
        is_overlap=is_overlap,
        ts_broker=ts_broker,
        time_mode=time_mode,
        broker_utc_offset_hours=broker_offset,
        untradable_window=untradable_window,
        untradable_reason=untradable_reason,
        policy_source="config",
    )
