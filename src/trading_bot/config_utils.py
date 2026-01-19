"""
Centralized config access + resolved settings logging for NDS Flow Scalper.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

ACTIVE_CONFIG_SECTIONS = (
    "trading_settings",
    "flow_settings",
    "momentum_settings",
    "risk_settings",
    "technical_settings",
    "sessions_config",
    "risk_manager_config",
)


def get_nested(config_payload: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(config_payload, dict):
        return default
    parts = key.split(".") if key else []
    value: Any = config_payload
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return default if value is None else value


def get_setting(config_payload: Any, key: str, default: Any = None) -> Any:
    if config_payload is None:
        return default
    if isinstance(config_payload, dict):
        return get_nested(config_payload, key, default)
    getter = getattr(config_payload, "get", None)
    if callable(getter):
        try:
            value = getter(key, default)
            return default if value is None else value
        except TypeError:
            return default
    return default


def resolve_active_settings(config_payload: Any) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    missing_sections: List[str] = []
    for section in ACTIVE_CONFIG_SECTIONS:
        section_payload = get_setting(config_payload, section, None)
        if not isinstance(section_payload, dict):
            section_payload = {}
            missing_sections.append(section)
        resolved[section] = section_payload
    return {
        "resolved": resolved,
        "missing_sections": missing_sections,
    }


def log_active_settings(config_payload: Any, log: logging.Logger | None = None) -> Dict[str, Any]:
    log = log or logger
    payload = resolve_active_settings(config_payload)
    message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    log.info("[NDS][CONFIG_ACTIVE] %s", message)
    return payload


def _load_mt5_creds_from_config(config_payload: Any) -> Dict[str, Any]:
    if config_payload is None:
        return {}

    getter = getattr(config_payload, "get_mt5_credentials", None)
    if callable(getter):
        try:
            creds = getter()
            if isinstance(creds, dict):
                return dict(creds)
        except Exception:
            return {}

    if isinstance(config_payload, dict):
        for key in ("mt5_credentials", "MT5_CREDENTIALS"):
            creds = config_payload.get(key)
            if isinstance(creds, dict):
                return dict(creds)

    creds = get_setting(config_payload, "mt5_credentials", None)
    if isinstance(creds, dict):
        return dict(creds)

    creds = get_setting(config_payload, "MT5_CREDENTIALS", None)
    if isinstance(creds, dict):
        return dict(creds)

    return {}


def _load_mt5_creds_from_file(file_path: Path) -> Dict[str, Any]:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_mt5_creds_from_env(env: Dict[str, str]) -> Dict[str, Any]:
    return {
        "login": env.get("MT5_LOGIN"),
        "password": env.get("MT5_PASSWORD"),
        "server": env.get("MT5_SERVER"),
        "mt5_path": env.get("MT5_PATH"),
        "real_time_enabled": env.get("MT5_REAL_TIME_ENABLED"),
        "tick_update_interval": env.get("MT5_TICK_UPDATE_INTERVAL"),
    }


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_mt5_credentials(
    config_payload: Any,
    credential_paths: Optional[Iterable[Path]] = None,
    env: Optional[Dict[str, str]] = None,
    log: logging.Logger | None = None,
) -> Dict[str, Any]:
    """Resolve MT5 credentials from env -> config -> file, with merged output."""
    log = log or logger
    env = env or os.environ

    sources: List[str] = []
    resolved: Dict[str, Any] = {}

    file_paths = list(credential_paths or [])
    for file_path in file_paths:
        if file_path.exists():
            file_creds = _load_mt5_creds_from_file(file_path)
            if file_creds:
                resolved.update(file_creds)
                sources.append(f"file:{file_path.name}")
                break

    config_creds = _load_mt5_creds_from_config(config_payload)
    if config_creds:
        resolved.update(config_creds)
        sources.append("central_config")

    env_creds = _load_mt5_creds_from_env(env)
    env_filtered = {k: v for k, v in env_creds.items() if v not in (None, "")}
    if env_filtered:
        resolved.update(env_filtered)
        sources.append("env")

    if "real_time_enabled" in resolved:
        resolved["real_time_enabled"] = _coerce_bool(resolved.get("real_time_enabled"))

    if "tick_update_interval" in resolved:
        resolved["tick_update_interval"] = _coerce_float(resolved.get("tick_update_interval"))

    required_keys = ("login", "password", "server")
    is_complete = all(resolved.get(k) not in (None, "") for k in required_keys)

    return {
        "credentials": resolved,
        "sources": sources,
        "is_complete": is_complete,
    }
