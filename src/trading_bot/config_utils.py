"""
Centralized config access + resolved settings logging for NDS Flow Scalper.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

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
