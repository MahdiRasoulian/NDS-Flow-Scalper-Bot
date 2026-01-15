"""
Centralized config access + resolved settings logging for NDS Flow Scalper.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from src.trading_bot.nds.distance_utils import DEFAULT_POINT_SIZE
from src.trading_bot.time_utils import DEFAULT_BROKER_OFFSET_HOURS, DEFAULT_SESSION_DEFINITIONS, DEFAULT_TIME_MODE

logger = logging.getLogger(__name__)


DEFAULT_ACTIVE_SETTINGS: Dict[str, Dict[str, Any]] = {
    "trading_settings": {
        "POINT_SIZE": DEFAULT_POINT_SIZE,
        "BROKER_UTC_OFFSET_HOURS": DEFAULT_BROKER_OFFSET_HOURS,
        "TIME_MODE": DEFAULT_TIME_MODE,
        "SESSION_DEFINITIONS": DEFAULT_SESSION_DEFINITIONS,
    },
    "flow_settings": {
        "BRK_MAX_DIST_ATR": 1.2,
        "BRK_MAX_AGE_BARS": 45,
        "IFVG_MAX_DIST_ATR": 3.0,
        "IFVG_MAX_AGE_BARS": 45,
        "FLOW_MAX_TOUCHES": 2,
        "FLOW_TOUCH_EXIT_ATR": 0.2,
        "FLOW_TOUCH_EXIT_PIPS": 5.0,
        "FLOW_TOUCH_PENETRATION_ATR": 0.05,
        "FLOW_TOUCH_PENALTY": 0.55,
        "FLOW_RETEST_POLICY": "FIRST_TOUCH",
        "FLOW_TOUCH_MIN_SEPARATION_BARS": 6,
        "FLOW_TOUCH_EXIT_CONFIRM_BARS": 2,
        "FLOW_TOUCH_COUNT_WINDOW_BARS": 200,
        "FLOW_CONSUME_ON_FIRST_VALID_TOUCH": True,
        "FLOW_NEAREST_ZONES": 5,
    },
    "momentum_settings": {
        "MOMO_ADX_MIN": 35.0,
        "MOMO_TIME_START": "10:00",
        "MOMO_TIME_END": "18:00",
        "MOMO_SESSION_ONLY": True,
        "MOMO_SESSION_ALLOWLIST": ["LONDON", "NEW_YORK", "OVERLAP"],
        "MOMO_BUFFER_ATR_MULT": 0.1,
        "MOMO_BUFFER_MIN_PIPS": 1.0,
    },
    "risk_settings": {
        "SCALP_ATR_SL_MULT": 1.5,
        "SL_MIN_PIPS": 10.0,
        "MIN_SL_PIPS": 20.0,
        "SL_MAX_PIPS": 40.0,
        "TP1_PIPS": 35.0,
        "TP2_ENABLED": True,
        "TP2_PIPS": 70.0,
        "SPREAD_MAX_PIPS": 2.5,
    },
}


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


def _merge_with_defaults(
    defaults: Dict[str, Any],
    overrides: Any,
    prefix: str = "",
) -> Tuple[Dict[str, Any], List[str]]:
    if not isinstance(overrides, dict):
        overrides = {}
    merged: Dict[str, Any] = {}
    defaulted: List[str] = []
    for key, default_value in defaults.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(default_value, dict):
            nested_override = overrides.get(key, {}) if isinstance(overrides, dict) else {}
            nested_merged, nested_defaulted = _merge_with_defaults(
                default_value,
                nested_override,
                prefix=path,
            )
            merged[key] = nested_merged
            defaulted.extend(nested_defaulted)
        else:
            if isinstance(overrides, dict) and key in overrides and overrides.get(key) is not None:
                merged[key] = overrides.get(key)
            else:
                merged[key] = default_value
                defaulted.append(path)
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key not in merged:
                merged[key] = value
    return merged, defaulted


def resolve_active_settings(config_payload: Any) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    defaults_used: List[str] = []
    for section, defaults in DEFAULT_ACTIVE_SETTINGS.items():
        overrides = get_setting(config_payload, section, {})
        merged, defaulted = _merge_with_defaults(defaults, overrides, prefix=section)
        resolved[section] = merged
        defaults_used.extend(defaulted)
    return {
        "resolved": resolved,
        "defaults_used": defaults_used,
    }


def log_active_settings(config_payload: Any, log: logging.Logger | None = None) -> Dict[str, Any]:
    log = log or logger
    payload = resolve_active_settings(config_payload)
    message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    log.info("[NDS][CONFIG_ACTIVE] %s", message)
    return payload
