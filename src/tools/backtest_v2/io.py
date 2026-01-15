from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from src.trading_bot.config_utils import get_setting

REQUIRED_COLS = ["time", "open", "high", "low", "close"]
TIME_ALIASES = ["time", "timestamp", "date", "datetime"]


def _infer_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_override(override: str) -> Dict[str, Any]:
    if "=" not in override:
        raise ValueError(f"Invalid override: {override}")
    key, raw_value = override.split("=", 1)
    value = _infer_value(raw_value)
    parts = key.split(".")
    out: Dict[str, Any] = {}
    cursor = out
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return out


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(updated.get(key), dict):
            updated[key] = deep_update(updated[key], value)
        else:
            updated[key] = value
    return updated


def apply_overrides(config: Dict[str, Any], overrides: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(config)
    for override in overrides:
        merged = deep_update(merged, override)
    return merged


def _coerce_time_column(df: pd.DataFrame) -> pd.DataFrame:
    columns = {str(c).strip().lower(): c for c in df.columns}
    for alias in TIME_ALIASES:
        if alias in columns:
            column_name = columns[alias]
            if column_name != "time":
                df = df.rename(columns={column_name: "time"})
            break
    return df


def load_ohlcv(path: str, dayfirst: bool = False) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if source.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(source)
    elif source.suffix.lower() == ".csv":
        df = pd.read_csv(source)
    else:
        raise ValueError("Unsupported file type. Use .xlsx/.xls/.csv")

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = _coerce_time_column(df)

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        else:
            df["volume"] = 0

    df["time"] = pd.to_datetime(df["time"], dayfirst=dayfirst, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    _validate_ohlcv(df)
    return df


def _validate_ohlcv(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning.")
    if not df["time"].is_monotonic_increasing:
        raise ValueError("Time column must be monotonic increasing.")


def slice_ohlcv(
    df: pd.DataFrame,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    rows: Optional[int] = None,
    days: Optional[int] = None,
) -> pd.DataFrame:
    sliced = df.copy()
    if date_from:
        start_ts = pd.to_datetime(date_from)
        sliced = sliced[sliced["time"] >= start_ts]
    if date_to:
        end_ts = pd.to_datetime(date_to)
        sliced = sliced[sliced["time"] <= end_ts]
    if days:
        max_time = sliced["time"].max()
        if pd.notna(max_time):
            sliced = sliced[sliced["time"] >= max_time - pd.Timedelta(days=days)]
    if rows:
        sliced = sliced.tail(rows)
    sliced = sliced.reset_index(drop=True)
    _validate_ohlcv(sliced)
    return sliced


def build_analyzer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    technical_settings = get_setting(config, "technical_settings", {}) or {}
    sessions_config = get_setting(config, "sessions_config", {}) or {}
    analyzer_settings = dict(technical_settings)
    adx_weak = get_setting(config, "technical_settings.ADX_THRESHOLD_WEAK", analyzer_settings.get("ADX_THRESHOLD_WEAK"))
    analyzer_settings["ADX_THRESHOLD_WEAK"] = adx_weak
    analyzer_settings["REAL_TIME_ENABLED"] = True
    analyzer_settings["USE_CURRENT_PRICE_FOR_ANALYSIS"] = True
    return {
        "ANALYZER_SETTINGS": analyzer_settings,
        "sessions_config": sessions_config,
        "TRADING_SESSIONS": sessions_config.get("TRADING_SESSIONS", {}) if isinstance(sessions_config, dict) else {},
    }


def load_config(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config payload must be a JSON object.")
    return payload
