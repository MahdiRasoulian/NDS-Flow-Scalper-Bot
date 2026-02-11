from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def resolve_sr_gate_settings(settings: Mapping[str, Any] | None) -> Dict[str, float]:
    """Shared SR gate settings to keep Analyzer and RiskManager in lockstep."""
    cfg = _as_mapping(settings)
    return {
        "proximity_block_atr": _to_float(cfg.get("STATIC_SR_PROXIMITY_BLOCK_ATR"), 0.25),
        "strong_touches_min": float(_to_int(cfg.get("STATIC_SR_STRONG_TOUCHES_MIN"), 3)),
        "break_disp_min_atr": _to_float(cfg.get("STATIC_SR_BREAK_DISPLACEMENT_ATR_MIN"), 0.5),
        "break_close_buffer_atr": _to_float(cfg.get("STATIC_SR_BREAK_CLOSE_BUFFER_ATR"), 0.02),
        "break_close_min_bars": float(_to_int(cfg.get("STATIC_SR_BREAK_CLOSE_MIN_BARS"), 1)),
        "pullback_hold_tolerance_atr": _to_float(cfg.get("STATIC_SR_PULLBACK_HOLD_TOLERANCE_ATR"), 0.05),
        "rejection_min_sweeps": float(_to_int(cfg.get("STATIC_SR_REJECTION_MIN_SWEEPS"), 1)),
    }


def _nearest_level_for_signal(signal: str, sr_context: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    sr_map = _as_mapping(sr_context)
    if signal == "BUY":
        return _as_mapping(sr_map.get("nearest_resistance")) or None
    if signal == "SELL":
        return _as_mapping(sr_map.get("nearest_support")) or None
    return None


def is_near_strong_level(
    *,
    signal: str,
    sr_context: Mapping[str, Any] | None,
    settings: Mapping[str, float],
) -> Tuple[bool, str, Dict[str, Any]]:
    level = _nearest_level_for_signal(signal, sr_context) or {}
    if not level:
        return False, "missing_level", {}

    dist_atr = _to_float(level.get("dist_atr"), 999.0)
    touches = _to_int(level.get("touches"), 0)
    near = dist_atr <= _to_float(settings.get("proximity_block_atr"), 0.25) and touches >= _to_int(
        settings.get("strong_touches_min"), 3
    )
    return near, f"dist_atr={dist_atr:.2f} touches={touches}", level


def compute_confirmation_flags(
    *,
    signal: str,
    signal_context: Mapping[str, Any] | None,
    entry_context: Mapping[str, Any] | None,
    level: Mapping[str, Any] | None,
    settings: Mapping[str, float],
) -> Dict[str, bool]:
    sig = _as_mapping(signal_context)
    ent = _as_mapping(entry_context)
    lvl = _as_mapping(level)

    disp_atr = _to_float(ent.get("disp_atr"), 0.0)
    close_count = _to_int(ent.get("sr_break_close_count"), 0)
    sweeps_count = _to_int(sig.get("sweeps_count"), 0)
    last_close = _to_float(ent.get("last_close"), float("nan"))
    last_open = _to_float(ent.get("last_open"), float("nan"))
    last_low = _to_float(ent.get("last_low"), float("nan"))
    last_high = _to_float(ent.get("last_high"), float("nan"))

    band_top = _to_float(lvl.get("band_top"), float("nan"))
    band_bottom = _to_float(lvl.get("band_bottom"), float("nan"))
    band_half = _to_float(ent.get("sr_band_half"), 0.0)
    close_buffer = max(0.0, band_half * _to_float(settings.get("break_close_buffer_atr"), 0.02))
    hold_tol = max(0.0, band_half * _to_float(settings.get("pullback_hold_tolerance_atr"), 0.05))

    has_band_top = band_top == band_top
    has_band_bottom = band_bottom == band_bottom
    has_last_open = last_open == last_open
    has_last_close = last_close == last_close
    has_last_low = last_low == last_low
    has_last_high = last_high == last_high

    break_close_confirmed = False
    pullback_hold_confirmed = False
    rejection_confirmed = False

    if signal == "BUY" and has_band_top:
        break_close_confirmed = (
            close_count >= _to_int(settings.get("break_close_min_bars"), 1)
            and disp_atr >= _to_float(settings.get("break_disp_min_atr"), 0.5)
        )
        pullback_hold_confirmed = break_close_confirmed and has_last_low and last_low >= band_top - hold_tol
        rejection_confirmed = (
            sweeps_count >= _to_int(settings.get("rejection_min_sweeps"), 1)
            and disp_atr >= _to_float(settings.get("break_disp_min_atr"), 0.5)
            and has_last_high
            and has_last_close
            and last_high >= band_top
            and last_close >= band_top + close_buffer
            and (not has_last_open or last_open <= band_top + close_buffer)
        )
    elif signal == "SELL" and has_band_bottom:
        break_close_confirmed = (
            close_count >= _to_int(settings.get("break_close_min_bars"), 1)
            and disp_atr >= _to_float(settings.get("break_disp_min_atr"), 0.5)
        )
        pullback_hold_confirmed = break_close_confirmed and has_last_high and last_high <= band_bottom + hold_tol
        rejection_confirmed = (
            sweeps_count >= _to_int(settings.get("rejection_min_sweeps"), 1)
            and disp_atr >= _to_float(settings.get("break_disp_min_atr"), 0.5)
            and has_last_low
            and has_last_close
            and last_low <= band_bottom
            and last_close <= band_bottom - close_buffer
            and (not has_last_open or last_open >= band_bottom - close_buffer)
        )

    explicit_break = bool(sig.get("break_close_confirmed") or ent.get("break_close_confirmed"))
    explicit_pullback = bool(sig.get("pullback_hold_confirmed") or ent.get("pullback_hold_confirmed"))
    explicit_rejection = bool(sig.get("rejection_confirmed") or ent.get("rejection_confirmed"))

    return {
        "break_close_confirmed": bool(break_close_confirmed or explicit_break),
        "pullback_hold_confirmed": bool(pullback_hold_confirmed or explicit_pullback),
        "rejection_confirmed": bool(rejection_confirmed or explicit_rejection),
    }


def allow_sr_override(
    *,
    signal: str,
    signal_context: Mapping[str, Any] | None,
    entry_context: Mapping[str, Any] | None,
    sr_context: Mapping[str, Any] | None,
    settings: Mapping[str, float],
) -> Tuple[bool, str, Dict[str, Any]]:
    near, near_reason, level = is_near_strong_level(signal=signal, sr_context=sr_context, settings=settings)
    if not near:
        return True, f"not_near_strong_level:{near_reason}", {}

    flags = compute_confirmation_flags(
        signal=signal,
        signal_context=signal_context,
        entry_context=entry_context,
        level=level,
        settings=settings,
    )
    if flags["break_close_confirmed"] or flags["pullback_hold_confirmed"] or flags["rejection_confirmed"]:
        return True, f"confirmation_ok:{near_reason}", flags

    return False, f"missing_confirmation:{near_reason}", flags
