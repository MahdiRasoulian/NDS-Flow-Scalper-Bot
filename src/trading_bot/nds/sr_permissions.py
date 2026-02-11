from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def resolve_sr_gate_settings(settings: Dict[str, Any]) -> Dict[str, float]:
    """Shared SR gate settings to keep Analyzer and RiskManager in lockstep."""
    return {
        "proximity_block_atr": float(settings.get("STATIC_SR_PROXIMITY_BLOCK_ATR", 0.25)),
        "strong_touches_min": float(settings.get("STATIC_SR_STRONG_TOUCHES_MIN", 3)),
        "break_disp_min_atr": float(settings.get("STATIC_SR_BREAK_DISPLACEMENT_ATR_MIN", 0.5)),
        "break_close_buffer_atr": float(settings.get("STATIC_SR_BREAK_CLOSE_BUFFER_ATR", 0.02)),
        "break_close_min_bars": float(settings.get("STATIC_SR_BREAK_CLOSE_MIN_BARS", 1)),
        "pullback_hold_tolerance_atr": float(settings.get("STATIC_SR_PULLBACK_HOLD_TOLERANCE_ATR", 0.05)),
        "rejection_min_sweeps": float(settings.get("STATIC_SR_REJECTION_MIN_SWEEPS", 1)),
    }


def _nearest_level_for_signal(signal: str, sr_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if signal == "BUY":
        return sr_context.get("nearest_resistance") if isinstance(sr_context, dict) else None
    if signal == "SELL":
        return sr_context.get("nearest_support") if isinstance(sr_context, dict) else None
    return None


def is_near_strong_level(
    *,
    signal: str,
    sr_context: Dict[str, Any],
    settings: Dict[str, float],
) -> Tuple[bool, str, Dict[str, Any]]:
    level = _nearest_level_for_signal(signal, sr_context) or {}
    if not level:
        return False, "missing_level", {}

    dist_atr = float(level.get("dist_atr") or 999.0)
    touches = int(level.get("touches") or 0)
    near = dist_atr <= float(settings["proximity_block_atr"]) and touches >= int(settings["strong_touches_min"])
    return near, f"dist_atr={dist_atr:.2f} touches={touches}", level


def compute_confirmation_flags(
    *,
    signal: str,
    signal_context: Dict[str, Any],
    entry_context: Dict[str, Any],
    level: Dict[str, Any],
    settings: Dict[str, float],
) -> Dict[str, bool]:
    disp_atr = float(entry_context.get("disp_atr") or 0.0)
    close_count = int(entry_context.get("sr_break_close_count") or 0)
    sweeps_count = int(signal_context.get("sweeps_count") or 0)
    last_close = entry_context.get("last_close")
    last_open = entry_context.get("last_open")
    last_low = entry_context.get("last_low")
    last_high = entry_context.get("last_high")

    band_top = level.get("band_top")
    band_bottom = level.get("band_bottom")
    band_half = float(entry_context.get("sr_band_half") or 0.0)
    close_buffer = max(0.0, band_half * float(settings["break_close_buffer_atr"]))
    hold_tol = max(0.0, band_half * float(settings["pullback_hold_tolerance_atr"]))

    break_close_confirmed = False
    pullback_hold_confirmed = False
    rejection_confirmed = False

    if signal == "BUY" and band_top is not None:
        break_close_confirmed = (
            close_count >= int(settings["break_close_min_bars"]) and disp_atr >= float(settings["break_disp_min_atr"])
        )
        pullback_hold_confirmed = (
            break_close_confirmed
            and last_low is not None
            and float(last_low) >= float(band_top) - hold_tol
        )
        rejection_confirmed = (
            sweeps_count >= int(settings["rejection_min_sweeps"])
            and disp_atr >= float(settings["break_disp_min_atr"])
            and last_high is not None
            and last_close is not None
            and float(last_high) >= float(band_top)
            and float(last_close) >= float(band_top) + close_buffer
            and (last_open is None or float(last_open) <= float(band_top) + close_buffer)
        )
    elif signal == "SELL" and band_bottom is not None:
        break_close_confirmed = (
            close_count >= int(settings["break_close_min_bars"]) and disp_atr >= float(settings["break_disp_min_atr"])
        )
        pullback_hold_confirmed = (
            break_close_confirmed
            and last_high is not None
            and float(last_high) <= float(band_bottom) + hold_tol
        )
        rejection_confirmed = (
            sweeps_count >= int(settings["rejection_min_sweeps"])
            and disp_atr >= float(settings["break_disp_min_atr"])
            and last_low is not None
            and last_close is not None
            and float(last_low) <= float(band_bottom)
            and float(last_close) <= float(band_bottom) - close_buffer
            and (last_open is None or float(last_open) >= float(band_bottom) - close_buffer)
        )

    explicit_break = bool(signal_context.get("break_close_confirmed") or entry_context.get("break_close_confirmed"))
    explicit_pullback = bool(signal_context.get("pullback_hold_confirmed") or entry_context.get("pullback_hold_confirmed"))
    explicit_rejection = bool(signal_context.get("rejection_confirmed") or entry_context.get("rejection_confirmed"))

    return {
        "break_close_confirmed": bool(break_close_confirmed or explicit_break),
        "pullback_hold_confirmed": bool(pullback_hold_confirmed or explicit_pullback),
        "rejection_confirmed": bool(rejection_confirmed or explicit_rejection),
    }


def allow_sr_override(
    *,
    signal: str,
    signal_context: Dict[str, Any],
    entry_context: Dict[str, Any],
    sr_context: Dict[str, Any],
    settings: Dict[str, float],
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
