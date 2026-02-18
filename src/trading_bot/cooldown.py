"""Cooldown gatekeeping for NDS scalping bot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.trading_bot.time_utils import parse_timestamp, to_utc_time


def _normalize_utc(value: Any) -> Optional[datetime]:
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    # parse_timestamp returns UTC-aware for strings/epoch; keep explicit UTC normalization
    return to_utc_time(parsed, time_mode="UTC")


def _normalize_candle_times_to_utc(candle_times):
    try:
        import pandas as pd  # local import to keep module lightweight in non-pandas contexts

        normalized = pd.to_datetime(candle_times, errors="coerce", utc=True)
        return normalized
    except Exception:
        return candle_times


def _normalize_open_positions_times(open_positions: List[Dict[str, Any]], signal: str) -> List[datetime]:
    normalized: List[datetime] = []
    for pos in open_positions:
        if str(pos.get("side", "")).upper() != signal:
            continue
        ts = _normalize_utc(pos.get("open_time"))
        if ts is not None:
            normalized.append(ts)
    return normalized


def _iter_times(series_like):
    if hasattr(series_like, "tolist"):
        try:
            return list(series_like.tolist())
        except Exception:
            pass
    if hasattr(series_like, "_values"):
        try:
            return list(series_like._values)
        except Exception:
            pass
    try:
        return list(series_like)
    except Exception:
        return []


@dataclass
class CooldownDecision:
    allowed: bool
    reason: str
    details: Dict[str, Any]


def warn_deprecated_cooldown_settings(config: Any, logger) -> None:
    trading_rules = _get_trading_rules(config)
    deprecated_keys = [
        "MIN_CANDLES_BETWEEN",
        "MIN_TIME_BETWEEN_TRADES_MINUTES",
    ]
    for key in deprecated_keys:
        if key in trading_rules:
            logger.warning(
                "[COOLDOWN][DEPRECATED] trading_rules.%s is deprecated and ignored. "
                "Use trading_rules.MIN_CANDLES_BETWEEN_TRADES instead.",
                key,
            )


def get_min_candles_between_trades(config: Any, default: int = 0) -> int:
    trading_rules = _get_trading_rules(config)
    try:
        return int(trading_rules.get("MIN_CANDLES_BETWEEN_TRADES", default) or default)
    except (TypeError, ValueError):
        return int(default)


def summarize_positions(positions: Iterable[Dict[str, Any]]) -> Tuple[int, int, int, List[int]]:
    buy_count = 0
    sell_count = 0
    tickets: List[int] = []
    for pos in positions:
        if str(pos.get("side", "")).upper() == "BUY":
            buy_count += 1
        elif str(pos.get("side", "")).upper() == "SELL":
            sell_count += 1
        ticket = pos.get("position_ticket")
        if ticket is not None:
            tickets.append(int(ticket))
    return buy_count + sell_count, buy_count, sell_count, sorted(tickets)


def resolve_exposure_bias(positions: Iterable[Dict[str, Any]]) -> str:
    _, buy_count, sell_count, _ = summarize_positions(positions)
    if buy_count and sell_count:
        return "MIXED"
    if buy_count:
        return "BUY"
    if sell_count:
        return "SELL"
    return "NONE"


def evaluate_cooldown(
    *,
    signal: str,
    min_candles_between: int,
    df,
    open_positions: List[Dict[str, Any]],
    pending_orders: Optional[List[Dict[str, Any]]] = None,
    last_trade_candle_time: Optional[datetime],
    last_trade_direction: Optional[str],
) -> CooldownDecision:
    signal = str(signal or "NONE").upper()
    if signal not in ("BUY", "SELL"):
        return CooldownDecision(True, "NO_SIGNAL", {"signal": signal})

    exposure_bias = resolve_exposure_bias(open_positions)
    if exposure_bias == "MIXED":
        _, buy_count, sell_count, tickets = summarize_positions(open_positions)
        return CooldownDecision(
            False,
            "MIXED_EXPOSURE",
            {
                "signal": signal,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "tickets": tickets,
            },
        )

    pending_orders = pending_orders or []
    if open_positions or pending_orders:
        return CooldownDecision(
            False,
            "EXPOSURE_PRESENT",
            {
                "signal": signal,
                "open_positions": len(open_positions),
                "pending_orders": len(pending_orders),
                "exposure_bias": exposure_bias,
            },
        )

    last_trade_time = None
    if open_positions:
        same_side_open_times = _normalize_open_positions_times(open_positions, signal)
        if same_side_open_times:
            last_trade_time = max(same_side_open_times)

    if last_trade_time is None and last_trade_direction == signal:
        last_trade_time = _normalize_utc(last_trade_candle_time)
    else:
        last_trade_time = _normalize_utc(last_trade_time)

    if not last_trade_time or min_candles_between <= 0:
        return CooldownDecision(
            True,
            "NO_COOLDOWN",
            {
                "signal": signal,
                "last_trade_bar": last_trade_time,
                "min_candles": min_candles_between,
                "exposure_bias": exposure_bias,
            },
        )

    if df is None or getattr(df, "empty", False):
        return CooldownDecision(
            True,
            "NO_CANDLES",
            {
                "signal": signal,
                "last_trade_bar": last_trade_time,
                "min_candles": min_candles_between,
                "exposure_bias": exposure_bias,
            },
        )

    candle_times = _normalize_candle_times_to_utc(df["time"])
    if last_trade_time is None:
        return CooldownDecision(
            True,
            "NO_COOLDOWN",
            {
                "signal": signal,
                "last_trade_bar": None,
                "min_candles": min_candles_between,
                "exposure_bias": exposure_bias,
                "note": "last_trade_time_unparseable",
            },
        )

    try:
        candles_passed = int((candle_times > last_trade_time).sum())
        current_bar_time = candle_times.iloc[-1]
    except TypeError:
        normalized_times = [_normalize_utc(ts) for ts in _iter_times(candle_times)]
        normalized_times = [ts for ts in normalized_times if ts is not None]
        candles_passed = int(sum(1 for ts in normalized_times if ts > last_trade_time))
        current_bar_time = normalized_times[-1] if normalized_times else None
    if candles_passed <= min_candles_between:
        return CooldownDecision(
            False,
            "COOLDOWN_BLOCKED",
            {
                "signal": signal,
                "last_trade_bar": last_trade_time,
                "current_bar": current_bar_time,
                "diff": candles_passed,
                "min_candles": min_candles_between,
                "exposure_bias": exposure_bias,
            },
        )

    return CooldownDecision(
        True,
        "COOLDOWN_OK",
        {
            "signal": signal,
            "last_trade_bar": last_trade_time,
            "current_bar": current_bar_time,
            "diff": candles_passed,
            "min_candles": min_candles_between,
            "exposure_bias": exposure_bias,
        },
    )


def _get_trading_rules(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict):
        return config.get("trading_rules", {}) or {}
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter("trading_rules", {}) or {}
    return {}
