"""Cooldown gatekeeping for NDS scalping bot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


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

    last_trade_time = None
    if open_positions:
        same_side_positions = [
            pos for pos in open_positions if str(pos.get("side", "")).upper() == signal
        ]
        if same_side_positions:
            last_trade_time = max(
                (pos.get("open_time") for pos in same_side_positions if pos.get("open_time")),
                default=None,
            )

    if last_trade_time is None and last_trade_direction == signal:
        last_trade_time = last_trade_candle_time

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

    candle_times = df["time"]
    candles_passed = int((candle_times > last_trade_time).sum())
    current_bar_time = candle_times.iloc[-1]
    if candles_passed < min_candles_between:
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
