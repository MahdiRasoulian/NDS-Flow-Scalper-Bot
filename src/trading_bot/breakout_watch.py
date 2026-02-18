from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class BreakoutWatchState(str, Enum):
    PENDING = "PENDING"
    TRIGGERED = "TRIGGERED"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class BreakoutWatch:
    direction: str
    trigger_price: float
    expiration_candles: int
    structural_reference: Dict[str, Any]
    original_score: float
    finalized_payload: Dict[str, Any]
    signal_snapshot: Dict[str, Any]
    watch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: BreakoutWatchState = BreakoutWatchState.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    triggered_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    cancelled_reason: Optional[str] = None
    elapsed_candles: int = 0


@dataclass
class BreakoutValidationResult:
    passed: bool
    reasons: List[str]
    metrics: Dict[str, Any]


@dataclass
class BreakoutValidationConfig:
    min_body_ratio: float = 0.60
    max_counter_wick_to_body: float = 0.30
    min_buy_rsi: float = 50.0
    max_sell_rsi: float = 50.0
    min_atr_ratio: float = 0.90
    require_liquidity_sweep: bool = True


def _candle_quality(direction: str, trigger_price: float, candle: Dict[str, float], cfg: BreakoutValidationConfig) -> tuple[bool, Dict[str, float], List[str]]:
    reasons: List[str] = []
    op = float(candle.get("open", 0.0) or 0.0)
    cl = float(candle.get("close", 0.0) or 0.0)
    hi = float(candle.get("high", 0.0) or 0.0)
    lo = float(candle.get("low", 0.0) or 0.0)

    rng = max(hi - lo, 1e-9)
    body = abs(cl - op)
    body_ratio = body / rng
    upper_wick = max(hi - max(op, cl), 0.0)
    lower_wick = max(min(op, cl) - lo, 0.0)
    wick_to_body = (upper_wick if direction == "BUY" else lower_wick) / max(body, 1e-9)

    if direction == "BUY" and not (cl > trigger_price):
        reasons.append("breakout_close_not_above_level")
    if direction == "SELL" and not (cl < trigger_price):
        reasons.append("breakout_close_not_below_level")
    if body_ratio < cfg.min_body_ratio:
        reasons.append("weak_breakout_body")
    if wick_to_body > cfg.max_counter_wick_to_body:
        reasons.append("counter_wick_too_large")

    return len(reasons) == 0, {
        "body_ratio": body_ratio,
        "wick_to_body": wick_to_body,
        "body": body,
        "range": rng,
    }, reasons


def revalidate_breakout(
    watch: BreakoutWatch,
    current_market_state: Dict[str, Any],
    cfg: Optional[BreakoutValidationConfig] = None,
) -> BreakoutValidationResult:
    """Pure validation pipeline for context-aware breakout execution."""
    cfg = cfg or BreakoutValidationConfig()
    direction = str(watch.direction or "").upper()
    reasons: List[str] = []

    rsi = float(current_market_state.get("rsi", 50.0) or 50.0)
    ema_slope = float(current_market_state.get("ema_slope", 0.0) or 0.0)
    atr = float(current_market_state.get("atr", 0.0) or 0.0)
    atr_baseline = float(current_market_state.get("atr_baseline", atr) or atr or 1e-9)
    inside_sr = bool(current_market_state.get("inside_sr", False))
    liquidity_sweep = bool(current_market_state.get("liquidity_sweep", False))

    candle = current_market_state.get("breakout_candle") if isinstance(current_market_state.get("breakout_candle"), dict) else {}
    quality_ok, candle_metrics, candle_reasons = _candle_quality(direction, watch.trigger_price, candle, cfg)
    if not quality_ok:
        reasons.extend(candle_reasons)

    if direction == "BUY":
        if rsi < cfg.min_buy_rsi or ema_slope < 0:
            reasons.append("momentum_not_aligned")
    elif direction == "SELL":
        if rsi > cfg.max_sell_rsi or ema_slope > 0:
            reasons.append("momentum_not_aligned")
    else:
        reasons.append("invalid_watch_direction")

    if inside_sr:
        reasons.append("inside_sr_zone")

    atr_ratio = atr / max(atr_baseline, 1e-9)
    if atr_ratio < cfg.min_atr_ratio:
        reasons.append("atr_contracted")

    if cfg.require_liquidity_sweep and not liquidity_sweep:
        reasons.append("missing_liquidity_sweep")

    metrics = {
        "rsi": rsi,
        "ema_slope": ema_slope,
        "atr": atr,
        "atr_baseline": atr_baseline,
        "atr_ratio": atr_ratio,
        "inside_sr": inside_sr,
        "liquidity_sweep": liquidity_sweep,
        **candle_metrics,
    }
    return BreakoutValidationResult(passed=len(reasons) == 0, reasons=reasons, metrics=metrics)


class BreakoutWatchManager:
    def __init__(self, logger) -> None:
        self._logger = logger
        self._watches: Dict[str, BreakoutWatch] = {}

    def add(self, watch: BreakoutWatch) -> BreakoutWatch:
        self._watches[watch.watch_id] = watch
        self._logger.info("[BREAKOUT_WATCH][STATE] id=%s -> %s trigger=%.2f exp=%s", watch.watch_id, watch.state.value, watch.trigger_price, watch.expiration_candles)
        return watch

    def pending(self) -> List[BreakoutWatch]:
        return [w for w in self._watches.values() if w.state == BreakoutWatchState.PENDING]

    def all(self) -> List[BreakoutWatch]:
        return list(self._watches.values())

    def on_new_candle(self) -> None:
        for watch in self.pending():
            watch.elapsed_candles += 1
            if watch.elapsed_candles >= watch.expiration_candles:
                watch.state = BreakoutWatchState.EXPIRED
                watch.cancelled_reason = "expiration"
                self._logger.info("[BREAKOUT_WATCH][STATE] id=%s -> %s elapsed=%s", watch.watch_id, watch.state.value, watch.elapsed_candles)

    def mark_triggered(self, watch: BreakoutWatch) -> None:
        watch.state = BreakoutWatchState.TRIGGERED
        watch.triggered_at = datetime.now(timezone.utc)
        self._logger.info("[BREAKOUT_WATCH][STATE] id=%s -> %s", watch.watch_id, watch.state.value)

    def mark_executed(self, watch: BreakoutWatch) -> None:
        watch.state = BreakoutWatchState.EXECUTED
        watch.executed_at = datetime.now(timezone.utc)
        self._logger.info("[BREAKOUT_WATCH][STATE] id=%s -> %s", watch.watch_id, watch.state.value)

    def mark_cancelled(self, watch: BreakoutWatch, reason: str) -> None:
        watch.state = BreakoutWatchState.CANCELLED
        watch.cancelled_reason = reason
        self._logger.info("[BREAKOUT_WATCH][STATE] id=%s -> %s reason=%s", watch.watch_id, watch.state.value, reason)
