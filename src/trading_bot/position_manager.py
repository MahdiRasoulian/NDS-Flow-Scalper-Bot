"""Position management for TP1/TP2 scalping execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time

from src.trading_bot.contracts import PositionContract
from src.trading_bot.nds.distance_utils import pips_to_price, resolve_point_size_with_source


@dataclass
class PositionPlan:
    position_ticket: int
    symbol: str
    side: str
    entry_price: float
    tp1_price: float
    tp2_price: Optional[float]
    stop_loss: float
    volume: float
    partial_close_pct: float
    move_sl_to_be: bool
    trail_after_tp1: bool
    trail_atr_mult: float
    atr_value: Optional[float]
    counter_trend: bool
    be_trigger_pips: Optional[float]
    be_plus_pips: float
    tp_plan: str
    tp1_hit: bool = False
    partial_closed: bool = False
    sl_moved: bool = False
    trail_active: bool = False
    post_tp1_configured: bool = False
    last_trail_sl: Optional[float] = None
    notes: List[str] = field(default_factory=list)


class PositionManager:
    """Manages TP1/TP2 partial close, BE moves, and trailing logic."""

    def __init__(
        self,
        config: Dict[str, Any],
        mt5_client: Any,
        *,
        trade_tracker: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.mt5_client = mt5_client
        self.trade_tracker = trade_tracker
        self._logger = logger or logging.getLogger(__name__)
        self._plans: Dict[int, PositionPlan] = {}

    def reconcile_positions(self, open_positions: List[PositionContract]) -> None:
        active_tickets = {int(pos["position_ticket"]) for pos in open_positions}
        for ticket in list(self._plans.keys()):
            if ticket not in active_tickets:
                self._logger.info("[NDS][TP_MANAGER] position=%s removed from plan cache", ticket)
                self._plans.pop(ticket, None)

        for position in open_positions:
            ticket = int(position["position_ticket"])
            if ticket not in self._plans:
                plan = self._build_plan(position)
                if plan:
                    self._plans[ticket] = plan
                    metadata = self._get_trade_metadata(position)
                    self._logger.info(
                        "[PM][PLAN_META] ticket=%s tp_exec=%s tp1=%s tp2=%s tp_sent=%s partial_count=%s remaining_vol=%s",
                        ticket,
                        metadata.get("tp_execution_mode"),
                        metadata.get("tp1_price"),
                        metadata.get("tp2_price"),
                        metadata.get("tp_sent_to_broker"),
                        metadata.get("partial_close_count"),
                        metadata.get("remaining_volume_after_tp1"),
                    )
                    self._logger.info(
                        "[NDS][TP_PLAN] ticket=%s tp_plan=%s tp1=%.2f tp2=%s trail=%s be_trigger=%s",
                        ticket,
                        plan.tp_plan,
                        float(plan.tp1_price),
                        f"{float(plan.tp2_price):.2f}" if plan.tp2_price else "NONE",
                        plan.trail_after_tp1,
                        plan.be_trigger_pips,
                    )

    def manage_positions(self, open_positions: List[PositionContract]) -> None:
        if not open_positions:
            return
        self.reconcile_positions(open_positions)
        for position in open_positions:
            ticket = int(position["position_ticket"])
            plan = self._plans.get(ticket)
            if not plan:
                continue
            plan.volume = float(position.get("volume") or plan.volume)
            plan.stop_loss = float(position.get("sl") or plan.stop_loss)
            self._process_position(position, plan)

    def _get_trade_metadata(self, position: PositionContract) -> Dict[str, Any]:
        if self.trade_tracker is None:
            return {}
        ticket = int(position.get("position_ticket") or 0)
        record = self.trade_tracker.active_trades.get(ticket)
        if record and record.get("open_event", {}).get("metadata"):
            return record.get("open_event", {}).get("metadata", {}) or {}
        if hasattr(self.trade_tracker, "resolve_metadata_for_position"):
            return self.trade_tracker.resolve_metadata_for_position(position)
        return {}

    def _build_plan(self, position: PositionContract) -> Optional[PositionPlan]:
        metadata = self._get_trade_metadata(position)

        entry_context = {}
        analysis_snapshot = metadata.get("analysis_snapshot") or {}
        if isinstance(analysis_snapshot, dict):
            entry_context = (
                analysis_snapshot.get("entry_context")
                or (analysis_snapshot.get("context") or {}).get("entry_context")
                or {}
            )

        counter_trend = bool(entry_context.get("counter_trend"))
        market_metrics = metadata.get("market_metrics", {}) if isinstance(metadata, dict) else {}
        atr_value = market_metrics.get("atr_short") or market_metrics.get("atr")

        flow_settings = self.config.get("flow_settings", {}) if isinstance(self.config, dict) else {}
        risk_settings = self.config.get("risk_settings", {}) if isinstance(self.config, dict) else {}

        tp1_price = metadata.get("tp1_price")
        tp1_price = float(tp1_price) if tp1_price else float(position.get("tp") or 0.0)

        tp2_price = metadata.get("tp2_price")
        tp2_price = float(tp2_price) if tp2_price else None

        if not tp1_price:
            tp1_price, tp2_price, reason = self._synthesize_tp_targets(position, metadata)
            if tp1_price:
                self._logger.error(
                    "[PM][PLAN_META_MISSING] ticket=%s reason=%s broker_tp=%.2f entry=%.2f sl=%.2f",
                    position.get("position_ticket"),
                    reason,
                    float(position.get("tp") or 0.0),
                    float(position.get("entry_price") or 0.0),
                    float(position.get("sl") or 0.0),
                )
                self._persist_fallback_metadata(position, tp1_price, tp2_price, reason)
            else:
                self._logger.error(
                    "[PM][PLAN_META_ERROR] ticket=%s reason=%s broker_tp=%.2f entry=%.2f sl=%.2f",
                    position.get("position_ticket"),
                    reason,
                    float(position.get("tp") or 0.0),
                    float(position.get("entry_price") or 0.0),
                    float(position.get("sl") or 0.0),
                )
                return None

        trail_after_tp1 = bool(flow_settings.get("FLOW_TRAIL_AFTER_TP1", True))
        tp2_enabled = bool(risk_settings.get("TP2_ENABLED", True))
        if not tp2_enabled:
            tp2_price = None

        be_trigger_pips = (
            risk_settings.get("BE_TRIGGER_PIPS_COUNTERTREND")
            if counter_trend
            else risk_settings.get("BE_TRIGGER_PIPS_SCALP")
        )
        be_trigger_pips = float(be_trigger_pips) if be_trigger_pips is not None else None
        be_plus_pips = float(flow_settings.get("FLOW_TP1_MOVE_SL_TO_BE_PLUS_PIPS", 0.0) or 0.0)

        tp_execution_mode = metadata.get("tp_execution_mode")
        partial_close_pct = float(flow_settings.get("FLOW_TP1_PARTIAL_CLOSE_PCT", 0.5))
        move_sl_to_be = bool(flow_settings.get("FLOW_TP1_MOVE_SL_TO_BE", True))
        if tp_execution_mode == "SINGLE_TP":
            partial_close_pct = 0.0
            move_sl_to_be = False
            trail_after_tp1 = False
            tp2_price = None

        tp_plan = "single_tp"
        if trail_after_tp1:
            tp_plan = "trail_after_tp1"
        elif tp2_price is not None:
            tp_plan = "tp1_tp2"

        return PositionPlan(
            position_ticket=int(position["position_ticket"]),
            symbol=str(position.get("symbol") or ""),
            side=str(position.get("side") or "BUY").upper(),
            entry_price=float(position.get("entry_price") or 0.0),
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            stop_loss=float(position.get("sl") or 0.0),
            volume=float(position.get("volume") or 0.0),
            partial_close_pct=partial_close_pct,
            move_sl_to_be=move_sl_to_be,
            trail_after_tp1=trail_after_tp1,
            trail_atr_mult=float(flow_settings.get("FLOW_TRAIL_ATR_MULT", 2.0)),
            atr_value=float(atr_value) if atr_value else None,
            counter_trend=counter_trend,
            be_trigger_pips=be_trigger_pips,
            be_plus_pips=be_plus_pips,
            tp_plan=tp_plan,
        )

    def _process_position(self, position: PositionContract, plan: PositionPlan) -> None:
        current_price = float(position.get("current_price") or 0.0)
        if not current_price:
            return
        self._logger.info(
            "[PM][MANAGE] ticket=%s mode=%s price=%.2f tp1=%.2f hit_tp1=%s partial_done=%s",
            plan.position_ticket,
            plan.tp_plan,
            current_price,
            plan.tp1_price,
            plan.tp1_hit,
            plan.partial_closed,
        )

        if not plan.tp1_hit and self._price_reached_tp1(plan, current_price):
            plan.tp1_hit = True
            self._logger.info(
                "[NDS][TP1_HIT] ticket=%s price=%.2f tp1=%.2f",
                plan.position_ticket,
                current_price,
                plan.tp1_price,
            )

        if plan.tp1_hit:
            self._execute_tp1_partial_close(position, plan)
            self._configure_post_tp1(position, plan)

        self._maybe_move_sl_to_be(position, plan, current_price)
        self._maybe_update_trailing(position, plan, current_price)

    def _price_reached_tp1(self, plan: PositionPlan, price: float) -> bool:
        if plan.side == "BUY":
            return price >= plan.tp1_price
        return price <= plan.tp1_price

    def _execute_tp1_partial_close(self, position: PositionContract, plan: PositionPlan) -> None:
        if plan.partial_closed or plan.partial_close_pct <= 0:
            return
        close_volume = plan.volume * plan.partial_close_pct
        min_lot = self._get_gold_spec("MIN_LOT", 0.01)
        lot_step = self._get_gold_spec("LOT_STEP", 0.01)
        close_volume = self._round_volume(close_volume, lot_step)
        if close_volume < float(min_lot):
            self._logger.warning(
                "[NDS][TP1_PARTIAL_SKIP] ticket=%s reason=volume_below_min volume=%.3f min=%.3f",
                plan.position_ticket,
                close_volume,
                float(min_lot),
            )
            return

        result = self._close_with_retry(
            ticket=plan.position_ticket,
            volume=close_volume,
            comment="TP1 partial close",
            context="TP1_PARTIAL",
        )
        self._logger.info(
            "[PM][PARTIAL_CLOSE] ticket=%s requested_vol=%.3f result=%s retcode=%s",
            plan.position_ticket,
            close_volume,
            result,
            result.get("retcode") if isinstance(result, dict) else None,
        )
        if result and result.get("success"):
            plan.partial_closed = True
            if self.trade_tracker is not None:
                try:
                    remaining_volume = max(plan.volume - close_volume, 0.0)
                    self.trade_tracker.register_partial_close(
                        position_ticket=plan.position_ticket,
                        volume_closed=close_volume,
                        remaining_volume=remaining_volume,
                        reason="TP1",
                    )
                except Exception as exc:
                    self._logger.warning(
                        "[NDS][TP1_PARTIAL_TRACK_FAIL] ticket=%s error=%s",
                        plan.position_ticket,
                        exc,
                    )
            self._logger.info(
                "[NDS][TP1_PARTIAL_CLOSE] ticket=%s volume=%.3f price=%.2f",
                plan.position_ticket,
                close_volume,
                float(result.get("price") or 0.0),
            )
        else:
            self._logger.warning(
                "[NDS][TP1_PARTIAL_FAIL] ticket=%s result=%s",
                plan.position_ticket,
                result,
            )

    def _configure_post_tp1(self, position: PositionContract, plan: PositionPlan) -> None:
        if plan.post_tp1_configured:
            return

        if plan.trail_after_tp1:
            result = self._modify_with_retry(
                ticket=plan.position_ticket,
                new_sl=None,
                new_tp=0.0,
                context="TP_TRAIL_ARM",
            )
            self._logger.info(
                "[PM][SET_TP2_OR_TRAIL] ticket=%s trailing=true tp2=NONE result=%s retcode=%s",
                plan.position_ticket,
                result,
                result.get("retcode") if isinstance(result, dict) else None,
            )
            self._logger.info(
                "[NDS][TP_TRAIL_ARM] ticket=%s result=%s",
                plan.position_ticket,
                result,
            )
            if result and result.get("success"):
                plan.trail_active = True
                plan.post_tp1_configured = True
            else:
                self._logger.warning(
                    "[NDS][TP_TRAIL_FAIL] ticket=%s result=%s",
                    plan.position_ticket,
                    result,
                )
        elif plan.tp2_price is not None:
            result = self._modify_with_retry(
                ticket=plan.position_ticket,
                new_tp=plan.tp2_price,
                new_sl=None,
                context="TP2_SET",
            )
            self._logger.info(
                "[PM][SET_TP2_OR_TRAIL] ticket=%s trailing=false tp2=%.2f result=%s retcode=%s",
                plan.position_ticket,
                float(plan.tp2_price),
                result,
                result.get("retcode") if isinstance(result, dict) else None,
            )
            self._logger.info(
                "[NDS][TP2_SET] ticket=%s tp2=%.2f result=%s",
                plan.position_ticket,
                float(plan.tp2_price),
                result,
            )
            if result and result.get("success"):
                plan.post_tp1_configured = True
            else:
                self._logger.warning(
                    "[NDS][TP2_SET_FAIL] ticket=%s tp2=%.2f result=%s",
                    plan.position_ticket,
                    float(plan.tp2_price),
                    result,
                )
        else:
            self._logger.info(
                "[NDS][TP2_SKIP] ticket=%s reason=no_tp2_config",
                plan.position_ticket,
            )
            plan.post_tp1_configured = True

    def _maybe_move_sl_to_be(
        self, position: PositionContract, plan: PositionPlan, current_price: float
    ) -> None:
        if plan.sl_moved or not plan.move_sl_to_be:
            return

        if plan.tp1_hit:
            should_move = True
        elif plan.be_trigger_pips is not None:
            point_size = self._resolve_point_size()
            trigger_distance = pips_to_price(plan.be_trigger_pips, point_size)
            if plan.side == "BUY":
                should_move = current_price >= plan.entry_price + trigger_distance
            else:
                should_move = current_price <= plan.entry_price - trigger_distance
        else:
            should_move = False

        if not should_move:
            return

        new_sl = plan.entry_price
        if plan.be_plus_pips > 0:
            point_size = self._resolve_point_size()
            offset = pips_to_price(plan.be_plus_pips, point_size)
            if plan.side == "BUY":
                new_sl = plan.entry_price + offset
            else:
                new_sl = plan.entry_price - offset
        result = self._modify_with_retry(
            ticket=plan.position_ticket,
            new_sl=new_sl,
            new_tp=None,
            context="MOVE_SL",
        )
        self._logger.info(
            "[PM][MOVE_SL] ticket=%s new_sl=%.2f result=%s retcode=%s",
            plan.position_ticket,
            new_sl,
            result,
            result.get("retcode") if isinstance(result, dict) else None,
        )
        if result and result.get("success"):
            plan.sl_moved = True
            plan.stop_loss = new_sl
            self._logger.info(
                "[NDS][SL_BE] ticket=%s sl=%.2f",
                plan.position_ticket,
                new_sl,
            )
        else:
            self._logger.warning(
                "[NDS][SL_BE_FAIL] ticket=%s result=%s",
                plan.position_ticket,
                result,
            )

    def _maybe_update_trailing(
        self, position: PositionContract, plan: PositionPlan, current_price: float
    ) -> None:
        if not plan.trail_active or not plan.tp1_hit or not plan.atr_value:
            return

        trail_distance = float(plan.atr_value) * float(plan.trail_atr_mult)
        if trail_distance <= 0:
            return

        if plan.side == "BUY":
            new_sl = current_price - trail_distance
            if plan.sl_moved:
                new_sl = max(new_sl, plan.entry_price)
            if plan.stop_loss and new_sl <= plan.stop_loss:
                return
        else:
            new_sl = current_price + trail_distance
            if plan.sl_moved:
                new_sl = min(new_sl, plan.entry_price)
            if plan.stop_loss and new_sl >= plan.stop_loss:
                return

        result = self._modify_with_retry(
            ticket=plan.position_ticket,
            new_sl=new_sl,
            new_tp=None,
            context="TRAIL",
        )
        if result and result.get("success"):
            plan.stop_loss = new_sl
            plan.last_trail_sl = new_sl
            self._logger.info(
                "[NDS][TRAIL] ticket=%s sl=%.2f trail_dist=%.4f",
                plan.position_ticket,
                new_sl,
                trail_distance,
            )
        else:
            self._logger.warning(
                "[NDS][TRAIL_FAIL] ticket=%s result=%s",
                plan.position_ticket,
                result,
            )

    def _resolve_point_size(self) -> float:
        point_size, _ = resolve_point_size_with_source(
            self.config,
            default=None,
        )
        if not point_size or point_size <= 0:
            point_size = 0.01
        return point_size

    def _get_gold_spec(self, key: str, default: float) -> float:
        trading_settings = self.config.get("trading_settings", {}) if isinstance(self.config, dict) else {}
        gold_specs = trading_settings.get("GOLD_SPECIFICATIONS", {}) if isinstance(trading_settings, dict) else {}
        return float(
            gold_specs.get(key)
            or gold_specs.get(str(key).lower())
            or default
        )

    @staticmethod
    def _round_volume(volume: float, step: float) -> float:
        if step <= 0:
            return volume
        return round(volume / step) * step

    def _synthesize_tp_targets(
        self, position: PositionContract, metadata: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float], str]:
        entry_price = float(position.get("entry_price") or 0.0)
        side = str(position.get("side") or "BUY").upper()
        if not entry_price:
            return None, None, "entry_price_missing"

        risk_settings = self.config.get("risk_settings", {}) if isinstance(self.config, dict) else {}
        analysis_snapshot = metadata.get("analysis_snapshot") or {}
        entry_context = (
            analysis_snapshot.get("entry_context")
            or (analysis_snapshot.get("context") or {}).get("entry_context")
            or {}
        )
        counter_trend = bool(entry_context.get("counter_trend"))
        tp1_pips = float(
            risk_settings.get("COUNTERTREND_TP1_PIPS")
            if counter_trend and risk_settings.get("COUNTERTREND_TP1_PIPS") is not None
            else risk_settings.get("TP1_PIPS", 35.0)
        )
        if tp1_pips <= 0:
            return None, None, "tp1_pips_missing"

        point_size = self._resolve_point_size()
        tp1_distance = pips_to_price(tp1_pips, point_size)
        tp1_price = entry_price + tp1_distance if side == "BUY" else entry_price - tp1_distance

        tp2_price = None
        if bool(risk_settings.get("TP2_ENABLED", True)):
            tp2_pips = float(risk_settings.get("TP2_PIPS", tp1_pips * 2.0))
            if tp2_pips > 0:
                tp2_distance = pips_to_price(tp2_pips, point_size)
                tp2_price = entry_price + tp2_distance if side == "BUY" else entry_price - tp2_distance

        return tp1_price, tp2_price, "synthesized_from_config"

    def _persist_fallback_metadata(
        self,
        position: PositionContract,
        tp1_price: float,
        tp2_price: Optional[float],
        reason: str,
    ) -> None:
        if self.trade_tracker is None:
            return
        ticket = int(position.get("position_ticket") or 0)
        record = self.trade_tracker.active_trades.get(ticket)
        if not record:
            return
        open_event = record.get("open_event") or {}
        metadata = open_event.get("metadata", {}) or {}
        metadata.setdefault("tp1_price", tp1_price)
        if tp2_price is not None:
            metadata.setdefault("tp2_price", tp2_price)
        metadata.setdefault("tp_execution_mode", metadata.get("tp_execution_mode") or "TP1_PARTIAL_MANAGED")
        metadata.setdefault("tp_plan", metadata.get("tp_plan") or "fallback")
        metadata.setdefault("tp_fallback_reason", reason)
        open_event["metadata"] = metadata
        record["open_event"] = open_event

    def _modify_with_retry(
        self,
        *,
        ticket: int,
        new_sl: Optional[float],
        new_tp: Optional[float],
        context: str,
    ) -> Dict[str, Any]:
        retries = 2
        backoff = 0.2
        last_result: Dict[str, Any] = {}
        for attempt in range(retries + 1):
            result = self.mt5_client.modify_position(
                ticket=ticket,
                new_sl=new_sl,
                new_tp=new_tp,
            )
            last_result = result or {}
            if result and result.get("success"):
                return result
            if not self._is_retryable(result) or attempt == retries:
                break
            self._logger.warning(
                "[PM][MODIFY_RETRY] ticket=%s context=%s attempt=%s retcode=%s",
                ticket,
                context,
                attempt + 1,
                last_result.get("retcode"),
            )
            time.sleep(backoff)
        return last_result

    def _close_with_retry(
        self,
        *,
        ticket: int,
        volume: float,
        comment: str,
        context: str,
    ) -> Dict[str, Any]:
        retries = 1
        backoff = 0.2
        last_result: Dict[str, Any] = {}
        for attempt in range(retries + 1):
            result = self.mt5_client.close_position(
                ticket=ticket,
                volume=volume,
                comment=comment,
            )
            last_result = result or {}
            if result and result.get("success"):
                return result
            if not self._is_retryable(result) or attempt == retries:
                break
            self._logger.warning(
                "[PM][CLOSE_RETRY] ticket=%s context=%s attempt=%s retcode=%s",
                ticket,
                context,
                attempt + 1,
                last_result.get("retcode"),
            )
            time.sleep(backoff)
        return last_result

    @staticmethod
    def _is_retryable(result: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(result, dict):
            return False
        retcode = result.get("retcode")
        if isinstance(retcode, str):
            retryable_tokens = ("requote", "off quotes", "price changed", "timeout", "trade context busy")
            return any(token in retcode.lower() for token in retryable_tokens)
        if isinstance(retcode, int):
            return retcode in {10004, 10006, 10016, 10021, 10025}
        comment = result.get("comment")
        if isinstance(comment, str):
            retryable_tokens = ("requote", "off quotes", "price changed", "timeout", "trade context busy")
            return any(token in comment.lower() for token in retryable_tokens)
        return False
