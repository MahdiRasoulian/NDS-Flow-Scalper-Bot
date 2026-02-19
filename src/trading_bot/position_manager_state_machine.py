"""Finite state machine based position lifecycle manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
from typing import Any, Dict, List, Optional

from src.trading_bot.contracts import PositionContract
from src.trading_bot.nds.distance_utils import pips_to_price, resolve_point_size_with_source


class PositionStatus(Enum):
    STATUS_OPEN = auto()
    STATUS_WAIT_TP1 = auto()
    STATUS_WAIT_TP2 = auto()
    STATUS_CLOSED = auto()
    STATUS_FAILED = auto()


@dataclass
class PositionPlan:
    """State and execution plan for one MT5 position."""

    ticket: int
    symbol: str
    direction: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    volume: float
    status: PositionStatus = PositionStatus.STATUS_OPEN
    partial_closed: bool = False
    sl_moved_to_be: bool = False
    close_summary: Dict[str, Any] = field(default_factory=dict)


class PositionManager:
    """Strict finite-state machine for TP1/TP2 lifecycle management."""

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

    def manage_positions(self, open_positions: List[PositionContract]) -> None:
        open_by_ticket = {
            int(pos.get("position_ticket")): pos
            for pos in open_positions
            if pos.get("position_ticket") is not None
        }

        for ticket in list(self._plans.keys()):
            if ticket not in open_by_ticket:
                self._handle_broker_closed(ticket)

        for ticket, position in open_by_ticket.items():
            plan = self._plans.get(ticket)
            if plan is None:
                plan = self._build_plan(position)
                if plan is None:
                    continue
                self._plans[ticket] = plan
            plan.volume = float(position.get("volume") or plan.volume)
            plan.sl_price = float(position.get("sl") or plan.sl_price)
            market_price = float(position.get("current_price") or 0.0)
            if market_price <= 0:
                continue
            self._logger.info("[PM][MANAGE] ticket=%s status=%s price=%.2f", plan.ticket, plan.status.name, market_price)
            self._run_fsm_tick(plan, market_price)

    def _build_plan(self, position: PositionContract) -> Optional[PositionPlan]:
        metadata = self._get_trade_metadata(position)
        risk_plan = metadata.get("risk_plan") if isinstance(metadata.get("risk_plan"), dict) else {}
        entry_risk = ((metadata.get("entry_snapshot") or {}).get("risk") or {}) if isinstance(metadata, dict) else {}

        tp1 = (
            risk_plan.get("tp1_price")
            or metadata.get("tp1_price")
            or entry_risk.get("tp1")
            or position.get("tp")
        )
        tp2 = (
            risk_plan.get("tp2_price")
            or metadata.get("tp2_price")
            or entry_risk.get("tp2")
        )
        if tp1 is None or tp2 is None:
            risk_settings = self.config.get("risk_settings", {}) if isinstance(self.config, dict) else {}
            point_size = self._resolve_point_size()
            entry = float(position.get("entry_price") or 0.0)
            side = str(position.get("side") or "BUY").upper()
            tp1_pips = float(risk_settings.get("TP1_PIPS", 35.0) or 35.0)
            tp2_pips = float(risk_settings.get("TP2_PIPS", tp1_pips * 2.0) or (tp1_pips * 2.0))
            if entry > 0 and point_size > 0:
                if tp1 is None:
                    tp1 = entry + pips_to_price(tp1_pips, point_size) if side == "BUY" else entry - pips_to_price(tp1_pips, point_size)
                if tp2 is None:
                    tp2 = entry + pips_to_price(tp2_pips, point_size) if side == "BUY" else entry - pips_to_price(tp2_pips, point_size)
                self._logger.info("[PM][PLAN_FALLBACK] ticket=%s tp1=%.2f tp2=%.2f", position.get("position_ticket"), float(tp1), float(tp2))
            else:
                self._logger.warning("[PM][PLAN_SKIP] ticket=%s reason=missing_tp_targets", position.get("position_ticket"))
                return None

        self._logger.info("[PM][PLAN_META] ticket=%s metadata_keys=%s", position.get("position_ticket"), sorted(list(metadata.keys())) if isinstance(metadata, dict) else [])

        plan = PositionPlan(
            ticket=int(position["position_ticket"]),
            symbol=str(position.get("symbol") or ""),
            direction=str(position.get("side") or "BUY").upper(),
            entry_price=float(position.get("entry_price") or 0.0),
            sl_price=float(position.get("sl") or 0.0),
            tp1_price=float(tp1),
            tp2_price=float(tp2),
            volume=float(position.get("volume") or 0.0),
        )
        self._logger.info(
            "[PM][PLAN] ticket=%s entry=%.2f sl=%.2f tp1=%.2f tp2=%.2f",
            plan.ticket,
            plan.entry_price,
            plan.sl_price,
            plan.tp1_price,
            plan.tp2_price,
        )
        return plan

    def _run_fsm_tick(self, plan: PositionPlan, market_price: float) -> None:
        try:
            if plan.status == PositionStatus.STATUS_OPEN:
                self._clear_broker_tp(plan)
                self._logger.info("[PM][STATE] %s OPEN -> WAIT_TP1", plan.ticket)
                plan.status = PositionStatus.STATUS_WAIT_TP1

            if plan.status == PositionStatus.STATUS_WAIT_TP1:
                tp1_hit = self._crossed_tp1(plan, market_price)
                self._logger.info(
                    "[PM][tp1_evaluation] ticket=%s price=%.2f tp1=%.2f hit=%s",
                    plan.ticket,
                    float(market_price),
                    float(plan.tp1_price),
                    tp1_hit,
                )
                if not tp1_hit:
                    return
                if self._secure_at_tp1(plan):
                    plan.status = PositionStatus.STATUS_WAIT_TP2
                    self._logger.info("[PM][STATE] %s WAIT_TP1 -> WAIT_TP2", plan.ticket)
                return

            if plan.status == PositionStatus.STATUS_WAIT_TP2:
                if not self._crossed_tp2(plan, market_price):
                    return
                self._logger.info("[PM][EVENT] TP2 reached ticket=%s price=%.2f", plan.ticket, float(market_price))
                return

            if plan.status in {PositionStatus.STATUS_CLOSED, PositionStatus.STATUS_FAILED}:
                return

            self._logger.error("[PM][ERROR] ticket=%s unexpected_state=%s", plan.ticket, plan.status)
            plan.status = PositionStatus.STATUS_FAILED
        except Exception:
            self._logger.exception("[PM][STATE_ERROR] ticket=%s", plan.ticket)
            plan.status = PositionStatus.STATUS_FAILED

    def _secure_at_tp1(self, plan: PositionPlan) -> bool:
        self._logger.info("[PM][TP1_DETECTED] ticket=%s", plan.ticket)
        if not plan.partial_closed:
            partial_result = self._partial_close(plan)
            if not partial_result:
                return False
            plan.partial_closed = True
            self._record_partial_close(plan, partial_result)

        if not plan.sl_moved_to_be:
            new_sl = self._compute_sl_to_be(plan)
            if not self._modify_sl(plan, new_sl):
                return False
            plan.sl_moved_to_be = True
            plan.sl_price = new_sl

        if not self._set_tp2(plan):
            return False

        return True

    def _record_partial_close(self, plan: PositionPlan, partial_result: Dict[str, float]) -> None:
        if self.trade_tracker is None:
            return
        register = getattr(self.trade_tracker, "register_partial_close", None)
        if not callable(register):
            return
        close_volume = float(partial_result.get("close_volume") or 0.0)
        remaining_volume = float(partial_result.get("remaining_volume") or 0.0)
        try:
            register(
                position_ticket=plan.ticket,
                volume_closed=float(close_volume),
                remaining_volume=float(remaining_volume),
                reason="TP1_PARTIAL",
            )
        except Exception:
            self._logger.exception("[PM][PARTIAL_META_FAIL] ticket=%s", plan.ticket)

    def _clear_broker_tp(self, plan: PositionPlan) -> None:
        self._modify_position(plan.ticket, new_tp=0.0, context="CLEAR_BROKER_TP")

    @staticmethod
    def _floor_to_step(value: float, step: float) -> float:
        if step <= 0:
            return float(value)
        steps = int((float(value) / float(step)) + 1e-9)
        return round(steps * float(step), 8)

    def _resolve_volume_constraints(self, symbol: Optional[str]) -> Dict[str, float]:
        trading_settings = self.config.get("trading_settings", {}) if isinstance(self.config, dict) else {}
        gold_specs = trading_settings.get("GOLD_SPECIFICATIONS", {}) if isinstance(trading_settings, dict) else {}
        strategy_min = float(gold_specs.get("MIN_LOT", 0.01) or 0.01)
        lot_step = float(gold_specs.get("LOT_STEP", 0.01) or 0.01)
        broker_min = strategy_min
        if hasattr(self.mt5_client, "_get_symbol_info"):
            try:
                symbol_info = self.mt5_client._get_symbol_info(symbol or "")
                if symbol_info is not None and hasattr(symbol_info, "volume_min"):
                    raw_broker_min = getattr(symbol_info, "volume_min")
                    if raw_broker_min is not None:
                        broker_min = float(raw_broker_min)
            except Exception:
                self._logger.debug("[PM][VOLUME] broker volume_min unavailable", exc_info=True)
        min_volume = max(strategy_min, broker_min)
        return {
            "strategy_min": strategy_min,
            "broker_min": broker_min,
            "min_volume": min_volume,
            "lot_step": lot_step,
        }

    def _resolve_partial_close_pct(self) -> float:
        flow_settings = self.config.get("flow_settings", {}) if isinstance(self.config, dict) else {}
        pct = float(flow_settings.get("FLOW_TP1_PARTIAL_CLOSE_PCT", 0.5) or 0.5)
        return min(max(pct, 0.0), 1.0)

    def _partial_close(self, plan: PositionPlan) -> Optional[Dict[str, float]]:
        constraints = self._resolve_volume_constraints(plan.symbol)
        lot_step = constraints["lot_step"]
        min_volume = constraints["min_volume"]

        requested_volume = max(plan.volume * self._resolve_partial_close_pct(), 0.0)
        close_volume = self._floor_to_step(requested_volume, lot_step)
        remaining_volume = self._floor_to_step(plan.volume - close_volume, lot_step)

        if close_volume <= 0:
            self._logger.error("[PM][partial_close_request] ticket=%s invalid_volume=%.4f", plan.ticket, close_volume)
            return None

        if remaining_volume < min_volume:
            close_volume = self._floor_to_step(plan.volume, lot_step)
            remaining_volume = 0.0
            if close_volume < min_volume:
                self._logger.warning(
                    "[PM][partial_close_request] ticket=%s rejected full_close=%.4f min=%.4f",
                    plan.ticket,
                    close_volume,
                    min_volume,
                )
                return None

        if close_volume < min_volume and remaining_volume > 0:
            self._logger.warning(
                "[PM][partial_close_request] ticket=%s rejected close=%.4f min=%.4f remaining=%.4f",
                plan.ticket,
                close_volume,
                min_volume,
                remaining_volume,
            )
            return None

        self._logger.info(
            "[PM][partial_close_request] ticket=%s position_volume=%.4f requested=%.4f close=%.4f remaining=%.4f min=%.4f step=%.4f",
            plan.ticket,
            float(plan.volume),
            float(requested_volume),
            float(close_volume),
            float(remaining_volume),
            float(min_volume),
            float(lot_step),
        )

        result = self._call_mt5("close_position", ticket=plan.ticket, volume=close_volume)
        ok = bool(result and result.get("success"))
        if ok:
            self._logger.info("[PM][PARTIAL_CLOSE_CONFIRMED] ticket=%s volume=%.3f", plan.ticket, close_volume)
            return {
                "close_volume": float(close_volume),
                "remaining_volume": float(remaining_volume),
            }
        else:
            self._logger.warning("[PM][PARTIAL_CLOSE_FAIL] ticket=%s result=%s", plan.ticket, result)
            self._logger.warning("[NDS][TP1_PARTIAL_FAIL] ticket=%s result=%s", plan.ticket, result)
            self._logger.warning("[NDS][SL_BE_FAIL] ticket=%s result=%s", plan.ticket, result)
        return None

    def _modify_sl(self, plan: PositionPlan, new_sl: float) -> bool:
        result = self._modify_position(plan.ticket, new_sl=new_sl, context="MODIFY_SL")
        ok = bool(result and result.get("success"))
        if ok:
            self._logger.info("[PM][SL_BE_CONFIRMED] ticket=%s sl=%.2f", plan.ticket, new_sl)
        else:
            self._logger.warning("[PM][MOVE_SL_FAIL] ticket=%s result=%s", plan.ticket, result)
            self._logger.warning("[NDS][SL_BE_FAIL] ticket=%s result=%s", plan.ticket, result)
        return ok

    def _set_tp2(self, plan: PositionPlan) -> bool:
        result = self._modify_position(plan.ticket, new_tp=plan.tp2_price, context="SET_TP2")
        ok = bool(result and result.get("success"))
        if ok:
            self._logger.info("[PM][TP2_SENT_CONFIRMED] ticket=%s tp2=%.2f", plan.ticket, plan.tp2_price)
        else:
            self._logger.warning("[PM][SET_TP2_FAIL] ticket=%s result=%s", plan.ticket, result)
        return ok

    def _compute_sl_to_be(self, plan: PositionPlan) -> float:
        offset = pips_to_price(self._get_cover_pips(), self._resolve_point_size())
        if plan.direction == "BUY":
            return plan.entry_price + offset
        return plan.entry_price - offset

    def _crossed_tp1(self, plan: PositionPlan, price: float) -> bool:
        return (plan.direction == "BUY" and price >= plan.tp1_price) or (
            plan.direction == "SELL" and price <= plan.tp1_price
        )

    def _crossed_tp2(self, plan: PositionPlan, price: float) -> bool:
        return (plan.direction == "BUY" and price >= plan.tp2_price) or (
            plan.direction == "SELL" and price <= plan.tp2_price
        )

    def _handle_broker_closed(self, ticket: int) -> None:
        plan = self._plans.get(ticket)
        if plan is None:
            return
        plan.status = PositionStatus.STATUS_CLOSED
        close_summary = self._resolve_close_from_history(ticket)
        plan.close_summary = close_summary
        self._logger.info("[PM][CLOSED] ticket=%s summary=%s", ticket, close_summary)
        if self.trade_tracker is not None and ticket in getattr(self.trade_tracker, "active_trades", {}):
            record = self.trade_tracker.active_trades.get(ticket)
            if record:
                self.trade_tracker.register_pending_close(ticket, record, datetime.utcnow())
        self._plans.pop(ticket, None)

    def _resolve_close_from_history(self, ticket: int) -> Dict[str, Any]:
        deals = self._get_position_deals(ticket)
        if not deals:
            return {"position": ticket, "resolved": False}
        ordered = sorted(deals, key=lambda d: int(d.get("time", 0)))
        last = ordered[-1]
        return {
            "position": ticket,
            "resolved": True,
            "exit_price": last.get("price"),
            "profit": sum(float(deal.get("profit") or 0.0) for deal in ordered),
            "exit_reason": last.get("reason") or last.get("comment") or "UNKNOWN",
        }

    def _get_position_deals(self, ticket: int) -> List[Dict[str, Any]]:
        fn = getattr(self.mt5_client, "history_deals_get", None)
        if callable(fn):
            try:
                raw = fn(position=ticket)
            except TypeError:
                raw = fn(ticket)
            return self._normalize_deals(raw)

        mt5_call = getattr(self.mt5_client, "_mt5_call", None)
        if callable(mt5_call):
            try:
                import MetaTrader5 as mt5  # type: ignore

                raw = mt5_call(mt5.history_deals_get, position=ticket)
                if not raw:
                    now = datetime.utcnow()
                    raw = mt5_call(mt5.history_deals_get, now - timedelta(days=7), now)
                return [
                    deal
                    for deal in self._normalize_deals(raw)
                    if int(deal.get("position") or 0) == ticket
                ]
            except Exception:
                self._logger.exception("[PM][HISTORY_FAIL] ticket=%s", ticket)
        return []

    @staticmethod
    def _normalize_deals(raw: Any) -> List[Dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                return raw
            normalized: List[Dict[str, Any]] = []
            for deal in raw:
                normalized.append(
                    {
                        "ticket": getattr(deal, "ticket", None),
                        "position": getattr(deal, "position_id", None),
                        "price": getattr(deal, "price", None),
                        "profit": getattr(deal, "profit", 0.0),
                        "reason": getattr(deal, "reason", None),
                        "comment": getattr(deal, "comment", None),
                        "time": getattr(deal, "time", 0),
                    }
                )
            return normalized
        return []

    def _modify_position(self, ticket: int, *, new_sl: Optional[float] = None, new_tp: Optional[float] = None, context: str) -> Dict[str, Any]:
        result = self._call_mt5("modify_position", ticket=ticket, new_sl=new_sl, new_tp=new_tp)
        if not result:
            self._logger.warning("[PM][%s] ticket=%s empty_result", context, ticket)
            return {"success": False, "error": "empty_result"}
        return result

    def _call_mt5(self, method_name: str, **kwargs: Any) -> Dict[str, Any]:
        fn = getattr(self.mt5_client, method_name, None)
        if callable(fn):
            try:
                result = fn(**kwargs)
                if isinstance(result, dict):
                    return result
            except Exception:
                self._logger.exception("[PM][MT5_CALL_FAIL] method=%s kwargs=%s", method_name, kwargs)
                return {"success": False}

        if method_name == "close_position":
            fallback = getattr(self.mt5_client, "partial_close", None)
            if callable(fallback):
                try:
                    result = fallback(position=kwargs.get("ticket"), volume=kwargs.get("volume"))
                    if isinstance(result, dict):
                        return result
                    return {"success": bool(result)}
                except Exception:
                    self._logger.exception("[PM][MT5_FALLBACK_FAIL] method=partial_close kwargs=%s", kwargs)

        return {"success": False, "error": f"missing_method:{method_name}"}

    def _resolve_point_size(self) -> float:
        point_size, _ = resolve_point_size_with_source(self.config, default=None)
        if point_size and point_size > 0:
            return float(point_size)
        return 0.01

    def _get_cover_pips(self) -> float:
        flow_settings = self.config.get("flow_settings", {}) if isinstance(self.config, dict) else {}
        return float(flow_settings.get("FLOW_TP1_MOVE_SL_TO_BE_PLUS_PIPS", 0.0) or 0.0)

    def _get_trade_metadata(self, position: PositionContract) -> Dict[str, Any]:
        if self.trade_tracker is None:
            return {}
        ticket = int(position.get("position_ticket") or 0)
        record = getattr(self.trade_tracker, "active_trades", {}).get(ticket)
        if record and record.get("open_event", {}).get("metadata"):
            return record.get("open_event", {}).get("metadata", {}) or {}
        resolver = getattr(self.trade_tracker, "resolve_metadata_for_position", None)
        if callable(resolver):
            return resolver(position)
        return {}
