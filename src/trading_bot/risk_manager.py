"""Deterministic live risk manager (MARKET/LIMIT only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from config.settings import config
from src.trading_bot.config_utils import get_setting
from src.trading_bot.nds.distance_utils import (
    DEFAULT_POINT_SIZE,
    calculate_distance_metrics,
    pips_to_price,
    resolve_point_size_with_source,
)
from src.trading_bot.nds.models import FinalizedOrderParams, LivePriceSnapshot
from src.trading_bot.session_policy import SessionDecision, evaluate_session, session_weight_from_config

if TYPE_CHECKING:
    from src.trading_bot.nds.models import AnalysisResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Geometry:
    signal: str
    entry_price: float
    stop_loss: float
    take_profit: float
    take_profit2: Optional[float]
    sl_pips: float
    tp1_pips: float
    tp2_pips: float


@dataclass
class ScalpingRiskParameters:
    lot_size: float
    risk_amount: float
    risk_percent: float
    actual_risk_percent: float
    position_value: float
    margin_required: float
    leverage_used: float
    validation_passed: bool
    warnings: list
    notes: list
    calculation_details: Dict[str, Any]
    scalping_specific: Dict[str, Any]


class RiskEngine:
    """Pure risk engine for deterministic order acceptance."""

    def __init__(self, settings: Dict[str, Any], gold_specs: Dict[str, Any]):
        self.settings = settings
        self.gold_specs = gold_specs

    def resolve_entry_policy(
        self,
        *,
        signal: str,
        planned_entry: Optional[float],
        entry_model: str,
        market_entry: float,
        point_size: float,
    ) -> Tuple[bool, str, float, str, float]:
        model = str(entry_model or "MARKET").upper()
        if model == "STOP":
            return False, "NONE", 0.0, "STOP entry model is disabled in live mode.", 0.0

        if model not in {"MARKET", "LIMIT"}:
            model = "MARKET"

        if model == "MARKET":
            entry_price = float(market_entry)
        else:
            if planned_entry is None:
                return False, "NONE", 0.0, "LIMIT entry missing planned_entry.", 0.0
            entry_price = float(planned_entry)
            if signal == "BUY" and entry_price >= market_entry:
                return False, "NONE", 0.0, "LIMIT BUY must be below market.", 0.0
            if signal == "SELL" and entry_price <= market_entry:
                return False, "NONE", 0.0, "LIMIT SELL must be above market.", 0.0

        deviation_pips = float(
            calculate_distance_metrics(
                entry_price=entry_price,
                current_price=market_entry,
                point_size=point_size,
            ).get("dist_pips")
            or 0.0
        )
        return True, model, entry_price, "ok", deviation_pips

    def build_geometry(
        self,
        *,
        signal: str,
        entry_price: float,
        analysis_payload: Dict[str, Any],
        point_size: float,
    ) -> Geometry:
        risk_settings = analysis_payload.get("_risk_settings", {})
        atr = analysis_payload.get("atr") or analysis_payload.get("atr_value") or 0.0
        atr = float(atr or 0.0)

        sl_pips_cfg = float(risk_settings.get("SL_MIN_PIPS", risk_settings.get("MIN_SL_PIPS", 12.0)) or 12.0)
        max_sl_pips = float(risk_settings.get("SL_MAX_PIPS_SCALP", max(sl_pips_cfg, 60.0)) or max(sl_pips_cfg, 60.0))
        atr_sl_mult = float(risk_settings.get("SCALP_ATR_SL_MULT", 1.0) or 1.0)
        atr_sl_pips = (atr * atr_sl_mult) / point_size if atr > 0 and point_size > 0 else sl_pips_cfg
        sl_pips = max(sl_pips_cfg, min(max_sl_pips, atr_sl_pips))

        tp1_pips = float(risk_settings.get("TP1_PIPS", 35.0) or 35.0)
        tp2_enabled = bool(risk_settings.get("TP2_ENABLED", True))
        tp2_pips = float(risk_settings.get("TP2_PIPS", tp1_pips * 2.0) or (tp1_pips * 2.0)) if tp2_enabled else 0.0

        sl_dist = pips_to_price(sl_pips, point_size)
        tp1_dist = pips_to_price(tp1_pips, point_size)
        tp2_dist = pips_to_price(tp2_pips, point_size) if tp2_enabled else 0.0

        if signal == "BUY":
            sl = entry_price - sl_dist
            tp1 = entry_price + tp1_dist
            tp2 = entry_price + tp2_dist if tp2_enabled else None
        else:
            sl = entry_price + sl_dist
            tp1 = entry_price - tp1_dist
            tp2 = entry_price - tp2_dist if tp2_enabled else None

        return Geometry(
            signal=signal,
            entry_price=float(entry_price),
            stop_loss=float(sl),
            take_profit=float(tp1),
            take_profit2=float(tp2) if tp2 is not None else None,
            sl_pips=float(sl_pips),
            tp1_pips=float(tp1_pips),
            tp2_pips=float(tp2_pips),
        )

    @staticmethod
    def compute_rr(*, entry_price: float, stop_loss: float, take_profit: float, point_size: float) -> Tuple[float, float, float]:
        sl_pips = abs(float(entry_price) - float(stop_loss)) / float(point_size)
        tp1_pips = abs(float(take_profit) - float(entry_price)) / float(point_size)
        rr = (tp1_pips / sl_pips) if sl_pips > 0 else 0.0
        return float(sl_pips), float(tp1_pips), float(rr)

    @staticmethod
    def validate_rr(*, rr: float, min_rr: float, sl_pips: float, tp1_pips: float) -> Tuple[bool, str]:
        if sl_pips <= 0:
            return False, "SL distance must be positive."
        if tp1_pips <= 0:
            return False, "TP1 distance must be positive."
        if rr < min_rr:
            return False, f"RR ratio below minimum ({rr:.4f} < {min_rr:.4f})."
        return True, "ok"

    def size_position(
        self,
        *,
        account_equity: float,
        risk_amount_usd: float,
        sl_pips: float,
    ) -> ScalpingRiskParameters:
        point = float(self.gold_specs.get("point", DEFAULT_POINT_SIZE) or DEFAULT_POINT_SIZE)
        tick_value = float(self.gold_specs.get("tick_value_per_lot", 1.0) or 1.0)
        min_lot = float(self.gold_specs.get("min_lot", 0.01) or 0.01)
        max_lot = float(self.gold_specs.get("max_lot", 50.0) or 50.0)
        step = float(self.gold_specs.get("lot_step", 0.01) or 0.01)

        sl_price = max(point, pips_to_price(sl_pips, point))
        loss_per_lot = max(1e-9, (sl_price / point) * tick_value)
        raw_lot = risk_amount_usd / loss_per_lot if loss_per_lot > 0 else 0.0
        stepped = round(raw_lot / step) * step if step > 0 else raw_lot
        lot = max(min_lot, min(max_lot, stepped))

        actual_risk = lot * loss_per_lot
        valid = lot > 0 and actual_risk > 0

        return ScalpingRiskParameters(
            lot_size=float(lot),
            risk_amount=float(risk_amount_usd),
            risk_percent=(float(risk_amount_usd) / float(account_equity) * 100.0) if account_equity > 0 else 0.0,
            actual_risk_percent=(float(actual_risk) / float(account_equity) * 100.0) if account_equity > 0 else 0.0,
            position_value=0.0,
            margin_required=0.0,
            leverage_used=0.0,
            validation_passed=valid,
            warnings=[],
            notes=[],
            calculation_details={"raw_lot": raw_lot, "loss_per_lot": loss_per_lot},
            scalping_specific={"sl_distance": sl_price},
        )

    @staticmethod
    def enforce_invariants(*, signal: str, entry: float, sl: float, tp1: float) -> Tuple[bool, str]:
        if signal == "BUY":
            if not (sl < entry < tp1):
                return False, "BUY geometry invariant failed."
        elif signal == "SELL":
            if not (tp1 < entry < sl):
                return False, "SELL geometry invariant failed."
        else:
            return False, "Invalid signal."
        return True, "ok"

    @staticmethod
    def build_decision(
        *,
        signal: str,
        order_type: str,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lot_size: float,
        risk_amount_usd: float,
        rr_ratio: float,
        deviation_pips: float,
        decision_notes: List[str],
        is_trade_allowed: bool,
        reject_reason: Optional[str],
        tp2: Optional[float],
        sl_pips: float,
        tp1_pips: float,
        tp2_pips: float,
        min_rr: float,
    ) -> FinalizedOrderParams:
        return FinalizedOrderParams(
            signal=signal,
            order_type=order_type,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            risk_amount_usd=risk_amount_usd,
            rr_ratio=rr_ratio,
            deviation_pips=deviation_pips,
            decision_notes=decision_notes,
            is_trade_allowed=is_trade_allowed,
            reject_reason=reject_reason,
            take_profit2=tp2,
            tp2=tp2,
            final_entry=entry_price,
            final_stop_loss=stop_loss,
            final_take_profit=take_profit,
            final_sl=stop_loss,
            final_tp=take_profit,
            lot=lot_size,
            rr_tp1=rr_ratio,
            rr_tp2=None,
            rr_checked="TP1",
            rr_validate_mode="TP1",
            min_rr_effective=min_rr,
            min_rr_source="risk_settings.MIN_RR_RATIO",
            sl_pips=sl_pips,
            tp1_pips=tp1_pips,
            tp2_pips=tp2_pips,
            tp_execution_mode="TP1_PARTIAL_MANAGED" if tp2 is not None else "TP1_ONLY",
            tp1_virtual_trigger=False,
            calculated_take_profit=take_profit,
            broker_take_profit=0.0 if tp2 is not None else take_profit,
        )


class ScalpingRiskManager:
    VERSION_TAG = "2026-02-20-deterministic-risk-engine"

    def __init__(self, overrides: Optional[Dict[str, Any]] = None, logger: logging.Logger = None):
        full_config = config.get_full_config()
        if overrides:
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(full_config.get(key), dict):
                    full_config[key].update(value)
                else:
                    full_config[key] = value
        self._logger = logger or logging.getLogger(__name__)
        self.settings = self._merge_with_config(full_config, {})
        self.config = self.settings

        trading_settings = full_config.get("trading_settings", {})
        specs = trading_settings.get("GOLD_SPECIFICATIONS", {}) if isinstance(trading_settings, dict) else {}
        self.GOLD_SPECS = {
            "point": float(specs.get("POINT", specs.get("point", DEFAULT_POINT_SIZE)) or DEFAULT_POINT_SIZE),
            "digits": int(specs.get("DIGITS", specs.get("digits", 2)) or 2),
            "contract_size": float(specs.get("CONTRACT_SIZE", specs.get("contract_size", 100.0)) or 100.0),
            "tick_value_per_lot": float(specs.get("TICK_VALUE_PER_LOT", specs.get("tick_value_per_lot", 1.0)) or 1.0),
            "min_lot": float(specs.get("MIN_LOT", specs.get("min_lot", 0.01)) or 0.01),
            "max_lot": float(specs.get("MAX_LOT", specs.get("max_lot", 50.0)) or 50.0),
            "lot_step": float(specs.get("LOT_STEP", specs.get("lot_step", 0.01)) or 0.01),
        }

        self.daily_risk_used = 0.0
        self.daily_profit_loss = 0.0
        self.active_positions = 0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.scalping_positions: List[Dict[str, Any]] = []
        self.scalping_stats: Dict[str, Any] = {"total_scalps": 0, "winning_scalps": 0}

        self.last_signal_confidence = 0.0
        self.last_adx = 0.0
        self.last_session = "UNKNOWN"
        self._logger.info("[RISK][VERSION] %s", self.VERSION_TAG)

    @staticmethod
    def _merge_with_config(full_config: Dict[str, Any], _: Dict[str, Any]) -> Dict[str, Any]:
        merged = {}
        for section in ("trading_settings", "risk_settings", "risk_manager_config"):
            block = full_config.get(section, {})
            if isinstance(block, dict):
                merged.update(block)
        return merged

    def _get_gold_spec(self, key: str, default: Any = None) -> Any:
        k = str(key or "")
        lower = k.lower()
        upper = k.upper()
        if lower in self.GOLD_SPECS:
            return self.GOLD_SPECS[lower]
        map_up = {
            "POINT": "point",
            "DIGITS": "digits",
            "CONTRACT_SIZE": "contract_size",
            "TICK_VALUE_PER_LOT": "tick_value_per_lot",
            "MIN_LOT": "min_lot",
            "MAX_LOT": "max_lot",
            "LOT_STEP": "lot_step",
        }
        if upper in map_up:
            return self.GOLD_SPECS.get(map_up[upper], default)
        return default

    def _normalize_analysis_payload(self, analysis: Union[AnalysisResult, Dict[str, Any]]) -> Dict[str, Any]:
        return dict(analysis) if isinstance(analysis, dict) else asdict(analysis)

    def _resolve_adx_from_signal(self, signal_data: Optional[Dict[str, Any]]) -> Tuple[float, str]:
        if not isinstance(signal_data, dict):
            return 0.0, "missing"
        for key in ("adx", "ADX"):
            val = signal_data.get(key)
            if val is not None:
                return float(val), f"payload:{key}"
        mm = signal_data.get("market_metrics", {}) if isinstance(signal_data.get("market_metrics"), dict) else {}
        if mm.get("adx") is not None:
            return float(mm.get("adx")), "payload:market_metrics.adx"
        return 0.0, "missing"

    def _resolve_confidence_from_signal(self, signal_data: Optional[Dict[str, Any]]) -> float:
        if not isinstance(signal_data, dict):
            return 0.0
        return float(signal_data.get("confidence") or 0.0)

    def _resolve_session_decision(self, signal_data: Optional[Dict[str, Any]]) -> Tuple[SessionDecision, str]:
        if isinstance(signal_data, dict) and signal_data.get("session"):
            payload = {
                "session_name": str(signal_data.get("session") or "UNKNOWN").upper(),
                "is_tradable": bool(signal_data.get("session_activity", True)),
                "weight": float(signal_data.get("session_weight", 1.0) or 1.0),
                "policy_mode": str(signal_data.get("session_policy", "payload")),
                "block_reason": signal_data.get("session_block_reason"),
            }
            return SessionDecision.from_payload(payload), "payload"
        return evaluate_session(None, self.config), "computed"

    def get_current_scalping_session(self) -> str:
        return str(self.last_session or "UNKNOWN")

    def can_scalp(self, account_equity: float, signal_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        reasons: List[str] = []
        s = self.settings

        max_daily_percent = float(s.get("MAX_DAILY_RISK_PERCENT", 1.0) or 1.0)
        used_percent = (self.daily_risk_used / account_equity) * 100.0 if account_equity > 0 else 0.0
        if used_percent >= max_daily_percent:
            reasons.append("Daily risk limit reached")

        if self.consecutive_losses >= int(s.get("MAX_CONSECUTIVE_LOSSES", 2) or 2):
            reasons.append("Consecutive loss guard")

        if self.active_positions >= int(s.get("MAX_POSITIONS", 4) or 4):
            reasons.append("Active position limit")

        if self.trades_today >= int(s.get("MAX_DAILY_TRADES", 20) or 20):
            reasons.append("Daily trades limit")

        session_decision, _ = self._resolve_session_decision(signal_data)
        self.last_session = session_decision.session_name
        conf = self._resolve_confidence_from_signal(signal_data)
        adx, _ = self._resolve_adx_from_signal(signal_data)
        self.last_signal_confidence = conf
        self.last_adx = adx
        if not session_decision.is_tradable:
            reasons.append(session_decision.block_reason or "Non-tradable session")

        return (len(reasons) == 0), ("ALLOWED" if not reasons else " | ".join(reasons))

    def calculate_scalping_position_size(
        self,
        account_equity: float,
        risk_amount_usd: float,
        entry_price: float,
        stop_loss: float,
        symbol: str = "XAUUSD",
    ) -> ScalpingRiskParameters:
        point = float(self._get_gold_spec("point", DEFAULT_POINT_SIZE) or DEFAULT_POINT_SIZE)
        sl_pips = abs(float(entry_price) - float(stop_loss)) / point
        engine = RiskEngine(self.settings, self.GOLD_SPECS)
        return engine.size_position(
            account_equity=float(account_equity or 0.0),
            risk_amount_usd=float(risk_amount_usd or 0.0),
            sl_pips=float(sl_pips),
        )

    def _compute_scalping_sl_tp(self, signal: str, entry_price: float, analysis_payload: Dict[str, Any], point_size: float) -> Dict[str, float]:
        engine = RiskEngine(self.settings, self.GOLD_SPECS)
        payload = dict(analysis_payload or {})
        payload["_risk_settings"] = self.settings
        g = engine.build_geometry(signal=signal, entry_price=entry_price, analysis_payload=payload, point_size=point_size)
        return {
            "stop_loss": g.stop_loss,
            "take_profit": g.take_profit,
            "take_profit2": g.take_profit2,
            "sl_pips": g.sl_pips,
            "tp1_pips": g.tp1_pips,
            "tp2_pips": g.tp2_pips,
            "sl_source": "deterministic",
            "tp1_source": "deterministic",
            "tp2_source": "deterministic" if g.take_profit2 is not None else "disabled",
        }

    def finalize_order(
        self,
        analysis: Union[AnalysisResult, Dict[str, Any]],
        live: Union[LivePriceSnapshot, Dict[str, Any]],
        symbol: str,
        config: Dict[str, Any],
    ) -> FinalizedOrderParams:
        analysis_payload = self._normalize_analysis_payload(analysis)
        live_payload = live if isinstance(live, dict) else asdict(live)

        signal = str(analysis_payload.get("signal") or "NONE").upper()
        if signal not in {"BUY", "SELL"}:
            return FinalizedOrderParams(
                signal=signal,
                order_type="NONE",
                symbol=symbol,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["No actionable signal."],
                is_trade_allowed=False,
                reject_reason="Signal is NONE/NEUTRAL.",
            )

        bid = live_payload.get("bid")
        ask = live_payload.get("ask")
        if bid is None or ask is None:
            return FinalizedOrderParams(
                signal=signal,
                order_type="NONE",
                symbol=symbol,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["Missing live bid/ask."],
                is_trade_allowed=False,
                reject_reason="Live snapshot incomplete.",
            )

        market_entry = float(ask if signal == "BUY" else bid)
        point_size, _ = resolve_point_size_with_source(config, default=self._get_gold_spec("point", DEFAULT_POINT_SIZE))

        engine = RiskEngine(self.settings, self.GOLD_SPECS)
        planned_entry = analysis_payload.get("entry_level") or analysis_payload.get("entry_price")
        ok_entry, order_type, entry_price, entry_reason, deviation_pips = engine.resolve_entry_policy(
            signal=signal,
            planned_entry=float(planned_entry) if planned_entry is not None else None,
            entry_model=str(analysis_payload.get("entry_model") or "MARKET"),
            market_entry=market_entry,
            point_size=point_size,
        )
        if not ok_entry:
            return engine.build_decision(
                signal=signal,
                order_type="NONE",
                symbol=symbol,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=[entry_reason],
                is_trade_allowed=False,
                reject_reason=entry_reason,
                tp2=None,
                sl_pips=0.0,
                tp1_pips=0.0,
                tp2_pips=0.0,
                min_rr=float(self.settings.get("MIN_RR_RATIO", self.settings.get("MIN_RISK_REWARD", 1.0)) or 1.0),
            )

        payload_for_geom = dict(analysis_payload)
        payload_for_geom["_risk_settings"] = self.settings
        geometry = engine.build_geometry(
            signal=signal,
            entry_price=float(entry_price),
            analysis_payload=payload_for_geom,
            point_size=float(point_size),
        )

        inv_ok, inv_reason = engine.enforce_invariants(
            signal=signal,
            entry=float(geometry.entry_price),
            sl=float(geometry.stop_loss),
            tp1=float(geometry.take_profit),
        )
        if not inv_ok:
            return engine.build_decision(
                signal=signal,
                order_type="NONE",
                symbol=symbol,
                entry_price=geometry.entry_price,
                stop_loss=geometry.stop_loss,
                take_profit=geometry.take_profit,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=deviation_pips,
                decision_notes=[inv_reason],
                is_trade_allowed=False,
                reject_reason=inv_reason,
                tp2=geometry.take_profit2,
                sl_pips=geometry.sl_pips,
                tp1_pips=geometry.tp1_pips,
                tp2_pips=geometry.tp2_pips,
                min_rr=float(self.settings.get("MIN_RR_RATIO", self.settings.get("MIN_RISK_REWARD", 1.0)) or 1.0),
            )

        sl_pips, tp1_pips, rr = engine.compute_rr(
            entry_price=geometry.entry_price,
            stop_loss=geometry.stop_loss,
            take_profit=geometry.take_profit,
            point_size=float(point_size),
        )
        min_rr = float(self.settings.get("MIN_RR_RATIO", self.settings.get("MIN_RISK_REWARD", 1.0)) or 1.0)
        rr_ok, rr_reason = engine.validate_rr(rr=rr, min_rr=min_rr, sl_pips=sl_pips, tp1_pips=tp1_pips)
        if not rr_ok:
            return engine.build_decision(
                signal=signal,
                order_type="NONE",
                symbol=symbol,
                entry_price=geometry.entry_price,
                stop_loss=geometry.stop_loss,
                take_profit=geometry.take_profit,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=rr,
                deviation_pips=deviation_pips,
                decision_notes=[rr_reason],
                is_trade_allowed=False,
                reject_reason=rr_reason,
                tp2=geometry.take_profit2,
                sl_pips=sl_pips,
                tp1_pips=tp1_pips,
                tp2_pips=geometry.tp2_pips,
                min_rr=min_rr,
            )

        account_equity = float(config.get("ACCOUNT_BALANCE") or config.get("account_balance") or 10000.0)
        risk_amount_usd = float(self.settings.get("RISK_AMOUNT_USD", self.settings.get("SCALPING_RISK_USD", 20.0)) or 20.0)
        size = engine.size_position(
            account_equity=account_equity,
            risk_amount_usd=risk_amount_usd,
            sl_pips=sl_pips,
        )
        if not size.validation_passed:
            return engine.build_decision(
                signal=signal,
                order_type="NONE",
                symbol=symbol,
                entry_price=geometry.entry_price,
                stop_loss=geometry.stop_loss,
                take_profit=geometry.take_profit,
                lot_size=0.0,
                risk_amount_usd=risk_amount_usd,
                rr_ratio=rr,
                deviation_pips=deviation_pips,
                decision_notes=["Position sizing validation failed."],
                is_trade_allowed=False,
                reject_reason="Sizing failure.",
                tp2=geometry.take_profit2,
                sl_pips=sl_pips,
                tp1_pips=tp1_pips,
                tp2_pips=geometry.tp2_pips,
                min_rr=min_rr,
            )

        return engine.build_decision(
            signal=signal,
            order_type=order_type,
            symbol=symbol,
            entry_price=geometry.entry_price,
            stop_loss=geometry.stop_loss,
            take_profit=geometry.take_profit,
            lot_size=float(size.lot_size),
            risk_amount_usd=risk_amount_usd,
            rr_ratio=rr,
            deviation_pips=deviation_pips,
            decision_notes=[
                "Deterministic entry policy accepted.",
                f"RR_TP1={rr:.4f} MIN_RR={min_rr:.4f}",
            ],
            is_trade_allowed=True,
            reject_reason=None,
            tp2=geometry.take_profit2,
            sl_pips=sl_pips,
            tp1_pips=tp1_pips,
            tp2_pips=geometry.tp2_pips,
            min_rr=min_rr,
        )

    def add_position(self, position_size: float):
        self.active_positions += 1
        self.trades_today += 1
        self.scalping_positions.append({"size": position_size, "time": datetime.now()})

    def remove_position(self, position_size: float, profit_loss: float):
        self.active_positions = max(0, self.active_positions - 1)
        self.daily_profit_loss += profit_loss
        self.daily_risk_used = max(0.0, self.daily_risk_used - abs(position_size) * 0.1)

    def get_scalping_summary(self) -> Dict[str, Any]:
        return {
            "can_scalp": self.can_scalp(1000.0)[0],
            "daily_pnl": self.daily_profit_loss,
            "active_positions": self.active_positions,
            "last_confidence": self.last_signal_confidence,
            "last_adx": self.last_adx,
            "last_session": self.last_session,
            "session_weight": session_weight_from_config(self.last_session, self.config),
        }


def create_scalping_risk_manager(overrides: Optional[Dict[str, Any]] = None, **kwargs) -> ScalpingRiskManager:
    return ScalpingRiskManager(overrides=overrides, **kwargs)
