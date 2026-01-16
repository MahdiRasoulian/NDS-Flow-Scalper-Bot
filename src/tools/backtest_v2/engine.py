from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd

from src.trading_bot.nds.analyzer import analyze_gold_market
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.nds.distance_utils import calculate_distance_metrics, resolve_point_size_from_config
from src.trading_bot.risk_manager import create_scalping_risk_manager
from src.trading_bot.config_utils import get_setting

from .io import build_analyzer_config
from .metrics import compute_trade_metrics, summarize_cycle_log, format_summary_text

logger = logging.getLogger(__name__)


@dataclass
class Position:
    id: str
    side: str
    symbol: str
    open_time: pd.Timestamp
    open_bar_index: int
    entry_price: float
    stop_loss: float
    take_profit: float
    lot: float
    order_type: str
    confidence: float
    score: float
    rr: float
    session: str
    planned_entry: float
    deviation_pips: float
    notes: List[str]

    close_time: Optional[pd.Timestamp] = None
    close_bar_index: Optional[int] = None
    close_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_usd: Optional[float] = None
    duration_bars: Optional[int] = None


@dataclass
class PendingOrder:
    id: str
    side: str
    symbol: str
    created_time: pd.Timestamp
    created_bar_index: int
    planned_entry: float
    stop_loss: float
    take_profit: float
    lot: float
    order_type: str
    confidence: float
    score: float
    rr: float
    session: str
    deviation_pips: float
    notes: List[str]

    filled_time: Optional[pd.Timestamp] = None
    filled_bar_index: Optional[int] = None
    filled_price: Optional[float] = None

    def to_position(self) -> Position:
        return Position(
            id=self.id,
            side=self.side,
            symbol=self.symbol,
            open_time=self.filled_time or self.created_time,
            open_bar_index=self.filled_bar_index if self.filled_bar_index is not None else self.created_bar_index,
            entry_price=float(self.filled_price if self.filled_price is not None else self.planned_entry),
            stop_loss=float(self.stop_loss),
            take_profit=float(self.take_profit),
            lot=float(self.lot),
            order_type=str(self.order_type),
            confidence=float(self.confidence),
            score=float(self.score),
            rr=float(self.rr),
            session=str(self.session),
            planned_entry=float(self.planned_entry),
            deviation_pips=float(self.deviation_pips),
            notes=list(self.notes or []),
        )


@dataclass
class BacktestConfig:
    symbol: str = "XAUUSD!"
    timeframe: str = "M15"
    warmup_bars: int = 300
    spread: float = 0.25
    slippage: float = 0.10
    commission_per_lot: float = 0.0

    allow_multiple_positions: bool = True
    max_positions: int = 5
    min_candles_between_trades: int = 10
    min_time_between_trades_minutes: int = 60
    daily_max_trades: int = 40

    enable_limit_orders: bool = True
    pending_expire_minutes: int = 60

    starting_equity: float = 1000.0
    max_daily_risk_percent: float = 6.0

    bars_per_day: int = 96

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _bars_per_day(timeframe: str) -> int:
        tf = str(timeframe).upper()
        mapping = {
            "M1": 1440,
            "M5": 288,
            "M15": 96,
            "M30": 48,
            "H1": 24,
            "H4": 6,
            "D1": 1,
        }
        return mapping.get(tf, 96)

    @classmethod
    def from_bot_config(
        cls,
        config: Dict[str, Any],
        *,
        warmup: Optional[int] = None,
        spread: Optional[float] = None,
        slippage: Optional[float] = None,
        starting_equity: Optional[float] = None,
    ) -> "BacktestConfig":
        trading_settings = get_setting(config, "trading_settings", {}) or {}
        trading_rules = get_setting(config, "trading_rules", {}) or {}
        gold_specs = trading_settings.get("GOLD_SPECIFICATIONS", {}) if isinstance(trading_settings, dict) else {}

        symbol = trading_settings.get("SYMBOL", "XAUUSD!")
        timeframe = trading_settings.get("TIMEFRAME", "M15")
        typical_spread = gold_specs.get("TYPICAL_SPREAD", 0.25)
        typical_slippage = gold_specs.get("TYPICAL_SLIPPAGE", 0.10)

        cfg = cls(
            symbol=symbol,
            timeframe=timeframe,
            warmup_bars=int(warmup) if warmup is not None else 300,
            spread=float(spread if spread is not None else typical_spread),
            slippage=float(slippage if slippage is not None else typical_slippage),
            allow_multiple_positions=bool(trading_rules.get("ALLOW_MULTIPLE_POSITIONS", True)),
            max_positions=int(trading_rules.get("MAX_POSITIONS", 5)),
            min_candles_between_trades=int(trading_rules.get("MIN_CANDLES_BETWEEN_TRADES", 10)),
            min_time_between_trades_minutes=int(trading_rules.get("MIN_TIME_BETWEEN_TRADES_MINUTES", 60)),
            daily_max_trades=int(trading_rules.get("DAILY_MAX_TRADES", 40)),
            starting_equity=float(starting_equity if starting_equity is not None else config.get("ACCOUNT_BALANCE", 1000.0)),
            max_daily_risk_percent=float(get_setting(config, "risk_settings.MAX_DAILY_RISK_PERCENT", 6.0) or 6.0),
        )
        cfg.bars_per_day = cls._bars_per_day(cfg.timeframe)
        return cfg


@dataclass
class BacktestResult:
    config: Dict[str, Any]
    overrides: Dict[str, Any]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    cycle_log: pd.DataFrame
    metrics: Dict[str, Any]
    diagnostics: Dict[str, Any]
    summary_text: str


class BacktestEngine:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        bt_config: Optional[BacktestConfig] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        self.config = config
        self.overrides = overrides or {}
        self.bt_config = bt_config or BacktestConfig.from_bot_config(config)
        self.analyzer_config = build_analyzer_config(config)
        self.risk_manager = create_scalping_risk_manager(overrides=config)
        self.point_size = resolve_point_size_from_config(config)
        trading_settings = get_setting(config, "trading_settings", {}) or {}
        gold_specs = trading_settings.get("GOLD_SPECIFICATIONS", {}) if isinstance(trading_settings, dict) else {}
        self.contract_size = float(gold_specs.get("CONTRACT_SIZE", gold_specs.get("contract_size", 100.0)) or 100.0)

    def _normalize_signal(self, signal_value: str) -> str:
        sig = (signal_value or "NONE").upper()
        if sig in {"NEUTRAL", "NONE"}:
            return "NONE"
        return sig

    def _normalize_analysis(self, result: Any) -> Dict[str, Any]:
        if result is None:
            return {}
        if isinstance(result, dict):
            payload = dict(result)
        else:
            payload = dict(getattr(result, "__dict__", {}) or {})
        if "signal" not in payload:
            for key in ("signal", "final_signal", "trade_signal", "direction"):
                if hasattr(result, key):
                    payload["signal"] = getattr(result, key)
                    break
        if "confidence" not in payload:
            for key in ("confidence", "conf", "confidence_pct"):
                if hasattr(result, key):
                    payload["confidence"] = getattr(result, key)
                    break
        if "score" not in payload:
            for key in ("score", "normalized_score", "final_score"):
                if hasattr(result, key):
                    payload["score"] = getattr(result, key)
                    break
        payload["signal"] = self._normalize_signal(payload.get("signal", "NONE"))
        conf = payload.get("confidence", 0.0) or 0.0
        if 0.0 <= float(conf) <= 1.0:
            conf = float(conf) * 100.0
        payload["confidence"] = float(conf)
        payload["score"] = float(payload.get("score", 0.0) or 0.0)
        return payload

    def _extract_adx(self, payload: Dict[str, Any]) -> float:
        if payload.get("adx") is not None:
            return float(payload.get("adx") or 0.0)
        if payload.get("adx_value") is not None:
            return float(payload.get("adx_value") or 0.0)
        indicators = payload.get("indicators") or {}
        if isinstance(indicators, dict):
            if indicators.get("adx") is not None:
                return float(indicators.get("adx") or 0.0)
            if indicators.get("adx_value") is not None:
                return float(indicators.get("adx_value") or 0.0)
            adx_analysis = indicators.get("adx_analysis")
            if isinstance(adx_analysis, dict) and adx_analysis.get("adx") is not None:
                return float(adx_analysis.get("adx") or 0.0)
        return 0.0

    def _extract_entry_meta(self, payload: Dict[str, Any]) -> Tuple[str, str, str, str, Dict[str, Any]]:
        entry_type = payload.get("entry_type") or "NONE"
        entry_model = payload.get("entry_model") or "NONE"
        tier = payload.get("tier") or "NONE"
        entry_source = payload.get("entry_source") or payload.get("entry_source", "NONE")
        context = payload.get("context") or {}
        entry_idea = payload.get("entry_idea") or context.get("entry_idea") or {}
        if entry_idea:
            entry_type = entry_idea.get("entry_type", entry_type)
            entry_model = entry_idea.get("entry_model", entry_model)
            tier = entry_idea.get("tier", tier)
            entry_source = entry_idea.get("zone", entry_source)
        entry_context = entry_idea.get("metrics") or context.get("entry_context") or {}
        return str(entry_type), str(entry_model), str(tier), str(entry_source), entry_context

    def _extract_session_info(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Robust session extractor.
        Live analyzer logs show fields like: sess=ASIA w=0.80 ...
        Backtest must support multiple payload layouts.
        """
        def _as_dict(x: Any) -> Dict[str, Any]:
            return x if isinstance(x, dict) else {}

        # 1) Preferred: explicit session_analysis dict
        session = _as_dict(payload.get("session_analysis"))

        # 2) Common fallbacks: top-level keys or nested meta/context/details
        if not session:
            # try top-level session-like keys
            session = {
                "current_session": payload.get("current_session") or payload.get("session") or payload.get("sess"),
                "weight": payload.get("session_weight") or payload.get("weight"),
                "session_activity": payload.get("session_activity") or payload.get("activity"),
                "is_active_session": payload.get("is_active_session"),
                "untradable": payload.get("untradable"),
                "untradable_reasons": payload.get("untradable_reasons"),
            }

        # 3) Try nested containers often used by analyzers
        if not session.get("current_session") and not session.get("weight"):
            for container_key in ("meta", "context", "details", "diagnostics", "indicators", "confidence_breakdown"):
                container = _as_dict(payload.get(container_key))
                if not container:
                    continue
                # look for session-like subdict
                cand = _as_dict(container.get("session_analysis") or container.get("session"))
                if cand:
                    session = {**session, **cand}
                    break
                # or keys directly inside container
                if any(k in container for k in ("current_session", "session", "sess", "session_weight", "weight")):
                    session = {
                        "current_session": container.get("current_session") or container.get("session") or container.get("sess"),
                        "weight": container.get("session_weight") or container.get("weight"),
                        "session_activity": container.get("session_activity") or container.get("activity"),
                        "is_active_session": container.get("is_active_session"),
                        "untradable": container.get("untradable"),
                        "untradable_reasons": container.get("untradable_reasons"),
                    }
                    break

        # Normalize outputs
        current_session = str(session.get("current_session") or "UNKNOWN")
        weight = session.get("weight", session.get("session_weight", None))

        try:
            weight_f = float(weight) if weight is not None else None
        except Exception:
            weight_f = None

        # IMPORTANT: do not allow UNKNOWN to zero-out confidence/tier in backtest
        if current_session in ("UNKNOWN", "NONE", ""):
            current_session = "UNKNOWN"
            if weight_f is None or weight_f <= 0:
                weight_f = 1.0  # safe neutral multiplier

        if weight_f is None:
            weight_f = 1.0

        session_activity = str(session.get("session_activity") or "")
        is_active = bool(session.get("is_active_session", True))
        untradable = bool(session.get("untradable", False))
        untradable_reasons = str(session.get("untradable_reasons", "-"))

        return {
            "session": current_session,
            "session_weight": float(weight_f),
            "session_activity": session_activity,
            "is_active_session": is_active,
            "untradable": untradable,
            "untradable_reasons": untradable_reasons,
        }


    def _entry_distance_pips(self, entry_price: Optional[float], reference_price: Optional[float]) -> float:
        if entry_price is None or reference_price is None:
            return 0.0
        metrics = calculate_distance_metrics(entry_price, reference_price, point_size=self.point_size)
        return float(metrics.get("dist_pips") or 0.0)

    def _calculate_pnl(self, side: str, entry: float, exit_price: float, lot: float) -> float:
        direction = 1.0 if side == "BUY" else -1.0
        return (exit_price - entry) * direction * self.contract_size * lot

    def _apply_exit_checks(self, position: Position, high: float, low: float) -> Optional[Tuple[str, float]]:
        if position.side == "BUY":
            sl_hit = low <= position.stop_loss
            tp_hit = high >= position.take_profit
            if sl_hit and tp_hit:
                return "SL", position.stop_loss
            if sl_hit:
                return "SL", position.stop_loss
            if tp_hit:
                return "TP", position.take_profit
        else:
            sl_hit = high >= position.stop_loss
            tp_hit = low <= position.take_profit
            if sl_hit and tp_hit:
                return "SL", position.stop_loss
            if sl_hit:
                return "SL", position.stop_loss
            if tp_hit:
                return "TP", position.take_profit
        return None

    def _pending_should_fill(self, order: PendingOrder, high: float, low: float) -> bool:
        if order.side == "BUY" and order.order_type == "STOP":
            return high >= order.planned_entry
        if order.side == "BUY" and order.order_type == "LIMIT":
            return low <= order.planned_entry
        if order.side == "SELL" and order.order_type == "STOP":
            return low <= order.planned_entry
        if order.side == "SELL" and order.order_type == "LIMIT":
            return high >= order.planned_entry
        return False

    def run(self, df: pd.DataFrame) -> BacktestResult:
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty.")

        bt_cfg = self.bt_config
        balance = float(bt_cfg.starting_equity)
        equity_curve: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []
        cycle_log: List[Dict[str, Any]] = []

        positions: List[Position] = []
        pending_orders: List[PendingOrder] = []

        last_trade_index = -10_000
        last_trade_time: Optional[pd.Timestamp] = None
        trades_today = 0
        current_day: Optional[datetime.date] = None
        trade_id_counter = 1

        for i, row in df.iterrows():
            current_time = pd.Timestamp(row["time"])
            open_ = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            if current_day is None or current_time.date() != current_day:
                current_day = current_time.date()
                trades_today = 0

            bid = close - (bt_cfg.spread / 2)
            ask = close + (bt_cfg.spread / 2)

            expired_orders = 0
            if pending_orders:
                for order in list(pending_orders):
                    if current_time - order.created_time > timedelta(minutes=bt_cfg.pending_expire_minutes):
                        pending_orders.remove(order)
                        expired_orders += 1

            filled_orders = 0
            if pending_orders:
                for order in list(pending_orders):
                    if self._pending_should_fill(order, high, low):
                        filled_orders += 1
                        order.filled_time = current_time
                        order.filled_bar_index = i
                        slippage = bt_cfg.slippage if order.side == "BUY" else -bt_cfg.slippage
                        order.filled_price = order.planned_entry + slippage
                        position = order.to_position()
                        positions.append(position)
                        pending_orders.remove(order)

            closed_positions = 0
            if positions:
                for position in list(positions):
                    exit_info = self._apply_exit_checks(position, high, low)
                    if exit_info:
                        exit_reason, target_price = exit_info
                        slippage = -bt_cfg.slippage if position.side == "BUY" else bt_cfg.slippage
                        exit_price = target_price + slippage
                        raw_pnl = self._calculate_pnl(position.side, position.entry_price, exit_price, position.lot)
                        commission = bt_cfg.commission_per_lot * position.lot * 2
                        pnl = raw_pnl - commission
                        position.close_time = current_time
                        position.close_bar_index = i
                        position.close_price = exit_price
                        position.exit_reason = exit_reason
                        position.pnl_usd = pnl
                        position.duration_bars = i - position.open_bar_index
                        balance += pnl
                        trades.append(asdict(position))
                        positions.remove(position)
                        closed_positions += 1

            unrealized = 0.0
            if positions:
                for position in positions:
                    unrealized += self._calculate_pnl(position.side, position.entry_price, close, position.lot)
            equity = balance + unrealized
            equity_curve.append({"time": current_time, "equity": equity, "balance": balance, "open_pnl": unrealized})

            cycle_payload: Dict[str, Any] = {
                "time": current_time,
                "bar_index": i,
                "price": close,
                "bid": bid,
                "ask": ask,
                "spread": bt_cfg.spread,
                "slippage": bt_cfg.slippage,
                "open_positions": len(positions),
                "pending_orders": len(pending_orders),
                "pending_filled": filled_orders,
                "pending_expired": expired_orders,
                "closed_positions": closed_positions,
                "analyzer_signal": "NONE",
                "final_signal": "NONE",
                "score": 0.0,
                "confidence": 0.0,
                "tier": "NONE",
                "entry_type": "NONE",
                "entry_model": "NONE",
                "planned_entry_model": "NONE",
                "order_type": "NONE",
                "entry_source": "NONE",
                "reject_reason": "-",
                "reject_details": "-",
                "gate_reason": "-",
                "risk_reject_reason": "-",
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "sl_pips": None,
                "tp_pips": None,
                "deviation_pips": None,
                "rr_ratio": None,
                "session": "UNKNOWN",
                "session_weight": 0.0,
                "session_activity": "",
                "is_active_session": True,
                "untradable": False,
                "entry_distance_pips": 0.0,
                "entry_distance_atr": 0.0,
                "zone_rejections": {},
                "retest_rejections": {},
            }

            if i < bt_cfg.warmup_bars:
                cycle_payload["reject_reason"] = "WARMUP"
                cycle_log.append(cycle_payload)
                continue

            min_confidence = float(get_setting(self.config, "technical_settings.SCALPING_MIN_CONFIDENCE", 0.0) or 0.0)
            if 0.0 <= min_confidence <= 1.0:
                min_confidence *= 100.0

            analyzer_window = df.iloc[: i + 1].copy()
            entry_factor = float(get_setting(self.config, "technical_settings.ENTRY_FACTOR", 0.2) or 0.2)

            raw_result = analyze_gold_market(
                dataframe=analyzer_window,
                timeframe=bt_cfg.timeframe,
                entry_factor=entry_factor,
                config=self.analyzer_config,
                scalping_mode=True,
            )
            result = self._normalize_analysis(raw_result)

            analyzer_signal = self._normalize_signal(result.get("signal", "NONE"))
            score = float(result.get("score", 0.0) or 0.0)
            confidence = float(result.get("confidence", 0.0) or 0.0)

            entry_type, entry_model, tier, entry_source, entry_context = self._extract_entry_meta(result)
            session_info = self._extract_session_info(result)
            adx_value = self._extract_adx(result)

            cycle_payload.update(
                {
                    "analyzer_signal": analyzer_signal,
                    "score": score,
                    "confidence": confidence,
                    "tier": tier,
                    "entry_type": entry_type,
                    "entry_model": entry_model,
                    "planned_entry_model": entry_model,
                    "entry_source": entry_source,
                    "session": session_info["session"],
                    "session_weight": session_info["session_weight"],
                    "session_activity": session_info["session_activity"],
                    "is_active_session": session_info["is_active_session"],
                    "untradable": session_info["untradable"],
                    "zone_rejections": entry_context.get("zone_rejections", {}) if isinstance(entry_context, dict) else {},
                    "retest_rejections": entry_context.get("retest_rejections", {}) if isinstance(entry_context, dict) else {},
                    "entry_distance_atr": entry_context.get("dist_atr") if isinstance(entry_context, dict) else 0.0,
                }
            )

            final_signal = analyzer_signal
            gate_reason = "-"
            reject_details = "-"

            enable_auto_trading = bool(get_setting(self.config, "trading_settings.ENABLE_AUTO_TRADING", True))
            if analyzer_signal not in ("BUY", "SELL"):
                final_signal = "NONE"
                gate_reason = "ANALYZER_NONE"
            elif confidence < min_confidence:
                final_signal = "NONE"
                gate_reason = "CONF_TOO_LOW"
                reject_details = f"{confidence:.1f} < {min_confidence:.1f}"
            elif session_info["untradable"]:
                final_signal = "NONE"
                gate_reason = "UNTRADABLE"
                reject_details = session_info["untradable_reasons"]
            elif not enable_auto_trading:
                final_signal = "NONE"
                gate_reason = "AUTO_TRADING_OFF"

            if final_signal in ("BUY", "SELL"):
                if not bt_cfg.allow_multiple_positions and positions:
                    final_signal = "NONE"
                    gate_reason = "SINGLE_POSITION_ONLY"
                elif len(positions) >= bt_cfg.max_positions:
                    final_signal = "NONE"
                    gate_reason = "MAX_POSITIONS"
                elif trades_today >= bt_cfg.daily_max_trades:
                    final_signal = "NONE"
                    gate_reason = "DAILY_MAX_TRADES"
                elif i - last_trade_index < bt_cfg.min_candles_between_trades:
                    final_signal = "NONE"
                    gate_reason = "MIN_CANDLES_BETWEEN"
                elif last_trade_time is not None:
                    delta_minutes = (current_time - last_trade_time).total_seconds() / 60.0
                    if delta_minutes < bt_cfg.min_time_between_trades_minutes:
                        final_signal = "NONE"
                        gate_reason = "MIN_TIME_BETWEEN"

            cycle_payload["final_signal"] = final_signal
            cycle_payload["gate_reason"] = gate_reason
            cycle_payload["reject_details"] = reject_details
            cycle_payload["reject_reason"] = gate_reason if gate_reason != "-" else "-"

            if final_signal not in ("BUY", "SELL"):
                cycle_log.append(cycle_payload)
                continue

            self.risk_manager.last_signal_confidence = float(confidence or 0.0)
            self.risk_manager.last_adx = float(adx_value or 0.0)
            self.risk_manager.last_session = str(session_info["session"] or "UNKNOWN")

            live_snapshot = LivePriceSnapshot(bid=bid, ask=ask, timestamp=str(current_time))
            finalized = self.risk_manager.finalize_order(
                analysis=result,
                live=live_snapshot,
                symbol=bt_cfg.symbol,
                config=self.config,
            )

            cycle_payload.update(
                {
                    "entry_price": finalized.entry_price,
                    "stop_loss": finalized.stop_loss,
                    "take_profit": finalized.take_profit,
                    "deviation_pips": finalized.deviation_pips,
                    "rr_ratio": finalized.rr_ratio,
                    "order_type": finalized.order_type,
                }
            )
            cycle_payload["entry_distance_pips"] = self._entry_distance_pips(finalized.entry_price, close)

            if not finalized.is_trade_allowed:
                cycle_payload["risk_reject_reason"] = finalized.reject_reason
                cycle_payload["reject_reason"] = finalized.reject_reason
                cycle_log.append(cycle_payload)
                continue

            sl_metrics = calculate_distance_metrics(
                entry_price=finalized.entry_price,
                current_price=finalized.stop_loss,
                point_size=self.point_size,
            )
            tp_metrics = calculate_distance_metrics(
                entry_price=finalized.entry_price,
                current_price=finalized.take_profit,
                point_size=self.point_size,
            )
            cycle_payload["sl_pips"] = float(sl_metrics.get("dist_pips") or 0.0)
            cycle_payload["tp_pips"] = float(tp_metrics.get("dist_pips") or 0.0)

            order_type = finalized.order_type.upper()
            cycle_payload["entry_model"] = order_type
            position_id = f"T{trade_id_counter:05d}"
            trade_id_counter += 1

            if order_type == "MARKET":
                slippage = bt_cfg.slippage if final_signal == "BUY" else -bt_cfg.slippage
                entry_price = finalized.entry_price + slippage
                position = Position(
                    id=position_id,
                    side=final_signal,
                    symbol=bt_cfg.symbol,
                    open_time=current_time,
                    open_bar_index=i,
                    entry_price=entry_price,
                    stop_loss=finalized.stop_loss,
                    take_profit=finalized.take_profit,
                    lot=finalized.lot_size,
                    order_type=order_type,
                    confidence=confidence,
                    score=score,
                    rr=finalized.rr_ratio,
                    session=session_info["session"],
                    planned_entry=finalized.entry_price,
                    deviation_pips=finalized.deviation_pips,
                    notes=list(finalized.decision_notes),
                )
                positions.append(position)
            else:
                if order_type == "LIMIT" and not bt_cfg.enable_limit_orders:
                    cycle_payload["risk_reject_reason"] = "LIMIT_DISABLED"
                    cycle_payload["reject_reason"] = "LIMIT_DISABLED"
                    cycle_log.append(cycle_payload)
                    continue
                pending_orders.append(
                    PendingOrder(
                        id=position_id,
                        side=final_signal,
                        symbol=bt_cfg.symbol,
                        created_time=current_time,
                        created_bar_index=i,
                        planned_entry=finalized.entry_price,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        lot=finalized.lot_size,
                        order_type=order_type,
                        confidence=confidence,
                        score=score,
                        rr=finalized.rr_ratio,
                        session=session_info["session"],
                        deviation_pips=finalized.deviation_pips,
                        notes=list(finalized.decision_notes),
                    )
                )

            trades_today += 1
            last_trade_index = i
            last_trade_time = current_time
            cycle_log.append(cycle_payload)

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        cycle_df = pd.DataFrame(cycle_log)

        if not equity_df.empty:
            equity_df = equity_df.set_index("time")
        if not cycle_df.empty:
            cycle_df = cycle_df.set_index("time")

        metrics = compute_trade_metrics(trades_df, equity_df, bt_cfg.starting_equity, bt_cfg.bars_per_day)
        diagnostics = summarize_cycle_log(cycle_df)
        summary_text = format_summary_text(metrics, diagnostics)

        return BacktestResult(
            config=self.config,
            overrides=self.overrides,
            trades=trades_df,
            equity_curve=equity_df,
            cycle_log=cycle_df,
            metrics=metrics,
            diagnostics=asdict(diagnostics),
            summary_text=summary_text,
        )
