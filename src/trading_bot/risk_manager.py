"""
مدیریت ریسک اسکلپینگ حرفه‌ای برای طلا (XAUUSD) - نسخه اسکلپینگ
بهینه‌شده برای معاملات M1-M5 با استراتژی اسکلپینگ NDS
نسخه یکپارچه با bot_config.json
"""

import logging
from typing import Dict, Optional, Any, Tuple, List, TYPE_CHECKING, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import math

from config.settings import config

from src.trading_bot.nds.distance_utils import (
    calculate_distance_metrics,
    DEFAULT_POINT_SIZE,
    pips_to_price,
    price_to_points,
    resolve_point_size_from_config,
    resolve_point_size_with_source,
)
from src.trading_bot.config_utils import get_setting
from src.trading_bot.time_utils import (
    get_broker_now,
    parse_timestamp,
    to_broker_time,
)
from src.trading_bot.session_policy import SessionDecision, evaluate_session, session_weight_from_config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.trading_bot.nds.models import AnalysisResult, FinalizedOrderParams, LivePriceSnapshot

from src.trading_bot.nds.models import FinalizedOrderParams, LivePriceSnapshot


@dataclass
class ScalpingRiskParameters:
    """پارامترهای ریسک محاسبه‌شده برای اسکلپینگ"""
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
    scalping_specific: Dict[str, Any]  # پارامترهای خاص اسکلپینگ

    def __str__(self):
        return (f"Lot: {self.lot_size:.3f}, "
                f"Risk: ${self.risk_amount:.2f} ({self.actual_risk_percent:.3f}%), "
                f"SL Distance: {self.scalping_specific.get('sl_distance', 0):.2f}$, "
                f"Valid: {self.validation_passed}")


class ScalpingRiskManager:
    """
    مدیر ریسک اسکلپینگ حرفه‌ای برای معاملات طلا
    با پشتیبانی کامل از اسکلپینگ با تایم‌فریم کوتاه
    """

    GOLD_SPECS = {}
    VERSION_TAG = "2025-02-14-stop-far-rr-repair"

    # ==================== تنظیمات پیش‌فرض اسکلپینگ ====================

    @property
    def DEFAULT_SCALPING_CONFIG(self):
        """تنظیمات اسکلپینگ مبتنی بر bot_config.json"""
        if hasattr(self, 'settings'):
            return self.settings.copy()

        full_config = config.get_full_config()
        return self._merge_with_config(full_config, {})

    @staticmethod
    def _normalize_gold_specs(gold_specs: Dict[str, Any]) -> Dict[str, Any]:
        if not gold_specs:
            return {}

        mapping = {
            'TICK_VALUE_PER_LOT': 'tick_value_per_lot',
            'POINT': 'point',
            'MIN_LOT': 'min_lot',
            'MAX_LOT': 'max_lot',
            'LOT_STEP': 'lot_step',
            'CONTRACT_SIZE': 'contract_size',
            'DIGITS': 'digits',
        }

        normalized = dict(gold_specs)
        for upper_key, lower_key in mapping.items():
            if lower_key in gold_specs:
                normalized[lower_key] = gold_specs[lower_key]
            elif upper_key in gold_specs:
                normalized[lower_key] = gold_specs[upper_key]

        return normalized

    # ==================== NEW: SAFE ACCESSOR / DEFAULTS ====================

    def _get_gold_spec(self, key: str, default: Any = None) -> Any:
        """
        Safe access for GOLD_SPECS normalized keys.
        """
        try:
            if isinstance(self.GOLD_SPECS, dict) and key in self.GOLD_SPECS:
                v = self.GOLD_SPECS.get(key)
                if v is not None:
                    return v
        except Exception:
            pass
        self._logger.warning("⚠️ GOLD_SPECS missing '%s' -> using default=%s", key, default)
        return default

    def _ensure_gold_specs(self) -> None:
        """
        Enforce presence of critical specs to prevent runtime KeyError.
        Defaults are conservative; prefer providing correct values in config.
        """
        if not isinstance(self.GOLD_SPECS, dict):
            self.GOLD_SPECS = {}

        # Conservative defaults (should be overridden by broker-specific config)
        self.GOLD_SPECS.setdefault("point", DEFAULT_POINT_SIZE)
        self.GOLD_SPECS.setdefault("digits", 2)
        self.GOLD_SPECS.setdefault("contract_size", 100.0)
        self.GOLD_SPECS.setdefault("tick_value_per_lot", 1.0)
        self.GOLD_SPECS.setdefault("min_lot", 0.01)
        self.GOLD_SPECS.setdefault("max_lot", 50.0)
        self.GOLD_SPECS.setdefault("lot_step", 0.01)

    def __init__(self, overrides: Optional[Dict[str, Any]] = None, logger: logging.Logger = None):
        """
        مقداردهی مدیر ریسک اسکلپینگ با ساختار یکپارچه و حرفه‌ای.

        Args:
            overrides: دیکشنری تنظیمات سفارشی برای بازنویسی مقادیر پایه
            logger: آبجکت لاگر برای ثبت وقایع
        """
        full_config = config.get_full_config()
        if overrides is not None:
            if not isinstance(overrides, dict):
                raise TypeError("overrides must be a dict when provided.")
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(full_config.get(key), dict):
                    full_config[key].update(value)
                else:
                    full_config[key] = value
        merged_config = self._merge_with_config(full_config, {})

        # ۲. مقداردهی لاگر
        self._logger = logger or logging.getLogger(__name__)
        self._logger.info("[RISK][VERSION] %s", self.VERSION_TAG)

        self._logger.info("🔄 Single Source of Truth loaded for RiskManager (ConfigManager + overrides).")

        # ۴. ذخیره تنظیمات نهایی در self.settings (منبع واحد حقیقت)
        self.settings = merged_config

        # جهت سازگاری با کدهای قدیمی که ممکن است از self.config استفاده کنند
        self.config = self.settings

        trading_settings = full_config.get('trading_settings', {})
        self.GOLD_SPECS = self._normalize_gold_specs(trading_settings.get('GOLD_SPECIFICATIONS', {}))
        self._ensure_gold_specs()  # ✅ جلوگیری از KeyError در محاسبات

        # ۵. وضعیت ردیابی ریسک اسکلپینگ (بدون تغییر)
        self.daily_risk_used = 0.0
        self.daily_profit_loss = 0.0
        self.active_positions = 0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.scalping_positions = []  # لیست پوزیشن‌های اسکلپینگ فعال


        self.last_signal_confidence = 0.0
        self.last_adx = 0.0
        self.last_session = "UNKNOWN"
        self._unknown_session_logged = False



        # ۶. آمار اسکلپینگ (بدون تغییر)
        self.scalping_stats = {
            'total_scalps': 0,
            'winning_scalps': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_duration': 0.0,
            'best_scalp': 0.0,
            'worst_scalp': 0.0,
        }

        self.last_update = datetime.now()

        # ۷. لاگ‌های نهایی برای تأیید صحت بارگذاری
        self._logger.info("✅ Scalping Risk Manager Initialized - Gold Scalping Optimized")
        self._logger.info(f"📊 Total parameters loaded: {len(self.settings)}")

        # نمایش مقادیر کلیدی در لاگ برای اطمینان از صحت Merge
        # استفاده از نام‌های داخلی که در Mapping تعریف کردیم
        min_conf = self.settings.get('SCALPING_MIN_CONFIDENCE', 'N/A')
        max_sl = self.settings.get('MAX_SL_DISTANCE', 'N/A')
        risk_usd = self.settings.get('SCALPING_RISK_USD', 'N/A')

        self._logger.info(f"📝 Key settings: Conf={min_conf}%, MaxSL={max_sl}$, Risk={risk_usd}$")

        # لاگ مشخصات Gold Specs برای عیب‌یابی
        self._logger.info(
            "🧩 GOLD_SPECS: point=%s digits=%s min_lot=%s max_lot=%s lot_step=%s contract_size=%s tick_value_per_lot=%s",
            self._get_gold_spec("POINT"),
            self._get_gold_spec("DIGITS"),
            self._get_gold_spec("MIN_LOT"),
            self._get_gold_spec("MAX_LOT"),
            self._get_gold_spec("LOT_STEP"),
            self._get_gold_spec("CONTRACT_SIZE"),
            self._get_gold_spec("TICK_VALUE_PER_LOT"),
        )

    def _merge_with_config(self, config: Dict, merged_config: Dict) -> Dict:
        """
        نسخه حرفه‌ای و یکپارچه ادغام تنظیمات با استفاده از Mapping داینامیک.
        مطابق با ساختار bot_config.json نسخه جدید.
        """

        # ۱. تعریف نگاشت (Mapping): {نام در فایل JSON : نام داخلی در RiskManager}
        mapping = {
            'risk_settings': {
                'MIN_RISK_DOLLARS': 'MIN_RISK_DOLLARS',
                'MIN_RISK_REWARD': 'MIN_RISK_REWARD',
                'MAX_RISK_REWARD': 'MAX_RISK_REWARD',
                'DEFAULT_RISK_REWARD': 'DEFAULT_RISK_REWARD',
                'RISK_AMOUNT_USD': 'SCALPING_RISK_USD',
                'MIN_CONFIDENCE': 'MIN_CONFIDENCE',
                'MAX_PRICE_DEVIATION_PIPS': 'MAX_PRICE_DEVIATION_PIPS',
                'MAX_ENTRY_ATR_DEVIATION': 'MAX_ENTRY_ATR_DEVIATION',
                'LIMIT_ORDER_MIN_CONFIDENCE': 'LIMIT_ORDER_MIN_CONFIDENCE',
                'SCALP_ATR_SL_MULT': 'SCALP_ATR_SL_MULT',
                'SL_MIN_PIPS': 'SL_MIN_PIPS',
                'SL_MAX_PIPS': 'SL_MAX_PIPS',
                'TP1_PIPS': 'TP1_PIPS',
                'TP2_ENABLED': 'TP2_ENABLED',
                'TP2_PIPS': 'TP2_PIPS',
                'SPREAD_MAX_PIPS': 'SPREAD_MAX_PIPS',
            },
            'technical_settings': {
                'ATR_WINDOW': 'ATR_WINDOW',
                'SWING_PERIOD': 'SWING_PERIOD',
                'ADX_WINDOW': 'ADX_WINDOW',
                'FVG_MIN_SIZE_MULTIPLIER': 'FVG_MIN_SIZE_MULTIPLIER',
                'MIN_ATR_DISTANCE_MULTIPLIER': 'MIN_ATR_DISTANCE_MULTIPLIER',
                'ENTRY_FACTOR': 'ENTRY_FACTOR',
                'FIXED_BUFFER': 'FIXED_BUFFER',
                'RANGE_TOLERANCE': 'RANGE_TOLERANCE',
                'MAX_SL_DISTANCE': 'MAX_SL_DISTANCE',
                'MIN_SL_DISTANCE': 'MIN_SL_DISTANCE',
                'SCALPING_MIN_CONFIDENCE': 'SCALPING_MIN_CONFIDENCE',
                'SCALPING_MAX_BARS_BACK': 'SCALPING_MAX_BARS_BACK',
                'SCALPING_MAX_DISTANCE_ATR': 'SCALPING_MAX_DISTANCE_ATR',
                'SCALPING_MIN_FVG_SIZE_ATR': 'SCALPING_MIN_FVG_SIZE_ATR',
                'MIN_RVOL_SCALPING': 'RVOL_THRESHOLD',  # تطبیق با نام RVOL_THRESHOLD در ثابت‌ها
                'ATR_SL_MULTIPLIER': 'ATR_SL_MULTIPLIER'
            },
            'risk_manager_config': {
                'MAX_RISK_PERCENT': 'MAX_RISK_PERCENT',
                'MIN_RISK_PERCENT': 'MIN_RISK_PERCENT',
                'MAX_DAILY_RISK_PERCENT': 'MAX_DAILY_RISK_PERCENT',
                'MAX_POSITIONS': 'MAX_POSITIONS',
                'MAX_DAILY_TRADES': 'MAX_DAILY_TRADES',
                'HIGH_CONFIDENCE': 'HIGH_CONFIDENCE',
                'MIN_RR_RATIO': 'MIN_RR_RATIO',
                'TARGET_RR_RATIO': 'TARGET_RR_RATIO',
                'MAX_LEVERAGE': 'MAX_LEVERAGE',
                'MAX_LOT': 'MAX_LOT_SIZE',  # مپ کردن MAX_LOT به نام مورد استفاده در محاسبات لات
                'MAX_LOT_SIZE': 'MAX_LOT_SIZE',
                'POSITION_TIMEOUT_MINUTES': 'POSITION_TIMEOUT_MINUTES'
            },
            'flow_settings': {
                'FLOW_TP1_MIN_RR': 'FLOW_TP1_MIN_RR',
                'FLOW_TP1_USE_OPPOSING_STRUCTURE': 'FLOW_TP1_USE_OPPOSING_STRUCTURE',
                'FLOW_TP1_PARTIAL_CLOSE_PCT': 'FLOW_TP1_PARTIAL_CLOSE_PCT',
                'FLOW_TP1_MOVE_SL_TO_BE': 'FLOW_TP1_MOVE_SL_TO_BE',
                'FLOW_TRAIL_ATR_MULT': 'FLOW_TRAIL_ATR_MULT',
                'FLOW_TRAIL_AFTER_TP1': 'FLOW_TRAIL_AFTER_TP1',
            }
        }

        # ۲. چرخه ادغام هوشمند (Smart Merge)
        for section_name, fields in mapping.items():
            # بررسی وجود بخش (مثلاً risk_settings) در کانفیگ ورودی
            if section_name in config:
                config_section = config[section_name]
                for json_key, internal_key in fields.items():
                    # اگر کلید در کانفیگ بود، مقدار را جایگزین کن
                    if json_key in config_section:
                        merged_config[internal_key] = config_section[json_key]

        # ۳. مدیریت بخش سشن‌ها (به دلیل ساختار دیکشنری تودرتو)
        if 'sessions_config' in config:
            s_config = config['sessions_config']

            # ادغام ضرایب اسکلپینگ سشن‌ها
            if 'SCALPING_SESSION_ADJUSTMENT' in s_config:
                merged_config['SCALPING_SESSION_MULTIPLIERS'] = s_config['SCALPING_SESSION_ADJUSTMENT']

            if 'SCALPING_HOLDING_TIMES' in s_config:
                merged_config['SCALPING_HOLDING_TIMES'] = s_config['SCALPING_HOLDING_TIMES']

            # ادغام حداقل وزن سشن
            if 'MIN_SESSION_WEIGHT' in s_config:
                merged_config['MIN_SESSION_WEIGHT'] = s_config['MIN_SESSION_WEIGHT']

        return merged_config

    # ==================== متدهای کمکی سشن‌ها ====================
    def get_current_scalping_session(
        self,
        dt: datetime = None,
        time_mode: Optional[str] = None,
        broker_utc_offset_hours: Optional[float] = None,
    ) -> str:
        """
        Detect current scalping session based on broker trading time.
        """
        resolved_mode = str(
            time_mode
            or get_setting(config, "trading_settings.TIME_MODE", "BROKER")
            or "BROKER"
        ).upper()
        resolved_offset = float(
            broker_utc_offset_hours
            if broker_utc_offset_hours is not None
            else get_setting(config, "trading_settings.BROKER_UTC_OFFSET_HOURS", 2) or 2
        )
        raw_dt = dt
        if dt is None:
            dt = get_broker_now(resolved_offset)
        else:
            parsed = parse_timestamp(dt) or dt
            dt = to_broker_time(parsed, resolved_offset, resolved_mode)

        decision = evaluate_session(dt, self.config)
        session = decision.session_name
        if session == "UNKNOWN" and not self._unknown_session_logged:
            self._unknown_session_logged = True
            self._logger.warning(
                "[RISK][SESSION][UNKNOWN] dt_raw=%s dt_broker=%s mode=%s offset=%.2f decision=%s",
                raw_dt,
                dt,
                resolved_mode,
                resolved_offset,
                decision.to_payload(),
            )
        return session

    def get_scalping_multiplier(self, session: str) -> float:
        """
        دریافت ضریب ریسک برای اسکلپینگ از منبع واحد تنظیمات.
        """
        # اولویت با تنظیمات داینامیک در self.settings است که در Init لود شده
        multipliers = self.settings.get('SCALPING_SESSION_MULTIPLIERS', {})

        # مقادیر از 0.1 (Dead Zone) تا 1.0 (Overlap) متغیر هستند
        if session in multipliers:
            multiplier = multipliers.get(session)
        else:
            fallback = session_weight_from_config(str(session).upper(), self.config)
            multiplier = float(fallback) if fallback is not None else 0.1

        self._logger.debug(f"🔍 Scalping Session Multiplier for {session}: {multiplier}")
        return multiplier

    def _resolve_session_decision(
        self,
        signal_data: Optional[Dict[str, Any]],
    ) -> Tuple[SessionDecision, str]:
        if isinstance(signal_data, dict):
            payload_decision = signal_data.get("session_decision")
            if not payload_decision and isinstance(signal_data.get("session"), dict):
                payload_decision = signal_data.get("session")
            session_analysis = (
                signal_data.get("session_analysis")
                if isinstance(signal_data.get("session_analysis"), dict)
                else {}
            )
            if not payload_decision and isinstance(session_analysis.get("session_decision"), dict):
                payload_decision = session_analysis.get("session_decision")
            if isinstance(payload_decision, dict):
                return SessionDecision.from_payload(payload_decision), "payload"

        ts_broker = None
        time_mode = None
        offset = None
        if isinstance(signal_data, dict):
            ts_broker = signal_data.get("ts_broker")
            time_mode = signal_data.get("time_mode")
            offset = signal_data.get("broker_utc_offset_hours")
            session_analysis = (
                signal_data.get("session_analysis")
                if isinstance(signal_data.get("session_analysis"), dict)
                else {}
            )
            if not ts_broker:
                ts_broker = session_analysis.get("ts_broker")
            if not time_mode:
                time_mode = session_analysis.get("time_mode")
            if offset is None:
                offset = session_analysis.get("broker_utc_offset_hours")

        resolved_ts = ts_broker or get_broker_now()
        return evaluate_session(resolved_ts, self.config), "computed"

    def _resolve_session_from_signal(
        self,
        signal_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        if not isinstance(signal_data, dict):
            return self.get_current_scalping_session(), "fallback"

        decision_payload = signal_data.get("session_decision")
        if isinstance(decision_payload, dict):
            try:
                decision = SessionDecision.from_payload(decision_payload)
                return decision.session_name, "payload"
            except Exception:
                pass

        session = signal_data.get("session")
        if isinstance(session, dict):
            session_name = session.get("session_name") or session.get("name")
            if session_name:
                return str(session_name).upper(), "payload"
        session_analysis = (
            signal_data.get("session_analysis")
            if isinstance(signal_data.get("session_analysis"), dict)
            else {}
        )
        if not session:
            session = session_analysis.get("current_session")
        if not session:
            analysis_trace = (
                signal_data.get("analysis_trace")
                if isinstance(signal_data.get("analysis_trace"), dict)
                else {}
            )
            market_state = (
                analysis_trace.get("market_state")
                if isinstance(analysis_trace.get("market_state"), dict)
                else {}
            )
            session = market_state.get("session")
        if session:
            return str(session).upper(), "payload"

        ts_broker = signal_data.get("ts_broker") or session_analysis.get("ts_broker")
        time_mode = signal_data.get("time_mode") or session_analysis.get("time_mode")
        offset = signal_data.get("broker_utc_offset_hours")
        if offset is None:
            offset = session_analysis.get("broker_utc_offset_hours")
        return (
            self.get_current_scalping_session(
                dt=ts_broker,
                time_mode=time_mode,
                broker_utc_offset_hours=offset,
            ),
            "fallback",
        )

    def _resolve_adx_from_signal(
        self,
        signal_data: Optional[Dict[str, Any]],
    ) -> Tuple[float, str]:
        if not isinstance(signal_data, dict):
            return 0.0, "missing"

        def _get_nested(payload: Dict[str, Any], path: str) -> Any:
            current: Any = payload
            for part in path.split("."):
                if not isinstance(current, dict):
                    return None
                current = current.get(part)
            return current

        candidates = [
            ("payload", signal_data.get("adx")),
            ("payload", signal_data.get("ADX")),
            ("payload", signal_data.get("adx_value")),
            ("payload", _get_nested(signal_data, "market_metrics.adx")),
            ("legacy", _get_nested(signal_data, "indicators.adx")),
            ("legacy", _get_nested(signal_data, "indicators.adx_value")),
            ("legacy", _get_nested(signal_data, "indicators.adx_analysis.adx")),
            ("legacy", _get_nested(signal_data, "analysis_trace.indicators.adx")),
            ("legacy", _get_nested(signal_data, "analysis_trace.market_state.adx")),
            ("legacy", _get_nested(signal_data, "context.market_metrics.adx")),
        ]

        for source, value in candidates:
            if value is None:
                continue
            try:
                adx_val = float(value or 0.0)
            except Exception:
                continue
            if adx_val > 0:
                return adx_val, source
        return 0.0, "missing"

    def _resolve_confidence_from_signal(self, signal_data: Optional[Dict[str, Any]]) -> float:
        if not isinstance(signal_data, dict):
            return float(getattr(self, "last_signal_confidence", 0.0) or 0.0)
        for key in ("confidence", "conf", "confidence_pct"):
            if signal_data.get(key) is None:
                continue
            try:
                return float(signal_data.get(key) or 0.0)
            except Exception:
                continue
        return float(getattr(self, "last_signal_confidence", 0.0) or 0.0)

    def get_max_holding_time(self, session: str) -> int:
        """دریافت حداکثر زمان نگهداری بر اساس سشن (دقیقه)."""
        holding_configs = self.settings.get('SCALPING_HOLDING_TIMES', {})

        # بازگشت مقدار (پیش‌فرض 60 دقیقه اگر سشن یافت نشد)
        # نکته: مقادیر باید از bot_config.json تامین شوند
        return holding_configs.get(session, 60)

    # ==================== متدهای اصلی ====================

    def calculate_scalping_position_size(self,
                                         account_equity: float,
                                         entry_price: float,
                                         stop_loss: float,
                                         take_profit: float,
                                         signal_confidence: float,
                                         atr_value: float = None,
                                         market_volatility: float = 1.0,
                                         session: str = None,
                                         max_risk_usd: float = None) -> 'ScalpingRiskParameters':
        """
        محاسبه حجم معامله اسکلپینگ با پارامترهای بهینه شده و تنظیمات یکپارچه
        """
        # مقداردهی اولیه
        params = ScalpingRiskParameters(
            lot_size=0.0,
            risk_amount=0.0,
            risk_percent=0.0,
            actual_risk_percent=0.0,
            position_value=0.0,
            margin_required=0.0,
            leverage_used=0.0,
            validation_passed=False,
            warnings=[],
            notes=[],
            calculation_details={},
            scalping_specific={}
        )

        # دسترسی به تنظیمات یکپارچه شده
        s = self.settings

        # 1. اعتبارسنجی اولیه برای اسکلپینگ
        if not self._validate_scalping_parameters(entry_price, stop_loss, take_profit,
                                                  signal_confidence, atr_value, params):
            return params

        # 2. محاسبه فاصله استاپ و اعتبارسنجی با ATR
        sl_distance = abs(entry_price - stop_loss)
        atr_multiplier = s.get('ATR_SL_MULTIPLIER', 1.5)

        if atr_value:
            # تطبیق استاپ با ATR
            optimal_sl_distance = atr_value * atr_multiplier
            if sl_distance > optimal_sl_distance * 1.5:
                params.warnings.append(
                    f"SL distance ({sl_distance:.2f}$) > 1.5x optimal ATR-based SL ({optimal_sl_distance:.2f}$)"
                )

        # 3. تعیین حداکثر ریسک دلاری برای اسکلپینگ
        if max_risk_usd is None:
            max_risk_usd = self._get_max_scalping_risk_usd(account_equity)

        # 4. محاسبه درصد ریسک بر اساس اعتماد
        base_risk_percent = self._calculate_scalping_risk_percent(signal_confidence, account_equity)

        # 5. تنظیم بر اساس سشن اسکلپینگ
        if session is None:
            session = self.get_current_scalping_session()
        session_multiplier = self.get_scalping_multiplier(session)

        # 6. تنظیم بر اساس نوسان بازار
        volatility_multiplier = self._calculate_scalping_volatility_multiplier(market_volatility)

        # 7. تنظیم بر اساس سابقه اسکلپینگ
        history_multiplier = self._calculate_scalping_history_multiplier()

        # 8. محاسبه ریسک نهایی اسکلپینگ
        final_risk_percent = base_risk_percent * session_multiplier * \
                             volatility_multiplier * history_multiplier

        # محدودیت‌های ریسک اسکلپینگ
        final_risk_percent = self._apply_scalping_risk_limits(final_risk_percent, account_equity, max_risk_usd)

        # 9. محاسبه ریسک دلاری
        risk_amount = min((account_equity * final_risk_percent) / 100, max_risk_usd)

        # 10. محاسبه حجم معامله اسکلپینگ
        lot_size = self._calculate_scalping_lot_size(entry_price, stop_loss, risk_amount, sl_distance)

        # 11. محاسبات مالی
        contract_size = float(self._get_gold_spec('CONTRACT_SIZE', 100.0))
        position_value = lot_size * contract_size * entry_price
        margin_required = self._calculate_scalping_margin(lot_size, entry_price)
        actual_risk = self._calculate_actual_scalping_risk(lot_size, entry_price, stop_loss)
        actual_risk_percent = (actual_risk / account_equity) * 100 if account_equity > 0 else 0.0

        # 12. محاسبه RR
        rr_ratio = abs(take_profit - entry_price) / sl_distance if sl_distance > 0 else 0

        # 13. پر کردن پارامترهای اسکلپینگ
        params.lot_size = lot_size
        params.risk_amount = risk_amount
        params.risk_percent = final_risk_percent
        params.actual_risk_percent = actual_risk_percent
        params.position_value = position_value
        params.margin_required = margin_required
        params.leverage_used = position_value / account_equity if account_equity > 0 else 0.0
        params.validation_passed = True

        # 14. اطلاعات خاص اسکلپینگ
        max_holding = self.get_max_holding_time(session)
        params.scalping_specific = {
            'sl_distance': sl_distance,
            'rr_ratio': rr_ratio,
            'session': session,
            'max_holding_minutes': max_holding,
            'optimal_exit_time': (datetime.now() + timedelta(minutes=max_holding * 0.7)).isoformat(),
            'atr_based': atr_value is not None,
            'atr_value': atr_value,
            'position_id': f"SCLP_{int(datetime.now().timestamp())}",
            'scalping_grade': self._calculate_scalping_grade(rr_ratio, sl_distance, signal_confidence)
        }

        # 15. جزئیات محاسبات
        params.calculation_details = {
            'base_risk_percent': base_risk_percent,
            'session_multiplier': session_multiplier,
            'volatility_multiplier': volatility_multiplier,
            'history_multiplier': history_multiplier,
            'final_risk_usd': risk_amount,
            'max_allowed_risk_usd': max_risk_usd,
            'stop_distance': sl_distance,
            'risk_reward_ratio': rr_ratio,
            'account_equity': account_equity,
            'timestamp': datetime.now().isoformat(),
            'scalping_mode': True
        }

        self._logger.info(f"📊 Scalping position calculated: {params}")
        return params

    def _normalize_analysis_payload(self, analysis: Union['AnalysisResult', Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize AnalysisResult/dataclass payloads to a dict."""
        if analysis is None:
            return {}
        if isinstance(analysis, dict):
            payload = dict(analysis)
        elif hasattr(analysis, "__dataclass_fields__"):
            payload = asdict(analysis)
        elif hasattr(analysis, "__dict__"):
            payload = dict(analysis.__dict__)
        else:
            return {}

        if payload.get("entry_level") is None and payload.get("entry_price") is not None:
            payload["entry_level"] = payload.get("entry_price")
        if payload.get("entry_price") is None and payload.get("entry_level") is not None:
            payload["entry_price"] = payload.get("entry_level")
        return payload

    def _resolve_signal_context(
        self,
        analysis_payload: Dict[str, Any],
        analysis_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve signal context (bias/reversal/trend) from analyzer payloads."""
        signal_context = analysis_payload.get("analysis_signal_context")
        if isinstance(signal_context, dict):
            return signal_context
        context_signal = (analysis_payload.get("context") or {}).get("analysis_signal_context")
        if isinstance(context_signal, dict):
            return context_signal
        context_signal = analysis_context.get("analysis_signal_context")
        if isinstance(context_signal, dict):
            return context_signal
        return {}

    def _apply_tp1_target_policy(
        self,
        *,
        signal: str,
        entry_context: Dict[str, Any],
        signal_context: Dict[str, Any],
        decision_notes: List[str],
    ) -> Dict[str, Any]:
        """Apply TP1 policy for counter-trend opposing-structure IFVG targets."""
        tp1_target_price = entry_context.get("tp1_target_price")
        tp1_target_source = entry_context.get("tp1_target_source")
        zone_type = entry_context.get("tp1_target_zone_type")
        zone_direction = entry_context.get("tp1_target_zone_direction")

        if tp1_target_price is None or tp1_target_source is None:
            return {"tp1_target_price": tp1_target_price, "action": "keep"}

        bias = str(signal_context.get("bias", "") or "")
        trend = str(signal_context.get("trend", "") or "")
        reversal_ok = bool(signal_context.get("reversal_ok"))
        counter_trend = (bias == "BULLISH" and signal == "SELL") or (bias == "BEARISH" and signal == "BUY")
        aligned_trend = (
            (trend == "DOWNTREND" and zone_direction == "BEARISH")
            or (trend == "UPTREND" and zone_direction == "BULLISH")
        )

        if counter_trend and not reversal_ok and zone_type == "IFVG" and aligned_trend:
            policy = str(self.settings.get("FLOW_TP1_COUNTERTREND_IFVG_POLICY", "fixed_pips")).lower()
            if policy == "reject":
                decision_notes.append(
                    "TP1 policy: reject counter-trend trade using aligned IFVG opposing structure."
                )
                return {
                    "tp1_target_price": None,
                    "action": "reject",
                    "reject_reason": "Counter-trend IFVG TP blocked without reversal confirmation.",
                }
            decision_notes.append(
                "TP1 policy: counter-trend aligned IFVG -> fixed_pips (reversal_ok=false)."
            )
            return {"tp1_target_price": None, "action": "fixed_pips"}

        return {"tp1_target_price": tp1_target_price, "action": "keep"}

    def _get_point_size(self, config_payload: Dict[str, Any]) -> float:
        """Resolve point size with default for XAUUSD mapping."""
        default_point_size = self._get_gold_spec("point", DEFAULT_POINT_SIZE)
        point_size, _ = resolve_point_size_with_source(config_payload, default=default_point_size)
        return point_size

    def _compute_scalping_sl_tp(
        self,
        signal: str,
        entry_price: float,
        atr_value: Optional[float],
        recent_low: Optional[float],
        recent_high: Optional[float],
        config_payload: Dict[str, Any],
        point_size: Optional[float] = None,
        tp1_target_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute scalping SL/TP with ATR and recent candle extrema reference."""
        settings = self.settings
        if point_size is None:
            point_size = self._get_point_size(config_payload)

        atr_mult = float(settings.get("SCALP_ATR_SL_MULT", 1.5))
        sl_min_pips = float(settings.get("SL_MIN_PIPS", 10.0))
        min_sl_pips = float(settings.get("MIN_SL_PIPS", sl_min_pips))
        sl_min_pips = max(sl_min_pips, min_sl_pips)
        sl_max_pips = float(settings.get("SL_MAX_PIPS", 40.0))
        tp1_pips = float(settings.get("TP1_PIPS", 35.0))
        tp2_enabled = bool(settings.get("TP2_ENABLED", True))
        tp2_pips = float(settings.get("TP2_PIPS", tp1_pips * 2.0))
        tp1_source = "fixed_pips"
        use_opposing = bool(settings.get("FLOW_TP1_USE_OPPOSING_STRUCTURE", True))
        min_rr = float(settings.get("FLOW_TP1_MIN_RR", 1.5))

        atr_value = float(atr_value) if atr_value is not None else 0.0
        atr_distance = atr_value * atr_mult if atr_value > 0 else 0.0

        ref_distance = 0.0
        if signal == "BUY" and recent_low is not None:
            ref_distance = max(0.0, float(entry_price) - float(recent_low))
        elif signal == "SELL" and recent_high is not None:
            ref_distance = max(0.0, float(recent_high) - float(entry_price))

        sl_source = "none"
        if atr_distance > 0 and ref_distance > 0:
            sl_distance = min(atr_distance, ref_distance)
            sl_source = "atr_ref_min"
        elif atr_distance > 0:
            sl_distance = atr_distance
            sl_source = "atr_only"
        else:
            sl_distance = ref_distance
            sl_source = "ref_only"

        if sl_distance <= 0:
            sl_distance = pips_to_price(sl_min_pips, point_size)
            sl_source = "min_pips"

        sl_metrics = calculate_distance_metrics(
            entry_price=float(entry_price),
            current_price=float(entry_price) + float(sl_distance),
            point_size=point_size,
        )
        sl_pips = float(sl_metrics.get("dist_pips") or 0.0)
        raw_sl_pips = sl_pips
        if sl_pips < sl_min_pips:
            sl_pips = sl_min_pips
            sl_distance = pips_to_price(sl_pips, point_size)
        elif sl_pips > sl_max_pips:
            sl_pips = sl_max_pips
            sl_distance = pips_to_price(sl_pips, point_size)

        if raw_sl_pips != sl_pips:
            self._logger.info(
                "[NDS][SL_CLAMP] raw=%.2f clamped=%.2f bounds=[%.2f,%.2f] point_size=%.4f",
                raw_sl_pips,
                sl_pips,
                sl_min_pips,
                sl_max_pips,
                point_size,
            )

        if signal == "BUY":
            stop_loss = float(entry_price) - sl_distance
            take_profit = float(entry_price) + pips_to_price(tp1_pips, point_size)
            if use_opposing and tp1_target_price is not None and tp1_target_price > entry_price:
                min_rr_price = float(entry_price) + (sl_distance * min_rr)
                take_profit = max(float(tp1_target_price), min_rr_price)
                tp1_source = "opposing_structure" if take_profit == float(tp1_target_price) else "min_rr_floor"
            tp2_price = (
                float(entry_price) + pips_to_price(tp2_pips, point_size)
                if tp2_enabled
                else None
            )
        else:
            stop_loss = float(entry_price) + sl_distance
            take_profit = float(entry_price) - pips_to_price(tp1_pips, point_size)
            if use_opposing and tp1_target_price is not None and tp1_target_price < entry_price:
                min_rr_price = float(entry_price) - (sl_distance * min_rr)
                take_profit = min(float(tp1_target_price), min_rr_price)
                tp1_source = "opposing_structure" if take_profit == float(tp1_target_price) else "min_rr_floor"
            tp2_price = (
                float(entry_price) - pips_to_price(tp2_pips, point_size)
                if tp2_enabled
                else None
            )

        tp_metrics = calculate_distance_metrics(
            entry_price=float(entry_price),
            current_price=float(take_profit),
            point_size=point_size,
        )
        tp1_pips = float(tp_metrics.get("dist_pips") or 0.0)

        if bool(settings.get("FLOW_TRAIL_AFTER_TP1", True)):
            tp2_price = None

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "tp1_pips": tp1_pips,
            "tp2_pips": tp2_pips,
            "tp2_price": tp2_price,
            "tp1_source": tp1_source,
            "sl_source": sl_source,
            "sl_pips": sl_pips,
            "raw_sl_pips": raw_sl_pips,
            "sl_distance": sl_distance,
            "atr_distance": atr_distance,
            "ref_distance": ref_distance,
            "point_size": point_size,
        }

    def _distance_sanity_check(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        point_size: float,
        atr_value: Optional[float],
    ) -> Tuple[bool, str]:
        """Validate distance conversions across pips/points/usd/atr."""
        tol_price = float(self.settings.get("DIST_SANITY_TOLERANCE_PRICE", 0.05))
        tol_ratio = float(self.settings.get("DIST_SANITY_TOLERANCE_RATIO", 0.02))

        metrics_sl = calculate_distance_metrics(
            entry_price=entry_price,
            current_price=stop_loss,
            point_size=point_size,
            atr_value=atr_value,
        )
        metrics_tp = calculate_distance_metrics(
            entry_price=entry_price,
            current_price=take_profit,
            point_size=point_size,
            atr_value=atr_value,
        )

        def _validate(metrics: Dict[str, Any], label: str) -> Optional[str]:
            dist_price = float(metrics.get("dist_price") or 0.0)
            dist_points = float(metrics.get("dist_points") or 0.0)
            dist_pips = float(metrics.get("dist_pips") or 0.0)
            dist_atr = metrics.get("dist_atr")

            expected_price = pips_to_price(dist_pips, point_size)
            expected_points = price_to_points(dist_price, point_size)

            if abs(expected_price - dist_price) > tol_price:
                return f"{label}:price_mismatch raw={dist_price:.4f} expected={expected_price:.4f}"
            if abs(expected_points - dist_points) > (dist_points * tol_ratio + 1e-6):
                return f"{label}:points_mismatch raw={dist_points:.4f} expected={expected_points:.4f}"
            if dist_atr is not None and atr_value:
                expected_atr = dist_price / float(atr_value) if float(atr_value) > 0 else None
                if expected_atr is not None and abs(float(dist_atr) - expected_atr) > tol_ratio:
                    return f"{label}:atr_mismatch raw={dist_atr:.4f} expected={expected_atr:.4f}"
            return None

        sl_issue = _validate(metrics_sl, "SL")
        if sl_issue:
            return False, sl_issue
        tp_issue = _validate(metrics_tp, "TP")
        if tp_issue:
            return False, tp_issue
        return True, "ok"

    def _apply_stop_far_from_market_policy(
        self,
        *,
        signal: str,
        order_type: str,
        planned_entry: float,
        market_entry: float,
        deviation_pips: float,
        point_size: float,
        confidence: float,
        analysis_payload: Dict[str, Any],
        analysis_context: Dict[str, Any],
        risk_settings: Dict[str, Any],
        risk_manager_config: Dict[str, Any],
        decision_notes: List[str],
    ) -> Optional[Dict[str, Any]]:
        if order_type != "STOP":
            return None

        stop_soft_pips = float(
            risk_manager_config.get(
                "STOP_MAX_DEVIATION_PIPS",
                risk_settings.get("MAX_PRICE_DEVIATION_PIPS", 70.0),
            )
        )
        stop_hard_pips = float(
            risk_manager_config.get(
                "STOP_HARD_REJECT_PIPS",
                max(stop_soft_pips * 1.5, stop_soft_pips + 10.0),
            )
        )
        limit_pips = float(
            risk_manager_config.get(
                "STOP_CONVERT_TO_LIMIT_PIPS",
                max(5.0, stop_soft_pips * 0.5),
            )
        )
        cap_pips = float(
            risk_manager_config.get(
                "MAX_ENTRY_CAP_PIPS",
                stop_soft_pips,
            )
        )
        trend_adx_min = float(
            risk_manager_config.get("TREND_STRENGTH_ADX_MIN", 25.0)
        )
        mean_rev_adx_max = float(
            risk_manager_config.get("MEAN_REVERSION_ADX_MAX", 18.0)
        )

        if deviation_pips < stop_soft_pips:
            decision_notes.append(
                f"STOP_FAR_POLICY:SKIP deviation_pips={deviation_pips:.1f} soft={stop_soft_pips:.1f}"
            )
            self._logger.info(
                "[NDS][STOP_FAR_POLICY] action=SKIP deviation_pips=%.2f soft=%.2f",
                deviation_pips,
                stop_soft_pips,
            )
            return None

        adx_val, adx_source = self._resolve_adx_from_signal(analysis_payload)
        market_metrics = analysis_payload.get("market_metrics") or analysis_context.get("market_metrics", {})
        volatility_state = str(market_metrics.get("volatility_state") or "").upper()

        decision_notes.append(
            "STOP_FAR_POLICY:deviation_pips="
            f"{deviation_pips:.1f} soft={stop_soft_pips:.1f} hard={stop_hard_pips:.1f} "
            f"adx={adx_val:.1f}({adx_source}) vol={volatility_state or 'NA'}"
        )

        if deviation_pips >= stop_hard_pips:
            decision_notes.append("STOP_FAR_POLICY:REJECT_HARD")
            self._logger.info(
                "[NDS][STOP_FAR_POLICY] action=REJECT_HARD deviation_pips=%.2f hard=%.2f",
                deviation_pips,
                stop_hard_pips,
            )
            return {
                "action": "REJECT_HARD",
                "order_type": "NONE",
                "entry_price": planned_entry,
                "reject_reason": "Stop too far.",
                "deviation_pips": deviation_pips,
            }

        is_trend_continuation = adx_val >= trend_adx_min if adx_val > 0 else False
        is_mean_reversion = (adx_val > 0 and adx_val <= mean_rev_adx_max) or volatility_state == "LOW"

        if is_trend_continuation:
            cap_pips = min(cap_pips, deviation_pips)
            cap_price = pips_to_price(cap_pips, point_size)
            if signal == "BUY":
                capped_entry = min(planned_entry, market_entry + cap_price)
            else:
                capped_entry = max(planned_entry, market_entry - cap_price)
            decision_notes.append(
                f"STOP_FAR_POLICY:CAP_ENTRY capped_entry={capped_entry:.2f} cap_pips={cap_pips:.1f}"
            )
            return {
                "action": "CAP_ENTRY",
                "order_type": "STOP",
                "entry_price": capped_entry,
                "reject_reason": None,
                "deviation_pips": float(
                    calculate_distance_metrics(
                        entry_price=capped_entry,
                        current_price=market_entry,
                        point_size=point_size,
                    ).get("dist_pips")
                    or 0.0
                ),
            }

        if is_mean_reversion:
            limit_min_conf = float(
                risk_settings.get(
                    "LIMIT_ORDER_MIN_CONFIDENCE",
                    self.settings.get("LIMIT_ORDER_MIN_CONFIDENCE", 0.0),
                )
            )
            if confidence < limit_min_conf:
                decision_notes.append(
                    f"STOP_FAR_POLICY:WAIT conf={confidence:.1f} < limit_min_conf={limit_min_conf:.1f}"
                )
                return {
                    "action": "WAIT",
                    "order_type": "WAIT",
                    "entry_price": planned_entry,
                    "reject_reason": "Stop far from market; wait for pullback.",
                    "deviation_pips": deviation_pips,
                }

            limit_price = pips_to_price(limit_pips, point_size)
            if signal == "BUY":
                limit_entry = market_entry - limit_price
            else:
                limit_entry = market_entry + limit_price
            decision_notes.append(
                f"STOP_FAR_POLICY:LIMIT limit_entry={limit_entry:.2f} limit_pips={limit_pips:.1f}"
            )
            return {
                "action": "LIMIT",
                "order_type": "LIMIT",
                "entry_price": limit_entry,
                "reject_reason": None,
                "deviation_pips": float(
                    calculate_distance_metrics(
                        entry_price=limit_entry,
                        current_price=market_entry,
                        point_size=point_size,
                    ).get("dist_pips")
                    or 0.0
                ),
            }

        decision_notes.append("STOP_FAR_POLICY:REJECT_NO_REGIME")
        return {
            "action": "REJECT_NO_REGIME",
            "order_type": "NONE",
            "entry_price": planned_entry,
            "reject_reason": "Stop far from market; no regime match.",
            "deviation_pips": deviation_pips,
        }

    def _attempt_rr_repair(
        self,
        *,
        signal: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        min_rr_ratio: float,
        point_size: float,
        atr_value: Optional[float],
        risk_manager_config: Dict[str, Any],
        decision_notes: List[str],
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """Attempt to repair RR by expanding TP within safe bounds."""
        rr_repair_enabled = bool(risk_manager_config.get("RR_REPAIR_ENABLED", True))
        if not rr_repair_enabled:
            return None, None, None
        rr_epsilon = float(risk_manager_config.get("RR_EPSILON", 1e-6))
        rr_target_buffer = float(
            risk_manager_config.get("RR_REPAIR_TARGET_BUFFER", max(rr_epsilon, 1e-4))
        )

        sl_metrics = calculate_distance_metrics(
            entry_price=entry_price,
            current_price=stop_loss,
            point_size=point_size,
        )
        tp_metrics = calculate_distance_metrics(
            entry_price=entry_price,
            current_price=take_profit,
            point_size=point_size,
        )
        sl_distance = float(sl_metrics.get("dist_price") or 0.0)
        tp_distance = float(tp_metrics.get("dist_price") or 0.0)
        sl_pips = float(sl_metrics.get("dist_pips") or 0.0)
        tp_pips = float(tp_metrics.get("dist_pips") or 0.0)

        rr_before = tp_distance / sl_distance if sl_distance > 0 else 0.0
        if rr_before + rr_epsilon >= min_rr_ratio:
            return None, None, None

        desired_rr = min_rr_ratio + rr_target_buffer
        desired_tp_distance = sl_distance * desired_rr
        max_tp_pips = float(
            risk_manager_config.get(
                "RR_REPAIR_MAX_TP_PIPS",
                max(tp_pips, float(self.settings.get("TP2_PIPS", tp_pips))),
            )
        )
        max_tp_atr_mult = float(risk_manager_config.get("RR_REPAIR_MAX_TP_ATR_MULT", 2.0))
        max_tp_price_atr = None
        if atr_value and float(atr_value) > 0:
            max_tp_price_atr = float(atr_value) * max_tp_atr_mult

        desired_tp_metrics = calculate_distance_metrics(
            entry_price=entry_price,
            current_price=entry_price + desired_tp_distance if signal == "BUY" else entry_price - desired_tp_distance,
            point_size=point_size,
        )
        desired_tp_pips = float(desired_tp_metrics.get("dist_pips") or 0.0)

        cap_reasons = []
        if desired_tp_pips > max_tp_pips:
            cap_reasons.append(
                f"tp_pips_cap {desired_tp_pips:.1f}>{max_tp_pips:.1f}"
            )
        if max_tp_price_atr is not None and desired_tp_distance > max_tp_price_atr:
            cap_reasons.append(
                f"tp_atr_cap {desired_tp_distance:.2f}>{max_tp_price_atr:.2f}"
            )

        decision_notes.append(
            "RR_CHECK sl_pips={sl_pips:.1f} tp_pips={tp_pips:.1f} "
            "rr={rr_before:.6f} min_rr={min_rr_ratio:.2f} target_rr={target_rr:.4f}"
            .format(
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                rr_before=rr_before,
                min_rr_ratio=min_rr_ratio,
                target_rr=desired_rr,
            )
        )

        if cap_reasons:
            decision_notes.append(
                f"RR_REPAIR_REJECT caps_exceeded={'|'.join(cap_reasons)}"
            )
            self._logger.info(
                "[NDS][RR_REPAIR] action=REJECT rr=%.2f min_rr=%.2f sl_pips=%.2f tp_pips=%.2f caps=%s",
                rr_before,
                min_rr_ratio,
                sl_pips,
                tp_pips,
                "|".join(cap_reasons),
            )
            return None, None, "TP cap exceeded for RR repair."

        new_take_profit = (
            entry_price + desired_tp_distance
            if signal == "BUY"
            else entry_price - desired_tp_distance
        )
        new_tp_metrics = calculate_distance_metrics(
            entry_price=entry_price,
            current_price=new_take_profit,
            point_size=point_size,
        )
        new_tp_pips = float(new_tp_metrics.get("dist_pips") or 0.0)
        rr_after = desired_tp_distance / sl_distance if sl_distance > 0 else 0.0

        decision_notes.append(
            "RR_REPAIR_TP "
            f"tp_pips={tp_pips:.1f}->{new_tp_pips:.1f} rr={rr_before:.6f}->{rr_after:.6f} "
            f"max_tp_pips={max_tp_pips:.1f} max_tp_atr_mult={max_tp_atr_mult:.2f}"
        )
        self._logger.info(
            "[NDS][RR_REPAIR] action=TP_ADJUST rr=%.6f->%.6f sl_pips=%.2f tp_pips=%.2f->%.2f target_rr=%.4f",
            rr_before,
            rr_after,
            sl_pips,
            tp_pips,
            new_tp_pips,
            desired_rr,
        )
        return new_take_profit, "rr_repair", None

    def finalize_order(
        self,
        analysis: Union['AnalysisResult', Dict[str, Any]],
        live: Union[LivePriceSnapshot, Dict[str, Any]],
        symbol: str,
        config: Dict[str, Any]
    ) -> FinalizedOrderParams:
        """
        Finalize an order decision using live market snapshot and unified risk settings.
        """
        analysis_payload = self._normalize_analysis_payload(analysis)
        live_payload = live if isinstance(live, dict) else asdict(live)
        point_size, point_source = resolve_point_size_with_source(
            config,
            default=self._get_gold_spec("point", DEFAULT_POINT_SIZE),
        )
        self._logger.info(
            "[NDS][POINT_SIZE] point_size=%.4f source=%s",
            point_size,
            point_source,
        )

        def _finalize(
            *,
            signal: str,
            order_type: str,
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
            take_profit2: Optional[float] = None,
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
                take_profit2=take_profit2,
                tp2=take_profit2,
                final_entry=entry_price,
                final_stop_loss=stop_loss,
                final_take_profit=take_profit,
                final_sl=stop_loss,
                final_tp=take_profit,
                lot=lot_size,
            )

        decision_notes: List[str] = []
        analysis_reasons = analysis_payload.get('reasons') or []
        decision_notes.extend(list(analysis_reasons))
        signal = analysis_payload.get('signal')
        if not signal or signal in ['NONE', 'NEUTRAL']:
            return _finalize(
                signal=signal or 'NONE',
                order_type='NONE',
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["No actionable signal from analyzer."],
                is_trade_allowed=False,
                reject_reason="Signal is NONE/NEUTRAL.",
            )

        planned_entry = analysis_payload.get('entry_level') or analysis_payload.get('entry_price')
        stop_loss = None
        take_profit = None
        confidence = analysis_payload.get('confidence')

        if confidence is None:
            confidence = 0.0

        bid = live_payload.get('bid')
        ask = live_payload.get('ask')
        if bid is None or ask is None:
            return _finalize(
                signal=signal,
                order_type='NONE',
                entry_price=planned_entry or 0.0,
                stop_loss=stop_loss or 0.0,
                take_profit=take_profit or 0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["Live snapshot missing bid/ask."],
                is_trade_allowed=False,
                reject_reason="Live prices unavailable.",
            )

        spread = float(ask) - float(bid)
        spread_metrics = calculate_distance_metrics(
            entry_price=0.0,
            current_price=spread,
            point_size=point_size,
        )
        spread_pips = float(spread_metrics.get("dist_pips") or 0.0)
        spread_max_pips = float(self.settings.get("SPREAD_MAX_PIPS", 2.5))
        if spread_pips > spread_max_pips:
            self._logger.info(
                "[NDS][RISK_GATE] allow=false reason=SPREAD_TOO_HIGH spread_pips=%.2f max=%.2f point_size=%.4f",
                spread_pips,
                spread_max_pips,
                point_size,
            )
            return _finalize(
                signal=signal,
                order_type='NONE',
                entry_price=planned_entry or 0.0,
                stop_loss=stop_loss or 0.0,
                take_profit=take_profit or 0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=[f"Spread too high ({spread_pips:.2f} > {spread_max_pips:.2f})"],
                is_trade_allowed=False,
                reject_reason="Spread too high.",
            )

        analysis_context = analysis_payload.get('context', {}) or {}
        entry_idea = (
            analysis_payload.get("entry_idea")
            or analysis_context.get("entry_idea", {})
            or {}
        )
        entry_model = (
            entry_idea.get("entry_model")
            or analysis_payload.get("entry_model")
            or "MARKET"
        )
        planned_entry = (
            entry_idea.get("entry_level")
            or analysis_payload.get("entry_level")
            or analysis_payload.get("entry_price")
        )
        market_metrics = analysis_payload.get('market_metrics') or analysis_context.get('market_metrics', {})
        atr_value = market_metrics.get('atr_short') or market_metrics.get('atr')

        if planned_entry is None:
            return _finalize(
                signal=signal,
                order_type='NONE',
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["Missing entry idea from analyzer."],
                is_trade_allowed=False,
                reject_reason="Missing entry idea.",
            )

        risk_settings = config.get('risk_settings', {})
        trading_settings = config.get('trading_settings', {})
        risk_manager_config = config.get('risk_manager_config', {})

        max_entry_atr_deviation = risk_settings.get('MAX_ENTRY_ATR_DEVIATION')
        min_rr_ratio = risk_manager_config.get('MIN_RR_RATIO')
        rr_epsilon = float(risk_manager_config.get("RR_EPSILON", 1e-6))

        if min_rr_ratio is None:
            return _finalize(
                signal=signal,
                order_type='NONE',
                entry_price=planned_entry or 0.0,
                stop_loss=stop_loss or 0.0,
                take_profit=take_profit or 0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["Missing risk settings in config."],
                is_trade_allowed=False,
                reject_reason="Risk settings missing from config.",
            )

        market_entry = ask if signal == 'BUY' else bid
        deviation = abs(planned_entry - market_entry)

        deviation_metrics = calculate_distance_metrics(
            entry_price=planned_entry,
            current_price=market_entry,
            point_size=point_size,
            atr_value=None,
        )
        deviation_pips = float(deviation_metrics.get("dist_pips") or 0.0)

        decision_notes.append(
            f"PlannedEntry={planned_entry:.2f} MarketEntry={market_entry:.2f} deviation_pips={deviation_pips:.1f} conf={float(confidence):.1f}"
        )

        entry_model = str(entry_model or "MARKET").upper()
        if entry_model == "STOP":
            order_type = "STOP"
        elif entry_model == "LIMIT":
            order_type = "LIMIT"
        else:
            order_type = "MARKET"
        entry_price = planned_entry if order_type in ("STOP", "LIMIT") else market_entry

        if order_type == "STOP":
            stop_policy = self._apply_stop_far_from_market_policy(
                signal=signal,
                order_type=order_type,
                planned_entry=planned_entry,
                market_entry=market_entry,
                deviation_pips=deviation_pips,
                point_size=point_size,
                confidence=float(confidence or 0.0),
                analysis_payload=analysis_payload,
                analysis_context=analysis_context,
                risk_settings=risk_settings,
                risk_manager_config=risk_manager_config,
                decision_notes=decision_notes,
            )
            if stop_policy:
                order_type = stop_policy.get("order_type", order_type)
                entry_price = stop_policy.get("entry_price", entry_price)
                deviation_pips = float(stop_policy.get("deviation_pips", deviation_pips))
                if stop_policy.get("reject_reason"):
                    return _finalize(
                        signal=signal,
                        order_type=order_type,
                        entry_price=entry_price,
                        stop_loss=stop_loss or 0.0,
                        take_profit=take_profit or 0.0,
                        lot_size=0.0,
                        risk_amount_usd=0.0,
                        rr_ratio=0.0,
                        deviation_pips=deviation_pips,
                        decision_notes=decision_notes,
                        is_trade_allowed=False,
                        reject_reason=stop_policy.get("reject_reason"),
                    )
                deviation = abs(entry_price - market_entry)
                deviation_metrics = calculate_distance_metrics(
                    entry_price=entry_price,
                    current_price=market_entry,
                    point_size=point_size,
                    atr_value=None,
                )
                deviation_pips = float(deviation_metrics.get("dist_pips") or deviation_pips)
                decision_notes.append(
                    f"STOP_FAR_POLICY:adjusted_entry_deviation_pips={deviation_pips:.1f}"
                )

        stop_revalidate_pips = risk_manager_config.get("STOP_REVALIDATE_PIPS")
        if order_type == "STOP" and stop_revalidate_pips is not None:
            stop_revalidate_pips = float(stop_revalidate_pips)
            if deviation_pips > stop_revalidate_pips:
                decision_notes.append(
                    f"STOP_REVALIDATE:deviation_pips={deviation_pips:.1f} threshold={stop_revalidate_pips:.1f}"
                )
                self._logger.info(
                    "[NDS][STOP_REVALIDATE] action=REJECT deviation_pips=%.2f threshold=%.2f",
                    deviation_pips,
                    stop_revalidate_pips,
                )
                return _finalize(
                    signal=signal,
                    order_type="NONE",
                    entry_price=entry_price,
                    stop_loss=stop_loss or 0.0,
                    take_profit=take_profit or 0.0,
                    lot_size=0.0,
                    risk_amount_usd=0.0,
                    rr_ratio=0.0,
                    deviation_pips=deviation_pips,
                    decision_notes=decision_notes,
                    is_trade_allowed=False,
                    reject_reason="Stop entry deviation exceeds revalidation threshold.",
                )

        if order_type == "STOP":
            if signal == "BUY" and entry_price <= ask:
                decision_notes.append("Stop already triggered; switching to MARKET.")
                order_type = "MARKET"
                entry_price = market_entry
            elif signal == "SELL" and entry_price >= bid:
                decision_notes.append("Stop already triggered; switching to MARKET.")
                order_type = "MARKET"
                entry_price = market_entry
        elif order_type == "LIMIT":
            if signal == "BUY" and entry_price >= ask:
                decision_notes.append("Limit already at/inside market; switching to MARKET.")
                order_type = "MARKET"
                entry_price = market_entry
            elif signal == "SELL" and entry_price <= bid:
                decision_notes.append("Limit already at/inside market; switching to MARKET.")
                order_type = "MARKET"
                entry_price = market_entry

        entry_context = (
            analysis_payload.get("entry_context")
            or analysis_context.get("entry_context", {})
            or {}
        )
        signal_context = self._resolve_signal_context(analysis_payload, analysis_context)
        tp1_target_source = entry_context.get("tp1_target_source")
        tp1_target_zone_type = entry_context.get("tp1_target_zone_type")
        tp1_target_zone_direction = entry_context.get("tp1_target_zone_direction")
        decision_notes.append(
            "CTX counter_trend={counter_trend} reversal_ok={reversal_ok} liquidity_ok={liquidity_ok} trend_ok={trend_ok}".format(
                counter_trend=entry_context.get("counter_trend"),
                reversal_ok=entry_context.get("reversal_ok"),
                liquidity_ok=entry_context.get("liquidity_ok"),
                trend_ok=entry_context.get("trend_ok"),
            )
        )
        if tp1_target_source or tp1_target_zone_type:
            decision_notes.append(
                f"TP1 target source={tp1_target_source} zone_type={tp1_target_zone_type} direction={tp1_target_zone_direction}"
            )
        recent_low = entry_context.get("recent_low")
        recent_high = entry_context.get("recent_high")
        tp1_target_price = entry_context.get("tp1_target_price")
        tp1_policy = self._apply_tp1_target_policy(
            signal=signal,
            entry_context=entry_context,
            signal_context=signal_context,
            decision_notes=decision_notes,
        )
        if tp1_policy.get("action") == "reject":
            return _finalize(
                signal=signal,
                order_type='NONE',
                entry_price=entry_price,
                stop_loss=stop_loss or 0.0,
                take_profit=take_profit or 0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=deviation_pips,
                decision_notes=decision_notes,
                is_trade_allowed=False,
                reject_reason=tp1_policy.get("reject_reason") or "TP1 policy reject.",
            )
        if "tp1_target_price" in tp1_policy:
            tp1_target_price = tp1_policy.get("tp1_target_price")
        sltp = self._compute_scalping_sl_tp(
            signal=signal,
            entry_price=entry_price,
            atr_value=atr_value,
            recent_low=recent_low,
            recent_high=recent_high,
            config_payload=config,
            point_size=point_size,
            tp1_target_price=tp1_target_price,
        )
        stop_loss = sltp.get("stop_loss")
        take_profit = sltp.get("take_profit")
        tp2_price = sltp.get("tp2_price")
        decision_notes.append("SL/TP computed by risk manager scalping model.")
        tp1_source = sltp.get("tp1_source", "fixed_pips")
        sl_source = sltp.get("sl_source", "unknown")
        sl_pips = float(sltp.get("sl_pips") or 0.0)
        tp1_pips = float(sltp.get("tp1_pips") or 0.0)
        decision_notes.append(
            f"SL model: {sl_source} sl_pips={sl_pips:.1f} tp1_source={tp1_source} tp1_pips={tp1_pips:.1f}"
        )
        tp1_partial = float(self.settings.get("FLOW_TP1_PARTIAL_CLOSE_PCT", 0.5))
        move_sl_to_be = bool(self.settings.get("FLOW_TP1_MOVE_SL_TO_BE", True))
        trail_atr_mult = float(self.settings.get("FLOW_TRAIL_ATR_MULT", 2.0))
        decision_notes.append(
            f"TP1 plan: {tp1_source} close {tp1_partial:.0%} at TP1; move SL to BE={move_sl_to_be}; trail {trail_atr_mult:.2f}x ATR after TP1."
        )
        if tp2_price is not None:
            self._logger.info("[NDS][TP2_PLAN] tp2=%.2f intent=runner optional=true", float(tp2_price))

        if stop_loss is None or take_profit is None:
            return _finalize(
                signal=signal,
                order_type='NONE',
                entry_price=entry_price,
                stop_loss=stop_loss or 0.0,
                take_profit=take_profit or 0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["Missing SL/TP from risk manager."],
                is_trade_allowed=False,
                reject_reason="Missing SL/TP.",
            )

        ok_dist, dist_reason = self._distance_sanity_check(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            point_size=point_size,
            atr_value=atr_value,
        )
        if not ok_dist:
            self._logger.warning("[NDS][DIST_SANITY_FAIL] %s", dist_reason)
            return _finalize(
                signal=signal,
                order_type='NONE',
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=deviation_pips,
                decision_notes=[f"Distance sanity failed: {dist_reason}"],
                is_trade_allowed=False,
                reject_reason="Distance sanity failed.",
            )

        sl_metrics = calculate_distance_metrics(
            entry_price=entry_price,
            current_price=stop_loss,
            point_size=point_size,
        )
        tp_metrics = calculate_distance_metrics(
            entry_price=entry_price,
            current_price=take_profit,
            point_size=point_size,
        )
        sl_distance = float(sl_metrics.get("dist_price") or 0.0)
        tp_distance = float(tp_metrics.get("dist_price") or 0.0)
        sl_pips = float(sl_metrics.get("dist_pips") or 0.0)
        tp_pips = float(tp_metrics.get("dist_pips") or 0.0)
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0
        decision_notes.append(
            f"RR_PRECHECK sl_pips={sl_pips:.1f} tp_pips={tp_pips:.1f} rr={rr_ratio:.2f}"
        )
        self._logger.info(
            "[NDS][RR_METRICS] sl_pips=%.2f tp_pips=%.2f rr=%.2f sl_src=%s tp_src=%s",
            sl_pips,
            tp_pips,
            rr_ratio,
            sl_source,
            tp1_source,
        )
        self._logger.info(
            "[NDS][RR_VALIDATE] rr_raw=%.8f min_rr=%.4f epsilon=%.6f sl_pips=%.2f tp_pips=%.2f",
            rr_ratio,
            float(min_rr_ratio),
            rr_epsilon,
            sl_pips,
            tp_pips,
        )

        # ===============================
        # ✅ FIX: inject last signal context for can_scalp session gating
        # ===============================
        try:
            self.last_signal_confidence = float(confidence) if confidence is not None else 0.0
        except Exception:
            self.last_signal_confidence = 0.0

        adx_val, _adx_source = self._resolve_adx_from_signal(analysis_payload)
        self.last_adx = float(adx_val or 0.0)

        if atr_value and max_entry_atr_deviation is not None:
            atr_deviation = deviation / atr_value if atr_value > 0 else 0.0
            if atr_deviation > max_entry_atr_deviation:
                decision_notes.append(
                    f"Entry deviation {atr_deviation:.2f} ATR > max {max_entry_atr_deviation:.2f}."
                )
                return _finalize(
                    signal=signal,
                    order_type='NONE',
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    lot_size=0.0,
                    risk_amount_usd=0.0,
                    rr_ratio=0.0,
                    deviation_pips=deviation_pips,
                    decision_notes=decision_notes,
                    is_trade_allowed=False,
                    reject_reason="Entry deviates beyond ATR threshold.",
                )

        sl_distance = entry_price - stop_loss if signal == 'BUY' else stop_loss - entry_price
        tp_distance = take_profit - entry_price if signal == 'BUY' else entry_price - take_profit
        if sl_distance <= 0 or tp_distance <= 0:
            return _finalize(
                signal=signal,
                order_type='NONE',
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=deviation_pips,
                decision_notes=["Invalid SL/TP distances from risk model."],
                is_trade_allowed=False,
                reject_reason="SL/TP distances invalid for signal.",
            )

        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0
        if rr_ratio + rr_epsilon < min_rr_ratio:
            repaired_tp, repair_source, repair_reject = self._attempt_rr_repair(
                signal=signal,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                min_rr_ratio=float(min_rr_ratio),
                point_size=point_size,
                atr_value=atr_value,
                risk_manager_config=risk_manager_config,
                decision_notes=decision_notes,
            )
            if repair_reject:
                return _finalize(
                    signal=signal,
                    order_type='NONE',
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    lot_size=0.0,
                    risk_amount_usd=0.0,
                    rr_ratio=rr_ratio,
                    deviation_pips=deviation_pips,
                    decision_notes=decision_notes,
                    is_trade_allowed=False,
                    reject_reason=repair_reject,
                )
            if repaired_tp is not None:
                take_profit = repaired_tp
                tp1_source = repair_source or tp1_source
                tp_metrics = calculate_distance_metrics(
                    entry_price=entry_price,
                    current_price=take_profit,
                    point_size=point_size,
                )
                tp_distance = float(tp_metrics.get("dist_price") or 0.0)
                tp_pips = float(tp_metrics.get("dist_pips") or 0.0)
                rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0
                decision_notes.append(
                    f"RR_POSTREPAIR sl_pips={sl_pips:.1f} tp_pips={tp_pips:.1f} rr={rr_ratio:.2f}"
                )
                self._logger.info(
                    "[NDS][RR_VALIDATE] rr_raw=%.8f min_rr=%.4f epsilon=%.6f sl_pips=%.2f tp_pips=%.2f post_repair=true",
                    rr_ratio,
                    float(min_rr_ratio),
                    rr_epsilon,
                    sl_pips,
                    tp_pips,
                )
            if rr_ratio + rr_epsilon < min_rr_ratio:
                decision_notes.append(
                    f"RR {rr_ratio:.2f} below minimum {min_rr_ratio:.2f}."
                )
                return _finalize(
                    signal=signal,
                    order_type='NONE',
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    lot_size=0.0,
                    risk_amount_usd=0.0,
                    rr_ratio=rr_ratio,
                    deviation_pips=deviation_pips,
                    decision_notes=decision_notes,
                    is_trade_allowed=False,
                    reject_reason="RR ratio below minimum.",
                )

        account_equity = config.get('ACCOUNT_BALANCE')
        max_risk_usd = risk_settings.get('RISK_AMOUNT_USD')
        if account_equity is None or max_risk_usd is None:
            return _finalize(
                signal=signal,
                order_type='NONE',
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=rr_ratio,
                deviation_pips=deviation_pips,
                decision_notes=["Missing account balance or risk amount in config."],
                is_trade_allowed=False,
                reject_reason="Risk amount settings missing.",
            )

        current_session, _session_source = self._resolve_session_from_signal(analysis_payload)
        risk_params = self.calculate_scalping_position_size(
            account_equity=account_equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_confidence=confidence,
            atr_value=atr_value,
            market_volatility=market_metrics.get('volatility_ratio', 1.0),
            session=current_session,
            max_risk_usd=max_risk_usd
        )

        if not risk_params.validation_passed:
            decision_notes.extend(risk_params.warnings)
            return _finalize(
                signal=signal,
                order_type='NONE',
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.0,
                risk_amount_usd=risk_params.risk_amount,
                rr_ratio=rr_ratio,
                deviation_pips=deviation_pips,
                decision_notes=decision_notes,
                is_trade_allowed=False,
                reject_reason="Risk validation failed.",
            )

        # ===============================
        # ✅ FIX: support both upper/lower keys for MIN/MAX lot
        # ===============================
        min_lot = self._get_gold_spec('MIN_LOT', 0.01)
        max_lot_spec = self._get_gold_spec('MAX_LOT', 50.0)
        lot_step = self._get_gold_spec('LOT_STEP', 0.01)

        max_lot_limit = risk_manager_config.get('MAX_LOT_SIZE')
        lot_size = risk_params.lot_size

        if min_lot is not None and lot_size < float(min_lot):
            decision_notes.append(f"Lot clamped to min {min_lot}.")
            lot_size = float(min_lot)

        if max_lot_limit is not None:
            max_lot = min(float(max_lot_spec), float(max_lot_limit)) if max_lot_spec is not None else float(max_lot_limit)
            if max_lot is not None and lot_size > max_lot:
                decision_notes.append(f"Lot clamped to max {max_lot}.")
                lot_size = max_lot

        self._logger.info(
            "[NDS][EXECUTION_READY] entry=%.2f sl=%.2f tp1=%.2f tp2=%s risk_usd=%.2f",
            float(entry_price),
            float(stop_loss),
            float(take_profit),
            f"{float(tp2_price):.2f}" if tp2_price is not None else "NONE",
            float(risk_params.risk_amount),
        )

        decision_notes.append(
            f"Final entry {entry_price:.2f} SL {stop_loss:.2f} TP {take_profit:.2f} lot {lot_size:.3f}."
        )

        return _finalize(
            signal=signal,
            order_type=order_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            risk_amount_usd=risk_params.risk_amount,
            rr_ratio=rr_ratio,
            deviation_pips=deviation_pips,
            decision_notes=decision_notes,
            is_trade_allowed=True,
            reject_reason=None,
            take_profit2=tp2_price,
        )


    def _validate_scalping_parameters(self, entry: float, sl: float, tp: float,
                                     confidence: float, atr_value: float,
                                     params: ScalpingRiskParameters) -> bool:
        """اعتبارسنجی پارامترهای اسکلپینگ با استفاده از settings یکپارچه"""
        errors = []
        s = self.settings

        # بررسی قیمت‌ها
        if entry <= 0 or sl <= 0 or tp <= 0:
            errors.append("Prices must be positive")

        # بررسی جهت SL/TP
        sl_distance = abs(entry - sl)
        is_valid_buy = (sl < entry) and (tp > entry)
        is_valid_sell = (sl > entry) and (tp < entry)

        if not (is_valid_buy or is_valid_sell):
            errors.append(f"Invalid SL/TP direction | Entry: {entry}, SL: {sl}, TP: {tp}")

        # بررسی اعتماد سیگنال برای اسکلپینگ
        min_confidence = s.get('SCALPING_MIN_CONFIDENCE', 55)
        if confidence < min_confidence:
            errors.append(f"Signal confidence ({confidence}%) below minimum ({min_confidence}%)")

        # بررسی فاصله استاپ برای اسکلپینگ (bounds in pips)
        point_size = resolve_point_size_from_config(self.settings, default=self._get_gold_spec("point", DEFAULT_POINT_SIZE))
        sl_metrics = calculate_distance_metrics(
            entry_price=entry,
            current_price=sl,
            point_size=point_size,
        )
        sl_pips = float(sl_metrics.get("dist_pips") or 0.0)
        min_sl_pips = float(s.get('SL_MIN_PIPS', 10.0))
        max_sl_pips = float(s.get('SL_MAX_PIPS', 40.0))

        if sl_pips < min_sl_pips:
            errors.append(f"Stop distance ({sl_pips:.2f} pips) too small (min: {min_sl_pips} pips)")

        if sl_pips > max_sl_pips:
            errors.append(f"Stop distance ({sl_pips:.2f} pips) too large (max: {max_sl_pips} pips)")

        # بررسی نسبت ریسک/پاداش برای اسکلپینگ
        rr_ratio = abs(tp - entry) / sl_distance if sl_distance > 0 else 0
        min_rr_ratio = s.get('MIN_RISK_REWARD', 1.0)

        rr_epsilon = float(s.get("RR_EPSILON", 1e-6))
        if rr_ratio + rr_epsilon < min_rr_ratio:
            errors.append(
                "Risk/Reward ratio ({rr:.6f}) below minimum ({min_rr:.2f}) "
                "after epsilon ({eps:.6f}).".format(
                    rr=rr_ratio,
                    min_rr=min_rr_ratio,
                    eps=rr_epsilon,
                )
            )

        # بررسی با ATR
        if atr_value and atr_value > 0:
            atr_multiplier = s.get('ATR_SL_MULTIPLIER', 1.5)
            optimal_sl = atr_value * atr_multiplier
            if sl_distance > optimal_sl * 2.0:
                errors.append(f"Stop distance ({sl_distance:.2f}$) > 2x ATR-based stop")

        if errors:
            params.warnings.extend(errors)
            self._logger.warning(f"❌ Scalping validation failed: {errors[:3]}")
            return False

        return True

    def _get_max_scalping_risk_usd(self, account_equity: float) -> float:
        """دریافت حداکثر ریسک دلاری برای اسکلپینگ"""
        s = self.settings
        max_risk_percent = s.get('MAX_RISK_PERCENT', 0.5)
        max_risk_usd = (account_equity * max_risk_percent) / 100

        # محدودیت مطلق اسکلپینگ از نگاشت جدید
        scalping_risk_limit = s.get('SCALPING_RISK_USD', 50.0)
        return min(max_risk_usd, scalping_risk_limit)

    def _calculate_scalping_risk_percent(self, confidence: float, account_equity: float) -> float:
        """محاسبه درصد ریسک برای اسکلپینگ بر اساس اعتماد"""
        s = self.settings
        min_confidence = s.get('SCALPING_MIN_CONFIDENCE', 55)
        high_confidence = s.get('HIGH_CONFIDENCE', 85)

        if confidence >= high_confidence:
            base_risk = 0.5
        elif confidence >= min_confidence:
            range_confidence = high_confidence - min_confidence
            normalized = (confidence - min_confidence) / range_confidence
            base_risk = 0.1 + (0.4 * normalized)
        else:
            base_risk = 0.0

        # اعمال حداقل ریسک دلاری
        min_risk_dollars = s.get('MIN_RISK_DOLLARS', 0.5)
        min_risk_percent = (min_risk_dollars / account_equity) * 100 if account_equity > 0 else 0.0
        return max(base_risk, min_risk_percent)

    def _calculate_scalping_volatility_multiplier(self, volatility: float) -> float:
        """محاسبه ضریب نوسان برای اسکلپینگ بر اساس VOLATILITY_STATES"""
        v_thresholds = config.get('technical_settings.VOLATILITY_STATES', {})

        if volatility < v_thresholds.get('MODERATE_VOLATILITY', {}).get('threshold', 0.8):
            return 0.7
        elif volatility > v_thresholds.get('HIGH_VOLATILITY', {}).get('threshold', 1.3):
            return 0.6
        elif 0.9 <= volatility <= 1.1:
            return 1.0
        else:
            return 0.8

    def _calculate_scalping_history_multiplier(self) -> float:
        """محاسبه ضریب بر اساس سابقه اسکلپینگ و تنظیمات یکپارچه"""
        s = self.settings
        multiplier = 1.0

        if self.consecutive_losses >= 2:
            multiplier *= 0.5
            self._logger.warning(f"Consecutive scalping losses: {self.consecutive_losses}")

        max_trades_per_day = s.get('MAX_DAILY_TRADES', 20)
        if self.trades_today >= max_trades_per_day * 0.8:
            reduction = 1.0 - (self.trades_today / max_trades_per_day)
            multiplier *= max(0.3, reduction)

        if self.scalping_stats['total_scalps'] > 10:
            win_rate = self.scalping_stats['winning_scalps'] / self.scalping_stats['total_scalps']
            if win_rate < 0.5:
                multiplier *= 0.7

        return max(0.2, multiplier)

    def _apply_scalping_risk_limits(self, risk_percent: float, account_equity: float,
                                   max_risk_usd: float) -> float:
        """اعمال محدودیت‌های ریسک با استفاده از تنظیمات یکپارچه"""
        s = self.settings

        min_risk_dollars = s.get('MIN_RISK_DOLLARS', 0.5)
        min_risk_percent = (min_risk_dollars / account_equity) * 100 if account_equity > 0 else 0.0
        risk_percent = max(risk_percent, min_risk_percent)

        max_daily_percent = s.get('MAX_DAILY_RISK_PERCENT', 1.0)
        daily_risk_left = max_daily_percent - ((self.daily_risk_used / account_equity) * 100 if account_equity > 0 else 0.0)
        risk_percent = min(risk_percent, max(0, daily_risk_left))

        max_risk_percent_from_usd = (max_risk_usd / account_equity) * 100 if account_equity > 0 else 0.0
        risk_percent = min(risk_percent, max_risk_percent_from_usd)

        return risk_percent

    def _calculate_scalping_lot_size(self, entry_price: float, stop_loss: float,
                                    risk_amount: float, sl_distance: float) -> float:
        """محاسبه حجم اسکلپینگ با دقت بالا (SAFE SPECS)"""
        tick_value_per_lot = float(self._get_gold_spec('TICK_VALUE_PER_LOT', 1.0))
        min_lot = float(self._get_gold_spec('MIN_LOT', 0.01))
        max_lot_spec = float(self._get_gold_spec('MAX_LOT', 50.0))
        lot_step = float(self._get_gold_spec('LOT_STEP', 0.01))

        risk_per_standard_lot = sl_distance * tick_value_per_lot

        if risk_per_standard_lot <= 0:
            return min_lot

        raw_lot = risk_amount / risk_per_standard_lot

        if lot_step > 0:
            steps = round(raw_lot / lot_step)
            calculated_lot = steps * lot_step
        else:
            calculated_lot = raw_lot

        # استفاده از تنظیمات مپ شده برای حداکثر حجم
        max_lot_limit = self.settings.get('MAX_LOT_SIZE', 2.0)
        max_lot = min(max_lot_spec, float(max_lot_limit))

        if calculated_lot > max_lot * 0.5:
            calculated_lot = max_lot * 0.5

        final_lot = max(min_lot, min(calculated_lot, max_lot))
        return round(final_lot, 3)

    def _calculate_scalping_margin(self, lot_size: float, entry_price: float) -> float:
        """محاسبه مارجین برای اسکلپینگ (SAFE SPECS)"""
        contract_size = float(self._get_gold_spec('CONTRACT_SIZE', 100.0))
        contract_value = float(lot_size) * contract_size * float(entry_price)
        leverage = self.settings.get('MAX_LEVERAGE', 50)
        margin = contract_value / leverage if leverage else contract_value
        return margin * 1.05

    def _calculate_actual_scalping_risk(self, lot_size: float, entry_price: float,
                                        stop_loss: float) -> float:
        """محاسبه ریسک واقعی اسکلپینگ"""
        sl_distance = abs(entry_price - stop_loss)
        tick_value_per_lot = float(self._get_gold_spec('TICK_VALUE_PER_LOT', 1.0))
        risk_per_tick = lot_size * tick_value_per_lot
        return sl_distance * risk_per_tick

    def _calculate_scalping_grade(self, rr_ratio: float, sl_distance: float,
                                  confidence: float) -> str:
        """محاسبه گرید کیفی اسکلپینگ با تنظیمات مپ شده"""
        score = 0
        s = self.settings

        # امتیاز RR
        min_rr = s.get('MIN_RISK_REWARD', 1.0)
        target_rr = s.get('DEFAULT_RISK_REWARD', 1.2)

        if rr_ratio >= target_rr * 1.25:
            score += 3
        elif rr_ratio >= target_rr:
            score += 2
        elif rr_ratio >= min_rr:
            score += 1

        # امتیاز SL distance
        max_sl = s.get('MAX_SL_DISTANCE', 10.0)
        if sl_distance <= max_sl * 0.5:
            score += 3
        elif sl_distance <= max_sl * 0.7:
            score += 2
        elif sl_distance <= max_sl:
            score += 1

        # امتیاز اعتماد
        high_conf = s.get('HIGH_CONFIDENCE', 85)
        min_conf = s.get('SCALPING_MIN_CONFIDENCE', 55)

        if confidence >= high_conf:
            score += 3
        elif confidence >= (high_conf + min_conf) / 2:
            score += 2
        elif confidence >= min_conf:
            score += 1

        grades = {8: "A+", 6: "A", 4: "B", 2: "C", 0: "D"}
        for threshold, grade in grades.items():
            if score >= threshold:
                return grade
        return "D"

    def update_scalping_trade_result(self, profit_loss: float, position_size: float,
                                     duration_minutes: float):
        """به‌روزرسانی وضعیت پس از بسته شدن معامله اسکلپینگ"""
        self.daily_profit_loss += profit_loss
        self.daily_risk_used += abs(profit_loss)
        self.scalping_stats['total_scalps'] += 1

        if profit_loss > 0:
            self.scalping_stats['winning_scalps'] += 1
            ws = self.scalping_stats['winning_scalps']
            self.scalping_stats['avg_win'] = ((self.scalping_stats['avg_win'] * (ws - 1) + profit_loss) / ws)
            self.consecutive_losses = 0
            if profit_loss > self.scalping_stats['best_scalp']:
                self.scalping_stats['best_scalp'] = profit_loss
        else:
            self.consecutive_losses += 1
            loss_count = self.scalping_stats['total_scalps'] - self.scalping_stats['winning_scalps']
            if loss_count > 0:
                self.scalping_stats['avg_loss'] = ((self.scalping_stats['avg_loss'] * (loss_count - 1) + abs(profit_loss)) / loss_count)
            if profit_loss < self.scalping_stats['worst_scalp']:
                self.scalping_stats['worst_scalp'] = profit_loss

        self.scalping_stats['avg_duration'] = ((self.scalping_stats['avg_duration'] * (self.scalping_stats['total_scalps'] - 1) + duration_minutes) / self.scalping_stats['total_scalps'])
        self.trades_today += 1
        self.active_positions = max(0, self.active_positions - 1)

        self._logger.info(f"Scalping trade result: PnL=${profit_loss:.2f}, Daily PnL=${self.daily_profit_loss:.2f}")

    def can_scalp(
        self,
        account_equity: float,
        signal_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        بررسی امکان اسکلپینگ جدید با تنظیمات مپ شده
        - بدون حذف منطق‌های قبلی
        - با سیاست سشن واحد و بدون قوانین پراکنده

        canonical payload keys (preferred in signal_data):
        - session, session_activity, ts_broker, time_mode, broker_utc_offset_hours
        - adx, plus_di, minus_di, confidence, score
        """
        reasons = []
        s = self.settings

        # ===============================
        # 1. Daily Risk Limit
        # ===============================
        max_daily_percent = s.get('MAX_DAILY_RISK_PERCENT', 1.0)
        daily_risk_used_percent = (
            (self.daily_risk_used / account_equity) * 100
            if account_equity > 0 else 0
        )

        if daily_risk_used_percent >= max_daily_percent:
            reasons.append(f"Daily risk limit reached ({daily_risk_used_percent:.1f}%)")

        # ===============================
        # 2. Consecutive Losses
        # ===============================
        if self.consecutive_losses >= 2:
            reasons.append(f"Consecutive losses: {self.consecutive_losses}")

        # ===============================
        # 3. Active Positions Limit
        # ===============================
        max_positions = s.get('MAX_POSITIONS', 4)
        if self.active_positions >= max_positions:
            reasons.append(f"Active positions: {self.active_positions}/{max_positions}")

        # ===============================
        # 4. Daily Trades Limit
        # ===============================
        max_trades = s.get('MAX_DAILY_TRADES', 20)
        if self.trades_today >= max_trades:
            reasons.append(f"Daily trade limit: {self.trades_today}/{max_trades}")

        # ===============================
        # 5. Scalping Session Handling (FIXED / ENFORCED)
        # ===============================
        session_decision, session_source = self._resolve_session_decision(signal_data)
        current_session = session_decision.session_name

        confidence = self._resolve_confidence_from_signal(signal_data)
        adx, adx_source = self._resolve_adx_from_signal(signal_data)

        self.last_signal_confidence = float(confidence or 0.0)
        self.last_adx = float(adx or 0.0)
        self.last_session = str(current_session or "UNKNOWN")

        if adx_source == "missing":
            self._logger.warning("[RISK][ADX] missing in payload; defaulting to 0.0")

        self._logger.info(
            "[RISK][SESSION_POLICY] session=%s(source=%s) tradable=%s mode=%s reason=%s weight=%.2f conf=%.1f adx=%.1f(source=%s)",
            current_session,
            session_source,
            bool(session_decision.is_tradable),
            session_decision.policy_mode,
            session_decision.block_reason or "-",
            float(session_decision.weight),
            confidence,
            adx,
            adx_source,
        )

        strict_match = bool(
            get_setting(self.config, "trading_settings.SESSION_STRICT_ASSERT_MATCH", False)
        )
        if strict_match and session_source == "payload":
            compare_ts = session_decision.ts_broker
            if compare_ts is None and isinstance(signal_data, dict):
                compare_ts = signal_data.get("ts_broker")
            computed = evaluate_session(compare_ts, self.config)
            if (
                computed.session_name != session_decision.session_name
                or computed.is_tradable != session_decision.is_tradable
                or abs(float(computed.weight) - float(session_decision.weight)) > 1e-6
            ):
                self._logger.error(
                    "[RISK][SESSION_POLICY][MISMATCH] payload=%s computed=%s",
                    session_decision.to_payload(),
                    computed.to_payload(),
                )
                raise ValueError("Session policy mismatch between payload and computed decision")

        if not session_decision.is_tradable:
            self._logger.info(
                "[RISK][SESSION] blocked | reason=%s",
                session_decision.block_reason or f"Non-optimal session: {current_session}",
            )
            reasons.append(session_decision.block_reason or f"Non-optimal session: {current_session}")


        # ===============================
        # 6. Final Decision
        # ===============================
        if reasons:
            return False, " | ".join(reasons)

        return True, "OK"

    def get_scalping_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه وضعیت اسکلپینگ"""
        current_session = self.get_current_scalping_session()
        decision = evaluate_session(get_broker_now(), self.config)
        return {
            'daily_risk_used': self.daily_risk_used,
            'daily_profit_loss': self.daily_profit_loss,
            'active_positions': self.active_positions,
            'consecutive_losses': self.consecutive_losses,
            'trades_today': self.trades_today,
            'scalping_stats': self.scalping_stats,
            'last_update': self.last_update.isoformat(),
            'can_scalp': self.can_scalp(1000)[0],
            'current_session': current_session,
            'session_tradable': decision.is_tradable,
            'session_multiplier': self.get_scalping_multiplier(current_session),
            'max_holding_minutes': self.get_max_holding_time(current_session)
        }


# تابع اصلی برای اسکلپینگ
def create_scalping_risk_manager(overrides: Optional[Dict[str, Any]] = None, **kwargs) -> ScalpingRiskManager:
    """
    ایجاد مدیر ریسک اسکلپینگ

    Args:
        overrides: دیکشنری تنظیمات سفارشی
        **kwargs: پارامترهای اضافی

    Returns:
        ScalpingRiskManager: نمونه ایجاد شده
    """
    if "config" in kwargs:
        config_override = kwargs.pop("config")
        if not isinstance(config_override, dict):
            raise TypeError("config must be a dict when passed for backward compatibility.")
        if overrides is None:
            overrides = config_override
        else:
            merged_overrides = dict(config_override)
            merged_overrides.update(overrides)
            overrides = merged_overrides
    return ScalpingRiskManager(overrides=overrides, **kwargs)


# تست عملکرد
if __name__ == "__main__":
    print("🧪 Testing Gold Scalping Risk Manager...")

    # استفاده از config متمرکز
    test_config = {
        'risk_manager_config': {
            'MAX_RISK_PERCENT': 0.5,
            'MIN_RISK_PERCENT': 0.05,
            'MAX_DAILY_RISK_PERCENT': 1.0,
            'MAX_POSITIONS': 3,
            'MAX_DAILY_TRADES': 20,
            'MIN_CONFIDENCE': 65,
            'HIGH_CONFIDENCE': 85,
            'MAX_SL_DISTANCE': 10.0,
            'MIN_SL_DISTANCE': 2.0,
            'ATR_SL_MULTIPLIER': 1.0,
            'MIN_RR_RATIO': 1.0,
            'TARGET_RR_RATIO': 1.2,
            'MAX_LEVERAGE': 50,
            'MAX_LOT_SIZE': 2.0,
            'MIN_RISK_USD': 5.0,
            'MAX_RISK_USD': 50.0,
            'POSITION_TIMEOUT_MINUTES': 60,
        }
    }

    # ایجاد مدیر ریسک اسکلپینگ
    srm = ScalpingRiskManager(overrides=test_config)

    # تست محاسبه حجم اسکلپینگ
    params = srm.calculate_scalping_position_size(
        account_equity=10000.0,
        entry_price=2150.0,
        stop_loss=2145.0,      # 5 دلار فاصله (اسکلپینگ)
        take_profit=2156.0,    # 6 دلار سود (RR=1.2)
        signal_confidence=80.0,
        atr_value=6.5,
        market_volatility=1.1,
        session='OVERLAP_PEAK',
        max_risk_usd=30.0
    )

    print(f"\n✅ Scalping Test Results:")
    print(f"   Lot Size: {params.lot_size:.3f}")
    print(f"   Risk Amount: ${params.risk_amount:.2f}")
    print(f"   Risk Percent: {params.risk_percent:.3f}%")
    print(f"   Actual Risk: {params.actual_risk_percent:.3f}%")
    print(f"   SL Distance: {params.scalping_specific.get('sl_distance', 0):.2f}$")
    print(f"   RR Ratio: {params.scalping_specific.get('rr_ratio', 0):.2f}")
    print(f"   Scalping Grade: {params.scalping_specific.get('scalping_grade', 'N/A')}")
    print(f"   Max Holding: {params.scalping_specific.get('max_holding_minutes', 0)}min")
    print(f"   Validation: {'PASS' if params.validation_passed else 'FAIL'}")

    if params.warnings:
        print(f"   Warnings: {params.warnings}")

    # تست بررسی امکان معامله
    can_scalp, reason = srm.can_scalp(10000.0)
    print(f"\n✅ Can Scalp: {can_scalp} - {reason}")

    # تست خلاصه وضعیت
    summary = srm.get_scalping_summary()
    print(f"\n✅ Current Session: {summary['current_session']}")
    print(f"   Session Friendly: {summary['session_friendly']}")
    print(f"   Session Multiplier: {summary['session_multiplier']:.2f}")

    print("\n✅ Gold Scalping Risk Manager test completed successfully!")
