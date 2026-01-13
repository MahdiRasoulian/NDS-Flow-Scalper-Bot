"""
مدیریت ریسک اسکلپینگ حرفه‌ای برای طلا (XAUUSD) - نسخه اسکلپینگ
بهینه‌شده برای معاملات M1-M5 با استراتژی اسکلپینگ NDS
نسخه یکپارچه با bot_config.json
"""

import logging
import numpy as np
from typing import Dict, Optional, Any, Tuple, List, TYPE_CHECKING, Union
from dataclasses import dataclass, asdict
from datetime import datetime, time, timedelta, timezone
import math

from config.settings import config

from src.trading_bot.nds.distance_utils import (
    calculate_distance_metrics,
    pips_to_price,
    price_to_points,
    points_to_pips,
)

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
        self.GOLD_SPECS.setdefault("point", 0.001)
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
            self._get_gold_spec("point"),
            self._get_gold_spec("digits"),
            self._get_gold_spec("min_lot"),
            self._get_gold_spec("max_lot"),
            self._get_gold_spec("lot_step"),
            self._get_gold_spec("contract_size"),
            self._get_gold_spec("tick_value_per_lot"),
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
                'LIMIT_ORDER_MIN_CONFIDENCE': 'LIMIT_ORDER_MIN_CONFIDENCE'
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
    @staticmethod
    def get_current_scalping_session(dt: datetime = None) -> str:
        """
        Detect current scalping session based on LOCAL trading time (UTC+3).
        This avoids false DEAD_ZONE detection caused by UTC mismatch.
        """

        # ===============================
        # 1. Define trading timezone offset
        # ===============================
        TRADING_UTC_OFFSET = 3  # Iraq / Middle East

        if dt is None:
            dt = datetime.utcnow() + timedelta(hours=TRADING_UTC_OFFSET)

        current_time = dt.time()

        sessions = config.get('sessions_config.SCALPING_SESSIONS', {})

        for session_name, session_data in sessions.items():
            start_hour = session_data.get('start', 0)
            end_hour = session_data.get('end', 0)

            start_time = time(start_hour, 0)
            end_time = time(end_hour, 0)

            # ===============================
            # Normal session (same day)
            # ===============================
            if start_time <= end_time:
                if start_time <= current_time < end_time:
                    return session_name

            # ===============================
            # Overnight session (e.g. 22 → 01)
            # ===============================
            else:
                if current_time >= start_time or current_time < end_time:
                    return session_name

        # ===============================
        # Fallback (safety)
        # ===============================
        return 'DEAD_ZONE'

    @staticmethod
    def is_scalping_friendly_session(session: str) -> bool:
        """
        بررسی مناسب بودن سشن برای اسکلپینگ.

        - این متد فقط «سازگاری پایه» سشن را بررسی می‌کند
        - تصمیم‌گیری نهایی (مانند DEAD_ZONE override) در can_scalp انجام می‌شود
        """

        # DEAD_ZONE به صورت پیش‌فرض مسدود نمی‌شود
        # منطق اجازه/عدم اجازه آن در can_scalp و بر اساس کیفیت سیگنال است
        if session == 'DEAD_ZONE':
            return True

        session_multiplier = config.get('sessions_config.SCALPING_SESSION_ADJUSTMENT', {}).get(session, 0)

        # سشن‌هایی با ضریب مناسب برای اسکلپینگ
        return session_multiplier >= 0.5

    def get_scalping_multiplier(self, session: str) -> float:
        """
        دریافت ضریب ریسک برای اسکلپینگ از منبع واحد تنظیمات.
        """
        # اولویت با تنظیمات داینامیک در self.settings است که در Init لود شده
        multipliers = self.settings.get('SCALPING_SESSION_MULTIPLIERS', {})

        # مقادیر از 0.1 (Dead Zone) تا 1.0 (Overlap) متغیر هستند
        multiplier = multipliers.get(session, 0.5)

        self._logger.debug(f"🔍 Scalping Session Multiplier for {session}: {multiplier}")
        return multiplier

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
        contract_size = float(self._get_gold_spec('contract_size', 100.0))
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
            return analysis
        if hasattr(analysis, "__dataclass_fields__"):
            return asdict(analysis)
        if hasattr(analysis, "__dict__"):
            return dict(analysis.__dict__)
        return {}

    def _get_point_size(self, config_payload: Dict[str, Any]) -> float:
        """Resolve point size with default for XAUUSD mapping."""
        trading_settings = config_payload.get("trading_settings", {}) if isinstance(config_payload, dict) else {}
        gold_specs = self._normalize_gold_specs(trading_settings.get("GOLD_SPECIFICATIONS", {}))
        point_size = (
            gold_specs.get("point")
            or gold_specs.get("POINT")
            or self._get_gold_spec("point", 0.001)
        )
        try:
            point_size = float(point_size)
        except Exception:
            point_size = 0.001
        if point_size <= 0:
            point_size = 0.001
        return point_size

    def _compute_scalping_sl_tp(
        self,
        signal: str,
        entry_price: float,
        atr_value: Optional[float],
        recent_low: Optional[float],
        recent_high: Optional[float],
        config_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute scalping SL/TP with ATR and recent candle extrema reference."""
        settings = self.settings
        point_size = self._get_point_size(config_payload)

        atr_mult = float(settings.get("SCALP_ATR_SL_MULT", 1.5))
        sl_min_pips = float(settings.get("SL_MIN_PIPS", 10.0))
        sl_max_pips = float(settings.get("SL_MAX_PIPS", 40.0))
        tp1_pips = float(settings.get("TP1_PIPS", 35.0))
        tp2_pips = float(settings.get("TP2_PIPS", tp1_pips * 2.0))

        atr_value = float(atr_value) if atr_value is not None else 0.0
        atr_distance = atr_value * atr_mult if atr_value > 0 else 0.0

        ref_distance = 0.0
        if signal == "BUY" and recent_low is not None:
            ref_distance = max(0.0, float(entry_price) - float(recent_low))
        elif signal == "SELL" and recent_high is not None:
            ref_distance = max(0.0, float(recent_high) - float(entry_price))

        if atr_distance > 0 and ref_distance > 0:
            sl_distance = min(atr_distance, ref_distance)
        elif atr_distance > 0:
            sl_distance = atr_distance
        else:
            sl_distance = ref_distance

        if sl_distance <= 0:
            sl_distance = pips_to_price(sl_min_pips, point_size)

        sl_points = price_to_points(sl_distance, point_size)
        sl_pips = points_to_pips(sl_points)
        raw_sl_pips = sl_pips
        if sl_pips < sl_min_pips:
            sl_pips = sl_min_pips
            sl_distance = pips_to_price(sl_pips, point_size)
        elif sl_pips > sl_max_pips:
            sl_pips = sl_max_pips
            sl_distance = pips_to_price(sl_pips, point_size)

        if raw_sl_pips != sl_pips:
            self._logger.info(
                "[NDS][SL_CLAMP] raw=%.2f clamped=%.2f bounds=[%.2f,%.2f]",
                raw_sl_pips,
                sl_pips,
                sl_min_pips,
                sl_max_pips,
            )

        if signal == "BUY":
            stop_loss = float(entry_price) - sl_distance
            take_profit = float(entry_price) + pips_to_price(tp1_pips, point_size)
            tp2_price = float(entry_price) + pips_to_price(tp2_pips, point_size)
        else:
            stop_loss = float(entry_price) + sl_distance
            take_profit = float(entry_price) - pips_to_price(tp1_pips, point_size)
            tp2_price = float(entry_price) - pips_to_price(tp2_pips, point_size)

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "tp1_pips": tp1_pips,
            "tp2_pips": tp2_pips,
            "tp2_price": tp2_price,
            "sl_pips": sl_pips,
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

        decision_notes: List[str] = []
        analysis_reasons = analysis_payload.get('reasons') or []
        decision_notes.extend(list(analysis_reasons))
        signal = analysis_payload.get('signal')
        if not signal or signal in ['NONE', 'NEUTRAL']:
            return FinalizedOrderParams(
                signal=signal or 'NONE',
                order_type='NONE',
                symbol=symbol,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["No actionable signal from analyzer."],
                is_trade_allowed=False,
                reject_reason="Signal is NONE/NEUTRAL."
            )

        planned_entry = analysis_payload.get('entry_price')
        stop_loss = None
        take_profit = None
        confidence = analysis_payload.get('confidence')

        if confidence is None:
            confidence = 0.0

        bid = live_payload.get('bid')
        ask = live_payload.get('ask')
        if bid is None or ask is None:
            return FinalizedOrderParams(
                signal=signal,
                order_type='NONE',
                symbol=symbol,
                entry_price=planned_entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["Live snapshot missing bid/ask."],
                is_trade_allowed=False,
                reject_reason="Live prices unavailable."
            )

        spread = float(ask) - float(bid)
        point_size = self._get_point_size(config)
        spread_points = price_to_points(spread, point_size)
        spread_pips = points_to_pips(spread_points)
        spread_max_pips = float(self.settings.get("SPREAD_MAX_PIPS", 2.5))
        if spread_pips > spread_max_pips:
            self._logger.info(
                "[NDS][RISK_GATE] allow=false reason=SPREAD_TOO_HIGH spread_pips=%.2f max=%.2f",
                spread_pips,
                spread_max_pips,
            )
            return FinalizedOrderParams(
                signal=signal,
                order_type='NONE',
                symbol=symbol,
                entry_price=planned_entry,
                stop_loss=stop_loss or 0.0,
                take_profit=take_profit or 0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=[f"Spread too high ({spread_pips:.2f} > {spread_max_pips:.2f})"],
                is_trade_allowed=False,
                reject_reason="Spread too high."
            )

        analysis_context = analysis_payload.get('context', {}) or {}
        entry_idea = (
            analysis_payload.get("entry_idea")
            or analysis_context.get("entry_idea", {})
            or {}
        )
        entry_model = entry_idea.get("entry_model") or analysis_payload.get("entry_model") or "MARKET"
        planned_entry = entry_idea.get("entry_level")
        market_metrics = analysis_payload.get('market_metrics') or analysis_context.get('market_metrics', {})
        atr_value = market_metrics.get('atr_short') or market_metrics.get('atr')

        if planned_entry is None:
            planned_entry = analysis_payload.get('entry_price')

        if planned_entry is None:
            return FinalizedOrderParams(
                signal=signal,
                order_type='NONE',
                symbol=symbol,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["Missing entry idea from analyzer."],
                is_trade_allowed=False,
                reject_reason="Missing entry idea."
            )

        risk_settings = config.get('risk_settings', {})
        trading_settings = config.get('trading_settings', {})
        risk_manager_config = config.get('risk_manager_config', {})

        max_entry_atr_deviation = risk_settings.get('MAX_ENTRY_ATR_DEVIATION')
        min_rr_ratio = risk_manager_config.get('MIN_RR_RATIO')

        if min_rr_ratio is None:
            return FinalizedOrderParams(
                signal=signal,
                order_type='NONE',
                symbol=symbol,
                entry_price=planned_entry or 0.0,
                stop_loss=stop_loss or 0.0,
                take_profit=take_profit or 0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["Missing risk settings in config."],
                is_trade_allowed=False,
                reject_reason="Risk settings missing from config."
            )

        market_entry = ask if signal == 'BUY' else bid
        deviation = abs(planned_entry - market_entry)

        gold_specs = self._normalize_gold_specs(trading_settings.get('GOLD_SPECIFICATIONS', {}))
        point_size = gold_specs.get('point') or self._get_gold_spec('point', 0.001)
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

        order_type = "STOP" if str(entry_model).upper() == "STOP" else "MARKET"
        entry_price = planned_entry if order_type == "STOP" else market_entry

        if order_type == "STOP":
            if signal == "BUY" and planned_entry <= ask:
                decision_notes.append("Stop already triggered; switching to MARKET.")
                order_type = "MARKET"
                entry_price = market_entry
            elif signal == "SELL" and planned_entry >= bid:
                decision_notes.append("Stop already triggered; switching to MARKET.")
                order_type = "MARKET"
                entry_price = market_entry

        entry_context = (
            analysis_payload.get("entry_context")
            or analysis_context.get("entry_context", {})
            or {}
        )
        recent_low = entry_context.get("recent_low")
        recent_high = entry_context.get("recent_high")
        sltp = self._compute_scalping_sl_tp(
            signal=signal,
            entry_price=entry_price,
            atr_value=atr_value,
            recent_low=recent_low,
            recent_high=recent_high,
            config_payload=config,
        )
        stop_loss = sltp.get("stop_loss")
        take_profit = sltp.get("take_profit")
        tp2_price = sltp.get("tp2_price")
        decision_notes.append("SL/TP computed by risk manager scalping model.")
        if tp2_price is not None:
            self._logger.info("[NDS][TP2_PLAN] tp2=%.2f intent=runner optional=true", float(tp2_price))

        if stop_loss is None or take_profit is None:
            return FinalizedOrderParams(
                signal=signal,
                order_type='NONE',
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss or 0.0,
                take_profit=take_profit or 0.0,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=0.0,
                decision_notes=["Missing SL/TP from risk manager."],
                is_trade_allowed=False,
                reject_reason="Missing SL/TP."
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
            return FinalizedOrderParams(
                signal=signal,
                order_type='NONE',
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=deviation_pips,
                decision_notes=[f"Distance sanity failed: {dist_reason}"],
                is_trade_allowed=False,
                reject_reason="Distance sanity failed."
            )

        # ===============================
        # ✅ FIX: inject last signal context for can_scalp session gating
        # ===============================
        try:
            self.last_signal_confidence = float(confidence) if confidence is not None else 0.0
        except Exception:
            self.last_signal_confidence = 0.0

        adx_val = market_metrics.get('adx')
        if adx_val is None:
            # تلاش برای پیدا کردن ADX در payloadهای دیگر (اگر موجود باشد)
            adx_val = analysis_payload.get('adx') or analysis_context.get('adx')

        try:
            self.last_adx = float(adx_val) if adx_val is not None else 0.0
        except Exception:
            self.last_adx = 0.0

        if atr_value and max_entry_atr_deviation is not None:
            atr_deviation = deviation / atr_value if atr_value > 0 else 0.0
            if atr_deviation > max_entry_atr_deviation:
                decision_notes.append(
                    f"Entry deviation {atr_deviation:.2f} ATR > max {max_entry_atr_deviation:.2f}."
                )
                return FinalizedOrderParams(
                    signal=signal,
                    order_type='NONE',
                    symbol=symbol,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    lot_size=0.0,
                    risk_amount_usd=0.0,
                    rr_ratio=0.0,
                    deviation_pips=deviation_pips,
                    decision_notes=decision_notes,
                    is_trade_allowed=False,
                    reject_reason="Entry deviates beyond ATR threshold."
                )

        sl_distance = entry_price - stop_loss if signal == 'BUY' else stop_loss - entry_price
        tp_distance = take_profit - entry_price if signal == 'BUY' else entry_price - take_profit
        if sl_distance <= 0 or tp_distance <= 0:
            return FinalizedOrderParams(
                signal=signal,
                order_type='NONE',
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=0.0,
                deviation_pips=deviation_pips,
                decision_notes=["Invalid SL/TP distances from risk model."],
                is_trade_allowed=False,
                reject_reason="SL/TP distances invalid for signal."
            )

        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0

        if rr_ratio < min_rr_ratio:
            decision_notes.append(
                f"RR {rr_ratio:.2f} below minimum {min_rr_ratio:.2f}."
            )
            return FinalizedOrderParams(
                signal=signal,
                order_type='NONE',
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=rr_ratio,
                deviation_pips=deviation_pips,
                decision_notes=decision_notes,
                is_trade_allowed=False,
                reject_reason="RR ratio below minimum."
            )

        account_equity = config.get('ACCOUNT_BALANCE')
        max_risk_usd = risk_settings.get('RISK_AMOUNT_USD')
        if account_equity is None or max_risk_usd is None:
            return FinalizedOrderParams(
                signal=signal,
                order_type='NONE',
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.0,
                risk_amount_usd=0.0,
                rr_ratio=rr_ratio,
                deviation_pips=deviation_pips,
                decision_notes=["Missing account balance or risk amount in config."],
                is_trade_allowed=False,
                reject_reason="Risk amount settings missing."
            )

        current_session = self.get_current_scalping_session()
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
            return FinalizedOrderParams(
                signal=signal,
                order_type='NONE',
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.0,
                risk_amount_usd=risk_params.risk_amount,
                rr_ratio=rr_ratio,
                deviation_pips=deviation_pips,
                decision_notes=decision_notes,
                is_trade_allowed=False,
                reject_reason="Risk validation failed."
            )

        # ===============================
        # ✅ FIX: support both upper/lower keys for MIN/MAX lot
        # ===============================
        min_lot = (
            gold_specs.get('min_lot')
            or gold_specs.get('MIN_LOT')
            or self._get_gold_spec('min_lot', 0.01)
        )
        max_lot_spec = (
            gold_specs.get('max_lot')
            or gold_specs.get('MAX_LOT')
            or self._get_gold_spec('max_lot', 50.0)
        )

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

        return FinalizedOrderParams(
            signal=signal,
            order_type=order_type,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            risk_amount_usd=risk_params.risk_amount,
            rr_ratio=rr_ratio,
            deviation_pips=deviation_pips,
            decision_notes=decision_notes,
            is_trade_allowed=True,
            reject_reason=None
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
        point_size = self._get_gold_spec('point', 0.001)
        sl_points = price_to_points(sl_distance, point_size)
        sl_pips = points_to_pips(sl_points)
        min_sl_pips = float(s.get('SL_MIN_PIPS', 10.0))
        max_sl_pips = float(s.get('SL_MAX_PIPS', 40.0))

        if sl_pips < min_sl_pips:
            errors.append(f"Stop distance ({sl_pips:.2f} pips) too small (min: {min_sl_pips} pips)")

        if sl_pips > max_sl_pips:
            errors.append(f"Stop distance ({sl_pips:.2f} pips) too large (max: {max_sl_pips} pips)")

        # بررسی نسبت ریسک/پاداش برای اسکلپینگ
        rr_ratio = abs(tp - entry) / sl_distance if sl_distance > 0 else 0
        min_rr_ratio = s.get('MIN_RISK_REWARD', 1.0)

        if rr_ratio < min_rr_ratio:
            errors.append(f"Risk/Reward ratio ({rr_ratio:.2f}) below minimum ({min_rr_ratio})")

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
        tick_value_per_lot = float(self._get_gold_spec('tick_value_per_lot', 1.0))
        min_lot = float(self._get_gold_spec('min_lot', 0.01))
        max_lot_spec = float(self._get_gold_spec('max_lot', 50.0))
        lot_step = float(self._get_gold_spec('lot_step', 0.01))

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
        contract_size = float(self._get_gold_spec('contract_size', 100.0))
        contract_value = float(lot_size) * contract_size * float(entry_price)
        leverage = self.settings.get('MAX_LEVERAGE', 50)
        margin = contract_value / leverage if leverage else contract_value
        return margin * 1.05

    def _calculate_actual_scalping_risk(self, lot_size: float, entry_price: float,
                                        stop_loss: float) -> float:
        """محاسبه ریسک واقعی اسکلپینگ"""
        sl_distance = abs(entry_price - stop_loss)
        tick_value_per_lot = float(self._get_gold_spec('tick_value_per_lot', 1.0))
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

    def can_scalp(self, account_equity: float) -> Tuple[bool, str]:
        """
        بررسی امکان اسکلپینگ جدید با تنظیمات مپ شده
        - بدون حذف منطق‌های قبلی
        - با DEAD_ZONE override واقعی و enforce شده
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
        current_session = self.get_current_scalping_session()

        # --- LOG: وضعیت سشن و ورودی‌های تصمیم ---
        try:
            friendly = bool(self.is_scalping_friendly_session(current_session))
        except Exception:
            friendly = False

        confidence = float(getattr(self, 'last_signal_confidence', 0.0) or 0.0)
        adx = float(getattr(self, 'last_adx', 0.0) or 0.0)

        self._logger.info(
            "[RISK][SESSION] current_session=%s friendly=%s last_conf=%.1f last_adx=%.1f",
            current_session, friendly, confidence, adx
        )

        # ✅ CRITICAL FIX:
        # چون is_scalping_friendly_session('DEAD_ZONE') == True است،
        # باید منطق DEAD_ZONE را جداگانه و صریح enforce کنیم؛ وگرنه override هیچوقت اجرا نمی‌شود.
        if current_session == 'DEAD_ZONE':
            # --- LOG: ورود به مسیر DEAD_ZONE ---
            conf_th = 65.0
            adx_th = 20.0
            self._logger.info(
                "[RISK][DEAD_ZONE] evaluating override | conf=%.1f(th=%.1f) adx=%.1f(th=%.1f)",
                confidence, conf_th, adx, adx_th
            )

            if confidence >= conf_th and adx >= adx_th:
                # ✅ اجازه معامله در DEAD_ZONE
                self.session_risk_multiplier = 0.4

                # --- LOG: پذیرش override ---
                self._logger.info(
                    "[RISK][DEAD_ZONE] override ACCEPTED | session_risk_multiplier=%.2f | conf=%.1f adx=%.1f",
                    float(getattr(self, "session_risk_multiplier", 1.0) or 1.0),
                    confidence,
                    adx,
                )

                # ✅ FIX: self.logger وجود ندارد، باید self._logger استفاده شود
                self._logger.info(
                    f"🔥 DEAD_ZONE override accepted | "
                    f"Confidence={confidence:.1f}% | ADX={adx:.1f}"
                )
            else:
                # --- LOG: رد override با دلیل دقیق ---
                fail_reasons = []
                if confidence < conf_th:
                    fail_reasons.append(f"conf {confidence:.1f} < {conf_th:.1f}")
                if adx < adx_th:
                    fail_reasons.append(f"adx {adx:.1f} < {adx_th:.1f}")
                self._logger.info(
                    "[RISK][DEAD_ZONE] override REJECTED | %s",
                    " & ".join(fail_reasons) if fail_reasons else "unknown"
                )

                reasons.append(f"Non-optimal session: {current_session}")
        else:
            # سایر سشن‌ها طبق منطق قبلی
            # --- LOG: مسیر non-DEAD_ZONE ---
            self._logger.info(
                "[RISK][SESSION] non-deadzone path | session=%s friendly=%s",
                current_session,
                friendly
            )

            if not friendly:
                # --- LOG: رد به دلیل unfriendly بودن سشن ---
                self._logger.info(
                    "[RISK][SESSION] blocked | reason=Non-optimal session: %s",
                    current_session
                )
                reasons.append(f"Non-optimal session: {current_session}")


        # ===============================
        # 6. Final Decision
        # ===============================
        if reasons:
            return False, " | ".join(reasons)

        return True, "OK"

    def get_scalping_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه وضعیت اسکلپینگ"""
        current_session = self.get_current_scalping_session()
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
            'session_friendly': self.is_scalping_friendly_session(current_session),
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
