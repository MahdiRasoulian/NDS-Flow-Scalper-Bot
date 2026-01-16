# src/trading_bot/nds/analyzer.py
"""
آنالایزر اصلی NDS برای طلا - نسخه بازنویسی شده با منطق امتیازدهی سازگار

Config keys (whitelisted via ANALYSIS_CONFIG_KEYS):
- ATR_WINDOW, ADX_WINDOW
- SCALPING_MIN_CONFIDENCE, MIN_CONFIDENCE
- MIN_RVOL_SCALPING, MIN_SESSION_WEIGHT, MIN_STRUCTURE_SCORE
- DEFAULT_TIMEFRAME, MIN_RR, ATR_BUFFER_MULTIPLIER
- ADX_OVERRIDE_THRESHOLD, ADX_OVERRIDE_PERSISTENCE_BARS, ADX_OVERRIDE_REQUIRE_BOS
"""
# CHANGELOG:
# - Added directional bias computation + counter-trend gate (with reversal confirmation) to block signals
#   that fight strong trends unless CHOCH/sweep-based reversal criteria are met.
# - Attached signal_context (bias/reversal/indicator state) into result payload for bot/backtest reporting.
# - Preserved legacy scoring, confidence, entry idea logic, and payload schema; no execution/risk logic added.
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from collections import deque
_SCORE_HIST = deque(maxlen=400)

import math
from statistics import mean, pstdev

import pandas as pd

from .constants import (
    ANALYSIS_CONFIG_KEYS,
    SESSION_MAPPING,
)
from .models import (
    AnalysisResult, SessionAnalysis,
    OrderBlock, FVG, FVGType, MarketStructure, MarketTrend
)
from .indicators import IndicatorCalculator
from .smc import SMCAnalyzer
from .distance_utils import (
    calculate_distance_metrics,
    normalize_spread,
    pips_to_price,
    resolve_pip_size_from_config,
    resolve_point_size_with_source,
)
from src.trading_bot.config_utils import get_setting, resolve_active_settings
from src.trading_bot.time_utils import (
    in_time_window,
    parse_timestamp,
    to_broker_time,
    DEFAULT_BROKER_OFFSET_HOURS,
    DEFAULT_TIME_MODE,
    DEFAULT_SESSION_DEFINITIONS,
    normalize_session_definitions,
)
from src.trading_bot.session_policy import evaluate_session, normalize_session_payload

logger = logging.getLogger(__name__)


class GoldNDSAnalyzer:
    """
    نسخه ماژولار و بهینه‌شده آنالایزر طلا (XAUUSD) - نسخه 6.0
    """

    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        if df.empty:
            raise ValueError("DataFrame is empty. Cannot initialize analyzer.")

        self.df = df.copy()
        self.config = config or {}
        self.debug_analyzer = False
        self._resolved_settings_payload = resolve_active_settings(self.config)
        self._resolved_settings = self._resolved_settings_payload.get("resolved", {})

        technical_settings = (self.config or {}).get('technical_settings', {})
        sessions_config = (self.config or {}).get('sessions_config', {})
        self.GOLD_SETTINGS = technical_settings.copy()
        self.TRADING_SESSIONS = sessions_config.get('BASE_TRADING_SESSIONS', {}).copy()
        self.timeframe_specifics = technical_settings.get('TIMEFRAME_SPECIFICS', {})
        self.swing_period_map = technical_settings.get('SWING_PERIOD_MAP', {})

        self.atr: Optional[float] = None
        self._point_size: Optional[float] = None
        self._point_size_source: Optional[str] = None
        self._pip_size: Optional[float] = None
        self.time_mode = DEFAULT_TIME_MODE
        self.broker_utc_offset = DEFAULT_BROKER_OFFSET_HOURS
        self.session_definitions: Dict[str, Dict[str, Any]] = dict(DEFAULT_SESSION_DEFINITIONS)

        self._apply_custom_config()
        self._init_time_settings()
        self._validate_dataframe()
        self.timeframe = self._detect_timeframe()
        self._apply_timeframe_settings()

        self._score_hist = _SCORE_HIST
        self._threshold_ema = {"buy": None, "sell": None}

        self._log_info(
            "[NDS][INIT] initialized candles=%s timeframe=%s",
            len(self.df),
            self.timeframe,
        )

    def _apply_custom_config(self) -> None:
        """اعمال تنظیمات سفارشی از config خارجی"""
        if self.config is None:
            return

        analyzer_config, sessions_config = self._extract_config_payload()
        if analyzer_config:
            self._apply_analyzer_settings(analyzer_config)

        if sessions_config:
            self._apply_sessions_config(sessions_config)

        flow_settings = self._resolved_settings.get("flow_settings", {})
        momentum_settings = self._resolved_settings.get("momentum_settings", {})
        risk_settings = self._resolved_settings.get("risk_settings", {})
        trading_settings = self._resolved_settings.get("trading_settings", {})
        for section in (flow_settings, momentum_settings, risk_settings, trading_settings):
            if isinstance(section, dict):
                self.GOLD_SETTINGS.update(section)

    def _init_time_settings(self) -> None:
        self.time_mode = str(get_setting(self.config, "trading_settings.TIME_MODE", DEFAULT_TIME_MODE) or DEFAULT_TIME_MODE).upper()
        self.broker_utc_offset = float(
            get_setting(self.config, "trading_settings.BROKER_UTC_OFFSET_HOURS", DEFAULT_BROKER_OFFSET_HOURS)
            or DEFAULT_BROKER_OFFSET_HOURS
        )
        session_defs = get_setting(self.config, "trading_settings.SESSION_DEFINITIONS", None)
        if not session_defs:
            session_defs = get_setting(self.config, "sessions_config.BASE_TRADING_SESSIONS", {}) or self.TRADING_SESSIONS
        session_defs = normalize_session_definitions(session_defs)
        if not session_defs:
            session_defs = dict(DEFAULT_SESSION_DEFINITIONS)
        self.session_definitions = session_defs

    def _extract_config_payload(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """دریافت تنظیمات مجاز آنالایزر و سشن‌ها از کانفیگ اصلی"""
        analyzer_config: Dict[str, Any] = {}
        sessions_config: Dict[str, Any] = {}

        analyzer_config = get_setting(self.config, "ANALYZER_SETTINGS", None)
        if not analyzer_config:
            analyzer_config = get_setting(self.config, "technical_settings", {}) or {}

        debug_flag = get_setting(self.config, "DEBUG_ANALYZER", None)
        if debug_flag is not None and 'DEBUG_ANALYZER' not in analyzer_config:
            analyzer_config = {**analyzer_config, 'DEBUG_ANALYZER': debug_flag}

        sessions_config = get_setting(self.config, "TRADING_SESSIONS", None)
        if sessions_config is None:
            sessions_config = get_setting(self.config, "sessions_config.TRADING_SESSIONS", {}) or {}

        return analyzer_config, sessions_config

    def _apply_analyzer_settings(self, analyzer_config: Dict[str, Any]) -> None:
        """اعمال تنظیمات تحلیل با وایت‌لیست دقیق"""
        self.debug_analyzer = bool(analyzer_config.get('DEBUG_ANALYZER', self.debug_analyzer))

        type_map = {
            'ATR_WINDOW': int,
            'ADX_WINDOW': int,
            'SCALPING_MIN_CONFIDENCE': float,
            'MIN_CONFIDENCE': float,
            'MIN_RVOL_SCALPING': float,
            'MIN_SESSION_WEIGHT': float,
            'MIN_STRUCTURE_SCORE': float,
            'DEFAULT_TIMEFRAME': str,
            'MIN_RR': float,
            'ATR_BUFFER_MULTIPLIER': float,
            'ADX_OVERRIDE_THRESHOLD': float,
            'ADX_OVERRIDE_PERSISTENCE_BARS': int,
            'ADX_OVERRIDE_REQUIRE_BOS': bool,
            'ENABLE_LEGACY_SWING_ENTRY': bool,
            'FLOW_STOP_BUFFER_ATR': float,
            'FLOW_RETEST_POLICY': str,
            'FLOW_TOUCH_PENETRATION_ATR': float,
            'FLOW_MAX_TOUCHES': int,
            'FLOW_TOUCH_PENALTY': float,
            'FLOW_TOUCH_EXIT_ATR': float,
            'FLOW_TOUCH_EXIT_PIPS': float,
            'FLOW_TOUCH_MIN_SEPARATION_BARS': int,
            'FLOW_TOUCH_EXIT_CONFIRM_BARS': int,
            'FLOW_TOUCH_COUNT_WINDOW_BARS': int,
            'FLOW_TOUCH_COOLDOWN_BARS': int,
            'FLOW_CONSUME_ON_FIRST_VALID_TOUCH': bool,
            'FLOW_ZONE_KEY_PRECISION': int,
            'FLOW_NEAREST_ZONES': int,
            'FLOW_SETUP_WEIGHTS': dict,
            'FLOW_SETUP_TOP_K': int,
            'FLOW_SETUP_DISPLACEMENT_ATR_TARGET': float,
            'MOMO_SESSION_ALLOWLIST': list,
            'MIN_SL_PIPS': float,
            'SMC_MIN_CANDLES': int,
            'SMC_MAX_FVG_COUNT': int,
            'SMC_MAX_OB_COUNT': int,
            'SMC_MAX_FVG_AGE_BARS': int,
            'SMC_MAX_OB_AGE_BARS': int,
            'SMC_MIN_FVG_SIZE_ATR': float,
            'SMC_MIN_OB_SIZE_ATR': float,
            'SMC_OB_RELAX_MIN_SIZE_MULT': float,
            'SMC_OB_FALLBACK_TOP_K': int,
            'SMC_OB_FALLBACK_MAX_AGE_MULT': float,
            'SMC_OB_FALLBACK_MAX_DIST_MULT': float,
            'SMC_OB_FALLBACK_MIN_STRENGTH': float,
            'SMC_ZONE_MAX_DIST_ATR': float,
            'SMC_ZONE_TIGHTEN_MULT': float,
            'SMC_FVG_RANK_WEIGHTS': dict,
            'SMC_OB_RANK_WEIGHTS': dict,
            'SCALPING_THRESHOLD_MODE': str,
            'SIGNAL_THRESHOLD_MODE': str,
            'SIGNAL_BUY_PERCENTILE': float,
            'SIGNAL_SELL_PERCENTILE': float,
            'PERCENTILE_MIN_HISTORY': int,
            'SIGNAL_MIN_THRESHOLD_SPREAD': float,
            'SIGNAL_BUY_THRESHOLD_MIN': float,
            'SIGNAL_BUY_THRESHOLD_MAX': float,
            'SIGNAL_SELL_THRESHOLD_MIN': float,
            'SIGNAL_SELL_THRESHOLD_MAX': float,
            'SIGNAL_THRESHOLD_EMA_ALPHA': float,
            'INTEGRITY_LOW_LIQUIDITY_RVOL_MAX': float,
            'INTEGRITY_LOW_LIQUIDITY_SESSION_ACTIVITY': str,
            'INTEGRITY_LOW_LIQUIDITY_FORCE_NONE': bool,
            'INTEGRITY_EXCEPTIONAL_SETUP_SCORE': float,
            'REGIME_ADX_WEAK_MAX': float,
            'REGIME_TREND_WEIGHT_MULT_LOW_ADX': float,
            'REGIME_FVG_WEIGHT_MULT_LOW_RVOL': float,
            'REGIME_OB_WEIGHT_MULT_LOW_RVOL': float,
        }

        validated_config: Dict[str, Any] = {}
        ignored_keys: List[str] = []

        for key, value in analyzer_config.items():
            if key == 'DEBUG_ANALYZER':
                continue
            if key not in ANALYSIS_CONFIG_KEYS:
                ignored_keys.append(key)
                continue

            target_type = type_map.get(key)
            parsed_value = None
            if target_type is None:
                if isinstance(value, (int, float, bool, str)):
                    parsed_value = value
            elif target_type is bool:
                if isinstance(value, bool):
                    parsed_value = value
                elif isinstance(value, str):
                    parsed_value = value.strip().lower() in {'1', 'true', 'yes', 'y'}
            elif target_type is list:
                if isinstance(value, (list, tuple, set)):
                    parsed_value = [str(v).strip() for v in value if str(v).strip()]
                elif isinstance(value, str):
                    parsed_value = [v.strip() for v in value.split(",") if v.strip()]
            elif target_type is dict:
                if isinstance(value, dict):
                    parsed_value = value
                else:
                    parsed_value = None
            else:
                try:
                    parsed_value = target_type(value)
                except (TypeError, ValueError):
                    parsed_value = None

            if parsed_value is None:
                self._log_debug("[NDS][INIT] ignored setting %s=%s", key, value)
                continue

            validated_config[key] = parsed_value

        if validated_config:
            self.GOLD_SETTINGS.update(validated_config)
            self._log_debug("[NDS][INIT] applied analyzer settings=%s", len(validated_config))

        if ignored_keys:
            self._log_debug("[NDS][INIT] ignored non-analysis keys=%s", sorted(set(ignored_keys)))

    def _apply_sessions_config(self, sessions_config: Dict[str, Any]) -> None:
        """اعمال تنظیمات سشن‌ها از config متمرکز"""
        for session_name, session_data in sessions_config.items():
            if not isinstance(session_data, dict):
                continue

            converted_session = {
                'start': session_data.get('start', 0),
                'end': session_data.get('end', 0),
                'weight': session_data.get('weight', 0.5)
            }
            standard_name = SESSION_MAPPING.get(session_name, session_name)

            if standard_name in self.TRADING_SESSIONS:
                self.TRADING_SESSIONS[standard_name].update(converted_session)
            else:
                self.TRADING_SESSIONS[standard_name] = converted_session

            self._log_debug("[NDS][SESSIONS] applied config %s=%s", standard_name, converted_session)
        normalized = normalize_session_definitions(self.TRADING_SESSIONS)
        if normalized:
            self.session_definitions = normalized

    def _log_debug(self, message: str, *args: Any) -> None:
        if self.debug_analyzer:
            logger.debug(message, *args)

    def _log_info(self, message: str, *args: Any) -> None:
        logger.info(message, *args)

    def _normalize_volatility_state(self, volatility_state: Optional[str]) -> str:
        if not volatility_state:
            return "MODERATE_VOLATILITY"
        state = str(volatility_state).upper()
        mapping = {
            "HIGH": "HIGH_VOLATILITY",
            "LOW": "LOW_VOLATILITY",
            "MODERATE": "MODERATE_VOLATILITY",
        }
        if state in mapping:
            return mapping[state]
        if state.endswith("_VOLATILITY"):
            return state
        return "MODERATE_VOLATILITY"

    def _append_reason(self, reasons: List[str], reason: str) -> None:
        reasons.append(reason)
        self._log_debug("[NDS][REASONS] %s", reason)

    def _validate_dataframe(self) -> None:
        """اعتبارسنجی DataFrame ورودی"""
        required_columns = ['time', 'open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df['time'] = pd.to_datetime(self.df['time'], errors="coerce")
        try:
            if getattr(self.df["time"].dt, "tz", None) is not None:
                self.df["time"] = self.df["time"].dt.tz_convert(None)
        except Exception:
            pass

        if len(self.df) > 1 and self.df['time'].iloc[0] > self.df['time'].iloc[-1]:
            logger.warning("[NDS][INIT] DataFrame not sorted chronologically. Sorting...")
            self.df = self.df.sort_values('time').reset_index(drop=True)

        if 'volume' not in self.df.columns:
            self.df['volume'] = 1.0

    def _detect_timeframe(self) -> str:
        """شناسایی تایم‌فریم با استفاده از میانه اختلاف زمانی"""
        default_tf = self.GOLD_SETTINGS.get('DEFAULT_TIMEFRAME', 'M15')

        if len(self.df) < 2:
            return default_tf

        deltas = self.df['time'].diff().dt.total_seconds().dropna()
        sample = deltas.head(200)
        if sample.empty:
            return default_tf

        median = float(sample.median())
        if median <= 0:
            return default_tf

        filtered = sample[sample <= median * 3]
        median = float(filtered.median()) if not filtered.empty else median

        tf_map = {
            'M1': 60,
            'M5': 300,
            'M15': 900,
            'M30': 1800,
            'H1': 3600,
            'H4': 14400,
            'D1': 86400,
        }
        closest_tf = None
        closest_diff = None
        for tf_name, seconds in tf_map.items():
            diff = abs(median - seconds)
            if closest_diff is None or diff < closest_diff:
                closest_tf = tf_name
                closest_diff = diff

        if closest_tf and closest_diff is not None and closest_diff <= tf_map[closest_tf] * 0.1:
            return closest_tf

        return default_tf

    def _apply_timeframe_settings(self) -> None:
        """اعمال تنظیمات اختصاصی تایم‌فریم"""
        tf_settings = self.timeframe_specifics.get(self.timeframe)
        if not tf_settings:
            self._log_debug("[NDS][INIT] missing TIMEFRAME_SPECIFICS timeframe=%s", self.timeframe)
            return
        self.GOLD_SETTINGS.update(tf_settings)

    def _analyze_trading_sessions(self, volume_analysis: Dict[str, Any]) -> SessionAnalysis:
        """تحلیل جامع سشن‌های معاملاتی

        سیاست جدید:
        - current_session/weight/activity فقط کیفیت هستند.
        - is_active_session فقط وقتی False می‌شود که واقعاً untradable باشیم (market closed, data ناقص, spread غیرعادی, ...).
        """
        last_time = self.df['time'].iloc[-1]
        parsed_time = parse_timestamp(last_time)
        broker_time = to_broker_time(parsed_time, self.broker_utc_offset, self.time_mode) if parsed_time else None

        session_decision = evaluate_session(broker_time or last_time, self.config)
        logger.info(
            "[NDS][TIME] mode=%s broker_offset=%s ts_raw=%s ts_broker=%s session=%s overlap=%s",
            self.time_mode,
            self.broker_utc_offset,
            last_time,
            broker_time,
            session_decision.session_name,
            session_decision.is_overlap,
        )
        self._log_info(
            "[NDS][SESSION_POLICY] session=%s tradable=%s weight=%.2f mode=%s reason=%s time_mode=%s offset=%.2f",
            session_decision.session_name,
            bool(session_decision.is_tradable),
            float(session_decision.weight),
            session_decision.policy_mode,
            session_decision.block_reason or "-",
            session_decision.time_mode,
            float(session_decision.broker_utc_offset_hours or 0.0),
        )

        # ---- RVOL (NaN-safe) ----
        rvol = volume_analysis.get('rvol', 1.0)
        try:
            rvol = float(rvol)
        except Exception:
            rvol = 1.0
        if pd.isna(rvol):
            rvol = 1.0

        volume_trend = str(volume_analysis.get('volume_trend', 'NEUTRAL') or 'NEUTRAL').upper()

        # ---- Activity (Quality only) ----
        if rvol > 1.2 or volume_trend == 'INCREASING':
            session_activity = 'HIGH'
        elif rvol < 0.8 and volume_trend == 'DECREASING':
            session_activity = 'LOW'
        else:
            session_activity = 'NORMAL'

        # ---- Determine "untradable" (Active/Inactive واقعی) ----
        # این کلیدها ممکن است در پروژه نباشند؛ اگر نباشند هیچ مشکلی نیست.
        market_status = str(volume_analysis.get("market_status", "") or "").upper()  # e.g. OPEN/CLOSED/HALTED
        data_ok = volume_analysis.get("data_ok", None)  # True/False if available

        spread_pips = volume_analysis.get("spread_pips", volume_analysis.get("spread", None))
        max_spread_pips = volume_analysis.get("max_spread_pips", volume_analysis.get("max_spread", None))

        untradable_reasons = []
        untradable = False

        # 1) invalid/parse failure from session decision (خیلی مهم)
        if not bool(session_decision.is_tradable):
            untradable = True
            untradable_reasons.append(session_decision.block_reason or "session_untradable")

        # 2) market status (optional)
        if market_status in ("CLOSED", "HALTED"):
            untradable = True
            untradable_reasons.append(f"market_status={market_status}")

        # 3) data ok flag (optional)
        if data_ok is False:
            untradable = True
            untradable_reasons.append("data_ok=False")

        # 4) spread sanity (optional)
        if spread_pips is not None and max_spread_pips is not None:
            try:
                if float(spread_pips) > float(max_spread_pips):
                    untradable = True
                    untradable_reasons.append(
                        f"spread_pips={float(spread_pips):.4f}>max={float(max_spread_pips):.4f}"
                    )
            except Exception:
                # اگر قابل تبدیل نبود، تصمیم‌گیری را به این معیار وابسته نکن
                pass

        is_active_session = (not untradable)

        analysis = SessionAnalysis(
            current_session=session_decision.session_name,
            session_weight=session_decision.weight,
            weight=session_decision.weight,
            gmt_hour=broker_time.hour if isinstance(broker_time, datetime) else 0,
            # سیاست جدید: active فقط برای untradable false می‌شود
            is_active_session=is_active_session,
            is_overlap=session_decision.is_overlap,
            session_activity=session_activity,
            optimal_trading=session_decision.weight >= 1.2,
            ts_broker=broker_time if isinstance(broker_time, datetime) else None,
            time_mode=self.time_mode,
            broker_utc_offset_hours=self.broker_utc_offset,
            policy_mode=session_decision.policy_mode,
            is_tradable=session_decision.is_tradable,
            block_reason=session_decision.block_reason,
            untradable=untradable,
            untradable_reasons=",".join(untradable_reasons) if untradable_reasons else None,
            session_decision=session_decision.to_payload(),
        )

        # لاگ ارتقا یافته: active و دلیل untradable هم چاپ می‌شود
        try:
            reasons_str = ",".join(untradable_reasons) if untradable_reasons else "-"
            self._log_info(
                "[NDS][SESSIONS] current=%s weight=%.2f activity=%s overlap=%s active=%s untradable=%s reasons=%s rvol=%.2f trend=%s",
                analysis.current_session,
                float(analysis.weight),
                analysis.session_activity,
                bool(analysis.is_overlap),
                bool(analysis.is_active_session),
                bool(untradable),
                reasons_str,
                float(rvol),
                volume_trend,
            )
        except Exception:
            # لاگ نباید تحلیل را fail کند
            pass

        return analysis


    def _hour_in_session(self, hour: int, start: int, end: int) -> bool:
        """بررسی ساعت داخل بازه [start, end) با پشتیبانی از عبور از نیمه‌شب."""
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end


    def _is_valid_trading_session(self, check_time: datetime) -> Dict[str, Any]:
        """بررسی سشن معاملاتی با پشتیبانی از نام‌های جدید و قدیم

        سیاست جدید:
        - is_valid دیگر به معنی "primary session" نیست.
        - is_valid فقط یعنی timestamp معتبر/قابل‌تحلیل است (نه تعطیلی بازار).
        - تشخیص untradable در _analyze_trading_sessions انجام می‌شود (با data_ok/spread/market_status).
        """
        decision = evaluate_session(check_time, self.config)
        return {
            'is_valid': bool(decision.is_tradable),
            'is_overlap': bool(decision.is_overlap),
            'session': decision.session_name,
            'weight': decision.weight,
            'hour': decision.ts_broker.hour if decision.ts_broker else 0,
            'optimal_trading': decision.weight >= 1.0,
        }


    def generate_trading_signal(
        self,
        timeframe: str = 'M15',
        entry_factor: float = 0.5,
        scalping_mode: bool = True,
    ) -> AnalysisResult:
        """
        تولید سیگنال نهایی با ادغام سیستم تشخیص ساختار الترا پرو و تاییدیه ADX/Volume
        نسخه تحلیل‌محور: خروجی فقط ایده معاملاتی (بدون منطق اجرا/ریسک).
        """
        mode = "Scalping" if scalping_mode else "Regular"
        self._log_info("[NDS][INIT] start analysis mode=%s analysis_only=true", mode)
        point_size, source = self._resolve_point_size()
        pip_size = self._resolve_pip_size(point_size)
        self._log_info(
            "[NDS][POINT_SIZE] point_size=%.4f source=%s",
            point_size,
            source,
        )
        self._log_info(
            "[NDS][PIP_SIZE] pip_size=%.4f",
            pip_size,
        )

        try:
            atr_window = self.GOLD_SETTINGS.get('ATR_WINDOW', 14)
            self.df, atr_v = IndicatorCalculator.calculate_atr(self.df, atr_window)
            self.atr = atr_v

            current_close = float(self.df['close'].iloc[-1])

            adx_window = self.GOLD_SETTINGS.get('ADX_WINDOW', 14)
            self.df, adx_v, plus_di, minus_di, di_trend = IndicatorCalculator.calculate_adx(self.df, adx_window)

            self._log_info(
                "[NDS][INDICATORS] atr=%.2f adx=%.2f plus_di=%.2f minus_di=%.2f di_trend=%s",
                atr_v,
                adx_v,
                plus_di,
                minus_di,
                di_trend,
            )

            if scalping_mode:
                atr_short_df, _ = IndicatorCalculator.calculate_atr(self.df.copy(), 7)
                atr_short_value = float(atr_short_df['atr_7'].iloc[-1])
            else:
                atr_short_value = None

            atr_for_scoring = atr_short_value if (scalping_mode and atr_short_value) else atr_v

            volume_analysis = IndicatorCalculator.analyze_volume(self.df, 5 if scalping_mode else 20)
            spread_payload = self._normalize_spread_from_df(point_size, pip_size)
            if spread_payload:
                volume_analysis.update(spread_payload)
            max_spread_pips = float(self.GOLD_SETTINGS.get("SPREAD_MAX_PIPS", 2.5))
            volume_analysis.setdefault("max_spread_pips", max_spread_pips)
            volume_analysis.setdefault("max_spread", max_spread_pips)
            volatility_state = self._normalize_volatility_state(self._determine_volatility(atr_v, atr_for_scoring))
            session_analysis = self._analyze_trading_sessions(volume_analysis)
            session_activity = str(
                getattr(session_analysis, "session_activity", None)
                or getattr(session_analysis, "activity", None)
                or "UNKNOWN"
            ).upper()
            low_liq_session = str(self.GOLD_SETTINGS.get("INTEGRITY_LOW_LIQUIDITY_SESSION_ACTIVITY", "LOW")).upper()
            low_liq_rvol_max = float(self.GOLD_SETTINGS.get("INTEGRITY_LOW_LIQUIDITY_RVOL_MAX", 0.55))
            low_liquidity = float(volume_analysis.get("rvol", 1.0)) <= low_liq_rvol_max and session_activity == low_liq_session

            last_candle = self.df.iloc[-1]
            self._log_debug(
                "[NDS][INDICATORS] last_candle time=%s open=%.2f high=%.2f low=%.2f close=%.2f rvol=%.2f",
                last_candle['time'],
                last_candle['open'],
                last_candle['high'],
                last_candle['low'],
                last_candle['close'],
                float(volume_analysis.get('rvol', 1.0)),
            )

            smc = SMCAnalyzer(self.df, self.atr, self.GOLD_SETTINGS)
            integrity_flags: List[str] = []
            if low_liquidity:
                integrity_flags.append("low_liquidity")
            min_smc_candles = int(self.GOLD_SETTINGS.get("SMC_MIN_CANDLES", 300))
            smc_ready = len(self.df) >= min_smc_candles
            if not smc_ready:
                integrity_flags.append(f"smc_skipped_candles<{min_smc_candles}")
                self._log_info(
                    "[NDS][INTEGRITY] SMC skipped (candles=%s min_required=%s)",
                    len(self.df),
                    min_smc_candles,
                )
                swings = []
                fvgs = []
                order_blocks = []
                sweeps = []
            else:
                swings = smc.detect_swings(timeframe)
                fvgs = smc.detect_fvgs()
                order_blocks = smc.detect_order_blocks()
                sweeps = smc.detect_liquidity_sweeps(swings)

            structure = smc.determine_market_structure(
                swings=swings,
                lookback_swings=4,
                volume_analysis=volume_analysis,
                volatility_state=volatility_state,
                adx_value=adx_v,
            )

            self._log_info("[NDS][SMC][STRUCTURE] %s", structure)
            try:
                breakers = getattr(structure, "breakers", []) or []
                ifvgs = getattr(structure, "inversion_fvgs", []) or []
                self._log_info(
                    "[NDS][FLOW_ZONES] detected breakers=%s ifvg=%s",
                    len(breakers),
                    len(ifvgs),
                )
            except Exception as _flow_e:
                self._log_debug("[NDS][FLOW_ZONES] detection log failed: %s", _flow_e)

            final_structure = self._apply_adx_override(structure, adx_v, plus_di, minus_di)

            score, reasons, score_breakdown = self._calculate_scoring_system(
                structure=final_structure,
                adx_value=adx_v,
                volume_analysis=volume_analysis,
                fvgs=fvgs,
                sweeps=sweeps,
                order_blocks=order_blocks,
                current_price=current_close,
                swings=swings,
                atr_value=atr_for_scoring,
                session_analysis=session_analysis,
            )

            confidence = self._calculate_confidence(
                score,
                volatility_state,
                session_analysis,
                volume_analysis,
                scalping_mode,
                sweeps=sweeps,
            )

            analysis_signal_context = self._build_signal_context(
                structure=final_structure,
                score=score,
                confidence=confidence,
                adx_value=adx_v,
                plus_di=plus_di,
                minus_di=minus_di,
                sweeps=sweeps,
                volatility_state=volatility_state,
                scalping_mode=scalping_mode,
            )

            self._log_info(
                "[NDS][BIAS] bias=%s trend=%s adx=%.2f di=%.2f/%.2f structure_score=%.1f",
                analysis_signal_context.get("bias"),
                analysis_signal_context.get("trend"),
                float(adx_v),
                float(plus_di),
                float(minus_di),
                float(analysis_signal_context.get("structure_score", 0.0)),
            )

            signal = self._determine_signal(
                score,
                confidence,
                volatility_state,
                scalping_mode,
                context=analysis_signal_context,
            )

            self._log_info(
                "[NDS][RESULT] score=%.1f confidence=%.1f signal=%s volatility=%s",
                score,
                confidence,
                signal,
                volatility_state,
            )

            result_payload = self._build_initial_result(
                signal=signal,
                confidence=confidence,
                score=score,
                reasons=reasons,
                structure=final_structure,
                atr_value=atr_v,
                atr_short_value=atr_short_value,
                adx_value=adx_v,
                plus_di=plus_di,
                minus_di=minus_di,
                volume_analysis=volume_analysis,
                recent_range=self._calculate_recent_range(scalping_mode),
                recent_position=self._calculate_recent_position(current_close, scalping_mode),
                volatility_state=volatility_state,
                session_analysis=session_analysis,
                current_price=current_close,
                timeframe=timeframe,
                score_breakdown=score_breakdown,
                scalping_mode=scalping_mode,
            )

            try:
                if "context" not in result_payload or result_payload.get("context") is None:
                    result_payload["context"] = {}
                result_payload["context"]["analysis_signal_context"] = analysis_signal_context
            except Exception as _ctx_e:
                self._log_debug("[NDS][SIGNAL][CONTEXT] failed to attach signal context: %s", _ctx_e)

            result_payload["analysis_signal_context"] = analysis_signal_context

            try:
                if "analysis_data" in result_payload and isinstance(result_payload["analysis_data"], dict):
                    result_payload["analysis_data"]["flow_zones"] = {
                        "breakers": getattr(final_structure, "breakers", []) or [],
                        "inversion_fvgs": getattr(final_structure, "inversion_fvgs", []) or [],
                    }
            except Exception as _flow_e:
                self._log_debug("[NDS][FLOW_ZONES] failed to attach flow zones: %s", _flow_e)

            pre_filter_signal = result_payload.get("signal")
            result_payload = self._apply_final_filters(result_payload, scalping_mode)
            post_filter_signal = result_payload.get("signal")
            signal = result_payload.get('signal')
            reasons = result_payload.get('reasons', reasons)

            ct_blocked = False
            ct_reason = "-"
            if signal in {"BUY", "SELL"} and analysis_signal_context:
                bias = analysis_signal_context.get("bias")
                strong_trend = bool(analysis_signal_context.get("strong_trend"))
                reversal_ok = bool(analysis_signal_context.get("reversal_ok"))
                counter_trend = (
                    (bias == "BULLISH" and signal == "SELL")
                    or (bias == "BEARISH" and signal == "BUY")
                )
                if counter_trend and strong_trend and not reversal_ok:
                    ct_blocked = True
                    ct_reason = "counter_trend_unconfirmed"
                    self._append_reason(
                        reasons,
                        f"Counter-trend blocked final gate (bias={bias}, strong={strong_trend}, reversal_ok={reversal_ok})",
                    )
                    signal = "NONE"
                    result_payload["signal"] = "NONE"

                self._log_info(
                    "[NDS][CTGATE] signal=%s bias=%s allow=%s reason=%s conf=%.1f score=%.1f",
                    signal,
                    bias,
                    not ct_blocked,
                    ct_reason,
                    confidence,
                    score,
                )

            self._log_info(
                "[NDS][FILTER_SUMMARY] pre=%s post=%s ct_blocked=%s reason=%s",
                pre_filter_signal,
                post_filter_signal,
                ct_blocked,
                ct_reason,
            )

            entry_price = None
            stop_loss = None
            take_profit = None
            entry_type = "NONE"
            entry_model = "NONE"
            entry_source = None
            entry_context = {}
            entry_idea: Dict[str, Any] = {}

            entry_idea = self.select_entry_idea(
                df=self.df,
                structure=final_structure,
                market_metrics={
                    "atr": atr_v,
                    "atr_short": atr_short_value,
                    "adx": adx_v,
                    "plus_di": plus_di,
                    "minus_di": minus_di,
                    "current_price": current_close,
                    "signal": signal,
                },
                session_analysis=session_analysis,
                signal_context=analysis_signal_context,
                volume_analysis=volume_analysis,
                scalping_mode=scalping_mode,
                entry_factor=entry_factor,
                fvgs=fvgs,
                order_blocks=order_blocks,
            )

            if low_liquidity and entry_idea.get("signal") in {"BUY", "SELL"}:
                exceptional_threshold = float(self.GOLD_SETTINGS.get("INTEGRITY_EXCEPTIONAL_SETUP_SCORE", 0.78))
                force_none = bool(self.GOLD_SETTINGS.get("INTEGRITY_LOW_LIQUIDITY_FORCE_NONE", True))
                setup_score = None
                if isinstance(entry_idea.get("zone"), dict):
                    setup_score = entry_idea["zone"].get("setup_score")
                if setup_score is None and isinstance(entry_idea.get("metrics"), dict):
                    setup_score = entry_idea["metrics"].get("setup_score")
                setup_score = float(setup_score or 0.0)
                if force_none and setup_score < exceptional_threshold:
                    entry_idea["blocked_setup"] = entry_idea.get("zone")
                    entry_idea["signal"] = "NONE"
                    entry_idea["entry_level"] = None
                    entry_idea["entry_type"] = "NONE"
                    entry_idea["entry_model"] = "NONE"
                    entry_idea["reject_reason"] = "INTEGRITY_LOW_LIQUIDITY"
                    self._append_reason(
                        reasons,
                        f"Low liquidity gate (rvol={float(volume_analysis.get('rvol', 0.0)):.2f}, session_activity={session_activity})",
                    )
                else:
                    confidence = round(float(confidence) * 0.7, 1)
                    self._append_reason(
                        reasons,
                        f"Low liquidity penalty (setup_score={setup_score:.2f} >= {exceptional_threshold:.2f})",
                    )
            result_payload["confidence"] = confidence

            entry_price = entry_idea.get("entry_level")
            entry_type = entry_idea.get("entry_type", "NONE")
            entry_model = entry_idea.get("entry_model", "NONE")
            entry_source = entry_idea.get("zone")
            entry_context = entry_idea.get("metrics", {}) or {}
            entry_reason = entry_idea.get("reason")
            if entry_reason:
                reasons.append(entry_reason)

            signal = entry_idea.get("signal", "NONE") or "NONE"
            result_payload["signal"] = signal
            result_payload["entry_reason"] = entry_reason
            result_payload["entry_level"] = entry_price
            result_payload["entry_price"] = entry_price

            if signal in {"BUY", "SELL"} and entry_price is not None:
                tp1_target = self._resolve_opposing_structure_target(
                    signal=signal,
                    entry_price=float(entry_price),
                    fvgs=fvgs,
                    order_blocks=order_blocks,
                )
                entry_context = dict(entry_context)
                entry_context["tp1_target_price"] = tp1_target.get("price")
                entry_context["tp1_target_source"] = tp1_target.get("source")
                entry_context["tp1_target_reason"] = tp1_target.get("reason")

            entry_signal_context = {
                "signal": signal,
                "entry_price": entry_price,
                "entry_level": entry_price,
                "stop_loss": None,
                "take_profit": None,
                "take_profit2": None,
                "entry_type": entry_type,
                "entry_source": entry_source,
                "entry_reason": entry_reason,
                "entry_model": entry_model,
                "tier": entry_idea.get("tier", "NONE"),
                "metrics": entry_context,
            }

            try:
                if "context" not in result_payload or result_payload.get("context") is None:
                    result_payload["context"] = {}
                result_payload["entry_type"] = entry_type
                result_payload["entry_model"] = entry_model
                result_payload["entry_idea"] = entry_idea
                result_payload["entry_source"] = entry_source
                result_payload["entry_context"] = entry_context
                result_payload["context"]["entry_type"] = entry_type
                result_payload["context"]["entry_model"] = entry_model
                result_payload["context"]["entry_idea"] = entry_idea
                result_payload["context"]["entry_source"] = entry_source
                result_payload["context"]["entry_context"] = entry_context
                result_payload["signal_context"] = entry_signal_context
                result_payload["context"]["signal_context"] = entry_signal_context
            except Exception as _ctx_e:
                self._log_debug("[NDS][SIGNAL][CONTEXT] failed to attach flow entry context: %s", _ctx_e)

            analysis_trace = self._build_analysis_trace(
                signal=signal,
                confidence=confidence,
                score=score,
                volume_analysis=volume_analysis,
                session_analysis=session_analysis,
                signal_context=analysis_signal_context,
                entry_idea=entry_idea,
                reasons=reasons,
                integrity_flags=integrity_flags,
            )
            result_payload["analysis_trace"] = analysis_trace
            result_payload["context"]["analysis_trace"] = analysis_trace

            top_setup = analysis_trace.get("setup", {})
            gates = analysis_trace.get("gates", {})
            reasons_str = ",".join(analysis_trace.get("decision_notes", [])[:6])
            self._log_info(
                "[NDS][SUMMARY] signal=%s score=%.1f conf=%.1f gates=%s top_setup=%s reasons=[%s]",
                signal,
                float(score),
                float(confidence),
                gates,
                {
                    "type": top_setup.get("selected_zone_type"),
                    "id": top_setup.get("selected_zone_id"),
                    "retest": top_setup.get("retest_reason"),
                    "touches": top_setup.get("touches"),
                    "dist_atr": top_setup.get("distance_to_zone_atr"),
                },
                reasons_str,
            )

            return self._build_analysis_result(
                signal=signal,
                confidence=confidence,
                score=score,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasons=reasons,
                context=result_payload,
                timeframe=timeframe,
                current_price=current_close,
            )

        except Exception as e:
            logger.error("[NDS][RESULT] analysis failed: %s", str(e), exc_info=True)
            return self._create_error_result(str(e), timeframe, current_close=None)


    def _calculate_recent_range(self, scalping_mode: bool) -> float:
        """محاسبه محدوده قیمت اخیر"""
        lookback_bars = 20 if scalping_mode else 96
        recent_high = float(self.df['high'].tail(lookback_bars).max())
        recent_low = float(self.df['low'].tail(lookback_bars).min())
        return recent_high - recent_low

    def _calculate_recent_position(self, current_price: float, scalping_mode: bool) -> float:
        """محاسبه موقعیت قیمت در محدوده اخیر"""
        lookback_bars = 20 if scalping_mode else 96
        recent_high = float(self.df['high'].tail(lookback_bars).max())
        recent_low = float(self.df['low'].tail(lookback_bars).min())
        recent_range = recent_high - recent_low
        return (current_price - recent_low) / recent_range if recent_range > 0 else 0.5

    def _resolve_opposing_structure_target(
        self,
        signal: str,
        entry_price: float,
        fvgs: List[FVG],
        order_blocks: List[OrderBlock],
    ) -> Dict[str, Any]:
        """Find nearest opposing structure level for dynamic TP1 planning."""
        if signal not in {"BUY", "SELL"}:
            return {"price": None, "source": None, "reason": "no_signal"}
        if entry_price is None:
            return {"price": None, "source": None, "reason": "no_entry"}

        candidates: List[Tuple[float, str]] = []

        for fvg in fvgs:
            if getattr(fvg, "filled", False) or getattr(fvg, "stale", False):
                continue
            if signal == "BUY" and fvg.type == FVGType.BEARISH:
                target = float(fvg.bottom)
                if target > entry_price:
                    candidates.append((target, "BEARISH_FVG"))
            elif signal == "SELL" and fvg.type == FVGType.BULLISH:
                target = float(fvg.top)
                if target < entry_price:
                    candidates.append((target, "BULLISH_FVG"))

        for ob in order_blocks:
            if getattr(ob, "stale", False):
                continue
            if signal == "BUY" and ob.type == "BEARISH_OB":
                target = float(ob.low)
                if target > entry_price:
                    candidates.append((target, "BEARISH_OB"))
            elif signal == "SELL" and ob.type == "BULLISH_OB":
                target = float(ob.high)
                if target < entry_price:
                    candidates.append((target, "BULLISH_OB"))

        if not candidates:
            return {"price": None, "source": None, "reason": "no_opposing_structure"}

        if signal == "BUY":
            target_price, source = min(candidates, key=lambda c: c[0])
        else:
            target_price, source = max(candidates, key=lambda c: c[0])

        return {"price": float(target_price), "source": source, "reason": "nearest_opposing_structure"}

    def _score_flow_setup(
        self,
        zone: Dict[str, Any],
        dist_atr: float,
        max_dist_atr: float,
        signal: str,
        session_analysis: SessionAnalysis,
        volume_analysis: Dict[str, Any],
        signal_context: Dict[str, Any],
    ) -> Dict[str, float]:
        settings = self.GOLD_SETTINGS
        weights = settings.get("FLOW_SETUP_WEIGHTS") or {
            "retest_quality": 0.25,
            "freshness": 0.2,
            "proximity": 0.2,
            "displacement": 0.15,
            "trend_alignment": 0.1,
            "liquidity": 0.1,
        }

        retest_reason = str(zone.get("retest_reason", "") or "").upper()
        retest_quality_map = {
            "CLOSE_RECLAIM": 1.0,
            "CLOSE_REJECT": 1.0,
            "MID_TOUCH_DISPLACEMENT": 0.9,
            "WICK_REJECTION": 0.8,
            "NO_CONFIRMED_TOUCH": 0.45,
            "FIRST_TOUCH_UNCONFIRMED": 0.35,
            "CONSUMED_AFTER_FIRST_TOUCH": 0.25,
        }
        retest_quality = retest_quality_map.get(retest_reason, 0.5)

        touch_count = int(zone.get("touch_count", 1))
        freshness_map = {1: 1.0, 2: 0.75, 3: 0.55}
        freshness = freshness_map.get(touch_count, 0.35)

        proximity = 1.0 if dist_atr <= 0 else max(0.0, 1.0 - (dist_atr / max(max_dist_atr, 0.01)))
        disp_target = float(settings.get("FLOW_SETUP_DISPLACEMENT_ATR_TARGET", 1.0))
        displacement = min(1.0, float(zone.get("disp_atr", 0.0)) / max(disp_target, 0.01))

        bias = str(signal_context.get("bias", "") or "")
        trend_alignment = 0.7
        if bias in {"BULLISH", "BEARISH"}:
            trend_alignment = 1.0 if ((bias == "BULLISH" and signal == "BUY") or (bias == "BEARISH" and signal == "SELL")) else 0.4

        session_weight = float(getattr(session_analysis, "session_weight", 1.0))
        rvol = float(volume_analysis.get("rvol", 1.0))
        liquidity = min(1.0, (0.6 * min(1.0, rvol / 1.0)) + (0.4 * min(1.0, session_weight / 1.2)))

        total_weight = sum(float(w) for w in weights.values()) or 1.0
        setup_score = (
            float(weights.get("retest_quality", 0.0)) * retest_quality
            + float(weights.get("freshness", 0.0)) * freshness
            + float(weights.get("proximity", 0.0)) * proximity
            + float(weights.get("displacement", 0.0)) * displacement
            + float(weights.get("trend_alignment", 0.0)) * trend_alignment
            + float(weights.get("liquidity", 0.0)) * liquidity
        ) / total_weight

        return {
            "retest_quality": retest_quality,
            "freshness": freshness,
            "proximity": proximity,
            "displacement": displacement,
            "trend_alignment": trend_alignment,
            "liquidity": liquidity,
            "setup_score": setup_score,
        }

    def _normalize_structure_score(self, raw_score: Optional[float]) -> float:
        """نرمال‌سازی structure_score به بازه 0..100"""
        if raw_score is None:
            return 0.0
        score = float(raw_score)
        normalized = score
        if score <= 1.05:
            normalized = score * 100
        elif score <= 10:
            normalized = score * 10
        else:
            normalized = score
        normalized = max(0.0, min(100.0, float(normalized)))
        self._log_debug(
            "[NDS][SCORE] structure_score raw=%.2f normalized=%.2f",
            score,
            normalized,
        )
        return normalized

    def _bounded(self, value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))

    def _min_stop_distance(self, atr_value: float) -> float:
        """Minimum logical SL distance from entry (in price units, e.g., USD for XAUUSD).

        This is a safety guardrail to prevent inverted/degenerate SL placement due to
        swing-anchor selection or noisy structure levels.
        """
        try:
            k = float(self.GOLD_SETTINGS.get("MIN_STOP_ATR_MULT", 0.35))
        except Exception:
            k = 0.35
        return max(0.01, float(atr_value) * k)

    def _percentile(self, data: List[float], pct: float) -> float:
        """Compute percentile with linear interpolation."""
        if not data:
            return 50.0
        pct = max(0.0, min(1.0, float(pct)))
        values = sorted(float(v) for v in data)
        if len(values) == 1:
            return values[0]
        idx = pct * (len(values) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return values[lo]
        frac = idx - lo
        return (values[lo] * (1.0 - frac)) + (values[hi] * frac)

    def _resolve_point_size(self) -> Tuple[float, str]:
        default_point_size = self.GOLD_SETTINGS.get("POINT_SIZE")
        point_size, source = resolve_point_size_with_source(self.config, default=default_point_size)
        self._point_size = point_size
        self._point_size_source = source
        return point_size, source

    def _resolve_pip_size(self, point_size: float) -> float:
        pip_size = resolve_pip_size_from_config(self.config, point_size, default=None)
        self._pip_size = pip_size
        return pip_size

    def _normalize_spread_from_df(self, point_size: float, pip_size: float) -> Dict[str, Optional[float]]:
        raw_spread = None
        raw_unit = None
        if "spread_price" in self.df.columns:
            raw_spread = self.df["spread_price"].iloc[-1]
            raw_unit = "price"
        elif "spread_pips" in self.df.columns:
            raw_spread = self.df["spread_pips"].iloc[-1]
            raw_unit = "pips"
        elif "spread_points" in self.df.columns:
            raw_spread = self.df["spread_points"].iloc[-1]
            raw_unit = "points"
        elif "spread" in self.df.columns:
            raw_spread = self.df["spread"].iloc[-1]
            raw_unit = "points"

        if raw_spread is None:
            return {}

        normalized = normalize_spread(raw_spread, point_size, pip_size, raw_unit or "price")
        if normalized.get("spread_price") is None:
            return {}

        max_spread_pips = float(self.GOLD_SETTINGS.get("SPREAD_MAX_PIPS", 2.5))
        self._log_info(
            "[NDS][SPREAD][NORMALIZE] raw_spread=%s raw_unit=%s spread_price=%.4f spread_points=%.2f "
            "spread_pips=%.2f point_size=%.4f pip_size=%.4f max_spread_pips=%.2f",
            normalized.get("raw_spread"),
            normalized.get("raw_unit"),
            float(normalized.get("spread_price") or 0.0),
            float(normalized.get("spread_points") or 0.0),
            float(normalized.get("spread_pips") or 0.0),
            float(normalized.get("point_size") or 0.0),
            float(normalized.get("pip_size") or 0.0),
            max_spread_pips,
        )

        return {
            "spread_price": normalized.get("spread_price"),
            "spread_points": normalized.get("spread_points"),
            "spread_pips": normalized.get("spread_pips"),
            "spread": normalized.get("spread_pips"),
            "spread_raw": normalized.get("raw_spread"),
            "spread_raw_unit": normalized.get("raw_unit"),
        }

    def _get_point_size(self) -> float:
        """Return configured point size with default."""
        if self._point_size is not None:
            return self._point_size
        point_size, source = self._resolve_point_size()
        self._log_debug(
            "[NDS][POINT_SIZE] point_size=%.4f source=%s",
            point_size,
            source,
        )
        return point_size

    def _normalize_session_name(self, session_name: str) -> str:
        if not session_name:
            return "OTHER"
        raw = str(session_name).upper()
        alias_map = {
            "OVERLAP_PEAK": "OVERLAP",
            "LONDON_NY_OVERLAP": "OVERLAP",
            "NY_OVERLAP": "OVERLAP",
            "NEWYORK": "NEW_YORK",
            "NEWYORK_SESSION": "NEW_YORK",
        }
        if raw in alias_map:
            return alias_map[raw]
        normalized = SESSION_MAPPING.get(raw, raw)
        if normalized == "OVERLAP_PEAK":
            normalized = "OVERLAP"
        return normalized

    def _is_time_in_window(self, ts: datetime, start_str: str, end_str: str) -> Tuple[bool, Optional[str]]:
        """Check if timestamp is within [start, end) window (supports midnight wrap)."""
        broker_time = to_broker_time(parse_timestamp(ts) or ts, self.broker_utc_offset, self.time_mode)
        return in_time_window(broker_time, start_str, end_str)

    def _compute_scalping_sl_tp(
        self,
        signal: str,
        entry_price: float,
        atr_value: Optional[float],
    ) -> Dict[str, Any]:
        """Compute scalping SL/TP using ATR and recent candle references."""
        settings = self.GOLD_SETTINGS
        point_size = self._get_point_size()

        atr_mult = float(settings.get("SCALP_ATR_SL_MULT", 1.5))
        sl_min_pips = float(settings.get("SL_MIN_PIPS", 10.0))
        min_sl_pips = float(settings.get("MIN_SL_PIPS", sl_min_pips))
        sl_min_pips = max(sl_min_pips, min_sl_pips)
        sl_max_pips = float(settings.get("SL_MAX_PIPS", 40.0))
        tp1_pips = float(settings.get("TP1_PIPS", 35.0))
        tp2_enabled = bool(settings.get("TP2_ENABLED", True))
        tp2_pips = float(settings.get("TP2_PIPS", tp1_pips * 2.0))

        recent_slice = self.df.tail(2)
        recent_low = float(recent_slice["low"].min()) if not recent_slice.empty else None
        recent_high = float(recent_slice["high"].max()) if not recent_slice.empty else None

        atr_value = float(atr_value) if atr_value is not None else 0.0
        atr_distance = atr_value * atr_mult if atr_value > 0 else 0.0

        ref_distance = 0.0
        if signal == "BUY" and recent_low is not None:
            ref_distance = max(0.0, float(entry_price) - float(recent_low))
        elif signal == "SELL" and recent_high is not None:
            ref_distance = max(0.0, float(recent_high) - float(entry_price))

        if atr_distance > 0 and ref_distance > 0:
            sl_distance = min(atr_distance, ref_distance)
        else:
            sl_distance = max(atr_distance, ref_distance)

        if sl_distance <= 0:
            sl_distance = pips_to_price(sl_min_pips, point_size)

        sl_metrics = calculate_distance_metrics(
            entry_price=float(entry_price),
            current_price=float(entry_price) + float(sl_distance),
            point_size=point_size,
        )
        sl_pips = float(sl_metrics.get("dist_pips") or 0.0)
        raw_sl_pips = sl_pips
        clamp_reason = None

        if sl_pips < sl_min_pips:
            sl_pips = sl_min_pips
            sl_distance = pips_to_price(sl_pips, point_size)
            clamp_reason = "min"
        elif sl_pips > sl_max_pips:
            sl_pips = sl_max_pips
            sl_distance = pips_to_price(sl_pips, point_size)
            clamp_reason = "max"

        if clamp_reason:
            self._log_info(
                "[NDS][SL_CLAMP] reason=%s raw=%.2f clamped=%.2f bounds=[%.2f,%.2f] point_size=%.4f",
                clamp_reason,
                raw_sl_pips,
                sl_pips,
                sl_min_pips,
                sl_max_pips,
                point_size,
            )

        if signal == "BUY":
            stop_loss = float(entry_price) - sl_distance
            take_profit = float(entry_price) + pips_to_price(tp1_pips, point_size)
            tp2_price = (
                float(entry_price) + pips_to_price(tp2_pips, point_size)
                if tp2_enabled
                else None
            )
        else:
            stop_loss = float(entry_price) + sl_distance
            take_profit = float(entry_price) - pips_to_price(tp1_pips, point_size)
            tp2_price = (
                float(entry_price) - pips_to_price(tp2_pips, point_size)
                if tp2_enabled
                else None
            )

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
            "recent_low": recent_low,
            "recent_high": recent_high,
        }

    def select_entry_idea(
        self,
        df: pd.DataFrame,
        structure: MarketStructure,
        market_metrics: Dict[str, Any],
        session_analysis: SessionAnalysis,
        signal_context: Dict[str, Any],
        volume_analysis: Dict[str, Any],
        scalping_mode: bool,
        entry_factor: float,
        fvgs: List[FVG],
        order_blocks: List[OrderBlock],
    ) -> Dict[str, Any]:
        """Single authoritative entry idea selector for Flow-tier logic."""
        base_signal = str(market_metrics.get("signal", "NONE") or "NONE")
        if base_signal not in {"BUY", "SELL"}:
            override_entry = self._select_flow_entry_override(
                structure=structure,
                market_metrics=market_metrics,
                session_analysis=session_analysis,
                signal_context=signal_context,
                volume_analysis=volume_analysis,
                scalping_mode=scalping_mode,
            )
            if override_entry.get("signal") in {"BUY", "SELL"}:
                return override_entry
            self._log_info(
                "[NDS][FLOW_DECISION] tier=NONE type=NONE allowed=false reject=BASE_SIGNAL_NONE reason=no_base_signal",
            )
            return {
                "signal": "NONE",
                "tier": "NONE",
                "entry_type": "NONE",
                "entry_model": "NONE",
                "entry_level": None,
                "zone": None,
                "confidence": 0.0,
                "reason": "no_base_signal",
                "reject_reason": "BASE_SIGNAL_NONE",
                "metrics": {},
            }

        flow_entry = self._select_flow_entry(
            signal=base_signal,
            structure=structure,
            current_price=float(market_metrics.get("current_price", 0.0) or 0.0),
            atr_value=float(market_metrics.get("atr_short") or market_metrics.get("atr") or 0.0),
            adx_value=float(market_metrics.get("adx") or 0.0),
            session_analysis=session_analysis,
            volume_analysis=volume_analysis,
            scalping_mode=scalping_mode,
            signal_context=signal_context,
        )
        if flow_entry.get("signal") in {"BUY", "SELL"}:
            return flow_entry

        allow_legacy = bool(self.GOLD_SETTINGS.get("ENABLE_LEGACY_SWING_ENTRY", False))
        if not allow_legacy:
            return flow_entry

        legacy_entry = self._select_legacy_entry_idea(
            signal=base_signal,
            fvgs=fvgs,
            order_blocks=order_blocks,
            structure=structure,
            atr_value=float(market_metrics.get("atr_short") or market_metrics.get("atr") or 0.0),
            entry_factor=entry_factor,
            current_price=float(market_metrics.get("current_price", 0.0) or 0.0),
            adx_value=float(market_metrics.get("adx") or 0.0),
        )
        if legacy_entry.get("signal") in {"BUY", "SELL"}:
            return legacy_entry

        return flow_entry

    def _select_flow_entry_override(
        self,
        structure: MarketStructure,
        market_metrics: Dict[str, Any],
        session_analysis: SessionAnalysis,
        signal_context: Dict[str, Any],
        volume_analysis: Dict[str, Any],
        scalping_mode: bool,
    ) -> Dict[str, Any]:
        """Allow high-quality flow setups to override a neutral base score."""
        settings = self.GOLD_SETTINGS
        current_price = float(market_metrics.get("current_price", 0.0) or 0.0)
        atr_value = float(market_metrics.get("atr_short") or market_metrics.get("atr") or 0.0)
        adx_value = float(market_metrics.get("adx") or 0.0)

        min_setup = float(settings.get("FLOW_OVERRIDE_MIN_SETUP_SCORE", 0.7))
        min_conf = float(settings.get("FLOW_OVERRIDE_MIN_CONFIDENCE", 0.6))
        max_touches = int(settings.get("FLOW_OVERRIDE_MAX_TOUCHES", settings.get("FLOW_MAX_TOUCHES", 5)))
        ct_bonus = float(settings.get("FLOW_OVERRIDE_COUNTERTREND_SCORE_BONUS", 0.08))
        allowed_retests = {
            "CLOSE_RECLAIM",
            "CLOSE_REJECT",
            "WICK_REJECTION",
            "MID_TOUCH_DISPLACEMENT",
        }

        candidates = []
        for side in ("BUY", "SELL"):
            entry = self._select_flow_entry(
                signal=side,
                structure=structure,
                current_price=current_price,
                atr_value=atr_value,
                adx_value=adx_value,
                session_analysis=session_analysis,
                volume_analysis=volume_analysis,
                scalping_mode=scalping_mode,
                signal_context=signal_context,
            )
            if entry.get("signal") not in {"BUY", "SELL"}:
                continue

            zone = entry.get("zone") if isinstance(entry, dict) else None
            if not isinstance(zone, dict):
                continue

            retest_reason = str(zone.get("retest_reason") or "").upper()
            touch_count = int(zone.get("touch_count", 0))
            setup_score = zone.get("setup_score")
            if setup_score is None and isinstance(entry.get("metrics"), dict):
                setup_score = entry["metrics"].get("setup_score")
            setup_score = float(setup_score or 0.0)
            entry_conf = float(entry.get("confidence", 0.0) or 0.0)

            if retest_reason and retest_reason not in allowed_retests:
                continue
            if touch_count > max_touches:
                continue

            bias = str(signal_context.get("bias", "") or "")
            strong_trend = bool(signal_context.get("strong_trend"))
            reversal_ok = bool(signal_context.get("reversal_ok"))
            counter_trend = (bias == "BULLISH" and side == "SELL") or (bias == "BEARISH" and side == "BUY")
            required_setup = min_setup + (ct_bonus if (counter_trend and strong_trend and not reversal_ok) else 0.0)

            if setup_score < required_setup or entry_conf < min_conf:
                continue

            entry.setdefault("metrics", {})
            entry["metrics"]["override"] = True
            entry["metrics"]["override_reason"] = "flow_override"
            entry["reason"] = f"{entry.get('reason', 'flow_override')} | override"
            candidates.append(entry)

        if not candidates:
            return {
                "signal": "NONE",
                "tier": "NONE",
                "entry_type": "NONE",
                "entry_model": "NONE",
                "entry_level": None,
                "zone": None,
                "confidence": 0.0,
                "reason": "flow_override_no_match",
                "reject_reason": "BASE_SIGNAL_NONE",
                "metrics": {},
            }

        candidates.sort(
            key=lambda e: (
                -float((e.get("zone") or {}).get("setup_score", 0.0) or 0.0),
                -float(e.get("confidence", 0.0) or 0.0),
            )
        )
        best = candidates[0]
        self._log_info(
            "[NDS][FLOW_OVERRIDE] signal=%s setup_score=%.2f conf=%.2f reason=%s",
            best.get("signal"),
            float((best.get("zone") or {}).get("setup_score", 0.0) or 0.0),
            float(best.get("confidence", 0.0) or 0.0),
            best.get("reason"),
        )
        return best

    def _select_legacy_entry_idea(
        self,
        signal: str,
        fvgs: List[FVG],
        order_blocks: List[OrderBlock],
        structure: MarketStructure,
        atr_value: float,
        entry_factor: float,
        current_price: float,
        adx_value: float,
    ) -> Dict[str, Any]:
        """Legacy swing fallback, disabled by default."""
        idea = self._build_entry_idea(
            signal=signal,
            fvgs=fvgs,
            order_blocks=order_blocks,
            structure=structure,
            atr_value=atr_value,
            entry_factor=entry_factor,
            current_price=current_price,
            adx_value=adx_value,
        )
        entry_level = idea.get("entry_level")
        if entry_level is None:
            return {
                "signal": "NONE",
                "tier": "D",
                "entry_type": "LEGACY",
                "entry_model": "NONE",
                "entry_level": None,
                "zone": idea.get("zone_meta"),
                "confidence": float(idea.get("confidence", 0.0) or 0.0),
                "reason": "legacy_no_entry",
                "reject_reason": "LEGACY_NO_ENTRY",
                "metrics": idea.get("metrics", {}) or {},
            }

        self._log_info(
            "[NDS][FLOW_DECISION] tier=D type=LEGACY model=%s signal=%s allowed=true reason=%s conf=%.2f",
            idea.get("entry_model", "MARKET"),
            signal,
            idea.get("reason", "-"),
            float(idea.get("confidence", 0.0) or 0.0),
        )
        return {
            "signal": signal,
            "tier": "D",
            "entry_type": "LEGACY",
            "entry_model": idea.get("entry_model", "MARKET"),
            "entry_level": entry_level,
            "zone": idea.get("zone_meta"),
            "confidence": float(idea.get("confidence", 0.0) or 0.0),
            "reason": idea.get("reason", "legacy_entry"),
            "reject_reason": None,
            "metrics": idea.get("metrics", {}) or {},
        }

    def _select_flow_entry(
        self,
        signal: str,
        structure: MarketStructure,
        current_price: float,
        atr_value: float,
        adx_value: float,
        session_analysis: SessionAnalysis,
        volume_analysis: Dict[str, Any],
        scalping_mode: bool,
        signal_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Select entry tier (Breaker/IFVG/Momentum) with strict hierarchy."""
        if signal not in {"BUY", "SELL"}:
            return {
                "signal": "NONE",
                "tier": "NONE",
                "entry_type": "NONE",
                "entry_model": "NONE",
                "entry_level": None,
                "zone": None,
                "confidence": 0.0,
                "reason": "signal=NONE",
                "reject_reason": "BASE_SIGNAL_NONE",
                "metrics": {},
            }

        settings = self.GOLD_SETTINGS
        point_size = self._get_point_size()
        entry_context: Dict[str, Any] = {}

        recent_slice = self.df.tail(2)
        recent_low = float(recent_slice["low"].min()) if not recent_slice.empty else None
        recent_high = float(recent_slice["high"].max()) if not recent_slice.empty else None

        spread_price = volume_analysis.get("spread_price")
        spread_buffer_price = 0.0
        if spread_price is not None:
            try:
                spread_buffer_price = float(spread_price) * 1.2
            except Exception:
                spread_buffer_price = 0.0
        buffer_atr = float(settings.get("FLOW_STOP_BUFFER_ATR", 0.05))
        atr_buffer = float(atr_value) * buffer_atr if atr_value else 0.0
        buffer_price = max(spread_buffer_price, atr_buffer)

        breakers = getattr(structure, "breakers", []) or []
        ifvgs = getattr(structure, "inversion_fvgs", []) or []

        rejection_counts = {
            "too_far": 0,
            "too_old": 0,
            "too_many_touches": 0,
            "stale": 0,
            "ineligible": 0,
        }
        retest_rejection_counts = {
            "TOO_MANY_TOUCHES": 0,
            "FIRST_TOUCH_UNCONFIRMED": 0,
            "NO_CONFIRMED_TOUCH": 0,
            "CONSUMED_AFTER_FIRST_TOUCH": 0,
        }

        def _tier_candidates(zones: List[Dict[str, Any]], side: str, max_dist_atr: float, max_age: int) -> List[Dict[str, Any]]:
            candidates = []
            max_touches = int(settings.get("FLOW_MAX_TOUCHES", 2))
            nearest_limit = int(settings.get("FLOW_NEAREST_ZONES", 5))
            pre_candidates: List[Dict[str, Any]] = []
            for zone in zones:
                retest_reason = str(zone.get("retest_reason") or "").upper()
                if not bool(zone.get("eligible", True)):
                    if retest_reason in retest_rejection_counts:
                        retest_rejection_counts[retest_reason] += 1
                    if retest_reason == "TOO_MANY_TOUCHES":
                        rejection_counts["too_many_touches"] += 1
                    else:
                        rejection_counts["ineligible"] += 1
                    self._log_info(
                        "[NDS][FLOW][ZONE_REJECT] tier=FLOW type=%s reason=INELIGIBLE retest_reason=%s idx=%s touches=%s",
                        zone.get("type"),
                        retest_reason,
                        zone.get("retest_index") or zone.get("index"),
                        zone.get("touch_count"),
                    )
                    continue
                z_type = str(zone.get("type", ""))
                if side == "BUY" and "BULLISH" not in z_type:
                    continue
                if side == "SELL" and "BEARISH" not in z_type:
                    continue
                if zone.get("stale"):
                    rejection_counts["stale"] += 1
                    continue
                top = float(zone.get("top", 0.0))
                bottom = float(zone.get("bottom", 0.0))
                if top <= 0 or bottom <= 0:
                    continue
                if bottom <= current_price <= top:
                    dist_price = 0.0
                    boundary = "inside"
                elif current_price > top:
                    dist_price = abs(current_price - top)
                    boundary = "upper"
                else:
                    dist_price = abs(current_price - bottom)
                    boundary = "lower"
                dist_atr = dist_price / float(atr_value) if atr_value > 0 else 999.0
                age = int(zone.get("age_bars", 9999))
                pre_candidates.append(
                    {
                        "zone": zone,
                        "dist_price": dist_price,
                        "dist_atr": dist_atr,
                        "age": age,
                        "boundary": boundary,
                        "top": top,
                        "bottom": bottom,
                    }
                )
            if not pre_candidates:
                return candidates
            pre_candidates.sort(key=lambda item: item["dist_atr"])
            scoped_candidates = pre_candidates[:max(1, nearest_limit)]
            for payload in scoped_candidates:
                zone = payload["zone"]
                dist_price = payload["dist_price"]
                dist_atr = payload["dist_atr"]
                age = payload["age"]
                boundary = payload["boundary"]
                top = payload["top"]
                bottom = payload["bottom"]
                self._log_debug(
                    "[NDS][FLOW_DEBUG][ZONE_DISTANCE] zone_id=%s type=%s top=%.5f bottom=%.5f price=%.5f dist=%.5f dist_atr=%.3f max=%.3f boundary=%s",
                    zone.get("zone_id"),
                    zone.get("type"),
                    top,
                    bottom,
                    float(current_price),
                    dist_price,
                    dist_atr,
                    max_dist_atr,
                    boundary,
                )
                if dist_atr > max_dist_atr:
                    rejection_counts["too_far"] += 1
                    ref_time = None
                    ref_idx = None
                    if "time" in self.df.columns:
                        ref_time = self.df["time"].iloc[-1]
                    if self.df.index is not None and len(self.df.index) > 0:
                        ref_idx = self.df.index[-1]
                    if hasattr(ref_time, "isoformat"):
                        ref_time = ref_time.isoformat()
                    self._log_info(
                        "[NDS][FLOW][ZONE_REJECT] tier=FLOW reason=TOO_FAR zone_id=%s zone_type=%s idx=%s "
                        "dist_price=%.5f dist_atr=%.2f max_dist_atr=%.2f boundary=%s ref_price=%.5f ref_time=%s ref_idx=%s",
                        zone.get("zone_id"),
                        zone.get("type"),
                        zone.get("retest_index") or zone.get("index"),
                        dist_price,
                        dist_atr,
                        max_dist_atr,
                        boundary,
                        float(current_price),
                        ref_time,
                        ref_idx,
                    )
                    continue
                if age > max_age:
                    rejection_counts["too_old"] += 1
                    self._log_info(
                        "[NDS][FLOW][ZONE_REJECT] tier=FLOW type=%s reason=TOO_OLD age=%s max=%s idx=%s",
                        zone.get("type"),
                        age,
                        max_age,
                        zone.get("retest_index") or zone.get("index"),
                    )
                    continue
                touch_count = int(zone.get("touch_count", 1))
                if touch_count > max_touches:
                    rejection_counts["too_many_touches"] += 1
                    self._log_info(
                        "[NDS][FLOW][ZONE_REJECT] tier=FLOW type=%s reason=TOO_MANY_TOUCHES touches=%s max=%s idx=%s",
                        zone.get("type"),
                        touch_count,
                        max_touches,
                        zone.get("retest_index") or zone.get("index"),
                    )
                    continue
                confidence = float(zone.get("confidence", 0.0))
                freshness_penalty = float(settings.get("FLOW_TOUCH_PENALTY", 0.55))
                util = confidence - (0.5 * dist_atr) - (0.1 * (age / max_age if max_age > 0 else 1.0))
                if touch_count > 1:
                    util *= freshness_penalty
                setup_scores = self._score_flow_setup(
                    zone=zone,
                    dist_atr=dist_atr,
                    max_dist_atr=max_dist_atr,
                    signal=side,
                    session_analysis=session_analysis,
                    volume_analysis=volume_analysis,
                    signal_context=signal_context,
                )
                candidates.append({**zone, "dist_atr": dist_atr, "util": util, **setup_scores})
            return candidates

        def _resolve_entry_model(side: str, top: float, bottom: float) -> Tuple[str, float, str]:
            if side == "BUY":
                if bottom <= current_price <= top:
                    return "MARKET", float(current_price), "price_inside_zone"
                return "STOP", float(top) + buffer_price, "price_outside_zone"
            if bottom <= current_price <= top:
                return "MARKET", float(current_price), "price_inside_zone"
            return "STOP", float(bottom) - buffer_price, "price_outside_zone"

        brk_max_dist = float(settings.get("BRK_MAX_DIST_ATR", 0.5))
        brk_max_age = int(settings.get("BRK_MAX_AGE_BARS", 60))
        brk_candidates = _tier_candidates(breakers, signal, brk_max_dist, brk_max_age)
        if not brk_candidates:
            self._log_debug(
                "[NDS][FLOW_TIER] tier=A breakers skipped (zones=%d max_dist_atr=%.2f max_age=%d)",
                len(breakers),
                brk_max_dist,
                brk_max_age,
            )

        ifvg_max_dist = float(settings.get("IFVG_MAX_DIST_ATR", 3.0))
        ifvg_max_age = int(settings.get("IFVG_MAX_AGE_BARS", 60))
        ifvg_candidates = _tier_candidates(ifvgs, signal, ifvg_max_dist, ifvg_max_age)
        if not ifvg_candidates:
            self._log_debug(
                "[NDS][FLOW_TIER] tier=B ifvg skipped (zones=%d max_dist_atr=%.2f max_age=%d)",
                len(ifvgs),
                ifvg_max_dist,
                ifvg_max_age,
            )

        self._log_info(
            "[NDS][FLOW_ZONES] breakers=%d eligible=%d ifvg=%d eligible=%d",
            len(breakers),
            sum(1 for z in breakers if bool(z.get("eligible", True))),
            len(ifvgs),
            sum(1 for z in ifvgs if bool(z.get("eligible", True))),
        )
        all_zones = breakers + ifvgs
        if all_zones:
            touch_values = [int(z.get("touch_count", 0)) for z in all_zones]
            avg_touches = sum(touch_values) / len(touch_values) if touch_values else 0.0
            max_touches = max(touch_values) if touch_values else 0
            eligible_count = sum(1 for z in all_zones if bool(z.get("eligible", True)))
            last_time = self.df["time"].iloc[-1] if "time" in self.df.columns else None
            self._log_debug(
                "[NDS][FLOW_DEBUG][ZONE_SUMMARY] time=%s zones=%d eligible=%d avg_touches=%.2f max_touches=%d breakers=%d ifvg=%d",
                last_time,
                len(all_zones),
                eligible_count,
                avg_touches,
                max_touches,
                len(breakers),
                len(ifvgs),
            )

        entry_type = "NONE"
        entry_source = None
        entry_model = "NONE"
        entry_level = None
        entry_reason = "no_zone"
        tier = "NONE"
        entry_confidence = 0.0
        reject_reason = "NO_ELIGIBLE_ZONE"

        top_k = int(settings.get("FLOW_SETUP_TOP_K", 3))
        runner_ups: List[Dict[str, Any]] = []
        if brk_candidates:
            brk_candidates.sort(
                key=lambda z: (
                    -float(z.get("setup_score", 0.0)),
                    float(z.get("dist_atr", 999.0)),
                    int(z.get("age_bars", 9999)),
                )
            )
            pick = brk_candidates[0]
            runner_ups = brk_candidates[1:top_k]
            entry_type = "BREAKER"
            entry_source = pick
            entry_model, entry_level, entry_reason = _resolve_entry_model(signal, float(pick.get("top")), float(pick.get("bottom")))
            entry_reason = f"tier=A breaker retest ({entry_reason})"
            tier = "A"
            entry_confidence = float(pick.get("confidence", 0.0))
            reject_reason = None
        elif ifvg_candidates:
            ifvg_candidates.sort(
                key=lambda z: (
                    -float(z.get("setup_score", 0.0)),
                    float(z.get("dist_atr", 999.0)),
                    int(z.get("age_bars", 9999)),
                )
            )
            pick = ifvg_candidates[0]
            runner_ups = ifvg_candidates[1:top_k]
            entry_type = "IFVG"
            entry_source = pick
            entry_model, entry_level, entry_reason = _resolve_entry_model(signal, float(pick.get("top")), float(pick.get("bottom")))
            entry_reason = f"tier=B inversion fvg ({entry_reason})"
            tier = "B"
            entry_confidence = float(pick.get("confidence", 0.0))
            reject_reason = None

        momentum_reason = None
        momentum_block_reason = None
        if entry_level is None:
            momo_adx_min = float(settings.get("MOMO_ADX_MIN", 35.0))
            time_start = settings.get("MOMO_TIME_START", "10:00")
            time_end = settings.get("MOMO_TIME_END", "18:00")
            session_only = bool(settings.get("MOMO_SESSION_ONLY", True))

            now_ts = self.df["time"].iloc[-1]
            parsed_ts = parse_timestamp(now_ts)
            broker_ts = to_broker_time(parsed_ts, self.broker_utc_offset, self.time_mode) if parsed_ts else None
            in_window, time_error = self._is_time_in_window(now_ts, time_start, time_end)
            if time_error:
                momentum_block_reason = "time_parse_failed"
                momentum_reason = f"time_parse_failed:{time_error}"
                self._log_info(
                    "[NDS][FLOW_TIER] tier=C momentum blocked (reason=%s)",
                    momentum_block_reason,
                )
                if momentum_block_reason in {"adx_below_min", "bias_mismatch", "liquidity_blocked", "untradable_market"}:
                    self._log_info(
                        "[NDS][MOMO_BLOCK] reason=%s session=%s ts_broker=%s",
                        momentum_block_reason,
                        session_name,
                        broker_ts,
                    )
                self._log_info(
                    "[NDS][MOMO_BLOCK] reason=time_parse_failed start=%s end=%s ts_broker=%s",
                    time_start,
                    time_end,
                    broker_ts,
                )
                in_window = False
            elif not in_window:
                momentum_block_reason = "time_outside_window"
                momentum_reason = "time_outside_window"
                self._log_info(
                    "[NDS][FLOW_TIER] tier=C momentum blocked (reason=%s)",
                    momentum_block_reason,
                )
                self._log_info(
                    "[NDS][MOMO_BLOCK] reason=time_outside_window start=%s end=%s ts_broker=%s",
                    time_start,
                    time_end,
                    broker_ts,
                )

            session_name = self._normalize_session_name(
                getattr(session_analysis, "current_session", "OTHER")
            )
            session_ok = True
            if session_only:
                allowlist = settings.get("MOMO_SESSION_ALLOWLIST")
                if isinstance(allowlist, str):
                    allowlist = [s.strip().upper() for s in allowlist.split(",") if s.strip()]
                elif isinstance(allowlist, (list, tuple, set)):
                    allowlist = [str(s).strip().upper() for s in allowlist if str(s).strip()]
                else:
                    allowlist = [
                        name for name, data in self.session_definitions.items()
                        if bool(data.get("allow_momentum", True))
                    ] or ["LONDON", "NEW_YORK", "OVERLAP"]
                session_ok = session_name in set(allowlist)

            liquidity_ok = bool(getattr(session_analysis, "is_active_session", True))
            market_status = str(volume_analysis.get("market_status", "") or "").upper()
            if market_status in {"CLOSED", "HALTED"}:
                liquidity_ok = False

            bias = str(signal_context.get("bias", "") or "")
            bias_ok = (bias == "BULLISH" and signal == "BUY") or (bias == "BEARISH" and signal == "SELL") or not bias

            adx_ok = adx_value >= momo_adx_min
            time_ok = in_window
            session_ok = session_ok

            if adx_ok and time_ok and session_ok and liquidity_ok and bias_ok:
                buffer_atr = float(settings.get("MOMO_BUFFER_ATR_MULT", 0.1))
                buffer_min_pips = float(settings.get("MOMO_BUFFER_MIN_PIPS", 1.0))
                buffer_price = pips_to_price(buffer_min_pips, point_size)
                if atr_value > 0:
                    buffer_price = max(buffer_price, float(atr_value) * buffer_atr)
                prev_high = float(self.df["high"].iloc[-2])
                prev_low = float(self.df["low"].iloc[-2])
                if signal == "BUY":
                    entry_level = prev_high + buffer_price
                    if current_price >= entry_level:
                        entry_model = "MARKET"
                        entry_reason = "tier=C momentum triggered"
                    else:
                        entry_model = "STOP"
                        entry_reason = "tier=C momentum breakout"
                else:
                    entry_level = prev_low - buffer_price
                    if current_price <= entry_level:
                        entry_model = "MARKET"
                        entry_reason = "tier=C momentum triggered"
                    else:
                        entry_model = "STOP"
                        entry_reason = "tier=C momentum breakout"

                entry_type = "MOMENTUM"
                entry_source = {
                    "type": "MOMENTUM_BREAKOUT",
                    "prev_high": prev_high,
                    "prev_low": prev_low,
                    "buffer_price": buffer_price,
                    "time_window": f"{time_start}-{time_end}",
                    "session": session_name,
                }
                tier = "C"
                entry_confidence = min(1.0, max(0.35, float(adx_value) / max(momo_adx_min, 1.0)))
                reject_reason = None
            else:
                if momentum_block_reason is None:
                    if not adx_ok:
                        momentum_block_reason = "adx_below_min"
                    elif not session_ok:
                        momentum_block_reason = "session_blocked"
                        self._log_info(
                            "[NDS][MOMO_BLOCK] reason=session_blocked session=%s allowlist=%s",
                            session_name,
                            allowlist,
                        )
                    elif not liquidity_ok:
                        momentum_block_reason = "untradable_market" if market_status in {"CLOSED", "HALTED"} else "liquidity_blocked"
                    elif not bias_ok:
                        momentum_block_reason = "bias_mismatch"
                    else:
                        momentum_block_reason = "time_outside_window"
                momentum_reason = momentum_block_reason
                self._log_info(
                    "[NDS][FLOW_TIER] tier=C momentum blocked (reason=%s)",
                    momentum_block_reason,
                )

        if entry_level is None:
            entry_reason = entry_reason if momentum_reason is None else f"{entry_reason}; momo_block={momentum_reason}"
            if reject_reason is None:
                reject_reason = "NO_ENTRY"

        if entry_level is not None:
            entry_context.update(
                self._calc_entry_distance_metrics(
                    entry_price=float(entry_level),
                    current_price=float(current_price),
                    symbol_meta={"point_size": point_size},
                    atr_value=atr_value,
                )
            )
            self._log_info(
                "[NDS][ENTRY_METRICS] type=%s entry=%.3f cur=%.3f point_size=%.4f dist_price=%.3f dist_points=%.2f dist_pips=%.2f dist_usd=%.3f dist_atr=%.2f",
                entry_type,
                float(entry_level),
                float(current_price),
                float(entry_context.get("point_size") or 0.0),
                float(entry_context.get("dist_price") or 0.0),
                float(entry_context.get("dist_points") or 0.0),
                float(entry_context.get("dist_pips") or 0.0),
                float(entry_context.get("dist_usd") or 0.0),
                float(entry_context.get("dist_atr") or 0.0),
            )

        entry_context.update(
            {
                "point_size": point_size,
                "momentum_reason": momentum_reason,
                "momentum_block_reason": momentum_block_reason,
                "buffer_price": buffer_price,
                "recent_low": recent_low,
                "recent_high": recent_high,
                "zone_rejections": dict(rejection_counts),
                "retest_rejections": dict(retest_rejection_counts),
            }
        )
        if isinstance(entry_source, dict):
            entry_context.setdefault("setup_score", entry_source.get("setup_score"))

        allowed = entry_level is not None
        if allowed:
            self._log_info(
                "[NDS][FLOW_DECISION] tier=%s type=%s model=%s signal=%s allowed=true reason=%s conf=%.2f",
                tier,
                entry_type,
                entry_model,
                signal,
                entry_reason,
                entry_confidence,
            )
        else:
            self._log_info(
                "[NDS][FLOW_DECISION] tier=%s type=%s allowed=false reject=%s reason=%s",
                tier,
                entry_type,
                reject_reason,
                entry_reason,
            )

        return {
            "signal": signal if entry_level is not None else "NONE",
            "tier": tier if entry_level is not None else "NONE",
            "entry_type": entry_type if entry_level is not None else "NONE",
            "entry_model": entry_model if entry_level is not None else "NONE",
            "entry_level": entry_level,
            "zone": entry_source,
            "confidence": entry_confidence,
            "reason": entry_reason,
            "reject_reason": reject_reason,
            "metrics": entry_context,
            "runner_ups": runner_ups,
        }

    def _calc_entry_distance_metrics(
        self,
        entry_price: float,
        current_price: float,
        symbol_meta: Optional[Dict[str, Any]] = None,
        atr_value: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """Calculate entry distance metrics with explicit points/pips/ATR mapping."""
        meta = symbol_meta if isinstance(symbol_meta, dict) else {}
        point_size = meta.get("point_size")
        if point_size is None:
            point_size = self._get_point_size()
        atr_source = atr_value
        if atr_source is None and meta:
            atr_source = meta.get("atr")
        if atr_source is None:
            atr_source = self.atr
        return calculate_distance_metrics(
            entry_price=entry_price,
            current_price=current_price,
            point_size=point_size,
            atr_value=atr_source,
        )

    def _apply_adx_override(
        self,
        structure: MarketStructure,
        adx_value: float,
        plus_di: float,
        minus_di: float,
    ) -> MarketStructure:
        """بهبود منطق override ADX با تاییدیه اضافی"""
        threshold = float(self.GOLD_SETTINGS.get('ADX_OVERRIDE_THRESHOLD', 30.0))
        persistence_bars = int(self.GOLD_SETTINGS.get('ADX_OVERRIDE_PERSISTENCE_BARS', 3))
        require_bos = bool(self.GOLD_SETTINGS.get('ADX_OVERRIDE_REQUIRE_BOS', True))

        if adx_value <= threshold or structure.trend.value != "RANGING":
            return structure

        dominance = None
        if plus_di > minus_di:
            dominance = "BULLISH"
        elif minus_di > plus_di:
            dominance = "BEARISH"

        if dominance is None:
            return structure

        di_persist = False
        if 'plus_di' in self.df.columns and 'minus_di' in self.df.columns:
            recent = self.df[['plus_di', 'minus_di']].tail(max(persistence_bars, 1))
            if dominance == "BULLISH":
                di_persist = bool((recent['plus_di'] > recent['minus_di']).all())
            else:
                di_persist = bool((recent['minus_di'] > recent['plus_di']).all())

        bos_confirmed = structure.bos in {"BULLISH_BOS", "BEARISH_BOS"}
        if require_bos and not bos_confirmed:
            self._log_debug("[NDS][FILTER] ADX override blocked: BOS required")
            return structure

        if not di_persist:
            self._log_debug("[NDS][FILTER] ADX override blocked: DI persistence not met")
            return structure

        new_trend = MarketTrend.UPTREND if dominance == "BULLISH" else MarketTrend.DOWNTREND
        self._log_info("[NDS][FILTER] ADX override applied trend=%s", new_trend.value)
        return MarketStructure(
            trend=new_trend,
            bos=structure.bos,
            choch=structure.choch,
            last_high=structure.last_high,
            last_low=structure.last_low,
            current_price=structure.current_price,
            range_width=structure.range_width,
            range_mid=structure.range_mid,
            bos_choch_confidence=structure.bos_choch_confidence,
            volume_analysis=structure.volume_analysis,
            volatility_state=structure.volatility_state,
            adx_value=adx_value,
            structure_score=structure.structure_score,
        )

    def _determine_volatility(self, atr_long: float, atr_short: float) -> str:
        """تعیین وضعیت نوسان"""
        volatility_ratio = atr_short / atr_long if atr_long > 0 else 1.0

        if volatility_ratio > 1.3:
            return "HIGH_VOLATILITY"
        if volatility_ratio > 0.8:
            return "MODERATE_VOLATILITY"
        return "LOW_VOLATILITY"

    def _calculate_scoring_system(
        self,
        structure: MarketStructure,
        adx_value: float,
        volume_analysis: Dict[str, Any],
        fvgs: List[FVG],
        sweeps: List[Any],
        order_blocks: List[OrderBlock],
        current_price: float,
        swings: List[Any],
        atr_value: float,
        session_analysis: SessionAnalysis,
    ) -> Tuple[float, List[str], Dict[str, Any]]:
        """سیستم امتیازدهی سازگار با وزن‌های مشخص"""
        settings = self.GOLD_SETTINGS
        weights = {
            'structure': 30,
            'trend': 20,
            'fvg': 15,
            'sweeps': 10,
            'order_blocks': 10,
            'volume_session': 15,
        }

        reasons: List[str] = []
        breakdown: Dict[str, Any] = {
            'weights': weights,
            'sub_scores': {},
            'raw_signals': {},
            'modifiers': {},
        }

        structure_score = self._normalize_structure_score(getattr(structure, 'structure_score', 0.0))
        bos_component = 0.0
        choch_component = 0.0
        if structure.bos == "BULLISH_BOS":
            bos_component = 1.0
            self._append_reason(reasons, "✅ Bullish BOS")
        elif structure.bos == "BEARISH_BOS":
            bos_component = -1.0
            self._append_reason(reasons, "🔻 Bearish BOS")

        if structure.choch == "BULLISH_CHOCH":
            choch_component = 1.0
            self._append_reason(reasons, "✅ Bullish CHOCH")
        elif structure.choch == "BEARISH_CHOCH":
            choch_component = -1.0
            self._append_reason(reasons, "🔻 Bearish CHOCH")

        structure_component = self._bounded((structure_score - 50.0) / 50.0)
        structure_sub = self._bounded(
            0.45 * bos_component + 0.35 * choch_component + 0.2 * structure_component
        )
        breakdown['sub_scores']['structure'] = structure_sub
        breakdown['raw_signals']['structure_score'] = structure_score

        trend_dir = 0.0
        if structure.trend.value == "UPTREND":
            trend_dir = 1.0
            self._append_reason(reasons, f"📈 Uptrend (ADX: {adx_value:.1f})")
        elif structure.trend.value == "DOWNTREND":
            trend_dir = -1.0
            self._append_reason(reasons, f"📉 Downtrend (ADX: {adx_value:.1f})")

        trend_strength = min(1.0, max(0.0, adx_value / 40.0))
        rvol = float(volume_analysis.get('rvol', 1.0))
        rvol_strength = min(1.0, max(0.0, (rvol - 0.8) / 1.2))
        trend_sub = self._bounded(trend_dir * (0.7 * trend_strength + 0.3 * rvol_strength))
        breakdown['sub_scores']['trend'] = trend_sub
        breakdown['raw_signals']['adx'] = adx_value
        breakdown['raw_signals']['rvol'] = rvol

        regime_adx_max = float(settings.get("REGIME_ADX_WEAK_MAX", 18.0))
        trend_weight_mult = float(settings.get("REGIME_TREND_WEIGHT_MULT_LOW_ADX", 0.6))
        if adx_value < regime_adx_max:
            trend_sub *= trend_weight_mult
            breakdown['sub_scores']['trend'] = trend_sub
            breakdown.setdefault('modifiers', {})
            breakdown['modifiers']['regime_trend_weight'] = {
                'applied': True,
                'adx': round(adx_value, 2),
                'threshold': round(regime_adx_max, 2),
                'mult': round(trend_weight_mult, 3),
            }
        else:
            breakdown.setdefault('modifiers', {})
            breakdown['modifiers']['regime_trend_weight'] = {
                'applied': False,
                'adx': round(adx_value, 2),
                'threshold': round(regime_adx_max, 2),
            }

        fvg_sub = 0.0
        valid_fvgs = [f for f in fvgs if not f.filled]
        recent_fvgs = [f for f in valid_fvgs if (len(self.df) - 1 - f.index) <= 10]
        fvg_values: List[float] = []
        for fvg in recent_fvgs:
            if fvg.bottom <= current_price <= fvg.top:
                size_ratio = fvg.size / atr_value if atr_value > 0 else 0.0
                size_score = min(1.0, size_ratio / 2.0)
                strength_score = min(1.0, fvg.strength / 2.0)
                base_score = min(1.0, 0.5 * size_score + 0.5 * strength_score)
                sign = 1.0 if fvg.type.value == "BULLISH_FVG" else -1.0
                alignment = 1.0
                if (sign > 0 and structure.trend.value == "DOWNTREND") or (
                    sign < 0 and structure.trend.value == "UPTREND"
                ):
                    alignment = 0.6
                fvg_value = self._bounded(sign * base_score * alignment)
                fvg_values.append(fvg_value)
                breakdown['raw_signals'][f"fvg_{fvg.index}"] = {
                    'type': fvg.type.value,
                    'size': fvg.size,
                    'strength': fvg.strength,
                    'score': fvg_value,
                }
                self._append_reason(
                    reasons,
                    f"{'🟢' if sign > 0 else '🔴'} {fvg.type.value} (Size: ${fvg.size:.2f})",
                )
            else:
                self._log_debug(
                    "[NDS][SMC][FVG] skipped index=%s price=%.2f range=(%.2f,%.2f)",
                    fvg.index,
                    current_price,
                    fvg.bottom,
                    fvg.top,
                )

        if fvg_values:
            fvg_sub = self._bounded(sum(fvg_values) / len(fvg_values))
        fvg_weight_mult = float(settings.get("REGIME_FVG_WEIGHT_MULT_LOW_RVOL", 0.7))
        if rvol < 0.8:
            fvg_sub *= fvg_weight_mult
            breakdown.setdefault('modifiers', {})
            breakdown['modifiers']['regime_fvg_weight'] = {
                'applied': True,
                'rvol': round(rvol, 2),
                'mult': round(fvg_weight_mult, 3),
            }
        breakdown['sub_scores']['fvg'] = fvg_sub

        sweep_values: List[float] = []
        for idx, sweep in enumerate(sweeps[-3:]):
            sweep_type = getattr(sweep, 'type', 'UNKNOWN')
            sweep_penetration = float(getattr(sweep, 'penetration', 0.0))
            sweep_strength = float(getattr(sweep, 'strength', 1.0))
            sign = 1.0 if sweep_type == 'BULLISH_SWEEP' else -1.0
            penetration_ratio = sweep_penetration / atr_value if atr_value > 0 else 0.0
            penetration_score = min(1.0, penetration_ratio / 1.5)
            strength_score = min(1.0, sweep_strength / 2.0)
            sweep_value = self._bounded(sign * (0.6 * penetration_score + 0.4 * strength_score))
            sweep_values.append(sweep_value)
            breakdown['raw_signals'][f"sweep_{idx}"] = {
                'type': sweep_type,
                'penetration': sweep_penetration,
                'strength': sweep_strength,
                'score': sweep_value,
            }
            self._append_reason(
                reasons,
                f"{'✅' if sign > 0 else '🔻'} {sweep_type} (Penetration: ${sweep_penetration:.2f})",
            )

        sweep_sub = self._bounded(sum(sweep_values) / len(sweep_values)) if sweep_values else 0.0
        breakdown['sub_scores']['sweeps'] = sweep_sub

        ob_values: List[float] = []
        recent_obs = order_blocks[-5:] if order_blocks else []
        for idx, ob in enumerate(recent_obs):
            ob_mid = getattr(ob, 'mid', (ob.high + ob.low) / 2)
            distance_atr = abs(current_price - ob_mid) / atr_value if atr_value > 0 else 999.0
            if distance_atr > 1.0:
                self._log_debug(
                    "[NDS][SMC][OB] skipped type=%s distance_atr=%.2f",
                    ob.type,
                    distance_atr,
                )
                continue
            sign = 1.0 if ob.type == 'BULLISH_OB' else -1.0
            distance_score = max(0.0, 1.0 - distance_atr)
            strength_score = min(1.0, ob.strength / 2.0)
            ob_value = self._bounded(sign * (0.6 * strength_score + 0.4 * distance_score))
            ob_values.append(ob_value)
            breakdown['raw_signals'][f"ob_{idx}"] = {
                'type': ob.type,
                'strength': ob.strength,
                'distance_atr': distance_atr,
                'score': ob_value,
            }
            self._append_reason(
                reasons,
                f"{'🟢' if sign > 0 else '🔴'} {ob.type} (Strength: {ob.strength:.1f})",
                )

        ob_sub = self._bounded(sum(ob_values) / len(ob_values)) if ob_values else 0.0
        ob_weight_mult = float(settings.get("REGIME_OB_WEIGHT_MULT_LOW_RVOL", 0.7))
        if rvol < 0.8:
            ob_sub *= ob_weight_mult
            breakdown.setdefault('modifiers', {})
            breakdown['modifiers']['regime_ob_weight'] = {
                'applied': True,
                'rvol': round(rvol, 2),
                'mult': round(ob_weight_mult, 3),
            }
        breakdown['sub_scores']['order_blocks'] = ob_sub

        session_weight = float(session_analysis.weight)
        rvol_component = self._bounded((rvol - 1.0) / 1.0)
        session_component = self._bounded((session_weight - 0.8) / 0.4)
        volume_session_sub = self._bounded(0.6 * rvol_component + 0.4 * session_component)
        breakdown['sub_scores']['volume_session'] = volume_session_sub
        breakdown['modifiers']['session_weight'] = session_weight

        total_weighted = sum(
            weights[name] * breakdown['sub_scores'][name]
            for name in weights
        )

        # --- Structure sanity dampening (prevents fake trends / weak confirmations) ---
        # اگر هیچ BOS/CHOCH نداریم و ADX هم پایین است، اجازه ندهیم امتیاز به ناحیه سیگنال نزدیک شود.
        try:
            sanity_adx_max = float(settings.get('SANITY_ADX_MAX', 18.0))
            sanity_damp = float(settings.get('SANITY_NO_BOS_CHOCH_DAMP', 0.55))
            sanity_min_structure = float(settings.get('SANITY_MIN_STRUCTURE_SCORE', 35.0))
        except Exception:
            sanity_adx_max, sanity_damp, sanity_min_structure = 18.0, 0.55, 35.0

        no_confirm = (getattr(structure, "bos", "NONE") == "NONE" and getattr(structure, "choch", "NONE") == "NONE")
        weak_adx = adx_value < sanity_adx_max
        weak_struct = structure_score < sanity_min_structure

        if no_confirm and weak_adx:
            applied_damp = sanity_damp * (0.8 if weak_struct else 1.0)  # اگر ساختار هم ضعیف است، کمی سخت‌تر
            total_weighted *= applied_damp
            breakdown.setdefault('modifiers', {})
            breakdown['modifiers']['sanity_no_bos_choch'] = {
                'applied': True,
                'adx': round(adx_value, 2),
                'structure_score': round(structure_score, 2),
                'damp': round(applied_damp, 3),
                'reason': 'no BOS/CHOCH with low ADX',
            }
            self._append_reason(
                reasons,
                f"⚠️ Weak confirmation (no BOS/CHOCH, ADX: {adx_value:.1f}) → score dampened"
            )
            self._log_debug(
                "[NDS][SANITY] dampened total_weighted by %.3f (no_confirm=%s weak_adx=%s weak_struct=%s)",
                applied_damp,
                no_confirm,
                weak_adx,
                weak_struct,
            )
        else:
            breakdown.setdefault('modifiers', {})
            breakdown['modifiers']['sanity_no_bos_choch'] = {
                'applied': False,
                'adx': round(adx_value, 2),
                'structure_score': round(structure_score, 2),
            }

        score = 50.0 + 0.5 * total_weighted

        score = max(0.0, min(100.0, score))
        breakdown['summary'] = {
            'total_weighted': total_weighted,
            'score': score,
            'structure_score': structure_score,
        }

        self._log_debug(
            "[NDS][SCORE] total_weighted=%.2f score=%.2f sub_scores=%s",
            total_weighted,
            score,
            breakdown['sub_scores'],
        )

        # INFO-level scoring trace (single-line, parse-friendly)
        try:
            contribs = {k: float(weights[k]) * float(breakdown['sub_scores'][k]) for k in weights}
            self._log_info(
                "[NDS][SCORE_BREAKDOWN] structure_sub=%.4f (w=%.2f c=%.4f) "
                "trend_sub=%.4f (w=%.2f c=%.4f) "
                "fvg_sub=%.4f (w=%.2f c=%.4f) "
                "sweeps_sub=%.4f (w=%.2f c=%.4f) "
                "ob_sub=%.4f (w=%.2f c=%.4f) "
                "volume_session_sub=%.4f (w=%.2f c=%.4f) "
                "-> total_weighted=%.4f formula=50+0.5*total clamp(0..100) score=%.2f",
                float(breakdown['sub_scores']['structure']), float(weights['structure']), float(contribs['structure']),
                float(breakdown['sub_scores']['trend']), float(weights['trend']), float(contribs['trend']),
                float(breakdown['sub_scores']['fvg']), float(weights['fvg']), float(contribs['fvg']),
                float(breakdown['sub_scores']['sweeps']), float(weights['sweeps']), float(contribs['sweeps']),
                float(breakdown['sub_scores']['order_blocks']), float(weights['order_blocks']), float(contribs['order_blocks']),
                float(breakdown['sub_scores']['volume_session']), float(weights['volume_session']), float(contribs['volume_session']),
                float(total_weighted), float(score),
            )
        except Exception:
            pass



        return score, reasons, breakdown




    def _sigmoid(self, x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _clamp(self, lo: float, x: float, hi: float) -> float:
        return max(lo, min(x, hi))

    def _safe_mult(self, m: float) -> float:
        try:
            m = float(m)
        except Exception:
            m = 1.0
        return self._clamp(0.05, m, 1.5)

    def _soft_floor_rational(self, conf_pen: float, floor: float, cap: float, t: float) -> float:
        try:
            conf_pen = float(conf_pen)
        except Exception:
            conf_pen = 0.0
        conf_pen = max(0.0, conf_pen)
        return floor + (cap - floor) * (conf_pen / (conf_pen + t))


    def _calculate_confidence(
        self,
        normalized_score: float,
        volatility_state: str,
        session_analysis: SessionAnalysis,
        volume_analysis: Dict[str, Any],
        scalping_mode: bool = True,
        sweeps: Optional[list] = None,
    ) -> float:
        """
        Design 3 (Z-score probability-of-edge) + log-space penalties + soft-floor
        - بدون hard clamp کف 10
        - پنالتی near-neutral به شکل پیوسته (جایگزین 42..58)
        - traceable logs
        """

        score = float(normalized_score)

        # ------------------------------
        # Parameters (TUNE)
        # ------------------------------
        C_MIN = 2.0
        C_MAX = 85.0

        # z-edge mapping
        Z0 = 0.8
        KZ = 0.45

        # neutral softness
        D0 = 4.5
        KD = 2.0
        NEUTRAL_FLOOR = 0.55

        # penalty strengths
        ALPHA_SESSION = 0.70
        ALPHA_RVOL = 0.55
        ALPHA_NEUTRAL = 0.75
        ALPHA_VOL = 0.40

        # soft-floor shaping
        FLOOR = 2.0
        T = 18.0

        # ------------------------------
        # Volatility adjustment (soft)
        # ------------------------------
        volatility_state = self._normalize_volatility_state(volatility_state)
        vol_mult = 1.0
        if volatility_state == 'HIGH_VOLATILITY':
            vol_mult = 1.08 if scalping_mode else 0.85
        elif volatility_state == 'LOW_VOLATILITY':
            vol_mult = 0.92 if scalping_mode else 1.05
        vol_mult_s = self._safe_mult(vol_mult)

        # ------------------------------
        # Session multiplier (soft)
        # ------------------------------
        session_name = str(
            getattr(session_analysis, "current_session", None)
            or getattr(session_analysis, "session", None)
            or getattr(session_analysis, "name", None)
            or "UNKNOWN"
        )

        # IMPORTANT: support both 'session_weight' and 'weight'
        _session_weight = (
            getattr(session_analysis, "session_weight", None)
            if getattr(session_analysis, "session_weight", None) is not None
            else getattr(session_analysis, "weight", None)
        )
        try:
            session_weight = float(_session_weight if _session_weight is not None else 1.0)
        except Exception:
            session_weight = 1.0

        upstream_active = bool(getattr(session_analysis, "is_active_session", True))

        session_activity = str(
            getattr(session_analysis, "session_activity", None)
            or getattr(session_analysis, "activity", None)
            or "UNKNOWN"
        ).upper()

        strong_signal = abs(score - 50) > 15

        session_mult = 1.0
        if session_weight < 0.6:
            session_mult *= 0.75
        if strong_signal and session_weight > 0.8:
            session_mult *= 1.15

        # ------------------------------
        # Untradable gate (unchanged)
        # ------------------------------
        market_status = str(volume_analysis.get("market_status", "") or "").upper()
        data_ok = volume_analysis.get("data_ok", None)
        spread_pips = volume_analysis.get("spread_pips", volume_analysis.get("spread", None))
        max_spread_pips = volume_analysis.get("max_spread_pips", volume_analysis.get("max_spread", None))

        untradable = False
        untradable_reasons = []

        if market_status in ("CLOSED", "HALTED"):
            untradable = True
            untradable_reasons.append(f"market_status={market_status}")

        if data_ok is False:
            untradable = True
            untradable_reasons.append("data_ok=False")

        if spread_pips is not None and max_spread_pips is not None:
            try:
                if float(spread_pips) > float(max_spread_pips):
                    untradable = True
                    untradable_reasons.append(
                        f"spread_pips={float(spread_pips):.4f}>max={float(max_spread_pips):.4f}"
                    )
            except Exception:
                pass

        effective_active = upstream_active
        if (not upstream_active) and (not untradable):
            effective_active = True

        if not effective_active:
            session_mult *= 0.8

        session_mult_s = self._safe_mult(session_mult)

        # ------------------------------
        # RVOL multiplier (soft)
        # ------------------------------
        rvol_mult = 1.0
        current_rvol = volume_analysis.get('rvol', 1.0)
        try:
            current_rvol = float(current_rvol)
        except Exception:
            current_rvol = 1.0

        if current_rvol > 1.2:
            rvol_mult *= 1.1
        elif current_rvol < 0.8:
            rvol_mult *= 0.9

        rvol_mult_s = self._safe_mult(rvol_mult)

        # ------------------------------
        # Neutral softness
        # ------------------------------
        d = abs(score - 50.0)
        neutral_mult = NEUTRAL_FLOOR + (1.0 - NEUTRAL_FLOOR) * self._sigmoid((d - D0) / KD)
        neutral_mult_s = self._safe_mult(neutral_mult)

        # ------------------------------
        # Z-score history (probability of edge)
        # ✅ از history shared استفاده می‌کنیم
        # ------------------------------
        hist = list(self._score_hist)

        if len(hist) >= 30:
            mu = mean(hist)
            sigma = pstdev(hist)
            sigma = max(1.0, float(sigma))
            z = (score - float(mu)) / sigma
            z_edge = abs(z)
            p_edge = self._sigmoid((z_edge - Z0) / KZ)
            conf_raw = C_MIN + (C_MAX - C_MIN) * p_edge
            z_mode = f"hist(n={len(hist)})"
        else:
            p_edge = self._sigmoid((d - 6.0) / 2.5)
            conf_raw = C_MIN + (C_MAX - C_MIN) * p_edge
            mu = 50.0
            sigma = 0.0
            z = 0.0
            z_edge = 0.0
            z_mode = f"warmup(n={len(hist)})"

        # ------------------------------
        # Apply penalties
        # ------------------------------
        vol_eff = vol_mult_s ** ALPHA_VOL
        session_eff = session_mult_s ** ALPHA_SESSION
        rvol_eff = rvol_mult_s ** ALPHA_RVOL
        neutral_eff = neutral_mult_s ** ALPHA_NEUTRAL

        combined_eff = vol_eff * session_eff * rvol_eff * neutral_eff
        conf_pen = conf_raw * combined_eff

        # ------------------------------
        # Sweep bonus (soft)
        # ------------------------------
        sweeps_count = len(sweeps) if sweeps else 0
        sweep_mult = 1.0
        if sweeps_count > 0 and strong_signal:
            sweep_mult = 1.03
            conf_pen *= sweep_mult

        # ------------------------------
        # Soft-floor final
        # ------------------------------
        conf_final = self._soft_floor_rational(conf_pen, floor=FLOOR, cap=C_MAX, t=T)
        conf_final = self._clamp(0.0, conf_final, 100.0)

        # ------------------------------
        # ✅ Update history AFTER computation (shared)
        # ------------------------------
        try:
            self._score_hist.append(score)
        except Exception:
            pass

        # ------------------------------
        # Logging
        # ------------------------------
        try:
            reasons_str = ",".join(untradable_reasons) if untradable_reasons else "-"
            self._log_info(
                "[NDS][CONF_V2] score=%.2f d=%.2f vol=%s vol_mult=%.3f vol_eff=%.3f(a=%.2f) | "
                "sess=%s w=%.2f act=%s upstream=%s eff_active=%s untradable=%s reasons=%s strong=%s "
                "session_mult=%.3f session_eff=%.3f(a=%.2f) | "
                "rvol=%.2f rvol_mult=%.3f rvol_eff=%.3f(a=%.2f) | "
                "neutral_mult=%.3f neutral_eff=%.3f(a=%.2f) | "
                "z_mode=%s mu=%.2f sigma=%.2f z=%.3f z_edge=%.3f p_edge=%.4f conf_raw=%.2f | "
                "combined_eff=%.4f sweeps=%d sweep_mult=%.3f conf_pen=%.2f | "
                "soft_floor(f=%.1f,t=%.1f,cap=%.1f)->conf=%.2f",
                score, d,
                volatility_state, vol_mult_s, vol_eff, ALPHA_VOL,
                session_name, session_weight, session_activity,
                bool(upstream_active), bool(effective_active),
                bool(untradable), reasons_str, bool(strong_signal),
                session_mult_s, session_eff, ALPHA_SESSION,
                float(current_rvol), rvol_mult_s, rvol_eff, ALPHA_RVOL,
                neutral_mult_s, neutral_eff, ALPHA_NEUTRAL,
                z_mode, float(mu), float(sigma),
                float(z), float(z_edge), float(p_edge), float(conf_raw),
                float(combined_eff),
                int(sweeps_count), float(sweep_mult), float(conf_pen),
                FLOOR, T, C_MAX, float(conf_final),
            )
        except Exception:
            pass

        return round(conf_final, 1)


    def _compute_directional_bias(
        self,
        structure: MarketStructure,
        adx_value: float,
        plus_di: float,
        minus_di: float,
        scalping_mode: bool,
    ) -> Dict[str, Any]:
        """Compute directional bias using structure trend + ADX + DI dominance."""
        settings = self.GOLD_SETTINGS
        try:
            adx_trend_min = float(settings.get("CT_ADX_TREND_MIN", 18.0))
        except Exception:
            adx_trend_min = 18.0
        try:
            adx_full = float(settings.get("CT_ADX_FULL", 35.0))
        except Exception:
            adx_full = 35.0
        try:
            di_margin = float(settings.get("CT_DI_MARGIN", 6.0))
        except Exception:
            di_margin = 6.0

        trend_value = getattr(structure, "trend", MarketTrend.RANGING).value
        di_diff = float(plus_di) - float(minus_di)
        di_dominance = "NEUTRAL"
        if di_diff >= di_margin:
            di_dominance = "BULLISH"
        elif di_diff <= -di_margin:
            di_dominance = "BEARISH"

        adx_strength = 0.0
        if adx_full > adx_trend_min:
            adx_strength = (float(adx_value) - adx_trend_min) / (adx_full - adx_trend_min)
        adx_strength = self._clamp(0.0, adx_strength, 1.0)

        bias = "NEUTRAL"
        reason_parts = [f"trend={trend_value}", f"adx={adx_value:.1f}", f"di_diff={di_diff:.1f}"]

        if trend_value == "UPTREND":
            bias = "BULLISH"
        elif trend_value == "DOWNTREND":
            bias = "BEARISH"

        if trend_value == "RANGING" and di_dominance != "NEUTRAL" and adx_value >= adx_trend_min:
            bias = di_dominance
            reason_parts.append("range_bias_from_di")

        conflict = False
        if bias != "NEUTRAL" and di_dominance != "NEUTRAL" and bias != di_dominance:
            if abs(di_diff) >= di_margin * 1.25:
                conflict = True
                bias = "NEUTRAL"
                reason_parts.append("bias_neutralized_conflict")

        strong_trend = bool(
            bias in {"BULLISH", "BEARISH"}
            and adx_value >= adx_trend_min
            and di_dominance == bias
            and not conflict
        )

        trend_strength = adx_strength
        if di_dominance == bias and bias != "NEUTRAL":
            trend_strength = self._clamp(0.0, adx_strength + 0.15, 1.0)

        reason = "; ".join(reason_parts)
        return {
            "bias": bias,
            "strong_trend": strong_trend,
            "trend_strength": float(trend_strength),
            "reason": reason,
        }

    def _compute_reversal_context(
        self,
        structure: MarketStructure,
        score: float,
        confidence: float,
        adx_value: float,
        plus_di: float,
        minus_di: float,
        structure_score: float,
        sweeps: List[Any],
        scalping_mode: bool,
    ) -> Dict[str, Any]:
        """Compute reversal eligibility for counter-trend allowance."""
        settings = self.GOLD_SETTINGS
        try:
            min_d = float(settings.get("CT_REVERSAL_MIN_D", 12.0))
        except Exception:
            min_d = 12.0

        trend_value = getattr(structure, "trend", MarketTrend.RANGING).value
        choch_value = getattr(structure, "choch", "NONE")
        sweeps_count = len(sweeps) if sweeps else 0
        score_distance = abs(float(score) - 50.0)

        choch_reversal = False
        if trend_value == "UPTREND" and choch_value == "BEARISH_CHOCH":
            choch_reversal = True
        elif trend_value == "DOWNTREND" and choch_value == "BULLISH_CHOCH":
            choch_reversal = True

        sweep_reversal = bool(sweeps_count > 0 and score_distance >= min_d)

        reversal_ok = bool(choch_reversal or sweep_reversal)
        reason_parts = [
            f"trend={trend_value}",
            f"choch={choch_value}",
            f"sweeps={sweeps_count}",
            f"d={score_distance:.1f}",
            f"min_d={min_d:.1f}",
        ]
        if choch_reversal:
            reason_parts.append("reversal=choch")
        elif sweep_reversal:
            reason_parts.append("reversal=sweep_score")
        else:
            reason_parts.append("reversal=none")

        di_diff = float(plus_di) - float(minus_di)
        di_dominance = "NEUTRAL"
        if di_diff >= 6.0:
            di_dominance = "BULLISH"
        elif di_diff <= -6.0:
            di_dominance = "BEARISH"

        return {
            "reversal_ok": reversal_ok,
            "reason": "; ".join(reason_parts),
            "sweeps_count": sweeps_count,
            "structure_score": float(structure_score),
            "adx": float(adx_value),
            "di_dominance": di_dominance,
            "di_diff": float(di_diff),
        }

    def _build_signal_context(
        self,
        structure: MarketStructure,
        score: float,
        confidence: float,
        adx_value: float,
        plus_di: float,
        minus_di: float,
        sweeps: List[Any],
        volatility_state: str,
        scalping_mode: bool,
    ) -> Dict[str, Any]:
        """Compose bias + reversal + counter-trend risk context for downstream gating."""
        structure_score = self._normalize_structure_score(getattr(structure, "structure_score", 0.0))
        bias_context = self._compute_directional_bias(
            structure=structure,
            adx_value=adx_value,
            plus_di=plus_di,
            minus_di=minus_di,
            scalping_mode=scalping_mode,
        )
        reversal_context = self._compute_reversal_context(
            structure=structure,
            score=score,
            confidence=confidence,
            adx_value=adx_value,
            plus_di=plus_di,
            minus_di=minus_di,
            structure_score=structure_score,
            sweeps=sweeps,
            scalping_mode=scalping_mode,
        )

        bias = bias_context.get("bias")
        trend_strength = float(bias_context.get("trend_strength", 0.0))
        structure_strength = self._clamp(0.0, structure_score / 100.0, 1.0)
        counter_trend_risk = self._clamp(0.0, 0.65 * trend_strength + 0.35 * structure_strength, 1.0)
        if bias == "NEUTRAL":
            counter_trend_risk *= 0.3
        if reversal_context.get("reversal_ok"):
            counter_trend_risk *= 0.35

        directional_bias = "RANGING"
        if bias == "BULLISH":
            directional_bias = "UPTREND"
        elif bias == "BEARISH":
            directional_bias = "DOWNTREND"

        return {
            "bias": bias,
            "directional_bias": directional_bias,
            "strong_trend": bias_context.get("strong_trend"),
            "trend_strength": trend_strength,
            "bias_reason": bias_context.get("reason"),
            "reversal_ok": reversal_context.get("reversal_ok"),
            "reversal_reason": reversal_context.get("reason"),
            "sweeps_count": reversal_context.get("sweeps_count"),
            "counter_trend_risk": float(counter_trend_risk),
            "structure_score": float(structure_score),
            "adx": float(adx_value),
            "plus_di": float(plus_di),
            "minus_di": float(minus_di),
            "di_dominance": reversal_context.get("di_dominance"),
            "di_diff": reversal_context.get("di_diff"),
            "trend": structure.trend.value if structure else "UNKNOWN",
            "bos": structure.bos if structure else "NONE",
            "choch": structure.choch if structure else "NONE",
            "volatility_state": volatility_state,
        }


    def _determine_signal(
            self,
            normalized_score: float,
            confidence: float,
            volatility_state: str,
            scalping_mode: bool = True,
            context: Optional[Dict[str, Any]] = None,
        ) -> str:
            """تعیین سیگنال با آستانه‌های پویا + گیت‌های اعتماد/نزدیک خنثی/ضد روند

            نکته:
            - آستانه‌های پیش‌فرض مطابق منطق قبلی حفظ شده‌اند.
            - برای بک‌تست/آپتیمایز می‌توانید از طریق config این آستانه‌ها را override کنید:
                technical_settings.SCALPING_BUY_THRESHOLD
                technical_settings.SCALPING_SELL_THRESHOLD

            همچنین برای جلوگیری از خطای انسانی (مثل گرید شما)، اگر این کلیدها را
            اشتباهاً زیر trading_settings گذاشته باشید هم پشتیبانی می‌شود:
                trading_settings.SCALPING_BUY_THRESHOLD
                trading_settings.SCALPING_SELL_THRESHOLD
            """
            settings = self.GOLD_SETTINGS

            # --- Default thresholds (legacy behaviour)
            if scalping_mode:
                volatility_state = self._normalize_volatility_state(volatility_state)
                if volatility_state == 'HIGH_VOLATILITY':
                    buy_threshold = 60
                    sell_threshold = 40
                elif volatility_state == 'LOW_VOLATILITY':
                    buy_threshold = 65
                    sell_threshold = 35
                else:
                    buy_threshold = 55
                    sell_threshold = 45
            else:
                buy_threshold = 65
                sell_threshold = 35

            # --- Configurable overrides (keeps backward compatibility)
            # We accept overrides in BOTH technical_settings and trading_settings to prevent misconfiguration.
            try:
                cfg = self.config if isinstance(self.config, dict) else {}
                ts = cfg.get("technical_settings", {}) if isinstance(cfg.get("technical_settings", {}), dict) else {}
                tr = cfg.get("trading_settings", {}) if isinstance(cfg.get("trading_settings", {}), dict) else {}

                # prefer technical_settings; fallback to trading_settings
                buy_ovr = ts.get("SCALPING_BUY_THRESHOLD", None)
                sell_ovr = ts.get("SCALPING_SELL_THRESHOLD", None)

                if buy_ovr is None:
                    buy_ovr = tr.get("SCALPING_BUY_THRESHOLD", None)
                if sell_ovr is None:
                    sell_ovr = tr.get("SCALPING_SELL_THRESHOLD", None)

                # allow numeric strings too
                if buy_ovr is not None:
                    buy_threshold = float(buy_ovr)
                if sell_ovr is not None:
                    sell_threshold = float(sell_ovr)
            except Exception:
                # never let config parsing break signal generation
                pass

            threshold_mode = str(settings.get("SIGNAL_THRESHOLD_MODE") or "STATIC").upper()
            if scalping_mode:
                threshold_mode = str(settings.get("SCALPING_THRESHOLD_MODE") or threshold_mode).upper()
            if threshold_mode == "PERCENTILE":
                hist = list(self._score_hist)
                min_hist = settings.get("PERCENTILE_MIN_HISTORY")
                if min_hist is None:
                    min_hist = settings.get("SIGNAL_MIN_HISTORY")
                try:
                    min_hist = int(min_hist)
                except (TypeError, ValueError):
                    min_hist = 60
                if len(hist) >= max(1, min_hist):
                    buy_pct_raw = settings.get("SIGNAL_BUY_PERCENTILE")
                    sell_pct_raw = settings.get("SIGNAL_SELL_PERCENTILE")
                    try:
                        buy_pct = float(buy_pct_raw)
                        sell_pct = float(sell_pct_raw)
                    except (TypeError, ValueError):
                        buy_pct = 0.8
                        sell_pct = 0.2
                    buy_threshold = self._percentile(hist, buy_pct)
                    sell_threshold = self._percentile(hist, sell_pct)
                    min_spread = float(settings.get("SIGNAL_MIN_THRESHOLD_SPREAD", 3.0))
                    if (buy_threshold - sell_threshold) < min_spread:
                        mid = (buy_threshold + sell_threshold) / 2.0
                        buy_threshold = mid + (min_spread / 2.0)
                        sell_threshold = mid - (min_spread / 2.0)
                    self._log_debug(
                        "[NDS][THRESH] mode=percentile n=%s buy_pct=%.2f sell_pct=%.2f buy=%.2f sell=%.2f",
                        len(hist),
                        buy_pct,
                        sell_pct,
                        buy_threshold,
                        sell_threshold,
                    )
                else:
                    self._log_info(
                        "[NDS][THRESH] mode=percentile warmup n=%s min=%s -> fallback=STATIC",
                        len(hist),
                        min_hist,
                    )

            try:
                buy_min_raw = settings.get("SIGNAL_BUY_THRESHOLD_MIN")
                buy_max_raw = settings.get("SIGNAL_BUY_THRESHOLD_MAX")
                sell_min_raw = settings.get("SIGNAL_SELL_THRESHOLD_MIN")
                sell_max_raw = settings.get("SIGNAL_SELL_THRESHOLD_MAX")
                if None not in (buy_min_raw, buy_max_raw, sell_min_raw, sell_max_raw):
                    buy_min = float(buy_min_raw)
                    buy_max = float(buy_max_raw)
                    sell_min = float(sell_min_raw)
                    sell_max = float(sell_max_raw)
                    buy_threshold = min(max(float(buy_threshold), buy_min), buy_max)
                    sell_threshold = min(max(float(sell_threshold), sell_min), sell_max)
            except Exception:
                pass

            ema_alpha_raw = settings.get("SIGNAL_THRESHOLD_EMA_ALPHA")
            try:
                ema_alpha = float(ema_alpha_raw)
            except (TypeError, ValueError):
                ema_alpha = 0.0
            if 0.0 < ema_alpha < 1.0:
                prev_buy = self._threshold_ema.get("buy")
                prev_sell = self._threshold_ema.get("sell")
                if prev_buy is None:
                    prev_buy = buy_threshold
                if prev_sell is None:
                    prev_sell = sell_threshold
                buy_threshold = (ema_alpha * buy_threshold) + ((1.0 - ema_alpha) * prev_buy)
                sell_threshold = (ema_alpha * sell_threshold) + ((1.0 - ema_alpha) * prev_sell)
                self._threshold_ema["buy"] = buy_threshold
                self._threshold_ema["sell"] = sell_threshold

            # Safety: ensure thresholds are sensible
            try:
                buy_threshold = float(buy_threshold)
                sell_threshold = float(sell_threshold)
            except Exception:
                buy_threshold, sell_threshold = 55.0, 45.0

            # If user misconfigures (e.g., buy <= sell), widen minimally to avoid degenerate logic
            if buy_threshold <= sell_threshold:
                mid = (buy_threshold + sell_threshold) / 2.0
                buy_threshold = mid + 1.0
                sell_threshold = mid - 1.0

            if normalized_score >= buy_threshold:
                base_signal = "BUY"
            elif normalized_score <= sell_threshold:
                base_signal = "SELL"
            else:
                return "NONE"

            volatility_state = self._normalize_volatility_state(volatility_state)
            try:
                min_conf_base = float(settings.get("SIGNAL_MIN_CONF", 42.0))
            except Exception:
                min_conf_base = 42.0
            try:
                min_conf_hv = float(settings.get("SIGNAL_MIN_CONF_HV", min_conf_base))
            except Exception:
                min_conf_hv = min_conf_base
            try:
                min_conf_lv = float(settings.get("SIGNAL_MIN_CONF_LV", min_conf_base))
            except Exception:
                min_conf_lv = min_conf_base

            if volatility_state == "HIGH_VOLATILITY":
                min_conf = min_conf_hv
            elif volatility_state == "LOW_VOLATILITY":
                min_conf = min_conf_lv
            else:
                min_conf = min_conf_base

            if confidence < min_conf:
                self._log_debug(
                    "[NDS][SIGNAL_GATE] blocked by confidence base_signal=%s conf=%.1f min=%.1f vol=%s",
                    base_signal,
                    confidence,
                    min_conf,
                    volatility_state,
                )
                return "NONE"

            try:
                neutral_band = float(settings.get("SIGNAL_NEUTRAL_BAND", 4.0))
            except Exception:
                neutral_band = 4.0
            try:
                neutral_extra = float(settings.get("SIGNAL_NEUTRAL_CONF_EXTRA", 8.0))
            except Exception:
                neutral_extra = 8.0

            if abs(float(normalized_score) - 50.0) < neutral_band:
                if confidence < (min_conf + neutral_extra):
                    self._log_debug(
                        "[NDS][SIGNAL_GATE] blocked near-neutral base_signal=%s score=%.1f conf=%.1f min=%.1f extra=%.1f",
                        base_signal,
                        normalized_score,
                        confidence,
                        min_conf,
                        neutral_extra,
                    )
                    return "NONE"

            signal = base_signal
            if context:
                bias = context.get("bias")
                strong_trend = bool(context.get("strong_trend"))
                reversal_ok = bool(context.get("reversal_ok"))
                ct_risk = float(context.get("counter_trend_risk", 0.0) or 0.0)
                ct_risk_max = float(settings.get("CT_RISK_MAX", 0.68))
                if strong_trend and bias in {"BULLISH", "BEARISH"}:
                    if (bias == "BULLISH" and base_signal == "SELL") or (
                        bias == "BEARISH" and base_signal == "BUY"
                    ):
                        allow = reversal_ok or (ct_risk <= ct_risk_max)
                        if not allow:
                            self._log_info(
                                "[NDS][CTGATE] signal=%s bias=%s allow=%s reason=%s conf=%.1f score=%.1f",
                                base_signal,
                                bias,
                                allow,
                                "counter_trend_risk",
                                confidence,
                                normalized_score,
                            )
                            return "NONE"
                        self._log_info(
                            "[NDS][CTGATE] signal=%s bias=%s allow=%s reason=%s conf=%.1f score=%.1f",
                            base_signal,
                            bias,
                            allow,
                            "reversal_ok" if reversal_ok else "ct_risk_ok",
                            confidence,
                            normalized_score,
                        )

            return signal





    def _build_initial_result(
        self,
        signal: str,
        confidence: float,
        score: float,
        reasons: List[str],
        structure: MarketStructure,
        atr_value: float,
        atr_short_value: Optional[float],
        adx_value: float,
        plus_di: float,
        minus_di: float,
        volume_analysis: Dict[str, Any],
        recent_range: float,
        recent_position: float,
        volatility_state: str,
        session_analysis: SessionAnalysis,
        current_price: float,
        timeframe: str,
        score_breakdown: Dict[str, Any],
        scalping_mode: bool = True,
    ) -> Dict[str, Any]:
        """ساخت ساختار نتیجه اولیه"""
        normalized_structure_score = self._normalize_structure_score(
            getattr(structure, 'structure_score', 0.0)
        )

        current_rvol = float(volume_analysis.get('rvol', 1.0))
        if pd.isna(current_rvol):
            current_rvol = 1.0

        session_payload = normalize_session_payload(session_analysis.session_decision)
        if not session_payload:
            session_payload = {
                "session_name": session_analysis.current_session,
                "weight": session_analysis.session_weight,
                "activity": session_analysis.session_activity,
                "policy_mode": session_analysis.policy_mode,
                "is_tradable": session_analysis.is_tradable,
                "block_reason": session_analysis.block_reason,
                "is_overlap": session_analysis.is_overlap,
                "time_mode": session_analysis.time_mode,
                "broker_utc_offset_hours": session_analysis.broker_utc_offset_hours,
                "ts_broker": (
                    session_analysis.ts_broker.isoformat()
                    if session_analysis.ts_broker is not None
                    else None
                ),
            }

        result = {
            "signal": signal,
            "confidence": confidence,
            "score": round(score, 1),
            "reasons": reasons[:8],
            "session": session_payload,
            "session_name": session_payload.get("session_name"),
            "session_activity": getattr(session_analysis, "session_activity", None),
            "ts_broker": (
                session_analysis.ts_broker.isoformat()
                if getattr(session_analysis, "ts_broker", None) is not None
                else None
            ),
            "time_mode": getattr(session_analysis, "time_mode", None),
            "broker_utc_offset_hours": getattr(session_analysis, "broker_utc_offset_hours", None),
            "adx": float(adx_value),
            "plus_di": float(plus_di),
            "minus_di": float(minus_di),
            "structure": {
                "trend": structure.trend.value,
                "bos": structure.bos,
                "choch": structure.choch,
                "last_high": structure.last_high.price if structure.last_high else None,
                "last_low": structure.last_low.price if structure.last_low else None,
                "range_width": round(structure.range_width, 2) if structure.range_width else 0,
                "range_mid": round(structure.range_mid, 2) if structure.range_mid else 0,
                "structure_score": normalized_structure_score,
            },
            "market_metrics": {
                "atr": round(atr_value, 2),
                "adx": round(adx_value, 1),
                "plus_di": round(plus_di, 1),
                "minus_di": round(minus_di, 1),
                "recent_range": round(recent_range, 2),
                "recent_position": round(recent_position, 2),
                "volatility_state": volatility_state,
                "current_rvol": round(current_rvol, 2),
            },
            "analysis_data": {
                "volume_analysis": volume_analysis,
                "score_breakdown": score_breakdown,
            },
            "session_analysis": {
                "current_session": session_analysis.current_session,
                "session_weight": session_analysis.session_weight,
                "is_active_session": session_analysis.is_active_session,
                "optimal_trading": session_analysis.optimal_trading,
                "weight": session_analysis.weight,
                "session_activity": session_analysis.session_activity,
                "policy_mode": session_analysis.policy_mode,
                "is_tradable": session_analysis.is_tradable,
                "block_reason": session_analysis.block_reason,
                "untradable": session_analysis.untradable,
                "untradable_reasons": session_analysis.untradable_reasons,
                "session_decision": session_analysis.session_decision,
                "ts_broker": (
                    session_analysis.ts_broker.isoformat()
                    if session_analysis.ts_broker is not None
                    else None
                ),
                "time_mode": session_analysis.time_mode,
                "broker_utc_offset_hours": session_analysis.broker_utc_offset_hours,
            },
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "timeframe": timeframe,
            "scalping_mode": scalping_mode,
        }

        if scalping_mode and atr_short_value:
            result["market_metrics"]["atr_short"] = round(atr_short_value, 2)

        return result

    def _build_analysis_trace(
        self,
        signal: str,
        confidence: float,
        score: float,
        volume_analysis: Dict[str, Any],
        session_analysis: SessionAnalysis,
        signal_context: Dict[str, Any],
        entry_idea: Dict[str, Any],
        reasons: List[str],
        integrity_flags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        min_session_weight = float(self.GOLD_SETTINGS.get("MIN_SESSION_WEIGHT", 0.35))
        min_rvol = float(self.GOLD_SETTINGS.get("MIN_RVOL_SCALPING", 0.35))
        adx_threshold = float(self.GOLD_SETTINGS.get("ADX_THRESHOLD_WEAK", 20.0))

        spread_pips = volume_analysis.get("spread_pips", volume_analysis.get("spread"))
        max_spread_pips = volume_analysis.get("max_spread_pips", volume_analysis.get("max_spread"))
        spread_ok = True
        if spread_pips is not None and max_spread_pips is not None:
            spread_ok = float(spread_pips) <= float(max_spread_pips)

        session_weight = float(getattr(session_analysis, "session_weight", session_analysis.weight))
        session_ok = bool(getattr(session_analysis, "is_active_session", True)) and session_weight >= min_session_weight

        market_status = str(volume_analysis.get("market_status", "") or "").upper()
        liquidity_ok = market_status not in {"CLOSED", "HALTED"} and float(volume_analysis.get("rvol", 1.0)) >= min_rvol

        trend_ok = float(signal_context.get("adx", 0.0)) >= adx_threshold

        zone = entry_idea.get("zone") if isinstance(entry_idea, dict) else None
        setup_payload = {}
        if isinstance(zone, dict):
            setup_payload = {
                "selected_zone_type": zone.get("type"),
                "selected_zone_id": zone.get("zone_id"),
                "retest_reason": zone.get("retest_reason"),
                "freshness": zone.get("fresh"),
                "touches": zone.get("touch_count"),
                "distance_to_zone_atr": zone.get("dist_atr"),
                "setup_score": zone.get("setup_score"),
            }

        market_state = {
            "trend": signal_context.get("trend"),
            "volatility_state": signal_context.get("volatility_state"),
            "volume_zone": volume_analysis.get("volume_zone"),
            "session": getattr(session_analysis, "current_session", None),
        }

        decision_notes = []
        for reason in reasons:
            if reason not in decision_notes:
                decision_notes.append(reason)
            if len(decision_notes) >= 8:
                break
        if signal == "NONE":
            if not spread_ok:
                decision_notes.append("gate:spread")
            if not session_ok:
                decision_notes.append("gate:session")
            if not liquidity_ok:
                decision_notes.append("gate:liquidity")
            if not trend_ok:
                decision_notes.append("gate:trend")
        if integrity_flags:
            decision_notes.extend([f"integrity:{flag}" for flag in integrity_flags])

        return {
            "signal": signal,
            "confidence": round(float(confidence), 2),
            "normalized_score": round(float(score), 2),
            "market_state": market_state,
            "gates": {
                "spread_ok": spread_ok,
                "session_ok": session_ok,
                "liquidity_ok": liquidity_ok,
                "trend_ok": trend_ok,
            },
            "setup": setup_payload,
            "decision_notes": decision_notes,
        }

    def _adaptive_min_rvol(self, base_min_rvol: float, structure_score: float) -> float:
        if structure_score >= 90.0:
            return base_min_rvol * 0.5
        if structure_score >= 70.0:
            return base_min_rvol * 0.75
        return base_min_rvol

    def _apply_final_filters(self, analysis_result: Dict[str, Any], scalping_mode: bool = True) -> Dict[str, Any]:
        """
        اعمال فیلترهای نهایی با اتصال به تنظیمات مرکزی
        """
        original_signal = analysis_result.get('signal', 'NONE')
        reasons = analysis_result.get('reasons', [])

        self._log_debug("[NDS][FILTER] start signal=%s", original_signal)

        if original_signal != 'NONE':
            settings = self.GOLD_SETTINGS

            cooldown_start = str(settings.get("FLOW_NY_OPEN_COOLDOWN_START", "15:00"))
            cooldown_end = str(settings.get("FLOW_NY_OPEN_COOLDOWN_END", "15:15"))
            last_time = self.df["time"].iloc[-1] if "time" in self.df.columns else None
            if last_time is not None:
                in_cooldown, time_error = self._is_time_in_window(last_time, cooldown_start, cooldown_end)
                if time_error:
                    self._log_debug("[NDS][FILTER] cooldown time parse failed: %s", time_error)
                elif in_cooldown:
                    analysis_result['signal'] = 'NONE'
                    self._append_reason(
                        reasons,
                        f"Cooling-off window active ({cooldown_start}-{cooldown_end} broker time)"
                    )

            if analysis_result.get('signal') in {'BUY', 'SELL'}:
                atr_spike_mult = float(settings.get("FLOW_ATR_SPIKE_MULT", 3.0))
                market_metrics = analysis_result.get('market_metrics', {})
                atr_value = float(market_metrics.get("atr", 0.0) or 0.0)
                if atr_value > 0 and "high" in self.df.columns and "low" in self.df.columns:
                    last_range = float(self.df["high"].iloc[-1]) - float(self.df["low"].iloc[-1])
                    if last_range > (atr_spike_mult * atr_value):
                        analysis_result['signal'] = 'NONE'
                        self._append_reason(
                            reasons,
                            f"ATR spike filter ({last_range:.2f} > {atr_spike_mult:.2f}x ATR)"
                        )

            if scalping_mode:
                base_min_rvol = settings.get('MIN_RVOL_SCALPING', 0.75)
                current_rvol = analysis_result.get('market_metrics', {}).get('current_rvol', 1.0)
                structure = analysis_result.get('structure', {})
                structure_score = float(structure.get('structure_score', 0.0))
                adaptive_min_rvol = self._adaptive_min_rvol(base_min_rvol, structure_score)

                self._log_debug(
                    "[NDS][FILTER] rvol base=%.2f current=%.2f structure_score=%.1f adaptive=%.2f",
                    base_min_rvol,
                    current_rvol,
                    structure_score,
                    adaptive_min_rvol,
                )

                if current_rvol < adaptive_min_rvol:
                    analysis_result['signal'] = 'NONE'
                    self._append_reason(
                        reasons,
                        f"Volume too low (RVOL: {current_rvol:.2f} < {adaptive_min_rvol:.2f})"
                    )
                elif adaptive_min_rvol < base_min_rvol:
                    self._append_reason(
                        reasons,
                        f"Volume accepted due to high structure score ({structure_score:.1f})"
                    )

            current_confidence = analysis_result.get('confidence', 0.0)
            if scalping_mode:
                min_confidence = settings.get('SCALPING_MIN_CONFIDENCE', 45.0)
                confidence_type = "SCALPING"
            else:
                min_confidence = settings.get('MIN_CONFIDENCE', 50.0)
                confidence_type = "NORMAL"

            self._log_debug(
                "[NDS][FILTER] confidence current=%.1f%% %s_min=%.1f%%",
                current_confidence,
                confidence_type,
                min_confidence,
            )

            if current_confidence < min_confidence:
                analysis_result['signal'] = 'NONE'
                self._append_reason(
                    reasons,
                    f"Confidence too low ({current_confidence:.1f}% < {min_confidence}%)"
                )

            session_analysis = analysis_result.get('session_analysis', {})
            session_weight = float(session_analysis.get('weight', 0.5))
            min_session_weight = settings.get('MIN_SESSION_WEIGHT', 0.3)

            self._log_debug(
                "[NDS][FILTER] session weight=%.2f min=%.2f",
                session_weight,
                min_session_weight,
            )

            if session_weight < min_session_weight:
                analysis_result['signal'] = 'NONE'
                self._append_reason(
                    reasons,
                    f"Low session weight ({session_weight:.2f} < {min_session_weight})"
                )

            structure = analysis_result.get('structure', {})
            structure_score = float(structure.get('structure_score', 0.0))
            min_structure_score = settings.get('MIN_STRUCTURE_SCORE', 20.0)


            # --- Hard structure sanity gate (scalping) ---
            # اگر BOS/CHOCH نداریم و ADX پایین است، حتی با score متوسط اجازه سیگنال نده.
            market_metrics = analysis_result.get('market_metrics', {})
            adx_v = float(market_metrics.get('adx', 0.0) or 0.0)
            bos_v = structure.get('bos', 'NONE')
            choch_v = structure.get('choch', 'NONE')

            sanity_reject_adx_max = float(settings.get('SANITY_ADX_REJECT_MAX', 18.0))
            sanity_reject_structure = float(settings.get('SANITY_STRUCTURE_REJECT_SCORE', 40.0))

            if (not scalping_mode) and analysis_result.get('signal', 'NONE') != 'NONE':
                if bos_v == 'NONE' and choch_v == 'NONE' and adx_v < sanity_reject_adx_max and structure_score < sanity_reject_structure:
                    analysis_result['signal'] = 'NONE'
                    self._append_reason(
                        reasons,
                        f"Rejected: no BOS/CHOCH with low ADX (ADX: {adx_v:.1f}, structure: {structure_score:.1f})"
                    )
                    self._log_debug(
                        "[NDS][FILTER][SANITY] reject no_confirm adx=%.2f structure=%.2f (max_adx=%.2f min_struct=%.2f)",
                        adx_v,
                        structure_score,
                        sanity_reject_adx_max,
                        sanity_reject_structure,
                    )

            self._log_debug(
                "[NDS][FILTER] structure score=%.1f min=%.1f",
                structure_score,
                min_structure_score,
            )

            if structure_score < min_structure_score:
                if scalping_mode:
                    self._append_reason(
                        reasons,
                        f"Weak structure noted (Score: {structure_score:.1f} < {min_structure_score})"
                    )
                    self._log_debug(
                        "[NDS][FILTER] scalping mode keeps signal despite weak structure score=%.1f min=%.1f",
                        structure_score,
                        min_structure_score,
                    )
                else:
                    analysis_result['signal'] = 'NONE'
                    self._append_reason(
                        reasons,
                        f"Weak market structure (Score: {structure_score:.1f} < {min_structure_score})"
                    )

            if analysis_result.get('signal') in {'BUY', 'SELL'}:
                context = analysis_result.get("context") or {}
                signal_context = context.get("analysis_signal_context", {}) if isinstance(context, dict) else {}
                bias = signal_context.get("bias")
                strong_trend = bool(signal_context.get("strong_trend"))
                reversal_ok = bool(signal_context.get("reversal_ok"))
                if strong_trend and bias in {"BULLISH", "BEARISH"}:
                    counter_signal = (
                        (bias == "BULLISH" and analysis_result.get("signal") == "SELL")
                        or (bias == "BEARISH" and analysis_result.get("signal") == "BUY")
                    )
                    if counter_signal and not reversal_ok:
                        analysis_result['signal'] = 'NONE'
                        self._append_reason(
                            reasons,
                            f"Counter-trend blocked (bias={bias}, strong_trend={strong_trend}, reversal_ok={reversal_ok})",
                        )
                        self._log_debug(
                            "[NDS][FILTER] counter-trend blocked bias=%s strong=%s reversal_ok=%s",
                            bias,
                            strong_trend,
                            reversal_ok,
                        )

        analysis_result['reasons'] = reasons
        final_signal = analysis_result.get('signal', 'NONE')

        if original_signal != final_signal:
            self._log_debug(
                "[NDS][FILTER] changed signal original=%s final=%s reasons=%s",
                original_signal,
                final_signal,
                reasons,
            )
        else:
            self._log_debug("[NDS][FILTER] result signal=%s", final_signal)

        return analysis_result

    def _select_swing_anchor(self, structure: MarketStructure, signal: str) -> Optional[float]:
        if signal == "BUY" and structure.last_low:
            return structure.last_low.price
        if signal == "SELL" and structure.last_high:
            return structure.last_high.price
        return None

    def _build_entry_idea(
        self,
        signal: str,
        fvgs: List[FVG],
        order_blocks: List[OrderBlock],
        structure: MarketStructure,
        atr_value: float,
        entry_factor: float,
        current_price: float,
        adx_value: float,
    ) -> Dict[str, Optional[float]]:
        """ساخت ایده ورود/خروج بدون منطق اجرا یا مدیریت ریسک

        بهبودها:
        - جلوگیری از Entryهای مرگبار با فیلتر فاصله/سن زون (ATR-based) و ignore کردن stale zones
        - انتخاب زون بهینه با utility score (strength - dist_pen*dist_atr - age_pen*age_norm)
        - لاگ‌های دقیق: چرا کاندیدها رد شدند و چرا یک زون انتخاب شد
        """

        idea = {
            "entry_level": None,
            "entry_price": None,
            "entry_model": "MARKET",
            "entry_type": "LEGACY",
            "reason": None,
        }

        # ---------------------------------------------------------------------
        # ✅ متادیتا برای لاگ/CSV
        # ---------------------------------------------------------------------
        idea.setdefault("source", "NONE")              # FVG / OB / FALLBACK / NONE
        idea.setdefault("zone_meta", None)             # اطلاعات zone منتخب
        idea.setdefault("entry_distance_pips", None)   # فاصله entry تا current_price بر حسب pip
        idea.setdefault("entry_distance_points", None) # فاصله entry تا current_price بر حسب point
        idea.setdefault("entry_distance_usd", None)    # فاصله entry تا current_price بر حسب USD
        idea.setdefault("entry_distance_atr", None)    # فاصله entry تا current_price بر حسب ATR
        idea.setdefault("metrics", {})                 # متریک‌های کمکی

        # -------------------------
        # Logging: start
        # -------------------------
        self._log_debug(
            "[NDS][ENTRY_IDEA] start signal=%s atr=%.2f entry_factor=%.2f price=%.2f adx=%.2f",
            signal,
            float(atr_value) if atr_value else 0.0,
            float(entry_factor) if entry_factor else 0.0,
            float(current_price) if current_price else 0.0,
            float(adx_value) if adx_value else 0.0,
        )

        # Guard: ATR
        if atr_value is None or atr_value <= 0:
            idea["reason"] = "Invalid ATR for entry idea"
            self._log_info(
                "[NDS][ENTRY_IDEA][INVALID] %s | signal=%s price=%.2f",
                idea["reason"],
                signal,
                float(current_price),
            )
            return idea

        # -------------------------
        # Settings
        # -------------------------
        buffer_mult = float(self.GOLD_SETTINGS.get('ATR_BUFFER_MULTIPLIER', 0.5))
        min_rr = float(self.GOLD_SETTINGS.get('MIN_RR', 1.5))

        tp_multiplier = 1.5
        if adx_value is not None and adx_value > 40:
            tp_multiplier = 2.0
        elif adx_value is not None and adx_value > 25:
            tp_multiplier = 1.7

        # -------------------------
        # NEW: Entry policy (distance/age aware)
        # -------------------------
        # اگر خواستی بعداً scalping_mode را از بیرون پاس بدهی، اینجا را توسعه می‌دهیم.
        scalping_mode = True

        max_dist_atr = float(self.GOLD_SETTINGS.get(
            "ENTRY_MAX_DISTANCE_ATR_SCALP" if scalping_mode else "ENTRY_MAX_DISTANCE_ATR_NORMAL",
            0.8 if scalping_mode else 1.5
        ))
        max_age_bars = int(self.GOLD_SETTINGS.get(
            "ENTRY_MAX_AGE_BARS_SCALP" if scalping_mode else "ENTRY_MAX_AGE_BARS_NORMAL",
            60 if scalping_mode else 200
        ))
        dist_pen = float(self.GOLD_SETTINGS.get("ENTRY_DIST_PENALTY", 0.65))
        age_pen = float(self.GOLD_SETTINGS.get("ENTRY_AGE_PENALTY", 0.15))

        # حداقل اندازه zone بر اساس ATR (همان منطقی که در کد شما بود اما قابل تنظیم‌تر)
        min_zone_size_mult = float(self.GOLD_SETTINGS.get("ENTRY_MIN_ZONE_SIZE_ATR_MULT", 0.10))  # پیش‌فرض 0.1 ATR
        min_zone_size = float(atr_value) * min_zone_size_mult

        # برای جلوگیری از log spam
        max_log_candidates = int(self.GOLD_SETTINGS.get("ENTRY_LOG_CANDIDATE_LIMIT", 12))
        max_log_candidates = max(3, min(max_log_candidates, 50))

        idea["metrics"]["entry_policy"] = {
            "scalping_mode": bool(scalping_mode),
            "max_dist_atr": float(max_dist_atr),
            "max_age_bars": int(max_age_bars),
            "dist_pen": float(dist_pen),
            "age_pen": float(age_pen),
            "min_zone_size": float(min_zone_size),
            "min_zone_size_mult": float(min_zone_size_mult),
            "tp_multiplier": float(tp_multiplier),
            "min_rr": float(min_rr),
            "buffer_mult": float(buffer_mult),
        }

        # -------------------------
        # Helpers for selection
        # -------------------------
        try:
            last_idx = len(self.df) - 1
        except Exception:
            last_idx = 0

        def _strength_norm(x: float) -> float:
            try:
                x = float(x)
            except Exception:
                x = 0.0
            return x / (x + 1.0) if x > 0 else 0.0

        def _zone_utility(strength: float, dist_atr_val: float, age_bars_val: int) -> float:
            s = _strength_norm(strength)
            a = min(1.0, max(0.0, (age_bars_val / float(max_age_bars)) if max_age_bars > 0 else 1.0))
            return (1.0 * s) - (dist_pen * float(dist_atr_val)) - (age_pen * a)

        def _dist_atr_from_mid(mid_price: float) -> float:
            return abs(float(current_price) - float(mid_price)) / float(atr_value) if atr_value > 0 else 999.0

        # -------------------------
        # Fallback gating (همان منطق شما + لاگ‌ها حفظ شده)
        # -------------------------
        valid_fvgs = [f for f in fvgs if not getattr(f, "filled", False)]

        fallback_min_adx = float(self.GOLD_SETTINGS.get('FALLBACK_MIN_ADX', 30.0))
        fallback_min_structure = float(self.GOLD_SETTINGS.get('FALLBACK_MIN_STRUCTURE_SCORE', 80.0))
        normalized_structure_score = self._normalize_structure_score(getattr(structure, 'structure_score', 0.0))
        trend_value = structure.trend.value if getattr(structure, 'trend', None) else "RANGING"
        safe_adx = float(adx_value) if adx_value is not None else 0.0

        allow_fallback = (
            trend_value in {"UPTREND", "DOWNTREND"}
            and safe_adx >= fallback_min_adx
            and normalized_structure_score >= fallback_min_structure
        )

        self._log_debug(
            "[NDS][ENTRY_IDEA][FALLBACK] allow=%s trend=%s adx=%.1f(>=%.1f) structure_score=%.1f(>=%.1f)",
            allow_fallback,
            trend_value,
            safe_adx,
            fallback_min_adx,
            normalized_structure_score,
            fallback_min_structure,
        )
        self._log_info(
            "[NDS][ENTRY_IDEA][FALLBACK] allow=%s trend=%s adx=%.1f structure_score=%.1f thresholds(adx>=%.1f score>=%.1f)",
            allow_fallback,
            trend_value,
            safe_adx,
            normalized_structure_score,
            fallback_min_adx,
            fallback_min_structure,
        )

        idea["metrics"]["fallback_allow"] = bool(allow_fallback)
        idea["metrics"]["fallback_min_adx"] = float(fallback_min_adx)
        idea["metrics"]["fallback_min_structure"] = float(fallback_min_structure)
        idea["metrics"]["structure_score_norm"] = float(normalized_structure_score)
        idea["metrics"]["trend_value"] = trend_value
        idea["metrics"]["safe_adx"] = float(safe_adx)

        # -------------------------
        # Finalize trade (کد شما حفظ شده)
        # -------------------------
        def finalize_trade(entry: float, reason: str) -> Dict[str, Optional[float]]:
            """Finalize entry idea without execution/risk calculations."""
            if entry is None:
                idea["reason"] = "Invalid entry (None)"
                self._log_info("[NDS][ENTRY_IDEA][INVALID] %s", idea["reason"])
                return idea

            entry = float(entry)
            idea["entry_level"] = entry
            idea["entry_price"] = entry
            idea["reason"] = reason

            zone_meta = idea.get("zone_meta") or {}
            top = zone_meta.get("top")
            bottom = zone_meta.get("bottom")
            if top is None:
                top = zone_meta.get("high")
            if bottom is None:
                bottom = zone_meta.get("low")
            if top is not None and bottom is not None:
                try:
                    top_f = float(top)
                    bottom_f = float(bottom)
                    if bottom_f <= float(current_price) <= top_f:
                        idea["entry_model"] = "MARKET"
                    else:
                        idea["entry_model"] = "STOP"
                except Exception:
                    pass

            entry_metrics = self._calc_entry_distance_metrics(
                entry_price=entry,
                current_price=current_price,
                symbol_meta={"point_size": self._get_point_size()},
                atr_value=atr_value,
            )
            idea["entry_distance_pips"] = entry_metrics.get("dist_pips")
            idea["entry_distance_points"] = entry_metrics.get("dist_points")
            idea["entry_distance_usd"] = entry_metrics.get("dist_usd")
            idea["entry_distance_atr"] = entry_metrics.get("dist_atr")
            idea["metrics"]["point_size"] = entry_metrics.get("point_size")
            idea["metrics"]["dist_price"] = entry_metrics.get("dist_price")
            idea["metrics"]["dist_points"] = entry_metrics.get("dist_points")

            self._log_info(
                "[NDS][ENTRY_METRICS] entry=%.3f cur=%.3f point_size=%.4f dist_usd=%.4f dist_points=%.2f dist_pips=%.2f dist_atr=%s",
                entry,
                float(current_price),
                float(entry_metrics.get("point_size") or 0.0),
                float(entry_metrics.get("dist_usd") or 0.0),
                float(entry_metrics.get("dist_points") or 0.0),
                float(entry_metrics.get("dist_pips") or 0.0),
                entry_metrics.get("dist_atr"),
            )

            self._log_info(
                "[NDS][ENTRY_IDEA] finalized signal=%s src=%s entry=%.2f model=%s dist_pips=%s dist_atr=%s reason=%s",
                signal,
                idea.get("source", "NONE"),
                entry,
                idea.get("entry_model", "MARKET"),
                idea.get("entry_distance_pips"),
                idea.get("entry_distance_atr"),
                reason,
            )
            return idea

        # -------------------------
        # Swing anchor selection (حفظ)
        # -------------------------
        swing_anchor = self._select_swing_anchor(structure, signal)
        self._log_debug(
            "[NDS][ENTRY_IDEA][ANCHOR] signal=%s swing_anchor=%s atr=%.2f buffer_mult=%.2f",
            signal,
            f"{float(swing_anchor):.2f}" if swing_anchor else "None",
            float(atr_value),
            float(buffer_mult),
        )

        # -------------------------
        # Candidate evaluation + logging (NEW)
        # -------------------------
        def _log_reject(prefix: str, why: str, meta: dict) -> None:
            # برای جلوگیری از log spam، فقط debug
            try:
                self._log_debug(
                    "%s reject=%s | idx=%s strength=%.2f dist_atr=%.3f age=%s size=%.2f meta=%s",
                    prefix,
                    why,
                    meta.get("index"),
                    float(meta.get("strength", 0.0)),
                    float(meta.get("dist_atr", 999.0)),
                    meta.get("age"),
                    float(meta.get("size", 0.0)),
                    meta.get("brief", "-"),
                )
            except Exception:
                pass

        def _summarize_candidates(prefix: str, kept: list, rejected: dict) -> None:
            # kept: list of meta
            # rejected: {reason: count}
            try:
                rej_str = ", ".join([f"{k}:{v}" for k, v in rejected.items()]) if rejected else "-"
                self._log_info(
                    "%s candidates kept=%d rejected={%s}",
                    prefix,
                    len(kept),
                    rej_str,
                )
            except Exception:
                pass

        # -------------------------
        # BUY branch
        # -------------------------
        if signal == "BUY":

            # ---------- FVG BUY selection (distance/age/stale aware) ----------
            fvg_rej = {}
            fvg_kept = []
            fvg_debug_list = []  # محدود برای نمایش

            for f in valid_fvgs:
                # stale/filled safety
                if getattr(f, "filled", False):
                    fvg_rej["filled"] = fvg_rej.get("filled", 0) + 1
                    continue
                if getattr(f, "stale", False):
                    fvg_rej["stale"] = fvg_rej.get("stale", 0) + 1
                    continue

                # type must be bullish
                f_type = getattr(f, "type", None)
                f_type_val = f_type.value if hasattr(f_type, "value") else str(f_type)
                if f_type_val not in ("BULLISH_FVG", "BULLISH"):
                    fvg_rej["type"] = fvg_rej.get("type", 0) + 1
                    continue

                top = float(getattr(f, "top", 0.0))
                bottom = float(getattr(f, "bottom", 0.0))
                size = float(getattr(f, "size", abs(top - bottom)))
                strength = float(getattr(f, "strength", 0.0))
                idx = int(getattr(f, "index", 0))
                age = max(0, last_idx - idx)

                # pullback شرط شما: zone زیر قیمت (top < current_price)
                if not (top < float(current_price)):
                    fvg_rej["not_pullback"] = fvg_rej.get("not_pullback", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": None, "age": age, "size": size, "brief": f"top>=px ({top:.2f}>={current_price:.2f})"}
                    if len(fvg_debug_list) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][BUY][FVG]", "not_pullback", meta)
                    continue

                # min size
                if size < min_zone_size:
                    fvg_rej["too_small"] = fvg_rej.get("too_small", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": None, "age": age, "size": size, "brief": f"size<{min_zone_size:.2f}"}
                    if len(fvg_debug_list) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][BUY][FVG]", "too_small", meta)
                    continue

                mid = (top + bottom) / 2.0
                dist_atr_val = _dist_atr_from_mid(mid)

                if dist_atr_val > max_dist_atr:
                    fvg_rej["too_far"] = fvg_rej.get("too_far", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": dist_atr_val, "age": age, "size": size, "brief": f"dist_atr>{max_dist_atr:.2f}"}
                    if len(fvg_debug_list) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][BUY][FVG]", "too_far", meta)
                    continue

                if age > max_age_bars:
                    fvg_rej["too_old"] = fvg_rej.get("too_old", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": dist_atr_val, "age": age, "size": size, "brief": f"age>{max_age_bars}"}
                    if len(fvg_debug_list) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][BUY][FVG]", "too_old", meta)
                    continue

                util = _zone_utility(strength, dist_atr_val, age)
                meta = {
                    "obj": f,
                    "index": idx,
                    "top": top,
                    "bottom": bottom,
                    "mid": mid,
                    "size": size,
                    "strength": strength,
                    "age": age,
                    "dist_atr": dist_atr_val,
                    "util": util,
                }
                fvg_kept.append(meta)

            # لاگ خلاصه FVG BUY
            _summarize_candidates("[NDS][ENTRY_IDEA][BUY][FVG]", fvg_kept, fvg_rej)

            best_fvg_meta = max(fvg_kept, key=lambda m: float(m["util"])) if fvg_kept else None
            if best_fvg_meta:
                f = best_fvg_meta["obj"]

                # انتخاب نهایی را info کنیم
                self._log_info(
                    "[NDS][ENTRY_IDEA][BUY][FVG][PICK] idx=%s zone=[%.2f-%.2f] size=%.2f strength=%.2f age=%d dist_atr=%.3f util=%.4f",
                    best_fvg_meta["index"],
                    best_fvg_meta["bottom"],
                    best_fvg_meta["top"],
                    best_fvg_meta["size"],
                    best_fvg_meta["strength"],
                    best_fvg_meta["age"],
                    best_fvg_meta["dist_atr"],
                    best_fvg_meta["util"],
                )

                # متادیتا
                idea["source"] = "FVG"
                idea["entry_type"] = "FVG"
                idea["zone_meta"] = {
                    "type": (getattr(f, "type").value if hasattr(getattr(f, "type", None), "value") else str(getattr(f, "type", "FVG"))),
                    "index": best_fvg_meta["index"],
                    "top": float(best_fvg_meta["top"]),
                    "bottom": float(best_fvg_meta["bottom"]),
                    "size": float(best_fvg_meta["size"]),
                    "strength": float(best_fvg_meta["strength"]),
                    "age_bars": int(best_fvg_meta["age"]),
                    "dist_atr": float(best_fvg_meta["dist_atr"]),
                    "util": float(best_fvg_meta["util"]),
                }

                fvg_height = abs(float(best_fvg_meta["top"]) - float(best_fvg_meta["bottom"]))
                entry = float(best_fvg_meta["top"]) - (fvg_height * float(entry_factor))

                self._log_debug(
                    "[NDS][ENTRY_IDEA][PRE] src=FVG signal=BUY entry=%.2f fvg_height=%.2f",
                    entry,
                    fvg_height,
                )
                return finalize_trade(entry, f"Bullish FVG idea (util={best_fvg_meta['util']:.3f}, strength={best_fvg_meta['strength']:.1f})")

            # ---------- OB BUY selection (distance/age/stale aware) ----------
            ob_rej = {}
            ob_kept = []

            bullish_obs_all = [ob for ob in order_blocks if getattr(ob, "type", "") == "BULLISH_OB"]
            for ob in bullish_obs_all:
                if getattr(ob, "stale", False):
                    ob_rej["stale"] = ob_rej.get("stale", 0) + 1
                    continue

                high = float(getattr(ob, "high", 0.0))
                low = float(getattr(ob, "low", 0.0))
                size = abs(high - low)
                strength = float(getattr(ob, "strength", 0.0))
                idx = int(getattr(ob, "index", 0))
                age = max(0, last_idx - idx)

                # min size (برای OB هم اعمال می‌کنیم)
                if size < min_zone_size:
                    ob_rej["too_small"] = ob_rej.get("too_small", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": None, "age": age, "size": size, "brief": f"size<{min_zone_size:.2f}"}
                    if len(ob_kept) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][BUY][OB]", "too_small", meta)
                    continue

                mid = (high + low) / 2.0
                dist_atr_val = _dist_atr_from_mid(mid)

                if dist_atr_val > max_dist_atr:
                    ob_rej["too_far"] = ob_rej.get("too_far", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": dist_atr_val, "age": age, "size": size, "brief": f"dist_atr>{max_dist_atr:.2f}"}
                    if len(ob_kept) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][BUY][OB]", "too_far", meta)
                    continue

                if age > max_age_bars:
                    ob_rej["too_old"] = ob_rej.get("too_old", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": dist_atr_val, "age": age, "size": size, "brief": f"age>{max_age_bars}"}
                    if len(ob_kept) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][BUY][OB]", "too_old", meta)
                    continue

                util = _zone_utility(strength, dist_atr_val, age)
                ob_kept.append({
                    "obj": ob,
                    "index": idx,
                    "high": high,
                    "low": low,
                    "size": size,
                    "strength": strength,
                    "age": age,
                    "dist_atr": dist_atr_val,
                    "util": util,
                })

            _summarize_candidates("[NDS][ENTRY_IDEA][BUY][OB]", ob_kept, ob_rej)

            best_ob_meta = max(ob_kept, key=lambda m: float(m["util"])) if ob_kept else None
            if best_ob_meta:
                ob = best_ob_meta["obj"]
                self._log_info(
                    "[NDS][ENTRY_IDEA][BUY][OB][PICK] idx=%s range=[%.2f-%.2f] size=%.2f strength=%.2f age=%d dist_atr=%.3f util=%.4f",
                    best_ob_meta["index"],
                    best_ob_meta["low"],
                    best_ob_meta["high"],
                    best_ob_meta["size"],
                    best_ob_meta["strength"],
                    best_ob_meta["age"],
                    best_ob_meta["dist_atr"],
                    best_ob_meta["util"],
                )

                idea["source"] = "OB"
                idea["entry_type"] = "OB"
                idea["zone_meta"] = {
                    "type": getattr(ob, "type", "BULLISH_OB"),
                    "index": best_ob_meta["index"],
                    "high": float(best_ob_meta["high"]),
                    "low": float(best_ob_meta["low"]),
                    "size": float(best_ob_meta["size"]),
                    "strength": float(best_ob_meta["strength"]),
                    "age_bars": int(best_ob_meta["age"]),
                    "dist_atr": float(best_ob_meta["dist_atr"]),
                    "util": float(best_ob_meta["util"]),
                }

                entry = float(best_ob_meta["low"]) + (float(best_ob_meta["high"]) - float(best_ob_meta["low"])) * 0.3
                self._log_debug(
                    "[NDS][ENTRY_IDEA][PRE] src=OB signal=BUY entry=%.2f",
                    entry,
                )
                return finalize_trade(entry, f"Bullish OB idea (util={best_ob_meta['util']:.3f}, strength={best_ob_meta['strength']:.1f})")

            # ---------- No zone found => fallback handling (حفظ منطق) ----------
            self._log_debug(
                "[NDS][ENTRY_IDEA] no BUY zone found (or all rejected by policy) | valid_fvgs=%d obs=%d allow_fallback=%s",
                len(valid_fvgs),
                len(order_blocks),
                allow_fallback,
            )

            if not allow_fallback:
                idea["reason"] = (
                    f"No valid FVG/OB for BUY within policy; fallback disabled "
                    f"(ADX={safe_adx:.1f}/Min:{fallback_min_adx:.1f} or Structure={normalized_structure_score:.1f}/Min:{fallback_min_structure:.1f} or Trend={trend_value})"
                )
                self._log_debug("[NDS][ENTRY_IDEA] bullish fallback blocked reason=%s", idea["reason"])
                self._log_info("[NDS][ENTRY_IDEA][BLOCK] %s", idea["reason"])
                return idea

            idea["source"] = "FALLBACK"
            idea["entry_type"] = "FALLBACK"
            idea["zone_meta"] = {"mode": "strong_structure_only", "side": "BUY", "policy": {"max_dist_atr": max_dist_atr, "max_age_bars": max_age_bars}}

            fallback_entry = float(current_price) - (float(atr_value) * 0.3)
            self._log_info(
                "[NDS][ENTRY_IDEA][BUY][FALLBACK] entry=%.2f (atr=%.2f) allow_fallback=%s",
                fallback_entry,
                float(atr_value),
                allow_fallback,
            )
            return finalize_trade(fallback_entry, "Fallback bullish idea (only allowed in strong structure)")

        # -------------------------
        # SELL branch
        # -------------------------
        if signal == "SELL":

            # ---------- FVG SELL selection (distance/age/stale aware) ----------
            fvg_rej = {}
            fvg_kept = []

            for f in valid_fvgs:
                if getattr(f, "filled", False):
                    fvg_rej["filled"] = fvg_rej.get("filled", 0) + 1
                    continue
                if getattr(f, "stale", False):
                    fvg_rej["stale"] = fvg_rej.get("stale", 0) + 1
                    continue

                f_type = getattr(f, "type", None)
                f_type_val = f_type.value if hasattr(f_type, "value") else str(f_type)
                if f_type_val not in ("BEARISH_FVG", "BEARISH"):
                    fvg_rej["type"] = fvg_rej.get("type", 0) + 1
                    continue

                top = float(getattr(f, "top", 0.0))
                bottom = float(getattr(f, "bottom", 0.0))
                size = float(getattr(f, "size", abs(top - bottom)))
                strength = float(getattr(f, "strength", 0.0))
                idx = int(getattr(f, "index", 0))
                age = max(0, last_idx - idx)

                # pullback شرط شما برای SELL: bottom > current_price (زون بالای قیمت)
                if not (bottom > float(current_price)):
                    fvg_rej["not_pullback"] = fvg_rej.get("not_pullback", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": None, "age": age, "size": size, "brief": f"bottom<=px ({bottom:.2f}<={current_price:.2f})"}
                    if len(fvg_kept) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][SELL][FVG]", "not_pullback", meta)
                    continue

                if size < min_zone_size:
                    fvg_rej["too_small"] = fvg_rej.get("too_small", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": None, "age": age, "size": size, "brief": f"size<{min_zone_size:.2f}"}
                    if len(fvg_kept) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][SELL][FVG]", "too_small", meta)
                    continue

                mid = (top + bottom) / 2.0
                dist_atr_val = _dist_atr_from_mid(mid)

                if dist_atr_val > max_dist_atr:
                    fvg_rej["too_far"] = fvg_rej.get("too_far", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": dist_atr_val, "age": age, "size": size, "brief": f"dist_atr>{max_dist_atr:.2f}"}
                    if len(fvg_kept) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][SELL][FVG]", "too_far", meta)
                    continue

                if age > max_age_bars:
                    fvg_rej["too_old"] = fvg_rej.get("too_old", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": dist_atr_val, "age": age, "size": size, "brief": f"age>{max_age_bars}"}
                    if len(fvg_kept) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][SELL][FVG]", "too_old", meta)
                    continue

                util = _zone_utility(strength, dist_atr_val, age)
                fvg_kept.append({
                    "obj": f,
                    "index": idx,
                    "top": top,
                    "bottom": bottom,
                    "mid": mid,
                    "size": size,
                    "strength": strength,
                    "age": age,
                    "dist_atr": dist_atr_val,
                    "util": util,
                })

            _summarize_candidates("[NDS][ENTRY_IDEA][SELL][FVG]", fvg_kept, fvg_rej)

            best_fvg_meta = max(fvg_kept, key=lambda m: float(m["util"])) if fvg_kept else None
            if best_fvg_meta:
                f = best_fvg_meta["obj"]

                self._log_info(
                    "[NDS][ENTRY_IDEA][SELL][FVG][PICK] idx=%s zone=[%.2f-%.2f] size=%.2f strength=%.2f age=%d dist_atr=%.3f util=%.4f",
                    best_fvg_meta["index"],
                    best_fvg_meta["bottom"],
                    best_fvg_meta["top"],
                    best_fvg_meta["size"],
                    best_fvg_meta["strength"],
                    best_fvg_meta["age"],
                    best_fvg_meta["dist_atr"],
                    best_fvg_meta["util"],
                )

                idea["source"] = "FVG"
                idea["entry_type"] = "FVG"
                idea["zone_meta"] = {
                    "type": (getattr(f, "type").value if hasattr(getattr(f, "type", None), "value") else str(getattr(f, "type", "FVG"))),
                    "index": best_fvg_meta["index"],
                    "top": float(best_fvg_meta["top"]),
                    "bottom": float(best_fvg_meta["bottom"]),
                    "size": float(best_fvg_meta["size"]),
                    "strength": float(best_fvg_meta["strength"]),
                    "age_bars": int(best_fvg_meta["age"]),
                    "dist_atr": float(best_fvg_meta["dist_atr"]),
                    "util": float(best_fvg_meta["util"]),
                }

                fvg_height = abs(float(best_fvg_meta["top"]) - float(best_fvg_meta["bottom"]))
                entry = float(best_fvg_meta["bottom"]) + (fvg_height * float(entry_factor))

                self._log_debug(
                    "[NDS][ENTRY_IDEA][PRE] src=FVG signal=SELL entry=%.2f fvg_height=%.2f",
                    entry,
                    fvg_height,
                )
                return finalize_trade(entry, f"Bearish FVG idea (util={best_fvg_meta['util']:.3f}, strength={best_fvg_meta['strength']:.1f})")

            # ---------- OB SELL selection (distance/age/stale aware) ----------
            ob_rej = {}
            ob_kept = []

            bearish_obs_all = [ob for ob in order_blocks if getattr(ob, "type", "") == "BEARISH_OB"]
            for ob in bearish_obs_all:
                if getattr(ob, "stale", False):
                    ob_rej["stale"] = ob_rej.get("stale", 0) + 1
                    continue

                high = float(getattr(ob, "high", 0.0))
                low = float(getattr(ob, "low", 0.0))
                size = abs(high - low)
                strength = float(getattr(ob, "strength", 0.0))
                idx = int(getattr(ob, "index", 0))
                age = max(0, last_idx - idx)

                if size < min_zone_size:
                    ob_rej["too_small"] = ob_rej.get("too_small", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": None, "age": age, "size": size, "brief": f"size<{min_zone_size:.2f}"}
                    if len(ob_kept) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][SELL][OB]", "too_small", meta)
                    continue

                mid = (high + low) / 2.0
                dist_atr_val = _dist_atr_from_mid(mid)

                if dist_atr_val > max_dist_atr:
                    ob_rej["too_far"] = ob_rej.get("too_far", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": dist_atr_val, "age": age, "size": size, "brief": f"dist_atr>{max_dist_atr:.2f}"}
                    if len(ob_kept) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][SELL][OB]", "too_far", meta)
                    continue

                if age > max_age_bars:
                    ob_rej["too_old"] = ob_rej.get("too_old", 0) + 1
                    meta = {"index": idx, "strength": strength, "dist_atr": dist_atr_val, "age": age, "size": size, "brief": f"age>{max_age_bars}"}
                    if len(ob_kept) < max_log_candidates:
                        _log_reject("[NDS][ENTRY_IDEA][SELL][OB]", "too_old", meta)
                    continue

                util = _zone_utility(strength, dist_atr_val, age)
                ob_kept.append({
                    "obj": ob,
                    "index": idx,
                    "high": high,
                    "low": low,
                    "size": size,
                    "strength": strength,
                    "age": age,
                    "dist_atr": dist_atr_val,
                    "util": util,
                })

            _summarize_candidates("[NDS][ENTRY_IDEA][SELL][OB]", ob_kept, ob_rej)

            best_ob_meta = max(ob_kept, key=lambda m: float(m["util"])) if ob_kept else None
            if best_ob_meta:
                ob = best_ob_meta["obj"]

                self._log_info(
                    "[NDS][ENTRY_IDEA][SELL][OB][PICK] idx=%s range=[%.2f-%.2f] size=%.2f strength=%.2f age=%d dist_atr=%.3f util=%.4f",
                    best_ob_meta["index"],
                    best_ob_meta["low"],
                    best_ob_meta["high"],
                    best_ob_meta["size"],
                    best_ob_meta["strength"],
                    best_ob_meta["age"],
                    best_ob_meta["dist_atr"],
                    best_ob_meta["util"],
                )

                idea["source"] = "OB"
                idea["entry_type"] = "OB"
                idea["zone_meta"] = {
                    "type": getattr(ob, "type", "BEARISH_OB"),
                    "index": best_ob_meta["index"],
                    "high": float(best_ob_meta["high"]),
                    "low": float(best_ob_meta["low"]),
                    "size": float(best_ob_meta["size"]),
                    "strength": float(best_ob_meta["strength"]),
                    "age_bars": int(best_ob_meta["age"]),
                    "dist_atr": float(best_ob_meta["dist_atr"]),
                    "util": float(best_ob_meta["util"]),
                }

                entry = float(best_ob_meta["high"]) - (float(best_ob_meta["high"]) - float(best_ob_meta["low"])) * 0.3
                self._log_debug(
                    "[NDS][ENTRY_IDEA][PRE] src=OB signal=SELL entry=%.2f",
                    entry,
                )
                return finalize_trade(entry, f"Bearish OB idea (util={best_ob_meta['util']:.3f}, strength={best_ob_meta['strength']:.1f})")

            # ---------- No zone found => fallback handling ----------
            self._log_debug(
                "[NDS][ENTRY_IDEA] no SELL zone found (or all rejected by policy) | valid_fvgs=%d obs=%d allow_fallback=%s",
                len(valid_fvgs),
                len(order_blocks),
                allow_fallback,
            )

            if not allow_fallback:
                idea["reason"] = (
                    f"No valid FVG/OB for SELL within policy; fallback disabled "
                    f"(ADX={safe_adx:.1f}/Min:{fallback_min_adx:.1f} or Structure={normalized_structure_score:.1f}/Min:{fallback_min_structure:.1f} or Trend={trend_value})"
                    
                )
                self._log_debug("[NDS][ENTRY_IDEA] bearish fallback blocked reason=%s", idea["reason"])
                self._log_info("[NDS][ENTRY_IDEA][BLOCK] %s", idea["reason"])
                return idea

            idea["source"] = "FALLBACK"
            idea["entry_type"] = "FALLBACK"
            idea["zone_meta"] = {"mode": "strong_structure_only", "side": "SELL", "policy": {"max_dist_atr": max_dist_atr, "max_age_bars": max_age_bars}}

            fallback_entry = float(current_price) + (float(atr_value) * 0.3)
            self._log_info(
                "[NDS][ENTRY_IDEA][SELL][FALLBACK] entry=%.2f (atr=%.2f) allow_fallback=%s",
                fallback_entry,
                float(atr_value),
                allow_fallback,
            )
            return finalize_trade(fallback_entry, "Fallback bearish idea (only allowed in strong structure)")

        return idea


    def _build_analysis_result(
        self,
        signal: str,
        confidence: float,
        score: float,
        entry_price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        reasons: List[str],
        context: Dict[str, Any],
        timeframe: str,
        current_price: float,
    ) -> AnalysisResult:
        """ساخت خروجی نهایی تحلیل‌محور"""
        return AnalysisResult(
            signal=signal,
            confidence=round(confidence, 1),
            score=round(score, 1),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons[:12],
            context=context,
            timestamp=datetime.now().isoformat(),
            timeframe=timeframe,
            current_price=current_price,
        )

    def _create_error_result(
        self,
        error_message: str,
        timeframe: str,
        current_close: Optional[float],
    ) -> AnalysisResult:
        """ایجاد نتیجه خطا"""
        return AnalysisResult(
            signal="NONE",
            confidence=0.0,
            score=50.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasons=[f"Error: {error_message}"],
            context={"error": True},
            timestamp=datetime.now().isoformat(),
            timeframe=timeframe,
            current_price=current_close or 0.0,
        )


def analyze_gold_market(
    dataframe: pd.DataFrame,
    timeframe: str = 'M15',
    entry_factor: float = 0.25,
    config: Optional[Dict[str, Any]] = None,
    scalping_mode: bool = True,
) -> AnalysisResult:
    """
    تابع اصلی برای تحلیل بازار طلا و ایجاد confirming signal
    """
    if dataframe is None or dataframe.empty:
        return AnalysisResult(
            signal="NONE",
            confidence=0.0,
            score=50.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasons=["DataFrame is empty"],
            context={"error": True},
            timestamp=datetime.now().isoformat(),
            timeframe=timeframe,
            current_price=0.0,
        )

    try:
        mode = "Scalping" if scalping_mode else "Regular"
        logger.info(
            "[NDS][INIT] create analyzer mode=%s timeframe=%s candles=%s",
            mode,
            timeframe,
            len(dataframe),
        )

        analyzer = GoldNDSAnalyzer(dataframe, config=config)
        result = analyzer.generate_trading_signal(timeframe, entry_factor, scalping_mode)

        return result

    except Exception as e:
        logger.error("[NDS][RESULT] analysis failed: %s", str(e), exc_info=True)
        return AnalysisResult(
            signal="NONE",
            confidence=0.0,
            score=50.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasons=[f"Analysis error: {str(e)}"],
            context={"error": True},
            timestamp=datetime.now().isoformat(),
            timeframe=timeframe,
            current_price=0.0,
        )
