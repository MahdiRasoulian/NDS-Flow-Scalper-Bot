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
from datetime import datetime, time
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
    OrderBlock, FVG, MarketStructure, MarketTrend
)
from .indicators import IndicatorCalculator
from .smc import SMCAnalyzer
from .distance_utils import (
    calculate_distance_metrics,
    pips_to_price,
    price_to_points,
    points_to_pips,
)

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

        technical_settings = (self.config or {}).get('technical_settings', {})
        sessions_config = (self.config or {}).get('sessions_config', {})
        self.GOLD_SETTINGS = technical_settings.copy()
        self.TRADING_SESSIONS = sessions_config.get('BASE_TRADING_SESSIONS', {}).copy()
        self.timeframe_specifics = technical_settings.get('TIMEFRAME_SPECIFICS', {})
        self.swing_period_map = technical_settings.get('SWING_PERIOD_MAP', {})

        self.atr: Optional[float] = None

        self._apply_custom_config()
        self._validate_dataframe()
        self.timeframe = self._detect_timeframe()
        self._apply_timeframe_settings()

        self._score_hist = _SCORE_HIST

        self._log_info(
            "[NDS][INIT] initialized candles=%s timeframe=%s",
            len(self.df),
            self.timeframe,
        )

    def _apply_custom_config(self) -> None:
        """اعمال تنظیمات سفارشی از config خارجی"""
        if not self.config:
            return

        analyzer_config, sessions_config = self._extract_config_payload()
        if analyzer_config:
            self._apply_analyzer_settings(analyzer_config)

        if sessions_config:
            self._apply_sessions_config(sessions_config)

    def _extract_config_payload(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """دریافت تنظیمات مجاز آنالایزر و سشن‌ها از کانفیگ اصلی"""
        analyzer_config: Dict[str, Any] = {}
        sessions_config: Dict[str, Any] = {}

        if 'ANALYZER_SETTINGS' in self.config:
            analyzer_config = self.config.get('ANALYZER_SETTINGS', {}) or {}
        elif 'technical_settings' in self.config:
            analyzer_config = self.config.get('technical_settings', {}) or {}

        if 'DEBUG_ANALYZER' in self.config and 'DEBUG_ANALYZER' not in analyzer_config:
            analyzer_config = {**analyzer_config, 'DEBUG_ANALYZER': self.config.get('DEBUG_ANALYZER')}

        if 'TRADING_SESSIONS' in self.config:
            sessions_config = self.config.get('TRADING_SESSIONS', {}) or {}
        else:
            sessions_config = self.config.get('sessions_config', {}).get('TRADING_SESSIONS', {}) or {}

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

        self.df['time'] = pd.to_datetime(self.df['time'], utc=True)

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
        hour = last_time.hour

        session_info = self._is_valid_trading_session(last_time)

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

        spread = volume_analysis.get("spread", None)       # numeric if available
        max_spread = volume_analysis.get("max_spread", None)  # numeric if available

        untradable_reasons = []
        untradable = False

        # 1) invalid/parse failure from session_info (خیلی مهم)
        if not bool(session_info.get('is_valid', True)):
            untradable = True
            untradable_reasons.append("invalid_time")

        # 2) market status (optional)
        if market_status in ("CLOSED", "HALTED"):
            untradable = True
            untradable_reasons.append(f"market_status={market_status}")

        # 3) data ok flag (optional)
        if data_ok is False:
            untradable = True
            untradable_reasons.append("data_ok=False")

        # 4) spread sanity (optional)
        if spread is not None and max_spread is not None:
            try:
                if float(spread) > float(max_spread):
                    untradable = True
                    untradable_reasons.append(f"spread={float(spread):.4f}>max={float(max_spread):.4f}")
            except Exception:
                # اگر قابل تبدیل نبود، تصمیم‌گیری را به این معیار وابسته نکن
                pass

        is_active_session = (not untradable)

        analysis = SessionAnalysis(
            current_session=session_info.get('session', 'OTHER'),
            session_weight=session_info.get('weight', 0.5),
            weight=session_info.get('weight', 0.5),
            gmt_hour=hour,
            # سیاست جدید: active فقط برای untradable false می‌شود
            is_active_session=is_active_session,
            is_overlap=session_info.get('is_overlap', False),
            session_activity=session_activity,
            optimal_trading=session_info.get(
                'optimal_trading',
                session_info.get('weight', 0.5) >= 1.2
            )
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
        if not isinstance(check_time, datetime):
            try:
                check_time = pd.to_datetime(check_time)
            except (ValueError, TypeError):
                return {
                    'is_valid': False, 'is_overlap': False, 'session': 'INVALID',
                    'weight': 0.0, 'optimal_trading': False
                }

        hour = check_time.hour
        raw_sessions = self.TRADING_SESSIONS

        sessions = {}
        for config_name, data in raw_sessions.items():
            standard_name = SESSION_MAPPING.get(config_name, config_name)
            if standard_name not in sessions or data.get('weight', 0) > sessions[standard_name].get('weight', 0):
                sessions[standard_name] = data

        session_name = 'OTHER'
        session_weight = 0.5
        is_overlap = False

        def check_in_session(name: str) -> Tuple[bool, float]:
            session = sessions.get(name)
            if session and self._hour_in_session(hour, session.get('start', 0), session.get('end', 0)):
                return True, session.get('weight', 0.5)
            return False, 0.0

        in_overlap, weight = check_in_session('OVERLAP')
        if in_overlap:
            session_name = 'OVERLAP'
            session_weight = weight
            is_overlap = True
        else:
            in_london, weight = check_in_session('LONDON')
            if in_london:
                session_name = 'LONDON'
                session_weight = weight
            else:
                in_ny, weight = check_in_session('NEW_YORK')
                if in_ny:
                    session_name = 'NEW_YORK'
                    session_weight = weight
                else:
                    in_asia, weight = check_in_session('ASIA')
                    if in_asia:
                        session_name = 'ASIA'
                        session_weight = weight

        optimal_trading = session_weight >= 1.0

        return {
            # سیاست جدید: معتبر بودن timestamp
            'is_valid': True,
            'is_overlap': is_overlap,
            'session': session_name,
            'weight': session_weight,
            'hour': hour,
            'optimal_trading': optimal_trading
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
            volatility_state = self._normalize_volatility_state(self._determine_volatility(atr_v, atr_for_scoring))
            session_analysis = self._analyze_trading_sessions(volume_analysis)

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

            entry_price = entry_idea.get("entry_level")
            entry_type = entry_idea.get("entry_type", "NONE")
            entry_model = entry_idea.get("entry_model", "NONE")
            entry_source = entry_idea.get("zone")
            entry_context = entry_idea.get("metrics", {}) or {}
            if entry_idea.get("reason"):
                reasons.append(entry_idea["reason"])

            signal = entry_idea.get("signal", "NONE") or "NONE"
            result_payload["signal"] = signal

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
                result_payload["signal_context"] = entry_idea
                result_payload["context"]["signal_context"] = entry_idea
            except Exception as _ctx_e:
                self._log_debug("[NDS][SIGNAL][CONTEXT] failed to attach flow entry context: %s", _ctx_e)

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

    def _get_point_size(self) -> float:
        """Return configured point size with default."""
        try:
            return float(self.GOLD_SETTINGS.get("POINT_SIZE", 0.001))
        except Exception:
            return 0.001

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
        try:
            start_parts = [int(p) for p in str(start_str).split(":")]
            end_parts = [int(p) for p in str(end_str).split(":")]
            start_h, start_m = (start_parts + [0, 0])[:2]
            end_h, end_m = (end_parts + [0, 0])[:2]
            start_t = time(start_h, start_m)
            end_t = time(end_h, end_m)
            now_t = ts.time()
        except Exception as exc:
            return False, f"time_parse_failed:{exc}"

        if start_t == end_t:
            return False, "time_window_zero"
        if start_t < end_t:
            return (start_t <= now_t < end_t), None
        return (now_t >= start_t or now_t < end_t), None

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
        sl_max_pips = float(settings.get("SL_MAX_PIPS", 40.0))
        tp1_pips = float(settings.get("TP1_PIPS", 35.0))
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

        sl_points = price_to_points(sl_distance, point_size)
        sl_pips = points_to_pips(sl_points)
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
                "[NDS][SL_CLAMP] reason=%s sl_pips=%.2f bounds=[%.2f,%.2f]",
                clamp_reason,
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

        spread = volume_analysis.get("spread")
        spread_buffer = 0.0
        if spread is not None:
            try:
                spread_buffer = float(spread) * 1.2
            except Exception:
                spread_buffer = 0.0
        buffer_atr = float(settings.get("FLOW_STOP_BUFFER_ATR", 0.05))
        atr_buffer = float(atr_value) * buffer_atr if atr_value else 0.0
        buffer_price = max(spread_buffer, atr_buffer)

        breakers = getattr(structure, "breakers", []) or []
        ifvgs = getattr(structure, "inversion_fvgs", []) or []

        rejection_counts = {
            "too_far": 0,
            "too_old": 0,
            "too_many_touches": 0,
            "stale": 0,
            "ineligible": 0,
        }

        def _tier_candidates(zones: List[Dict[str, Any]], side: str, max_dist_atr: float, max_age: int) -> List[Dict[str, Any]]:
            candidates = []
            max_touches = int(settings.get("FLOW_MAX_TOUCHES", 2))
            for zone in zones:
                if not bool(zone.get("eligible", True)):
                    rejection_counts["ineligible"] += 1
                    continue
                z_type = str(zone.get("type", ""))
                if side == "BUY" and "BULLISH" not in z_type:
                    continue
                if side == "SELL" and "BEARISH" not in z_type:
                    continue
                if zone.get("stale"):
                    rejection_counts["stale"] += 1
                    continue
                mid = float(zone.get("mid", 0.0))
                if mid <= 0:
                    continue
                dist_atr = abs(float(current_price) - mid) / float(atr_value) if atr_value > 0 else 999.0
                age = int(zone.get("age_bars", 9999))
                if dist_atr > max_dist_atr:
                    rejection_counts["too_far"] += 1
                    continue
                if age > max_age:
                    rejection_counts["too_old"] += 1
                    continue
                touch_count = int(zone.get("touch_count", 1))
                if touch_count > max_touches:
                    rejection_counts["too_many_touches"] += 1
                    continue
                confidence = float(zone.get("confidence", 0.0))
                freshness_penalty = float(settings.get("FLOW_TOUCH_PENALTY", 0.55))
                util = confidence - (0.5 * dist_atr) - (0.1 * (age / max_age if max_age > 0 else 1.0))
                if touch_count > 1:
                    util *= freshness_penalty
                candidates.append({**zone, "dist_atr": dist_atr, "util": util})
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

        ifvg_max_dist = float(settings.get("IFVG_MAX_DIST_ATR", 0.5))
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
            len(brk_candidates),
            len(ifvgs),
            len(ifvg_candidates),
        )

        entry_type = "NONE"
        entry_source = None
        entry_model = "NONE"
        entry_level = None
        entry_reason = "no_zone"
        tier = "NONE"
        entry_confidence = 0.0
        reject_reason = "NO_ELIGIBLE_ZONE"

        if brk_candidates:
            pick = max(brk_candidates, key=lambda z: float(z.get("util", 0.0)))
            entry_type = "BREAKER"
            entry_source = pick
            entry_model, entry_level, entry_reason = _resolve_entry_model(signal, float(pick.get("top")), float(pick.get("bottom")))
            entry_reason = f"tier=A breaker retest ({entry_reason})"
            tier = "A"
            entry_confidence = float(pick.get("confidence", 0.0))
            reject_reason = None
        elif ifvg_candidates:
            pick = max(ifvg_candidates, key=lambda z: float(z.get("util", 0.0)))
            entry_type = "IFVG"
            entry_source = pick
            entry_model, entry_level, entry_reason = _resolve_entry_model(signal, float(pick.get("top")), float(pick.get("bottom")))
            entry_reason = f"tier=B inversion fvg ({entry_reason})"
            tier = "B"
            entry_confidence = float(pick.get("confidence", 0.0))
            reject_reason = None

        momentum_reason = None
        if entry_level is None:
            momo_adx_min = float(settings.get("MOMO_ADX_MIN", 35.0))
            time_start = settings.get("MOMO_TIME_START", "10:00")
            time_end = settings.get("MOMO_TIME_END", "18:00")
            session_only = bool(settings.get("MOMO_SESSION_ONLY", True))

            now_ts = self.df["time"].iloc[-1]
            in_window, time_error = self._is_time_in_window(now_ts, time_start, time_end)
            if time_error:
                momentum_reason = f"time_block:{time_error}"
                self._log_info("[NDS][FLOW_TIER] tier=C momentum blocked (%s)", momentum_reason)
                in_window = False

            session_name = self._normalize_session_name(
                getattr(session_analysis, "current_session", "OTHER")
            )
            session_ok = True
            if session_only:
                session_ok = session_name in {"LONDON", "NEW_YORK", "OVERLAP"}

            liquidity_ok = bool(getattr(session_analysis, "is_active_session", True))
            market_status = str(volume_analysis.get("market_status", "") or "").upper()
            if market_status in {"CLOSED", "HALTED"}:
                liquidity_ok = False

            strong_trend = bool(signal_context.get("strong_trend"))
            time_ok = in_window or (strong_trend and time_error is None)

            bias = str(signal_context.get("bias", "") or "")
            bias_ok = (bias == "BULLISH" and signal == "BUY") or (bias == "BEARISH" and signal == "SELL") or not bias

            if adx_value >= momo_adx_min and time_ok and (session_ok or strong_trend) and liquidity_ok and bias_ok:
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
                momentum_reason = (
                    f"adx={adx_value:.1f}/{momo_adx_min:.1f} "
                    f"window={in_window} session_ok={session_ok} liquidity_ok={liquidity_ok} bias_ok={bias_ok}"
                )
                self._log_debug("[NDS][FLOW_TIER] tier=C momentum skipped (%s)", momentum_reason)

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

        entry_context.update(
            {
                "point_size": point_size,
                "momentum_reason": momentum_reason,
                "buffer_price": buffer_price,
                "recent_low": recent_low,
                "recent_high": recent_high,
                "zone_rejections": dict(rejection_counts),
            }
        )

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
            point_size = self.GOLD_SETTINGS.get("POINT_SIZE", 0.001)
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
        spread = volume_analysis.get("spread", None)
        max_spread = volume_analysis.get("max_spread", None)

        untradable = False
        untradable_reasons = []

        if market_status in ("CLOSED", "HALTED"):
            untradable = True
            untradable_reasons.append(f"market_status={market_status}")

        if data_ok is False:
            untradable = True
            untradable_reasons.append("data_ok=False")

        if spread is not None and max_spread is not None:
            try:
                if float(spread) > float(max_spread):
                    untradable = True
                    untradable_reasons.append(f"spread={float(spread):.4f}>max={float(max_spread):.4f}")
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

            settings = self.GOLD_SETTINGS
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

        result = {
            "signal": signal,
            "confidence": confidence,
            "score": round(score, 1),
            "reasons": reasons[:8],
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
            },
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "timeframe": timeframe,
            "scalping_mode": scalping_mode,
        }

        if scalping_mode and atr_short_value:
            result["market_metrics"]["atr_short"] = round(atr_short_value, 2)

        return result

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
                symbol_meta={"point_size": self.GOLD_SETTINGS.get("POINT_SIZE", 0.001)},
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
                "[NDS][ENTRY_METRICS] entry=%.3f cur=%.3f point=%.4f dist_usd=%.4f dist_points=%.2f dist_pips=%.2f dist_atr=%s",
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
