"""
ربات اصلی معاملات NDS برای طلا - نسخه اسکلپینگ
نسخه یکپارچه با risk_manager.py
نسخه بهبود یافته با:
- سازگاری کامل با mt5_client.py (Real-Time + positions/pending)
- رفع مشکل عدم تشخیص بسته شدن پوزیشن (مانیتورینگ پیوسته + تشخیص pending vs position)
- یکپارچه‌سازی قرارداد خروجی Analyzer (AnalysisResult/dataclass -> dict)
- بهبود گزارش‌گیری lifecycle (OPEN/UPDATE/CLOSE) + تلگرام
- اصلاح ناسازگاری NONE/NEUTRAL و جلوگیری از ترید روی سیگنال خنثی
"""

import sys
import time
import atexit
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple


# پیدا کردن مسیر اصلی پروژه (nds_bot)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # nds_bot
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# اضافه کردن پوشه src به مسیرها
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

logger = logging.getLogger(__name__)

# ایمپورت‌های پروژه
from config.settings import config
from src.utils.telegram_notifier import TelegramNotifier

# ایمپورت مدیر ریسک اسکلپینگ
try:
    from src.trading_bot.risk_manager import create_scalping_risk_manager
    logger.info("✅ Scalping Risk Manager module imported successfully")
except ImportError as e:
    logger.critical(f"❌ Scalping Risk Manager module not found: {e}")
    print(f"\n❌ خطا: ماژول مدیریت ریسک اسکلپینگ یافت نشد")
    print(f"   لطفاً از وجود فایل‌های زیر اطمینان حاصل کنید:")
    print(f"   - src/trading_bot/risk_manager.py")
    sys.exit(1)

from src.trading_bot.state import BotState
from src.trading_bot.execution_reporting import generate_execution_report
from src.trading_bot.contracts import ExecutionEvent, PositionContract, compute_pips
from src.trading_bot.nds.distance_utils import (
    calculate_distance_metrics,
    resolve_point_size_with_source,
)
from src.trading_bot.config_utils import log_active_settings
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.realtime_price import RealTimePriceMonitor
from src.trading_bot.session_policy import evaluate_session, normalize_session_payload
from src.trading_bot.trade_tracker import TradeTracker
from src.trading_bot.position_state import PositionStateStore
from src.trading_bot.position_manager import PositionManager
from src.trading_bot.cooldown import (
    CooldownDecision,
    evaluate_cooldown,
    get_min_candles_between_trades,
    resolve_exposure_bias,
    summarize_positions,
    warn_deprecated_cooldown_settings,
)
from src.trading_bot.user_controls import UserControls
from src.ui.cli import print_banner, print_help, update_config_interactive

# ایمپورت آنالایزر جدید به صورت ماژولار
try:
    from src.trading_bot.nds.analyzer import GoldNDSAnalyzer
    try:
        # در برخی نسخه‌ها ممکن است تابع analyze_gold_market وجود نداشته باشد (فقط کلاس)
        from src.trading_bot.nds.analyzer import analyze_gold_market
    except Exception:
        analyze_gold_market = None
    logger.info("✅ NDS analyzer module imported successfully")
except ImportError as e:
    logger.critical(f"❌ NDS analyzer module not found: {e}")
    print(f"\n❌ خطا: ماژول تحلیل NDS یافت نشد")
    print(f"   لطفاً از وجود فایل‌های زیر اطمینان حاصل کنید:")
    print(f"   - src/trading_bot/nds/analyzer.py")
    print(f"   - src/trading_bot/nds/models.py")
    print(f"   - src/trading_bot/nds/indicators.py")
    print(f"   - src/trading_bot/nds/smc.py")
    sys.exit(1)

# متغیر گلوبال برای سیگنال هندلر (برای دسترسی از بیرون کلاس)
bot_state_global = None


class NDSBot:
    """
    کلاس اصلی ربات NDS برای اسکلپینگ طلا - نسخه Real-Time
    شامل منطق ترید، مدیریت چرخه تحلیل و ارتباط با کاربر
    """

    def __init__(self, mt5_client_cls, risk_manager_cls=None, analyzer_cls=None, analyze_func=None):
        global bot_state_global
        self.bot_state = BotState()
        bot_state_global = self.bot_state

        # DI
        self.MT5Client_cls = mt5_client_cls
        self.RiskManager_cls = risk_manager_cls

        self.analyze_market_func = analyze_func or analyze_gold_market

        self.mt5_client = None
        self.risk_manager = None
        self.config = config
        self.analyzer_config = None
        self.analyzer = None  # instance of GoldNDSAnalyzer (preferred)
        log_active_settings(self.config, logger)

        self.price_monitor = RealTimePriceMonitor(config=self.config, bot_state=self.bot_state, logger=logger)
        self.trade_tracker = TradeTracker()
        self.user_controls = UserControls(self, logger)
        self.position_manager = None

        self.notifier = TelegramNotifier()

        # مانیتورینگ معامله
        self._last_trade_monitor_ts = 0.0
        self._trade_monitor_interval_sec = 2.0  # هر 2 ثانیه بررسی تریدها (قابل تغییر)
        self._logged_deprecated_cooldown = False
        self._last_open_position_tickets: set[int] = set()
        self._latest_open_positions: List[PositionContract] = []
        self._latest_pending_orders: List[Dict[str, Any]] = []
        self.position_state_store = PositionStateStore(Path("reports") / "state" / "positions.json")
        self.position_state_store.load()
        self._shutdown_started = False
        self._cleanup_done = False

    # ----------------------------
    # Helpers
    # ----------------------------
    def _result_to_dict(self, result: Any) -> Dict[str, Any]:
        """سازگارکننده خروجی آنالایزر به قرارداد قابل مصرف توسط bot.py و risk_manager.

        پشتیبانی:
        - dict (همان را برمی‌گرداند)
        - AnalysisResult/dataclass/obj (از __dict__ + getattr استخراج می‌کند)

        نکته مهم:
        برخی خروجی‌ها (مثل AnalysisResult) ممکن است signal را به صورت property نگه دارند
        و در __dict__ نباشد؛ بنابراین علاوه بر __dict__، با getattr هم فیلدهای کلیدی را
        استخراج می‌کنیم تا signal=BUY/SELL/NONE از دست نرود.
        """
        if result is None:
            return {}

        # 1) Dict output
        if isinstance(result, dict):
            return self._normalize_result_dict(result)

        # 2) Object/dataclass output
        d: Dict[str, Any] = {}
        if hasattr(result, "__dict__"):
            d.update(dict(getattr(result, "__dict__", {}) or {}))

        # 3) Harvest critical fields even if they are properties (not in __dict__)
        def _safe_get(obj, name, default=None):
            try:
                return getattr(obj, name)
            except Exception:
                return default

        # signal
        signal_val = d.get("signal", None)
        if signal_val is None:
            for cand in ("signal", "final_signal", "trade_signal", "direction"):
                v = _safe_get(result, cand, None)
                if v is not None:
                    signal_val = v
                    break
        if signal_val is not None:
            d["signal"] = signal_val

        # confidence
        if d.get("confidence", None) is None:
            for cand in ("confidence", "conf", "confidence_pct"):
                v = _safe_get(result, cand, None)
                if v is not None:
                    d["confidence"] = v
                    break

        # score
        if d.get("score", None) is None:
            for cand in ("score", "normalized_score", "final_score"):
                v = _safe_get(result, cand, None)
                if v is not None:
                    d["score"] = v
                    break

        # optional context
        if d.get("indicators", None) is None:
            v = _safe_get(result, "indicators", None)
            if v is not None:
                d["indicators"] = v
        if d.get("session_analysis", None) is None:
            v = _safe_get(result, "session_analysis", None)
            if v is not None:
                d["session_analysis"] = v

        return self._normalize_result_dict(d)


    def _normalize_result_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw analyzer dict into bot contract."""
        if not isinstance(d, dict):
            return {}

        ctx = d.get("context") if isinstance(d.get("context"), dict) else {}

        # --- signal ---
        d["signal"] = self._normalize_signal(d.get("signal", "NONE"))

        # --- confidence normalization (0..100) ---
        conf = d.get("confidence", 0) or 0
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        # اگر خروجی 0..1 بود، به درصد تبدیل کن
        if 0.0 <= conf_f <= 1.0:
            conf_f *= 100.0
        d["confidence"] = conf_f

        # --- score normalization ---
        try:
            d["score"] = float(d.get("score", 0) or 0)
        except Exception:
            d["score"] = 0.0

        # --- reasons ---
        if not d.get("reasons"):
            if isinstance(ctx.get("reasons"), list):
                d["reasons"] = ctx["reasons"]
            else:
                d["reasons"] = []

        # --- market_metrics ---
        market_metrics = d.get("market_metrics") if isinstance(d.get("market_metrics"), dict) else {}
        if ctx:
            for src_k, dst_k in (
                ("atr", "atr"),
                ("atr_short", "atr_short"),
                ("adx", "adx"),
                ("plus_di", "plus_di"),
                ("minus_di", "minus_di"),
                ("rvol", "current_rvol"),
            ):
                if dst_k not in market_metrics and src_k in ctx:
                    market_metrics[dst_k] = ctx.get(src_k)
        d["market_metrics"] = market_metrics

        # --- spread normalization (from analyzer volume analysis) ---
        analysis_data = ctx.get("analysis_data") if isinstance(ctx.get("analysis_data"), dict) else {}
        volume_analysis = analysis_data.get("volume_analysis") if isinstance(analysis_data.get("volume_analysis"), dict) else {}
        if not volume_analysis and isinstance(d.get("analysis_data"), dict):
            maybe_volume = d.get("analysis_data", {}).get("volume_analysis")
            if isinstance(maybe_volume, dict):
                volume_analysis = maybe_volume

        spread_pips = volume_analysis.get("spread_pips", volume_analysis.get("spread"))
        spread_price = volume_analysis.get("spread_price")
        if d.get("spread_pips") is None and spread_pips is not None:
            d["spread_pips"] = spread_pips
        if d.get("spread_price") is None and spread_price is not None:
            d["spread_price"] = spread_price
        if d.get("spread_raw") is None and volume_analysis.get("spread_raw") is not None:
            d["spread_raw"] = volume_analysis.get("spread_raw")
        if d.get("spread_raw_unit") is None and volume_analysis.get("spread_raw_unit") is not None:
            d["spread_raw_unit"] = volume_analysis.get("spread_raw_unit")
        if d.get("spread") is None and spread_pips is not None:
            d["spread"] = spread_pips

        # --- canonical session/time keys ---
        if not d.get("session_analysis") and isinstance(ctx.get("session_analysis"), dict):
            d["session_analysis"] = ctx.get("session_analysis")
        session_analysis = d.get("session_analysis") if isinstance(d.get("session_analysis"), dict) else {}
        raw_session = d.get("session") or ctx.get("session")
        session_payload = {}
        if isinstance(raw_session, dict):
            session_payload = normalize_session_payload(raw_session)
        elif isinstance(session_analysis.get("session_decision"), dict):
            session_payload = normalize_session_payload(session_analysis.get("session_decision"))
        elif raw_session:
            session_payload = {"session_name": str(raw_session).upper()}

        if not d.get("ts_broker"):
            d["ts_broker"] = session_analysis.get("ts_broker")
        if not d.get("ts_broker") and ctx.get("ts_broker") is not None:
            d["ts_broker"] = ctx.get("ts_broker")
        if not d.get("time_mode"):
            d["time_mode"] = session_analysis.get("time_mode")
        if not d.get("time_mode") and ctx.get("time_mode") is not None:
            d["time_mode"] = ctx.get("time_mode")
        if d.get("broker_utc_offset_hours") is None:
            d["broker_utc_offset_hours"] = session_analysis.get("broker_utc_offset_hours")
        if d.get("broker_utc_offset_hours") is None and ctx.get("broker_utc_offset_hours") is not None:
            d["broker_utc_offset_hours"] = ctx.get("broker_utc_offset_hours")

        if not session_payload and d.get("ts_broker"):
            session_payload = normalize_session_payload(
                evaluate_session(d.get("ts_broker"), self.config).to_payload()
            )

        if session_payload:
            session_payload.setdefault("time_mode", d.get("time_mode"))
            session_payload.setdefault("broker_utc_offset_hours", d.get("broker_utc_offset_hours"))
            session_payload.setdefault("ts_broker", d.get("ts_broker"))
            session_payload.setdefault(
                "weight",
                session_analysis.get("weight", session_analysis.get("session_weight", 0.0)),
            )
            session_payload.setdefault("activity", session_analysis.get("session_activity", "NORMAL"))
            d["session"] = session_payload
            d.setdefault("session_name", session_payload.get("session_name"))
            if not d.get("session_decision"):
                d["session_decision"] = session_payload
        else:
            d["session"] = raw_session
        if not d.get("session_activity"):
            d["session_activity"] = session_analysis.get("session_activity")
        if not d.get("session_activity") and ctx.get("session_activity") is not None:
            d["session_activity"] = ctx.get("session_activity")

        # --- canonical indicator keys ---
        if d.get("adx") is None:
            d["adx"] = market_metrics.get("adx")
        if d.get("plus_di") is None:
            d["plus_di"] = market_metrics.get("plus_di")
        if d.get("minus_di") is None:
            d["minus_di"] = market_metrics.get("minus_di")

        # --- structure ---
        structure = d.get("structure") if isinstance(d.get("structure"), dict) else {}
        if ctx and isinstance(ctx.get("structure"), dict):
            structure.update(ctx["structure"])
        if "last_high" not in structure and "high" in structure:
            structure["last_high"] = structure.get("high")
        if "last_low" not in structure and "low" in structure:
            structure["last_low"] = structure.get("low")
        d["structure"] = structure

        # --- entry idea extraction ---
        entry_idea = ctx.get("entry_idea") if isinstance(ctx.get("entry_idea"), dict) else None
        if entry_idea:
            entry_level = entry_idea.get("entry_level") or entry_idea.get("entry_price")
            if d.get("entry_level") is None and entry_level is not None:
                d["entry_level"] = entry_level
            if d.get("entry_price") is None and entry_level is not None:
                d["entry_price"] = entry_level
            if d.get("entry_model") is None and entry_idea.get("entry_model") is not None:
                d["entry_model"] = entry_idea.get("entry_model")
            if entry_idea.get("reason") and not d.get("entry_reason"):
                d["entry_reason"] = entry_idea.get("reason")

        if d.get("entry_level") is None and d.get("entry_price") is not None:
            d["entry_level"] = d.get("entry_price")
        if d.get("entry_price") is None and d.get("entry_level") is not None:
            d["entry_price"] = d.get("entry_level")

        # --- session info ---
        if ctx and isinstance(ctx.get("session"), dict) and "session_analysis" not in d:
            d["session_analysis"] = ctx.get("session")

        # ✅ NEW: analyzer most likely stores it here
        if ctx and isinstance(ctx.get("session_analysis"), dict) and "session_analysis" not in d:
            d["session_analysis"] = ctx.get("session_analysis")


        if "scalping_mode" not in d:
            d["scalping_mode"] = True

        return d

    def _extract_adx_value(self, result: Dict[str, Any]) -> float:
        """Extract ADX from normalized analyzer payload with legacy fallbacks."""
        if not isinstance(result, dict):
            return 0.0
        adx_candidates = [
            result.get("adx"),
            result.get("ADX"),
            result.get("adx_value"),
        ]
        market_metrics = result.get("market_metrics") if isinstance(result.get("market_metrics"), dict) else {}
        adx_candidates.append(market_metrics.get("adx"))
        indicators = result.get("indicators") if isinstance(result.get("indicators"), dict) else {}
        adx_candidates.append(indicators.get("adx"))
        adx_candidates.append(indicators.get("adx_value"))
        adx_analysis = indicators.get("adx_analysis") if isinstance(indicators.get("adx_analysis"), dict) else {}
        adx_candidates.append(adx_analysis.get("adx"))

        for candidate in adx_candidates:
            if candidate is None:
                continue
            try:
                return float(candidate or 0.0)
            except Exception:
                continue
        return 0.0

    def _normalize_signal(self, signal_value: str) -> str:
        """
        استانداردسازی سیگنال:
        Analyzer: BUY/SELL/NONE
        برخی نسخه‌ها: NEUTRAL
        """
        sig = (signal_value or "NONE").upper()
        if sig == "NEUTRAL":
            sig = "NONE"
        if sig not in ("BUY", "SELL", "NONE"):
            # هر چیزی غیر از BUY/SELL را خنثی در نظر بگیر
            sig = "NONE"
        return sig

    def _get_bot_magic_numbers(self) -> List[int]:
        magic_config = self.config.get("trading_settings.MAGIC_NUMBERS", None)
        if magic_config is None:
            magic_config = self.config.get("trading_settings.MAGIC_NUMBER", None)
        if magic_config is None:
            return [202401, 202402, 202403]
        if isinstance(magic_config, (list, tuple, set)):
            return [int(m) for m in magic_config if str(m).strip() != ""]
        try:
            return [int(magic_config)]
        except (TypeError, ValueError):
            return [202401, 202402, 202403]

    def _filter_positions_by_magic(self, positions: List[PositionContract]) -> List[PositionContract]:
        allowed_magics = set(self._get_bot_magic_numbers())
        if not allowed_magics:
            return positions
        return [pos for pos in positions if int(pos.get("magic", 0) or 0) in allowed_magics]

    def _filter_orders_by_magic(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        allowed_magics = set(self._get_bot_magic_numbers())
        if not allowed_magics:
            return orders
        return [order for order in orders if int(order.get("magic", 0) or 0) in allowed_magics]

    def _log_positions_summary(self, positions: List[PositionContract]) -> None:
        total, buy_count, sell_count, tickets = summarize_positions(positions)
        logger.info(
            "[POSITIONS] Open positions summary: count=%s | BUY=%s SELL=%s | tickets=%s",
            total,
            buy_count,
            sell_count,
            tickets,
        )

    def _log_signal_state(self, *, bias: str, pending_count: int = 0) -> None:
        logger.info(
            "[SIGNAL_STATE] current_bias=%s last_trade_bar=%s last_trade_dir=%s pending=%s",
            bias,
            self.bot_state.last_trade_candle_time,
            self.bot_state.last_trade_direction,
            pending_count,
        )

    def _log_cooldown_decision(self, decision: CooldownDecision) -> None:
        details = decision.details or {}
        if decision.allowed:
            logger.info(
                "[COOLDOWN] last_trade_bar=%s current_bar=%s diff=%s min=%s => ALLOWED",
                details.get("last_trade_bar"),
                details.get("current_bar"),
                details.get("diff"),
                details.get("min_candles"),
            )
        else:
            logger.info(
                "[COOLDOWN] last_trade_bar=%s current_bar=%s diff=%s min=%s => BLOCKED",
                details.get("last_trade_bar"),
                details.get("current_bar"),
                details.get("diff"),
                details.get("min_candles"),
            )

    def _maybe_monitor_trades(self, force: bool = False):
        """مانیتورینگ معاملات با throttle برای جلوگیری از فشار"""
        if self._cleanup_done:
            return
        if self.mt5_client is not None and hasattr(self.mt5_client, "connected"):
            if not self.mt5_client.connected:
                return
        now = time.time()
        if force or (now - self._last_trade_monitor_ts) >= self._trade_monitor_interval_sec:
            self._last_trade_monitor_ts = now
            self._monitor_open_trades()

    # ----------------------------
    # Initialize
    # ----------------------------
    def initialize(self) -> bool:
        """🔥 مقداردهی اولیه ربات و اتصال به سرویس‌ها (نسخه Real-Time حرفه‌ای - اصلاح‌شده)"""
        logger.info("🔧 در حال راه‌اندازی ربات اسکلپینگ Real-Time...")
        print("\n🔧 در حال راه‌اندازی ربات اسکلپینگ Real-Time...")

        try:
            if not self._logged_deprecated_cooldown:
                warn_deprecated_cooldown_settings(self.config, logger)
                self._logged_deprecated_cooldown = True

            # ------------------------------------------------------------
            # 1) ایجاد MT5 Client
            # ------------------------------------------------------------
            if self.mt5_client is None:
                self.mt5_client = self.MT5Client_cls()
            if self.position_manager is None:
                self.position_manager = PositionManager(
                    self.config,
                    self.mt5_client,
                    trade_tracker=self.trade_tracker,
                    logger=logger,
                )

            # ------------------------------------------------------------
            # 2) اعمال تنظیمات Real-Time از bot_config.json روی MT5Client
            # ------------------------------------------------------------
            try:
                tick_interval = self.config.get("trading_settings.TICK_UPDATE_INTERVAL", 1.0)
            except Exception:
                tick_interval = 1.0

            # اگر MT5Client شما ConnectionConfig دارد، مستقیم همان را تنظیم کن
            try:
                if hasattr(self.mt5_client, "connection_config") and self.mt5_client.connection_config:
                    self.mt5_client.connection_config.real_time_enabled = True
                    self.mt5_client.connection_config.tick_update_interval = float(tick_interval)
                    logger.info(f"✅ Real-Time enabled | tick_update_interval={tick_interval}s")
                else:
                    logger.debug("ℹ️ MT5Client has no connection_config; skipping real-time config injection.")
            except Exception as e:
                logger.warning(f"⚠️ Unable to apply real-time settings to MT5Client: {e}")

            # ------------------------------------------------------------
            # 3) اتصال به MT5
            # ------------------------------------------------------------
            if not self.mt5_client.connect():
                logger.error("❌ اتصال به MT5 ناموفق بود.")
                print("❌ اتصال به MT5 ناموفق بود. فایل config/mt5_credentials.json و مسیر mt5_path را بررسی کنید.")
                return False

            # ------------------------------------------------------------
            # 4) آپدیت موجودی (Equity/Balance)
            # ------------------------------------------------------------
            account_info = self.mt5_client.get_account_info()
            if account_info:
                current_equity = account_info.get("equity") or account_info.get("balance") or 0.0
                try:
                    self.config.update_setting("ACCOUNT_BALANCE", current_equity)
                except Exception:
                    pass
                logger.info(f"💰 حساب متصل شد | موجودی لحظه‌ای: ${current_equity:,.2f}")
            else:
                logger.warning("⚠️ اتصال برقرار شد اما account_info دریافت نشد (mt5.account_info=None).")

            # ------------------------------------------------------------
            # 5) شروع مانیتورینگ قیمت (سیستم داخلی پروژه)
            # ------------------------------------------------------------
            if getattr(self, "price_monitor", None) is not None:
                try:
                    self.price_monitor.set_mt5_client(self.mt5_client)
                    self.price_monitor.start()
                except Exception as e:
                    logger.warning(f"⚠️ Price monitor failed to start: {e}")
            else:
                logger.debug("ℹ️ price_monitor not available on bot instance; skipping.")

            # ------------------------------------------------------------
            # 6) آماده‌سازی آنالایزر
            # ------------------------------------------------------------
            logger.info("🧠 در حال هماهنگ‌سازی تنظیمات آنالایزر با استراتژی SMC...")

            try:
                self.analyzer_config = self.config.get_full_config_for_analyzer()
            except Exception:
                # fallback حداقلی
                self.analyzer_config = {
                    "ANALYZER_SETTINGS": self.config.get("technical_settings", {}) if hasattr(self.config, "get") else {},
                    "TRADING_SESSIONS": {},
                }

            if "ANALYZER_SETTINGS" not in self.analyzer_config or not isinstance(self.analyzer_config.get("ANALYZER_SETTINGS"), dict):
                self.analyzer_config["ANALYZER_SETTINGS"] = self.config.get("technical_settings", {})

            tech_settings = self.analyzer_config.get("ANALYZER_SETTINGS", {}) or {}
            try:
                adx_weak = self.config.get("technical_settings.ADX_THRESHOLD_WEAK", tech_settings.get("ADX_THRESHOLD_WEAK"))
            except Exception:
                adx_weak = tech_settings.get("ADX_THRESHOLD_WEAK")

            analyzer_settings = {
                **tech_settings,
                "ADX_THRESHOLD_WEAK": adx_weak,
                "REAL_TIME_ENABLED": True,
                "USE_CURRENT_PRICE_FOR_ANALYSIS": True,
            }
            self.analyzer_config = {**self.analyzer_config, "ANALYZER_SETTINGS": analyzer_settings}

            # ------------------------------------------------------------
            # 6.1) ایجاد نمونه آنالایزر (GoldNDSAnalyzer) با کانفیگ نهایی
            # ------------------------------------------------------------
            self.analyzer = None  # مسیر A: analyzer instance نمی‌سازیم؛ از analyze_gold_market استفاده می‌کنیم
            logger.info("✅ Analyzer will be used via module function analyze_gold_market (no instance in initialize).")


            # ------------------------------------------------------------
            # 7) ایجاد Risk Manager
            # ------------------------------------------------------------
            try:
                scalping_config = {
                    "risk_manager_config": self.config.get_risk_manager_config() if hasattr(self.config, "get_risk_manager_config") else {},
                    "trading_rules": {
                        "MIN_CANDLES_BETWEEN_TRADES": self.config.get("trading_rules.MIN_CANDLES_BETWEEN_TRADES", 3),
                    },
                    "risk_settings": {
                        "MAX_PRICE_DEVIATION_PIPS": self.config.get("risk_settings.MAX_PRICE_DEVIATION_PIPS", 50.0),
                    },
                }
                self.risk_manager = create_scalping_risk_manager(overrides=scalping_config)
            except Exception as e:
                logger.error(f"⚠️ RiskManager creation failed: {e}", exc_info=True)
                # fallback حداقلی (اگر تابع اجازه دهد)
                self.risk_manager = create_scalping_risk_manager(overrides={})

            logger.info("✅ ربات با موفقیت عملیاتی شد.")
            try:
                self._log_real_time_status()
            except Exception:
                pass

            # ------------------------------------------------------------
            # 8) همگام‌سازی اولیه وضعیت معاملات با MT5 (در صورت وجود)
            # ------------------------------------------------------------
            try:
                logger.info("🔄 همگام‌سازی اولیه وضعیت معاملات با MT5...")
                self._maybe_monitor_trades(force=True)
            except Exception as e:
                logger.warning(f"⚠️ Initial trade sync failed: {e}")

            return True

        except Exception as e:
            logger.critical(f"❌ خطای بحرانی در Initialize: {e}", exc_info=True)
            return False

    def _log_real_time_status(self):
        """🔥 گزارش وضعیت واقعی و داینامیک سیستم"""
        try:
            symbol = self.config.get("trading_settings.SYMBOL")
            current_price = self.price_monitor.get_current_price(symbol)

            conn_status = "✅ Connected" if self.mt5_client and getattr(self.mt5_client, "connected", False) else "❌ Disconnected"
            monitor_status = "✅ Active" if getattr(self.mt5_client, "real_time_monitor", None) else "⚠️ Inactive"

            max_dev = self.config.get("risk_settings.MAX_PRICE_DEVIATION_PIPS")
            min_candles = self.config.get("trading_rules.MIN_CANDLES_BETWEEN_TRADES")

            status_report = f"""
        🎯 گزارش وضعیت لحظه‌ای سیستم (Real-Time)
        ==========================================
        📊 وضعیت اتصال: {conn_status}
        🎯 مانیتور قیمت MT5: {monitor_status}
        💰 اکوئیتی جاری: ${self.config.get('ACCOUNT_BALANCE'):,.2f}

        📈 وضعیت بازار لحظه‌ای:
        نماد: {symbol}
        Bid: {current_price.get('bid', 0.0):.2f} | Ask: {current_price.get('ask', 0.0):.2f}
        اسپرد: {current_price.get('spread', 0.0):.2f}
        منبع قیمت: {current_price.get('source', 'Unknown')}

        ⚙️ پارامترهای فعال معاملاتی:
        فاصله استراحت: {min_candles} کندل
        حداکثر انحراف مجاز: {max_dev} Pips
        آپدیت قیمت: هر {self.config.get('trading_settings.TICK_UPDATE_INTERVAL')} ثانیه
        ==========================================
        """
            logger.info(status_report)
            print(status_report)

        except Exception as e:
            logger.error(f"❌ خطا در تولید گزارش وضعیت: {e}", exc_info=True)

    

    def _log_trade_decision(
        self,
        *,
        cycle_number: int,
        analyzer_signal: str,
        final_signal: str,
        score: float,
        confidence: float,
        min_confidence: float,
        price: float,
        spread: float,
        spread_price: float | None = None,
        spread_raw: float | None = None,
        spread_raw_unit: str | None = None,
        session: str = "",
        session_weight: float = 0.0,
        session_activity: str = "",
        is_active_session: bool = True,
        untradable: bool = False,
        reject_reason: str = "-",
        reject_details: str = "-",
    ) -> None:
        """لاگ متمرکز و یک خطی برای تحلیل دقیق تصمیمات ربات"""
        try:
            spread_fields = ""
            if spread_price is not None:
                spread_fields += f" spread_price={spread_price:.5f}"
            if spread_raw is not None:
                unit = spread_raw_unit or ""
                spread_fields += f" spread_raw={float(spread_raw):.5f}{unit}"
            logger.info(
                f"[BOT][DECISION] cycle={cycle_number} analyzer={analyzer_signal} final={final_signal} "
                f"score={score:.1f} conf={confidence:.1f} min_conf={min_confidence:.1f} "
                f"price={price:.2f} spread_pips={spread:.5f}{spread_fields} sess={session} weight={session_weight:.2f} "
                f"act={is_active_session} untradable={untradable} reason={reject_reason} details={reject_details}"
            )
        except Exception:
            pass


    # ----------------------------
    # Main Cycle
    # ----------------------------
    def run_analysis_cycle(self, cycle_number: int):
        """اجرای یک سیکل کامل تحلیل بازار اسکلپینگ با فیلتر فاصله کندلی + مانیتورینگ ترید"""
        SYMBOL = self.config.get("trading_settings.SYMBOL")
        TIMEFRAME = self.config.get("trading_settings.TIMEFRAME")
        BARS_TO_FETCH = self.config.get("trading_settings.BARS_TO_FETCH")
        ENABLE_AUTO_TRADING = self.config.get("trading_settings.ENABLE_AUTO_TRADING")
        ENABLE_DRY_RUN = self.config.get("trading_settings.ENABLE_DRY_RUN")

        MIN_CANDLES_BETWEEN = get_min_candles_between_trades(self.config, default=0)
        MAX_POS = self.config.get("trading_rules.MAX_POSITIONS")
        WAIT_CLOSE = self.config.get("trading_rules.WAIT_FOR_CLOSE_BEFORE_NEW_TRADE")

        ENTRY_FACTOR = self.config.get("technical_settings.ENTRY_FACTOR")
        MIN_CONFIDENCE = self.config.get("technical_settings.SCALPING_MIN_CONFIDENCE")

        try:
            MIN_CONFIDENCE = float(MIN_CONFIDENCE or 0)
        except Exception:
            MIN_CONFIDENCE = 0.0
        if 0.0 <= MIN_CONFIDENCE <= 1.0:
            MIN_CONFIDENCE *= 100.0

        ACCOUNT_BALANCE = self.config.get("ACCOUNT_BALANCE")

        logger.info(
            f"⚙️ تنظیمات نهایی بارگذاری شد: Timeframe={TIMEFRAME}, Min_Candles_Between={MIN_CANDLES_BETWEEN}"
        )
        logger.info(f"\n{'='*60}\n🔄 سیکل تحلیل اسکلپینگ #{cycle_number} | ⏰ {datetime.now().strftime('%H:%M:%S')}\n{'='*60}")

        try:
            # 0) مانیتورینگ تریدها
            self._maybe_monitor_trades(force=True)

            logger.info(f"📥 دریافت داده‌های {SYMBOL}...")
            df = self.mt5_client.get_historical_data(symbol=SYMBOL, timeframe=TIMEFRAME, bars=BARS_TO_FETCH)

            if df is None or len(df) < 100:
                logger.error("❌ داده کافی دریافت نشد")
                return

            current_price = float(df['close'].iloc[-1])
            logger.info(f"✅ {len(df)} کندل دریافت شد | قیمت جاری: ${current_price:.2f}")

            logger.info("🧠 اجرای تحلیل NDS اسکلپینگ...")

            # --- اجرای تحلیل ---
            try:
                raw_result = self.analyze_market_func(
                    dataframe=df, timeframe=TIMEFRAME, entry_factor=ENTRY_FACTOR,
                    config=self.analyzer_config, scalping_mode=True
                )
                result = self._result_to_dict(raw_result)
            except Exception as e:
                logger.error(f"❌ خطا در اجرای تحلیل: {e}", exc_info=True)
                return

            if not result:
                logger.warning("❌ تحلیل نتیجه خالی برگرداند")
                return

            # --- استخراج داده‌ها برای لاگ تصمیم‌گیری ---
            analyzer_signal = self._normalize_signal(result.get("signal", "NONE"))
            score = float(result.get("score", 0.0) or 0.0)
            confidence = float(result.get("confidence", 0.0) or 0.0)
            current_spread = float(result.get("spread_pips") or result.get("spread") or 0.0)
            spread_price = result.get("spread_price")
            spread_raw = result.get("spread_raw")
            spread_raw_unit = result.get("spread_raw_unit")

            sess = result.get("session_analysis") or {}
            session_payload = result.get("session") if isinstance(result.get("session"), dict) else {}
            session_name = str(
                session_payload.get("session_name")
                or sess.get("current_session")
                or result.get("session")
                or "UNKNOWN"
            )
            session_weight = float(
                session_payload.get("weight")
                or sess.get("weight", sess.get("session_weight", 0.0))
                or 0.0
            )
            session_activity = str(
                session_payload.get("activity")
                or sess.get("session_activity", "")
            )
            is_active_session = bool(sess.get("is_active_session", True))
            untradable = bool(sess.get("untradable", False))
            untradable_reasons = str(sess.get("untradable_reasons", "-"))

            # --- استخراج ADX به شکل مقاوم (برای DEAD_ZONE override در RiskManager) ---
            adx_value = self._extract_adx_value(result)
            if result.get("adx") is None:
                result["adx"] = adx_value

            # --- منطق تصمیم‌گیری (Decision Logic) ---
            final_signal = analyzer_signal
            reject_reason = "-"
            reject_details = "-"

            if analyzer_signal not in ("BUY", "SELL"):
                final_signal = "NONE"
                reject_reason = "ANALYZER_NONE"
            elif confidence < MIN_CONFIDENCE:
                final_signal = "NONE"
                reject_reason = "CONF_TOO_LOW"
                reject_details = f"{confidence:.1f} < {MIN_CONFIDENCE:.1f}"
            elif untradable:
                final_signal = "NONE"
                reject_reason = "UNTRADABLE"
                reject_details = untradable_reasons
            elif not ENABLE_AUTO_TRADING:
                final_signal = "NONE"
                reject_reason = "AUTO_TRADING_OFF"

            # ثبت لاگ متمرکز تصمیم
            self._log_trade_decision(
                cycle_number=cycle_number, analyzer_signal=analyzer_signal, final_signal=final_signal,
                score=score, confidence=confidence, min_confidence=MIN_CONFIDENCE,
                price=current_price, spread=current_spread, spread_price=spread_price,
                spread_raw=spread_raw, spread_raw_unit=spread_raw_unit, session=session_name,
                session_weight=session_weight, session_activity=session_activity,
                is_active_session=is_active_session, untradable=untradable,
                reject_reason=reject_reason, reject_details=reject_details
            )

            # نمایش نتایج در کنسول (همان تابع قبلی شما)
            result["signal"] = final_signal  # آپدیت سیگنال نهایی در دیکشنری
            self.display_results(result)

            self.bot_state.analysis_count += 1
            self.bot_state.last_analysis = datetime.now()

            open_positions = self._latest_open_positions or self.get_open_positions_info()
            pending_orders = self._latest_pending_orders or self.get_pending_orders_info()
            exposure_bias = resolve_exposure_bias(open_positions)
            self._log_positions_summary(open_positions)
            self._log_signal_state(bias=exposure_bias, pending_count=len(pending_orders))

            if result.get("error"):
                logger.warning("⚠️ سیگنال حاوی خطاست")
                return

            # --- اجرای معامله ---
            if final_signal in ("BUY", "SELL"):
                symbol = result.get("symbol") or self.config.get("trading_settings.SYMBOL")
                pending_tickets = []
                if hasattr(self, "trade_tracker") and self.trade_tracker:
                    pending_tickets = self.trade_tracker.get_pending_close_tickets_for_symbol(symbol)
                if pending_tickets:
                    logger.warning(
                        "[TRADE_BLOCK] symbol=%s reason=pending_close tickets=%s",
                        symbol,
                        pending_tickets,
                    )
                    return

                cooldown_decision = evaluate_cooldown(
                    signal=final_signal,
                    min_candles_between=MIN_CANDLES_BETWEEN,
                    df=df,
                    open_positions=open_positions,
                    last_trade_candle_time=self.bot_state.last_trade_candle_time,
                    last_trade_direction=self.bot_state.last_trade_direction,
                )
                if cooldown_decision.reason == "MIXED_EXPOSURE":
                    logger.info(
                        "[COOLDOWN] mixed exposure detected (BUY/SELL open) => BLOCKED"
                    )
                    return
                if cooldown_decision.reason in {"COOLDOWN_BLOCKED", "COOLDOWN_OK"}:
                    self._log_cooldown_decision(cooldown_decision)
                if not cooldown_decision.allowed:
                    return

                # محدودیت تعداد پوزیشن
                open_positions = self.get_open_positions_count()
                if open_positions >= MAX_POS:
                    logger.info(f"⏸️ حداکثر پوزیشن باز ({MAX_POS}) تکمیل است.")
                    return

                # بررسی ریسک منیجر
                if self.risk_manager:
                    # ✅ CRITICAL: ست کردن ورودی‌های DEAD_ZONE override قبل از can_scalp()
                    try:
                        self.risk_manager.last_signal_confidence = float(confidence or 0.0)
                        self.risk_manager.last_adx = float(adx_value or 0.0)
                        self.risk_manager.last_session = str(session_name or "UNKNOWN")
                    except Exception:
                        # اگر به هر دلیل attribute set نشد، اجازه نمی‌دهیم سیکل کرش کند
                        pass

                    logger.info(
                        "[BOT][RISK][PAYLOAD] session=%s adx=%.1f ts_broker=%s",
                        session_name,
                        float(adx_value or 0.0),
                        result.get("ts_broker"),
                    )

                    # ✅ لاگ برای اثبات اینکه RiskManager مقدار واقعی دریافت کرده
                    logger.info(
                        "[RISK][SESSION] session=%s weight=%.2f act=%s untradable=%s | "
                        "signal=%s score=%.1f conf=%.1f adx=%.1f price=%.2f",
                        session_name,
                        session_weight,
                        is_active_session,
                        untradable,
                        final_signal,
                        score,
                        confidence,
                        adx_value,
                        current_price,
                    )
                    if session_name == "DEAD_ZONE":
                        logger.info(
                            "[RISK][DEAD_ZONE][INPUT] conf=%.1f adx=%.1f -> can_scalp() will evaluate override",
                            float(getattr(self.risk_manager, "last_signal_confidence", 0.0) or 0.0),
                            float(getattr(self.risk_manager, "last_adx", 0.0) or 0.0),
                        )

                    can_trade, reason = self.risk_manager.can_scalp(
                        account_equity=ACCOUNT_BALANCE,
                        signal_data=result,
                    )
                    if not can_trade:
                        logger.info(f"⏸️ ریسک منیجر: {reason}")
                        return

                if not ENABLE_DRY_RUN:
                    trade_success = self.execute_scalping_trade(result, df)
                    if trade_success:
                        self.bot_state.last_trade_candle_time = df["time"].iloc[-1]
                        self.bot_state.last_trade_wall_time = datetime.now()
                        self.bot_state.last_trade_time = self.bot_state.last_trade_wall_time
                        self.bot_state.last_trade_direction = final_signal
                        logger.info(f"✅ معامله ثبت شد")
                        self._maybe_monitor_trades(force=True)
                else:
                    logger.info("🔧 حالت آزمایشی فعال است (Dry Run)")
            else:
                # لاگ تکمیلی برای زمانی که سیگنال تایید نشد
                if reject_reason != "-":
                    logger.info(f"⏸️ تصمیم رد شد | دلیل: {reject_reason} | {reject_details}")

            self._maybe_monitor_trades(force=True)

        except Exception as e:
            logger.error(f"❌ خطا در سیکل تحلیل: {e}", exc_info=True)


    # ----------------------------
    # Positions/Pending (MT5)
    # ----------------------------
    def get_open_positions_count(self) -> int:
        """دریافت تعداد پوزیشن‌های باز برای نماد با سازگاری با MT5Client"""
        SYMBOL = self.config.get("trading_settings.SYMBOL")
        try:
            positions = self.mt5_client.get_open_positions(symbol=SYMBOL)
            positions = self._filter_positions_by_magic(positions or [])
            if not positions:
                logger.debug(f"No open positions found for {SYMBOL}")
                return 0
            count = len(positions)
            logger.debug(f"Found {count} open positions for {SYMBOL}")
            return count
        except Exception as e:
            logger.error(f"⚠️ خطا در دریافت تعداد پوزیشن‌های باز: {e}", exc_info=True)
            return 0

    def get_open_positions_info(self) -> List[PositionContract]:
        """
        دریافت اطلاعات دقیق پوزیشن‌های باز
        سازگار با mt5_client.get_open_positions که لیست dict برمی‌گرداند
        """
        SYMBOL = self.config.get("trading_settings.SYMBOL")
        try:
            positions: List[PositionContract] = self.mt5_client.get_open_positions(symbol=SYMBOL)
            positions = self._filter_positions_by_magic(positions or [])
            if not positions:
                logger.debug(f"No open positions information available for {SYMBOL}")
                return []

            for pos in positions:
                logger.debug(
                    "Position #%s: %s %.3f @ $%.2f | cur=$%.2f | pnl=$%.2f",
                    pos["position_ticket"],
                    pos["side"],
                    pos["volume"],
                    pos["entry_price"],
                    pos["current_price"],
                    pos["profit"],
                )

            logger.info(f"Retrieved {len(positions)} open positions for {SYMBOL}")
            return positions

        except Exception as e:
            logger.error(f"⚠️ خطا در دریافت اطلاعات پوزیشن‌ها: {e}", exc_info=True)
            return []

    def get_pending_orders_info(self) -> List[Dict[str, Any]]:
        """دریافت سفارش‌های pending برای جلوگیری از false-close در tracker"""
        SYMBOL = self.config.get("trading_settings.SYMBOL")
        try:
            if hasattr(self.mt5_client, "get_pending_orders"):
                orders = self.mt5_client.get_pending_orders(symbol=SYMBOL)
                return self._filter_orders_by_magic(orders or [])
            return []
        except Exception as e:
            logger.error(f"⚠️ خطا در دریافت pending orders: {e}", exc_info=True)
            return []

    # ----------------------------
    # Display
    # ----------------------------
    def display_results(self, result: dict):
        """نمایش نتایج تحلیل در کنسول (نسخه بهبود یافته با حفظ تمامی فیلدها)"""
        if not result:
            logger.warning("No results to display")
            print("❌ هیچ نتیجه‌ای برای نمایش وجود ندارد")
            return

        scalping_mode = bool(result.get("scalping_mode", False))
        mode_text = "اسکلپینگ" if scalping_mode else "معمولی"
        signal_value = result.get("signal", "NONE")
        confidence = result.get("confidence", 0)

        logger.info(f"📊 نمایش نتایج تحلیل {mode_text}: signal={signal_value}, confidence={confidence}%")

        if result.get("error"):
            print(f"\n❌ خطا در تحلیل:")
            for reason in result.get("reasons", ["Unknown error"]):
                print(f"   ⚠️  {reason}")
            return

        print(f"\n📊 نتایج تحلیل {mode_text}:")
        print(f"   signal: {signal_value}")
        print(f"   confidence: {confidence}%")
        print(f"   score: {result.get('score', 0)}/100")

        if scalping_mode:
            print(f"   mode: 🎯 SCALPING")

        market_metrics = result.get("market_metrics", {}) or {}
        if market_metrics:
            atr = market_metrics.get("atr")
            if atr and atr > 0:
                print(f"   ATR: ${atr:.2f}")

            if scalping_mode:
                atr_short = market_metrics.get("atr_short")
                if atr_short and atr_short > 0:
                    print(f"   ATR (Short): ${atr_short:.2f}")

            structure = result.get("structure", {}) or {}
            if structure:
                print(f"\n🏛️  ساختار بازار:")
                print(f"   روند: {structure.get('trend', 'N/A')}")
                print(f"   BOS: {structure.get('bos', 'N/A')}")
                print(f"   CHoCH: {structure.get('choch', 'N/A')}")

                if structure.get("last_high") and structure.get("last_low"):
                    print(f"   High: ${structure.get('last_high'):.2f}")
                    print(f"   Low: ${structure.get('last_low'):.2f}")

            adx = market_metrics.get("adx")
            if adx is not None:
                try:
                    adx_val = float(adx)
                    print(f"   ADX: {adx_val:.1f}")
                except Exception:
                    pass

                plus_di = market_metrics.get("plus_di", 0)
                minus_di = market_metrics.get("minus_di", 0)
                try:
                    print(f"   +DI: {float(plus_di):.1f} | -DI: {float(minus_di):.1f}")
                    trend_str = "صعودی" if plus_di > minus_di else ("نزولی" if minus_di > plus_di else "خنثی")
                    print(f"   قدرت روند: {trend_str}")
                except Exception:
                    pass

            vol_ratio = market_metrics.get("volatility_ratio")
            if vol_ratio:
                print(f"   نسبت نوسان: {vol_ratio:.2f}")

            rvol = market_metrics.get("current_rvol")
            if rvol:
                print(f"   حجم نسبی (RVOL): {rvol:.1f}x")

        reasons = result.get("reasons", []) or []
        if reasons:
            print(f"\n📈 دلایل:")
            for i, reason in enumerate(reasons[:3], 1):
                print(f"   {i}. {reason}")

        # پارامترهای ورود
        def _format_optional_price(value: Optional[float]) -> str:
            try:
                if value is None:
                    return "N/A"
                value_f = float(value)
            except Exception:
                return "N/A"
            if value_f == 0:
                return "N/A"
            return f"${value_f:.2f}"

        entry_price_val = result.get("entry_level") or result.get("entry_price")
        entry_model = result.get("entry_model") or result.get("entry_type") or "N/A"
        entry_context = result.get("entry_context") or (result.get("context") or {}).get("entry_context", {}) or {}
        tp1_target = entry_context.get("tp1_target_price")
        tp1_source = entry_context.get("tp1_target_source")
        if entry_price_val not in (None, 0, 0.0):
            print("\n🧾 ایده ورود (Analyzer - برنامه‌ریزی اولیه):")
            print(f"   قیمت ورود پیشنهادی: {_format_optional_price(entry_price_val)}")
            print(f"   مدل ورود: {entry_model}")
            if tp1_target:
                print(f"   هدف TP1 اولیه: {_format_optional_price(tp1_target)} ({tp1_source or 'N/A'})")
            print("   ℹ️  SL/TP نهایی بعد از RiskManager محاسبه می‌شود.")

        final_entry = result.get("final_entry")
        final_sl = result.get("final_stop_loss") or result.get("final_sl")
        final_tp = result.get("final_take_profit") or result.get("final_tp")
        if any(val not in (None, 0, 0.0) for val in (final_entry, final_sl, final_tp)):
            print("\n🧮 تصمیم نهایی (RiskManager):")
            print(f"   قیمت ورود نهایی: {_format_optional_price(final_entry)}")
            print(f"   استاپ لاس نهایی: {_format_optional_price(final_sl)}")
            print(f"   تیک پروفیت نهایی: {_format_optional_price(final_tp)}")

            rr = result.get("risk_reward_ratio")
            if rr:
                try:
                    print(f"   نسبت ریسک/پاداش: {float(rr):.2f}:1")
                except Exception:
                    pass

            pos_size = result.get("position_size")
            if pos_size:
                try:
                    print(f"   حجم معامله: {float(pos_size):.3f} لات")
                except Exception:
                    pass

        quality = result.get("quality")
        if quality:
            q_map = {"HIGH": "⭐⭐⭐", "MEDIUM": "⭐⭐", "LOW": "⭐"}
            print(f"   کیفیت سیگنال: {quality} {q_map.get(quality, '')}")

    # ----------------------------
    # Trade Execution
    # ----------------------------
    # ----------------------------
    # Trade Geometry Guards
    # ----------------------------
    def _resolve_entry_idea(self, signal_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
        """Resolve entry idea fields from analyzer payload (no SL/TP)."""
        entry_idea = (
            signal_data.get("entry_idea")
            if isinstance(signal_data.get("entry_idea"), dict)
            else None
        )
        entry_level = (
            (entry_idea or {}).get("entry_level")
            or (entry_idea or {}).get("entry_price")
            or signal_data.get("entry_level")
            or signal_data.get("entry_price")
        )
        entry_model = (
            (entry_idea or {}).get("entry_model")
            or signal_data.get("entry_model")
        )

        try:
            entry_level = float(entry_level) if entry_level is not None else None
        except Exception:
            entry_level = None

        if entry_model is not None:
            entry_model = str(entry_model).upper()

        return entry_level, entry_model

    def _validate_entry_idea(self, signal_data: Dict[str, Any]) -> Tuple[bool, str, Optional[float], Optional[str]]:
        """Validate analyzer entry idea without requiring SL/TP."""
        side = self._normalize_signal(signal_data.get("signal", "NONE"))
        if side not in ("BUY", "SELL"):
            return False, f"Invalid signal={side}", None, None

        entry_level, entry_model = self._resolve_entry_idea(signal_data)
        if entry_level is None:
            return False, "Missing entry_level from analyzer", None, entry_model
        if entry_model in (None, "", "NONE"):
            return False, "Missing entry_model from analyzer", entry_level, entry_model

        return True, "OK", entry_level, entry_model

    def _validate_trade_geometry(self, side: str, entry: Optional[float], sl: Optional[float], tp: Optional[float]) -> Tuple[bool, str]:
        """Hard validation of SL/TP placement relative to entry."""
        side = self._normalize_signal(side)
        if side not in ("BUY", "SELL"):
            return False, f"Invalid side={side}"

        if entry is None or sl is None or tp is None:
            return False, f"Missing levels: entry={entry} sl={sl} tp={tp}"

        if side == "BUY":
            if not (sl < entry < tp):
                return False, f"Invalid BUY geometry: sl={sl:.2f} entry={entry:.2f} tp={tp:.2f}"
        else:
            if not (tp < entry < sl):
                return False, f"Invalid SELL geometry: tp={tp:.2f} entry={entry:.2f} sl={sl:.2f}"

        return True, "OK"

    def execute_scalping_trade(self, signal_data: dict, df=None) -> bool:
        """🔥 اجرای معامله اسکلپینگ با Real-Time، ثبت گزارش و ذخیره JSON"""
        SYMBOL = self.config.get("trading_settings.SYMBOL")
        TIMEFRAME = self.config.get("trading_settings.TIMEFRAME")

        # ایمنی: سیگنال باید BUY/SELL باشد
        signal_data["signal"] = self._normalize_signal(signal_data.get("signal", "NONE"))
        if signal_data["signal"] not in ("BUY", "SELL"):
            logger.info(f"⏸️ execute_scalping_trade skipped | signal={signal_data.get('signal')}")
            return False

        logger.info(f"🚀 شروع فرآیند اجرای معامله اسکلپینگ Real-Time: signal={signal_data.get('signal', 'N/A')}")

        if signal_data.get("error"):
            logger.error(f"❌ سیگنال حاوی خطاست، معامله اجرا نمی‌شود: {signal_data.get('reasons', ['Unknown error'])}")
            print("❌ سیگنال حاوی خطاست، معامله اجرا نمی‌شود")
            return False

        # ------------------------------------------------------------
        # Guardrail #1: only validate entry idea (no SL/TP before RiskManager)
        # ------------------------------------------------------------
        try:
            ok, reason, entry_level, entry_model = self._validate_entry_idea(signal_data)
            if not ok:
                logger.error("❌ Missing/invalid entry idea from Analyzer | %s", reason)
                print(f"❌ ایده ورود نامعتبر است: {reason}")
                return False
            logger.info(
                "🧾 Entry idea validated | entry_level=%.2f entry_model=%s",
                float(entry_level or 0.0),
                entry_model,
            )
        except Exception as g_err:
            logger.warning(f"⚠️ Entry idea validation failed unexpectedly: {g_err}", exc_info=True)

        try:
            # قیمت Real-Time از PriceMonitor داخلی
            current_price_data = self.price_monitor.get_current_price(SYMBOL)
            if current_price_data.get("source") in ["no_data", "error"]:
                logger.error(f"❌ نمی‌توان قیمت Real-Time را دریافت کرد: {current_price_data.get('error', 'Unknown error')}")
                print("❌ دریافت قیمت Real-Time ناموفق")
                return False

            logger.info(
                "🎯 Real-Time Price Check: Symbol=%s Bid=%.2f Ask=%.2f Spread=%.2f Source=%s",
                SYMBOL,
                float(current_price_data.get("bid", 0.0) or 0.0),
                float(current_price_data.get("ask", 0.0) or 0.0),
                float(current_price_data.get("spread", 0.0) or 0.0),
                current_price_data.get("source", "Unknown"),
            )
            print(f"🎯 قیمت لحظه‌ای: Bid: {current_price_data['bid']:.2f}, Ask: {current_price_data['ask']:.2f}")

            market_metrics = signal_data.get("market_metrics", {}) or {}
            current_atr = market_metrics.get("atr")
            atr_short = market_metrics.get("atr_short")

            if current_atr:
                logger.info(f"📈 ATR معامله اسکلپینگ: ${float(current_atr):.2f}")
                print(f"📈 ATR معامله: ${float(current_atr):.2f}")

            if atr_short:
                logger.info(f"📈 ATR کوتاه‌مدت: ${float(atr_short):.2f}")
                print(f"📈 ATR کوتاه‌مدت: ${float(atr_short):.2f}")

            if not self.risk_manager:
                logger.error("❌ مدیر ریسک اسکلپینگ وجود ندارد")
                print("❌ مدیر ریسک اسکلپینگ وجود ندارد")
                return False

            live_snapshot = LivePriceSnapshot(
                bid=current_price_data["bid"],
                ask=current_price_data["ask"],
                timestamp=current_price_data.get("timestamp"),
            )

            config_payload = self.config.get_full_config()

            # ------------------------------------------------------------------
            # ✅ افزودنی (بدون حذف/تغییر منطق موجود):
            # لاگ خروجی Analyzer دقیقاً قبل از finalize_order
            # ------------------------------------------------------------------
            logger.info(
                "🧾 Analyzer out | signal=%s score=%s conf=%s entry=%s sl=%s tp=%s src=%s dist_pips=%s dist_atr=%s",
                signal_data.get("signal"),
                signal_data.get("score"),
                signal_data.get("confidence"),
                signal_data.get("entry_level") or signal_data.get("entry_price"),
                signal_data.get("stop_loss"),
                signal_data.get("take_profit"),
                (signal_data.get("context") or {}).get("entry_source"),
                (signal_data.get("context") or {}).get("entry_distance_pips"),
                (signal_data.get("context") or {}).get("entry_distance_atr"),
            )
            entry_context = (
                signal_data.get("entry_context")
                or (signal_data.get("context") or {}).get("entry_context", {})
                or {}
            )
            planned_entry = signal_data.get("entry_level") or signal_data.get("entry_price")
            planned_model = signal_data.get("entry_model") or signal_data.get("entry_type") or "N/A"
            print("\n🧾 ایده ورود (Analyzer - برنامه‌ریزی اولیه):")
            if planned_entry not in (None, 0, 0.0):
                print(f"   قیمت ورود پیشنهادی: {planned_entry:.2f}")
            print(f"   مدل ورود: {planned_model}")
            if entry_context.get("tp1_target_price"):
                print(
                    "   هدف TP1 اولیه: %.2f (%s)"
                    % (
                        float(entry_context.get("tp1_target_price")),
                        entry_context.get("tp1_target_source") or "N/A",
                    )
                )
            print("   ℹ️  SL/TP نهایی بعد از RiskManager محاسبه می‌شود.")
            # ------------------------------------------------------------------

            finalized = self.risk_manager.finalize_order(
                analysis=signal_data,
                live=live_snapshot,
                symbol=SYMBOL,
                config=config_payload,
            )

            # ------------------------------------------------------------------
            # ✅ افزودنی (بدون حذف/تغییر منطق موجود):
            # لاگ خروجی RiskManager بلافاصله بعد از finalize_order
            # ------------------------------------------------------------------
            logger.info(
                "🧮 RiskManager | allowed=%s order_type=%s deviation_pips=%.1f rr=%.2f lot=%.3f reason=%s notes=%s",
                finalized.is_trade_allowed,
                finalized.order_type,
                finalized.deviation_pips,
                finalized.rr_ratio,
                finalized.lot_size,
                finalized.reject_reason,
                finalized.decision_notes[-3:],
            )
            print(
                "🧮 تصمیم نهایی (RiskManager): entry=%.2f sl=%.2f tp=%.2f rr=%.2f"
                % (
                    float(finalized.entry_price),
                    float(finalized.stop_loss),
                    float(finalized.take_profit),
                    float(finalized.rr_ratio),
                )
            )
            # ------------------------------------------------------------------

            if not finalized.is_trade_allowed:
                logger.warning(f"❌ Trade rejected by RiskManager: {finalized.reject_reason}")
                print(f"❌ RiskManager معامله را رد کرد: {finalized.reject_reason}")
                return False


            # ------------------------------------------------------------
            # Guardrail #2: اعتبارسنجی هندسه معامله (Finalized output)
            # ------------------------------------------------------------
            try:
                ok2, reason2 = self._validate_trade_geometry(
                    signal_data.get("signal", "NONE"),
                    float(finalized.entry_price),
                    float(finalized.stop_loss),
                    float(finalized.take_profit),
                )
                if not ok2:
                    logger.error("❌ Invalid trade geometry after RiskManager finalize | %s", reason2)
                    print(f"❌ هندسه معامله بعد از RiskManager نامعتبر است: {reason2}")
                    return False
            except Exception as g2_err:
                logger.warning(f"⚠️ Post-finalize geometry validation failed unexpectedly: {g2_err}", exc_info=True)

            signal_data.update(
                {
                    "final_entry": finalized.final_entry or finalized.entry_price,
                    "final_stop_loss": finalized.final_stop_loss or finalized.stop_loss,
                    "final_take_profit": finalized.final_take_profit or finalized.take_profit,
                    "final_sl": finalized.final_sl or finalized.stop_loss,
                    "final_tp": finalized.final_tp or finalized.take_profit,
                    "final_volume": finalized.lot or finalized.lot_size,
                    "order_type": finalized.order_type,
                    "decision_reasons": finalized.decision_notes,
                }
            )

            order_type = finalized.order_type
            lot_size = finalized.lot_size
            price_deviation_pips = finalized.deviation_pips
            current_session = None
            scalping_grade = signal_data.get("quality", "N/A")
            if hasattr(self.risk_manager, "get_current_scalping_session"):
                current_session = self.risk_manager.get_current_scalping_session()

            decision_summary = (
                f"Decision Summary | type={order_type} "
                f"entry={finalized.entry_price:.2f} sl={finalized.stop_loss:.2f} "
                f"tp={finalized.take_profit:.2f} volume={finalized.lot_size:.3f} "
                f"deviation_pips={price_deviation_pips:.1f}"
            )
            logger.info(decision_summary)
            print(f"✅ {decision_summary}")
            if finalized.decision_notes:
                notes_text = " | ".join(finalized.decision_notes)
                logger.info(f"Decision Notes: {notes_text}")
                print(f"📝 {notes_text}")

            logger.info(f"📤 ارسال سفارش اسکلپینگ ({order_type}) به بروکر: {signal_data['signal']} {lot_size:.3f} لات")
            print(f"📤 ارسال سفارش اسکلپینگ ({order_type}) به بروکر...")

            order_result = None

            if str(order_type).lower() == "market":
                if hasattr(self.mt5_client, "send_order_real_time"):
                    order_result = self.mt5_client.send_order_real_time(
                        symbol=SYMBOL,
                        order_type=signal_data["signal"],
                        volume=lot_size,
                        sl_price=finalized.stop_loss,
                        tp_price=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
                else:
                    order_result = self.mt5_client.send_order(
                        symbol=SYMBOL,
                        order_type=signal_data["signal"],
                        volume=lot_size,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
            elif str(order_type).lower() == "stop":
                stop_order_type = f"{signal_data['signal']}_STOP"
                if hasattr(self.mt5_client, "send_stop_order"):
                    order_result = self.mt5_client.send_stop_order(
                        symbol=SYMBOL,
                        order_type=stop_order_type,
                        volume=lot_size,
                        stop_price=finalized.entry_price,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
                elif hasattr(self.mt5_client, "send_pending_order"):
                    order_result = self.mt5_client.send_pending_order(
                        symbol=SYMBOL,
                        order_type=stop_order_type,
                        volume=lot_size,
                        pending_price=finalized.entry_price,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
                else:
                    order_result = self.mt5_client.send_order_with_type(
                        symbol=SYMBOL,
                        order_type=stop_order_type,
                        volume=lot_size,
                        price=finalized.entry_price,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
            else:
                # Limit/Pending
                limit_order_type = f"{signal_data['signal']}_LIMIT"  # BUY_LIMIT / SELL_LIMIT

                if hasattr(self.mt5_client, "send_limit_order"):
                    order_result = self.mt5_client.send_limit_order(
                        symbol=SYMBOL,
                        order_type=limit_order_type,
                        volume=lot_size,
                        limit_price=finalized.entry_price,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
                elif hasattr(self.mt5_client, "send_pending_order"):
                    order_result = self.mt5_client.send_pending_order(
                        symbol=SYMBOL,
                        order_type=limit_order_type,
                        volume=lot_size,
                        pending_price=finalized.entry_price,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
                else:
                    order_result = self.mt5_client.send_order(
                        symbol=SYMBOL,
                        order_type=limit_order_type,
                        volume=lot_size,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                        order_action="LIMIT",
                    )

            # ارزیابی نتیجه
            success = False
            order_id = None
            position_ticket = None
            actual_entry_price = finalized.entry_price
            actual_sl = finalized.stop_loss
            actual_tp = finalized.take_profit

            if isinstance(order_result, dict):
                success = bool(order_result.get("success"))
                order_id = order_result.get("order_ticket") or order_result.get("ticket")
                position_ticket = order_result.get("position_ticket")
                actual_entry_price = float(order_result.get("entry_price", actual_entry_price) or actual_entry_price)
                actual_sl = float(order_result.get("stop_loss", actual_sl) or actual_sl)
                actual_tp = float(order_result.get("take_profit", actual_tp) or actual_tp)
                signal_data["execution_time"] = order_result.get("time", datetime.now())
            elif isinstance(order_result, int):
                success = True
                order_id = order_result

            needs_reconcile = any(
                value in (None, 0, 0.0) for value in (actual_entry_price, actual_sl, actual_tp)
            )
            if needs_reconcile and position_ticket:
                for attempt in range(3):
                    try:
                        positions = self.mt5_client.get_open_positions(symbol=SYMBOL)
                        matched = next(
                            (pos for pos in positions if pos.get("position_ticket") == position_ticket),
                            None,
                        )
                        if matched:
                            actual_entry_price = matched.get("entry_price") or actual_entry_price
                            actual_sl = matched.get("sl") or actual_sl
                            actual_tp = matched.get("tp") or actual_tp
                        if all(value not in (None, 0, 0.0) for value in (actual_entry_price, actual_sl, actual_tp)):
                            break
                    except Exception as reconcile_error:
                        logger.debug(
                            "⚠️ Fill reconciliation attempt %s failed: %s",
                            attempt + 1,
                            reconcile_error,
                        )
                    time.sleep(0.2)

            if success and order_id:
                signal_data["order_ticket"] = order_id
                signal_data["position_ticket"] = position_ticket
                entry_idea = signal_data.get("entry_idea") or (signal_data.get("context") or {}).get("entry_idea", {})
                entry_context = (
                    signal_data.get("entry_context")
                    or (signal_data.get("context") or {}).get("entry_context", {})
                    or {}
                )
                entry_source = entry_idea.get("zone") or signal_data.get("entry_source")
                entry_type = entry_idea.get("entry_type") or signal_data.get("entry_type")
                entry_tier = entry_idea.get("tier") or signal_data.get("tier")
                retest_reason = None
                touch_count = None
                if isinstance(entry_source, dict):
                    retest_reason = entry_source.get("retest_reason")
                    touch_count = entry_source.get("touch_count")

                point_size, point_source = resolve_point_size_with_source(
                    config_payload,
                    default=self.risk_manager._get_gold_spec("point"),
                )
                logger.info(
                    "[NDS][POINT_SIZE] point_size=%.4f source=%s",
                    point_size,
                    point_source,
                )
                spread_price = float(current_price_data.get("spread", 0.19) or 0.19)
                spread_metrics = calculate_distance_metrics(
                    entry_price=0.0,
                    current_price=spread_price,
                    point_size=point_size,
                )
                spread_pips = float(spread_metrics.get("dist_pips") or 0.0)
                sl_metrics = calculate_distance_metrics(
                    entry_price=actual_entry_price,
                    current_price=actual_sl,
                    point_size=point_size,
                )
                tp1_metrics = calculate_distance_metrics(
                    entry_price=actual_entry_price,
                    current_price=actual_tp,
                    point_size=point_size,
                )
                tp2_price = getattr(finalized, "tp2", None) or getattr(finalized, "take_profit2", None)
                tp2_metrics = (
                    calculate_distance_metrics(
                        entry_price=actual_entry_price,
                        current_price=tp2_price,
                        point_size=point_size,
                    )
                    if tp2_price
                    else {}
                )
                sl_pips = float(sl_metrics.get("dist_pips") or 0.0)
                tp1_pips = float(tp1_metrics.get("dist_pips") or 0.0)
                tp2_pips = float(tp2_metrics.get("dist_pips") or 0.0)
                flow_settings = config_payload.get("flow_settings", {}) if isinstance(config_payload, dict) else {}
                risk_settings = config_payload.get("risk_settings", {}) if isinstance(config_payload, dict) else {}
                trail_after_tp1 = bool(flow_settings.get("FLOW_TRAIL_AFTER_TP1", True))
                tp2_enabled = bool(risk_settings.get("TP2_ENABLED", True))
                tp_plan = "single_tp"
                if trail_after_tp1:
                    tp_plan = "trail_after_tp1"
                elif tp2_enabled and tp2_price is not None:
                    tp_plan = "tp1_tp2"

                logger.info(
                    "✅ [TRADE][OPEN] ticket=%s position=%s symbol=%s side=%s entry=%.2f sl=%.2f tp=%.2f vol=%.3f order_type=%s",
                    order_id,
                    position_ticket,
                    SYMBOL,
                    signal_data["signal"],
                    float(actual_entry_price),
                    float(actual_sl),
                    float(actual_tp),
                    float(lot_size),
                    order_type,
                )
                open_time_utc = datetime.utcnow()
                logger.info(
                    "[OPEN] order_ticket=%s position_ticket=%s symbol=%s magic=%s open_time=%s entry=%.2f sl=%.2f tp=%.2f volume=%.3f",
                    order_id,
                    position_ticket,
                    SYMBOL,
                    getattr(finalized, "magic", None),
                    open_time_utc.isoformat(),
                    float(actual_entry_price),
                    float(actual_sl),
                    float(actual_tp),
                    float(lot_size),
                )
                print(f"✅ سفارش {order_type} ارسال شد - ticket={order_id} | حجم: {lot_size:.3f} لات")

                open_event: ExecutionEvent = {
                    "event_type": "OPEN",
                    "event_time": open_time_utc,
                    "symbol": SYMBOL,
                    "order_ticket": order_id,
                    "position_ticket": position_ticket,
                    "side": signal_data["signal"],
                    "volume": lot_size,
                    "entry_price": actual_entry_price,
                    "exit_price": None,
                    "sl": actual_sl,
                    "tp": actual_tp,
                    "profit": None,
                    "pips": None,
                    "reason": None,
                    "metadata": {
                        "confidence": signal_data.get("confidence", 0),
                        "scalping_grade": scalping_grade,
                        "timeframe": TIMEFRAME,
                        "risk_amount": getattr(finalized, "risk_amount_usd", None),
                        "session": current_session,
                        "order_type": order_type,
                        "magic": getattr(finalized, "magic", None),
                        "comment": order_result.get("comment") if isinstance(order_result, dict) else None,
                        "price_deviation_pips": price_deviation_pips,
                        "market_metrics": market_metrics,
                        "decision_notes": finalized.decision_notes,
                        "analysis_snapshot": signal_data,
                        "rr_ratio": getattr(finalized, "rr_ratio", None),
                        "entry_type": entry_type,
                        "tier": entry_tier,
                        "retest_reason": retest_reason,
                        "touch_count": touch_count,
                        "dist_pips": entry_context.get("dist_pips"),
                        "sl_pips": sl_pips,
                        "tp1_pips": tp1_pips,
                        "tp2_pips": tp2_pips,
                        "spread_pips": spread_pips,
                        "point_size": point_size,
                        "tp2_price": tp2_price,
                        "tp1_price": actual_tp,
                        "tp_plan": tp_plan,
                        "tp2_enabled": tp2_enabled,
                        "trail_after_tp1": trail_after_tp1,
                    },
                }
                self.trade_tracker.add_trade_open(open_event)
                self.bot_state.add_trade(success=True)

                if df is None or df.empty:
                    self.bot_state.last_trade_wall_time = datetime.now()
                    self.bot_state.last_trade_time = self.bot_state.last_trade_wall_time
                    self.bot_state.last_trade_direction = signal_data.get("signal")

                if hasattr(self.risk_manager, "add_position"):
                    self.risk_manager.add_position(lot_size)

                generate_execution_report(
                    logger=logger,
                    event=open_event,
                    df=df,
                )

                try:
                    tg_payload = {
                        "signal": signal_data.get("signal"),
                        "entry_price": float(actual_entry_price),
                        "stop_loss": float(actual_sl),
                        "take_profit": float(actual_tp),
                        "confidence": float(signal_data.get("confidence") or 0.0),
                        "order_type": order_type,
                        "order_ticket": order_id,
                        "position_ticket": position_ticket,
                    }
                    self.notifier.send_signal_notification(params=tg_payload, symbol=SYMBOL)
                except Exception as t_err:
                    logger.warning(f"⚠️ خطای غیربحرانی در ارسال تلگرام: {t_err}", exc_info=True)

                self._maybe_monitor_trades(force=True)
                return True

            else:
                # این بخش قبلاً اجرا نمی‌شد، اکنون در بلوک else قرار گرفته است
                logger.error(f"❌ ارسال سفارش اسکلپینگ {order_type} ناموفق بود | result={order_result}")
                print(f"❌ ارسال سفارش اسکلپینگ {order_type} ناموفق بود")
                self.bot_state.add_trade(success=False)
                return False

        except Exception as e:
            logger.error(f"❌ خطا در اجرای معامله اسکلپینگ Real-Time: {e}", exc_info=True)
            print(f"❌ خطا در اجرای معامله اسکلپینگ Real-Time: {e}")
            self.bot_state.add_trade(success=False)
            return False


    def execute_trade(self, signal_data: dict, df=None) -> bool:
        """سازگاری با کدهای قدیمی"""
        return self.execute_scalping_trade(signal_data, df)

    # ----------------------------
    # Trade Monitoring (Open/Close)
    # ----------------------------
    def _monitor_open_trades(self):
        """
        🔥 مانیتورینگ هوشمند:
        - بروزرسانی سود/قیمت برای پوزیشن‌های باز
        - جلوگیری از false-close با بررسی pending orders
        - تشخیص بسته‌شدن پوزیشن و ارسال نتیجه به تلگرام
        """
        if not hasattr(self, "trade_tracker"):
            return

        try:
            SYMBOL = self.config.get("trading_settings.SYMBOL")
            open_positions = self.get_open_positions_info()
            pending_orders = self.get_pending_orders_info()
            self._latest_open_positions = open_positions
            self._latest_pending_orders = pending_orders
            self.bot_state.active_positions = open_positions

            self._log_positions_summary(open_positions)
            if self.position_manager is not None:
                try:
                    self.position_manager.manage_positions(open_positions)
                except Exception as manager_error:
                    logger.error(
                        "⚠️ خطا در PositionManager: %s",
                        manager_error,
                        exc_info=True,
                    )
            exposure_bias = resolve_exposure_bias(open_positions)
            if open_positions or pending_orders:
                self.bot_state.active_signal_direction = exposure_bias if exposure_bias != "NONE" else None
            else:
                self.bot_state.active_signal_direction = None
            self._log_signal_state(bias=exposure_bias, pending_count=len(pending_orders))

            now = datetime.utcnow()
            added_count, updated_count, closed_candidates = self.trade_tracker.reconcile_with_open_positions(
                open_positions,
                reconcile_time=now,
            )
            state_result = self.position_state_store.reconcile(open_positions, now)

            if added_count or updated_count:
                logger.debug("🔄 Trade reconciliation: added=%s updated=%s", added_count, updated_count)

            closed_records = {
                record.get("trade_identity", {}).get("position_ticket"): record
                for record in closed_candidates
            }
            current_tickets = {pos["position_ticket"] for pos in open_positions}
            closed_tickets = self._last_open_position_tickets - current_tickets
            self._last_open_position_tickets = current_tickets

            def _record_from_state(state_record: Dict[str, Any]) -> Dict[str, Any]:
                if not state_record:
                    return {}
                opened_at = state_record.get("open_time") or now
                open_event: ExecutionEvent = {
                    "event_type": "OPEN",
                    "event_time": opened_at,
                    "symbol": state_record.get("symbol"),
                    "order_ticket": None,
                    "position_ticket": state_record.get("position_ticket"),
                    "side": state_record.get("side"),
                    "volume": state_record.get("volume"),
                    "entry_price": state_record.get("entry_price"),
                    "exit_price": None,
                    "sl": None,
                    "tp": None,
                    "profit": None,
                    "pips": None,
                    "reason": None,
                    "metadata": {
                        "magic": state_record.get("magic"),
                        "comment": state_record.get("comment"),
                        "detected_by": "recovery_scan",
                    },
                }
                return self.trade_tracker.normalize_trade_record(
                    {
                        "trade_identity": {
                            "order_ticket": None,
                            "position_ticket": state_record.get("position_ticket"),
                            "symbol": state_record.get("symbol"),
                            "magic": state_record.get("magic"),
                            "comment": state_record.get("comment"),
                            "opened_at": opened_at,
                            "detected_by": "recovery_scan",
                        },
                        "open_event": open_event,
                        "last_update_event": open_event,
                        "close_event": {},
                        "status": state_record.get("status") or "OPEN",
                    }
                )

            state_closed_map = {
                int(record.get("position_ticket")): record
                for record in state_result.closed_positions
                if record.get("position_ticket")
            }

            for position_ticket in sorted(
                set(closed_records.keys()) | closed_tickets | set(state_closed_map.keys())
            ):
                if not position_ticket:
                    continue
                record = (
                    closed_records.get(position_ticket)
                    or self.trade_tracker.active_trades.get(position_ticket)
                    or _record_from_state(state_closed_map.get(position_ticket, {}))
                )
                if not record:
                    continue
                record = self.trade_tracker.normalize_trade_record(record)
                if not record:
                    continue
                identity = record.get("trade_identity", {})

                close_event = record.get("close_event") if isinstance(record.get("close_event"), dict) else {}
                first_seen = close_event.get("event_time") or now
                if self.trade_tracker.register_pending_close(position_ticket, record, first_seen):
                    logger.info(
                        "[CLOSE_DETECT] ticket=%s symbol=%s side=%s open_time=%s",
                        position_ticket,
                        identity.get("symbol"),
                        record.get("open_event", {}).get("side"),
                        identity.get("opened_at"),
                    )

            close_config = self.config.get("close_tracking", {}) if hasattr(self.config, "get") else {}
            lookback_hours = int(close_config.get("HISTORY_LOOKBACK_HOURS", 72))
            timeout_sec = float(close_config.get("CLOSE_CONFIRM_TIMEOUT_SEC", 180))
            backoff_base = float(close_config.get("CLOSE_RETRY_BACKOFF_SEC", 5))
            backoff_max = float(close_config.get("CLOSE_RETRY_BACKOFF_MAX_SEC", 30))

            pending_checks, pending_timeouts = self.trade_tracker.get_pending_close_candidates(
                now=now,
                base_backoff_sec=backoff_base,
                max_backoff_sec=backoff_max,
                timeout_sec=timeout_sec,
            )

            for position_ticket, payload in pending_checks:
                record = payload.get("record", {})
                identity = record.get("trade_identity", {})
                open_event = record.get("open_event", {})
                opened_at = identity.get("opened_at")
                retries = int(payload.get("retries") or 0)
                from_time = (opened_at - timedelta(hours=1 + retries)) if opened_at else None
                to_time = now + timedelta(minutes=1 + retries)
                history = self.mt5_client.get_position_history(
                    position_ticket,
                    lookback_hours=lookback_hours,
                    symbol=identity.get("symbol"),
                    magic=identity.get("magic"),
                    open_time=opened_at,
                    volume=open_event.get("volume"),
                    side=open_event.get("side"),
                    from_time=from_time,
                    to_time=to_time,
                )
                if not history or not history.get("close_time"):
                    self.trade_tracker.mark_pending_attempt(position_ticket, now)
                    logger.warning(
                        "[CLOSE_PENDING] ticket=%s retries=%s backoff=%.1fs",
                        position_ticket,
                        payload.get("retries"),
                        min(backoff_base * (2 ** int(payload.get("retries") or 0)), backoff_max),
                    )
                    continue

                symbol = identity.get("symbol") or SYMBOL
                side = record.get("open_event", {}).get("side")
                entry_price = record.get("open_event", {}).get("entry_price")
                exit_price = history.get("exit_price") or record.get("last_update_event", {}).get("metadata", {}).get("current_price")
                profit = history.get("total_profit")
                close_time = history.get("close_time")
                reason = history.get("reason")
                duration_sec = None
                opened_at = identity.get("opened_at")
                if opened_at and close_time:
                    duration_sec = (close_time - opened_at).total_seconds()

                point_size, point_source = resolve_point_size_with_source(
                    self.config,
                    default=self.risk_manager._get_gold_spec("point"),
                )
                logger.info(
                    "[NDS][POINT_SIZE] point_size=%.4f source=%s",
                    point_size,
                    point_source,
                )
                pips_val = compute_pips(symbol, entry_price or 0.0, exit_price or 0.0, config_payload=self.config)

                close_event: ExecutionEvent = {
                    "event_type": "CLOSE",
                    "event_time": close_time or datetime.utcnow(),
                    "symbol": symbol,
                    "order_ticket": identity.get("order_ticket"),
                    "position_ticket": position_ticket,
                    "side": side,
                    "volume": record.get("open_event", {}).get("volume"),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "sl": record.get("open_event", {}).get("sl"),
                    "tp": record.get("open_event", {}).get("tp"),
                    "profit": profit,
                    "pips": pips_val,
                    "reason": reason,
                    "metadata": {"history": history, "duration_sec": duration_sec},
                }

                self.trade_tracker.close_trade_event(close_event)
                generate_execution_report(logger=logger, event=close_event)

                logger.info(
                    "[CLOSE] ticket=%s closed_time=%s pnl=%.2f reason=%s",
                    position_ticket,
                    close_time,
                    float(profit or 0.0),
                    reason or "Manual/Other",
                )

                if hasattr(self, "notifier") and self.notifier is not None:
                    try:
                        self.notifier.send_trade_close_notification(
                            symbol=symbol,
                            signal_type=side or "Unknown",
                            profit_usd=float(profit or 0.0),
                            pips=float(pips_val or 0.0),
                            reason=reason or "Manual/Other",
                        )
                        logger.info(f"✅ گزارش تلگرام برای بسته‌شدن پوزیشن #{position_ticket} ارسال شد.")
                    except Exception as tel_err:
                        logger.error(f"⚠️ خطا در ارسال نوتیفیکیشن تلگرام: {tel_err}", exc_info=True)

                state_record = state_closed_map.get(position_ticket)
                if state_record is not None:
                    state_record["status"] = "CLOSED"
                    state_record["close_time"] = close_time or now

            for position_ticket, payload in pending_timeouts:
                record = payload.get("record", {})
                identity = record.get("trade_identity", {})
                open_event = record.get("open_event", {})
                opened_at = identity.get("opened_at")
                close_event: ExecutionEvent = {
                    "event_type": "CLOSE_UNKNOWN",
                    "event_time": now,
                    "symbol": identity.get("symbol") or SYMBOL,
                    "order_ticket": identity.get("order_ticket"),
                    "position_ticket": position_ticket,
                    "side": open_event.get("side"),
                    "volume": open_event.get("volume"),
                    "entry_price": open_event.get("entry_price"),
                    "exit_price": None,
                    "sl": open_event.get("sl"),
                    "tp": open_event.get("tp"),
                    "profit": None,
                    "pips": None,
                    "reason": "history_timeout",
                    "metadata": {"duration_sec": (now - opened_at).total_seconds() if opened_at else None},
                }
                self.trade_tracker.finalize_unknown_close(position_ticket, close_event)
                generate_execution_report(logger=logger, event=close_event)
                logger.warning(
                    "[CLOSE_UNKNOWN] ticket=%s symbol=%s timeout=%.1fs",
                    position_ticket,
                    identity.get("symbol"),
                    timeout_sec,
                )

            for state_record, volume_delta in state_result.partial_positions:
                position_ticket = state_record.get("position_ticket")
                if not position_ticket:
                    continue
                opened_at = state_record.get("open_time")
                history = self.mt5_client.get_position_history(
                    int(position_ticket),
                    lookback_hours=lookback_hours,
                    symbol=state_record.get("symbol"),
                    magic=state_record.get("magic"),
                    open_time=opened_at,
                    volume=volume_delta,
                    side=state_record.get("side"),
                    from_time=state_record.get("last_reconcile"),
                    to_time=now,
                )
                partial_profit = history.get("total_profit") if history else None
                logger.info(
                    "[PARTIAL_CLOSE] ticket=%s volume_delta=%.3f pnl=%s match=%s",
                    position_ticket,
                    float(volume_delta or 0.0),
                    partial_profit,
                    history.get("match_method") if history else None,
                )
                if hasattr(self, "notifier") and self.notifier is not None:
                    try:
                        self.notifier.send_trade_partial_close_notification(
                            symbol=state_record.get("symbol") or SYMBOL,
                            signal_type=state_record.get("side") or "Unknown",
                            profit_usd=float(partial_profit or 0.0) if partial_profit is not None else 0.0,
                            pips=None,
                            reason="Partial Close",
                            volume=float(volume_delta or 0.0),
                        )
                    except Exception as tel_err:
                        logger.error(f"⚠️ خطا در ارسال نوتیفیکیشن تلگرام: {tel_err}", exc_info=True)

            self.position_state_store.save()

        except Exception as e:
            logger.error(f"⚠️ خطا در فرآیند مانیتورینگ معاملات: {e}", exc_info=True)

    # ----------------------------
    # Cleanup/Summary
    # ----------------------------
    def cleanup(self):
        """تمیزکاری منابع و قطع اتصال"""
        if self._cleanup_done:
            logger.info("🧹 cleanup already completed; skipping duplicate call.")
            return
        self._shutdown_started = True
        logger.info("🧹 در حال ذخیره وضعیت و تمیزکاری...")
        print("\n🧹 در حال ذخیره وضعیت...")

        try:
            # یک بار آخر مانیتورینگ تا closeها ثبت شوند
            self._maybe_monitor_trades(force=True)
        except Exception:
            pass

        try:
            if self.mt5_client and getattr(self.mt5_client, "connected", False):
                logger.info("قطع اتصال MT5...")
                self.mt5_client.disconnect()
                logger.info("✅ اتصال MT5 قطع شد")
                print("✅ اتصال MT5 قطع شد")
        except Exception as e:
            logger.error(f"⚠️ خطا در قطع اتصال MT5: {e}", exc_info=True)
            print(f"⚠️ خطا در قطع اتصال MT5: {e}")
        finally:
            self._cleanup_done = True

    def print_summary(self):
        """چاپ گزارش نهایی عملکرد"""
        logger.info("📊 چاپ گزارش نهایی عملکرد اسکلپینگ")

        stats = self.bot_state.get_statistics()
        hours = int(stats["runtime_seconds"] // 3600)
        minutes = int((stats["runtime_seconds"] % 3600) // 60)
        seconds = int(stats["runtime_seconds"] % 60)

        print(f"\n{'📊' * 20}")
        print("خلاصه نهایی اجرا اسکلپینگ")
        print(f"{'📊' * 20}")

        print(f"⏱️  زمان اجرا: {hours}:{minutes:02d}:{seconds:02d}")
        print(f"📈 تعداد تحلیل‌ها: {stats['analysis_count']}")
        print(f"💰 تعداد معاملات: {stats['trade_count']}")

        if stats["trade_count"] > 0:
            print(f"✅ معاملات موفق: {stats['successful_trades']}")
            print(f"❌ معاملات ناموفق: {stats['failed_trades']}")
            print(f"📊 نرخ موفقیت: {stats['success_rate']:.1f}%")

        print(f"💵 سود کل: ${stats['total_profit']:.2f}")
        print(f"📊 سود روزانه: ${stats['daily_pnl']:.2f}")
        print(f"📉 ضررهای متوالی: {stats['consecutive_losses']}")

        open_positions = self.get_open_positions_count()
        print(f"📊 پوزیشن‌های باز در پایان: {open_positions}")

        if open_positions > 0:
            logger.warning(f"⚠️  توجه: {open_positions} پوزیشن هنوز باز است")
            print(f"⚠️  توجه: {open_positions} پوزیشن هنوز باز است")

        logger.info("✅ ربات اسکلپینگ با موفقیت متوقف شد")
        print("\n✅ ربات اسکلپینگ با موفقیت متوقف شد")

    # ----------------------------
    # Main Loop
    # ----------------------------
    def run(self):
        """متد اصلی اجرای حلقه ربات"""
        logger.info("🚀 شروع اجرای ربات NDS اسکلپینگ")

        print_banner()
        print_help()

        atexit.register(self.cleanup)

        if not self._initialize_robot():
            return

        cycle_number = 0
        logger.info(f"🔁 شروع حلقه اصلی ربات اسکلپینگ، cycle_number={cycle_number}")

        try:
            self._run_main_loop(cycle_number)
        except KeyboardInterrupt:
            logger.info("🛑 توقف توسط کاربر (KeyboardInterrupt)")
            print("\n\n🛑 توقف توسط کاربر")
        finally:
            self._execute_shutdown_procedure()

    def _initialize_robot(self) -> bool:
        if not self.initialize():
            logger.critical("❌ راه‌اندازی ربات ناموفق بود")
            print("❌ راه‌اندازی ربات ناموفق بود")
            return False
        return True

    def _run_main_loop(self, start_cycle: int):
        cycle_number = start_cycle

        while self.bot_state.running:
            cycle_number += 1

            if not self.bot_state.paused:
                self._execute_analysis_cycle(cycle_number)

            if self.bot_state.running and not self.bot_state.paused:
                self._wait_for_next_cycle()

            self._handle_pause_mode()

    def _execute_analysis_cycle(self, cycle_number: int):
        logger.info(f"🔁 اجرای سیکل اسکلپینگ #{cycle_number}")
        self.run_analysis_cycle(cycle_number)

    def _wait_for_next_cycle(self):
        ANALYSIS_INTERVAL_MINUTES = self.config.get("trading_settings.ANALYSIS_INTERVAL_MINUTES")
        wait_time = ANALYSIS_INTERVAL_MINUTES * 60

        logger.info(f"⏳ انتظار برای سیکل بعدی: {ANALYSIS_INTERVAL_MINUTES} دقیقه")
        print(f"\n⏳ تحلیل بعدی در {ANALYSIS_INTERVAL_MINUTES} دقیقه...")
        print("   (فشار دهید: P=توقف, S=وضعیت, Q=خروج)")

        # در زمان انتظار، user_controls خودش loop دارد؛ بعد از پایان، مانیتور کنیم تا closeها از دست نرود
        self.user_controls.wait_with_controls(wait_time)
        self._maybe_monitor_trades(force=True)

    def _handle_pause_mode(self):
        while self.bot_state.paused and self.bot_state.running:
            logger.info("⏸️  ربات در حالت توقف")
            print("\n⏸️  ربات متوقف شده")
            print("   P=ادامه, Q=خروج, C=تنظیمات")

            action = self.user_controls.get_user_action()

            if action == "pause":
                self._resume_robot()
            elif action == "quit":
                self._stop_robot_during_pause()
                break
            elif action == "config":
                self._update_config_during_pause()
            else:
                # حتی در pause هم گهگاهی مانیتور معاملات را انجام بده
                self._maybe_monitor_trades()
                time.sleep(0.5)

    def _resume_robot(self):
        self.bot_state.paused = False
        logger.info("▶️  ربات ادامه یافت")
        print("▶️  ربات ادامه یافت")

    def _stop_robot_during_pause(self):
        self.bot_state.running = False
        logger.info("👋 درخواست خروج در حالت توقف")

    def _update_config_during_pause(self):
        logger.info("⚙️  به‌روزرسانی تنظیمات در حالت توقف")
        update_config_interactive()

    def _execute_shutdown_procedure(self):
        if self._shutdown_started and self._cleanup_done:
            logger.info("🧹 shutdown already completed; skipping duplicate shutdown procedure.")
            return
        self._shutdown_started = True
        logger.info("🧹 شروع فرآیند تمیزکاری نهایی")

        # ابتدا summary (هنوز اتصال برقرار است)
        try:
            self.print_summary()
        except Exception as e:
            logger.error(f"⚠️ خطا در چاپ summary: {e}", exc_info=True)

        # سپس cleanup
        self.cleanup()

        logger.info("🏁 پایان اجرای ربات اسکلپینگ")
