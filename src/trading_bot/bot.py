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
from datetime import datetime
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
    price_to_points,
    points_to_pips,
)
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.realtime_price import RealTimePriceMonitor
from src.trading_bot.trade_tracker import TradeTracker
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

        self.price_monitor = RealTimePriceMonitor(config=self.config, bot_state=self.bot_state, logger=logger)
        self.trade_tracker = TradeTracker()
        self.user_controls = UserControls(self, logger)

        self.notifier = TelegramNotifier()

        # مانیتورینگ معامله
        self._last_trade_monitor_ts = 0.0
        self._trade_monitor_interval_sec = 2.0  # هر 2 ثانیه بررسی تریدها (قابل تغییر)

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
            if d.get("entry_price") is None and entry_idea.get("entry_price") is not None:
                d["entry_price"] = entry_idea.get("entry_price")
            if d.get("stop_loss") is None and entry_idea.get("stop_loss") is not None:
                d["stop_loss"] = entry_idea.get("stop_loss")
            if d.get("take_profit") is None and entry_idea.get("take_profit") is not None:
                d["take_profit"] = entry_idea.get("take_profit")
            if entry_idea.get("reason") and not d.get("entry_reason"):
                d["entry_reason"] = entry_idea.get("reason")

        # --- session info ---
        if ctx and isinstance(ctx.get("session"), dict) and "session_analysis" not in d:
            d["session_analysis"] = ctx.get("session")

        # ✅ NEW: analyzer most likely stores it here
        if ctx and isinstance(ctx.get("session_analysis"), dict) and "session_analysis" not in d:
            d["session_analysis"] = ctx.get("session_analysis")


        if "scalping_mode" not in d:
            d["scalping_mode"] = True

        return d

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

    def _maybe_monitor_trades(self, force: bool = False):
        """مانیتورینگ معاملات با throttle برای جلوگیری از فشار"""
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
            # ------------------------------------------------------------
            # 1) ایجاد MT5 Client
            # ------------------------------------------------------------
            if self.mt5_client is None:
                self.mt5_client = self.MT5Client_cls()

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
                        "MIN_CANDLES_BETWEEN": self.config.get("trading_rules.MIN_CANDLES_BETWEEN", 3),
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
            min_candles = self.config.get("trading_rules.MIN_CANDLES_BETWEEN")

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
            logger.info(
                f"[BOT][DECISION] cycle={cycle_number} analyzer={analyzer_signal} final={final_signal} "
                f"score={score:.1f} conf={confidence:.1f} min_conf={min_confidence:.1f} "
                f"price={price:.2f} spread={spread:.5f} sess={session} weight={session_weight:.2f} "
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

        MIN_CANDLES_BETWEEN = self.config.get("trading_rules.MIN_CANDLES_BETWEEN")
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

        logger.info(f"⚙️ تنظیمات نهایی بارگذاری شد: Timeframe={TIMEFRAME}, Min_Candles_Between={MIN_CANDLES_BETWEEN}")
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

            # --- استراحت کندلی ---
            if self.bot_state.last_trade_candle_time and not df.empty:
                last_trade_time = self.bot_state.last_trade_candle_time
                candles_passed = len(df[df["time"] > last_trade_time])
                if candles_passed < MIN_CANDLES_BETWEEN:
                    wait_needed = MIN_CANDLES_BETWEEN - candles_passed
                    logger.info(f"⏸️ استراحت کندلی: {candles_passed}/{MIN_CANDLES_BETWEEN}")
                    self._maybe_monitor_trades()
                    return

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
            current_spread = float(result.get("spread", 0.0) or 0.0)

            sess = result.get("session_analysis") or {}
            session_name = str(sess.get("current_session", "UNKNOWN"))
            session_weight = float(sess.get("weight", sess.get("session_weight", 0.0)) or 0.0)
            session_activity = str(sess.get("session_activity", ""))
            is_active_session = bool(sess.get("is_active_session", True))
            untradable = bool(sess.get("untradable", False))
            untradable_reasons = str(sess.get("untradable_reasons", "-"))

            # --- استخراج ADX به شکل مقاوم (برای DEAD_ZONE override در RiskManager) ---
            # چون خروجی analyzer ممکن است کلیدهای متفاوت داشته باشد، چند مسیر را پوشش می‌دهیم.
            adx_value = 0.0
            try:
                if result.get("adx") is not None:
                    adx_value = float(result.get("adx") or 0.0)
                elif result.get("adx_value") is not None:
                    adx_value = float(result.get("adx_value") or 0.0)
                else:
                    ind = result.get("indicators") or {}
                    if isinstance(ind, dict):
                        if ind.get("adx") is not None:
                            adx_value = float(ind.get("adx") or 0.0)
                        elif ind.get("adx_value") is not None:
                            adx_value = float(ind.get("adx_value") or 0.0)
                        elif ind.get("adx_analysis") and isinstance(ind.get("adx_analysis"), dict):
                            adx_value = float(ind["adx_analysis"].get("adx", 0.0) or 0.0)
            except Exception:
                adx_value = 0.0

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
                price=current_price, spread=current_spread, session=session_name,
                session_weight=session_weight, session_activity=session_activity,
                is_active_session=is_active_session, untradable=untradable,
                reject_reason=reject_reason, reject_details=reject_details
            )

            # نمایش نتایج در کنسول (همان تابع قبلی شما)
            result["signal"] = final_signal  # آپدیت سیگنال نهایی در دیکشنری
            self.display_results(result)

            self.bot_state.analysis_count += 1
            self.bot_state.last_analysis = datetime.now()

            if result.get("error"):
                logger.warning("⚠️ سیگنال حاوی خطاست")
                return

            # --- اجرای معامله ---
            if final_signal in ("BUY", "SELL"):
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

                    can_trade, reason = self.risk_manager.can_scalp(account_equity=ACCOUNT_BALANCE)
                    if not can_trade:
                        logger.info(f"⏸️ ریسک منیجر: {reason}")
                        return

                if not ENABLE_DRY_RUN:
                    trade_success = self.execute_scalping_trade(result, df)
                    if trade_success:
                        self.bot_state.last_trade_candle_time = df["time"].iloc[-1]
                        self.bot_state.last_trade_wall_time = datetime.now()
                        self.bot_state.last_trade_time = self.bot_state.last_trade_wall_time
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
                return orders or []
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
        if result.get("entry_price"):
            ep = float(result.get("entry_price") or 0)
            sl = float(result.get("stop_loss") or 0)
            tp = float(result.get("take_profit") or 0)

            print(f"\n💰 پارامترهای ورود:")
            print(f"   قیمت ورود: ${ep:.2f}")
            print(f"   استاپ لاس: ${sl:.2f}")
            print(f"   تیک پروفیت: ${tp:.2f}")

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
    def _extract_trade_levels(self, signal_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Extract entry/sl/tp from either root keys or nested analyzer context."""
        entry = signal_data.get("entry_price")
        sl = signal_data.get("stop_loss")
        tp = signal_data.get("take_profit")

        # برخی خروجی‌ها ممکن است از RiskManager نهایی شده باشند
        if entry is None and signal_data.get("final_entry") is not None:
            entry = signal_data.get("final_entry")
        if sl is None and signal_data.get("final_stop_loss") is not None:
            sl = signal_data.get("final_stop_loss")
        if tp is None and signal_data.get("final_take_profit") is not None:
            tp = signal_data.get("final_take_profit")

        try:
            entry_f = float(entry) if entry is not None else None
        except Exception:
            entry_f = None
        try:
            sl_f = float(sl) if sl is not None else None
        except Exception:
            sl_f = None
        try:
            tp_f = float(tp) if tp is not None else None
        except Exception:
            tp_f = None

        return entry_f, sl_f, tp_f

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
        # Guardrail #1: اعتبارسنجی هندسه معامله (Analyzer output)
        # ------------------------------------------------------------
        try:
            entry, sl, tp = self._extract_trade_levels(signal_data)
            # اگر آنالایزر level ارائه داده باشد، باید هندسه صحیح باشد
            if entry is not None or sl is not None or tp is not None:
                ok, reason = self._validate_trade_geometry(signal_data.get("signal", "NONE"), entry, sl, tp)
                if not ok:
                    logger.error("❌ Invalid trade geometry from Analyzer | %s", reason)
                    print(f"❌ هندسه معامله نامعتبر است: {reason}")
                    return False
        except Exception as g_err:
            logger.warning(f"⚠️ Geometry validation failed unexpectedly: {g_err}", exc_info=True)

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
                signal_data.get("entry_price"),
                signal_data.get("stop_loss"),
                signal_data.get("take_profit"),
                (signal_data.get("context") or {}).get("entry_source"),
                (signal_data.get("context") or {}).get("entry_distance_pips"),
                (signal_data.get("context") or {}).get("entry_distance_atr"),
            )
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
                    "final_entry": finalized.entry_price,
                    "final_stop_loss": finalized.stop_loss,
                    "final_take_profit": finalized.take_profit,
                    "final_volume": finalized.lot_size,
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

                point_size = self.risk_manager._get_point_size(config_payload)
                spread_price = float(current_price_data.get("spread", 0.19) or 0.19)
                spread_pips = points_to_pips(price_to_points(spread_price, point_size))
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
                print(f"✅ سفارش {order_type} ارسال شد - ticket={order_id} | حجم: {lot_size:.3f} لات")

                open_event: ExecutionEvent = {
                    "event_type": "OPEN",
                    "event_time": datetime.now(),
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
                        "tp2_price": tp2_price,
                    },
                }
                self.trade_tracker.add_trade_open(open_event)
                self.bot_state.add_trade(success=True)

                if df is None or df.empty:
                    self.bot_state.last_trade_wall_time = datetime.now()
                    self.bot_state.last_trade_time = self.bot_state.last_trade_wall_time

                if hasattr(self.risk_manager, "add_position"):
                    self.risk_manager.add_position(lot_size)

                generate_execution_report(
                    logger=logger,
                    event=open_event,
                    df=df,
                )

                try:
                    self.notifier.send_signal_notification(params=signal_data, symbol=SYMBOL)
                except Exception as t_err:
                    logger.warning(f"⚠️ خطای غیربحرانی در ارسال تلگرام: {t_err}", exc_info=True)

                self._maybe_monitor_trades(force=True)
                return True

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
            added_count, updated_count, closed_candidates = self.trade_tracker.reconcile_with_open_positions(open_positions)

            if added_count or updated_count:
                logger.debug("🔄 Trade reconciliation: added=%s updated=%s", added_count, updated_count)

            for record in closed_candidates:
                identity = record.get("trade_identity", {})
                position_ticket = identity.get("position_ticket")
                if not position_ticket:
                    continue

                history = self.mt5_client.get_position_history(position_ticket)
                if not history or not history.get("close_time"):
                    self.trade_tracker.mark_trade_unknown(position_ticket, "history_not_found")
                    logger.debug("⏳ Close not confirmed for position %s. Will retry.", position_ticket)
                    continue

                symbol = identity.get("symbol") or SYMBOL
                side = record.get("open_event", {}).get("side")
                entry_price = record.get("open_event", {}).get("entry_price")
                exit_price = history.get("exit_price") or record.get("last_update_event", {}).get("metadata", {}).get("current_price")
                profit = history.get("total_profit")
                close_time = history.get("close_time")
                reason = history.get("reason")

                pips_val = compute_pips(symbol, entry_price or 0.0, exit_price or 0.0)

                close_event: ExecutionEvent = {
                    "event_type": "CLOSE",
                    "event_time": close_time or datetime.now(),
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
                    "metadata": {"history": history},
                }

                self.trade_tracker.close_trade_event(close_event)
                generate_execution_report(logger=logger, event=close_event)

                logger.info(
                    "[TRADE][CLOSE] position=%s profit=%.2f pips=%.1f reason=%s",
                    position_ticket,
                    float(profit or 0.0),
                    float(pips_val or 0.0),
                    reason,
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

        except Exception as e:
            logger.error(f"⚠️ خطا در فرآیند مانیتورینگ معاملات: {e}", exc_info=True)

    # ----------------------------
    # Cleanup/Summary
    # ----------------------------
    def cleanup(self):
        """تمیزکاری منابع و قطع اتصال"""
        logger.info("🧹 در حال ذخیره وضعیت و تمیزکاری...")
        print("\n🧹 در حال ذخیره وضعیت...")

        try:
            # یک بار آخر مانیتورینگ تا closeها ثبت شوند
            self._maybe_monitor_trades(force=True)
        except Exception:
            pass

        try:
            if self.mt5_client:
                logger.info("قطع اتصال MT5...")
                self.mt5_client.disconnect()
                logger.info("✅ اتصال MT5 قطع شد")
                print("✅ اتصال MT5 قطع شد")
        except Exception as e:
            logger.error(f"⚠️ خطا در قطع اتصال MT5: {e}", exc_info=True)
            print(f"⚠️ خطا در قطع اتصال MT5: {e}")

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
        logger.info("🧹 شروع فرآیند تمیزکاری نهایی")

        # ابتدا summary (هنوز اتصال برقرار است)
        try:
            self.print_summary()
        except Exception as e:
            logger.error(f"⚠️ خطا در چاپ summary: {e}", exc_info=True)

        # سپس cleanup
        self.cleanup()

        logger.info("🏁 پایان اجرای ربات اسکلپینگ")
