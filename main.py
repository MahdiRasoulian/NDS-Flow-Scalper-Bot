"""
NDS Trading Bot Pro - Main Entry Point
نسخه ماژولار - منطبق با ساختار src.trading_bot.*
✅ منبع حقیقت واحد: bot_config.json
✅ بدون وابستگی به config/settings.py (حذف شده)
✅ خروج ایمن با SIGINT
"""

import sys
import os
import json
import signal
import logging
from pathlib import Path
from typing import Any, Dict, Optional


# ------------------------------------------------------------
# 1) Standard Path Setup (Safe / Non-duplicative)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# 2) Imports (Project)
#    نکته: هیچ import از config.settings نباید وجود داشته باشد.
# ------------------------------------------------------------
try:
    from src.utils.logger import setup_windows_encoding, setup_logging
    from src.trading_bot.bot import NDSBot
    from src.trading_bot.mt5_client import MT5Client
    from src.trading_bot.nds.analyzer import analyze_gold_market
except ImportError as e:
    print(f"❌ خطای ساختار پروژه: {e}")
    print("نکته: مطمئن شوید فایل‌ها در مسیرهای صحیح قرار دارند.")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ------------------------------------------------------------
# 3) Config Loading (Single Source: bot_config.json)
# ------------------------------------------------------------
def _find_bot_config_path() -> Path:
    """
    مسیر bot_config.json را به صورت مقاوم پیدا می‌کند.
    اولویت‌ها:
      1) ./config/bot_config.json
      2) ./bot_config.json
      3) هر جایی داخل پروژه که config/bot_config.json وجود داشت (fallback ساده)
    """
    candidates = [
        PROJECT_ROOT / "config" / "bot_config.json",
        PROJECT_ROOT / "bot_config.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    # fallback: جست‌وجوی محدود
    for p in PROJECT_ROOT.rglob("bot_config.json"):
        return p

    raise FileNotFoundError("bot_config.json not found in project.")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("bot_config.json must be a JSON object (dict).")
    return data


class ConfigAdapter:
    """
    آداپتر سبک برای کانفیگ پروژه بر پایه bot_config.json
    - get با کلیدهای dot.notation
    - متدهای کمکی رایج که bot/mt5_client معمولا نیاز دارند
    """

    def __init__(self, data: Dict[str, Any], source_path: Optional[Path] = None):
        self._config: Dict[str, Any] = data or {}
        self._source_path = source_path

    def get(self, key: str, default: Any = None) -> Any:
        if not key:
            return default
        # پشتیبانی از کلید تو در تو: "risk_settings.RISK_AMOUNT_USD"
        parts = key.split(".")
        cur: Any = self._config
        for part in parts:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def get_full_config(self) -> Dict[str, Any]:
        return dict(self._config)

    # --- APIهای موردنیاز در bot.py (بر اساس کد شما) ---
    def update_setting(self, key: str, value: Any) -> None:
        # کلید ساده (نه dot) را در ریشه می‌نویسد، چون کد bot.py شما همین رفتار را انتظار دارد
        self._config[key] = value
        self._persist_if_possible()

    def get_mt5_credentials(self) -> Optional[Dict[str, Any]]:
        creds = self._config.get("mt5_credentials") or self._config.get("MT5_CREDENTIALS")
        return creds if isinstance(creds, dict) else None

    def save_mt5_credentials(self, creds: Dict[str, Any]) -> None:
        if not isinstance(creds, dict):
            return
        # طبق عرف، در bot_config.json بهتر است در mt5_credentials ذخیره شود
        self._config["mt5_credentials"] = creds
        self._persist_if_possible()

    def get_full_config_for_analyzer(self) -> Dict[str, Any]:
        # اگر در پروژه شما ساختار خاصی دارید، اینجا همان را برمی‌گردانیم.
        # فعلاً کل کانفیگ + تکنیکال‌ها به عنوان ANALYZER_SETTINGS
        cfg = self.get_full_config()
        tech = cfg.get("technical_settings", {}) or {}
        if "ANALYZER_SETTINGS" not in cfg:
            cfg["ANALYZER_SETTINGS"] = tech
        return cfg

    def get_risk_manager_config(self) -> Dict[str, Any]:
        rm = self._config.get("risk_manager_config", {}) or {}
        return rm if isinstance(rm, dict) else {}

    def get_sessions_config(self) -> Dict[str, Any]:
        sc = self._config.get("sessions_config", {}) or {}
        return sc if isinstance(sc, dict) else {}

    def get_technical_settings(self) -> Dict[str, Any]:
        ts = self._config.get("technical_settings", {}) or {}
        return ts if isinstance(ts, dict) else {}

    def _persist_if_possible(self) -> None:
        # اگر مایل باشید، این بخش bot_config.json را هم آپدیت می‌کند.
        # اگر دوست ندارید config در زمان اجرا نوشته شود، این را کامنت کنید.
        try:
            if self._source_path and self._source_path.exists():
                with open(self._source_path, "w", encoding="utf-8") as f:
                    json.dump(self._config, f, ensure_ascii=False, indent=2)
        except Exception:
            # برای جلوگیری از شکست برنامه، persist را silent می‌کنیم
            pass


# ------------------------------------------------------------
# 4) Logging Setup
# ------------------------------------------------------------
setup_windows_encoding()

# تلاش برای تزریق کانفیگ به setup_logging (اگر نسخه‌ی جدید logger.py این را پشتیبانی کند)
logger = logging.getLogger(__name__)


def _setup_logging_safely(cfg: Dict[str, Any]) -> None:
    try:
        # اگر setup_logging(config_dict=...) پشتیبانی شود
        setup_logging(config_dict=cfg)
    except TypeError:
        # نسخه قدیمی‌تر: بدون پارامتر
        setup_logging()


def _print_active_settings(full_config: Dict[str, Any]) -> None:
    trading_settings = full_config.get("trading_settings", {}) or {}
    risk_settings = full_config.get("risk_settings", {}) or {}

    print("\n⚙️  تنظیمات نهایی (Active):")
    print(f"  • نماد: {trading_settings.get('SYMBOL', 'XAUUSD!')}")
    print(f"  • تایم‌فریم: {trading_settings.get('TIMEFRAME', 'M5')}")
    print(f"  • بازه تحلیل: {trading_settings.get('ANALYSIS_INTERVAL_MINUTES', 5)} دقیقه")
    print(f"  • BARS: {trading_settings.get('BARS_TO_FETCH', 'N/A')}")
    print(f"  • AutoTrading: {trading_settings.get('ENABLE_AUTO_TRADING', False)}")
    print(f"  • DryRun: {trading_settings.get('ENABLE_DRY_RUN', False)}")

    print("\n🛡️  تنظیمات ریسک (Config):")
    if "RISK_AMOUNT_USD" in risk_settings:
        print(f"  • ریسک ثابت دلاری: ${risk_settings.get('RISK_AMOUNT_USD', 0.0)}")
    if "RISK_PERCENT" in risk_settings:
        print(f"  • ریسک درصدی: {risk_settings.get('RISK_PERCENT', 0.0)}%")
    if "MAX_PRICE_DEVIATION_PIPS" in risk_settings:
        print(f"  • Max Deviation: {risk_settings.get('MAX_PRICE_DEVIATION_PIPS', 0)} pips")


# ------------------------------------------------------------
# 5) Main
# ------------------------------------------------------------
def main() -> None:
    # پاکسازی کنسول (اختیاری)
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass

    print("🚀 NDS Gold Scalping Bot - در حال اجرا ...")

    try:
        # Load config from bot_config.json
        cfg_path = _find_bot_config_path()
        print(f"⏳ در حال بارگذاری تنظیمات از: {cfg_path}")
        full_config = _load_json(cfg_path)
        config_manager = ConfigAdapter(full_config, source_path=cfg_path)

        # Setup logging using loaded config (if supported)
        _setup_logging_safely(full_config)
        global logger
        logger = logging.getLogger(__name__)

        if not full_config:
            print("⚠️  هشدار: کانفیگ خالی است. bot_config.json را بررسی کنید.")
            logger.warning("Full config is empty.")

        # Minimal MT5 credentials check
        creds = config_manager.get_mt5_credentials()
        if not creds or not all(k in creds for k in ("login", "password", "server")):
            print("⚠️  اطلاعات MT5 کامل نیست. بخش mt5_credentials در bot_config.json را بررسی کنید.")
            logger.warning("MT5 credentials incomplete or missing in bot_config.json.")

        # Print active settings snapshot
        _print_active_settings(full_config)

        print("\n📦 ماژول‌های فعال:")
        print(f"  • MT5 Client: {MT5Client.__name__}")
        print("  • Risk Manager: managed inside NDSBot.initialize()")
        print("  • Analyzer: analyze_gold_market (NDS/SMC Modular)")

        # MT5Client factory: inject config into instance after creation (hardened)
        def mt5_factory():
            client = MT5Client(logger=logging.getLogger("src.trading_bot.mt5_client"))
            # اگر MT5Client شما قابلیت config داخلی دارد، تزریق می‌کنیم
            try:
                client.config = config_manager  # type: ignore[attr-defined]
                # اگر متد load config دوباره لازم است
                if hasattr(client, "_load_connection_config"):
                    client.connection_config = client._load_connection_config()  # type: ignore[attr-defined]
            except Exception:
                pass
            return client

        # Create bot
        bot = NDSBot(
            mt5_client_cls=mt5_factory,      # به جای کلاس مستقیم، فکتوری می‌دهیم
            risk_manager_cls=None,
            analyzer_cls=None,
            analyze_func=analyze_gold_market
        )

        # تزریق کانفیگ واحد به bot (اگر bot.config داشته باشد)
        try:
            bot.config = config_manager  # اگر در bot استفاده می‌شود
            # analyzer config هم از همین منبع
            bot.analyzer_config = config_manager.get_full_config_for_analyzer()
            # اگر price_monitor قبلاً با config قبلی ساخته شده، اینجا جایگزین می‌کنیم
            if hasattr(bot, "price_monitor") and bot.price_monitor:
                try:
                    bot.price_monitor.config = config_manager  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass

        # Signal handling (Safe stop: do NOT sys.exit immediately)
        def signal_handler(sig, frame):
            print("\n🛑 درخواست توقف دریافت شد. ربات به صورت ایمن متوقف می‌شود...")
            logger.info("SIGINT received. Requesting safe shutdown...")
            try:
                if hasattr(bot, "bot_state") and bot.bot_state:
                    bot.bot_state.running = False
            except Exception:
                pass

        signal.signal(signal.SIGINT, signal_handler)

        # Run bot
        print("\n🎯 شروع چرخه معاملاتی اسکلپینگ طلا")
        bot.run()

        print("\n✅ اجرای ربات پایان یافت.")
        logger.info("Bot run finished normally.")

    except KeyboardInterrupt:
        print("\n🛑 توقف توسط کاربر (KeyboardInterrupt)")
        logger.info("KeyboardInterrupt in main().")

    except Exception as e:
        print(f"\n💥 خطای بحرانی در لایه Main: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.critical("Critical failure in main", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
