# config/settings.py
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    مدیریت متمرکز تنظیمات:
    - bot_config.json = تنظیمات عمومی (منبع واحد حقیقت برای تنظیمات ربات)
    - mt5_credentials.json = اطلاعات حساس MT5 (فایل جدا)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._config = {}
            cls._instance._base_dir = Path(__file__).resolve().parent  # پوشه config
            cls._instance._load_config()
        return cls._instance

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _config_path(self) -> Path:
        return self._base_dir / "bot_config.json"

    def _mt5_credentials_path(self) -> Path:
        return self._base_dir / "mt5_credentials.json"

    def _load_json_file(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_config(self) -> None:
        """بارگذاری bot_config.json"""
        config_path = self._config_path()
        if not config_path.exists():
            raise FileNotFoundError(f"bot_config.json not found at: {config_path}")

        self._config = self._load_json_file(config_path)
        self._validate_required_keys()
        logger.info("✅ Config loaded from config/bot_config.json")

    def _validate_required_keys(self) -> None:
        required_paths = [
            "bot_name",
            "version",
            "ACCOUNT_BALANCE",
            "trading_rules",
            "risk_settings",
            "trading_settings",
            "technical_settings",
            "sessions_config",
            "risk_manager_config",
            "LOG_LEVEL",
            "DEBUG_LOG_FILE",
        ]

        for key in required_paths:
            self._require_path(key)

    def _require_path(self, key: str) -> Any:
        """اگر وجود نداشته باشد KeyError می‌دهد (برای validation داخلی)."""
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"Missing required config key: {key}")
        return value

    def _get_path_soft(self, key: str, default: Any = None) -> Any:
        """نسخه نرم: اگر نبود default برگرداند."""
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return default if value is None else value

    # -----------------------------
    # Public API
    # -----------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """
        دریافت مقدار با Dot-Notation.
        - اگر نبود: default
        - اگر موجود ولی None: default
        """
        return self._get_path_soft(key, default)

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.get(key, default)

    def get_trading_rules(self) -> Dict[str, Any]:
        return self.get("trading_rules", {}) or {}

    def get_risk_settings(self) -> Dict[str, Any]:
        return self.get("risk_settings", {}) or {}

    def get_trading_settings(self) -> Dict[str, Any]:
        return self.get("trading_settings", {}) or {}

    def get_technical_settings(self) -> Dict[str, Any]:
        return self.get("technical_settings", {}) or {}

    def get_sessions_config(self) -> Dict[str, Any]:
        return self.get("sessions_config", {}) or {}

    def get_risk_manager_config(self) -> Dict[str, Any]:
        return self.get("risk_manager_config", {}) or {}

    def get_full_config_for_analyzer(self) -> Dict[str, Any]:
        technical = self.get_technical_settings()
        sessions = self.get_sessions_config()
        return {
            "ANALYZER_SETTINGS": technical.copy(),
            "sessions_config": sessions.copy(),
            "TRADING_SESSIONS": sessions.get("TRADING_SESSIONS", {}) if isinstance(sessions, dict) else {},
        }

    def get_full_config(self) -> Dict[str, Any]:
        return dict(self._config)  # shallow copy

    # -----------------------------
    # MT5 Credentials
    # -----------------------------
    def get_mt5_credentials(self) -> Dict[str, Any]:
        """
        اولویت خواندن:
        1) bot_config.json -> mt5_credentials (اگر در آینده اضافه کردی)
        2) config/mt5_credentials.json (فایل جداگانه)
        """
        # 1) Optional: from bot_config.json
        creds_from_main = self.get("mt5_credentials", None)
        if isinstance(creds_from_main, dict) and creds_from_main:
            normalized = self._normalize_mt5_credentials(creds_from_main)
            if self._mt5_creds_complete(normalized):
                return normalized

        # 2) From mt5_credentials.json in config folder (ABSOLUTE PATH)
        creds_path = self._mt5_credentials_path()
        if creds_path.exists():
            try:
                data = self._load_json_file(creds_path)
                normalized = self._normalize_mt5_credentials(data)
                return normalized
            except Exception as e:
                logger.error(f"❌ Failed to read MT5 credentials: {e}")
                return {}

        return {}

    def save_mt5_credentials(self, credentials: Dict[str, Any]) -> None:
        """ذخیره امن credentials در config/mt5_credentials.json"""
        creds_path = self._mt5_credentials_path()
        try:
            normalized = self._normalize_mt5_credentials(credentials)
            creds_path.parent.mkdir(parents=True, exist_ok=True)
            with creds_path.open("w", encoding="utf-8") as f:
                json.dump(normalized, f, indent=2, ensure_ascii=False)
            logger.info("✅ MT5 credentials saved to config/mt5_credentials.json")
        except Exception as e:
            logger.error(f"❌ Error saving MT5 credentials: {e}")

    def _normalize_mt5_credentials(self, creds: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize keys/types to what project expects."""
        if not isinstance(creds, dict):
            return {}

        out = dict(creds)

        # Ensure required keys exist with correct names
        # (اگر کسی با LOGIN / SERVER آمده باشد هم ساپورت می‌کنیم)
        if "login" not in out and "LOGIN" in out:
            out["login"] = out.get("LOGIN")
        if "password" not in out and "PASSWORD" in out:
            out["password"] = out.get("PASSWORD")
        if "server" not in out and "SERVER" in out:
            out["server"] = out.get("SERVER")

        # Types
        try:
            if "login" in out and out["login"] is not None:
                out["login"] = int(out["login"])
        except Exception:
            pass

        # Defaults (optional)
        out.setdefault("timeout", 30)
        out.setdefault("retry_count", 3)
        out.setdefault("real_time_enabled", True)
        out.setdefault("tick_update_interval", 1.0)

        return out

    def _mt5_creds_complete(self, creds: Dict[str, Any]) -> bool:
        return bool(
            isinstance(creds, dict)
            and creds.get("login")
            and creds.get("password")
            and creds.get("server")
        )

    # -----------------------------
    # Update setting (in-memory)
    # -----------------------------
    def update_setting(self, key: str, value: Any) -> None:
        keys = key.split(".")
        ref = self._config
        for k in keys[:-1]:
            if k not in ref or not isinstance(ref[k], dict):
                ref[k] = {}
            ref = ref[k]
        ref[keys[-1]] = value
        logger.info(f"Config updated (in-memory): {key} = {value}")


config = ConfigManager()
