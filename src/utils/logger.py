# src/utils/logger.py

import sys
import os
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

# نگاشت سطوح لاگ
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_windows_encoding() -> None:
    """تنظیمات encoding برای ویندوز (قبل از هر چیز)"""
    if os.name != "nt":
        return

    # تغییر کدپیج کنسول به UTF-8
    try:
        os.system("chcp 65001 > nul")
    except Exception:
        pass

    # پیکربندی stdout و stderr برای UTF-8
    try:
        # Python 3.7+
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # جایگزین برای پایتون‌های قدیمی‌تر
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            # اگر هیچ‌کدام ممکن نبود، بی‌صدا ادامه بده
            pass


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists() or not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _discover_bot_config() -> Dict[str, Any]:
    """
    تلاش برای پیدا کردن bot_config.json بدون وابستگی به config/settings.py.
    اولویت‌ها:
      1) env: BOT_CONFIG_PATH
      2) ./config/bot_config.json
      3) ./bot_config.json
      4) (optional) جستجو در بالا دستی‌ها تا 3 سطح
    """
    # 1) BOT_CONFIG_PATH
    env_path = os.getenv("BOT_CONFIG_PATH")
    if env_path:
        data = _safe_read_json(Path(env_path).expanduser().resolve())
        if data:
            return data

    # 2) project-root guess: از محل همین فایل به سمت بالا
    # src/utils/logger.py -> src/utils -> src -> project_root
    here = Path(__file__).resolve()
    project_root = here.parents[2] if len(here.parents) >= 3 else Path.cwd()

    candidates = [
        project_root / "config" / "bot_config.json",
        project_root / "bot_config.json",
        Path.cwd() / "config" / "bot_config.json",
        Path.cwd() / "bot_config.json",
    ]

    for c in candidates:
        data = _safe_read_json(c)
        if data:
            return data

    # 3) fallback: جستجوی محدود (تا 3 سطح بالا) برای پروژه‌هایی که از جای دیگری اجرا می‌شوند
    p = Path.cwd().resolve()
    for _ in range(3):
        for c in (p / "config" / "bot_config.json", p / "bot_config.json"):
            data = _safe_read_json(c)
            if data:
                return data
        if p.parent == p:
            break
        p = p.parent

    return {}


def _get_from_dict(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    خواندن کلیدهای تو در تو با dot-notation.
    مثال: 'logging.LOG_LEVEL'
    """
    if not d or not key:
        return default
    cur: Any = d
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


class SafeStreamHandler(logging.StreamHandler):
    """Handler ایمن برای ویندوز با UTF-8 و حذف کاراکترهای مشکل‌دار در صورت نیاز"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            try:
                msg = self.format(record)
                msg_clean = re.sub(r"[^\x00-\x7F]+", "", msg)
                stream = self.stream
                stream.write(msg_clean + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


def setup_logging(
    *,
    config_dict: Optional[Dict[str, Any]] = None,
    force_reconfigure: bool = True,
) -> None:
    """
    تنظیمات یکپارچه logging برای کل پروژه بدون وابستگی به config.settings.

    Args:
        config_dict: اگر main.py کانفیگ را خودش می‌خواند، می‌تواند اینجا تزریق کند.
        force_reconfigure: اگر True باشد همه handlerهای قبلی پاک می‌شوند (برای جلوگیری از duplication).
    """
    # 1) load config
    cfg = config_dict if isinstance(config_dict, dict) else _discover_bot_config()

    # 2) read settings (priority: env -> json -> defaults)
    env_level = os.getenv("LOG_LEVEL") or os.getenv("NDS_LOG_LEVEL")
    log_level_str = (env_level or _get_from_dict(cfg, "LOG_LEVEL") or _get_from_dict(cfg, "logging.LOG_LEVEL") or "INFO").upper()
    log_level = LOG_LEVEL_MAP.get(log_level_str, logging.INFO)

    env_log_file = os.getenv("DEBUG_LOG_FILE") or os.getenv("NDS_DEBUG_LOG_FILE")
    log_file = env_log_file or _get_from_dict(cfg, "DEBUG_LOG_FILE") or _get_from_dict(cfg, "logging.DEBUG_LOG_FILE") or "debug.log"

    # اگر مسیر نسبی است، داخل project root قرار بده تا پراکنده نشود
    log_path = Path(log_file)
    if not log_path.is_absolute():
        here = Path(__file__).resolve()
        project_root = here.parents[2] if len(here.parents) >= 3 else Path.cwd()
        log_path = (project_root / log_path).resolve()

    # ساخت پوشه مقصد
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # اگر نتوانست پوشه بسازد، fallback به cwd
        log_path = (Path.cwd() / Path(log_file).name).resolve()

    # 3) reset handlers
    if force_reconfigure:
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)

    # 4) formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 5) file handler (utf-8)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 6) stream handler (safe)
    stream_handler = SafeStreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # 7) apply
    logging.root.setLevel(log_level)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stream_handler)

    # 8) small startup log (optional)
    logging.getLogger(__name__).info(
        "Logging initialized | level=%s | file=%s",
        log_level_str,
        str(log_path),
    )
