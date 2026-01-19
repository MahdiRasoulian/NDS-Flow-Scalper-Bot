import os
import time
import math
import logging
import threading
import queue
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)


def _to_float(x: Any) -> Optional[float]:
    """Best-effort float conversion; returns None if not possible."""
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        # guard against nan/inf
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    try:
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in ("none", "null", "nan", "inf", "-inf"):
            return None
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _get_first(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


class TelegramNotifier:
    """
    Non-blocking Telegram notifier suitable for scalping bots.
    - Uses a worker thread + queue to avoid blocking the trading loop.
    - Robust to different payload schemas (Analyzer vs RiskManager vs Bot decision summaries).
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_sec: float = 10.0,
        queue_maxsize: int = 500,
        max_retries: int = 2,
        retry_backoff_sec: float = 1.25,
    ):
        # Credentials: prefer env; DO NOT hardcode secrets in code.
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        self.timeout_sec = float(timeout_sec)
        self.max_retries = int(max_retries)
        self.retry_backoff_sec = float(retry_backoff_sec)

        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.warning(
                "TelegramNotifier disabled: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set."
            )
            self.api_url = None
        else:
            self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

        # HTTP session for lower overhead
        self._http = requests.Session()

        # Queue for non-blocking sends
        self.msg_queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=queue_maxsize)

        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    # -----------------------------
    # Worker / transport layer
    # -----------------------------
    def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                message = self.msg_queue.get()
                if message is None:
                    self.msg_queue.task_done()
                    break
                self._send_request(message)
                self.msg_queue.task_done()
            except Exception as e:
                # Never allow worker to die
                logger.error(f"TelegramNotifier worker error: {e}")

    def _send_request(self, message: str):
        if not self.enabled or not self.api_url:
            return

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post(self.api_url, json=payload, timeout=self.timeout_sec)
                if resp.status_code == 200:
                    return
                last_err = f"status={resp.status_code} body={resp.text}"
                logger.error(f"Telegram API error (attempt {attempt+1}): {last_err}")
            except Exception as e:
                last_err = str(e)
                logger.error(f"Telegram send failed (attempt {attempt+1}): {e}")

            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_sec * (attempt + 1))

        if last_err:
            logger.error(f"Telegram message ultimately failed: {last_err}")

    def close(self, drain: bool = False):
        """Optional: stop worker. drain=True waits until queue is empty."""
        try:
            if drain:
                self.msg_queue.join()
        except Exception:
            pass
        self._stop_event.set()
        try:
            self.msg_queue.put_nowait(None)
        except Exception:
            pass

    def _enqueue(self, message: str):
        """Non-blocking enqueue; drop safely if overloaded."""
        if not self.enabled:
            return
        try:
            self.msg_queue.put_nowait(message)
        except queue.Full:
            logger.warning("TelegramNotifier queue is full; dropping message to avoid blocking.")

    # -----------------------------
    # Payload normalization
    # -----------------------------
    def _extract_trade_fields(self, params: Any) -> Dict[str, Any]:
        """
        Normalize different payload schemas into canonical fields:
          signal, entry, sl, tp, confidence, rr, timestamp
        """
        # Object -> dict-ish
        if isinstance(params, dict):
            d = params
        else:
            # best effort attribute access
            d = {}
            for k in (
                "signal",
                "side",
                "order_side",
                "type",
                "entry",
                "entry_price",
                "entry_level",
                "final_entry",
                "sl",
                "sl_price",
                "stop_loss",
                "final_sl",
                "tp",
                "tp_price",
                "take_profit",
                "final_tp",
                "tp1",
                "confidence",
                "score",
                "order_type",
            ):
                if hasattr(params, k):
                    d[k] = getattr(params, k)

        # Signal / side
        sig = (
            _get_first(d, ("signal", "side", "order_side", "type"))
            or "NONE"
        )
        sig = str(sig).upper().strip()
        if sig in ("NEUTRAL",):
            sig = "NONE"

        # Entry/SL/TP (support analyzer + risk manager + bot decision)
        entry = _to_float(_get_first(d, ("final_entry", "entry", "entry_price", "entry_level")))
        sl = _to_float(_get_first(d, ("final_sl", "sl", "sl_price", "stop_loss")))
        tp = _to_float(_get_first(d, ("final_tp", "tp", "tp_price", "take_profit", "tp1")))
        conf = _to_float(_get_first(d, ("confidence",)))
        score = _to_float(_get_first(d, ("score",)))

        # RR
        rr = None
        if entry is not None and sl is not None and tp is not None:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk > 0:
                rr = round(reward / risk, 2)

        return {
            "signal": sig,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "confidence": conf,
            "score": score,
            "rr": rr,
        }

    # -----------------------------
    # Public notifications
    # -----------------------------
    def send_signal_notification(self, params: Any, symbol: str):
        """
        Sends a professional Persian message for a new signal/trade plan.
        Works with Analyzer payload (may have no SL/TP) AND RiskManager finalized payload.
        """
        f = self._extract_trade_fields(params)
        sig = f["signal"]

        if sig not in ("BUY", "SELL"):
            return

        entry = f["entry"]
        sl = f["sl"]
        tp = f["tp"]
        conf = f["confidence"]
        score = f["score"]
        rr = f["rr"]

        side_emoji = "🟢 #BUY" if sig == "BUY" else "🔴 #SELL"

        # Format numbers safely
        def fmt(v: Optional[float]) -> str:
            return "N/A" if v is None else f"{v:,.2f}"

        # If SL/TP absent (e.g., analyzer-only), show Pending rather than crashing/misleading 0.00
        sl_txt = fmt(sl) if sl is not None else "Pending (RiskManager)"
        tp_txt = fmt(tp) if tp is not None else "Pending (RiskManager)"

        rr_txt = "N/A"
        if rr is not None:
            rr_txt = f"1:{rr}"
        elif entry is not None and sl is not None and tp is None:
            rr_txt = "TP pending"
        elif entry is not None and tp is not None and sl is None:
            rr_txt = "SL pending"

        conf_txt = "N/A" if conf is None else f"{conf:.1f}%"
        score_txt = "" if score is None else f"\n📈 <b>امتیاز:</b> <code>{score:.1f}/100</code>"

        message = (
            f"🚀 <b>سیگنال جدید اسکلپینگ {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🔔 <b>نوع پوزیشن:</b> {side_emoji}\n"
            f"🎯 <b>قیمت ورود:</b> <code>{fmt(entry)}</code>\n"
            f"🛑 <b>حد ضرر (SL):</b> <code>{sl_txt}</code>\n"
            f"✅ <b>حد سود (TP):</b> <code>{tp_txt}</code>\n"
            f"📊 <b>نسبت R/R:</b> <code>{rr_txt}</code>\n"
            f"🛡 <b>سطح اطمینان:</b> <code>{conf_txt}</code>"
            f"{score_txt}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"⏰ <b>زمان:</b> {datetime.now().strftime('%H:%M:%S')}\n"
            f"🤖 <i>NDS Flow Scalper</i>"
        )

        self._enqueue(message)

    def send_trade_close_notification(
        self,
        symbol: str,
        signal_type: str,
        profit_usd: float,
        pips: float,
        reason: str,
    ):
        result_emoji = "✅ #PROFIT" if profit_usd > 0 else "❌ #LOSS"
        trend_emoji = "💰" if profit_usd > 0 else "📉"

        # Safe formatting
        profit_txt = f"${profit_usd:,.2f}" if profit_usd is not None else "N/A"
        pips_txt = f"{pips:,.1f} Pips" if pips is not None else "N/A"
        signal_type = (signal_type or "").upper().strip()

        message = (
            f"{trend_emoji} <b>معامله {symbol} بسته شد</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🏁 <b>نتیجه:</b> {result_emoji}\n"
            f"👤 <b>نوع معامله:</b> {signal_type}\n"
            f"💵 <b>سود/ضرر دلار:</b> <code>{profit_txt}</code>\n"
            f"📏 <b>مقدار جابجایی:</b> <code>{pips_txt}</code>\n"
            f"📝 <b>علت خروج:</b> {reason}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"⏰ <b>زمان بسته شدن:</b> {datetime.now().strftime('%H:%M:%S')}\n"
            f"📊 <i>NDS Scalping Performance</i>"
        )

        self._enqueue(message)
