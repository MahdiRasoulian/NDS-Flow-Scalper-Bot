import os
import requests
import logging
import threading
import queue
from datetime import datetime

# تنظیمات لاگر
logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self):

            self.token = os.getenv("TELEGRAM_BOT_TOKEN", "8528114862:AAGfpVR-ytNUf0IwKHYRmvITV5EAuHFV-xQ")
            self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "-1003385933201")
            
            self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            
            # سیستم Queue برای جلوگیری از لگ در اسکلپینگ
            self.msg_queue = queue.Queue()
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()

    def _process_queue(self):
        """پردازشگر پس‌زمینه برای ارسال پیام‌ها"""
        while True:
            message = self.msg_queue.get()
            if message is None: break
            self._send_request(message)
            self.msg_queue.task_done()

    def _send_request(self, message):
        """ارسال نهایی به API تلگرام با مدیریت خطا"""
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(self.api_url, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Telegram API Error: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    def send_signal_notification(self, params, symbol: str):
        """
        ارسال سیگنال حرفه‌ای به زبان فارسی
        :param params: شیء از کلاس EntryParameters یا دیکشنری سیگنال
        """
        # استخراج داده‌ها (پشتیبانی از هر دو حالت شیء یا دیکشنری)
        if isinstance(params, dict):
            sig_type = params.get('signal', 'NEUTRAL')
            ep = params.get('entry_price', 0)
            sl = params.get('stop_loss', 0)
            tp = params.get('take_profit', 0)
            conf = params.get('confidence', 0)
        else:
            sig_type = params.signal
            ep = params.entry_price
            sl = params.stop_loss
            tp = params.take_profit
            conf = params.confidence

        if sig_type == 'NEUTRAL': return

        # محاسبه Risk to Reward
        risk = abs(ep - sl)
        reward = abs(tp - ep)
        rr = round(reward / risk, 2) if risk != 0 else 0
        
        # تعیین ایموجی جهت معامله
        side_emoji = "🟢 #BUY" if sig_type == "BUY" else "🔴 #SELL"
        
        # ساخت متن پیام شکیل
        message = (
            f"🚀 <b>سیگنال جدید اسکلپینگ {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🔔 <b>نوع پوزیشن:</b> {side_emoji}\n"
            f"🎯 <b>قیمت ورود:</b> <code>{ep:,.2f}</code>\n"
            f"🛑 <b>حد ضرر (SL):</b> <code>{sl:,.2f}</code>\n"
            f"✅ <b>حد سود (TP):</b> <code>{tp:,.2f}</code>\n"
            f"📊 <b>نسبت R/R:</b> <code>1:{rr}</code>\n"
            f"🛡 <b>سطح اطمینان:</b> <code>{conf}%</code>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"⏰ <b>زمان:</b> {datetime.now().strftime('%H:%M:%S')}\n"
            f"🤖 <i>NDS Gold Analyzer Bot</i>"
        )
        
        # افزودن به صف ارسال (Non-blocking)
        self.msg_queue.put(message)


    def send_trade_close_notification(self, symbol: str, signal_type: str, profit_usd: float, pips: float, reason: str):
        """
        ارسال گزارش بسته‌شدن معامله
        :param reason: دلیل بسته شدن (TP, SL, Manual, Time-out)
        """
        result_emoji = "✅ #PROFIT" if profit_usd > 0 else "❌ #LOSS"
        trend_emoji = "💰" if profit_usd > 0 else "📉"
        
        message = (
            f"{trend_emoji} <b>معامله {symbol} بسته شد</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🏁 <b>نتیجه:</b> {result_emoji}\n"
            f"👤 <b>نوع معامله:</b> {signal_type}\n"
            f"💵 <b>سود/ضرر دلار:</b> <code>${profit_usd:,.2f}</code>\n"
            f"📏 <b>مقدار جابجایی:</b> <code>{pips:,.1f} Pips</code>\n"
            f"📝 <b>علت خروج:</b> {reason}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"⏰ <b>زمان بسته شدن:</b> {datetime.now().strftime('%H:%M:%S')}\n"
            f"📊 <i>NDS Scalping Performance Management</i>"
        )
        self.msg_queue.put(message)    
