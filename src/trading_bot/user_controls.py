"""User input handling helpers for the NDS bot."""

import os
import sys
import threading
import time
from datetime import datetime, timedelta

from src.ui.cli import print_help, update_config_interactive


class UserControls:
    """مدیریت ورودی و کنترل‌های کاربر"""

    def __init__(self, bot, logger):
        self.bot = bot
        self.logger = logger

    def get_user_action(self, timeout: float = 0.1) -> str:
        """دریافت عمل کاربر (پشتیبانی کامل از ویندوز و تردینگ)"""
        try:
            if os.name == 'nt':
                import msvcrt

                key_pressed = [None]

                def check_key():
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key:
                            key_pressed[0] = key.decode('utf-8', errors='ignore').lower()

                key_thread = threading.Thread(target=check_key)
                key_thread.daemon = True
                key_thread.start()
                key_thread.join(timeout=timeout)

                key = key_pressed[0]
            else:
                import select
                if select.select([sys.stdin], [], [], timeout)[0]:
                    key = sys.stdin.read(1).lower()
                else:
                    key = None

            if key:
                self.logger.debug(f"User action detected: {key}")
                key_map = {
                    'q': 'quit', 'p': 'pause', 's': 'status', 'c': 'config',
                    't': 'toggle_trading', 'r': 'toggle_risk', 'd': 'toggle_dry_run',
                    'k': 'skip', 'h': 'help'
                }
                return key_map.get(key, '')
        except Exception as e:
            self.logger.debug(f"خطا در دریافت ورودی کاربر: {e}")

        return ''

    def handle_user_action(self, action: str):
        """مدیریت دستورات کاربر"""
        self.logger.info(f"User action: {action}")

        action_handlers = {
            'quit': lambda: setattr(self.bot.bot_state, 'running', False) or self.logger.info("👋 درخواست خروج") or print("\n👋 درخواست خروج"),
            'pause': lambda: (
                setattr(self.bot.bot_state, 'paused', not self.bot.bot_state.paused),
                self.logger.info(f"⏸️  ربات {'متوقف شد' if self.bot.bot_state.paused else 'ادامه یافت'}"),
                print(f"\n⏸️  ربات {'متوقف شد' if self.bot.bot_state.paused else 'ادامه یافت'}")
            ),
            'status': lambda: (self.logger.info("📊 نمایش وضعیت ربات"), self.print_status()),
            'config': lambda: (self.logger.info("⚙️  به‌روزرسانی تنظیمات"), update_config_interactive()),
            'toggle_trading': lambda: (
                self.bot.config.update_setting('trading_settings.ENABLE_AUTO_TRADING', not self.bot.config.get('trading_settings.ENABLE_AUTO_TRADING')),
                self.logger.info(f"🤖 معاملات خودکار {'فعال' if not self.bot.config.get('trading_settings.ENABLE_AUTO_TRADING') else 'غیرفعال'} شد"),
                print(f"\n🤖 معاملات خودکار {'فعال' if not self.bot.config.get('trading_settings.ENABLE_AUTO_TRADING') else 'غیرفعال'} شد")
            ),
            'toggle_risk': lambda: (
                self.bot.config.update_setting('trading_settings.ENABLE_RISK_MANAGER', not self.bot.config.get('trading_settings.ENABLE_RISK_MANAGER')),
                self.logger.info(f"🛡️  مدیر ریسک {'فعال' if not self.bot.config.get('trading_settings.ENABLE_RISK_MANAGER') else 'غیرفعال'} شد"),
                print(f"\n🛡️  مدیر ریسک {'فعال' if not self.bot.config.get('trading_settings.ENABLE_RISK_MANAGER') else 'غیرفعال'} شد")
            ),
            'toggle_dry_run': lambda: (
                self.bot.config.update_setting('trading_settings.ENABLE_DRY_RUN', not self.bot.config.get('trading_settings.ENABLE_DRY_RUN')),
                self.logger.info(f"🔧 حالت آزمایشی {'فعال' if not self.bot.config.get('trading_settings.ENABLE_DRY_RUN') else 'غیرفعال'} شد"),
                print(f"\n🔧 حالت آزمایشی {'فعال' if not self.bot.config.get('trading_settings.ENABLE_DRY_RUN') else 'غیرفعال'} شد")
            ),
            'skip': lambda: (self.logger.info("⏩ رد کردن زمان انتظار"), print("\n⏩ رد کردن زمان انتظار")),
            'help': lambda: (self.logger.info("📖 نمایش راهنما"), print_help())
        }

        handler = action_handlers.get(action)
        if handler:
            handler()

    def wait_with_controls(self, seconds):
        """انتظار هوشمند همراه با مانیتورینگ مداوم پوزیشن‌ها"""
        next_time = datetime.now() + timedelta(seconds=seconds)
        next_time_str = next_time.strftime('%H:%M:%S')

        msg = f"⏳ انتظار برای سیکل بعدی... تحلیل شماره بعدی در ساعت {next_time_str} انجام خواهد شد."
        self.logger.info(msg)
        print(f"\n{msg}")
        print("   (P=توقف، S=وضعیت، C=تنظیمات، Q=خروج)")

        start_wait = time.time()
        last_monitor_time = time.time()

        while time.time() - start_wait < seconds:
            if not self.bot.bot_state.running or self.bot.bot_state.paused:
                break

            if time.time() - last_monitor_time > 3.0:
                self.bot._monitor_open_trades()
                last_monitor_time = time.time()

            action = self.get_user_action()
            if action:
                self.handle_user_action(action)
                if action == 'status':
                    print(f"\n{msg}")

            time.sleep(0.5)

    def print_status(self):
        """نمایش وضعیت لحظه‌ای ربات"""
        stats = self.bot.bot_state.get_statistics()

        trading_cfg = self.bot.config.get('trading_settings')
        tech_cfg = self.bot.config.get('technical_settings')

        symbol = trading_cfg['SYMBOL']
        timeframe = trading_cfg['TIMEFRAME']
        min_conf = tech_cfg['SCALPING_MIN_CONFIDENCE']

        self.logger.info(f"📊 وضعیت ربات: {symbol} | {timeframe} | Conf: {min_conf}%")

        print(f"\n" + "=" * 45)
        print(f"📊 وضعیت ربات: {symbol} ({timeframe})")
        print(f"   حداقل اعتماد تنظیمی: {min_conf}%")

        hours = int(stats['runtime_seconds'] // 3600)
        minutes = int((stats['runtime_seconds'] % 3600) // 60)
        print(f"   زمان اجرا: {hours}:{minutes:02d}")

        print(f"   تحلیل‌ها: {stats['analysis_count']} | معاملات: {stats['trade_count']}")

        if stats['trade_count'] > 0:
            print(f"   نرخ موفقیت: {stats['success_rate']:.1f}%")

        print(f"   سود کل: ${stats['total_profit']:.2f} | روزانه: ${stats['daily_pnl']:.2f}")

        open_positions = self.bot.get_open_positions_count()
        print(f"   پوزیشن‌های باز: {open_positions}")

        if open_positions > 0:
            positions_info = self.bot.get_open_positions_info()
            for pos in positions_info[:3]:
                ticket = pos.get('ticket')
                p_type = pos.get('type')
                volume = pos.get('volume', 0.0) or 0.0
                profit = pos.get('profit', 0.0) or 0.0

                profit_color = "🟢" if profit >= 0 else "🔴"

                print(f"   └─ #{ticket}: {p_type} {volume}L -> {profit_color}${profit:.2f}")

        if self.bot.risk_manager and hasattr(self.bot.risk_manager, 'get_scalping_summary'):
            try:
                scalping_summary = self.bot.risk_manager.get_scalping_summary()
                print(
                    f"   سشن: {scalping_summary.get('current_session', 'N/A')} "
                    f"({'✅' if scalping_summary.get('session_friendly') else '❌'})"
                )
            except Exception:
                pass

        if hasattr(self.bot, 'trade_tracker'):
            try:
                daily_stats = self.bot.trade_tracker.get_daily_stats()

                if daily_stats.get('total_trades', 0) > 0 or daily_stats.get('active_trades', 0) > 0:
                    print(f"   📊 آمار سیستم ردیابی:")
                    print(f"      • معاملات امروز: {daily_stats.get('total_trades', 0)}")

                    if daily_stats.get('total_trades', 0) > 0:
                        win_rate = daily_stats.get('win_rate', 0.0) or 0.0
                        total_p = daily_stats.get('total_profit', 0.0) or 0.0
                        max_p = daily_stats.get('max_daily_profit', 0.0) or 0.0

                        print(f"      • وین ریت: {win_rate:.1f}%")
                        print(f"      • سود امروز: ${total_p:.2f}")
                        print(f"      • حداکثر سود: ${max_p:.2f}")

                    if daily_stats.get('active_trades', 0) > 0:
                        print(f"      • معاملات فعال: {daily_stats.get('active_trades', 0)}")
                        active_trades = list(self.bot.trade_tracker.active_trades.items())[:2]
                        for ticket, trade in active_trades:
                            raw_profit = trade.get('current_profit', 0.0)
                            if raw_profit is None:
                                raw_profit = 0.0

                            profit_color = "🟢" if raw_profit >= 0 else "🔴"
                            signal_type = trade.get('signal_type') or trade.get('type', 'UNKNOWN')
                            signal_emoji = "📈" if "BUY" in str(signal_type).upper() else "📉"

                            print(f"         {signal_emoji} #{ticket}: {profit_color}${raw_profit:.2f}")

                    if daily_stats.get('closed_trades', 0) > 0:
                        print(f"      • معاملات بسته: {daily_stats.get('closed_trades', 0)}")
            except Exception as e:
                self.logger.warning(f"⚠️ جزئیات خطا در نمایش آمار: {e}")

        print("=" * 45)
