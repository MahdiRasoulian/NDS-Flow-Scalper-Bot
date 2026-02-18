# src/trading_bot/state.py

from datetime import datetime, timezone
from typing import Any, Optional

from src.trading_bot.time_utils import parse_timestamp, to_utc_time


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _normalize_utc(value: Any) -> Optional[datetime]:
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    return to_utc_time(parsed, time_mode="UTC")


class BotState:
    """مدیریت وضعیت ربات"""

    def __init__(self):
        self.running = True
        self.paused = False
        self.analysis_count = 0
        self.trade_count = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit = 0.0
        self.start_time = _utc_now()
        self.last_analysis = None
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.active_positions = []
        self.last_trade_time = None  # سازگاری با نسخه‌های قدیمی
        self.last_trade_wall_time = None
        self.last_trade_candle_time = None
        self.last_trade_direction = None
        self.active_signal_direction = None

    def set_last_analysis(self, value: Any) -> None:
        self.last_analysis = _normalize_utc(value)

    def set_last_trade_times(self, *, wall_time: Any = None, candle_time: Any = None) -> None:
        if wall_time is not None:
            normalized_wall = _normalize_utc(wall_time)
            self.last_trade_wall_time = normalized_wall
            self.last_trade_time = normalized_wall
        if candle_time is not None:
            self.last_trade_candle_time = _normalize_utc(candle_time)

    def add_trade(self, success: bool, profit: float = 0.0):
        """ثبت معامله"""
        self.trade_count += 1
        self.set_last_trade_times(wall_time=_utc_now())

        if success:
            self.successful_trades += 1
        else:
            self.failed_trades += 1
            self.consecutive_losses += 1

        self.daily_pnl += profit
        self.total_profit += profit

        if success and self.consecutive_losses > 0:
            self.consecutive_losses = 0

    def get_statistics(self) -> dict:
        """دریافت آمار ربات"""
        runtime = _utc_now() - self.start_time

        stats = {
            'runtime_seconds': runtime.total_seconds(),
            'analysis_count': self.analysis_count,
            'trade_count': self.trade_count,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'success_rate': (self.successful_trades / self.trade_count * 100) if self.trade_count > 0 else 0,
            'total_profit': self.total_profit,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'active_positions': len(self.active_positions),
            'last_trade_time': self.last_trade_time.strftime('%H:%M:%S') if self.last_trade_time else 'N/A',
            'last_trade_wall_time': self.last_trade_wall_time.strftime('%H:%M:%S') if self.last_trade_wall_time else 'N/A',
            'last_trade_candle_time': self.last_trade_candle_time.strftime('%H:%M:%S') if self.last_trade_candle_time else 'N/A',
            'last_trade_direction': self.last_trade_direction or 'N/A',
            'active_signal_direction': self.active_signal_direction or 'N/A',
        }
        return stats
