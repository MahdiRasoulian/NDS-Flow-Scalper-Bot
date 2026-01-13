# src/trading_bot/state.py

from datetime import datetime

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
        self.start_time = datetime.now()
        self.last_analysis = None
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.active_positions = []
        self.last_trade_time = None  # سازگاری با نسخه‌های قدیمی
        self.last_trade_wall_time = None
        self.last_trade_candle_time = None
        
    def add_trade(self, success: bool, profit: float = 0.0):
        """ثبت معامله"""
        self.trade_count += 1
        self.last_trade_time = datetime.now()
        self.last_trade_wall_time = self.last_trade_time
        
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
        runtime = datetime.now() - self.start_time
        
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
        }
        return stats
