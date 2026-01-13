"""Real-time price monitoring utilities."""

from datetime import datetime
from typing import Any, Dict, Optional


class RealTimePriceMonitor:
    """مانیتورینگ و کش قیمت‌های Real-Time"""

    def __init__(self, config, bot_state, logger):
        self.config = config
        self.bot_state = bot_state
        self.logger = logger
        self.mt5_client = None
        self.real_time_prices: Dict[str, Dict[str, Any]] = {}
        self.last_tick_time: Dict[str, datetime] = {}
        self._last_price_log = datetime.min

    def set_mt5_client(self, mt5_client) -> None:
        self.mt5_client = mt5_client

    def start(self) -> None:
        """🔥 شروع مانیتورینگ Real-Time قیمت‌ها"""
        if not self.mt5_client or not self.mt5_client.connected:
            self.logger.warning("⚠️ Cannot start Real-Time monitor: MT5 not connected")
            return

        try:
            if hasattr(self.mt5_client, 'real_time_monitor') and self.mt5_client.real_time_monitor:
                self.logger.info("✅ Using MT5Client Real-Time monitor (no local thread)")
                return

            self.logger.info("🎯 Real-Time Price Monitor ready (direct tick fetch)")

        except Exception as e:
            self.logger.error(f"❌ Error starting Real-Time monitor: {e}")

    def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """🔥 دریافت قیمت لحظه‌ای از کش یا دریافت مستقیم"""
        try:
            if self.mt5_client and self.mt5_client.connected:
                tick = self.mt5_client.get_current_tick(symbol)
                if tick:
                    self.real_time_prices[symbol] = {
                        'bid': tick.get('bid', 0),
                        'ask': tick.get('ask', 0),
                        'last': tick.get('last', 0),
                        'time': tick.get('time', datetime.now()),
                        'spread': tick.get('spread', 0)
                    }
                    self.last_tick_time[symbol] = datetime.now()
                    return {
                        'bid': tick.get('bid', 0),
                        'ask': tick.get('ask', 0),
                        'last': tick.get('last', 0),
                        'time': tick.get('time', datetime.now()),
                        'spread': tick.get('spread', 0),
                        'source': tick.get('source', 'mt5_client')
                    }

            return {
                'bid': 0,
                'ask': 0,
                'last': 0,
                'time': datetime.now(),
                'spread': 0,
                'source': 'no_data',
                'error': 'No price data available'
            }

        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return {
                'bid': 0,
                'ask': 0,
                'last': 0,
                'time': datetime.now(),
                'spread': 0,
                'source': 'error',
                'error': str(e)
            }
