"""
MT5 Client بهینه‌شده برای استراتژی NDS - نسخه Real-Time
اتصال دائمی + دریافت لحظه‌ای قیمت + WebSocket Emulation + پشتیبانی کامل از انواع سفارش
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import time
import threading
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import queue
from collections import deque

logger = logging.getLogger(__name__)

from src.trading_bot.contracts import normalize_position

@dataclass
class ConnectionConfig:
    """تنظیمات اتصال MT5"""
    login: int = 0
    password: str = ""
    server: str = ""
    mt5_path: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    timeout: int = 30
    retry_count: int = 3
    real_time_enabled: bool = True  # 🔥 فعال‌سازی Real-Time
    tick_update_interval: float = 1.0  # 🔥 هر ۱ ثانیه قیمت را آپدیت کن

@dataclass
class RealTimePrice:
    """ساختار داده قیمت لحظه‌ای"""
    symbol: str
    bid: float
    ask: float
    last: float
    time: datetime
    volume: int = 0
    spread: float = 0.0
    
    def is_valid(self) -> bool:
        """بررسی اعتبار قیمت"""
        return (self.bid > 0 and self.ask > 0 and 
                self.bid < self.ask and  # bid باید کمتر از ask باشد
                abs(self.last) > 0)  # last نباید صفر باشد

class RealTimeMonitor:
    """مانیتور Real-Time برای دریافت لحظه‌ای قیمت‌ها"""
    
    def __init__(self, client: 'MT5Client', update_interval: float = 1.0):
        self.client = client
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.price_cache: Dict[str, RealTimePrice] = {}
        self.price_history: Dict[str, deque] = {}  # تاریخچه 100 قیمت آخر
        self.callbacks = []
        self.lock = threading.Lock()
        self._error_count = 0
        self._last_error_time = None
        
    def start(self):
        """شروع مانیتورینگ Real-Time"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("✅ Real-Time Price Monitor started")
        
    def stop(self):
        """توقف مانیتورینگ"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("⏹️ Real-Time Price Monitor stopped")
        
    def _monitor_loop(self):
        """حلقه اصلی دریافت قیمت‌ها"""
        logger.debug("🔄 RealTimeMonitor loop started")
        
        while self.running:
            try:
                symbols = list(self.client.symbol_cache.keys())
                if not symbols:
                    logger.warning("⚠️ No symbols in cache for RealTimeMonitor")
                    time.sleep(self.update_interval)
                    continue
                
                # 🔥 دریافت قیمت‌های لحظه‌ای برای تمام نمادهای فعال
                for symbol in symbols:
                    if not self.running:
                        break
                    
                    self._fetch_and_update_price(symbol)
                
            except Exception as e:
                self._handle_monitor_error(e)
                
            time.sleep(self.update_interval)
    
    def _fetch_and_update_price(self, symbol: str):
        """دریافت و به‌روزرسانی قیمت برای یک نماد"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            
            if not tick:
                logger.debug(f"⚠️ No tick data for {symbol}")
                return
            
            # 🔥 اعتبارسنجی قیمت‌های tick
            if tick.bid <= 0 or tick.ask <= 0:
                logger.warning(f"❌ Invalid tick prices for {symbol}: bid={tick.bid}, ask={tick.ask}")
                return
            
            # 🔥 تعیین قیمت last معتبر
            if hasattr(tick, 'last') and tick.last and tick.last > 0:
                last_price = tick.last
            else:
                last_price = tick.bid  # fallback to bid
                logger.debug(f"🔄 Using bid as last for {symbol} (tick.last={tick.last})")
            
            # 🔥 اعتبارسنجی spread
            spread = tick.ask - tick.bid
            if spread < 0:
                logger.error(f"❌ Negative spread for {symbol}: bid={tick.bid}, ask={tick.ask}")
                return
            
            price = RealTimePrice(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                last=last_price,
                time=datetime.now(),
                volume=tick.volume,
                spread=spread
            )
            
            # 🔥 اعتبارسنجی نهایی
            if not price.is_valid():
                logger.error(f"❌ Invalid RealTimePrice for {symbol}: {price}")
                return
            
            with self.lock:
                # فقط اگر قیمت جدید است یا تغییر کرده، ذخیره کن
                old_price = self.price_cache.get(symbol)
                if not old_price or self._has_price_changed(old_price, price):
                    self.price_cache[symbol] = price
                    
                    # ذخیره تاریخچه
                    if symbol not in self.price_history:
                        self.price_history[symbol] = deque(maxlen=100)
                    self.price_history[symbol].append(price)
                    
                    # لاگ تغییر قیمت
                    if old_price:
                        logger.debug(f"📈 Price updated for {symbol}: {old_price.bid:.2f}→{price.bid:.2f}")
            
            # اجرای callback‌ها
            self._execute_callbacks(price)
            
        except Exception as e:
            logger.error(f"❌ Error fetching price for {symbol}: {e}")
    
    def _has_price_changed(self, old_price: RealTimePrice, new_price: RealTimePrice) -> bool:
        """بررسی تغییر قیمت (برای جلوگیری از به‌روزرسانی‌های غیرضروری)"""
        # برای طلا، تغییر کمتر از 0.1 را نادیده بگیر
        return (abs(old_price.bid - new_price.bid) > 0.1 or
                abs(old_price.ask - new_price.ask) > 0.1)
    
    def _execute_callbacks(self, price: RealTimePrice):
        """اجرای callback‌ها"""
        if not self.callbacks:
            return
        
        for callback in self.callbacks[:]:  # کپی لیست برای ایمنی
            try:
                callback(price)
            except Exception as e:
                logger.error(f"❌ Callback error: {e}")
                # اگر callback مرتب خطا می‌دهد، حذفش کن
                try:
                    self.callbacks.remove(callback)
                except ValueError:
                    pass
    
    def _handle_monitor_error(self, error: Exception):
        """مدیریت خطاهای مانیتور"""
        self._error_count += 1
        current_time = datetime.now()
        
        # لاگ خطا با جزئیات
        error_msg = f"❌ Real-Time monitor error #{self._error_count}: {error}"
        
        # اگر خطاها زیاد شدند، هشدار بده
        if self._error_count > 10:
            if self._last_error_time and (current_time - self._last_error_time).total_seconds() < 60:
                logger.critical(f"🚨 CRITICAL: Multiple Real-Time monitor errors in short time!")
        
        self._last_error_time = current_time
        logger.error(error_msg)
        
        # اگر خطاها خیلی زیاد شدند، کمی صبر کن
        if self._error_count > 50:
            logger.warning("⚠️ Too many errors, sleeping for 5 seconds...")
            time.sleep(5)
            self._error_count = 0
    
    def get_current_price(self, symbol: str) -> Optional[RealTimePrice]:
        """دریافت قیمت لحظه‌ای"""
        with self.lock:
            price = self.price_cache.get(symbol)
            if price and price.is_valid():
                return price
            return None
    
    def get_current_price_dict(self, symbol: str) -> Optional[Dict[str, Any]]:
        """دریافت قیمت لحظه‌ای به صورت دیکشنری"""
        price = self.get_current_price(symbol)
        if price:
            return {
                'bid': price.bid,
                'ask': price.ask,
                'last': price.last,
                'time': price.time,
                'volume': price.volume,
                'spread': price.spread,
                'source': 'real_time_monitor'
            }
        return None
    
    def get_price_history(self, symbol: str, count: int = 10) -> List[RealTimePrice]:
        """دریافت تاریخچه قیمت‌ها"""
        with self.lock:
            history = self.price_history.get(symbol, deque())
            # فقط قیمت‌های معتبر را برگردان
            valid_prices = [p for p in list(history)[-count:] if p.is_valid()]
            return valid_prices if valid_prices else []
    
    def get_price_stats(self, symbol: str) -> Dict[str, Any]:
        """دریافت آمار قیمت"""
        with self.lock:
            history = self.price_history.get(symbol, deque())
            if not history:
                return {}
            
            prices = list(history)[-20:]  # 20 قیمت آخر
            if not prices:
                return {}
            
            bid_prices = [p.bid for p in prices]
            ask_prices = [p.ask for p in prices]
            
            return {
                'current_bid': bid_prices[-1],
                'current_ask': ask_prices[-1],
                'avg_bid': sum(bid_prices) / len(bid_prices),
                'avg_ask': sum(ask_prices) / len(ask_prices),
                'min_bid': min(bid_prices),
                'max_bid': max(bid_prices),
                'spread_avg': sum(p.spread for p in prices) / len(prices),
                'update_count': len(prices)
            }
    
    def register_callback(self, callback):
        """ثبت تابع برای دریافت به‌روزرسانی‌های قیمت"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            logger.debug(f"✅ Registered callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def unregister_callback(self, callback):
        """حذف تابع callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"✅ Unregistered callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def get_status(self) -> Dict[str, Any]:
        """دریافت وضعیت مانیتور"""
        with self.lock:
            return {
                'running': self.running,
                'symbol_count': len(self.price_cache),
                'error_count': self._error_count,
                'callback_count': len(self.callbacks),
                'last_error_time': self._last_error_time,
                'update_interval': self.update_interval
            }

class MT5Client:
    """
    کلاس مدیریت اتصال به MetaTrader 5 - نسخه Real-Time
    پشتیبانی کامل از انواع سفارش: Market, Limit, Stop, Stop Limit
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        مقداردهی اولیه کلاینت
        """
        self.connected = False
        self._logger = logger or logging.getLogger(__name__)
        self.data_cache = {}
        self.symbol_cache = {}
        self.session_start = None
        self._last_equity_log = None
        
        # متغیرهای مستقیم برای دسترسی راحت‌تر و مقداردهی از بیرون
        self.login = None
        self.password = None
        self.server = None
        self.mt5_path = None
        
        # 🔥 مانیتور Real-Time
        self.real_time_monitor = None
        
        # اتصال به config متمرکز
        self.config = None
        try:
            from config.settings import config
            self.config = config
            self._logger.info("✅ Config loaded from config/bot_config.json")
        except ImportError as e:
            self._logger.warning(f"⚠️  Config not found: {e}. Using defaults.")
        
        # بارگیری تنظیمات اتصال از فایل‌ها
        self.connection_config = self._load_connection_config()
        
        # 🔥 همگام‌سازی متغیرهای مستقیم با کانفیگ بارگذاری شده
        if self.connection_config:
            self.login = self.connection_config.login
            self.password = self.connection_config.password
            self.server = self.connection_config.server
            self.mt5_path = self.connection_config.mt5_path

        # 🔥 کش قیمت‌های لحظه‌ای
        self.tick_cache: Dict[str, Dict[str, Any]] = {}
        self.last_tick_time: Dict[str, datetime] = {}

    def _load_connection_config(self) -> Any: # فرض شده خروجی ConnectionConfig است
        """بارگذاری تنظیمات اتصال با رعایت اولویت‌ها"""
        # اینجا از کلاس داخلی یا ساختار شما استفاده شده است
        try:
            from src.trading_bot.nds.models import ConnectionConfig
        except ImportError:
            # یک کلاس موقت اگر مدل یافت نشد
            class ConnectionConfig:
                def __init__(self):
                    self.login = 0; self.password = ""; self.server = ""
                    self.mt5_path = None; self.timeout = 30; self.retry_count = 3
                    self.real_time_enabled = True; self.tick_update_interval = 1.0
        
        config_obj = ConnectionConfig()
        sources = []
        
        # اولویت 1: config متمرکز (Settings.py)
        if self.config:
            try:
                creds = self.config.get_mt5_credentials()
                if creds:
                    config_obj.login = creds.get('login', config_obj.login)
                    config_obj.password = creds.get('password', config_obj.password)
                    config_obj.server = creds.get('server', config_obj.server)
                    config_obj.mt5_path = creds.get('mt5_path', config_obj.mt5_path)
                    config_obj.real_time_enabled = creds.get('real_time_enabled', True)
                    config_obj.tick_update_interval = creds.get('tick_update_interval', 1.0)
                    sources.append("central config")
            except Exception as e:
                self._logger.warning(f"Failed to load from central config: {e}")
        
        # اولویت 2: بررسی فایل‌های فیزیکی اگر موارد بالا خالی بودند
        if not all([config_obj.login, config_obj.password, config_obj.server]):
            possible_paths = [
                Path.cwd() / "config" / "mt5_credentials.json",
                Path.cwd() / "mt5_credentials.json",
                Path(__file__).parent.parent / "config" / "mt5_credentials.json"
            ]
            
            for file_path in possible_paths:
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            creds = json.load(f)
                        if all(k in creds for k in ['login', 'password', 'server']):
                            config_obj.login = creds.get('login')
                            config_obj.password = creds.get('password')
                            config_obj.server = creds.get('server')
                            config_obj.mt5_path = creds.get('mt5_path', config_obj.mt5_path)
                            sources.append(f"file: {file_path.name}")
                            break
                    except Exception as e:
                        self._logger.debug(f"Skip {file_path}: {e}")

        # لاگ نتیجه نهایی بارگذاری
        if all([config_obj.login, config_obj.password, config_obj.server]):
            self._logger.info(f"✅ Connection credentials loaded from: {', '.join(sources)}")
        else:
            self._logger.warning("⚠️ MT5 credentials are still incomplete.")
        
        return config_obj

    def connect(self) -> bool:
        """اتصال به MT5 با استفاده از آخرین داده‌های موجود در کلاس"""
        if self.connected:
            return True

        # 🔥 نکته مهم: همگام‌سازی نهایی قبل از اتصال
        # اگر از بیرون (bot.py) مقادیری ست شده، اولویت با آن‌هاست
        final_login = self.login or self.connection_config.login
        final_password = self.password or self.connection_config.password
        final_server = self.server or self.connection_config.server
        final_path = self.mt5_path or self.connection_config.mt5_path

        if not all([final_login, final_password, final_server]):
            self._logger.error(f"❌ Cannot connect: Missing credentials. (Login: {final_login}, Server: {final_server})")
            return False

        for attempt in range(1, self.connection_config.retry_count + 1):
            try:
                self._logger.info(f"🔄 Connection attempt {attempt}/{self.connection_config.retry_count}...")
                
                # Initialize
                init_params = {"timeout": self.connection_config.timeout * 1000}
                if final_path:
                    init_params["path"] = str(final_path)
                
                if not mt5.initialize(**init_params):
                    self._logger.error(f"❌ Initialize failed: {mt5.last_error()}")
                    time.sleep(2)
                    continue

                # Login
                if not mt5.login(login=int(final_login), password=final_password, server=final_server):
                    self._logger.error(f"❌ Login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    time.sleep(2)
                    continue

                # Success
                account_info = mt5.account_info()
                if account_info:
                    self.connected = True
                    self.session_start = datetime.now()
                    
                    # لاگ پیروزی!
                    self._logger.info(f"✅ Connected to {final_server} | Account: {final_login} | Equity: {account_info.equity}")
                    
                    # شروع مانیتورینگ Real-Time
                    if self.connection_config.real_time_enabled and self.real_time_monitor is None:
                        try:
                            from src.trading_bot.mt5_client import RealTimeMonitor # اصلاح مسیر ایمپورت بر اساس پروژه شما
                            self.real_time_monitor = RealTimeMonitor(self, self.connection_config.tick_update_interval)
                            self.real_time_monitor.start()
                        except Exception as e:
                            self._logger.error(f"⚠️ Could not start RealTimeMonitor: {e}")

                    return True
            
            except Exception as e:
                self._logger.error(f"💥 Critical connection error: {e}")
                time.sleep(2)

        return False

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """دریافت و لاگ اطلاعات حساب"""
        if not self.connected:
            return None
        
        acc = mt5.account_info()
        if acc is None:
            return None
            
        info = {
            'login': acc.login, 'balance': acc.balance, 'equity': acc.equity,
            'margin': acc.margin, 'free_margin': acc.margin_free,
            'leverage': acc.leverage, 'server': acc.server
        }
        self._logger.info(f"📊 Balance: ${info['balance']:.2f} | Equity: ${info['equity']:.2f}")
        return info
    
    def get_current_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """🔥 دریافت قیمت لحظه‌ای (Real-Time)"""
        if not self.connected:
            self._logger.error("❌ Not connected to MT5")
            return None
        
        try:
            # اولویت 1: از مانیتور Real-Time
            if self.real_time_monitor:
                price = self.real_time_monitor.get_current_price(symbol)
                if price:
                    # 🔥 اعتبارسنجی قیمت‌های مانیتور
                    if price.bid > 0 and price.ask > 0:
                        valid_last = price.last if hasattr(price, 'last') and price.last > 0 else price.bid
                        self._logger.debug(f"📡 RealTimeMonitor price for {symbol}: bid={price.bid:.2f}, ask={price.ask:.2f}, last={valid_last:.2f}")
                        return {
                            'bid': price.bid,
                            'ask': price.ask,
                            'last': valid_last,
                            'time': price.time,
                            'volume': price.volume,
                            'spread': price.ask - price.bid,
                            'source': 'real_time_monitor'
                        }
                    else:
                        self._logger.warning(f"⚠️ RealTimeMonitor returned invalid prices for {symbol}: bid={price.bid}, ask={price.ask}")
            
            # اولویت 2: دریافت مستقیم از MT5
            if not self._select_symbol(symbol):
                self._logger.warning(f"⚠️ Symbol {symbol} not selected in Market Watch")
                return None
            
            self._logger.debug(f"📡 Fetching direct tick for {symbol} from MT5...")
            tick = mt5.symbol_info_tick(symbol)
            
            if not tick:
                error = mt5.last_error()
                self._logger.error(f"❌ Failed to get tick for {symbol}: {error}")
                return None
            
            # 🔥 اعتبارسنجی کامل قیمت tick
            if tick.bid <= 0 or tick.ask <= 0:
                self._logger.error(f"❌ Invalid tick prices for {symbol}: bid={tick.bid}, ask={tick.ask}")
                return None
            
            # 🔥 تعیین قیمت last معتبر
            if hasattr(tick, 'last') and tick.last and tick.last > 0:
                valid_last = tick.last
            else:
                valid_last = tick.bid  # fallback to bid
                self._logger.debug(f"⚠️ Using bid as last for {symbol} (tick.last was {tick.last})")
            
            result = {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': valid_last,
                'time': datetime.fromtimestamp(tick.time),
                'volume': tick.volume,
                'spread': tick.ask - tick.bid,
                'source': 'direct_fetch'
            }
            
            # 🔥 لاگ قیمت برای دیباگ
            self._logger.debug(f"✅ Direct tick for {symbol}: bid={tick.bid:.2f}, ask={tick.ask:.2f}, last={valid_last:.2f}, spread={tick.ask - tick.bid:.2f}")
            
            # ذخیره در کش
            self.tick_cache[symbol] = result
            self.last_tick_time[symbol] = datetime.now()
            
            return result
            
        except Exception as e:
            self._logger.error(f"❌ Error getting current tick for {symbol}: {e}", exc_info=True)
            return None
    
    def _cleanup_cache(self):
        """پاکسازی کش قدیمی"""
        try:
            current_time = datetime.now()
            keys_to_remove = []
            
            for key, (_, timestamp) in self.data_cache.items():
                if (current_time - timestamp).total_seconds() > 300:  # 5 دقیقه
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.data_cache[key]
                
        except Exception as e:
            self._logger.warning(f"Error cleaning cache: {e}")
    
    def get_historical_data(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """دریافت داده‌های تاریخی"""
        if not self.connected:
            self._logger.error("❌ Not connected to MT5")
            return None
        
        # 🔥 کاهش زمان کش - برای Real-Time
        cache_key = f"{symbol}_{timeframe}_{bars}"
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            # فقط 10 ثانیه کش می‌ماند (به جای 60 ثانیه)
            cache_age = (datetime.now() - timestamp).total_seconds()
            if cache_age < 10:
                self._logger.debug(f"📦 Using cached data for {symbol} (age: {cache_age:.1f}s)")
                # 🔥 بررسی اعتبار داده‌های کش شده
                if not cached_data.empty and cached_data['close'].iloc[-1] > 0:
                    return cached_data.copy()
                else:
                    self._logger.warning(f"⚠️ Cached data invalid for {symbol}, fetching fresh...")
            else:
                self._logger.debug(f"🔄 Cache expired for {symbol} (age: {cache_age:.1f}s)")
        
        # مپ تایم‌فریم
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        tf = timeframe_map.get(timeframe.upper())
        if not tf:
            self._logger.error(f"❌ Invalid timeframe: {timeframe}")
            return None
        
        try:
            # انتخاب نماد
            if not self._select_symbol(symbol):
                self._logger.error(f"❌ Failed to select symbol: {symbol}")
                return None
            
            # دریافت داده
            self._logger.debug(f"📥 Fetching {bars} bars of {symbol} {timeframe} from MT5...")
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            
            if rates is None or len(rates) == 0:
                self._logger.error(f"❌ No data returned for {symbol} {timeframe}")
                return None
            
            # تبدیل به DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.sort_values('time').reset_index(drop=True)
            
            # اعمال تغییرات لازم
            if 'tick_volume' in df.columns:
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            # 🔥 دیباگ: لاگ آخرین کندل قبل از آپدیت
            if len(df) > 0:
                original_close = df['close'].iloc[-1]
                original_high = df['high'].iloc[-1]
                original_low = df['low'].iloc[-1]
                self._logger.debug(f"📊 Original last candle: close={original_close:.2f}, high={original_high:.2f}, low={original_low:.2f}")
            
            # 🔥 افزودن قیمت لحظه‌ای به آخرین رکورد
            tick = self.get_current_tick(symbol)
            if tick and len(df) > 0:
                # اعتبارسنجی قیمت tick
                if 'last' in tick and tick['last'] and tick['last'] > 0:
                    use_price = tick['last']
                    price_source = 'tick.last'
                elif 'bid' in tick and tick['bid'] and tick['bid'] > 0:
                    use_price = tick['bid']
                    price_source = 'tick.bid'
                    self._logger.warning(f"⚠️ Using bid price ({tick['bid']:.2f}) as last was invalid for {symbol}")
                else:
                    use_price = None
                    self._logger.error(f"❌ Invalid tick data for {symbol}: {tick}")
                
                if use_price:
                    # به‌روزرسانی close با قیمت لحظه‌ای
                    df.loc[len(df)-1, 'close'] = use_price
                    
                    # اگر high/low لحظه‌ای بیشتر/کمتر است، به‌روزرسانی کن
                    if use_price > df.loc[len(df)-1, 'high']:
                        df.loc[len(df)-1, 'high'] = use_price
                    if use_price < df.loc[len(df)-1, 'low']:
                        df.loc[len(df)-1, 'low'] = use_price
                    
                    self._logger.debug(f"🔄 Updated last candle with {price_source}: {use_price:.2f}")
                else:
                    self._logger.warning(f"⚠️ No valid real-time price for {symbol}, using historical close")
            else:
                if not tick:
                    self._logger.warning(f"⚠️ No tick data available for {symbol}")
                if len(df) == 0:
                    self._logger.error(f"❌ DataFrame is empty for {symbol}")
            
            # 🔥 دیباگ: لاگ آخرین کندل بعد از آپدیت
            if len(df) > 0:
                updated_close = df['close'].iloc[-1]
                updated_high = df['high'].iloc[-1]
                updated_low = df['low'].iloc[-1]
                self._logger.debug(f"📊 Updated last candle: close={updated_close:.2f}, high={updated_high:.2f}, low={updated_low:.2f}")
            
            # ذخیره در کش (با زمان کم)
            self.data_cache[cache_key] = (df.copy(), datetime.now())
            self._cleanup_cache()
            
            self._logger.info(f"✅ Data fetched: {symbol} {timeframe}, {len(df)} bars (Real-Time updated)")
            return df
            
        except Exception as e:
            self._logger.error(f"❌ Error fetching data for {symbol} {timeframe}: {e}", exc_info=True)
            return None
    
    def _select_symbol(self, symbol: str) -> bool:
        """انتخاب نماد در Market Watch"""
        if symbol in self.symbol_cache:
            return True
        
        try:
            if not mt5.symbol_select(symbol, True):
                error = mt5.last_error()
                self._logger.error(f"Failed to select {symbol}: {error}")
                return False
            
            self.symbol_cache[symbol] = True
            
            # 🔥 ثبت در مانیتور Real-Time
            if self.real_time_monitor:
                # اضافه کردن به مانیتورینگ
                pass
                
            return True
            
        except Exception as e:
            self._logger.error(f"Error selecting symbol: {e}")
            return False
    
    def send_order_real_time(self, symbol: str, order_type: str, volume: float, 
                            sl_price: float = None, tp_price: float = None,
                            sl_distance: float = None, tp_distance: float = None,
                            comment: str = "") -> Dict[str, Any]:
        """🔥 ارسال سفارش با قیمت لحظه‌ای و مدیریت Real-Time (نسخه Hardened)"""

        if not self.connected:
            return {'error': 'Not connected to MT5', 'success': False}

        try:
            # 🔹 اطمینان از فعال بودن نماد
            if not mt5.symbol_select(symbol, True):
                return {'error': f'Failed to select symbol {symbol}', 'success': False}

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'error': f'Symbol info not available for {symbol}', 'success': False}

            digits = symbol_info.digits
            point = symbol_info.point
            stop_level = symbol_info.trade_stops_level * point

            # 🔹 دریافت قیمت لحظه‌ای
            tick = self.get_current_tick(symbol)
            if not tick:
                return {'error': 'Failed to get real-time price', 'success': False}

            current_bid = tick['bid']
            current_ask = tick['ask']

            # 🔹 تعیین نوع سفارش
            if order_type.upper() == 'BUY':
                entry_price = current_ask
                mt5_type = mt5.ORDER_TYPE_BUY
                stop_loss = (
                    entry_price - sl_distance if sl_distance else sl_price or 0
                )
                take_profit = (
                    entry_price + tp_distance if tp_distance else tp_price or 0
                )

            elif order_type.upper() == 'SELL':
                entry_price = current_bid
                mt5_type = mt5.ORDER_TYPE_SELL
                stop_loss = (
                    entry_price + sl_distance if sl_distance else sl_price or 0
                )
                take_profit = (
                    entry_price - tp_distance if tp_distance else tp_price or 0
                )
            else:
                return {'error': f'Invalid order type: {order_type}', 'success': False}

            # 🔹 نرمال‌سازی قیمت‌ها
            entry_price = round(entry_price, digits)
            stop_loss = round(stop_loss, digits) if stop_loss else 0
            take_profit = round(take_profit, digits) if take_profit else 0

            # 🔹 بررسی حداقل فاصله مجاز SL/TP
            if stop_loss:
                if abs(entry_price - stop_loss) < stop_level:
                    return {'error': f'SL too close to price (min {stop_level})', 'success': False}

            if take_profit:
                if abs(entry_price - take_profit) < stop_level:
                    return {'error': f'TP too close to price (min {stop_level})', 'success': False}

            # 🔹 انتخاب filling mode سازگار
            filling_type = (
                mt5.ORDER_FILLING_IOC
                if symbol_info.filling_mode & mt5.ORDER_FILLING_IOC
                else mt5.ORDER_FILLING_RETURN
            )

            # 🔹 لاگ دقیق
            self._logger.info(f"""
    🔥 Real-Time Order Parameters:
    Symbol: {symbol}
    Type: {order_type}
    Entry: {entry_price}
    SL: {stop_loss}
    TP: {take_profit}
    Volume: {volume}
    Filling: {filling_type}
    """)


            request_comment = comment or "NDS_SCALP_V1"
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_type,
                "price": entry_price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 10,
                "magic": 202401,
                "comment": request_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_type,
            }

            request_time = datetime.now()
            result = mt5.order_send(request)

            if result is None:
                last_error = mt5.last_error()
                error_msg = f"order_send returned None | MT5 error: {last_error}"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False}

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed [{result.retcode}]: {result.comment}"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': result.retcode}

            order_ticket = getattr(result, "order", None)
            position_ticket = self.resolve_position_ticket(
                symbol=symbol,
                magic=request.get("magic"),
                comment=request_comment,
                opened_after_time=request_time,
                timeout_sec=5,
            )

            return {
                'success': True,
                'order_ticket': order_ticket,
                'position_ticket': position_ticket,
                'ticket': position_ticket or order_ticket,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'volume': volume,
                'time': datetime.now(),
                'comment': result.comment,
                'bid_at_entry': current_bid,
                'ask_at_entry': current_ask,
            }

        except Exception as e:
            error_msg = f"Real-Time order error: {e}"
            self._logger.error(error_msg)
            return {'error': error_msg, 'success': False}

    def resolve_position_ticket(
        self,
        symbol: str,
        magic: Optional[int],
        comment: Optional[str],
        opened_after_time: datetime,
        timeout_sec: int = 5,
    ) -> Optional[int]:
        """تلاش برای یافتن position_ticket بعد از ارسال سفارش."""
        if not self.connected:
            return None

        deadline = time.time() + timeout_sec
        matched_positions: List[Any] = []

        while time.time() < deadline:
            try:
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    matched_positions = [
                        pos for pos in positions
                        if (magic is None or pos.magic == magic)
                        and (comment is None or (pos.comment or "").strip() == comment.strip())
                    ]
                    if matched_positions:
                        break
            except Exception:
                matched_positions = []

            time.sleep(0.5)

        if not matched_positions:
            return None

        def _pos_time(pos_obj: Any) -> datetime:
            try:
                return datetime.fromtimestamp(pos_obj.time)
            except Exception:
                return datetime.min

        newest = max(matched_positions, key=_pos_time)
        pos_time = _pos_time(newest)
        if pos_time < opened_after_time - timedelta(seconds=2):
            return None
        return int(newest.ticket)

    def get_position_history(self, position_ticket: int, days_back: int = 2) -> Dict[str, Any]:
        """دریافت تاریخچه معاملات برای یک position_ticket جهت تایید بسته شدن."""
        if not self.connected:
            self._logger.warning("⚠️ Not connected to MT5")
            return {}

        try:
            now = datetime.now()
            from_time = now - timedelta(days=days_back)
            deals = mt5.history_deals_get(from_time, now)
            if deals is None:
                error = mt5.last_error()
                self._logger.warning(f"⚠️ Failed to get deals history: {error}")
                return {}

            position_deals = []
            for deal in deals:
                deal_position_id = getattr(deal, "position_id", None)
                deal_position = getattr(deal, "position", None)
                if deal_position_id == position_ticket or deal_position == position_ticket:
                    position_deals.append(deal)

            if not position_deals:
                return {}

            total_profit = 0.0
            exit_price = None
            close_time = None
            volume_closed = 0.0
            reason = "Manual/Other"

            for deal in position_deals:
                total_profit += float(getattr(deal, "profit", 0.0) or 0.0)
                entry_flag = getattr(deal, "entry", None)
                if entry_flag in (getattr(mt5, "DEAL_ENTRY_OUT", None), getattr(mt5, "DEAL_ENTRY_INOUT", None)):
                    exit_price = float(getattr(deal, "price", 0.0) or 0.0)
                    close_time = datetime.fromtimestamp(getattr(deal, "time", 0))
                    volume_closed += float(getattr(deal, "volume", 0.0) or 0.0)

                comment = (getattr(deal, "comment", "") or "").lower()
                if "tp" in comment or "sl" in comment:
                    reason = "TP/SL"

            return {
                "position_ticket": position_ticket,
                "total_profit": total_profit,
                "exit_price": exit_price,
                "close_time": close_time,
                "volume_closed": volume_closed,
                "reason": reason,
            }

        except Exception as e:
            self._logger.error(f"❌ Error getting position history: {e}", exc_info=True)
            return {}

    
    def send_order(self, symbol: str, order_type: str, volume: float, 
                   stop_loss: float = None, take_profit: float = None, 
                   comment: str = "", order_action: str = "MARKET") -> Optional[int]:
        """
        🔄 ارسال سفارش - نسخه بهبودیافته با پشتیبانی از انواع مختلف سفارش
        
        Args:
            symbol: نام نماد
            order_type: نوع سفارش ('BUY' یا 'SELL')
            volume: حجم سفارش
            stop_loss: قیمت استاپ‌لاس (اختیاری)
            take_profit: قیمت تیک‌پروفیت (اختیاری)
            comment: توضیحات سفارش
            order_action: نوع اجرای سفارش ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT')
            
        Returns:
            شماره تیکت سفارش یا None در صورت خطا
        """
        try:
            # استفاده از متد هوشمند send_order_with_type
            result = self.send_order_with_type(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment,
                order_action=order_action
            )
            
            # 🔥 FIX: بررسی اینکه result وجود دارد و None نیست
            if result is None:
                self._logger.error("❌ نتیجه ارسال سفارش None است (ارتباط با MT5 قطع شده؟)")
                return None
            
            # 🔥 FIX: بررسی نوع result و وجود کلید success
            if not isinstance(result, dict):
                self._logger.error(f"❌ نوع نتیجه نامعتبر است: {type(result)}")
                return None
            
            # 🔥 FIX: بررسی وجود کلید success در دیکشنری
            if 'success' not in result:
                self._logger.error(f"❌ کلید 'success' در نتیجه وجود ندارد. نتیجه: {result}")
                return None
            
            return result.get('ticket') if result.get('success') else None
            
        except Exception as e:
            self._logger.error(f"❌ خطا در تابع send_order: {e}")
            return None
    
    def send_limit_order(self, symbol: str, order_type: str, volume: float, 
                        limit_price: float, stop_loss: float = None, 
                        take_profit: float = None, comment: str = "") -> Dict[str, Any]:
        """
        📌 ارسال سفارش Limit (پندینگ)
        
        Args:
            symbol: نام نماد
            order_type: نوع سفارش ('BUY_LIMIT' یا 'SELL_LIMIT')
            volume: حجم سفارش
            limit_price: قیمت Limit
            stop_loss: قیمت استاپ‌لاس (اختیاری)
            take_profit: قیمت تیک‌پروفیت (اختیاری)
            comment: توضیحات سفارش
            
        Returns:
            دیکشنری حاوی نتیجه سفارش
        """
        if not self.connected:
            return {'error': 'Not connected to MT5', 'success': False}
        
        try:
            # 🔥 دریافت قیمت لحظه‌ای برای لاگ
            tick = self.get_current_tick(symbol)
            current_bid = tick.get('bid') if tick else 0
            current_ask = tick.get('ask') if tick else 0
            
            # تبدیل نوع سفارش به کد MT5
            order_type_map = {
                'BUY_LIMIT': mt5.ORDER_TYPE_BUY_LIMIT,
                'SELL_LIMIT': mt5.ORDER_TYPE_SELL_LIMIT
            }
            
            mt5_order_type = order_type_map.get(order_type.upper())
            if not mt5_order_type:
                return {'error': f'Invalid limit order type: {order_type}', 'success': False}
            
            # 🔥 اعتبارسنجی قیمت Limit
            if order_type.upper() == 'BUY_LIMIT' and limit_price >= current_ask:
                return {
                    'error': f'Buy Limit price ({limit_price}) must be below current ask ({current_ask})', 
                    'success': False
                }
            elif order_type.upper() == 'SELL_LIMIT' and limit_price <= current_bid:
                return {
                    'error': f'Sell Limit price ({limit_price}) must be above current bid ({current_bid})', 
                    'success': False
                }
            
            # ساخت درخواست Limit
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": limit_price,
                "sl": stop_loss if stop_loss else 0,
                "tp": take_profit if take_profit else 0,
                "deviation": 5,
                "magic": 202402,  # Magic متفاوت برای پندینگ‌ها
                "comment": f"{comment} | Limit Order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            
            # ارسال سفارش
            result = mt5.order_send(request)
            
            # 🔥 FIX: بررسی اینکه result برابر با None نباشد
            if result is None:
                error_msg = "MT5 returned None for order_send() - connection may be lost"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': None}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Limit order failed: {result.comment}"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': result.retcode}
            
            # لاگ موفقیت
            self._logger.info(f"""
            ✅ Limit Order Placed Successfully!
               Symbol: {symbol}
               Type: {order_type}
               Limit Price: {limit_price:.2f}
               Volume: {volume}
               Current Bid: {current_bid:.2f}
               Current Ask: {current_ask:.2f}
               Ticket: {result.order}
            """)
            
            return {
                'success': True,
                'ticket': result.order,
                'order_type': order_type,
                'limit_price': limit_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'volume': volume,
                'current_bid': current_bid,
                'current_ask': current_ask,
                'time': datetime.now(),
                'comment': result.comment
            }
            
        except Exception as e:
            error_msg = f"Limit order error: {e}"
            self._logger.error(error_msg)
            return {'error': error_msg, 'success': False}
    
    def send_pending_order(self, symbol: str, order_type: str, volume: float, 
                         pending_price: float, stop_loss: float = None, 
                         take_profit: float = None, comment: str = "") -> Dict[str, Any]:
        """
        ⏳ ارسال سفارش Pending (مستعار برای send_limit_order)
        
        Args:
            symbol: نام نماد
            order_type: نوع سفارش ('BUY_LIMIT', 'SELL_LIMIT', 'BUY_STOP', 'SELL_STOP')
            volume: حجم سفارش
            pending_price: قیمت پندینگ
            stop_loss: قیمت استاپ‌لاس (اختیاری)
            take_profit: قیمت تیک‌پروفیت (اختیاری)
            comment: توضیحات سفارش
            
        Returns:
            دیکشنری حاوی نتیجه سفارش
        """
        # پشتیبانی از انواع مختلف پندینگ
        if order_type.upper() in ['BUY_LIMIT', 'SELL_LIMIT']:
            return self.send_limit_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                limit_price=pending_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment
            )
        elif order_type.upper() in ['BUY_STOP', 'SELL_STOP']:
            return self.send_stop_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                stop_price=pending_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment
            )
        else:
            return {'error': f'Invalid pending order type: {order_type}', 'success': False}
    
    def send_order_with_type(self, symbol: str, order_type: str, volume: float, 
                           stop_loss: float = None, take_profit: float = None, 
                           comment: str = "", order_action: str = "MARKET",
                           limit_price: float = None, stop_price: float = None) -> Dict[str, Any]:
        """
        🎯 ارسال سفارش با مدیریت هوشمند انواع مختلف
        
        Args:
            symbol: نام نماد
            order_type: نوع سفارش ('BUY', 'SELL', 'BUY_LIMIT', 'SELL_LIMIT', 'BUY_STOP', 'SELL_STOP')
            volume: حجم سفارش
            stop_loss: قیمت استاپ‌لاس (اختیاری)
            take_profit: قیمت تیک‌پروفیت (اختیاری)
            comment: توضیحات سفارش
            order_action: نوع اجرای سفارش ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT')
            limit_price: قیمت Limit (برای سفارش‌های Limit)
            stop_price: قیمت Stop (برای سفارش‌های Stop)
            
        Returns:
            دیکشنری حاوی نتیجه کامل سفارش
        """
        order_type_upper = order_type.upper()
        order_action_upper = order_action.upper()
        
        try:
            # 🔥 دریافت قیمت لحظه‌ای برای اعتبارسنجی
            tick = self.get_current_tick(symbol)
            if not tick:
                return {'error': 'Failed to get real-time price', 'success': False}
            
            current_bid = tick['bid']
            current_ask = tick['ask']
            
            # 🎯 تشخیص نوع سفارش و ارسال مناسب
            if order_action_upper == 'MARKET':
                # سفارش مارکت
                if order_type_upper in ['BUY', 'SELL']:
                    return self.send_order_real_time(
                        symbol=symbol,
                        order_type=order_type,
                        volume=volume,
                        sl_price=stop_loss,
                        tp_price=take_profit,
                        comment=comment
                    )
                else:
                    return {'error': f'Invalid market order type: {order_type}', 'success': False}
            
            elif order_action_upper == 'LIMIT':
                # سفارش Limit
                if not limit_price:
                    return {'error': 'Limit price required for LIMIT orders', 'success': False}
                
                if order_type_upper in ['BUY_LIMIT', 'SELL_LIMIT']:
                    return self.send_limit_order(
                        symbol=symbol,
                        order_type=order_type,
                        volume=volume,
                        limit_price=limit_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        comment=comment
                    )
                else:
                    return {'error': f'Invalid limit order type: {order_type}', 'success': False}
            
            elif order_action_upper == 'STOP':
                # سفارش Stop
                if not stop_price:
                    return {'error': 'Stop price required for STOP orders', 'success': False}
                
                if order_type_upper in ['BUY_STOP', 'SELL_STOP']:
                    return self.send_stop_order(
                        symbol=symbol,
                        order_type=order_type,
                        volume=volume,
                        stop_price=stop_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        comment=comment
                    )
                else:
                    return {'error': f'Invalid stop order type: {order_type}', 'success': False}
            
            elif order_action_upper == 'STOP_LIMIT':
                # سفارش Stop Limit
                if not limit_price or not stop_price:
                    return {'error': 'Both stop price and limit price required for STOP_LIMIT orders', 'success': False}
                
                # MT5 از Stop Limit پشتیبانی نمی‌کند، از ترکیب Stop + Limit استفاده می‌کنیم
                self._logger.warning("⚠️ MT5 doesn't support STOP_LIMIT natively, using alternative approach")
                
                # اول یک Stop قرار می‌دهیم
                stop_result = self.send_stop_order(
                    symbol=symbol,
                    order_type=order_type,
                    volume=volume,
                    stop_price=stop_price,
                    comment=f"{comment} | Stop part of Stop-Limit"
                )
                
                if not stop_result.get('success'):
                    return stop_result
                
                return {
                    'success': True,
                    'ticket': stop_result.get('ticket'),
                    'order_type': order_type,
                    'action': 'STOP_LIMIT',
                    'stop_price': stop_price,
                    'limit_price': limit_price,
                    'volume': volume,
                    'message': 'Stop order placed (MT5 does not support Stop-Limit natively)',
                    'comment': comment
                }
            
            else:
                return {'error': f'Invalid order action: {order_action}', 'success': False}
                
        except Exception as e:
            error_msg = f"Order with type error: {e}"
            self._logger.error(error_msg)
            return {'error': error_msg, 'success': False}
    
    def send_stop_order(self, symbol: str, order_type: str, volume: float, 
                       stop_price: float, stop_loss: float = None, 
                       take_profit: float = None, comment: str = "") -> Dict[str, Any]:
        """
        ⚡ ارسال سفارش Stop
        
        Args:
            symbol: نام نماد
            order_type: نوع سفارش ('BUY_STOP' یا 'SELL_STOP')
            volume: حجم سفارش
            stop_price: قیمت Stop
            stop_loss: قیمت استاپ‌لاس (اختیاری)
            take_profit: قیمت تیک‌پروفیت (اختیاری)
            comment: توضیحات سفارش
            
        Returns:
            دیکشنری حاوی نتیجه سفارش
        """
        if not self.connected:
            return {'error': 'Not connected to MT5', 'success': False}
        
        try:
            # 🔥 دریافت قیمت لحظه‌ای برای اعتبارسنجی
            tick = self.get_current_tick(symbol)
            if not tick:
                return {'error': 'Failed to get real-time price', 'success': False}
            
            current_bid = tick['bid']
            current_ask = tick['ask']
            
            # تبدیل نوع سفارش به کد MT5
            order_type_map = {
                'BUY_STOP': mt5.ORDER_TYPE_BUY_STOP,
                'SELL_STOP': mt5.ORDER_TYPE_SELL_STOP
            }
            
            mt5_order_type = order_type_map.get(order_type.upper())
            if not mt5_order_type:
                return {'error': f'Invalid stop order type: {order_type}', 'success': False}
            
            # 🔥 اعتبارسنجی قیمت Stop
            if order_type.upper() == 'BUY_STOP' and stop_price <= current_ask:
                return {
                    'error': f'Buy Stop price ({stop_price}) must be above current ask ({current_ask})', 
                    'success': False
                }
            elif order_type.upper() == 'SELL_STOP' and stop_price >= current_bid:
                return {
                    'error': f'Sell Stop price ({stop_price}) must be below current bid ({current_bid})', 
                    'success': False
                }
            
            # ساخت درخواست Stop
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": stop_price,
                "sl": stop_loss if stop_loss else 0,
                "tp": take_profit if take_profit else 0,
                "deviation": 5,
                "magic": 202403,  # Magic متفاوت برای Stopها
                "comment": f"{comment} | Stop Order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            
            # ارسال سفارش
            result = mt5.order_send(request)
            
            # 🔥 FIX: بررسی اینکه result برابر با None نباشد
            if result is None:
                error_msg = "MT5 returned None for order_send() - connection may be lost"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': None}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Stop order failed: {result.comment}"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': result.retcode}
            
            # لاگ موفقیت
            self._logger.info(f"""
            ✅ Stop Order Placed Successfully!
               Symbol: {symbol}
               Type: {order_type}
               Stop Price: {stop_price:.2f}
               Volume: {volume}
               Current Bid: {current_bid:.2f}
               Current Ask: {current_ask:.2f}
               Ticket: {result.order}
            """)
            
            return {
                'success': True,
                'ticket': result.order,
                'order_type': order_type,
                'stop_price': stop_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'volume': volume,
                'current_bid': current_bid,
                'current_ask': current_ask,
                'time': datetime.now(),
                'comment': result.comment
            }
            
        except Exception as e:
            error_msg = f"Stop order error: {e}"
            self._logger.error(error_msg)
            return {'error': error_msg, 'success': False}
    
    def get_open_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """دریافت پوزیشن‌های باز
        
        Args:
            symbol: نام نماد (اختیاری). اگر مشخص شود فقط پوزیشن‌های آن نماد برگردانده می‌شوند.
        
        Returns:
            لیستی از دیکشنری‌های حاوی اطلاعات پوزیشن‌های باز
        """
        if not self.connected:
            self._logger.warning("⚠️ Not connected to MT5")
            return []
        
        try:
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            
            if positions is None:
                error = mt5.last_error()
                if error[0] != mt5.TRADE_RETCODE_NO_ERROR:
                    self._logger.warning(f"⚠️ No open positions found for {symbol if symbol else 'all symbols'}: {error}")
                return []
            
            if len(positions) == 0:
                self._logger.debug(f"📊 No open positions found for {symbol if symbol else 'all symbols'}")
                return []
            
            result: List[Dict[str, Any]] = []
            for pos in positions:
                raw_info = {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'entry_price': pos.price_open,
                    'current_price': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'magic': pos.magic,
                    'comment': pos.comment,
                    'time': datetime.fromtimestamp(pos.time),
                    'time_update': datetime.fromtimestamp(pos.time_update),
                }
                result.append(normalize_position(raw_info))
            
            self._logger.info(f"📊 Found {len(result)} open position(s) for {symbol if symbol else 'all symbols'}")
            return result
            
        except Exception as e:
            self._logger.error(f"❌ Error getting open positions: {e}", exc_info=True)
            return []
    
    def get_pending_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        دریافت سفارش‌های Pending (باز)
        
        Args:
            symbol: نام نماد (اختیاری)
            
        Returns:
            لیستی از سفارش‌های Pending
        """
        if not self.connected:
            self._logger.warning("⚠️ Not connected to MT5")
            return []
        
        try:
            orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
            
            if orders is None:
                error = mt5.last_error()
                if error[0] != mt5.TRADE_RETCODE_NO_ERROR:
                    self._logger.debug(f"⚠️ No pending orders found for {symbol if symbol else 'all symbols'}: {error}")
                return []
            
            result = []
            for order in orders:
                order_info = {
                    'ticket': order.ticket,
                    'symbol': order.symbol,
                    'type': self._get_order_type_name(order.type),
                    'volume': order.volume_current,
                    'price_open': order.price_open,
                    'price_current': order.price_current,
                    'sl': order.sl,
                    'tp': order.tp,
                    'magic': order.magic,
                    'comment': order.comment,
                    'time_setup': datetime.fromtimestamp(order.time_setup),
                    'time_expiration': datetime.fromtimestamp(order.time_expiration) if order.time_expiration > 0 else None,
                    'state': order.state,
                }
                result.append(order_info)
            
            self._logger.info(f"📋 Found {len(result)} pending order(s) for {symbol if symbol else 'all symbols'}")
            return result
            
        except Exception as e:
            self._logger.error(f"❌ Error getting pending orders: {e}")
            return []
    
    def _get_order_type_name(self, order_type: int) -> str:
        """تبدیل کد نوع سفارش به نام"""
        type_map = {
            mt5.ORDER_TYPE_BUY: 'BUY',
            mt5.ORDER_TYPE_SELL: 'SELL',
            mt5.ORDER_TYPE_BUY_LIMIT: 'BUY_LIMIT',
            mt5.ORDER_TYPE_SELL_LIMIT: 'SELL_LIMIT',
            mt5.ORDER_TYPE_BUY_STOP: 'BUY_STOP',
            mt5.ORDER_TYPE_SELL_STOP: 'SELL_STOP',
            mt5.ORDER_TYPE_BUY_STOP_LIMIT: 'BUY_STOP_LIMIT',
            mt5.ORDER_TYPE_SELL_STOP_LIMIT: 'SELL_STOP_LIMIT',
        }
        return type_map.get(order_type, f'UNKNOWN_{order_type}')
    
    def cancel_order(self, ticket: int) -> Dict[str, Any]:
        """
        لغو سفارش Pending
        
        Args:
            ticket: شماره تیکت سفارش
            
        Returns:
            نتیجه لغو سفارش
        """
        if not self.connected:
            return {'error': 'Not connected to MT5', 'success': False}
        
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
                "comment": "Cancelled by MT5Client",
            }
            
            result = mt5.order_send(request)
            
            # 🔥 FIX: بررسی اینکه result برابر با None نباشد
            if result is None:
                error_msg = "MT5 returned None for order_send() - connection may be lost"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': None}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Failed to cancel order {ticket}: {result.comment}"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': result.retcode}
            
            self._logger.info(f"✅ Order {ticket} cancelled successfully")
            return {
                'success': True,
                'ticket': ticket,
                'comment': result.comment,
                'time': datetime.now()
            }
            
        except Exception as e:
            error_msg = f"Error cancelling order {ticket}: {e}"
            self._logger.error(error_msg)
            return {'error': error_msg, 'success': False}
    
    def modify_order(self, ticket: int, new_price: float = None, 
                    new_sl: float = None, new_tp: float = None) -> Dict[str, Any]:
        """
        ویرایش سفارش Pending
        
        Args:
            ticket: شماره تیکت سفارش
            new_price: قیمت جدید (اختیاری)
            new_sl: استاپ‌لاس جدید (اختیاری)
            new_tp: تیک‌پروفیت جدید (اختیاری)
            
        Returns:
            نتیجه ویرایش سفارش
        """
        if not self.connected:
            return {'error': 'Not connected to MT5', 'success': False}
        
        try:
            # دریافت اطلاعات سفارش فعلی
            orders = mt5.orders_get(ticket=ticket)
            if not orders or len(orders) == 0:
                return {'error': f'Order {ticket} not found', 'success': False}
            
            order = orders[0]
            
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket,
                "price": new_price if new_price else order.price_open,
                "sl": new_sl if new_sl else order.sl,
                "tp": new_tp if new_tp else order.tp,
                "comment": f"Modified by MT5Client | Original: {order.comment}",
            }
            
            result = mt5.order_send(request)
            
            # 🔥 FIX: بررسی اینکه result برابر با None نباشد
            if result is None:
                error_msg = "MT5 returned None for order_send() - connection may be lost"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': None}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Failed to modify order {ticket}: {result.comment}"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': result.retcode}
            
            self._logger.info(f"✅ Order {ticket} modified successfully")
            return {
                'success': True,
                'ticket': ticket,
                'new_price': new_price,
                'new_sl': new_sl,
                'new_tp': new_tp,
                'comment': result.comment,
                'time': datetime.now()
            }
            
        except Exception as e:
            error_msg = f"Error modifying order {ticket}: {e}"
            self._logger.error(error_msg)
            return {'error': error_msg, 'success': False}
    
    def modify_position(self, ticket: int, new_sl: float = None, 
                       new_tp: float = None) -> Dict[str, Any]:
        """
        ویرایش پوزیشن باز
        
        Args:
            ticket: شماره تیکت پوزیشن
            new_sl: استاپ‌لاس جدید (اختیاری)
            new_tp: تیک‌پروفیت جدید (اختیاری)
            
        Returns:
            نتیجه ویرایش پوزیشن
        """
        if not self.connected:
            return {'error': 'Not connected to MT5', 'success': False}
        
        try:
            # دریافت اطلاعات پوزیشن فعلی
            positions = mt5.positions_get(ticket=ticket)
            if not positions or len(positions) == 0:
                return {'error': f'Position {ticket} not found', 'success': False}
            
            position = positions[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl if new_sl else position.sl,
                "tp": new_tp if new_tp else position.tp,
                "symbol": position.symbol,
                "comment": f"Modified by MT5Client",
            }
            
            result = mt5.order_send(request)
            
            # 🔥 FIX: بررسی اینکه result برابر با None نباشد
            if result is None:
                error_msg = "MT5 returned None for order_send() - connection may be lost"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': None}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Failed to modify position {ticket}: {result.comment}"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': result.retcode}
            
            self._logger.info(f"✅ Position {ticket} modified successfully")
            return {
                'success': True,
                'ticket': ticket,
                'new_sl': new_sl,
                'new_tp': new_tp,
                'symbol': position.symbol,
                'comment': result.comment,
                'time': datetime.now()
            }
            
        except Exception as e:
            error_msg = f"Error modifying position {ticket}: {e}"
            self._logger.error(error_msg)
            return {'error': error_msg, 'success': False}
    
    def close_position(self, ticket: int, volume: float = None, 
                      comment: str = "Closed by MT5Client") -> Dict[str, Any]:
        """
        بستن پوزیشن
        
        Args:
            ticket: شماره تیکت پوزیشن
            volume: حجمی که باید بسته شود (اختیاری - اگر None باشد، کل حجم بسته می‌شود)
            comment: توضیحات بستن
            
        Returns:
            نتیجه بستن پوزیشن
        """
        if not self.connected:
            return {'error': 'Not connected to MT5', 'success': False}
        
        try:
            # دریافت اطلاعات پوزیشن
            positions = mt5.positions_get(ticket=ticket)
            if not positions or len(positions) == 0:
                return {'error': f'Position {ticket} not found', 'success': False}
            
            position = positions[0]
            
            # تعیین حجم
            close_volume = volume if volume else position.volume
            
            # تعیین نوع معکوس برای بستن
            if position.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(position.symbol).bid
            else:  # SELL
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(position.symbol).ask
            
            # 🔥 FIX: بررسی اینکه قیمت معتبر باشد
            if price <= 0:
                return {'error': f'Invalid close price for {position.symbol}: {price}', 'success': False}
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": close_type,
                "price": price,
                "deviation": 10,
                "magic": 202404,  # Magic برای بستن
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            
            # 🔥 FIX: بررسی اینکه result برابر با None نباشد
            if result is None:
                error_msg = "MT5 returned None for order_send() - connection may be lost"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': None}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Failed to close position {ticket}: {result.comment}"
                self._logger.error(error_msg)
                return {'error': error_msg, 'success': False, 'retcode': result.retcode}
            
            profit = position.profit if hasattr(position, 'profit') else 0
            self._logger.info(f"✅ Position {ticket} closed successfully | Profit: ${profit:.2f}")
            
            return {
                'success': True,
                'ticket': ticket,
                'closed_volume': close_volume,
                'profit': profit,
                'price': price,
                'comment': result.comment,
                'time': datetime.now()
            }
            
        except Exception as e:
            error_msg = f"Error closing position {ticket}: {e}"
            self._logger.error(error_msg)
            return {'error': error_msg, 'success': False}
    
    def get_trading_history(self, days_back: int = 1) -> List[Dict[str, Any]]:
        """
        دریافت تاریخچه معاملات
        
        Args:
            days_back: تعداد روزهای گذشته
            
        Returns:
            لیستی از تاریخچه معاملات
        """
        if not self.connected:
            self._logger.warning("⚠️ Not connected to MT5")
            return []
        
        try:
            from_date = datetime.now() - timedelta(days=days_back)
            to_date = datetime.now()
            
            history = mt5.history_deals_get(from_date, to_date)
            if history is None:
                self._logger.debug(f"No trading history found for last {days_back} days")
                return []
            
            result = []
            for deal in history:
                deal_info = {
                    'ticket': deal.ticket,
                    'order': deal.order,
                    'position': deal.position_id,
                    'symbol': deal.symbol,
                    'type': 'BUY' if deal.type == 0 else 'SELL',
                    'entry': deal.entry,
                    'volume': deal.volume,
                    'price': deal.price,
                    'profit': deal.profit,
                    'commission': deal.commission,
                    'swap': deal.swap,
                    'fee': deal.fee,
                    'time': datetime.fromtimestamp(deal.time),
                    'magic': deal.magic,
                    'comment': deal.comment,
                }
                result.append(deal_info)
            
            self._logger.info(f"📜 Found {len(result)} historical deals in last {days_back} days")
            return result
            
        except Exception as e:
            self._logger.error(f"❌ Error getting trading history: {e}")
            return []
    
    def disconnect(self):
        """قطع اتصال از MT5"""
        # 🔥 توقف مانیتور Real-Time
        if self.real_time_monitor:
            self.real_time_monitor.stop()
            self.real_time_monitor = None
        
        if self.connected:
            try:
                mt5.shutdown()
                self.connected = False
                self.session_start = None
                self._logger.info("Disconnected from MT5 (Real-Time monitor stopped)")
            except Exception as e:
                self._logger.error(f"Error disconnecting: {e}")
