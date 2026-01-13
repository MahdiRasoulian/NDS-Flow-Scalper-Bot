import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

def extract_mt5_data(symbol: str, timeframe, start_date: datetime, end_date: datetime):
    """
    اتصال به MT5 و استخراج داده کندلی.
    """
    # ۱. اتصال به MT5
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return None

    # ۲. تنظیمات: اطمینان از اینکه MT5 به سرور متصل است
    if not mt5.terminal_info().connected:
        print("MT5 is not connected to the trading server. Please open the MT5 terminal.")
        mt5.shutdown()
        return None

    # ۳. تبدیل تاریخ به فرمت UTC/سرور
    timezone = pytz.timezone("Etc/UTC") 
    date_from = timezone.localize(start_date)
    date_to = timezone.localize(end_date)
    
    # ۴. استخراج کندل‌ها
    # mt5.copy_rates_range بهترین راه برای بک‌تست طولانی است
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    
    # ۵. قطع اتصال
    mt5.shutdown()

    # ۶. تبدیل به DataFrame
    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol}.")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    # مرتب‌سازی و انتخاب ستون‌های لازم
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"✅ Data extracted successfully: {len(df)} bars of {symbol} found.")
    return df

if __name__ == '__main__':
    # --- تنظیمات استخراج ---
    SYMBOL = "XAUUSD!"
    # تایم فریم مورد نظر (مثلاً 1 Hour)
    TIMEFRAME = mt5.TIMEFRAME_M15
    # محدوده زمانی مورد نظر (مثلاً 1 سال گذشته)
    START_DATE = datetime(2025, 11, 1)
    END_DATE = datetime(2026, 1, 9)
    
    # --- اجرای استخراج ---
    historical_data = extract_mt5_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)

    # --- ذخیره‌سازی داده ---
    if historical_data is not None:
        file_name = f"{SYMBOL}_{TIMEFRAME}_{START_DATE.year}-{END_DATE.year}.csv"
        # ⚠️ محل ذخیره: فایل را در همان پوشه‌ای ذخیره کنید که backtester.py در آن قرار دارد.
        historical_data.to_csv(file_name, index=False)
        print(f"✅ Data saved to: {file_name}")