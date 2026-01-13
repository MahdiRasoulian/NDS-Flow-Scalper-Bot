# src/ui/cli.py

import sys
from datetime import datetime
from config.settings import config

def get_user_input(prompt: str, default: str = "") -> str:
    """دریافت ورودی از کاربر"""
    try:
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            return user_input if user_input else default
        else:
            return input(f"{prompt}: ").strip()
    except (KeyboardInterrupt, EOFError):
        return ""

def print_banner():
    """چاپ بنر خوش‌آمدگویی"""
    SYMBOL = config.get('trading_settings.SYMBOL', None)
    TIMEFRAME = config.get('trading_settings.TIMEFRAME', None)
    RISK_PERCENT = config.get('risk_settings.RISK_PERCENT', None)
    MIN_CONFIDENCE = config.get('risk_settings.MIN_CONFIDENCE', None)
    MODE = config.get('trading_settings.MODE', None)
    MIN_TIME = config.get('trading_rules.MIN_TIME_BETWEEN_TRADES_MINUTES', None)
    MAX_POS = config.get('trading_rules.MAX_POSITIONS', None)

    print("\n" + "="*70)
    print("🎯 NDS Trading Bot Pro - نسخه ساختار یافته")
    print("="*70)
    print(f"📅 تاریخ: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"⏰ زمان شروع: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    print(f"📊 نماد: {SYMBOL}")
    print(f"⏱️  تایم‌فریم: {TIMEFRAME}")
    print(f"💰 ریسک: {RISK_PERCENT}%")
    print(f"🎯 حداقل اعتماد: {MIN_CONFIDENCE}%")
    print(f"🤖 حالت: {MODE}")
    print(f"⏳ حداقل فاصله بین معاملات: {MIN_TIME} دقیقه")
    print(f"📈 حداکثر پوزیشن‌های باز: {MAX_POS}")
    print(f"{'='*70}")

def print_help():
    """چاپ راهنمای دستورات"""
    print("\n🎮 دستورات کنترل:")
    print("   Q : خروج")
    print("   P : توقف/ادامه")
    print("   S : نمایش وضعیت")
    print("   C : تغییر تنظیمات")
    print("   T : تغییر حالت معامله")
    print("   R : تغییر حالت ریسک")
    print("   D : تغییر حالت آزمایشی")
    print("   H : نمایش راهنما")
    print("="*70)

def update_config_interactive():
    """به‌روزرسانی تنظیمات به صورت تعاملی"""
    print("\n⚙️  به‌روزرسانی تنظیمات:")
    
    settings_to_update = {
        '1': ('risk_settings.RISK_PERCENT', 'درصد ریسک هر معامله', 'float'),
        '2': ('risk_settings.MIN_CONFIDENCE', 'حداقل اعتماد سیگنال (%)', 'float'),
        '3': ('trading_settings.ANALYSIS_INTERVAL_MINUTES', 'فاصله تحلیل (دقیقه)', 'int'),
        '4': ('trading_settings.ENABLE_AUTO_TRADING', 'فعال کردن معاملات خودکار', 'bool'),
        '5': ('trading_settings.ENABLE_DRY_RUN', 'فعال کردن حالت آزمایشی', 'bool'),
        '6': ('trading_settings.SYMBOL', 'نماد معاملاتی', 'str'),
        '7': ('trading_settings.TIMEFRAME', 'تایم‌فریم', 'str'),
        '8': ('trading_rules.MIN_TIME_BETWEEN_TRADES_MINUTES', 'حداقل فاصله بین معاملات (دقیقه)', 'int'),
        '9': ('trading_rules.MAX_POSITIONS', 'حداکثر پوزیشن‌های باز', 'int'),
        '10': ('trading_rules.ALLOW_MULTIPLE_POSITIONS', 'اجازه پوزیشن‌های متعدد', 'bool'),
    }
    
    print("گزینه‌ها:")
    for key, (setting_key, description, _) in settings_to_update.items():
        current_value = config.get(setting_key, 'N/A')
        print(f"   {key}: {description} (فعلی: {current_value})")
    
    print("   0: بازگشت")
    
    choice = get_user_input("\nانتخاب کنید", "0")
    
    if choice in settings_to_update:
        setting_key, description, setting_type = settings_to_update[choice]
        current_value = config.get(setting_key, '')
        
        if setting_type == 'bool':
            new_value = not bool(current_value)
            config.update_setting(setting_key, new_value)
            status = "فعال" if new_value else "غیرفعال"
            print(f"✅ {description} {status} شد")
            
        else:
            new_value = get_user_input(f"مقدار جدید برای {description}", str(current_value))
            
            if new_value:
                try:
                    if setting_type == 'float':
                        new_value = float(new_value)
                    elif setting_type == 'int':
                        new_value = int(new_value)
                    
                    config.update_setting(setting_key, new_value)
                    print(f"✅ {description} به {new_value} تغییر یافت")
                except ValueError:
                    print("❌ مقدار وارد شده معتبر نیست")
    
    elif choice != '0':
        print("❌ انتخاب نامعتبر")
