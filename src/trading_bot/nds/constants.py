"""
ثابت‌های غیررفتاری NDS برای طلا
فقط شامل نگاشت‌ها/کلیدهای بدون اعداد قابل تنظیم
"""

# ==================== تبدیل نام سشن‌ها ====================

SESSION_MAPPING = {
    'LONDON_OPEN': 'LONDON',
    'LONDON_CORE': 'LONDON',
    'NY_OPEN': 'NEW_YORK',
    'NY_CORE': 'NEW_YORK',
    'OVERLAP_PEAK': 'OVERLAP',
    'ASIA': 'ASIA',
    'DEAD_ZONE': 'OTHER'
}

# ==================== کلیدهای مجاز برای تنظیمات تحلیل ====================

ANALYSIS_CONFIG_KEYS = frozenset({
    'ATR_WINDOW',
    'ADX_WINDOW',
    'ADX_THRESHOLD_WEAK',
    'ADX_THRESHOLD_STRONG',
    'SWING_PERIOD',
    'ATR_SL_MULTIPLIER',
    'MIN_RVOL_SCALPING',
    'SWING_MIN_CONFIDENCE',
    'MIN_STRUCTURE_SCORE',
    'SCALPING_MAX_BARS_BACK',
    'SCALPING_MAX_DISTANCE_ATR',
    'SCALPING_MIN_FVG_SIZE_ATR',
    'DAILY_ATR_FACTOR',
    'ATR_MULTIPLIER',
    'RVOL_THRESHOLD',
    'FVG_VOLUME_CONFIRMATION',
    'MIN_ATR_DISTANCE_MULTIPLIER',
    'FVG_MIN_SIZE_MULTIPLIER',
    'OB_MIN_SIZE_MULTIPLIER',
    'MIN_SWEEP_PENETRATION_MULTIPLIER',
    'MIN_CANDLE_RANGE_MULTIPLIER',
    'SCALPING_MIN_CONFIDENCE',
    'MIN_CONFIDENCE',
    'FIXED_BUFFER',
    'RANGE_TOLERANCE',
    'ENTRY_FACTOR',
    'MIN_SESSION_WEIGHT',
    'DEBUG_ANALYZER',
})
