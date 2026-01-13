# src/trading_bot/nds/indicators.py
"""
محاسبه اندیکاتورهای تکنیکال پیشرفته برای طلا
"""
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator
from ta.volume import VolumeWeightedAveragePrice
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    """کلاس محاسبه تمام اندیکاتورهای مورد نیاز NDS"""
    
    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 1000.0) -> float:
        """تقسیم ایمن با جلوگیری از تقسیم بر صفر"""
        if denominator != 0:
            return numerator / denominator
        else:
            return default
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> Tuple[pd.DataFrame, float]:
        """محاسبه ATR با چندین روش پشتیبان"""
        df = df.copy()
        atr_calculated = False
        methods_tried = []
        atr_value = 0.0
        
        # نام ستون منحصر به فرد بر اساس دوره
        atr_column = f'atr_{period}'
        
        try:
            # روش 1: کتابخانه TA
            atr_indicator = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=period,
                fillna=False
            )
            df[atr_column] = atr_indicator.average_true_range()
            atr_value = float(df[atr_column].iloc[-1])
            
            # ایجاد ستون 'ATR' برای سازگاری با کدهای قدیمی
            df['ATR'] = df[atr_column]
            
            if atr_value > 0 and not np.isnan(atr_value):
                atr_calculated = True
                methods_tried.append("TA Library")
        except Exception as e:
            logger.warning(f"TA Library ATR failed: {e}")
        
        if not atr_calculated:
            try:
                # روش 2: محاسبه دستی RMA (Wilder)
                hl = df['high'] - df['low']
                hc = abs(df['high'] - df['close'].shift(1))
                lc = abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                df[atr_column] = tr.ewm(alpha=1/period, adjust=False).mean()
                atr_value = float(df[atr_column].iloc[-1])
                
                # ایجاد ستون 'ATR' برای سازگاری با کدهای قدیمی
                df['ATR'] = df[atr_column]
                
                if atr_value > 0:
                    atr_calculated = True
                    methods_tried.append("Manual RMA")
            except Exception as e:
                logger.warning(f"Manual RMA ATR failed: {e}")
        
        # محاسبه ATR درصدی
        if 'close' in df.columns and atr_value > 0:
            df[f'atr_pct_{period}'] = df[atr_column] / df['close'] * 100
            df['ATR_PCT'] = df[f'atr_pct_{period}']  # ستون سازگاری
        
        # باندهای ATR
        df[f'atr_upper_{period}'] = df['close'] + (atr_value * 2)
        df[f'atr_lower_{period}'] = df['close'] - (atr_value * 2)
        df['ATR_UPPER'] = df[f'atr_upper_{period}']
        df['ATR_LOWER'] = df[f'atr_lower_{period}']
        
        logger.info(f"ATR Calculated: ${atr_value:.2f} | Methods: {', '.join(methods_tried)}")
        return df, max(0.5, atr_value)
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.DataFrame, float, float, float, str]:
        """محاسبه ADX و خطوط DI"""
        df = df.copy()
        
        try:
            # پاکسازی داده‌ها
            high = df['high'].replace([np.inf, -np.inf], np.nan).ffill()
            low = df['low'].replace([np.inf, -np.inf], np.nan).ffill()
            close = df['close'].replace([np.inf, -np.inf], np.nan).ffill()
            
            adx_indicator = ADXIndicator(
                high=high,
                low=low,
                close=close,
                window=period,
                fillna=True
            )
            
            df['adx'] = adx_indicator.adx()
            df['plus_di'] = adx_indicator.adx_pos()
            df['minus_di'] = adx_indicator.adx_neg()
            df['di_diff'] = df['plus_di'] - df['minus_di']
            
            # تشخیص قدرت روند
            df['trend_strength'] = pd.cut(
                df['adx'],
                bins=[0, 20, 30, 50, 100],
                labels=['Very Weak', 'Weak', 'Strong', 'Very Strong']
            )
            
            adx_value = float(df['adx'].iloc[-1])
            plus_di = float(df['plus_di'].iloc[-1])
            minus_di = float(df['minus_di'].iloc[-1])
            
            # تشخیص جهت روند بر اساس DI
            if plus_di > minus_di:
                di_trend = "BULLISH"
            elif minus_di > plus_di:
                di_trend = "BEARISH"
            else:
                di_trend = "NEUTRAL"
            
            logger.info(f"ADX Analysis: ADX={adx_value:.1f} | +DI={plus_di:.1f} | -DI={minus_di:.1f} | Trend={di_trend}")
            return df, adx_value, plus_di, minus_di, di_trend
            
        except Exception as e:
            logger.error(f"ADX Calculation Error: {e}")
            df['adx'] = 25.0
            df['plus_di'] = 20.0
            df['minus_di'] = 18.0
            df['trend_strength'] = 'Weak'
            return df, 25.0, 20.0, 18.0, "NEUTRAL"
    
    @staticmethod
    def calculate_daily_range(df: pd.DataFrame, timeframe: str, atr_value: float) -> Tuple[float, float, float, float]:
        """محاسبه رنج روزانه و موقعیت قیمت"""
        try:
            if len(df) < 96 * 2:  # حداقل 2 روز داده M15
                factor = 4.8 if timeframe == 'M15' else 2.4
                daily_range = atr_value * factor
                daily_high = float(df['high'].max())
                daily_low = float(df['low'].min())
                daily_mid = (daily_high + daily_low) / 2
                
                logger.info(f"Daily Range (estimated): ${daily_range:.2f} (factor: {factor:.1f}x)")
                return daily_range, daily_high, daily_low, daily_mid
            
            # ساخت کندل‌های روزانه واقعی
            df_daily = df.copy()
            df_daily['date'] = df_daily['time'].dt.date
            
            daily_stats = df_daily.groupby('date').agg({
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'open': 'first'
            }).reset_index()
            
            daily_stats['daily_range'] = daily_stats['high'] - daily_stats['low']
            
            # میانگین 20 روز اخیر
            lookback = min(20, len(daily_stats))
            daily_range = float(daily_stats['daily_range'].tail(lookback).mean())
            daily_high = float(daily_stats['high'].iloc[-1])
            daily_low = float(daily_stats['low'].iloc[-1])
            daily_mid = (daily_high + daily_low) / 2
            
            logger.info(f"Daily Range: ${daily_range:.2f} | High: ${daily_high:.2f} | Low: ${daily_low:.2f}")
            return daily_range, daily_high, daily_low, daily_mid
            
        except Exception as e:
            logger.error(f"Daily Range Calculation Error: {e}")
            factor = 4.8 if timeframe == 'M15' else 2.4
            daily_range = atr_value * factor
            daily_high = float(df['high'].max())
            daily_low = float(df['low'].min())
            daily_mid = (daily_high + daily_low) / 2
            return daily_range, daily_high, daily_low, daily_mid
    
    @staticmethod
    def analyze_volume(df: pd.DataFrame, ma_window: int = 20) -> Dict:
        """تحلیل حجم پیشرفته"""
        if 'volume' not in df.columns:
            return {
                'rvol': 1.0,
                'volume_zone': 'NORMAL',
                'volume_trend': 'NEUTRAL',
                'vwap': float(df['close'].iloc[-1]),
                'vwap_distance_pct': 0.0,
                'volume_cluster_center': float(df['close'].iloc[-1])
            }
        
        try:
            volume = df['volume'].replace(0, np.nan).ffill().fillna(1)
            
            # محاسبه RVOL
            volume_ma = volume.rolling(window=ma_window).mean()
            df['rvol'] = volume / volume_ma
            
            # تشخیص Volume Zones
            def classify_volume(rvol):
                if rvol > 2.0:
                    return 'VERY_HIGH'
                elif rvol > 1.5:
                    return 'HIGH'
                elif rvol < 0.5:
                    return 'VERY_LOW'
                elif rvol < 0.8:
                    return 'LOW'
                else:
                    return 'NORMAL'
            
            df['volume_zone'] = df['rvol'].apply(classify_volume)
            
            # Volume Trend
            volume_ma_short = volume.rolling(window=5).mean()
            volume_ma_long = volume.rolling(window=20).mean()
            df['volume_trend'] = np.where(
                volume_ma_short > volume_ma_long, 'INCREASING',
                np.where(volume_ma_short < volume_ma_long, 'DECREASING', 'STABLE')
            )
            
            # محاسبه VWAP
            try:
                vwap_ind = VolumeWeightedAveragePrice(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    volume=volume,
                    window=ma_window
                )
                df['vwap'] = vwap_ind.volume_weighted_average_price()
                df['vwap_distance_pct'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
            except:
                df['vwap'] = df['close']
                df['vwap_distance_pct'] = 0.0
            
            # Volume Clusters
            price_bins = pd.cut(df['close'], bins=20)
            volume_by_price = df.groupby(price_bins, observed=False)['volume'].sum()
            volume_cluster_center = volume_by_price.idxmax().mid if not volume_by_price.empty else float(df['close'].iloc[-1])
            
            current_data = {
                'rvol': float(df['rvol'].iloc[-1]),
                'volume_zone': df['volume_zone'].iloc[-1],
                'volume_trend': df['volume_trend'].iloc[-1],
                'vwap': float(df['vwap'].iloc[-1]),
                'vwap_distance_pct': float(df['vwap_distance_pct'].iloc[-1]),
                'volume_cluster_center': volume_cluster_center
            }
            
            logger.info(f"Volume Analysis: RVOL={current_data['rvol']:.2f}x | Zone={current_data['volume_zone']}")
            return current_data
            
        except Exception as e:
            logger.error(f"Volume Analysis Error: {e}")
            return {
                'rvol': 1.0,
                'volume_zone': 'NORMAL',
                'volume_trend': 'NEUTRAL',
                'vwap': float(df['close'].iloc[-1]),
                'vwap_distance_pct': 0.0,
                'volume_cluster_center': float(df['close'].iloc[-1])
            }
    
    @staticmethod
    def determine_volatility_state(df: pd.DataFrame, atr_value: float, daily_range: float, timeframe: str) -> Tuple[float, str]:
        """تشخیص حالت نوسان بازار"""
        try:
            if daily_range > 0:
                if timeframe == 'M15':
                    expected_atr = daily_range / 20  # تقریب برای M15
                else:
                    expected_atr = daily_range / 24
                
                volatility_ratio = IndicatorCalculator._safe_divide(atr_value, expected_atr, 1.0)
                
                if volatility_ratio > 1.5:
                    volatility_state = 'HIGH_VOLATILITY'
                elif volatility_ratio < 0.5:
                    volatility_state = 'LOW_VOLATILITY'
                else:
                    volatility_state = 'NORMAL_VOLATILITY'
            else:
                volatility_ratio = 1.0
                volatility_state = 'NORMAL_VOLATILITY'
            
            logger.info(f"Volatility State: {volatility_state} (Ratio: {volatility_ratio:.2f})")
            return volatility_ratio, volatility_state
            
        except Exception as e:
            logger.error(f"Volatility State Error: {e}")
            return 1.0, 'NORMAL_VOLATILITY'