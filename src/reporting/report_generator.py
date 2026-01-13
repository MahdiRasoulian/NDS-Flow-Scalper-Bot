"""
ماژول گزارش‌گیری حرفه‌ای برای ربات معاملاتی NDS
طراحی شده برای تریدرهای حرفه‌ای فارکس با تمرکز بر تحلیل عملکرد
نسخه: 2.0
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import os
from pathlib import Path
import json
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

# تنظیمات ظاهری نمودار برای نمایش حرفه‌ای‌تر
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [20, 12]
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

@dataclass
class TradeMetrics:
    """کلاس برای ذخیره متریک‌های معاملاتی"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    recovery_factor: float = 0.0

class ReportGenerator:
    """
    کلاس اصلی برای تولید گزارش‌های حرفه‌ای از ربات معاملاتی NDS
    """
    
    def __init__(self, 
                 output_dir: str = "trade_reports",
                 logger: Optional[logging.Logger] = None):
        """
        مقداردهی اولیه گزارش‌گیر
        
        Args:
            output_dir: پوشه خروجی برای گزارش‌ها
            logger: آبجکت لاگر (اختیاری)
        """
        self.output_dir = Path(output_dir)
        self._logger = logger or logging.getLogger(__name__)
        self._setup_directories()
        
        # کش برای داده‌های گزارش
        self._report_cache = {}
        
        # تنظیمات پیش‌فرض
        self._config = {
            'candle_width': 0.8,
            'volume_alpha': 0.7,
            'support_resistance_width': 1.5,
            'entry_marker_size': 120,
            'sl_tp_linewidth': 2,
            'chart_quality': 'high'  # high, medium, low
        }
        
        self._logger.info(f"✅ ReportGenerator initialized. Output directory: {self.output_dir}")
    
    def _setup_directories(self):
        """ایجاد پوشه‌های مورد نیاز برای گزارش‌ها"""
        directories = [
            self.output_dir / 'excel',
            self.output_dir / 'charts',
            self.output_dir / 'summaries',
            self.output_dir / 'daily',
            self.output_dir / 'weekly',
            self.output_dir / 'monthly',
            self.output_dir / 'performance'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self._logger.debug(f"Directory created/verified: {directory}")
    
    def _get_timestamp(self) -> str:
        """دریافت timestamp فعلی برای نام فایل‌ها"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _format_currency(self, value: float) -> str:
        """فرمت‌دهی مبالغ مالی"""
        if abs(value) >= 1000:
            return f"${value:,.0f}"
        return f"${value:.2f}"
    
    def _calculate_metrics(self, trades_data: List[Dict]) -> TradeMetrics:
        """محاسبه متریک‌های معاملاتی از داده‌های معاملات"""
        metrics = TradeMetrics()
        
        if not trades_data:
            return metrics
        
        # تبدیل به DataFrame برای محاسبات آسان‌تر
        trades_df = pd.DataFrame(trades_data)
        
        if 'profit' not in trades_df.columns:
            return metrics
        
        # محاسبات پایه
        metrics.total_trades = len(trades_df)
        metrics.winning_trades = len(trades_df[trades_df['profit'] > 0])
        metrics.losing_trades = len(trades_df[trades_df['profit'] < 0])
        
        if metrics.total_trades > 0:
            metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100
        
        # محاسبه سود و ضرر
        metrics.total_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
        metrics.total_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
        metrics.net_profit = trades_df['profit'].sum()
        
        # محاسبه Profit Factor
        if metrics.total_loss > 0:
            metrics.profit_factor = metrics.total_profit / metrics.total_loss
        
        # میانگین سود و ضرر
        if metrics.winning_trades > 0:
            metrics.avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean()
        
        if metrics.losing_trades > 0:
            metrics.avg_loss = trades_df[trades_df['profit'] < 0]['profit'].mean()
        
        # محاسبه نسبت ریسک به ریوارد (اگر داده‌ها موجود باشد)
        if 'risk_reward' in trades_df.columns:
            metrics.avg_rr = trades_df['risk_reward'].mean()
        
        # محاسبه حداکثر Drawdown
        if 'balance' in trades_df.columns or 'equity' in trades_df.columns:
            # شبیه‌سازی drawdown (در نسخه واقعی باید از داده‌های تاریخی استفاده شود)
            pass
        
        # محاسبه حداکثر برد/باخت متوالی
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        
        for profit in trades_df['profit']:
            if profit > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                consecutive_wins = max(consecutive_wins, current_streak)
                consecutive_losses = max(consecutive_losses, abs(current_streak) if current_streak < 0 else 0)
            elif profit < 0:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                consecutive_losses = max(consecutive_losses, abs(current_streak))
                consecutive_wins = max(consecutive_wins, current_streak if current_streak > 0 else 0)
        
        metrics.max_consecutive_wins = consecutive_wins
        metrics.max_consecutive_losses = consecutive_losses
        
        return metrics
    
    def save_excel_report(self, 
                          df: pd.DataFrame, 
                          signal_data: Dict, 
                          order_details: Optional[Dict] = None,
                          trades_history: Optional[List[Dict]] = None,
                          filename: Optional[str] = None) -> str:
        """
        ذخیره گزارش کامل Excel از تحلیل NDS و معامله
        
        Args:
            df: دیتافریم قیمت
            signal_data: نتایج تحلیل از NDSAnalyzer
            order_details: اطلاعات معامله (اختیاری)
            trades_history: تاریخچه معاملات (اختیاری)
            filename: نام فایل خروجی (اختیاری)
            
        Returns:
            str: مسیر فایل ذخیره شده
        """
        try:
            if filename is None:
                timestamp = self._get_timestamp()
                symbol = signal_data.get('symbol', 'XAUUSD')
                filename = self.output_dir / 'excel' / f"{symbol}_report_{timestamp}.xlsx"
            else:
                filename = Path(filename)
            
            # اطمینان از وجود پوشه
            filename.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # ===== 1. داده‌های خام =====
                df_raw = df.copy()
                if 'time' in df_raw.columns:
                    df_raw['time'] = pd.to_datetime(df_raw['time'])
                
                df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                # ===== 2. خلاصه اجرا =====
                summary_data = {
                    'گزارش تولید شده در': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'نماد': signal_data.get('symbol', 'XAUUSD'),
                    'تایم‌فریم': signal_data.get('timeframe', 'M15'),
                    'سیگنال': signal_data.get('signal', 'NEUTRAL'),
                    'اعتماد سیگنال': f"{signal_data.get('confidence', 0)}%",
                    'امتیاز سیگنال': signal_data.get('score', 0),
                    'وضعیت بازار': signal_data.get('market_state', 'NORMAL_MARKET'),
                    'منطقه PD': signal_data.get('pd_zone', 'N/A'),
                    'روند': signal_data.get('structure', {}).get('trend', 'N/A'),
                    'ATR فعلی': self._format_currency(signal_data.get('atr', 0)),
                    'ATR روزانه': self._format_currency(signal_data.get('daily_atr', 0)),
                    'نسبت ATR': f"{(signal_data.get('atr', 0) / signal_data.get('daily_atr', 1) if signal_data.get('daily_atr', 0) > 0 else 0):.2f}",
                }
                
                # اضافه کردن اطلاعات معامله اگر موجود باشد
                if order_details:
                    summary_data.update({
                        'ورود پیشنهادی': self._format_currency(order_details.get('entry', 0)),
                        'حد ضرر': self._format_currency(order_details.get('sl', 0)),
                        'حد سود': self._format_currency(order_details.get('tp', 0)),
                        'فاصله SL': self._format_currency(order_details.get('sl_distance', 0)),
                        'نسبت R:R': f"{order_details.get('rr_ratio', 0):.2f}",
                        'حجم معامله': f"{order_details.get('lot', 0):.3f} لات",
                        'مقدار ریسک': self._format_currency(order_details.get('lot_calculation', {}).get('actual_risk_amount', 0)),
                        'درصد ریسک': f"{order_details.get('lot_calculation', {}).get('actual_risk_percent', 0):.2f}%",
                    })
                
                # دلایل سیگنال
                reasons = signal_data.get('reasons', [])
                if reasons:
                    summary_data['دلایل سیگنال'] = ' | '.join(reasons[:3])  # فقط ۳ دلیل اول
                
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # ===== 3. تحلیل بازار =====
                analysis_data = []
                
                # تحلیل ساختار
                structure = signal_data.get('structure', {})
                
                # توابع کمکی برای استخراج امن قیمت و زمان
                def safe_get_price(data):
                    if isinstance(data, dict): return data.get('price', 0)
                    if isinstance(data, (int, float)): return data
                    return 0

                def safe_get_time(data):
                    if isinstance(data, dict): return data.get('time', 'N/A')
                    return 'N/A'

                analysis_data.append({
                    'بخش': 'ساختار بازار',
                    'پارامتر': 'روند',
                    'مقدار': structure.get('trend', 'N/A'),
                    'توضیحات': 'روند کلی بازار'
                })
                
                analysis_data.append({
                    'بخش': 'ساختار بازار',
                    'پارامتر': 'BOS',
                    'مقدار': structure.get('bos', 'N/A'),
                    'توضیحات': 'Break of Structure'
                })
                
                analysis_data.append({
                    'بخش': 'ساختار بازار',
                    'پارامتر': 'CHoCH',
                    'مقدار': structure.get('choch', 'N/A'),
                    'توضیحات': 'Change of Character'
                })
                
                # اصلاح بخش سقف اخیر
                lh_data = structure.get('last_high', 0)
                analysis_data.append({
                    'بخش': 'ساختار بازار',
                    'پارامتر': 'سقف اخیر',
                    'مقدار': safe_get_price(lh_data),
                    'توضیحات': f"زمان: {safe_get_time(lh_data)}"
                })
                
                # اصلاح بخش کف اخیر
                ll_data = structure.get('last_low', 0)
                analysis_data.append({
                    'بخش': 'ساختار بازار',
                    'پارامتر': 'کف اخیر',
                    'مقدار': safe_get_price(ll_data),
                    'توضیحات': f"زمان: {safe_get_time(ll_data)}"
                })
                
                # تحلیل نوسان
                analysis_data.append({
                    'بخش': 'نوسان',
                    'پارامتر': 'ATR فعلی',
                    'مقدار': signal_data.get('atr', 0),
                    'توضیحات': 'میانگین محدوده واقعی'
                })
                
                analysis_data.append({
                    'بخش': 'نوسان',
                    'پارامتر': 'ATR روزانه',
                    'مقدار': signal_data.get('daily_atr', 0),
                    'توضیحات': 'ATR روزانه نماد'
                })
                
                analysis_data.append({
                    'بخش': 'نوسان',
                    'پارامتر': 'وضعیت نوسان',
                    'مقدار': signal_data.get('volatility_state', 'NORMAL'),
                    'توضیحات': 'سطح نوسان بازار'
                })
                
                # تحلیل حجم
                analysis_data.append({
                    'بخش': 'حجم',
                    'پارامتر': 'RVOL',
                    'مقدار': signal_data.get('rvol', 0),
                    'توضیحات': 'حجم نسبی'
                })
                
                analysis_data.append({
                    'بخش': 'حجم',
                    'پارامتر': 'منطقه حجم',
                    'مقدار': signal_data.get('volume_zone', 'NORMAL'),
                    'توضیحات': 'منطقه حجمی بازار'
                })
                
                # تحلیل جلسه معاملاتی
                analysis_data.append({
                    'بخش': 'جلسه معاملاتی',
                    'پارامتر': 'جلسه فعال',
                    'مقدار': signal_data.get('session', {}).get('name', 'N/A'),
                    'توضیحات': f"وزن: {signal_data.get('session', {}).get('weight', 0)}"
                })
                
                # تحلیل قیمت
                if len(df) > 0:
                    current_price = df['close'].iloc[-1]
                    analysis_data.append({
                        'بخش': 'قیمت',
                        'پارامتر': 'قیمت فعلی',
                        'مقدار': current_price,
                        'توضیحات': 'آخرین قیمت بسته شدن'
                    })
                    
                    if 'high' in df.columns and 'low' in df.columns:
                        daily_high = df['high'].max()
                        daily_low = df['low'].min()
                        analysis_data.append({
                            'بخش': 'قیمت',
                            'پارامتر': 'بازه روزانه',
                            'مقدار': f"{daily_high} - {daily_low}",
                            'توضیحات': f"عرض: {daily_high - daily_low:.2f}"
                        })
                
                analysis_df = pd.DataFrame(analysis_data)
                analysis_df.to_excel(writer, sheet_name='Market_Analysis', index=False)
                
                # ===== 4. سوینگ‌ها =====
                if 'swings' in signal_data and signal_data['swings']:
                    swings_df = pd.DataFrame(signal_data['swings'])
                    if not swings_df.empty:
                        swings_df['time'] = pd.to_datetime(swings_df['time'])
                        swings_df['price_diff'] = swings_df['price'].diff()
                        swings_df['price_diff_pct'] = (swings_df['price'].diff() / swings_df['price'].shift(1)) * 100
                        swings_df.to_excel(writer, sheet_name='Swings', index=False)
                
                # ===== 5. FVGها =====
                if 'fvgs' in signal_data and signal_data['fvgs']:
                    fvgs_df = pd.DataFrame(signal_data['fvgs'])
                    if not fvgs_df.empty:
                        fvgs_df['time'] = pd.to_datetime(fvgs_df['time'])
                        fvgs_df['size'] = fvgs_df['top'] - fvgs_df['bottom']
                        fvgs_df['size_pct'] = (fvgs_df['size'] / fvgs_df['mid']) * 100
                        fvgs_df['age_bars'] = fvgs_df.get('age', 0)
                        fvgs_df.to_excel(writer, sheet_name='FVG', index=False)
                
                # ===== 6. Liquidity Sweeps =====
                if 'sweeps' in signal_data and signal_data['sweeps']:
                    sweeps_df = pd.DataFrame(signal_data['sweeps'])
                    if not sweeps_df.empty:
                        sweeps_df['time'] = pd.to_datetime(sweeps_df['time'])
                        sweeps_df.to_excel(writer, sheet_name='Sweeps', index=False)
                
                # ===== 7. Order Blocks =====
                if 'order_blocks' in signal_data and signal_data['order_blocks']:
                    blocks_df = pd.DataFrame(signal_data['order_blocks'])
                    if not blocks_df.empty:
                        blocks_df['time'] = pd.to_datetime(blocks_df['time'])
                        blocks_df['size'] = blocks_df['high'] - blocks_df['low']
                        blocks_df.to_excel(writer, sheet_name='Order_Blocks', index=False)
                
                # ===== 8. اطلاعات معامله =====
                if order_details:
                    order_df = pd.DataFrame([order_details])
                    order_df.to_excel(writer, sheet_name='Order_Details', index=False)
                
                # ===== 9. تاریخچه معاملات =====
                if trades_history:
                    trades_df = pd.DataFrame(trades_history)
                    if not trades_df.empty:
                        trades_df['time'] = pd.to_datetime(trades_df['time'])
                        trades_df.to_excel(writer, sheet_name='Trades_History', index=False)
                        
                        # محاسبه و ذخیره متریک‌ها
                        metrics = self._calculate_metrics(trades_history)
                        metrics_df = pd.DataFrame([asdict(metrics)])
                        metrics_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
                
                # ===== 10. اطلاعات حساب =====
                account_info = signal_data.get('account_info', {})
                if account_info:
                    account_df = pd.DataFrame([account_info])
                    account_df.to_excel(writer, sheet_name='Account_Info', index=False)
                
                # ===== 11. تنظیمات ربات =====
                bot_config = signal_data.get('bot_config', {})
                if bot_config:
                    config_df = pd.DataFrame([bot_config])
                    config_df.to_excel(writer, sheet_name='Bot_Config', index=False)
                
                # ===== 12. لاگ خطاها =====
                error_log = signal_data.get('error_log', [])
                if error_log:
                    error_df = pd.DataFrame(error_log)
                    error_df.to_excel(writer, sheet_name='Error_Log', index=False)
            
            self._logger.info(f"✅ Excel report saved to {filename}")
            return str(filename)
            
        except Exception as e:
            self._logger.error(f"❌ Error saving Excel report: {e}", exc_info=True)
            raise
    
    def plot_chart(self, 
                   df: pd.DataFrame, 
                   signal_data: Dict, 
                   order_details: Optional[Dict] = None,
                   filename: Optional[str] = None) -> str:
        """
        ایجاد نمودار حرفه‌ای از تحلیل NDS
        
        Args:
            df: دیتافریم قیمت
            signal_data: نتایج تحلیل
            order_details: اطلاعات معامله (اختیاری)
            filename: نام فایل خروجی (اختیاری)
            
        Returns:
            str: مسیر فایل ذخیره شده
        """
        try:
            if filename is None:
                timestamp = self._get_timestamp()
                symbol = signal_data.get('symbol', 'XAUUSD')
                filename = self.output_dir / 'charts' / f"{symbol}_chart_{timestamp}.png"
            else:
                filename = Path(filename)
            
            # اطمینان از وجود پوشه
            filename.parent.mkdir(parents=True, exist_ok=True)
            
            # تنظیم کیفیت نمودار بر اساس تنظیمات
            if self._config['chart_quality'] == 'high':
                plt.rcParams['figure.dpi'] = 150
                figsize = (22, 14)
            elif self._config['chart_quality'] == 'medium':
                plt.rcParams['figure.dpi'] = 120
                figsize = (20, 12)
            else:
                plt.rcParams['figure.dpi'] = 100
                figsize = (18, 10)
            
            # ایجاد نمودار با subplot‌های متعدد
            fig = plt.figure(figsize=figsize)
            
            # تعریف grid برای نمودارها
            gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
            
            # نمودار اصلی (قیمت)
            ax_price = fig.add_subplot(gs[0:4, :])
            
            # نمودار حجم
            ax_volume = fig.add_subplot(gs[4, :], sharex=ax_price)
            
            # نمودار ATR
            ax_atr = fig.add_subplot(gs[5, :], sharex=ax_price)
            
            # تبدیل زمان به datetime برای plotting
            if 'time' in df.columns:
                df['time_dt'] = pd.to_datetime(df['time'])
            else:
                # اگر ستون time وجود ندارد، از ایندکس استفاده کن
                df['time_dt'] = pd.date_range(end=datetime.now(), periods=len(df), freq='15min')
            
            # --- نمودار اصلی (قیمت) ---
            
            # رسم کندل‌استیک (ساده‌شده)
            for idx in range(len(df)):
                row = df.iloc[idx]
                
                # تعیین رنگ کندل
                if row['close'] >= row['open']:
                    color = 'green'
                    body_color = 'limegreen'
                else:
                    color = 'red'
                    body_color = 'lightcoral'
                
                # رسم بدنه کندل
                ax_price.plot([row['time_dt'], row['time_dt']], 
                             [row['open'], row['close']], 
                             color=color, linewidth=6, solid_capstyle='round')
                
                # رسم سایه کندل
                ax_price.plot([row['time_dt'], row['time_dt']], 
                             [row['low'], row['high']], 
                             color=color, linewidth=1)
            
            # رسم سوینگ‌ها با رنگ‌های مختلف
            if 'swings' in signal_data and signal_data['swings']:
                swings_df = pd.DataFrame(signal_data['swings'])
                if not swings_df.empty:
                    swings_df['time_dt'] = pd.to_datetime(swings_df['time'])
                    
                    # سوینگ Highها
                    high_swings = swings_df[swings_df['side'] == 'HIGH']
                    if not high_swings.empty:
                        colors_high = []
                        for swing_type in high_swings['type']:
                            if swing_type == 'HH':
                                colors_high.append('darkred')
                            elif swing_type == 'LH':
                                colors_high.append('red')
                            else:
                                colors_high.append('orange')
                        
                        ax_price.scatter(high_swings['time_dt'], high_swings['price'], 
                                       c=colors_high, marker='v', s=80, 
                                       label='Swing Highs', zorder=5, edgecolors='black', linewidth=1)
                    
                    # سوینگ Lowها
                    low_swings = swings_df[swings_df['side'] == 'LOW']
                    if not low_swings.empty:
                        colors_low = []
                        for swing_type in low_swings['type']:
                            if swing_type == 'LL':
                                colors_low.append('darkgreen')
                            elif swing_type == 'HL':
                                colors_low.append('green')
                            else:
                                colors_low.append('lime')
                        
                        ax_price.scatter(low_swings['time_dt'], low_swings['price'], 
                                       c=colors_low, marker='^', s=80, 
                                       label='Swing Lows', zorder=5, edgecolors='black', linewidth=1)
            
            # رسم FVGها
            if 'fvgs' in signal_data and signal_data['fvgs']:
                fvgs_df = pd.DataFrame(signal_data['fvgs'])
                if not fvgs_df.empty:
                    fvgs_df['time_dt'] = pd.to_datetime(fvgs_df['time'])
                    
                    for _, fvg in fvgs_df.iterrows():
                        if not fvg.get('filled', False):
                            color = 'limegreen' if fvg.get('type') == 'BULLISH_FVG' else 'lightcoral'
                            alpha = 0.25
                            
                            # رسم مستطیل FVG
                            rect = Rectangle(
                                (mdates.date2num(fvg['time_dt'] - timedelta(hours=2)), fvg['bottom']),
                                width=mdates.date2num(timedelta(hours=4)),
                                height=fvg['top'] - fvg['bottom'],
                                facecolor=color,
                                alpha=alpha,
                                edgecolor=color,
                                linewidth=1,
                                linestyle='--'
                            )
                            ax_price.add_patch(rect)
                            
                            # خط میانی FVG
                            ax_price.axhline(y=fvg['mid'], color=color, alpha=0.5, 
                                           linestyle='--', linewidth=0.8, zorder=1)
            
            # رسم Liquidity Sweeps - نسخه اصلاح شده
            if 'sweeps' in signal_data and signal_data['sweeps']:
                sweeps_df = pd.DataFrame(signal_data['sweeps'])
                if not sweeps_df.empty:
                    sweeps_df['time_dt'] = pd.to_datetime(sweeps_df['time'])
                    
                    for idx, sweep in sweeps_df.iterrows():
                        color = 'darkred' if sweep['type'] == 'BEARISH_SWEEP' else 'darkgreen'
                        marker = 'x' if sweep['type'] == 'BEARISH_SWEEP' else '+'
                        size = 120
                        
                        # فقط برای اولین نقطه برچسب می‌خورد
                        label = f"{sweep['type']}" if idx == 0 else None
                        ax_price.scatter(sweep['time_dt'], sweep['level'], 
                                       c=color, marker=marker, s=size, 
                                       label=label, 
                                       zorder=6, linewidths=2, edgecolors='black')
            
            # رسم خطوط حمایت و مقاومت
            if 'support_levels' in signal_data:
                for level in signal_data.get('support_levels', [])[:3]:  # فقط ۳ سطح اول
                    ax_price.axhline(y=level, color='green', alpha=0.4, 
                                    linestyle='--', linewidth=1.5, label='Support' if level == signal_data['support_levels'][0] else "")
            
            if 'resistance_levels' in signal_data:
                for level in signal_data.get('resistance_levels', [])[:3]:  # فقط ۳ سطح اول
                    ax_price.axhline(y=level, color='red', alpha=0.4, 
                                    linestyle='--', linewidth=1.5, label='Resistance' if level == signal_data['resistance_levels'][0] else "")
            
            # رسم خطوط ورود/خروج اگر معامله‌ای باشد
            current_price = df['close'].iloc[-1]
            ax_price.axhline(y=current_price, color='blue', alpha=0.3, 
                           linestyle='-', linewidth=1, label='Current Price')
            
            if order_details:
                # خطوط ورود، استاپ و تارگت
                entry_price = order_details.get('entry', 0)
                sl_price = order_details.get('sl', 0)
                tp_price = order_details.get('tp', 0)
                
                ax_price.axhline(y=entry_price, color='green', 
                               alpha=0.7, linestyle='-', linewidth=2, label='Entry')
                ax_price.axhline(y=sl_price, color='red', 
                               alpha=0.7, linestyle='-', linewidth=2, label='Stop Loss')
                ax_price.axhline(y=tp_price, color='blue', 
                               alpha=0.7, linestyle='--', linewidth=2, label='Take Profit')
                
                # علامت‌گذاری نقاط
                ax_price.scatter([df['time_dt'].iloc[-1]], [entry_price], 
                               color='green', marker='o', s=100, zorder=7, edgecolors='black')
                ax_price.scatter([df['time_dt'].iloc[-1]], [tp_price], 
                               color='blue', marker='*', s=150, zorder=7, edgecolors='black')
                
                # پر کردن منطقه ریسک/پاداش
                ax_price.fill_between([df['time_dt'].min(), df['time_dt'].max()], 
                                     sl_price, entry_price, 
                                     color='red', alpha=0.1, label='Risk Zone')
                ax_price.fill_between([df['time_dt'].min(), df['time_dt'].max()], 
                                     entry_price, tp_price, 
                                     color='green', alpha=0.1, label='Reward Zone')
            
            # تنظیمات نمودار اصلی
            symbol = signal_data.get('symbol', 'XAUUSD')
            timeframe = signal_data.get('timeframe', 'M15')
            signal = signal_data.get('signal', 'NEUTRAL')
            confidence = signal_data.get('confidence', 0)
            
            ax_price.set_title(f"{symbol} - {timeframe} | NDS Analysis: {signal} (Confidence: {confidence}%)", 
                             fontsize=16, fontweight='bold', pad=20)
            ax_price.set_ylabel('Price', fontsize=12)
            ax_price.legend(loc='upper left', fontsize=9, ncol=2)
            ax_price.grid(True, alpha=0.3)
            
            # فرمت تاریخ در محور X
            ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45)
            
            # --- نمودار حجم ---
            if 'volume' in df.columns:
                # تعیین رنگ‌های حجم بر اساس صعودی/نزولی بودن کندل
                colors_volume = ['green' if close >= open_ else 'red' 
                                for close, open_ in zip(df['close'], df['open'])]
                
                ax_volume.bar(df['time_dt'], df['volume'], color=colors_volume, 
                             alpha=self._config['volume_alpha'], width=0.8)
                
                if 'vol_ma' in df.columns:
                    ax_volume.plot(df['time_dt'], df['vol_ma'], color='yellow', 
                                  linewidth=1.5, label='Volume MA')
                
                ax_volume.set_ylabel('Volume', fontsize=10)
                ax_volume.legend(loc='upper left', fontsize=8)
                ax_volume.grid(True, alpha=0.2)
            
            # --- نمودار ATR (نوسان) ---
            if 'atr' in df.columns:
                ax_atr.plot(df['time_dt'], df['atr'], color='purple', 
                           linewidth=1.5, label='ATR')
                ax_atr.axhline(y=signal_data.get('atr', 0), color='red', alpha=0.5, 
                              linestyle='--', linewidth=1, label='Current ATR')
                
                if 'daily_atr' in signal_data and signal_data['daily_atr']:
                    ax_atr.axhline(y=signal_data['daily_atr'], color='green', alpha=0.5,
                                  linestyle='--', linewidth=1, label='Daily ATR')
                
                ax_atr.set_ylabel('ATR (Volatility)', fontsize=10)
                ax_atr.set_xlabel('Time', fontsize=10)
                ax_atr.legend(loc='upper left', fontsize=8)
                ax_atr.grid(True, alpha=0.2)
            
            # --- جعبه اطلاعات (Info Box) ---
            info_text = self._create_info_box_text(signal_data, order_details, df)
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
            ax_price.text(0.02, 0.98, info_text, transform=ax_price.transAxes, fontsize=9,
                         verticalalignment='top', bbox=props, fontfamily='monospace', linespacing=1.5)
            
            # اضافه کردن دلایل سیگنال
            reasons = signal_data.get('reasons', [])
            if reasons:
                reasons_text = "Reasons:\n" + "\n".join([f"• {r}" for r in reasons[:4]])
                ax_volume.text(0.02, 0.98, reasons_text, transform=ax_volume.transAxes, fontsize=8,
                              verticalalignment='top', 
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            # تنظیم فاصله‌ها
            plt.tight_layout()
            
            # ذخیره نمودار
            plt.savefig(filename, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
            plt.close(fig)
            
            self._logger.info(f"✅ Chart saved to {filename}")
            return str(filename)
            
        except Exception as e:
            self._logger.error(f"❌ Error plotting chart: {e}", exc_info=True)
            raise
    
    def _create_info_box_text(self, signal_data, order_details, df):
        """ایجاد متن جعبه اطلاعات"""
        info_text = f"""
Signal: {signal_data.get('signal', 'NEUTRAL')}
Confidence: {signal_data.get('confidence', 0)}%
Score: {signal_data.get('score', 50)}

Market Structure:
  Trend: {signal_data.get('structure', {}).get('trend', 'N/A')}
  BOS: {signal_data.get('structure', {}).get('bos', 'N/A')}
  CHoCH: {signal_data.get('structure', {}).get('choch', 'N/A')}
  PD Zone: {signal_data.get('pd_zone', 'N/A')}

Volatility:
  Current ATR: ${signal_data.get('atr', 0):.2f}
  Daily ATR: ${signal_data.get('daily_atr', 0):.2f}
  Ratio: {(signal_data.get('atr', 0) / signal_data.get('daily_atr', 1) if signal_data.get('daily_atr', 0) > 0 else 0):.2f}

Price Analysis:
  Current: ${df['close'].iloc[-1] if len(df) > 0 else 0:.2f}
  Daily High: ${df['high'].max() if len(df) > 0 else 0:.2f}
  Daily Low: ${df['low'].min() if len(df) > 0 else 0:.2f}
  Range: ${(df['high'].max() - df['low'].min()) if len(df) > 0 else 0:.2f}
        """
        
        if order_details:
            info_text += f"""
--- ORDER DETAILS ---
Side: {order_details.get('side', 'N/A')}
Entry: ${order_details.get('entry', 0):.2f}
SL: ${order_details.get('sl', 0):.2f}
TP: ${order_details.get('tp', 0):.2f}
SL Distance: ${order_details.get('sl_distance', 0):.2f}
R:R Ratio: {order_details.get('rr_ratio', 0):.2f}
Lot Size: {order_details.get('lot', 0):.3f}
Risk Amount: ${order_details.get('lot_calculation', {}).get('actual_risk_amount', 0):.2f}
Risk %: {order_details.get('lot_calculation', {}).get('actual_risk_percent', 0):.2f}%
"""
        
        # اطلاعات زمان
        info_text += f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return info_text
    
    def generate_trade_summary(self, 
                                signal_data: Dict, 
                                order_details: Optional[Dict] = None,
                                filename: Optional[str] = None) -> str:
            """
            ایجاد خلاصه متنی از معامله با پشتیبانی از فرمت‌های مختلف داده
            """
            try:
                if filename is None:
                    timestamp = self._get_timestamp()
                    symbol = signal_data.get('symbol', 'XAUUSD')
                    filename = self.output_dir / 'summaries' / f"{symbol}_summary_{timestamp}.txt"
                else:
                    filename = Path(filename)
                
                filename.parent.mkdir(parents=True, exist_ok=True)
                
                # تابع کمکی برای استخراج امن قیمت (چه عدد باشد چه دیکشنری)
                def safe_price(data):
                    if isinstance(data, (int, float)):
                        return float(data)
                    if isinstance(data, dict):
                        return float(data.get('price', 0))
                    return 0.0

                summary = []
                summary.append("=" * 70)
                summary.append(" " * 20 + "NDS TRADING BOT - TRADE SUMMARY")
                summary.append("=" * 70)
                summary.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                summary.append(f"Symbol: {signal_data.get('symbol', 'XAUUSD')}")
                summary.append(f"Timeframe: {signal_data.get('timeframe', 'M15')}")
                summary.append(f"Signal: {signal_data.get('signal', 'NEUTRAL')}")
                summary.append(f"Confidence: {signal_data.get('confidence', 0)}%")
                summary.append(f"Score: {signal_data.get('score', 50)}")
                summary.append("")
                
                # اطلاعات ساختار - اصلاح شده برای جلوگیری از خطا
                structure = signal_data.get('structure', {})
                summary.append("MARKET STRUCTURE:")
                summary.append(f"  Trend: {structure.get('trend', 'N/A')}")
                summary.append(f"  BOS: {structure.get('bos', 'N/A')}")
                summary.append(f"  CHoCH: {structure.get('choch', 'N/A')}")
                
                # استفاده از تابع safe_price برای جلوگیری از AttributeError
                last_high = safe_price(structure.get('last_high', 0))
                last_low = safe_price(structure.get('last_low', 0))
                
                summary.append(f"  Last High: ${last_high:.2f}")
                summary.append(f"  Last Low: ${last_low:.2f}")
                summary.append(f"  PD Zone: {signal_data.get('pd_zone', 'N/A')}")
                summary.append("")
                
                # اطلاعات نوسان
                summary.append("VOLATILITY ANALYSIS:")
                atr = float(signal_data.get('atr', 0))
                daily_atr = float(signal_data.get('daily_atr', 0))
                summary.append(f"  Current ATR: ${atr:.2f}")
                summary.append(f"  Daily ATR: ${daily_atr:.2f}")
                if daily_atr > 0:
                    summary.append(f"  ATR Ratio: {(atr / daily_atr):.2f}")
                summary.append(f"  Volatility State: {signal_data.get('volatility_state', 'NORMAL')}")
                summary.append("")
                
                # اطلاعات حجم
                summary.append("VOLUME ANALYSIS:")
                summary.append(f"  RVOL: {float(signal_data.get('rvol', 0)):.2f}x")
                summary.append(f"  Volume Zone: {signal_data.get('volume_zone', 'NORMAL')}")
                summary.append(f"  Volume Trend: {signal_data.get('volume_trend', 'N/A')}")
                summary.append("")
                
                # دلایل سیگنال
                reasons = signal_data.get('reasons', [])
                if reasons:
                    summary.append("SIGNAL REASONS:")
                    for reason in reasons:
                        summary.append(f"  • {reason}")
                    summary.append("")
                
                # اطلاعات معامله
                if order_details:
                    summary.append("TRADE EXECUTION DETAILS:")
                    summary.append(f"  Side: {order_details.get('side', 'N/A')}")
                    summary.append(f"  Entry: ${float(order_details.get('entry', 0)):.2f}")
                    summary.append(f"  Stop Loss: ${float(order_details.get('sl', 0)):.2f}")
                    summary.append(f"  Take Profit: ${float(order_details.get('tp', 0)):.2f}")
                    summary.append(f"  SL Distance: ${float(order_details.get('sl_distance', 0)):.2f}")
                    summary.append(f"  Risk/Reward: {float(order_details.get('rr_ratio', 0)):.2f}")
                    summary.append(f"  Lot Size: {float(order_details.get('lot', 0)):.3f}")
                    
                    lot_calc = order_details.get('lot_calculation', {})
                    if lot_calc:
                        summary.append(f"  Risk Amount: ${float(lot_calc.get('actual_risk_amount', 0)):.2f}")
                        summary.append(f"  Risk %: {float(lot_calc.get('actual_risk_percent', 0)):.2f}%")
                    summary.append("")
                
                # FVGها - مدیریت امن برای لیست‌ها
                fvgs = signal_data.get('fvgs', [])
                if isinstance(fvgs, list) and fvgs:
                    unfilled_fvgs = [f for f in fvgs if isinstance(f, dict) and not f.get('filled', True)]
                    if unfilled_fvgs:
                        summary.append("ACTIVE FVG ZONES:")
                        for fvg in unfilled_fvgs[:4]:
                            summary.append(f"  {fvg.get('type', 'N/A')}: "
                                        f"${float(fvg.get('bottom', 0)):.2f}-${float(fvg.get('top', 0)):.2f}")
                        summary.append("")
                
                summary.append("=" * 70)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("\n".join(summary))
                
                self._logger.info(f"✅ Trade summary saved to {filename}")
                return str(filename)
                
            except Exception as e:
                self._logger.error(f"❌ Error generating trade summary: {e}", exc_info=True)
                return "" # برگرداندن استرینگ خالی برای جلوگیری از کرش کل سیستم
    
    def generate_full_report(self, 
                             df: pd.DataFrame, 
                             signal_data: Dict, 
                             order_details: Optional[Dict] = None,
                             trades_history: Optional[List[Dict]] = None,
                             base_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        تولید گزارش کامل (Excel + Chart + Summary)
        
        Args:
            df: دیتافریم قیمت
            signal_data: نتایج تحلیل
            order_details: اطلاعات معامله
            trades_history: تاریخچه معاملات
            base_filename: نام پایه فایل (بدون پسوند)
            
        Returns:
            Dict: شامل مسیرهای فایل‌های ایجاد شده و نتایج
        """
        try:
            if base_filename is None:
                timestamp = self._get_timestamp()
                symbol = signal_data.get('symbol', 'XAUUSD')
                base_filename = f"{symbol}_report_{timestamp}"
            
            # تولید گزارش‌ها
            results = {}
            file_paths = {}
            
            # 1. گزارش Excel
            excel_file = self.output_dir / 'excel' / f"{base_filename}.xlsx"
            results['excel'] = self.save_excel_report(
                df, signal_data, order_details, trades_history, str(excel_file)
            )
            file_paths['excel'] = results['excel']
            
            # 2. نمودار
            chart_file = self.output_dir / 'charts' / f"{base_filename}.png"
            results['chart'] = self.plot_chart(
                df, signal_data, order_details, str(chart_file)
            )
            file_paths['chart'] = results['chart']
            
            # 3. خلاصه متنی
            summary_file = self.output_dir / 'summaries' / f"{base_filename}.txt"
            results['summary'] = self.generate_trade_summary(
                signal_data, order_details, str(summary_file)
            )
            file_paths['summary'] = results['summary']
            
            # لاگ نتیجه
            successful = [k for k, v in results.items() if v]
            
            if successful:
                self._logger.info(f"✅ Successfully generated: {', '.join(successful)}")
                
                # ایجاد گزارش ترکیبی
                combined_report = {
                    'generated_at': datetime.now().isoformat(),
                    'files': file_paths,
                    'signal': signal_data.get('signal'),
                    'confidence': signal_data.get('confidence'),
                    'symbol': signal_data.get('symbol'),
                    'timeframe': signal_data.get('timeframe')
                }
                
                # ذخیره گزارش ترکیبی
                combined_file = self.output_dir / 'performance' / f"{base_filename}_combined.json"
                with open(combined_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_report, f, indent=2, ensure_ascii=False)
                
                file_paths['combined'] = str(combined_file)
            
            return {
                'file_paths': file_paths,
                'results': results,
                'success': len(successful) == 3  # همه ۳ گزارش موفق بودند
            }
            
        except Exception as e:
            self._logger.error(f"❌ Error generating full report: {e}", exc_info=True)
            raise
    
    def generate_performance_report(self, 
                                    trades_data: List[Dict],
                                    time_period: str = 'daily') -> str:
        """
        تولید گزارش عملکرد بر اساس تاریخچه معاملات
        
        Args:
            trades_data: داده‌های معاملات
            time_period: daily, weekly, monthly
            
        Returns:
            str: مسیر فایل گزارش
        """
        try:
            if not trades_data:
                self._logger.warning("No trades data available for performance report")
                return ""
            
            # محاسبه متریک‌ها
            metrics = self._calculate_metrics(trades_data)
            
            # ایجاد گزارش
            timestamp = self._get_timestamp()
            report_file = self.output_dir / 'performance' / f"performance_{time_period}_{timestamp}.json"
            
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'time_period': time_period,
                'metrics': asdict(metrics),
                'trades_count': len(trades_data),
                'summary': {
                    'net_profit': self._format_currency(metrics.net_profit),
                    'win_rate': f"{metrics.win_rate:.2f}%",
                    'profit_factor': f"{metrics.profit_factor:.2f}",
                    'avg_rr': f"{metrics.avg_rr:.2f}"
                }
            }
            
            # ذخیره گزارش
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self._logger.info(f"✅ Performance report saved to {report_file}")
            return str(report_file)
            
        except Exception as e:
            self._logger.error(f"❌ Error generating performance report: {e}")
            raise

# تابع سازگاری برای استفاده قدیمی
def generate_full_report(df, signal_data, order_details=None, base_filename=None):
    """
    تابع سازگاری برای استفاده از نسخه قدیمی
    """
    generator = ReportGenerator()
    return generator.generate_full_report(df, signal_data, order_details, None, base_filename)