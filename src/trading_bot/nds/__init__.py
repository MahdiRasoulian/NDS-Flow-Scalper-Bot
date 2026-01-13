# src/trading_bot/nds/__init__.py
"""
NDS Analysis Package for Gold Trading
"""

from .analyzer import GoldNDSAnalyzer, analyze_gold_market
from .models import (
    SwingPoint, SwingType, FVG, FVGType, OrderBlock, 
    LiquiditySweep, MarketStructure, MarketTrend, 
    MarketState, SignalType,
    SessionAnalysis, VolumeAnalysis, MarketMetrics
)
from .constants import (
    ANALYSIS_CONFIG_KEYS, SESSION_MAPPING
)

__version__ = "4.0.0"
__author__ = "Senior ICT Trader & Python Expert"
__all__ = [
    'GoldNDSAnalyzer',
    'analyze_gold_market',
    # Models
    'SwingPoint', 'SwingType', 'FVG', 'FVGType', 'OrderBlock',
    'LiquiditySweep', 'MarketStructure', 'MarketTrend', 
    'MarketState', 'SignalType',
    'SessionAnalysis', 'VolumeAnalysis', 'MarketMetrics',
    # Constants
    'ANALYSIS_CONFIG_KEYS', 'SESSION_MAPPING'
]
