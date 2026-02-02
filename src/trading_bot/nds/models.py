# src/trading_bot/nds/models.py
"""
مدل‌های داده برای سیستم NDS
"""
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any

class SwingType(Enum):
    HIGH = "HIGH"
    LOW = "LOW"
    HH = "HH"
    LH = "LH"
    HL = "HL"
    LL = "LL"

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"
    NEUTRAL = "NONE"

class FVGType(Enum):
    BULLISH = "BULLISH_FVG"
    BEARISH = "BEARISH_FVG"

class MarketTrend(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGING = "RANGING"

class MarketState(Enum):
    HIGH_VOLATILITY_TREND = "HIGH_VOLATILITY_TREND"
    LOW_VOLATILITY_RANGE = "LOW_VOLATILITY_RANGE"
    NORMAL_MARKET = "NORMAL_MARKET"
    BREAKOUT = "BREAKOUT"

@dataclass
class SwingPoint:
    index: int
    price: float
    time: datetime
    type: SwingType
    side: str

    def __str__(self):
        return f"{self.type.value} @ {self.price:.2f} ({self.time.strftime('%Y-%m-%d %H:%M')})"

@dataclass
class FVG:
    type: FVGType
    top: float
    bottom: float
    mid: float
    time: datetime
    index: int
    filled: bool = False
    size: float = 0.0
    strength: float = 1.0  # قدرت FVG از 0.5 تا 2.0

    @property
    def height(self) -> float:
        return abs(self.top - self.bottom)

    def is_price_in_fvg(self, price: float) -> bool:
        return min(self.top, self.bottom) <= price <= max(self.top, self.bottom)

@dataclass
class OrderBlock:
    type: str  # 'BULLISH_OB' or 'BEARISH_OB'
    high: float
    low: float
    time: datetime
    index: int
    strength: float = 1.0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

@dataclass
class LiquiditySweep:
    time: datetime
    type: str
    level: float
    penetration: float
    description: str
    strength: float = 1.0

@dataclass
class MarketStructure:
    """ساختار بازار با قابلیت‌های پیشرفته تحلیل"""
    trend: MarketTrend
    bos: str
    choch: str
    last_high: Optional[SwingPoint]
    last_low: Optional[SwingPoint]
    current_price: float
    range_width: Optional[float] = None
    range_mid: Optional[float] = None
    bos_choch_confidence: float = 0.0  # 🔥 جدید: اطمینان از تشخیص
    volume_analysis: Optional[Dict] = None  # 🔥 جدید: تحلیل حجم
    volatility_state: Optional[str] = None  # 🔥 جدید: وضعیت نوسان
    adx_value: Optional[float] = None  # 🔥 جدید: قدرت روند
    structure_score: float = 0.0  # 🔥 جدید: امتیاز کلی ساختار

    def __str__(self):
        confidence_str = f"Confidence: {self.bos_choch_confidence:.1%}" if self.bos_choch_confidence > 0 else "Confidence: N/A"
        return (f"Trend: {self.trend.value}, BOS: {self.bos}, CHoCH: {self.choch}, "
                f"{confidence_str}, Range: {self.range_width or 0:.2f}, "
                f"Score: {self.structure_score:.1f}")

    def is_valid_structure(self) -> bool:
        """بررسی اعتبار ساختار شناسایی شده"""
        if self.bos == "NONE" and self.choch == "NONE":
            return False

        # حداقل اطمینان ۴۰٪
        if self.bos_choch_confidence < 0.4:
            return False

        # رنج معتبر (برای طلا حداقل ۲ ATR)
        if self.range_width and self.range_width < (self.current_price * 0.001):
            return False

        return True

    def get_structure_priority(self) -> int:
        """اولویت‌بندی ساختار برای معامله"""
        priority_map = {
            ("BULLISH_BOS", "NONE"): 10,
            ("BEARISH_BOS", "NONE"): 10,
            ("NONE", "BULLISH_CHOCH"): 8,
            ("NONE", "BEARISH_CHOCH"): 8,
            ("BULLISH_BOS", "BEARISH_CHOCH"): 6,  # تضاد
            ("BEARISH_BOS", "BULLISH_CHOCH"): 6,  # تضاد
        }
        return priority_map.get((self.bos, self.choch), 0)

@dataclass
class VolumeAnalysis:
    rvol: float
    volume_zone: str  # VERY_HIGH, HIGH, NORMAL, LOW, VERY_LOW
    volume_trend: str  # INCREASING, DECREASING, STABLE
    vwap: float
    vwap_distance_pct: float
    volume_cluster_center: float

@dataclass
class SessionAnalysis:
    current_session: str
    session_weight: float
    weight: float
    gmt_hour: int
    is_active_session: bool
    is_overlap: bool
    session_activity: str
    optimal_trading: bool
    ts_broker: Optional[datetime] = None
    time_mode: Optional[str] = None
    broker_utc_offset_hours: Optional[float] = None
    policy_mode: Optional[str] = None
    is_tradable: Optional[bool] = None
    block_reason: Optional[str] = None
    untradable: bool = False
    untradable_reasons: Optional[str] = None
    session_decision: Optional[Dict[str, Any]] = None

@dataclass
class MarketMetrics:
    atr_value: float
    atr_pct: float
    daily_range: float
    daily_high: float
    daily_low: float
    daily_mid: float
    daily_position: float
    adx_value: float
    plus_di: float
    minus_di: float
    di_trend: str
    volatility_ratio: float
    volatility_state: str

@dataclass
class AnalysisResult:
    signal: str
    confidence: float
    score: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasons: List[str]
    context: Dict[str, Any]
    timestamp: str
    timeframe: str
    current_price: float

@dataclass
class LivePriceSnapshot:
    bid: float
    ask: float
    timestamp: Optional[str] = None

@dataclass
class FinalizedOrderParams:
    # NOTE: In dataclasses, all non-default fields must come before default fields.
    signal: str
    order_type: str
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    risk_amount_usd: float
    rr_ratio: float
    deviation_pips: float
    decision_notes: List[str]
    is_trade_allowed: bool

    # Optional / default fields (must be after non-default fields)
    reject_reason: Optional[str] = None
    take_profit2: Optional[float] = None
    tp2: Optional[float] = None
    final_entry: Optional[float] = None
    final_stop_loss: Optional[float] = None
    final_take_profit: Optional[float] = None
    final_sl: Optional[float] = None
    final_tp: Optional[float] = None
    lot: Optional[float] = None
    rr_tp1: Optional[float] = None
    rr_tp2: Optional[float] = None
    rr_checked: Optional[str] = None
    min_rr_effective: Optional[float] = None
    min_rr_source: Optional[str] = None
    sl_pips: Optional[float] = None
    tp1_pips: Optional[float] = None
    tp2_pips: Optional[float] = None

    def to_standard_payload(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "is_trade_allowed": self.is_trade_allowed,
            "reject_reason": self.reject_reason,
            "order_type": self.order_type,
            "final_entry": self.final_entry if self.final_entry is not None else self.entry_price,
            "final_sl": (
                self.final_sl
                if self.final_sl is not None
                else (self.final_stop_loss if self.final_stop_loss is not None else self.stop_loss)
            ),
            "final_tp": (
                self.final_tp
                if self.final_tp is not None
                else (self.final_take_profit if self.final_take_profit is not None else self.take_profit)
            ),
            "lot": self.lot if self.lot is not None else self.lot_size,
            "rr_ratio": self.rr_ratio,
            "deviation_pips": self.deviation_pips,
            "decision_notes": list(self.decision_notes or []),
        }
