"""Memory-mapped file protocol definitions for the NDS MT5 bridge."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import struct
from typing import Dict, List, Union

MAGIC = 0x4E445342  # 'NDSB'
VERSION = 1

REQUEST_STRUCT = struct.Struct(
    "<IHHQq12sddddddddqqI"  # header + symbol + pricing + OHLC + times + market_state
)
RESPONSE_STRUCT = struct.Struct(
    "<IHHQdddddI256s"  # header + order + json payload
)

REQUEST_SIZE = REQUEST_STRUCT.size
RESPONSE_SIZE = RESPONSE_STRUCT.size
BUFFER_SIZE = 2048


@dataclass
class MarketSnapshot:
    symbol: str
    timestamp_ms: int
    bid: float
    ask: float
    spread: float
    ohlc_current: List[float]
    ohlc_previous: List[float]
    bar_time_current: int
    bar_time_previous: int
    market_state: int = 0
    sequence: int = 0
    flags: int = 0

    def pack(self) -> bytes:
        symbol_bytes = self.symbol.encode("ascii", errors="ignore")[:11]
        symbol_bytes = symbol_bytes + b"\x00" * (12 - len(symbol_bytes))
        return REQUEST_STRUCT.pack(
            MAGIC,
            VERSION,
            self.flags,
            self.sequence,
            self.timestamp_ms,
            symbol_bytes,
            self.bid,
            self.ask,
            self.spread,
            *self.ohlc_current,
            *self.ohlc_previous,
            self.bar_time_current,
            self.bar_time_previous,
            self.market_state,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "MarketSnapshot":
        fields = REQUEST_STRUCT.unpack(payload[:REQUEST_SIZE])
        (
            magic,
            version,
            flags,
            sequence,
            timestamp_ms,
            symbol_bytes,
            bid,
            ask,
            spread,
            o_cur,
            h_cur,
            l_cur,
            c_cur,
            o_prev,
            h_prev,
            l_prev,
            c_prev,
            bar_time_current,
            bar_time_previous,
            market_state,
        ) = fields
        if magic != MAGIC or version != VERSION:
            raise ValueError("Invalid bridge request header")
        symbol = symbol_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore")
        return cls(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            bid=bid,
            ask=ask,
            spread=spread,
            ohlc_current=[o_cur, h_cur, l_cur, c_cur],
            ohlc_previous=[o_prev, h_prev, l_prev, c_prev],
            bar_time_current=bar_time_current,
            bar_time_previous=bar_time_previous,
            market_state=market_state,
            sequence=sequence,
            flags=flags,
        )


@dataclass
class ExecutionCommand:
    action: Union[int, str]
    entry: float
    sl: float
    tp: float
    volume: float
    confidence: float = 0.0
    reason_codes: List[str] = field(default_factory=list)
    sequence: int = 0
    flags: int = 0

    def pack(self) -> bytes:
        payload = json.dumps(
            {
                "action": self.action,
                "entry": self.entry,
                "sl": self.sl,
                "tp": self.tp,
                "volume": self.volume,
                "confidence": self.confidence,
                "reason_codes": self.reason_codes,
            }
        ).encode("utf-8")
        payload = payload[:255]
        payload = payload + b"\x00" * (256 - len(payload))
        if isinstance(self.action, str):
            action_code = ACTION_TO_CODE.get(self.action.upper(), 0)
        else:
            action_code = int(self.action)
        return RESPONSE_STRUCT.pack(
            MAGIC,
            VERSION,
            action_code,
            self.sequence,
            self.entry,
            self.sl,
            self.tp,
            self.volume,
            self.confidence,
            self.flags,
            payload,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "ExecutionCommand":
        (
            magic,
            version,
            action_code,
            sequence,
            entry,
            sl,
            tp,
            volume,
            confidence,
            flags,
            json_payload,
        ) = RESPONSE_STRUCT.unpack(payload[:RESPONSE_SIZE])
        if magic != MAGIC or version != VERSION:
            raise ValueError("Invalid bridge response header")
        action = int(action_code)
        reason_codes: List[str] = []
        if json_payload:
            raw = json_payload.split(b"\x00", 1)[0]
            if raw:
                data = json.loads(raw.decode("utf-8"))
                reason_codes = data.get("reason_codes", [])
        return cls(
            action=action,
            entry=entry,
            sl=sl,
            tp=tp,
            volume=volume,
            confidence=confidence,
            reason_codes=reason_codes,
            sequence=sequence,
            flags=flags,
        )

    def to_payload(self) -> Dict[str, object]:
        if isinstance(self.action, str):
            signal = self.action.upper()
        else:
            signal = CODE_TO_ACTION.get(self.action, "NONE")
        return {
            "signal": signal,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "volume": self.volume,
            "confidence": self.confidence,
            "reason_codes": self.reason_codes,
        }


ACTION_TO_CODE = {
    "NONE": 0,
    "BUY": 1,
    "SELL": 2,
    "BUY_LIMIT": 3,
    "SELL_LIMIT": 4,
    "BUY_STOP": 5,
    "SELL_STOP": 6,
}
CODE_TO_ACTION = {value: key for key, value in ACTION_TO_CODE.items()}
