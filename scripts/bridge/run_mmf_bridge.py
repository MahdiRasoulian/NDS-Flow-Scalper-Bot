"""CLI entry point for the NDS MMF bridge."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

# --- اصلاح مسیر پروژه (فقط این بخش اضافه شد) ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ----------------------------------------------

try:
    from src.bridge.mmf_bridge import BridgeConfig, JsonLoggingDecisionWrapper, MMFBridgeServer
    from src.bridge.mmf_protocol import ACTION_TO_CODE, ExecutionCommand, MarketSnapshot
    from src.trading_bot.nds.analyzer import analyze_gold_market
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class CandleBuffer:
    max_bars: int
    candles: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def update(self, bar_time: int, candle: Dict[str, float]) -> None:
        self.candles[bar_time] = candle
        if len(self.candles) > self.max_bars:
            for key in sorted(self.candles.keys())[:-self.max_bars]:
                self.candles.pop(key, None)

    def to_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for bar_time in sorted(self.candles.keys()):
            candle = self.candles[bar_time]
            rows.append(
                {
                    "time": pd.to_datetime(bar_time, unit="s"),
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": candle.get("volume", 1.0),
                }
            )
        return pd.DataFrame(rows)


class NDSBridgeAdapter:
    def __init__(self, config_path: Path) -> None:
        self.config = self._load_config(config_path)
        trading_settings = self.config.get("trading_settings", {})
        technical_settings = self.config.get("technical_settings", {})
        self.timeframe = trading_settings.get("TIMEFRAME", "M5")
        self.entry_factor = float(technical_settings.get("ENTRY_FACTOR", 0.25))
        self.scalping_mode = True
        self.volume = float(
            trading_settings.get("GOLD_SPECIFICATIONS", {}).get("MIN_LOT", 0.01)
        )
        self.buffer = CandleBuffer(
            max_bars=int(trading_settings.get("BARS_TO_FETCH", 800))
        )

    def _load_config(self, path: Path) -> Dict[str, object]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _update_buffer(self, snapshot: MarketSnapshot) -> None:
        if snapshot.bar_time_previous:
            self.buffer.update(
                snapshot.bar_time_previous,
                {
                    "open": snapshot.ohlc_previous[0],
                    "high": snapshot.ohlc_previous[1],
                    "low": snapshot.ohlc_previous[2],
                    "close": snapshot.ohlc_previous[3],
                    "volume": 1.0,
                },
            )
        if snapshot.bar_time_current:
            self.buffer.update(
                snapshot.bar_time_current,
                {
                    "open": snapshot.ohlc_current[0],
                    "high": snapshot.ohlc_current[1],
                    "low": snapshot.ohlc_current[2],
                    "close": snapshot.ohlc_current[3],
                    "volume": 1.0,
                },
            )

    def __call__(self, snapshot: MarketSnapshot) -> ExecutionCommand:
        self._update_buffer(snapshot)
        dataframe = self.buffer.to_dataframe()
        result = analyze_gold_market(
            dataframe=dataframe,
            timeframe=self.timeframe,
            entry_factor=self.entry_factor,
            config=self.config,
            scalping_mode=self.scalping_mode,
        )
        signal = str(result.signal or "NONE").upper()
        action = ACTION_TO_CODE.get(signal, 0)
        volume = self.volume if action else 0.0
        return ExecutionCommand(
            action=action,
            entry=float(result.entry_price or 0.0),
            sl=float(result.stop_loss or 0.0),
            tp=float(result.take_profit or 0.0),
            volume=volume,
            confidence=float(result.confidence or 0.0),
            reason_codes=list(result.reasons or []),
        )


def run(decision_fn: Callable[[MarketSnapshot], ExecutionCommand]) -> None:
    server = MMFBridgeServer(BridgeConfig())
    server.run(JsonLoggingDecisionWrapper(decision_fn))


if __name__ == "__main__":
    adapter = NDSBridgeAdapter(Path("config/bot_config.json"))
    run(adapter)
