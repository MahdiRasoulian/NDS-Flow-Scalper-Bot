#!/usr/bin/env python3
"""Replay a saved trade JSON and verify TP1/TP2 management behavior."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config.settings import config as bot_config
from src.trading_bot.position_manager import PositionManager


@dataclass
class FakeMT5:
    closes: List[Dict[str, Any]]
    modifies: List[Dict[str, Any]]

    def close_position(self, ticket: int, volume: float = None, comment: str = "") -> Dict[str, Any]:
        payload = {"ticket": ticket, "volume": volume, "comment": comment, "price": None, "success": True}
        self.closes.append(payload)
        return payload

    def modify_position(self, ticket: int, new_sl: float = None, new_tp: float = None) -> Dict[str, Any]:
        payload = {"ticket": ticket, "new_sl": new_sl, "new_tp": new_tp, "success": True}
        self.modifies.append(payload)
        return payload


class DummyTradeTracker:
    def __init__(self, metadata: Dict[str, Any]):
        self.active_trades = {metadata["position_ticket"]: {"open_event": {"metadata": metadata}}}


def _load_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: scripts/replay_trade.py <trade.json>")
        return 1

    trade_path = Path(sys.argv[1])
    payload = _load_payload(trade_path)

    cfg = bot_config.get_full_config()
    cfg.update(payload.get("config", {}))

    mt5 = FakeMT5(closes=[], modifies=[])
    metadata = {
        "position_ticket": payload.get("position_ticket", 777),
        "tp2_price": payload.get("tp2_price"),
        "analysis_snapshot": payload.get("analysis_snapshot", {}),
        "market_metrics": payload.get("market_metrics", {}),
    }
    tracker = DummyTradeTracker(metadata)
    manager = PositionManager(cfg, mt5, trade_tracker=tracker)

    ticket = int(payload.get("position_ticket", 777))
    prices = payload.get("prices") or []
    if not prices:
        print("No prices provided in trade payload.")
        return 1

    base_position = {
        "position_ticket": ticket,
        "symbol": payload.get("symbol", "XAUUSD"),
        "side": payload.get("side", "BUY"),
        "volume": float(payload.get("volume", 1.0)),
        "entry_price": float(payload.get("entry_price")),
        "sl": float(payload.get("stop_loss")),
        "tp": float(payload.get("tp1_price")),
        "profit": 0.0,
        "magic": payload.get("magic", 0),
        "comment": payload.get("comment", "replay"),
        "open_time": datetime.utcnow(),
        "update_time": datetime.utcnow(),
    }

    for price in prices:
        position = dict(base_position)
        position["current_price"] = float(price)
        manager.manage_positions([position])

    print("Replay complete.")
    print("Partial closes:", mt5.closes)
    print("Modifications:", mt5.modifies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
