"""Stress test for MT5 pending orders (STOP + LIMIT) under concurrent tick load.

Expected outcome:
  - No order_send() returns None.
  - Pending orders are accepted with retcode=10009 (DONE) and cancelled cleanly.
  - Logs show structured pending order context without API-state errors.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict

from src.trading_bot.config_utils import resolve_mt5_credentials
from src.trading_bot.mt5_client import MT5Client


def _load_bot_config() -> Dict[str, Any]:
    candidates = [
        Path.cwd() / "config" / "bot_config.json",
        Path.cwd() / "bot_config.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return dict(data) if isinstance(data, dict) else {}
    return {}


def _tick_spammer(client: MT5Client, symbol: str, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        client.get_current_tick(symbol)
        time.sleep(0.2)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pending_stress_test")

    config_payload = _load_bot_config()
    credentials = resolve_mt5_credentials(
        config_payload,
        [
            Path.cwd() / "config" / "mt5_credentials.json",
            Path.cwd() / "mt5_credentials.json",
        ],
        log=logger,
    )

    if not credentials["is_complete"]:
        raise RuntimeError("MT5 credentials are incomplete; check env/config/file sources.")

    client = MT5Client(logger=logging.getLogger("src.trading_bot.mt5_client"))
    client.config = config_payload
    client.connection_config = client._load_connection_config()

    if not client.connect():
        raise RuntimeError("Failed to connect to MT5.")

    symbol = (
        config_payload.get("trading_settings", {}).get("SYMBOL")
        if isinstance(config_payload, dict)
        else None
    ) or "XAUUSD"

    tick = client.get_current_tick(symbol)
    if not tick:
        raise RuntimeError("Failed to fetch initial tick.")

    symbol_info = client._get_symbol_info(symbol)
    if not symbol_info:
        raise RuntimeError("Failed to fetch symbol info.")

    digits = symbol_info.digits
    min_distance = client._pending_min_distance(symbol_info)
    point = symbol_info.point or 0.01
    extra = max(min_distance, point * 10)

    stop_event = threading.Event()
    spam_thread = threading.Thread(target=_tick_spammer, args=(client, symbol, stop_event), daemon=True)
    spam_thread.start()

    iterations = 5
    for idx in range(iterations):
        tick = client.get_current_tick(symbol)
        if not tick:
            logger.warning("No tick; retrying iteration %s", idx)
            time.sleep(0.5)
            continue

        bid = tick["bid"]
        ask = tick["ask"]

        buy_stop = round(ask + extra, digits)
        buy_limit = round(bid - extra, digits)
        sell_stop = round(bid - extra, digits)
        sell_limit = round(ask + extra, digits)

        logger.info("Iteration %s/%s", idx + 1, iterations)

        orders = [
            ("BUY_STOP", buy_stop),
            ("BUY_LIMIT", buy_limit),
            ("SELL_STOP", sell_stop),
            ("SELL_LIMIT", sell_limit),
        ]

        for order_type, price in orders:
            if "STOP" in order_type:
                result = client.send_stop_order(
                    symbol=symbol,
                    order_type=order_type,
                    volume=0.01,
                    stop_price=price,
                    comment=f"stress_{order_type}_{idx}",
                )
            else:
                result = client.send_limit_order(
                    symbol=symbol,
                    order_type=order_type,
                    volume=0.01,
                    limit_price=price,
                    comment=f"stress_{order_type}_{idx}",
                )

            if not result.get("success"):
                logger.error("Order failed: %s", result)
                continue

            ticket = result.get("ticket")
            time.sleep(0.5)
            if ticket:
                cancel_result = client.cancel_order(ticket)
                logger.info("Cancel result: %s", cancel_result)

        time.sleep(1.0)

    stop_event.set()
    spam_thread.join(timeout=2)
    client.disconnect()


if __name__ == "__main__":
    main()
