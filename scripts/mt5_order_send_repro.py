#!/usr/bin/env python3
"""
Minimal MT5 order_send reproduction script.
Uses MT5Client build/sanitize path for MARKET and STOP orders.
"""

import argparse
import logging
from datetime import datetime

import MetaTrader5 as mt5

from src.trading_bot.mt5_client import MT5Client


def _print_request(label: str, request: dict) -> None:
    types = {key: type(value).__name__ for key, value in request.items()}
    print(f"\n[{label}] REQUEST @ {datetime.now().isoformat()}")
    print(request)
    print(f"[{label}] REQ_TYPES")
    print(types)


def main() -> int:
    parser = argparse.ArgumentParser(description="MT5 order_send reproduction (MARKET + STOP).")
    parser.add_argument("--symbol", default="XAUUSD!", help="Symbol to trade (default: XAUUSD!)")
    parser.add_argument("--volume", type=float, default=0.01, help="Order volume (default: 0.01)")
    parser.add_argument("--stop-offset", type=float, default=5.0, help="STOP price offset in points (default: 5.0)")
    parser.add_argument("--comment", default="NDS Scalping - REPRO", help="Order comment")
    parser.add_argument("--dry-run", action="store_true", help="Build requests only (skip order_send).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    client = MT5Client()

    if not client.connect():
        print("❌ Failed to connect to MT5.")
        return 1

    tick = client.get_current_tick(args.symbol)
    if not tick:
        print("❌ Failed to get tick data.")
        return 1

    bid = tick["bid"]

    market_request = client.build_order_request(
        order_action="MARKET",
        symbol=args.symbol,
        volume=args.volume,
        order_type="SELL",
        price=bid,
        stop_loss=bid + 4.0,
        take_profit=bid - 6.0,
        comment=args.comment,
        magic=202401,
        deviation=10,
        type_time=mt5.ORDER_TIME_GTC,
        type_filling=mt5.ORDER_FILLING_RETURN,
    )

    stop_request = client.build_order_request(
        order_action="STOP",
        symbol=args.symbol,
        volume=args.volume,
        order_type="SELL_STOP",
        price=bid - args.stop_offset,
        stop_loss=bid + 4.0,
        take_profit=bid - 6.0,
        comment=f"{args.comment} | Stop Order",
        magic=202403,
        deviation=5,
        type_time=mt5.ORDER_TIME_GTC,
        type_filling=mt5.ORDER_FILLING_RETURN,
    )

    market_request = client.sanitize_mt5_request(market_request)
    stop_request = client.sanitize_mt5_request(stop_request)

    _print_request("MARKET", market_request)
    _print_request("STOP", stop_request)

    if args.dry_run:
        print("ℹ️ Dry run enabled. Skipping order_send.")
        return 0

    market_result = client._order_send_with_retry(market_request, args.symbol, "repro_market")
    print(f"MARKET result: {market_result}")

    stop_result = client._order_send_with_retry(stop_request, args.symbol, "repro_stop")
    print(f"STOP result: {stop_result}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
