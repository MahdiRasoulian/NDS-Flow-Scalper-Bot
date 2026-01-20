from datetime import datetime, timedelta

from src.trading_bot.trade_tracker import TradeTracker


def simulate_close_reconciliation() -> None:
    tracker = TradeTracker()
    now = datetime.now()
    open_event = {
        "event_type": "OPEN",
        "event_time": now - timedelta(minutes=5),
        "symbol": "XAUUSD!",
        "order_ticket": 111,
        "position_ticket": 222,
        "side": "BUY",
        "volume": 0.1,
        "entry_price": 2350.5,
        "sl": 2345.0,
        "tp": 2360.0,
        "metadata": {},
    }
    tracker.add_trade_open(open_event)

    record = tracker.active_trades.get(222)
    tracker.register_pending_close(222, record, now)
    ready, timed_out = tracker.get_pending_close_candidates(
        now=now,
        base_backoff_sec=1,
        max_backoff_sec=4,
        timeout_sec=30,
    )
    print(f"Pending ready={len(ready)} timed_out={len(timed_out)}")

    close_event = {
        "event_type": "CLOSE",
        "event_time": now,
        "symbol": "XAUUSD!",
        "order_ticket": 111,
        "position_ticket": 222,
        "side": "BUY",
        "volume": 0.1,
        "entry_price": 2350.5,
        "exit_price": 2355.5,
        "sl": 2345.0,
        "tp": 2360.0,
        "profit": 50.0,
        "pips": 50,
        "reason": "TP/SL",
        "metadata": {"duration_sec": 300},
    }
    tracker.close_trade_event(close_event)
    print(f"Closed trades={len(tracker.closed_trades)} active={len(tracker.active_trades)}")


def simulate_reports() -> None:
    try:
        import pandas as pd
        from src.reporting.report_generator import ReportGenerator
    except ImportError:
        print("⚠️ pandas/matplotlib not available; skipping report generation simulation.")
        return

    now = datetime.now()
    df = pd.DataFrame(
        {
            "time": [now - timedelta(minutes=i) for i in range(20)][::-1],
            "open": [2348 + i * 0.1 for i in range(20)],
            "high": [2348.5 + i * 0.1 for i in range(20)],
            "low": [2347.8 + i * 0.1 for i in range(20)],
            "close": [2348.2 + i * 0.1 for i in range(20)],
            "volume": [100 + i for i in range(20)],
        }
    )
    signal_data = {
        "symbol": "XAUUSD!",
        "timeframe": "M5",
        "signal": "BUY",
        "confidence": 75,
        "score": 62,
        "structure": {"trend": "UP"},
        "atr": None,
        "daily_atr": None,
    }
    order_details = {
        "side": "BUY",
        "entry_actual": 2350.5,
        "sl_actual": 2345.0,
        "tp_actual": 2360.0,
        "rr_ratio": 1.2,
        "lot": 0.1,
    }
    report_gen = ReportGenerator(output_dir="trade_reports/scalping_reports")
    report_gen.generate_full_report(
        df=df,
        signal_data=signal_data,
        order_details=order_details,
        base_filename="SIM_OPEN_222",
    )
    close_event = {
        "event_type": "CLOSE",
        "event_time": now,
        "symbol": "XAUUSD!",
        "order_ticket": 111,
        "position_ticket": 222,
        "side": "BUY",
        "volume": 0.1,
        "entry_price": 2350.5,
        "exit_price": 2355.5,
        "sl": 2345.0,
        "tp": 2360.0,
        "profit": 50.0,
        "pips": 50,
        "reason": "TP",
        "metadata": {"duration_sec": 300},
    }
    report_gen.generate_close_report(close_event, base_filename="SIM_CLOSE_222")
    report_gen.update_daily_summary(close_event)


if __name__ == "__main__":
    simulate_close_reconciliation()
    simulate_reports()
