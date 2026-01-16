import argparse
from collections import Counter

import pandas as pd

from config.settings import config
from src.trading_bot.nds.analyzer import analyze_gold_market
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.risk_manager import create_scalping_risk_manager


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("CSV must include a 'time' column")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    return df


def run_backtest(csv_path: str, cycles: int, min_bars: int) -> None:
    df = _load_csv(csv_path)

    if len(df) < min_bars:
        raise ValueError(f"Not enough rows: {len(df)} < min_bars={min_bars}")

    start_index = min_bars
    end_index = min(len(df), start_index + cycles)

    analyzer_config = config.get_full_config_for_analyzer()
    config_payload = config.get_full_config()
    symbol = config_payload.get("trading_settings", {}).get("SYMBOL", "XAUUSD")
    timeframe = config_payload.get("trading_settings", {}).get("TIMEFRAME", "M5")
    entry_factor = config_payload.get("technical_settings", {}).get("ENTRY_FACTOR", 0.25)

    risk_manager = create_scalping_risk_manager()

    counters = Counter()
    reject_reasons = Counter()

    for i in range(start_index, end_index):
        window = df.iloc[: i + 1]
        result = analyze_gold_market(
            dataframe=window,
            timeframe=timeframe,
            entry_factor=entry_factor,
            config=analyzer_config,
            scalping_mode=True,
        )

        counters["cycles"] += 1
        signal = getattr(result, "signal", None) or "NONE"
        if signal not in ("BUY", "SELL"):
            counters["analyzer_none"] += 1
            continue

        counters["actionable_signals"] += 1

        close = float(window["close"].iloc[-1])
        bid = close
        ask = close + 0.01
        live_snapshot = LivePriceSnapshot(bid=bid, ask=ask, timestamp=str(window["time"].iloc[-1]))

        finalized = risk_manager.finalize_order(
            analysis=result,
            live=live_snapshot,
            symbol=symbol,
            config=config_payload,
        )

        if finalized.is_trade_allowed:
            counters["risk_allowed"] += 1
            counters["executed"] += 1
        else:
            counters["risk_rejected"] += 1
            reject_reasons[finalized.reject_reason or "UNKNOWN"] += 1

    print("\n=== Smoke Backtest Summary ===")
    print(f"Cycles processed: {counters['cycles']}")
    print(f"Analyzer actionable signals: {counters['actionable_signals']}")
    print(f"Analyzer NONE/neutral: {counters['analyzer_none']}")
    print(f"RiskManager allowed: {counters['risk_allowed']}")
    print(f"RiskManager rejected: {counters['risk_rejected']}")
    print(f"Executed (simulated): {counters['executed']}")

    if reject_reasons:
        print("\n--- Reject Reasons ---")
        for reason, count in reject_reasons.most_common():
            print(f"{reason}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NDS Flow Scalper smoke backtest summary")
    parser.add_argument(
        "--csv",
        default="scripts/XAUUSD!_5_2026-2026.csv",
        help="Path to CSV file",
    )
    parser.add_argument("--cycles", type=int, default=200, help="Number of cycles to simulate")
    parser.add_argument("--min-bars", type=int, default=120, help="Minimum bars before evaluation")
    args = parser.parse_args()

    run_backtest(args.csv, args.cycles, args.min_bars)


if __name__ == "__main__":
    main()
