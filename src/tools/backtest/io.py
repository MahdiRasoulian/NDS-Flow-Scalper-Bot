from __future__ import annotations

import pandas as pd

REQUIRED_COLS = ["time", "open", "high", "low", "close"]

def load_ohlcv(path: str, dayfirst: bool = False) -> pd.DataFrame:
    """
    Loads OHLCV from MT5-exported Excel/CSV and normalizes schema.

    Anti-leakage note:
    - We only load raw candles. No future-based features are computed here.
    """
    p = path.lower()

    if p.endswith(".xlsx") or p.endswith(".xls"):
        df = pd.read_excel(path)
    elif p.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type. Use .xlsx/.xls/.csv")

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Validate
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Found: {list(df.columns)}")

    # Volume can be 'volume' or 'tick_volume' in some exports
    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        else:
            df["volume"] = 0

    # Parse time (your sample: 1/9/2026 2:00)
    df["time"] = pd.to_datetime(df["time"], dayfirst=dayfirst, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Ensure numeric
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    # Basic sanity: high/low envelope
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"]  = df[["low", "open", "close"]].min(axis=1)

    return df
