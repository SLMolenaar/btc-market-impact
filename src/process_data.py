"""
process_data.py

Builds the analysis dataset from aggTrades only.

Price impact is measured as the log price change over a symmetric window
of N trades around each market order (realized impact). This is standard
when quote data is unavailable.

Output columns:
    timestamp_ms    trade timestamp
    price           trade price
    qty             trade size in BTC
    side            1 = buy market order, -1 = sell market order
    log_return      log(price[i+window] / price[i-window])
    signed_impact   side * log_return
    sigma_local     realized vol over previous 30 minutes
    volume_local    total traded volume over previous 30 minutes

Output: data/processed/impact_data.parquet

Run: python src/process_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SYMBOL        = "BTCUSDT"
MONTHS = ["2025-08", "2025-09"] # calm
# MONTHS = ["2025-11"] # stress
RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

WINDOW_MS      = 30 * 60 * 1000 # 30 minutes in milliseconds
IMPACT_WINDOW  = 10 # trades before/after for impact measurement
MIN_WINDOW_OBS = 20 # minimum trades in rolling window


def load_month(month: str) -> pd.DataFrame:
    path = RAW_DIR / "aggTrades" / f"{SYMBOL}-aggTrades-{month}.csv"
    df = pd.read_csv(path, header=None, names=[
        "agg_trade_id", "price", "qty",
        "first_trade_id", "last_trade_id",
        "timestamp_us", "is_buyer_maker", "is_best_match"
    ], dtype={
        "price": np.float64,
        "qty": np.float64,
        "timestamp_us": np.int64,
        "is_buyer_maker": bool
    })
    # Timestamps are in microseconds, convert to milliseconds
    df["timestamp_ms"] = df["timestamp_us"] // 1000
    df["side"] = np.where(df["is_buyer_maker"], -1, 1)
    return df[["timestamp_ms", "price", "qty", "side"]].sort_values("timestamp_ms").reset_index(drop=True)


def compute_impact(prices: np.ndarray, window: int) -> np.ndarray:
    """
    For each trade i, compute log(price[i+window] / price[i-window]).
    Trades within 'window' of the edges are set to NaN.
    """
    log_prices       = np.log(prices)
    shifted_forward  = np.roll(log_prices, -window)
    shifted_backward = np.roll(log_prices,  window)
    impact           = shifted_forward - shifted_backward
    impact[:window]  = np.nan
    impact[-window:] = np.nan
    return impact


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sigma_local and volume_local using a time-based rolling window.
    Fully vectorized via pandas time-based rolling on a DatetimeIndex.
    """
    df = df.copy()
    df["log_return_1"] = np.log(df["price"]).diff()
    df.index = pd.to_datetime(df["timestamp_ms"], unit="ms")

    window = f"{WINDOW_MS // 1000}s"

    df["sigma_local"]  = (
        df["log_return_1"]
        .rolling(window, min_periods=MIN_WINDOW_OBS)
        .std()
    )
    df["volume_local"] = (
        df["qty"]
        .rolling(window, min_periods=MIN_WINDOW_OBS)
        .sum()
    )

    df = df.drop(columns=["log_return_1"])
    df.index = range(len(df))
    return df


def process_month(month: str) -> pd.DataFrame:
    print(f"\nProcessing {month}...")

    path = RAW_DIR / "aggTrades" / f"{SYMBOL}-aggTrades-{month}.csv"
    if not path.exists():
        print(f"  File not found: {path.name}")
        return pd.DataFrame()

    df = load_month(month)
    print(f"  Loaded {len(df):,} trades")

    print("  Computing impact...")
    df["log_return"]    = compute_impact(df["price"].values, IMPACT_WINDOW)
    df["signed_impact"] = df["side"] * df["log_return"]

    print("  Computing rolling features...")
    df = compute_rolling_features(df)

    before = len(df)
    df = df.dropna(subset=["log_return", "sigma_local", "volume_local"])
    df = df[df["sigma_local"] > 0]
    print(f"  Dropped {before - len(df):,} rows with missing features")
    print(f"  Remaining: {len(df):,} trades")

    return df


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for month in MONTHS:
        df = process_month(month)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("No data processed. Run fetch_data.py first.")
        raise SystemExit(1)

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )

    out_path = PROCESSED_DIR / "impact_data.parquet"
    combined.to_parquet(out_path, index=False)

    t_min = pd.to_datetime(combined["timestamp_ms"].min(), unit="ms")
    t_max = pd.to_datetime(combined["timestamp_ms"].max(), unit="ms")

    print(f"\nTotal trades : {len(combined):,}")
    print(f"Date range   : {t_min} to {t_max}")
    print(f"Saved to     : {out_path}")