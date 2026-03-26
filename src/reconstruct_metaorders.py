"""
Reconstructs synthetic metaorders from public aggTrades using the algorithm
of Maitrier, Loeper & Bouchaud (2025), arXiv:2503.18199.

The algorithm:
  1. Assign each trade to one of N synthetic traders using a mapping function.
     Sampling is WITHOUT replacement per round of N trades, each trader is
     assigned exactly one trade before any trader is assigned a second.
     This is the key property the paper identifies as essential for recovering
     the square-root law.
  2. For each trader, group consecutive same-sign trades into metaorders.
     A metaorder ends when the trader's sign flips.
  3. For each metaorder, compute total size Q and normalized price impact I.

Parameters:
  N               number of synthetic traders (default 50)
  distribution    'homogeneous' or 'power_law'
  alpha           power-law exponent if distribution='power_law' (default 2.0)

Output columns:
  metaorder_id    unique identifier
  trader_id       synthetic trader (0 to N-1)
  start_ts        timestamp of first child order (ms)
  end_ts          timestamp of last child order (ms)
  n_child         number of child orders
  Q               total size in BTC
  Q_norm          Q normalized by daily volume (participation rate)
  sign            +1 (buy metaorder) or -1 (sell metaorder)
  I               normalized price impact: delta_P / (sigma_daily * sqrt(V_daily))
  sigma_daily     daily realized volatility at time of metaorder
  V_daily         daily volume at time of metaorder

Output: data/processed/metaorders.parquet

Run: python src/reconstruct_metaorders.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Algorithm parameters
N_TRADERS    = 20 # number of synthetic traders
DISTRIBUTION = "power_law" # 'homogeneous' or 'power_law'
ALPHA        = 2.0 # power-law exponent (only used if DISTRIBUTION='power_law')

# Minimum child orders for a metaorder to be included
MIN_CHILD_ORDERS = 10

# Impact measurement window: number of trades after metaorder end to measure price
# BTC/USDT trades ~12 times/sec, 200 trades ~ 17 seconds post-execution
IMPACT_WINDOW_TRADES = 200


def build_trader_weights(n: int, distribution: str, alpha: float) -> np.ndarray:
    """
    Build normalized trading frequency weights for N synthetic traders.

    Homogeneous: all traders trade equally often.
    Power-law: trader i has weight proportional to (i+1)^{-alpha}.
    A small number of traders dominate, as observed empirically.
    """
    if distribution == "homogeneous":
        weights = np.ones(n)
    elif distribution == "power_law":
        ranks = np.arange(1, n + 1, dtype=float)
        weights = ranks ** (-alpha)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return weights / weights.sum()


def assign_trader_ids(n_trades: int, n_traders: int, weights: np.ndarray) -> np.ndarray:
    """
    Assign a synthetic trader ID to each trade using sampling without replacement.

    Each round of N trades assigns exactly one trade to each trader via a
    weighted shuffle. This without-replacement property is essential for
    recovering the square-root law (Maitrier et al. 2025).

    Fully loopless using the Gumbel-max trick for weighted permutations:
      - Draw U ~ Uniform(0, 1) of shape (n_rounds, N)
      - Compute keys = log(U) / weights (broadcast across rows)
      - argsort each row descending gives a weighted shuffle without replacement

    This generates all rounds simultaneously in pure numpy with no Python loops.
    """
    n_full_rounds = n_trades // n_traders
    remainder     = n_trades % n_traders

    # Generate all full rounds simultaneously: shape (n_full_rounds, n_traders)
    U    = np.random.uniform(0, 1, size=(n_full_rounds, n_traders))
    keys = np.log(U) / weights          # Gumbel-max weighted permutation keys
    rounds = np.argsort(-keys, axis=1)  # descending argsort = weighted shuffle
    trader_ids = rounds.ravel()

    # Partial round for remainder trades
    if remainder > 0:
        u       = np.random.uniform(0, 1, size=n_traders)
        partial = np.argsort(-(np.log(u) / weights))[:remainder]
        trader_ids = np.concatenate([trader_ids, partial])

    return trader_ids.astype(np.int32)


def extract_metaorders(df: pd.DataFrame, trader_ids: np.ndarray) -> pd.DataFrame:
    """
    For each trader, find consecutive same-sign runs and define them as metaorders.
    Fully vectorized using pandas groupby and cumsum run-length encoding.
    Uses a minimal working dataframe to avoid memory issues on large datasets.
    """
    log_prices = np.log(df["price"].values)
    n = len(log_prices)

    work = pd.DataFrame({
        "trader_id":    trader_ids.astype(np.int32),
        "side":         df["side"].values.astype(np.int8),
        "qty":          df["qty"].values,
        "timestamp_ms": df["timestamp_ms"].values,
        "log_price":    log_prices,
    })

    # Detect sign changes within each trader sequence
    prev_sign          = work.groupby("trader_id")["side"].shift(1).fillna(0).astype(np.int8)
    work["run_change"] = (work["side"] != prev_sign).astype(np.int8)
    work["run_id"]     = work.groupby("trader_id")["run_change"].cumsum().astype(np.int32)

    # Unique metaorder key per (trader, run)
    work["mo_key"] = work["trader_id"].astype(np.int64) * 1_000_000 + work["run_id"].astype(np.int64)

    # Aggregate, use positional index for start/end to enable fast numpy lookup
    work["pos"] = np.arange(n, dtype=np.int64)

    agg = work.groupby("mo_key", sort=False).agg(
        trader_id =("trader_id",    "first"),
        start_pos =("pos",          "first"),
        end_pos   =("pos",          "last"),
        start_ts  =("timestamp_ms", "first"),
        end_ts    =("timestamp_ms", "last"),
        n_child   =("qty",          "count"),
        Q         =("qty",          "sum"),
        sign      =("side",         "first"),
    ).reset_index(drop=True)

    agg = agg[agg["n_child"] >= MIN_CHILD_ORDERS].reset_index(drop=True)

    # Compute price impact
    start_log_p = log_prices[agg["start_pos"].values]
    impact_pos  = np.minimum(agg["end_pos"].values + IMPACT_WINDOW_TRADES, n - 1)
    end_log_p   = log_prices[impact_pos]
    agg["raw_impact"] = (end_log_p - start_log_p) * agg["sign"].values

    agg = agg.drop(columns=["start_pos", "end_pos"])
    agg.insert(0, "metaorder_id", range(len(agg)))

    return agg


def compute_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily volume and realized volatility for normalization."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date
    df["log_return"] = np.log(df["price"]).diff()

    daily = df.groupby("date").agg(
        V_daily=("qty", "sum"),
        sigma_daily=("log_return", "std"),
    ).reset_index()

    return daily


def normalize_metaorders(metaorders_df: pd.DataFrame, daily_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize metaorder size and impact by daily volume and volatility.

    Q_norm = Q / V_daily          (participation rate)
    I      = raw_impact / sigma_daily   (volatility-normalized impact)

    This is the standard normalization from Maitrier et al. and the broader
    market impact literature, making impact comparable across days.
    """
    metaorders_df = metaorders_df.copy()
    metaorders_df["date"] = pd.to_datetime(
        metaorders_df["start_ts"], unit="ms"
    ).dt.date

    metaorders_df = metaorders_df.merge(daily_stats, on="date", how="left")

    metaorders_df["Q_norm"] = metaorders_df["Q"] / metaorders_df["V_daily"]
    metaorders_df["I"] = metaorders_df["raw_impact"] / metaorders_df["sigma_daily"]

    return metaorders_df.drop(columns=["raw_impact", "date"])


def verify_square_root_law(metaorders_df: pd.DataFrame):
    """
    Quick sanity check: fit log I ~ delta * log Q_norm.
    If reconstruction worked, delta should be close to 0.5.
    """
    from scipy import stats

    df = metaorders_df[metaorders_df["I"] > 0].copy()
    df = df[df["Q_norm"] > 0]

    log_q = np.log(df["Q_norm"].values)
    log_i = np.log(df["I"].values)

    # Remove outliers beyond 3 sigma
    mask = (np.abs(log_i - log_i.mean()) < 3 * log_i.std())
    log_q, log_i = log_q[mask], log_i[mask]

    slope, intercept, r, p, se = stats.linregress(log_q, log_i)

    print("\nSquare-root law verification:")
    print(f"  delta (slope) : {slope:.4f}")
    print(f"  95% CI        : [{slope - 1.96*se:.4f}, {slope + 1.96*se:.4f}]")
    print(f"  R²            : {r**2:.4f}")
    print(f"  delta ~ 0.5?  : {abs(slope - 0.5) < 0.15}")


if __name__ == "__main__":
    np.random.seed(42)

    print("Loading processed trades...")
    df = pd.read_parquet(PROCESSED_DIR / "impact_data.parquet")
    print(f"  {len(df):,} trades loaded")

    print(f"\nParameters:")
    print(f"  N traders    : {N_TRADERS}")
    print(f"  Distribution : {DISTRIBUTION}")
    if DISTRIBUTION == "power_law":
        print(f"  Alpha        : {ALPHA}")
    print(f"  Min children : {MIN_CHILD_ORDERS}")

    print("\nComputing daily stats...")
    daily_stats = compute_daily_stats(df)

    print("Building trader weights...")
    weights = build_trader_weights(N_TRADERS, DISTRIBUTION, ALPHA)

    print("Assigning synthetic trader IDs...")
    trader_ids = assign_trader_ids(len(df), N_TRADERS, weights)

    print("Extracting metaorders...")
    metaorders = extract_metaorders(df, trader_ids)
    print(f"  Extracted {len(metaorders):,} metaorders")

    metaorders_df = pd.DataFrame(metaorders)

    print("Normalizing...")
    metaorders_df = normalize_metaorders(metaorders_df, daily_stats)

    # Drop rows where normalization failed
    before = len(metaorders_df)
    metaorders_df = metaorders_df.dropna(subset=["Q_norm", "I"])
    metaorders_df = metaorders_df[
        (metaorders_df["sigma_daily"] > 0) &
        (metaorders_df["V_daily"] > 0)
    ]
    print(f"  Dropped {before - len(metaorders_df):,} rows with missing normalization")

    verify_square_root_law(metaorders_df)

    out_path = PROCESSED_DIR / "metaorders.parquet"
    metaorders_df.to_parquet(out_path, index=False)

    print(f"\nMetaorders saved to {out_path}")
    print(f"Total metaorders : {len(metaorders_df):,}")
    print(f"Median Q_norm    : {metaorders_df['Q_norm'].median():.2e}")
    print(f"Median n_child   : {metaorders_df['n_child'].median():.1f}")
    print("\nNext: open notebooks/01b_metaorders.ipynb")