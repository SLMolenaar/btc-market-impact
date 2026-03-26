"""
fetch_data.py
-------------
Downloads 3 months of BTC/USDT trade and order book data from
Binance's public data portal (data.binance.vision).

No API key required. No rate limits. Run this just once

What we download:
  - aggTrades: every aggregated market trade (size, price, side, timestamp)
  - bookTicker: best bid/ask at every update (lets us compute mid-price)

After running this script, run process_data.py to merge and clean.
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOL = "BTCUSDT"
# 3 months of data.
MONTHS = ["2025-12", "2026-01", "2026-02"]
BASE_URL = "https://data.binance.vision/data/spot/monthly"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> bool:
    """Download file with a progress bar. Returns True on success."""
    if dest.exists():
        print(f"  Already exists, skipping: {dest.name}")
        return True

    r = requests.get(url, stream=True, timeout=60)
    if r.status_code == 404:
        print(f"  Not found (404): {url}")
        return False
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f, tqdm(
            desc=dest.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return True


def unzip_and_clean(zip_path: Path) -> Path:
    """Unzip file, return path to the extracted CSV, delete the zip."""
    out_dir = zip_path.parent
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        csv_name = next(n for n in names if n.endswith(".csv"))
        zf.extract(csv_name, out_dir)

    zip_path.unlink()  # delete zip to save disk space
    return out_dir / csv_name


def fetch_agg_trades():
    """
    aggTrades columns (no header in file):
      agg_trade_id, price, qty, first_trade_id, last_trade_id,
      timestamp_ms, is_buyer_maker

    is_buyer_maker = True  → sell market order (buyer is the maker = passive)
    is_buyer_maker = False → buy market order  (buyer is the taker = aggressive)
    """
    print("\nFetching aggTrades")
    out_dir = RAW_DIR / "aggTrades"
    out_dir.mkdir(parents=True, exist_ok=True)

    for month in MONTHS:
        filename = f"{SYMBOL}-aggTrades-{month}"
        url = f"{BASE_URL}/aggTrades/{SYMBOL}/{filename}.zip"
        zip_path = out_dir / f"{filename}.zip"
        csv_path = out_dir / f"{filename}.csv"

        if csv_path.exists():
            print(f"  Already processed: {csv_path.name}")
            continue

        print(f"\n  Downloading {month}...")
        if download_file(url, zip_path):
            extracted = unzip_and_clean(zip_path)
            # Rename to consistent name if needed
            if extracted != csv_path:
                extracted.rename(csv_path)
            print(f"  Saved: {csv_path.name}")


def fetch_book_ticker():
    """
    bookTicker columns (no header in file):
      update_id, best_bid_price, best_bid_qty,
      best_ask_price, best_ask_qty, timestamp_ms

    This gives us best bid/ask at every order book update.
    We use this to compute mid-price before and after each trade.
    """
    print("\nFetching bookTicker")
    out_dir = RAW_DIR / "bookTicker"
    out_dir.mkdir(parents=True, exist_ok=True)

    for month in MONTHS:
        filename = f"{SYMBOL}-bookTicker-{month}"
        url = f"{BASE_URL}/bookTicker/{SYMBOL}/{filename}.zip"
        zip_path = out_dir / f"{filename}.zip"
        csv_path = out_dir / f"{filename}.csv"

        if csv_path.exists():
            print(f"  Already processed: {csv_path.name}")
            continue

        print(f"\n  Downloading {month}...")
        if download_file(url, zip_path):
            extracted = unzip_and_clean(zip_path)
            if extracted != csv_path:
                extracted.rename(csv_path)
            print(f"  Saved: {csv_path.name}")


def verify_downloads():
    """Print a summary of what was downloaded."""
    print("\nDownload summary")
    for data_type in ["aggTrades", "bookTicker"]:
        folder = RAW_DIR / data_type
        files = sorted(folder.glob("*.csv")) if folder.exists() else []
        total_mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"  {data_type}: {len(files)} files, {total_mb:.0f} MB")
        for f in files:
            mb = f.stat().st_size / 1e6
            print(f"    {f.name}  ({mb:.0f} MB)")


if __name__ == "__main__":
    print("Binance Historical Data Fetcher")
    print("Symbol:", SYMBOL)
    print("Months:", MONTHS)
    print("Output:", RAW_DIR.resolve())

    fetch_agg_trades()
    fetch_book_ticker()
    verify_downloads()

    print("\nDone. Run src/process_data.py next.")