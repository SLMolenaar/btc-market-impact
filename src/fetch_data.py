"""
fetch_data.py

Downloads BTC/USDT aggTrades from data.binance.vision.
No API key required.

Output: data/raw/aggTrades/BTCUSDT-aggTrades-YYYY-MM.csv

aggTrades columns (no header):
    agg_trade_id, price, qty, first_trade_id, last_trade_id,
    timestamp_ms, is_buyer_maker

Run: python src/fetch_data.py
"""

import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

SYMBOL   = "BTCUSDT"
MONTHS   = ["2025-12", "2026-01"]
BASE_URL = "https://data.binance.vision/data/spot/monthly"
RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return True

    r = requests.get(url, stream=True, timeout=60)
    if r.status_code == 404:
        print(f"  Not found (404): {url}")
        return False
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f, tqdm(
        desc=dest.name, total=total,
        unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return True


def unzip_and_clean(zip_path: Path, expected_csv: Path) -> bool:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        csv_name = next((n for n in names if n.endswith(".csv")), None)
        if not csv_name:
            return False
        zf.extract(csv_name, zip_path.parent)

    extracted = zip_path.parent / csv_name
    if extracted != expected_csv:
        extracted.rename(expected_csv)

    zip_path.unlink()
    return True


def fetch_agg_trades():
    out_dir = RAW_DIR / "aggTrades"
    out_dir.mkdir(parents=True, exist_ok=True)

    for month in MONTHS:
        filename = f"{SYMBOL}-aggTrades-{month}"
        csv_path = out_dir / f"{filename}.csv"
        zip_path = out_dir / f"{filename}.zip"

        if csv_path.exists():
            print(f"  Already processed: {csv_path.name}")
            continue

        url = f"{BASE_URL}/aggTrades/{SYMBOL}/{filename}.zip"
        print(f"\n  Downloading {month}...")
        if download_file(url, zip_path):
            unzip_and_clean(zip_path, csv_path)
            print(f"  Saved: {csv_path.name}")


def print_summary():
    folder = RAW_DIR / "aggTrades"
    files  = sorted(folder.glob("*.csv")) if folder.exists() else []
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"\nSummary: {len(files)} files, {total_mb:.0f} MB total")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    print(f"Symbol: {SYMBOL}")
    print(f"Months: {MONTHS}")
    print(f"Output: {RAW_DIR.resolve()}\n")

    fetch_agg_trades()
    print_summary()
    print("\nDone. Run src/process_data.py next.")
