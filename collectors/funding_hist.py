"""Per-coin hourly funding-rate history from Hyperliquid (public, years deep) -> data/funding/<COIN>.parquet"""
from __future__ import annotations

import glob
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "funding"
BARS = Path(__file__).resolve().parents[1] / "data" / "bars"
URL = "https://api.hyperliquid.xyz/info"


def fetch(coin: str, start_ms: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (coin, "k" + coin):
        rows, s = [], start_ms
        for _ in range(80):
            try:
                r = requests.post(URL, json={"type": "fundingHistory", "coin": name, "startTime": s}, timeout=(10, 30))
                r.raise_for_status(); d = r.json()
            except Exception:  # noqa: BLE001
                time.sleep(1.5); continue
            if not d:
                break
            rows += d
            nxt = int(d[-1]["time"]) + 1
            if nxt <= s or len(d) < 400:
                break
            s = nxt
            time.sleep(0.05)
        if rows:
            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df.time.astype("int64"), unit="ms").dt.floor("h")
            df = df[["fundingRate"]].astype(float)
            df = df[~df.index.duplicated()].sort_index()
            df.to_parquet(OUT / f"{coin}.parquet")
            return len(df)
    return 0


def run() -> int:
    coins = [Path(f).stem.replace("USDT", "") for f in glob.glob(str(BARS / "*.parquet"))]
    start = int((pd.Timestamp.now("UTC") - pd.Timedelta(days=365 * 2.6)).timestamp() * 1000)
    with ThreadPoolExecutor(5) as ex:
        return sum(ex.map(lambda c: fetch(c, start), coins))
