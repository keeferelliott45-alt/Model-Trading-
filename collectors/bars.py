"""Hourly OHLCV for top USDT perps from Bitget (public, free, deep history) -> data/bars/<SYM>.parquet + `bars` view in store."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "bars"
U = "https://api.bitget.com/api/v2/mix/market"


def top_symbols(n: int = 45) -> list[str]:
    d = requests.get(f"{U}/tickers", params={"productType": "usdt-futures"}, timeout=(10, 30)).json()["data"]
    df = pd.DataFrame(d)
    df["v"] = pd.to_numeric(df.usdtVolume, errors="coerce")
    return df.sort_values("v", ascending=False).symbol.head(n).tolist()


def fetch_symbol(sym: str, years: float = 2.5) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{sym}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    stop = int((pd.Timestamp.now("UTC") - pd.Timedelta(days=365 * years)).timestamp() * 1000)
    if old is not None and len(old):
        stop = max(stop, int(old.index.max().timestamp() * 1000) - 3_600_000 * 3)
    end, rows = int(time.time() * 1000), []
    while end > stop:
        for a in range(4):
            try:
                r = requests.get(f"{U}/history-candles", params={"symbol": sym, "productType": "usdt-futures", "granularity": "1H", "endTime": end, "limit": 200}, timeout=(10, 30))
                r.raise_for_status()
                data = r.json()["data"]
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.5 * (a + 1)); data = None
        if not data:
            break
        rows += data
        oldest = int(data[0][0])
        if oldest >= end:
            break
        end = oldest - 1
        time.sleep(0.07)
    if not rows:
        return old if old is not None else pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v", "qv"][: len(rows[0])]).astype({"ts": "int64"})
    df = df.astype({k: float for k in df.columns if k != "ts"})
    df.index = pd.to_datetime(df.ts, unit="ms")
    df = df.drop(columns="ts").sort_index()
    if old is not None:
        df = pd.concat([old, df])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)
    return df


def run(con=None, n=45) -> int:
    syms = top_symbols(n)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda s: (s, fetch_symbol(s)), syms))
    return sum(len(d) for _, d in res)
