"""Deribit perpetual funding-rate history (free, no key) -> daily mean funding series."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger("funding_connector")
URL = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
DAY_MS = 86_400_000


def fetch_funding(instrument: str, start: str, cache_dir: str = "cache", max_age_hours: int = 20) -> pd.Series:
    path = Path(cache_dir) / f"funding_{instrument}.csv"
    if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < max_age_hours:
        return pd.read_csv(path, index_col=0, parse_dates=True)["funding"]

    s = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end = int(time.time() * 1000)
    rows = []
    while s < end:
        e = min(s + 30 * DAY_MS, end)
        for attempt in range(4):
            try:
                r = requests.get(URL, params={"instrument_name": instrument, "start_timestamp": s, "end_timestamp": e}, timeout=30)
                r.raise_for_status()
                rows += r.json().get("result", [])
                break
            except Exception as ex:  # noqa: BLE001
                logger.warning("deribit %s attempt %d: %s", instrument, attempt + 1, ex)
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"deribit funding fetch failed for {instrument}")
        s = e
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no funding data for {instrument}")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    daily = df.groupby("date")["interest_8h"].mean().rename("funding")
    path.parent.mkdir(exist_ok=True)
    daily.to_frame().to_csv(path)
    return daily
