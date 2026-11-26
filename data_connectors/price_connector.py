"""Daily OHLC price connector with a fallback chain and local cache.

Order: yfinance -> stooq -> Google Finance CSV (via Sheets-equivalent endpoint is not
directly scriptable, so stooq is the practical free no-key fallback for equities/ETFs;
yfinance covers crypto tickers like BTC-USD/ETH-USD directly).
"""
from __future__ import annotations

import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger("price_connector")


class PriceFetchError(Exception):
    pass


def _cache_path(cache_dir: str, ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return Path(cache_dir) / f"{safe}.csv"


def _load_cache(cache_dir: str, ticker: str, max_age_hours: int) -> pd.DataFrame | None:
    path = _cache_path(cache_dir, ticker)
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _save_cache(cache_dir: str, ticker: str, df: pd.DataFrame) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(cache_dir, ticker))


def _fetch_yfinance(ticker: str, start: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if df is None or df.empty:
                raise PriceFetchError(f"yfinance returned no data for {ticker}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            out = df[["Close"]].rename(columns={"Close": "close"})
            out.index.name = "date"
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("yfinance attempt %d/%d failed for %s: %s", attempt + 1, retries, ticker, e)
            time.sleep(2 ** attempt)
    raise PriceFetchError(f"yfinance exhausted retries for {ticker}: {last_err}")


def _stooq_symbol(ticker: str) -> str:
    # stooq uses .us suffix for US equities/ETFs; crypto pairs use e.g. btcusd
    if ticker.endswith("-USD"):
        return ticker.replace("-USD", "").lower() + "usd"
    return ticker.lower() + ".us"


def _fetch_stooq(ticker: str, start: str, retries: int = 3) -> pd.DataFrame:
    import requests
    from io import StringIO

    symbol = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            text = resp.text
            if "Date,Open" not in text:
                raise PriceFetchError(f"stooq returned no usable data for {ticker} ({symbol})")
            df = pd.read_csv(StringIO(text), parse_dates=["Date"]).set_index("Date")
            df.index.name = "date"
            out = df[["Close"]].rename(columns={"Close": "close"})
            out = out[out.index >= pd.Timestamp(start)]
            if out.empty:
                raise PriceFetchError(f"stooq data for {ticker} empty after start filter")
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("stooq attempt %d/%d failed for %s: %s", attempt + 1, retries, ticker, e)
            time.sleep(2 ** attempt)
    raise PriceFetchError(f"stooq exhausted retries for {ticker}: {last_err}")


def fetch_prices(ticker: str, start: str, cache_dir: str = "cache", cache_max_age_hours: int = 20) -> pd.DataFrame:
    """Fetch daily close prices for `ticker` since `start`, using cache then a fallback chain."""
    cached = _load_cache(cache_dir, ticker, cache_max_age_hours)
    if cached is not None:
        logger.info("cache hit for %s", ticker)
        return cached

    errors = []
    for fetcher_name, fetcher in (("yfinance", _fetch_yfinance), ("stooq", _fetch_stooq)):
        try:
            df = fetcher(ticker, start)
            _save_cache(cache_dir, ticker, df)
            logger.info("fetched %s via %s (%d rows)", ticker, fetcher_name, len(df))
            return df
        except PriceFetchError as e:
            errors.append(f"{fetcher_name}: {e}")
            continue

    raise PriceFetchError(f"all sources failed for {ticker}: {' | '.join(errors)}")
