"""Backfill history (prices, macro, on-chain, funding) into the store from the connector registry."""
from __future__ import annotations

from data_connectors.funding_connector import fetch_funding
from data_connectors.macro import load_all
from data_connectors.price_connector import fetch_prices
from store.db import put_series
from research.features import ASSETS
from research.replicate_coins import COINS


def run(con) -> int:
    n = 0
    series, failed = load_all()
    for k, s in series.items():
        n += put_series(con, k, s, "macro_registry")
    for a in ASSETS + [f"{c}-USD" for c in COINS]:
        try:
            n += put_series(con, f"px_{a}", fetch_prices(a, "2016-01-01")["close"], "prices")
        except Exception:  # noqa: BLE001
            pass
    for inst in ("BTC-PERPETUAL", "ETH-PERPETUAL"):
        n += put_series(con, f"deribit_{inst}_funding_daily", fetch_funding(inst, "2019-01-01"), "deribit")
    return n
