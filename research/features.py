"""Daily feature panel: prices, macro, geopolitics, on-chain, cycle. All point-in-time (publication-lagged)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_connectors.macro import load_all
from data_connectors.price_connector import fetch_prices
from data_connectors.funding_connector import fetch_funding

HALVINGS = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-19"])
ASSETS = ["BTC-USD", "ETH-USD", "CRWV", "MU", "AMD", "AVGO", "CEG", "FCX", "GLD", "SLV", "DLR"]


def zs(s: pd.Series, w: int = 252) -> pd.Series:
    return (s - s.rolling(w, min_periods=60).mean()) / s.rolling(w, min_periods=60).std()


def load_panel(start: str = "2018-01-01"):
    idx = pd.bdate_range(start, pd.Timestamp.today().normalize())
    px = {}
    for a in ASSETS:
        px[a] = fetch_prices(a, start)["close"]
    prices = pd.DataFrame(px).reindex(idx).ffill()
    series, failed = load_all()
    macro = pd.DataFrame({k: v.reindex(v.index.union(idx)).ffill().reindex(idx) for k, v in series.items()})
    for a, inst in (("btc", "BTC-PERPETUAL"), ("eth", "ETH-PERPETUAL")):
        f = fetch_funding(inst, "2019-01-01")
        f.index = f.index + pd.tseries.offsets.BDay(1)
        f = f[~f.index.duplicated(keep='last')]
        macro[f"funding_{a}"] = f.reindex(f.index.union(idx)).ffill().reindex(idx)
    return prices, macro, failed


def build_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    F = {}
    for c in macro.columns:
        s = macro[c]
        F[f"{c}_z"] = zs(s)
        F[f"{c}_d5"] = (s / s.shift(5) - 1) if (s > 0).all() else s - s.shift(5)
        F[f"{c}_d20"] = (s / s.shift(20) - 1) if (s > 0).all() else s - s.shift(20)
    for a in ("btc", "eth"):
        fin, fout, sx = macro[f"cm_{a}_FlowInExUSD"], macro[f"cm_{a}_FlowOutExUSD"], macro[f"cm_{a}_SplyExUSD"]
        net = (fin - fout).rolling(7).sum()
        F[f"{a}_netflow7_z"] = zs(net, 365)
        F[f"{a}_netflow_pct_sply"] = net / sx
        F[f"{a}_exsupply_d30"] = sx / sx.shift(30) - 1
        F[f"{a}_flowin_spike"] = zs(fin.rolling(3).sum(), 180)
        F[f"{a}_mvrv_z"] = zs(macro[f"cm_{a}_CapMVRVCur"], 730)
        iss = macro[f"cm_{a}_IssTotUSD"]
        F[f"{a}_puell"] = iss / iss.rolling(365, min_periods=200).mean()
        F[f"{a}_active_z"] = zs(macro[f"cm_{a}_AdrActCnt"], 365)
    for a in prices.columns:
        p = prices[a]
        r = p.pct_change()
        F[f"{a}_ret1"] = r
        F[f"{a}_ret5"] = p / p.shift(5) - 1
        F[f"{a}_ret20"] = p / p.shift(20) - 1
        F[f"{a}_vol20"] = r.rolling(20).std()
        F[f"{a}_dd_ath"] = p / p.cummax() - 1
        F[f"{a}_ma200"] = p / p.rolling(200, min_periods=120).mean() - 1
        F[f"{a}_move_z"] = r / r.rolling(60).std().shift(1)
    # cycle
    days = pd.Series(np.nan, index=prices.index)
    for i, d in enumerate(prices.index):
        prev = HALVINGS[HALVINGS <= d]
        days.iloc[i] = (d - prev[-1]).days
    F["days_since_halving"] = days
    F["halving_phase"] = (days / 1461).clip(0, 1)
    F["ex_spx_rel_btc"] = np.log(prices["BTC-USD"]) - np.log(macro["spx"])
    return pd.DataFrame(F, index=prices.index).replace([np.inf, -np.inf], np.nan)
