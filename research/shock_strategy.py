"""E4: tradable test of alt post-surge continuation: long alt / short BTC, enter next close, hold h days, costs in."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_connectors.price_connector import fetch_prices
from research.replicate_coins import COINS

COST = 0.0040  # 40 bps round trip per leg pair (alt + BTC hedge), conservative for liquid alts


def events_and_returns(h: int, thresh: float = 2.5, start="2018-01-01") -> pd.DataFrame:
    btc = fetch_prices("BTC-USD", start)["close"]
    out = []
    for c in COINS:
        try:
            p = fetch_prices(f"{c}-USD", start)["close"]
        except Exception:
            continue
        p = p[~p.index.duplicated()].asfreq("D").ffill(limit=3).dropna()
        if len(p) < 700:
            continue
        r = p.pct_change()
        z5 = (p / p.shift(5) - 1) / (r.rolling(20).std().shift(1) * np.sqrt(5))
        b = btc.reindex(p.index).ffill()
        last = None
        for d in z5[z5 > thresh].index:
            if last is not None and (d - last).days < h:
                continue
            i = p.index.get_loc(d)
            if i + 1 + h >= len(p):
                continue
            e, x = p.index[i + 1], p.index[i + 1 + h]  # enter NEXT close: no same-bar lookahead
            alt = p[x] / p[e] - 1
            hedge = b[x] / b[e] - 1
            out.append({"coin": c, "signal_date": d, "year": d.year, "alt": alt, "btc": hedge, "net": alt - hedge - COST})
            last = d
    return pd.DataFrame(out).sort_values("signal_date")


def summarize(df: pd.DataFrame, h: int) -> dict:
    # collapse same-day events across coins into one equal-weight basket trade to remove clustering
    g = df.groupby("signal_date")["net"].mean()
    per_trade = g.mean()
    sharpe = per_trade / g.std() * np.sqrt(365 / h) if g.std() > 0 else np.nan
    return {"h": h, "coin_events": len(df), "basket_days": len(g), "net_mean_per_trade": per_trade,
            "hit": float((g > 0).mean()), "ann_sharpe_approx": sharpe,
            "pos_years": int((df.groupby("year")["net"].mean() > 0).sum()), "n_years": df["year"].nunique(),
            "ex_top_year_mean": df[df.year != df.groupby("year")["net"].sum().idxmax()]["net"].mean()}
