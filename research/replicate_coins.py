"""E3: out-of-sample replication of crypto post-shock continuation on coins NOT used in discovery (BTC/ETH)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from data_connectors.price_connector import fetch_prices

COINS = ["SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX", "LTC", "BCH", "XLM", "TRX", "DOT", "ATOM", "ETC", "XMR", "UNI", "AAVE", "NEAR", "ALGO", "FIL"]
HORIZONS = (5, 10, 20)
EVENTS = ("surge_5d_z2.5", "big_dn_1d_z3", "crash_5d_z2.5")


def run(start="2018-01-01", exp="E3"):
    btc = fetch_prices("BTC-USD", start)["close"]
    rows, used = [], []
    per = {}
    for c in COINS:
        try:
            p = fetch_prices(f"{c}-USD", start)["close"]
        except Exception:
            continue
        p = p[~p.index.duplicated()].asfreq("D").ffill(limit=3).dropna()
        if len(p) < 700:
            continue
        used.append(c)
        r = p.pct_change()
        vol = r.rolling(20).std().shift(1)
        z1 = r / r.rolling(60).std().shift(1)
        z5 = (p / p.shift(5) - 1) / (vol * np.sqrt(5))
        ev = {"surge_5d_z2.5": z5 > 2.5, "big_dn_1d_z3": z1 < -3, "crash_5d_z2.5": z5 < -2.5}
        b = btc.reindex(p.index).ffill()
        for h in HORIZONS:
            fwd = p.shift(-h) / p - 1
            fwd_x = fwd - (b.shift(-h) / b - 1)  # BTC-neutral
            for en, m in ev.items():
                idx, last = [], None
                for d in m[m.fillna(False)].index:
                    if last is None or (d - last).days >= h:
                        idx.append(d); last = d
                idx = [d for d in idx if pd.notna(fwd.get(d))]
                if len(idx) >= 5:
                    per[(c, en, h)] = (fwd.loc[idx].mean() - fwd.mean(), fwd_x.loc[idx].mean() - fwd_x.mean(), len(idx))
    for en in EVENTS:
        for h in HORIZONS:
            v = [(c, *per[(c, en, h)]) for c in used if (c, en, h) in per]
            if len(v) < 6:
                continue
            e, ex, n = np.array([x[1] for x in v]), np.array([x[2] for x in v]), sum(x[3] for x in v)
            t, p = stats.ttest_1samp(ex, 0)  # across-coin test of BTC-neutral edge
            rows.append({"exp": exp, "asset": f"{len(v)} alt coins", "event": en, "condition": "btc_neutral_across_coins",
                         "horizon": h, "n": n, "mean_fwd": e.mean(), "edge": ex.mean(), "hit_rate": float((ex > 0).mean()),
                         "t": t, "p_value": p})
    return pd.DataFrame(rows), used
