"""E1: 'after the move' event study for BTC/ETH conditioned on whale/exchange flow, valuation, funding, sentiment, cycle."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = [5, 10, 20, 40, 60]


def _first_cross(cond: pd.Series, cooldown: int = 20) -> pd.Series:
    out, last = np.zeros(len(cond), dtype=bool), -10**9
    for i, v in enumerate(cond.fillna(False).to_numpy()):
        if v and i - last >= cooldown:
            out[i] = True
            last = i
    return pd.Series(out, index=cond.index)


def base_events(X: pd.DataFrame, a: str) -> dict[str, pd.Series]:
    z5 = X[f"{a}_ret5"] / (X[f"{a}_vol20"] * np.sqrt(5))
    dd = X[f"{a}_dd_ath"]
    return {
        "big_up_1d_z3": X[f"{a}_move_z"] > 3,
        "big_dn_1d_z3": X[f"{a}_move_z"] < -3,
        "surge_5d_z2.5": z5 > 2.5,
        "crash_5d_z2.5": z5 < -2.5,
        "dd_cross_-30": _first_cross((dd < -0.30), 60),
        "dd_cross_-50": _first_cross((dd < -0.50), 120),
        "dd_cross_-70": _first_cross((dd < -0.70), 200),
        "ath_break": _first_cross(dd.ge(-1e-9) & dd.shift(10).lt(-0.05), 30),
        "extended_ma200_+60%": _first_cross(X[f"{a}_ma200"] > 0.6, 60),
        "below_ma200_-30%": _first_cross(X[f"{a}_ma200"] < -0.3, 60),
    }


def conditioners(X: pd.DataFrame, a: str) -> dict[str, pd.Series]:
    c = {"none": pd.Series(True, index=X.index)}
    nf, mv = X[f"{a}_netflow7_z"], X[f"{a}_mvrv_z"]
    c["exch_inflow_z>1 (distribution)"] = nf > 1
    c["exch_outflow_z<-1 (accumulation)"] = nf < -1
    c["flowin_spike>1.5"] = X[f"{a}_flowin_spike"] > 1.5
    c["mvrv_z>1 (hot)"] = mv > 1
    c["mvrv_z<-0.5 (cheap)"] = mv < -0.5
    fz = X.get(f"funding_{a}_z")
    if fz is not None:
        c["funding_z>1 (crowded long)"] = fz > 1
        c["funding_z<-1 (crowded short)"] = fz < -1
    fg = X["fear_greed_z"]
    c["fear_greed_z<-1 (fear)"] = fg < -1
    c["fear_greed_z>1 (greed)"] = fg > 1
    ph = X["halving_phase"]
    for lo, hi, nm in ((0, .25, "yr1 post-halving"), (.25, .5, "yr2"), (.5, .75, "yr3"), (.75, 1.01, "yr4 pre-halving")):
        c[f"cycle {nm}"] = (ph >= lo) & (ph < hi)
    c["stress hy_spread_z>1"] = X["hy_spread_z"] > 1
    c["geopol gpr_z>1"] = X["gpr_z"] > 1 if "gpr_z" in X else pd.Series(False, index=X.index)
    return c


def run_event_study(X: pd.DataFrame, prices: pd.DataFrame, assets=("BTC-USD", "ETH-USD"), min_n: int = 10, exp: str = "E1") -> pd.DataFrame:
    rows = []
    for a in assets:
        p = prices[a]
        base = base_events(X, a)
        key = a.split("-")[0].lower()
        cond = conditioners(X.rename(columns={f"{key}_netflow7_z": f"{a}_netflow7_z", f"{key}_mvrv_z": f"{a}_mvrv_z",
                                              f"{key}_flowin_spike": f"{a}_flowin_spike", f"funding_{key}_z": f"funding_{a}_z"}), a)
        for h in HORIZONS:
            fwd = (p.shift(-h) / p - 1).to_numpy()
            valid = ~np.isnan(fwd)
            uncond = fwd[valid]
            for (bn, bmask), (cn, cmask) in itertools.product(base.items(), cond.items()):
                m = (bmask & cmask).fillna(False).to_numpy() & valid
                idx, last = [], -10**9
                for i in np.where(m)[0]:
                    if i - last >= h:
                        idx.append(i)
                        last = i
                if len(idx) < min_n:
                    continue
                f = fwd[idx]
                t, pv = stats.ttest_ind(f, uncond, equal_var=False)
                half = len(p) // 2
                f1 = fwd[[i for i in idx if i < half]]
                f2 = fwd[[i for i in idx if i >= half]]
                rows.append({"exp": exp, "asset": a, "event": bn, "condition": cn, "horizon": h, "n": len(idx),
                             "mean_fwd": f.mean(), "uncond": uncond.mean(), "edge": f.mean() - uncond.mean(),
                             "hit_rate": float((f > 0).mean()), "t": t, "p_value": pv,
                             "edge_h1": (f1.mean() - uncond.mean()) if len(f1) >= 3 else np.nan,
                             "edge_h2": (f2.mean() - uncond.mean()) if len(f2) >= 3 else np.nan})
    return pd.DataFrame(rows)
