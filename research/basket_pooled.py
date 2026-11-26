"""E9: pooled daily model over the target basket only (BTC, ETH, CRWV, MU, AMD, AVGO, CEG, FCX, GLD, SLV, DLR). Feature-group ablation."""
from __future__ import annotations

import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ASSETS = ["BTC-USD", "ETH-USD", "CRWV", "MU", "AMD", "AVGO", "CEG", "FCX", "GLD", "SLV", "DLR"]
CLASS = {"BTC-USD": 0, "ETH-USD": 0, "CRWV": 1, "MU": 1, "AMD": 1, "AVGO": 1, "CEG": 1, "FCX": 2, "GLD": 3, "SLV": 3, "DLR": 1}
MACRO = ["vix_z", "vix_d5", "dxy_z", "dxy_d20", "hy_spread_z", "us10y_d20", "real10y_d20", "curve_z", "spx_d20", "ndx_d20", "semis_d20", "gpr_z", "epu_daily_z",
         "fin_stress_z", "usd_broad_d20", "oil_d20", "copper_d20", "breakeven10_d20", "defense_d20", "uranium_d20", "gold_miners_d20"]
CRYPTO = ["fear_greed_z", "fear_greed_d5", "stablecoins_d20", "halving_phase", "btc_netflow7_z", "btc_mvrv_z", "btc_puell", "funding_btc_z", "dvol_btc_z",
          "eth_netflow7_z", "eth_mvrv_z", "funding_eth_z", "dvol_eth_z", "cot_gold_z", "cot_silver_z", "cot_copper_z"]


def stack(X, P, h):
    rows = []
    fwd = np.log(P.shift(-h) / P)
    mkt = fwd[ASSETS].mean(axis=1)
    rets = {a: {k: X[f"{a}_{k}"] for k in ("ret1", "ret5", "ret20", "vol20", "dd_ath", "ma200", "move_z")} for a in ASSETS}
    xs = {k: pd.DataFrame({a: rets[a][k] for a in ASSETS}) for k in ("ret5", "ret20", "vol20", "move_z", "ret1")}
    xr = {k: v.rank(axis=1, pct=True) for k, v in xs.items()}
    for a in ASSETS:
        d = pd.DataFrame(index=X.index)
        for k, s in rets[a].items():
            d[f"price_{k}"] = s
        d["price_ret60"] = P[a] / P[a].shift(60) - 1
        for k in ("ret5", "ret20", "vol20", "move_z"):
            d[f"price_rk_{k}"] = xr[k][a]
        d["price_beta_mkt20"] = xs["ret20"][a] - xs["ret20"].mean(axis=1)
        for m in MACRO:
            if m in X: d["macro_" + m] = X[m]
        for m in CRYPTO:
            if m in X: d["crypto_" + m] = X[m]
        d["price_class"] = CLASS[a]; d["price_asset"] = ASSETS.index(a)
        d["fwd"] = fwd[a]; d["tgt_mn"] = fwd[a] - mkt; d["coin"] = a; d["date"] = X.index
        rows.append(d.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def walk(D, cols, h, target="fwd", min_days=700, step=63, seed=0):
    dates = np.sort(D.date.unique()); D = D.copy(); D["pred"] = np.nan
    for s in range(min_days, len(dates), step):
        cut = dates[max(s - h - 1, 0)]
        tr = D[(D.date <= cut)].dropna(subset=[target])
        tr = tr[tr[cols].notna().mean(axis=1) > 0.5]
        te = D[(D.date >= dates[s]) & (D.date < dates[min(s + step, len(dates) - 1)])]
        if len(tr) < 3000 or te.empty:
            continue
        lo, hi = tr[target].quantile([.02, .98])
        m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.6, reg_lambda=10, random_state=seed, verbose=-1, n_jobs=4)
        m.fit(tr[cols], tr[target].clip(lo, hi))
        D.loc[te.index, "pred"] = m.predict(te[cols])
    return D


def evaluate(D, h, target="fwd", n_perm=300, seed=0):
    d = D.dropna(subset=["pred", target]).sort_values("date")
    ics = d.groupby("date").apply(lambda g: stats.spearmanr(g.pred, g[target])[0] if len(g) >= 6 else np.nan).dropna()
    non = ics.iloc[::h]
    t = non.mean() / non.std() * np.sqrt(len(non))
    # time-series IC per asset, non-overlapping, with circular-shift permutation p
    per = {}
    for a, g in d.groupby("coin"):
        g = g.iloc[::h]
        if len(g) < 30: continue
        ic = stats.spearmanr(g.pred, g[target])[0]
        rng = np.random.default_rng(seed)
        null = [stats.spearmanr(np.roll(g.pred.to_numpy(), rng.integers(3, len(g) - 3)), g[target])[0] for _ in range(n_perm)]
        per[a] = (ic, (1 + sum(abs(x) >= abs(ic) for x in null)) / (1 + n_perm), len(g))
    return {"xs_ic": non.mean(), "xs_t": t, "xs_pos": float((non > 0).mean()), "n_dates": len(non)}, per
