"""E8: BTC+ETH-only predictor with feature-group ablation (price / whale-exchange flow / cycle / derivs / macro / sentiment)."""
from __future__ import annotations

import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def stack(X: pd.DataFrame, P: pd.DataFrame, h: int):
    rows = []
    for a, k in (("BTC-USD", "btc"), ("ETH-USD", "eth")):
        d = pd.DataFrame(index=X.index)
        for n in ("ret1", "ret5", "ret20", "vol20", "dd_ath", "ma200", "move_z"):
            d[f"price_{n}"] = X[f"{a}_{n}"]
        d["price_ret60"] = P[a] / P[a].shift(60) - 1
        d["price_ret120"] = P[a] / P[a].shift(120) - 1
        for n in ("netflow7_z", "netflow_pct_sply", "exsupply_d30", "flowin_spike", "mvrv_z", "puell", "active_z"):
            d[f"chain_{n}"] = X[f"{k}_{n}"]
        d["deriv_funding_z"] = X[f"funding_{k}_z"]
        d["deriv_funding_d5"] = X[f"funding_{k}_d5"]
        d["deriv_dvol_z"] = X[f"dvol_{k}_z"]
        d["deriv_dvol_d5"] = X[f"dvol_{k}_d5"]
        d["deriv_ivrv"] = X[f"dvol_{k}"] if f"dvol_{k}" in X else X[f"dvol_{k}_z"]
        d["cycle_days"] = X["days_since_halving"]; d["cycle_phase"] = X["halving_phase"]
        for n in ("fear_greed_z", "fear_greed_d5", "stablecoins_d20", "stablecoins_z"):
            d["sent_" + n] = X[n]
        for n in ("vix_z", "dxy_z", "dxy_d20", "hy_spread_z", "us10y_d20", "real10y_d20", "spx_d20", "ndx_d20", "gpr_z", "epu_daily_z", "fin_stress_z", "usd_broad_d20"):
            if n in X:
                d["macro_" + n] = X[n]
        other = "ETH-USD" if a == "BTC-USD" else "BTC-USD"
        d["price_rel_other20"] = X[f"{a}_ret20"] - X[f"{other}_ret20"]
        d["asset"] = 0 if k == "btc" else 1
        d["fwd"] = np.log(P[a].shift(-h) / P[a])
        d["coin"] = k
        d["date"] = d.index
        rows.append(d.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def walk(D, cols, h, min_days=900, step=63, seed=0):
    dates = np.sort(D.date.unique()); D = D.copy(); D["pred"] = np.nan
    for s in range(min_days, len(dates), step):
        cut = dates[max(s - h - 1, 0)]
        tr = D[(D.date <= cut)].dropna(subset=["fwd"])
        tr = tr[tr[cols].notna().mean(axis=1) > 0.5]
        te = D[(D.date >= dates[s]) & (D.date < dates[min(s + step, len(dates) - 1)])]
        if len(tr) < 600 or te.empty:
            continue
        lo, hi = tr.fwd.quantile([.02, .98])
        m = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.03, num_leaves=5, min_child_samples=60, subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.6, reg_lambda=10, random_state=seed, verbose=-1, n_jobs=4)
        m.fit(tr[cols], tr.fwd.clip(lo, hi))
        D.loc[te.index, "pred"] = m.predict(te[cols])
    return D


def evaluate(D, h, cost=0.0010, n_perm=400, seed=0):
    d = D.dropna(subset=["pred", "fwd"]).sort_values("date")
    out = {}
    res_all = []
    for coin, g in d.groupby("coin"):
        g = g.iloc[::h]
        ic = stats.spearmanr(g.pred, g.fwd)[0]
        rng = np.random.default_rng(seed)
        null = [stats.spearmanr(np.roll(g.pred.to_numpy(), rng.integers(3, len(g) - 3)), g.fwd)[0] for _ in range(n_perm)]
        p = (1 + sum(abs(x) >= abs(ic) for x in null)) / (1 + n_perm)
        pos = np.sign(g.pred - g.pred.expanding(min_periods=10).median().shift(1)).fillna(0)  # no look-ahead centering
        prev = pos.shift(1).fillna(0)
        net = pos * g.fwd - cost * (pos - prev).abs()
        res_all.append(net.rename(coin))
        out[coin] = {"n": len(g), "ic": ic, "p_perm": p, "hit": float((np.sign(g.pred) == np.sign(g.fwd)).mean()),
                     "strat_mean_bps": net.mean() * 1e4, "bh_mean_bps": g.fwd.mean() * 1e4,
                     "strat_sharpe": net.mean() / net.std() * np.sqrt(365 / h), "bh_sharpe": g.fwd.mean() / g.fwd.std() * np.sqrt(365 / h)}
    return out
