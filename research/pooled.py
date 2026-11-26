"""E6: Kaggle-style pooled cross-sectional model (G-Research crypto pattern): one LightGBM over all coins,
market-neutral target, purged walk-forward, daily cross-sectional IC + long/short basket net of costs."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX", "LTC", "BCH", "XLM", "TRX", "DOT", "ATOM", "ETC", "XMR", "AAVE", "NEAR", "ALGO", "FIL"]


def load(start="2018-01-01"):
    d = yf.download([f"{c}-USD" for c in COINS], start=start, progress=False, auto_adjust=True)
    close, vol = d["Close"], d["Volume"]
    close.columns = [c.split("-")[0] for c in close.columns]
    vol.columns = close.columns
    return close.asfreq("D").ffill(limit=3), vol.asfreq("D").ffill(limit=3)


def panel(close, vol, macro: pd.DataFrame, h: int) -> pd.DataFrame:
    r1 = close.pct_change()
    feats = {}
    for k in (1, 3, 5, 10, 20, 60):
        feats[f"ret{k}"] = close / close.shift(k) - 1
    feats["vol20"] = r1.rolling(20).std()
    feats["vol60"] = r1.rolling(60).std()
    feats["volratio"] = feats["vol20"] / feats["vol60"]
    feats["shock_z"] = r1 / r1.rolling(60).std().shift(1)
    feats["z5"] = feats["ret5"] / (feats["vol20"].shift(1) * np.sqrt(5))
    feats["dd60"] = close / close.rolling(60).max() - 1
    feats["ma50"] = close / close.rolling(50).mean() - 1
    dv = (close * vol)
    feats["dvol_z"] = (np.log1p(dv) - np.log1p(dv).rolling(60).mean()) / np.log1p(dv).rolling(60).std()
    feats["dvol_chg5"] = np.log1p(dv.rolling(5).mean()) - np.log1p(dv.rolling(20).mean())
    btc = r1["BTC"]
    beta = r1.rolling(90).cov(btc).div(btc.rolling(90).var(), axis=0)
    feats["beta90"] = beta
    feats["resid5"] = feats["ret5"] - beta * feats["ret5"]["BTC"].to_numpy()[:, None]
    # cross-sectional ranks of key signals (Kaggle-style normalisation)
    for k in ("ret5", "ret20", "z5", "vol20", "dvol_z", "resid5"):
        feats[f"{k}_rk"] = feats[k].rank(axis=1, pct=True)
    fwd = close.shift(-h) / close - 1
    tgt = fwd.sub(fwd.mean(axis=1), axis=0)  # market-neutral target
    rows = []
    for c in close.columns:
        df = pd.DataFrame({k: v[c] for k, v in feats.items()})
        df["coin"], df["tgt"], df["fwd"] = c, tgt[c], fwd[c]
        rows.append(df)
    P = pd.concat(rows).reset_index().rename(columns={"index": "date", "Date": "date"})
    M = macro.reindex(close.index).ffill()
    P = P.merge(M, left_on="date", right_index=True, how="left")
    P["coin_id"] = P["coin"].astype("category").cat.codes
    return P


def walk(P: pd.DataFrame, h: int, feat_cols, min_days=500, step=63, seed=0, noise=0.0):
    dates = np.sort(P["date"].unique())
    P = P.copy(); P["pred"] = np.nan
    for s in range(min_days, len(dates), step):
        tr_end = dates[max(s - h - 1, 0)]  # purge/embargo of h days
        te = P[(P.date >= dates[s]) & (P.date < dates[min(s + step, len(dates) - 1)])]
        tr = P[(P.date <= tr_end)].dropna(subset=["tgt"])
        tr = tr[tr[feat_cols].notna().mean(axis=1) > 0.6]
        if len(tr) < 3000 or te.empty:
            continue
        lo, hi = tr.tgt.quantile([.01, .99])
        Xtr = tr[feat_cols].copy()
        if noise:
            Xtr = Xtr + np.random.default_rng(seed).normal(0, noise, Xtr.shape) * Xtr.std().to_numpy()  # Jane Street-style noise aug
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=200, subsample=0.7,
                              subsample_freq=1, colsample_bytree=0.6, reg_lambda=10, random_state=seed, verbose=-1, n_jobs=4)
        m.fit(Xtr, tr.tgt.clip(lo, hi))
        P.loc[te.index, "pred"] = m.predict(te[feat_cols])
    return P


def evaluate(P: pd.DataFrame, h: int, k: int = 3, cost_side: float = 0.0010):
    d = P.dropna(subset=["pred", "tgt"])
    ic = d.groupby("date").apply(lambda g: stats.spearmanr(g.pred, g.tgt)[0] if len(g) > 8 else np.nan).dropna()
    nonov = ic.iloc[::h]
    t = nonov.mean() / nonov.std() * np.sqrt(len(nonov))
    # long top-k / short bottom-k basket, rebalance every h days, cost on both legs entry+exit
    pnl = []
    for dt in sorted(d.date.unique())[::h]:
        g = d[d.date == dt].sort_values("pred")
        if len(g) < 2 * k + 2:
            continue
        r = g.fwd.iloc[-k:].mean() - g.fwd.iloc[:k].mean() - 4 * cost_side
        pnl.append((dt, r))
    pnl = pd.Series(dict(pnl))
    sh = pnl.mean() / pnl.std() * np.sqrt(365 / h) if len(pnl) > 5 else np.nan
    return {"h": h, "mean_ic": ic.mean(), "ic_t_nonoverlap": t, "ic_pos_frac": float((nonov > 0).mean()), "n_rebal": len(pnl),
            "ls_net_per_rebal": pnl.mean(), "ls_hit": float((pnl > 0).mean()), "ann_sharpe": sh}, ic, pnl
