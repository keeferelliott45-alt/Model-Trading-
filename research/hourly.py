"""E7: Kaggle/G-Research-style pooled HOURLY cross-sectional model on 45 perps (450k rows), market-neutral target, purged walk-forward."""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
BARS = Path(__file__).resolve().parents[1] / "data" / "bars"


def load_bars(min_rows=6000):
    C, V, H, L, O = {}, {}, {}, {}, {}
    for f in glob.glob(str(BARS / "*.parquet")):
        d = pd.read_parquet(f)
        s = Path(f).stem.replace("USDT", "")
        if len(d) < min_rows:
            continue
        d = d[~d.index.duplicated()]
        full = pd.date_range(d.index.min(), d.index.max(), freq="h")
        if len(d) / len(full) < 0.97:  # drop session-based (stock/commodity) perps with gaps
            continue
        C[s], V[s], H[s], L[s], O[s] = d.c, d.qv if "qv" in d else d.v * d.c, d.h, d.l, d.o
    f = lambda x: pd.DataFrame(x).sort_index()
    return f(C), f(V), f(H), f(L), f(O)


def build(h: int, funding: bool = False):
    C, V, H, L, O = load_bars()
    idx = C.index
    r1 = np.log(C).diff()
    F = {}
    for k in (1, 2, 3, 6, 12, 24, 48, 96, 168):
        F[f"r{k}"] = np.log(C) - np.log(C.shift(k))
    for k in (24, 168):
        F[f"vol{k}"] = r1.rolling(k, min_periods=int(k * .8)).std()
    F["volratio"] = F["vol24"] / F["vol168"]
    F["range1"] = (H - L) / C
    F["range24"] = (H.rolling(24).max() - L.rolling(24).min()) / C
    F["body"] = (C - O) / (H - L + 1e-12)
    F["upwick"] = (H - np.maximum(C, O)) / (H - L + 1e-12)
    F["dvol_z"] = (np.log1p(V) - np.log1p(V).rolling(168, min_periods=130).mean()) / np.log1p(V).rolling(168, min_periods=130).std()
    F["dvol_chg"] = np.log1p(V.rolling(6).mean()) - np.log1p(V.rolling(48).mean())
    F["z24"] = F["r24"] / (F["vol168"] * np.sqrt(24))
    F["hi_dist48"] = C / H.rolling(48).max() - 1
    F["lo_dist48"] = C / L.rolling(48).min() - 1
    mkt = r1.mean(axis=1, skipna=True)
    btc = r1["BTC"] if "BTC" in r1 else mkt
    beta = r1.rolling(336, min_periods=250).cov(btc).div(btc.rolling(336, min_periods=250).var(), axis=0)
    F["beta"] = beta
    for k in (6, 24):
        F[f"mkt_r{k}"] = pd.DataFrame({c: mkt.rolling(k).sum() for c in C.columns})
        F[f"idio{k}"] = F[f"r{k}"] - beta * F[f"mkt_r{k}"]
    F["disp24"] = pd.DataFrame({c: r1.rolling(24, min_periods=18).std().mean(axis=1) for c in C.columns})
    for k in ("r1", "r6", "r24", "r96", "z24", "vol24", "dvol_z", "idio24", "range24"):
        F[f"{k}_rk"] = F[k].rank(axis=1, pct=True)
    hr = idx.hour.to_numpy()
    F["h_sin"] = pd.DataFrame({c: np.sin(2 * np.pi * hr / 24) for c in C.columns}, index=idx)
    F["h_cos"] = pd.DataFrame({c: np.cos(2 * np.pi * hr / 24) for c in C.columns}, index=idx)
    if funding:
        FD = {}
        for f in glob.glob(str(BARS.parent / "funding" / "*.parquet")):
            c = Path(f).stem
            if c in C.columns:
                FD[c] = pd.read_parquet(f).fundingRate
        FD = pd.DataFrame(FD).reindex(idx.union(pd.DataFrame(FD).index)).sort_index().ffill(limit=12).reindex(idx).reindex(columns=C.columns) * 1e4
        F["fund"] = FD
        F["fund_z"] = (FD - FD.rolling(336, min_periods=200).mean()) / FD.rolling(336, min_periods=200).std()
        F["fund_chg24"] = FD - FD.shift(24)
        F["fund_rk"] = FD.rank(axis=1, pct=True)
        F["fund_mkt"] = pd.DataFrame({c: FD.mean(axis=1) for c in C.columns}, index=idx)
        F["fund_x_r24"] = F["fund_rk"] * F["r24_rk"]
    fwd = np.log(C.shift(-h)) - np.log(C)
    tgt = fwd.sub(fwd.mean(axis=1), axis=0)
    rows = []
    for c in C.columns:
        d = pd.DataFrame({k: v[c] for k, v in F.items()})
        d["coin"], d["tgt"], d["fwd"] = c, tgt[c], fwd[c]
        rows.append(d)
    P = pd.concat(rows).rename_axis("date").reset_index()
    P["cid"] = P.coin.astype("category").cat.codes
    return P


def walk(P, h, feats, min_hours=24 * 180, step_h=24 * 14, sub=3, seed=0, noise=0.05):
    hours = np.sort(P.date.unique())
    P = P.copy(); P["pred"] = np.nan
    for s in range(min_hours, len(hours), step_h):
        cut = hours[max(s - h - 1, 0)]
        tr = P[(P.date <= cut) & P.tgt.notna()]
        tr = tr[(tr.date.dt.hour % sub == 0)].dropna(subset=feats, thresh=int(len(feats) * 0.8))
        te = P[(P.date >= hours[s]) & (P.date < hours[min(s + step_h, len(hours) - 1)])]
        if len(tr) < 20000 or te.empty:
            continue
        lo, hi = tr.tgt.quantile([.005, .995])
        X = tr[feats].astype(float)
        if noise:
            X = X + np.random.default_rng(seed + s).normal(0, noise, X.shape) * X.std().to_numpy()
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.6,
                              subsample_freq=1, colsample_bytree=0.6, reg_lambda=20, random_state=seed, verbose=-1, n_jobs=4)
        m.fit(X, tr.tgt.clip(lo, hi))
        P.loc[te.index, "pred"] = m.predict(te[feats].astype(float))
    return P


def evaluate(P, h, k=5, cost_leg=0.0012):
    d = P.dropna(subset=["pred", "tgt"])
    ic = d.groupby("date").apply(lambda g: stats.spearmanr(g.pred, g.tgt)[0] if len(g) > 12 else np.nan).dropna()
    non = ic.iloc[::h]
    t = non.mean() / non.std() * np.sqrt(len(non))
    pnl = []
    for dt in sorted(d.date.unique())[::h]:
        g = d[d.date == dt].sort_values("pred")
        if len(g) < 2 * k + 4:
            continue
        pnl.append((dt, g.fwd.iloc[-k:].mean() - g.fwd.iloc[:k].mean() - 2 * cost_leg))
    pnl = pd.Series(dict(pnl))
    gross = pnl + 2 * cost_leg
    out = {"h": h, "mean_ic": ic.mean(), "ic_t": t, "ic_pos": float((non > 0).mean()), "n_rebal": len(pnl),
           "gross_per_rebal_bps": gross.mean() * 1e4, "net_per_rebal_bps": pnl.mean() * 1e4,
           "ann_sharpe_net": pnl.mean() / pnl.std() * np.sqrt(8760 / h) if pnl.std() > 0 else np.nan}
    return out, ic, pnl
