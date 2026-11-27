"""E13: research-driven blend on the liquid perp pool. 24h horizon, 1-bar execution delay, walk-forward, costs.
Components: LightGBM (Kaggle/G-Research), Ridge (linear, robust), and literature factors with FIXED signs:
  momentum (Liu-Tsyvinski-Wu: 7d return skipping last day), short-term reversal (Peng-Shams), trend (CTREND-style MA signals),
  low volatility, illiquidity proxy (low dollar volume). Blends: equal-weight, shrunk trailing-ICIR, ML stack (LGBM+Ridge), + vol scaling."""
import sys, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
import research.hourly as H

HZ, DELAY, K, COST = 24, 1, 5, 0.0006


def xs_z(s: pd.Series, dates: pd.Series) -> pd.Series:
    r = s.groupby(dates).rank(pct=True) - 0.5
    return r / r.groupby(dates).transform("std")


def ridge_walk(P, feats, min_hours=24 * 180, step_h=24 * 14):
    hours = np.sort(P.date.unique()); out = pd.Series(np.nan, index=P.index)
    Z = P[feats].groupby(P.date).rank(pct=True) - 0.5  # rank-gauss-ish, robust to outliers
    Z = Z.fillna(0)
    for s in range(min_hours, len(hours), step_h):
        cut = hours[max(s - HZ - DELAY - 1, 0)]
        tr = (P.date <= cut) & P.tgt.notna() & (P.date.dt.hour % 3 == 0)
        te = (P.date >= hours[s]) & (P.date < hours[min(s + step_h, len(hours) - 1)])
        if tr.sum() < 20000 or not te.any():
            continue
        y = P.loc[tr, "tgt"].clip(*P.loc[tr, "tgt"].quantile([.005, .995]))
        m = Ridge(alpha=1000.0).fit(Z[tr], y)
        out[te] = m.predict(Z[te])
    return out


def main():
    t0 = time.time()
    P = H.build(HZ, delay=DELAY, target="demean")
    feats = [c for c in P.columns if c not in ("date", "coin", "tgt", "fwd", "pred", "cid")]
    d = P.date
    comp = {}
    lg = [H.walk(P, HZ, feats, seed=s)["pred"] for s in (0, 1, 2)]
    comp["lgbm"] = pd.Series(np.mean(lg, axis=0), index=P.index)
    comp["ridge"] = ridge_walk(P, feats)
    comp["f_momentum"] = P["r168"] - P["r24"]
    comp["f_reversal"] = -P["r24"]
    comp["f_trend"] = sum(np.sign(P[c]) for c in ("r24", "r48", "r96", "r168")) / 4
    comp["f_lowvol"] = -P["vol168"]
    comp["f_illiq"] = -P["dvol_z"]
    print(f"components built {time.time()-t0:.0f}s", flush=True)
    live = comp["lgbm"].notna() & comp["ridge"].notna()
    D = P.loc[live, ["date", "coin", "tgt", "fwd"]].copy()
    for k, v in comp.items():
        D[k] = xs_z(v[live], d[live])
    D = D.dropna(subset=["tgt"])
    stamps = np.sort(D.date.unique())[::HZ]  # one rebalance per day
    D = D[D.date.isin(stamps)].copy()
    names = list(comp)
    ic = pd.DataFrame({k: D.groupby("date").apply(lambda g, k=k: stats.spearmanr(g[k], g.tgt)[0]) for k in names})
    # trailing ICIR weights: IC of day t is known only after HZ+DELAY hours -> shift 2 daily steps
    roll = ic.shift(2).rolling(60, min_periods=20)
    icir = (roll.mean() / roll.std()).clip(lower=0).fillna(0)
    w = icir.div(icir.sum(axis=1).replace(0, np.nan), axis=0).fillna(1 / len(names))
    w = 0.5 * w + 0.5 / len(names)  # shrink toward equal weight
    D["blend_equal"] = D[names].mean(axis=1)
    D["blend_factors_only"] = D[[n for n in names if n.startswith("f_")]].mean(axis=1)
    D["blend_ml_stack"] = D[["lgbm", "ridge"]].mean(axis=1)
    D["blend_icir"] = sum(D[n] * D.date.map(w[n]) for n in names)
    signals = names + ["blend_equal", "blend_factors_only", "blend_ml_stack", "blend_icir"]
    rows, pnls = [], {}
    for s in signals:
        icd = D.groupby("date").apply(lambda g: stats.spearmanr(g[s], g.tgt)[0])
        prev, pnl, turn = pd.Series(dtype=float), [], []
        for dt, g in D.groupby("date"):
            g = g.sort_values(s)
            wt = pd.Series(0.0, index=g.coin)
            wt[g.coin.iloc[-K:].values] = 1 / K; wt[g.coin.iloc[:K].values] = -1 / K
            tv = wt.sub(prev, fill_value=0).abs().sum(); prev = wt
            pnl.append((dt, float((wt.values * g.fwd.values).sum()) - COST * tv)); turn.append(tv)
        x = pd.Series(dict(pnl)); pnls[s] = x
        half = x.index[len(x) // 2]
        sh = lambda z: z.mean() / z.std() * np.sqrt(365)
        rows.append(dict(signal=s, ic=icd.mean(), ic_t=icd.mean() / icd.std() * np.sqrt(len(icd)), net_bps=x.mean() * 1e4,
                         turnover=np.mean(turn), sharpe=sh(x), sharpe_h1=sh(x[x.index < half]), sharpe_h2=sh(x[x.index >= half]),
                         maxdd=float((x.cumsum() - x.cumsum().cummax()).min())))
    # volatility-scaled version of the best ex-ante choice (blend_icir): target 1% daily vol using trailing 30d pnl vol (known ex-ante)
    x = pnls["blend_icir"]; scale = (0.01 / x.shift(1).rolling(30, min_periods=10).std()).clip(0.25, 3).fillna(1)
    xv = x * scale; half = xv.index[len(xv) // 2]; sh = lambda z: z.mean() / z.std() * np.sqrt(365)
    rows.append(dict(signal="blend_icir_volscaled", ic=np.nan, ic_t=np.nan, net_bps=xv.mean() * 1e4, turnover=np.nan, sharpe=sh(xv),
                     sharpe_h1=sh(xv[xv.index < half]), sharpe_h2=sh(xv[xv.index >= half]), maxdd=float((xv.cumsum() - xv.cumsum().cummax()).min())))
    R = pd.DataFrame(rows)
    R.to_csv("reports/e13_blend.csv", index=False)
    ic.corr().round(2).to_csv("reports/e13_component_ic_corr.csv")
    pd.DataFrame(pnls).to_csv("reports/e13_daily_pnl.csv")
    pd.set_option("display.width", 220)
    print(R.round(3).to_string(index=False))
    print("\ncomponent IC correlation:\n", ic.corr().round(2).to_string())
    print("\nlast ICIR weights:", w.iloc[-1].round(3).to_dict())
    print(f"DONE {time.time()-t0:.0f}s")


main()
