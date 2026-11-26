"""E11 review: execution delay, beta-residual target, seed ensemble, Pearson (Kaggle metric) vs reversal baseline."""
import sys, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import research.hourly as H

BASKET = ["BTC","ETH","MU","XAU","XAG","COPPER","NVDA","QQQ","COIN","MSTR","CL","SPY","CRWV","AMD","AVGO","CEG","FCX"]
rows = []
_lb = H.load_bars
for uni_name, uni, minr in (("alt_pool_E7", None, 6000), ("target_basket", BASKET, 1000)):
    H.UNIVERSE = uni
    H.load_bars = lambda min_rows=6000, _m=minr: _lb(_m)
    for h in (6, 24):
        for delay in (0, 1):
            for tgt in ("demean", "beta"):
                t = time.time()
                P = H.build(h, delay=delay, target=tgt)
                feats = [c for c in P.columns if c not in ("date","coin","tgt","fwd","pred","cid")]
                preds = []
                for seed in (0, 1):
                    W = H.walk(P, h, feats, seed=seed, sub=2 if uni else 3, min_hours=24*150 if uni else 24*180)
                    preds.append(W["pred"])
                W["pred"] = np.mean(preds, axis=0)
                r, ic, pnl = H.evaluate(W, h, k=2 if uni else 5, cost_leg=0.0006)
                d = W.dropna(subset=["pred", "tgt"])
                pear = d.groupby("date").apply(lambda g: np.corrcoef(g.pred, g.tgt)[0, 1] if len(g) > 6 else np.nan).dropna().iloc[::h]
                base = P.dropna(subset=["tgt", "r1"])
                base = base[base.date.isin(d.date.unique())]
                rev = base.groupby("date").apply(lambda g: stats.spearmanr(-g.r1, g.tgt)[0] if len(g) > 6 else np.nan).dropna().iloc[::h]
                rows.append(dict(universe=uni_name, h=h, delay=delay, target=tgt, ic=r["mean_ic"], ic_t=r["ic_t"], pearson=pear.mean(),
                                 pearson_t=pear.mean()/pear.std()*np.sqrt(len(pear)), reversal_ic=rev.mean(), reversal_t=rev.mean()/rev.std()*np.sqrt(len(rev)),
                                 gross_bps=r["gross_per_rebal_bps"], net_bps=r["net_per_rebal_bps"], sharpe_net=r["ann_sharpe_net"], n=r["n_rebal"]))
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, round(time.time()-t), "s", flush=True)
pd.DataFrame(rows).to_csv("reports/e11_review.csv", index=False)
print("DONE")
