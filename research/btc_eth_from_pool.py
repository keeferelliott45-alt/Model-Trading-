"""E12: train on the whole liquid perp pool (information), trade ONLY BTC/ETH. 24h, 1-bar delay, 2 seeds."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import research.hourly as H

h, delay = 24, 1
P = H.build(h, delay=delay, target="demean")
feats = [c for c in P.columns if c not in ("date","coin","tgt","fwd","pred","cid")]
preds = [H.walk(P, h, feats, seed=s)["pred"] for s in (0, 1)]
P["pred"] = np.mean(preds, axis=0)
P[["date","coin","pred","tgt","fwd"]].dropna(subset=["pred"]).to_parquet("cache/pool24_delay1.parquet")
d = P.dropna(subset=["pred","tgt"])
out = {}
for c in ("BTC","ETH"):
    g = d[d.coin == c].sort_values("date").iloc[::h]
    out[c] = (len(g), stats.spearmanr(g.pred, g.tgt)[0])
print("per-asset IC vs pool-neutral target (n, ic):", out)
w = d[d.coin.isin(["BTC","ETH"])].pivot(index="date", columns="coin", values=["pred","fwd"]).dropna().iloc[::h]
sig = np.sign(w["pred"]["ETH"] - w["pred"]["BTC"])
pair = sig * (w["fwd"]["ETH"] - w["fwd"]["BTC"])
cost = 0.0006 * 2 * (sig.diff().abs().fillna(1))  # each flip trades both legs in and out
net = pair - cost
ann = np.sqrt(8760 / h)
half = net.index[len(net)//2]
print(f"ETH/BTC pair: n={len(net)} gross {pair.mean()*1e4:.1f}bps net {net.mean()*1e4:.1f}bps hit {(net>0).mean():.2f} sharpe {net.mean()/net.std()*ann:.2f} | 1st half {net[net.index<half].mean()/net[net.index<half].std()*ann:.2f} 2nd half {net[net.index>=half].mean()/net[net.index>=half].std()*ann:.2f}")
for c in ("BTC","ETH"):
    g = d[d.coin == c].set_index("date").sort_index().iloc[::h]
    thr = g.pred.expanding(50).quantile(.5).shift(1)
    pos = np.sign(g.pred - thr).fillna(0)
    r = pos * g.tgt - 0.0006 * 2 * pos.diff().abs().fillna(0)
    print(f"{c} vs alt-market (market-neutral) n={len(r)} net {r.mean()*1e4:.1f}bps sharpe {r.mean()/r.std()*ann:.2f}")
