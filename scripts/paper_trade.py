"""Daily paper-trading run for the frozen spec in paper/SPEC.md. Appends to paper/positions.csv and paper/daily.csv (committed)."""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collectors import bars
import research.hourly as H

H_, DELAY, K, COST = 24, 1, 5, 0.0006
POS, DAILY = ROOT / "paper" / "positions.csv", ROOT / "paper" / "daily.csv"
COLS = ["signal_ts", "coin", "weight", "pred", "entry_ts", "exit_ts", "entry_px", "exit_px", "ret"]


def closes():
    return pd.DataFrame({Path(f).stem.replace("USDT", ""): pd.read_parquet(f).c for f in bars.OUT.glob("*.parquet")}).sort_index()


def settle(pos: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    last = C.index.max()
    for i, r in pos.iterrows():
        e, x = pd.Timestamp(r.entry_ts), pd.Timestamp(r.exit_ts)
        if pd.isna(r.entry_px) and e <= last and e in C.index:
            pos.at[i, "entry_px"] = C.at[e, r.coin]
        if pd.isna(r.exit_px) and x <= last and x in C.index:
            pos.at[i, "exit_px"] = C.at[x, r.coin]
            pos.at[i, "ret"] = np.log(pos.at[i, "exit_px"] / pos.at[i, "entry_px"])
    return pos


def daily_table(pos: pd.DataFrame) -> pd.DataFrame:
    rows, prev = [], pd.Series(dtype=float)
    for ts, g in pos.groupby("signal_ts", sort=True):
        w = g.set_index("coin").weight
        turn = w.sub(prev, fill_value=0).abs().sum()
        prev = w
        done = g.ret.notna().all()
        gross = float((g.weight * g.ret).sum()) if done else np.nan
        rows.append({"signal_ts": ts, "settled": done, "gross": gross, "cost": COST * turn, "net": gross - COST * turn if done else np.nan,
                     "longs": ",".join(g[g.weight > 0].coin), "shorts": ",".join(g[g.weight < 0].coin)})
    d = pd.DataFrame(rows)
    d["cum_net"] = d.net.fillna(0).cumsum()
    return d


def main():
    if "--no-refresh" not in sys.argv:
        bars.run()
    pos = pd.read_csv(POS) if POS.exists() else pd.DataFrame(columns=COLS)
    C = closes()
    pos = settle(pos, C)
    P = H.build(H_, delay=DELAY, target="demean")
    last = P.date.max()
    already = len(pos) and (last - pd.Timestamp(pos.signal_ts.max())) < pd.Timedelta(hours=23)
    if not already:
        feats = [c for c in P.columns if c not in ("date", "coin", "tgt", "fwd", "pred", "cid")]
        tr = P[P.tgt.notna() & (P.date.dt.hour % 3 == 0)].dropna(subset=feats, thresh=int(len(feats) * .8))
        lo, hi = tr.tgt.quantile([.005, .995])
        now_rows = P[P.date == last].dropna(subset=feats, thresh=int(len(feats) * .8)).copy()
        import lightgbm as lgb
        preds = []
        for seed in (0, 1):
            X = tr[feats].astype(float)
            X = X + np.random.default_rng(seed).normal(0, .05, X.shape) * X.std().to_numpy()
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.6, subsample_freq=1,
                                  colsample_bytree=0.6, reg_lambda=20, random_state=seed, verbose=-1, n_jobs=4).fit(X, tr.tgt.clip(lo, hi))
            preds.append(m.predict(now_rows[feats].astype(float)))
        now_rows["pred"] = np.mean(preds, axis=0)
        r = now_rows.sort_values("pred")
        new = []
        for side, g in ((1, r.tail(K)), (-1, r.head(K))):
            for _, x in g.iterrows():
                e = last + pd.Timedelta(hours=DELAY)
                new.append({"signal_ts": str(last), "coin": x.coin, "weight": side / K, "pred": float(x.pred), "entry_ts": str(e),
                            "exit_ts": str(e + pd.Timedelta(hours=H_)), "entry_px": np.nan, "exit_px": np.nan, "ret": np.nan})
        pos = pd.concat([pos, pd.DataFrame(new)], ignore_index=True)
        print(f"signal {last}: LONG {list(r.tail(K).coin)} SHORT {list(r.head(K).coin)}")
    else:
        print(f"already have a signal within 23h of {last}; settling only")
    pos = settle(pos, C)
    POS.parent.mkdir(exist_ok=True)
    pos.to_csv(POS, index=False)
    d = daily_table(pos)
    d.to_csv(DAILY, index=False)
    s = d[d.settled]
    print(f"rebalances {len(d)}, settled {len(s)}, cum net {s.net.sum()*1e4:.1f} bps" + (f", mean {s.net.mean()*1e4:.1f} bps" if len(s) else ""))


main()
