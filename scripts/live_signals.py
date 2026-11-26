"""Paper-trading loop: refresh bars -> train pooled hourly model on all settled data -> log top/bottom-k signals -> settle past ones.
Usage: python scripts/live_signals.py [--h 24] [--k 3] [--no-refresh]"""
import sys, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import lightgbm as lgb, numpy as np, pandas as pd
from collectors import bars
from research.hourly import build
from store.db import connect, now

H = int(sys.argv[sys.argv.index("--h") + 1]) if "--h" in sys.argv else 24
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 3
MODEL = f"hourly_pooled_h{H}_k{K}"

def main():
    con = connect()
    if "--no-refresh" not in sys.argv:
        bars.run()
    P = build(H)
    feats = [c for c in P.columns if c not in ("date", "coin", "tgt", "fwd", "pred", "cid")]
    last = P.date.max()
    tr = P[(P.date <= last - pd.Timedelta(hours=H + 1)) & P.tgt.notna() & (P.date.dt.hour % 3 == 0)].dropna(subset=feats, thresh=int(len(feats) * .8))
    X = tr[feats].astype(float); lo, hi = tr.tgt.quantile([.005, .995])
    X = X + np.random.default_rng(0).normal(0, .05, X.shape) * X.std().to_numpy()
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.6, subsample_freq=1,
                          colsample_bytree=0.6, reg_lambda=20, random_state=0, verbose=-1, n_jobs=4).fit(X, tr.tgt.clip(lo, hi))
    now_rows = P[P.date == last].dropna(subset=feats, thresh=int(len(feats) * .8)).copy()
    now_rows["pred"] = m.predict(now_rows[feats].astype(float))
    r = now_rows.sort_values("pred")
    ts = last.strftime("%Y-%m-%dT%H:%M:%SZ")
    imp = dict(sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])[:5])
    if con.execute("SELECT 1 FROM signals WHERE ts=? AND model=?", (ts, MODEL)).fetchone() is None:
        for side, g in (("long", r.tail(K)), ("short", r.head(K))):
            for _, x in g.iterrows():
                con.execute("INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (ts, MODEL, x.coin, side, float(x.pred), H, f"market-neutral {H}h rank {'top' if side=='long' else 'bottom'}-{K} of {len(r)}; top feats {list(imp)}", json.dumps({"r24": x.get("r24"), "z24": x.get("z24")}, default=float), None, None))
        con.commit()
    # settle: outcome = market-neutral fwd return (side-adjusted) once H hours have elapsed
    px = pd.DataFrame({Path(f).stem.replace("USDT", ""): pd.read_parquet(f).c for f in __import__("glob").glob(str(bars.OUT / "*.parquet"))})
    fwd = np.log(px.shift(-H)) - np.log(px)
    mkt = fwd.mean(axis=1)
    for rowid, sts, asset, side in con.execute("SELECT rowid, ts, asset, direction FROM signals WHERE model=? AND outcome IS NULL", (MODEL,)).fetchall():
        t = pd.Timestamp(sts).tz_localize(None)
        if t in fwd.index and not np.isnan(fwd.loc[t, asset]):
            o = (fwd.loc[t, asset] - mkt.loc[t]) * (1 if side == "long" else -1)
            con.execute("UPDATE signals SET outcome=?, outcome_ts=? WHERE rowid=?", (float(o), now(), rowid))
    con.commit()
    done = pd.read_sql_query("SELECT * FROM signals WHERE model=? AND outcome IS NOT NULL", con, params=(MODEL,))
    print(f"{MODEL} @ {ts}: LONG {list(r.tail(K).coin)} SHORT {list(r.head(K).coin)}")
    print(f"settled signals: {len(done)}, mean outcome {done.outcome.mean()*1e4 if len(done) else float('nan'):.1f} bps, hit {(done.outcome>0).mean() if len(done) else float('nan'):.2f}")

main()
