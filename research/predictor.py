"""E2: walk-forward gradient-boosted predictor. Purged/embargoed expanding window, circular-shift permutation p-values."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def walk_forward(X: pd.DataFrame, y: pd.Series, h: int, min_train: int = 700, step: int = 63, seed: int = 0) -> pd.Series:
    """Predictions for each date using only models trained on data ending h days earlier (embargo)."""
    df = pd.concat([X, y.rename("__y")], axis=1)
    pred = pd.Series(np.nan, index=X.index)
    n = len(df)
    for start in range(min_train, n, step):
        end = min(start + step, n)
        tr = df.iloc[: start - h].dropna(subset=["__y"])
        tr = tr[tr.drop(columns="__y").notna().mean(axis=1) > 0.5]
        if len(tr) < 300:
            continue
        m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                              random_state=seed, verbose=-1, n_jobs=4)
        yt = tr["__y"].clip(tr["__y"].quantile(0.02), tr["__y"].quantile(0.98))
        m.fit(tr.drop(columns="__y"), yt)
        pred.iloc[start:end] = m.predict(df.iloc[start:end].drop(columns="__y"))
    return pred


def evaluate(pred: pd.Series, fwd: pd.Series, h: int, n_perm: int = 500, seed: int = 0) -> dict:
    d = pd.concat([pred, fwd], axis=1).dropna()
    d = d.iloc[::h] if h > 1 else d  # non-overlapping sample for honest n
    if len(d) < 30:
        return {}
    p, f = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    ic = stats.spearmanr(p, f)[0]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        k = rng.integers(5, len(p) - 5)
        null.append(stats.spearmanr(np.roll(p, k), f)[0])
    pv = (1 + np.sum(np.abs(null) >= abs(ic))) / (1 + n_perm)
    pos = np.sign(p - np.median(p))
    return {"n_oos": len(d), "ic": ic, "p_value": pv, "hit_rate": float((np.sign(f) == np.sign(p - np.median(p))).mean()),
            "ls_mean_per_period": float((pos * f).mean()), "bh_mean": float(f.mean())}
