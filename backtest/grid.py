"""Multi-horizon grid backtest with FDR correction and walk-forward fold consistency."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from signals.regime import flag_overreactions, rolling_regression_residuals

logger = logging.getLogger("grid")


def _non_overlapping(idx: np.ndarray, gap: int) -> np.ndarray:
    keep, last = [], -10**9
    for i in idx:
        if i - last >= gap:
            keep.append(i)
            last = i
    return np.array(keep, dtype=int)


def _forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def run_grid(
    prices: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    regime: pd.Series,
    lookbacks: list[int],
    horizons: list[int],
    sigma: float,
    alpha: float,
    n_folds: int = 4,
) -> pd.DataFrame:
    rows = []
    for asset, df in prices.items():
        close = df["close"].reindex(returns.index).ffill()
        for lb in lookbacks:
            resid = rolling_regression_residuals(returns[asset], regime, lb)
            flagged = flag_overreactions(resid, sigma)
            direction = -np.sign(resid)  # expected reversion direction
            for h in horizons:
                fwd = _forward_returns(close, h)
                rev = (direction * fwd)
                valid = rev.notna() & resid.notna()
                flag_idx = np.where(valid.to_numpy() & flagged.to_numpy())[0]
                flag_idx = _non_overlapping(flag_idx, h)
                ctrl_mask = valid.to_numpy() & ~flagged.to_numpy()
                if len(flag_idx) < 8 or ctrl_mask.sum() < 30:
                    continue
                rev_arr = rev.to_numpy()
                f = rev_arr[flag_idx]
                c = rev_arr[ctrl_mask]
                t, p = stats.ttest_ind(f, c, equal_var=False)

                folds = np.array_split(np.arange(len(rev_arr)), n_folds)
                fold_wins, fold_used = 0, 0
                for fo in folds:
                    fset = set(fo.tolist())
                    ff = [rev_arr[i] for i in flag_idx if i in fset]
                    cc = rev_arr[[i for i in np.where(ctrl_mask)[0] if i in fset]]
                    if len(ff) >= 3 and len(cc) >= 10:
                        fold_used += 1
                        fold_wins += int(np.mean(ff) > np.mean(cc))

                rows.append({
                    "asset": asset, "lookback": lb, "horizon": h,
                    "events": len(flag_idx),
                    "mean_reversion_ret": float(f.mean()),
                    "control_ret": float(c.mean()),
                    "edge": float(f.mean() - c.mean()),
                    "t_stat": float(t), "p_value": float(p),
                    "folds_used": fold_used, "folds_positive": fold_wins,
                })
                logger.info("%s lb=%d h=%d events=%d edge=%.4f p=%.4f", asset, lb, h, len(flag_idx), f.mean() - c.mean(), p)

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["p_fdr"] = multipletests(res["p_value"], alpha=alpha, method="fdr_bh")[1]
    res["p_bonf"] = multipletests(res["p_value"], alpha=alpha, method="bonferroni")[1]
    res["sig_raw"] = res["p_value"] < alpha
    res["sig_fdr"] = res["p_fdr"] < alpha
    res["sig_bonf"] = res["p_bonf"] < alpha
    res["direction"] = np.where(res["edge"] > 0, "reversion", "continuation")
    return res.sort_values("p_value").reset_index(drop=True)


def summarize(res: pd.DataFrame, alpha: float) -> str:
    if res.empty:
        return "No tests had enough events."
    n = len(res)
    lines = [
        f"Tests run: {n}",
        f"Significant raw (p<{alpha}): {int(res.sig_raw.sum())} "
        f"(~{n * alpha:.1f} expected by chance)",
        f"Survive FDR (BH): {int(res.sig_fdr.sum())}",
        f"Survive Bonferroni: {int(res.sig_bonf.sum())}",
        f"Raw-significant reversion: {int((res.sig_raw & (res.direction == 'reversion')).sum())}, "
        f"continuation: {int((res.sig_raw & (res.direction == 'continuation')).sum())}",
    ]
    return "\n".join(lines)
