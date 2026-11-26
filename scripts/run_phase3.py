"""Phase 3 (free positioning data): Deribit funding as regressor and as crowding filter, BTC/ETH."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.grid import _forward_returns, _non_overlapping  # noqa: E402
from data_connectors.funding_connector import fetch_funding  # noqa: E402
from data_connectors.price_connector import fetch_prices  # noqa: E402
from signals.regime import build_returns, regime_score, rolling_regression_residuals  # noqa: E402

cfg = yaml.safe_load((ROOT / "config" / "basket.yaml").read_text())
tickers = cfg["risk_on"] + cfg["risk_off"]
prices = {t: fetch_prices(t, cfg["history_start"], str(ROOT / cfg["cache_dir"]), cfg["cache_max_age_hours"]) for t in tickers}
returns = build_returns(prices)
regime = regime_score(returns, cfg["risk_on"], cfg["risk_off"], cfg["regime_window_days"])

PERP = {"BTC-USD": "BTC-PERPETUAL", "ETH-USD": "ETH-PERPETUAL"}


def multi_resid(y: pd.Series, X: pd.DataFrame, lb: int) -> pd.Series:
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    yv, Xv = df["y"].to_numpy(), np.column_stack([np.ones(len(df)), df[X.columns].to_numpy()])
    out = pd.Series(np.nan, index=df.index)
    for i in range(lb, len(df)):
        Xw, yw = Xv[i - lb:i], yv[i - lb:i]
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        sd = (yw - Xw @ beta).std(ddof=Xw.shape[1])
        if sd > 0:
            out.iloc[i] = (yv[i] - Xv[i] @ beta) / sd
    return out.reindex(y.index)


rows = []
for asset, inst in PERP.items():
    fund = fetch_funding(inst, "2019-01-01", str(ROOT / cfg["cache_dir"]))
    fz = ((fund - fund.rolling(90).mean()) / fund.rolling(90).std()).reindex(returns.index)
    close = prices[asset]["close"].reindex(returns.index).ffill()
    for lb in cfg["lookback_windows"]:
        variants = {
            "regime+funding": multi_resid(returns[asset], pd.concat([regime.rename("regime"), fz.shift(1).rename("fz")], axis=1), lb),
            "price_only": rolling_regression_residuals(returns[asset], regime, lb),
        }
        for vname, resid in variants.items():
            flagged = resid.abs() > cfg["residual_sigma_threshold"]
            crowded = flagged & (np.sign(resid) == np.sign(fz.shift(1))) & (fz.shift(1).abs() > 1)
            for h in cfg["holding_horizons"]:
                rev = -np.sign(resid) * _forward_returns(close, h)
                valid = (rev.notna() & resid.notna()).to_numpy()
                ctrl = rev.to_numpy()[valid & ~flagged.to_numpy()]
                for fname, mask in (("flagged", flagged), ("flagged_crowded", crowded)):
                    if vname == "price_only" and fname == "flagged":
                        continue  # identical to Phase 2
                    idx = _non_overlapping(np.where(valid & mask.to_numpy())[0], h)
                    if len(idx) < 8:
                        continue
                    f = rev.to_numpy()[idx]
                    t, p = stats.ttest_ind(f, ctrl, equal_var=False)
                    rows.append(dict(asset=asset, model=vname, subset=fname, lookback=lb, horizon=h,
                                     events=len(idx), mean_rev=f.mean(), control=ctrl.mean(),
                                     edge=f.mean() - ctrl.mean(), p_value=p))

res = pd.DataFrame(rows)
res["p_fdr"] = multipletests(res.p_value, method="fdr_bh")[1]
res["p_bonf"] = multipletests(res.p_value, method="bonferroni")[1]
res["direction"] = np.where(res.edge > 0, "reversion", "continuation")
res = res.sort_values("p_value").reset_index(drop=True)
(ROOT / "reports").mkdir(exist_ok=True)
res.to_csv(ROOT / "reports" / "phase3_funding_results.csv", index=False)
n = len(res)
summ = (f"Tests: {n}\nRaw p<0.05: {(res.p_value<0.05).sum()} (~{n*0.05:.1f} by chance)\n"
        f"Survive FDR: {(res.p_fdr<0.05).sum()}\nSurvive Bonferroni: {(res.p_bonf<0.05).sum()}\n"
        f"Raw-sig reversion: {((res.p_value<0.05)&(res.edge>0)).sum()}, continuation: {((res.p_value<0.05)&(res.edge<0)).sum()}")
top = res.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}")
rep = f"# Phase 3 (funding) report\n\n```\n{summ}\n```\n\nTop 12:\n\n```\n{top}\n```\n"
(ROOT / "reports" / "phase3_report.md").write_text(rep)
print(rep)
