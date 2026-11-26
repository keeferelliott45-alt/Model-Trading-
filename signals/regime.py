"""Regime score and rolling regression decomposition (Phase 1 core, swept over windows)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_returns(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """prices: {ticker: df with 'close' column} -> wide DataFrame of daily pct returns."""
    closes = pd.DataFrame({t: df["close"] for t, df in prices.items()})
    closes = closes.sort_index().ffill()
    returns = closes.pct_change()
    return returns


def regime_score(returns: pd.DataFrame, risk_on: list[str], risk_off: list[str], window: int) -> pd.Series:
    """z-scored (risk-on avg return - risk-off avg return), rolling `window`-day window."""
    risk_on_avg = returns[risk_on].mean(axis=1)
    risk_off_avg = returns[risk_off].mean(axis=1)
    spread = risk_on_avg - risk_off_avg
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    z = (spread - rolling_mean) / rolling_std
    return z.rename("regime_score")


def rolling_regression_residuals(asset_returns: pd.Series, regime: pd.Series, lookback: int) -> pd.Series:
    """For each day, regress asset_returns on regime over the trailing `lookback` window,
    return the standardized residual (today's actual minus predicted, divided by residual std
    of the fitting window)."""
    df = pd.concat([asset_returns.rename("y"), regime.rename("x")], axis=1).dropna()
    n = len(df)
    resid = pd.Series(index=df.index, dtype=float)

    y = df["y"].to_numpy()
    x = df["x"].to_numpy()

    for i in range(lookback, n):
        y_win = y[i - lookback:i]
        x_win = x[i - lookback:i]
        x_mean = x_win.mean()
        y_mean = y_win.mean()
        denom = ((x_win - x_mean) ** 2).sum()
        if denom == 0:
            continue
        beta = ((x_win - x_mean) * (y_win - y_mean)).sum() / denom
        alpha = y_mean - beta * x_mean
        pred_win = alpha + beta * x_win
        resid_win = y_win - pred_win
        resid_std = resid_win.std(ddof=1)
        if resid_std == 0 or np.isnan(resid_std):
            continue
        today_pred = alpha + beta * x[i]
        today_resid = y[i] - today_pred
        resid.iloc[i] = today_resid / resid_std

    return resid.reindex(asset_returns.index)


def flag_overreactions(std_residual: pd.Series, sigma_threshold: float) -> pd.Series:
    """Boolean series: True where |standardized residual| exceeds sigma_threshold."""
    return std_residual.abs() > sigma_threshold
