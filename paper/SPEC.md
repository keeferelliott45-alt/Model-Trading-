# Paper-trading spec (frozen 2026-11-27, do not tune while live)

Model: `research/hourly.py` pooled LightGBM, exactly the E11 config that survived review.
- Universe: Bitget USDT perps from the top-45-by-volume list with >=6000 hourly bars and >=97% hourly coverage (same filter as validation).
- Target: 24h log return, demeaned cross-sectionally; features at hour t; **entry at close of t+1** (1-bar delay); exit 24h later.
- Training: all settled history, every 3rd hour, noise aug 0.05, 2 seeds averaged (0,1).
- Portfolio: long top-5 / short bottom-5 by prediction, equal weight +-0.2, one rebalance per day.
- Costs: 6 bps per unit of turnover (taker fee + slippage).

Backtest expectation (E11, OOS walk-forward, delay=1): IC ~0.035, net ~+29 bps per daily rebalance, Sharpe ~1.5.

Kill / promote rules (decided before any live data):
- After 60 rebalances: stop if cumulative net <= 0 OR mean daily IC <= 0.
- After 120 rebalances: promote to small real capital only if net Sharpe >= 0.8 AND mean IC >= 0.015.
- Any code/param change resets the clock (new spec version).
