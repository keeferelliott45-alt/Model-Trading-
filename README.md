# Cross-Asset Overreaction Trading System

Build spec: Phase 1-2 implemented (config-driven basket, fallback price connector, rolling regression decomposition, multi-horizon grid with FDR/Bonferroni correction).

```bash
pip install -r requirements.txt
python scripts/run_backtest.py   # writes reports/phase2_report.md + grid_results.csv
```

Edit `config/basket.yaml` to change tickers, lookbacks, horizons.

## Phase 2 result (gate)
261 tests (11 assets x 4 lookbacks x 6 horizons). 16 raw-significant (~13 expected by chance); **0 survive FDR, 0 survive Bonferroni**. Raw-significant: 10 continuation, 6 reversion. ETH/BTC continuation at 5d is the most consistent pattern (0/4 folds positive for reversion) but does not survive correction. See `reports/phase2_report.md`.
