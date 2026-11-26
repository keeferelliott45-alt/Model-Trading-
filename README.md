# Cross-Asset Overreaction Trading System

Build spec: Phase 1-2 implemented (config-driven basket, fallback price connector, rolling regression decomposition, multi-horizon grid with FDR/Bonferroni correction).

```bash
pip install -r requirements.txt
python scripts/run_backtest.py   # writes reports/phase2_report.md + grid_results.csv
```

Edit `config/basket.yaml` to change tickers, lookbacks, horizons.

## Phase 2 result (gate)
261 tests (11 assets x 4 lookbacks x 6 horizons). 16 raw-significant (~13 expected by chance); **0 survive FDR, 0 survive Bonferroni**. Raw-significant: 10 continuation, 6 reversion. ETH/BTC continuation at 5d is the most consistent pattern (0/4 folds positive for reversion) but does not survive correction. See `reports/phase2_report.md`.

## Phase 3 result (free positioning data)
Deribit funding-rate (BTC/ETH) added as a regressor and as a crowding filter: 144 tests, 6 raw-significant (~7 by chance), all continuation; **0 survive FDR/Bonferroni**. See `reports/phase3_report.md`. Whale-flow (paid/keyed APIs) not built — gate not justified by results.
