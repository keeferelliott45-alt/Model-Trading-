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

## Research loop (hypothesis -> test -> new connector -> repeat)
`research/` + `data_connectors/macro.py`: 59 free series (FRED rates/credit/liquidity/EPU, Caldara-Iacoviello GPR geopolitics, VIX/DXY/oil/copper/defense/semis, Fear&Greed, Deribit funding+DVOL, CoinMetrics exchange flows/MVRV/supply, stablecoin supply, CFTC managed-money COT). Every hypothesis tested is logged in `reports/ledger.csv` with global BH/Bonferroni correction.
- E1 crypto "after the move" event study (BTC/ETH x whale flow/MVRV/funding/sentiment/halving cycle): candidate = shock **continuation**.
- E2 walk-forward LightGBM on 280 features, 10 assets: **no OOS skill** (IC ~0).
- E3 replication of E1 on 19 other coins, BTC-neutral: 5d surges continue (+7.8% over 20d, p=0.0001, 17/19 coins). Survivorship-biased.
- E4 tradable test (long alt / short BTC, 40bps costs): positive but dominated by 2021/2024; 2025-26 negative.
- E5 regime split: edge present only when BTC > 200d MA.

## Information store (`data/info.db`, local; not committed)
Tables: `series` (683 numeric series), `snapshots` (option term/skew, strike magnets, Polymarket odds), `events` (headlines), `signals` (every call, win or loss), `hypotheses`, `ledger_tests`, `notes` (lessons/dead ends), `sources` (catalog + status + limits), `collector_runs` (health). Raw API payloads archived gz under `data/raw/`.
Live collectors capture what can never be backfilled: Deribit option book, Hyperliquid/Bitget/OKX funding+OI+basis, BTC order-book depth, Polymarket odds, mempool/Blockchair, headlines; daily CFTC TFF.
Run: `python scripts/collect.py --backfill && python scripts/seed_knowledge.py` (then hourly `python scripts/collect.py`).
