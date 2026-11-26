"""Seed the sources catalog, hypotheses table and findings notes (idempotent)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from store.db import connect, now, put_note

SOURCES = [
 ("FRED", "macro", "fred.stlouisfed.org/graph/fredgraph.csv?id=", "free", "NO custom User-Agent (hangs)", "full", "ok"),
 ("Caldara-Iacoviello GPR", "geopolitics", "matteoiacoviello.com/gpr_files", "free", "daily xls, needs xlrd", "1985+", "ok"),
 ("CoinMetrics community", "onchain", "community-api.coinmetrics.io/v4", "free", "30 metrics BTC/ETH: exchange flows, MVRV, supply on exchanges, active addrs", "2016+", "ok"),
 ("Deribit", "derivs", "deribit.com/api/v2/public", "free", "funding history, DVOL, full option book (snapshot only)", "funding+DVOL yes; options no", "ok"),
 ("Hyperliquid info", "derivs", "api.hyperliquid.xyz/info", "free", "all perps funding/OI/mark in one call", "no (snapshot)", "ok"),
 ("Bitget public", "derivs", "api.bitget.com/api/v2/mix", "free", "20 req/s/IP; tickers, depth, funding", "funding hist yes, OI no", "ok"),
 ("OKX public", "derivs", "okx.com/api/v5/public", "free", "OI by instrument", "no (snapshot)", "ok"),
 ("Kraken/Coinbase books", "microstructure", "api.kraken.com / api.exchange.coinbase.com", "free", "L2 depth", "no (snapshot)", "ok"),
 ("Polymarket gamma", "odds/geopolitics", "gamma-api.polymarket.com/markets", "free", "crowd-implied probabilities", "no (snapshot)", "ok"),
 ("CFTC Socrata COT (disagg)", "positioning", "publicreporting.cftc.gov/resource/72hh-3qpy", "free", "commodities managed money, weekly", "2006+", "ok"),
 ("CFTC Socrata TFF", "positioning", "publicreporting.cftc.gov/resource/gpe5-46if", "free", "BTC/ES/NQ/UST/VIX leveraged funds, weekly", "2006+", "ok"),
 ("DefiLlama stablecoins", "liquidity", "stablecoins.llama.fi", "free", "total stablecoin supply", "2017+", "ok"),
 ("blockchain.info charts", "onchain", "api.blockchain.info/charts", "free", "hashrate, txs, fees", "full", "ok"),
 ("mempool.space / Blockchair", "onchain", "mempool.space/api, api.blockchair.com", "free", "fees, largest tx 24h", "no (snapshot)", "ok"),
 ("Cointelegraph RSS", "news", "cointelegraph.com/rss", "free", "headlines -> events table", "no", "ok"),
 ("Alternative.me Fear&Greed", "sentiment", "api.alternative.me/fng", "free", "daily", "2018+", "ok"),
 ("yfinance", "prices", "yahoo", "free", "rate-limits (429 crumb) but works w/ retry", "full", "ok"),
 ("Binance futures", "derivs", "fapi.binance.com", "free", "HTTP 451 from this env", "-", "blocked"),
 ("Bybit", "derivs", "api.bybit.com", "free", "HTTP 403 from this env", "-", "blocked"),
 ("GDELT doc API", "geopolitics", "api.gdeltproject.org", "free", "timeouts/429 from this env; gkg zip files reachable", "-", "flaky"),
 ("Wikipedia pageviews", "attention", "wikimedia.org/api/rest_v1", "free", "429 without UA, UA hangs elsewhere; retry later", "-", "flaky"),
 ("CFTC cftc.gov direct", "positioning", "cftc.gov/dea", "free", "403 -> use Socrata instead", "-", "blocked"),
 ("Etherscan", "onchain", "api.etherscan.io", "free key", "needs key (100k/day free) -> real whale wallets", "yes", "needs_key"),
 ("Arkham / Whale Alert / Glassnode", "onchain", "-", "paid/keyed", "entity-labelled whale flow", "-", "needs_key"),
]
HYP = [
 ("H1_phase1_mr", "Price-only residual mean reversion on 11-asset basket", "rejected", 33, 0.012, 1.0, "3/33 raw sig, all continuation"),
 ("H2_grid_fdr", "Multi-horizon grid 4 lookbacks x 6 horizons x 11 assets", "rejected", 261, 0.005, 0.607, "16 raw sig, 0 survive BH/Bonferroni"),
 ("H3_funding", "Deribit funding regressor/crowding filter on BTC/ETH reversion", "rejected", 144, 0.0036, 0.52, "0 survive"),
 ("H4_lgbm_per_asset", "Per-asset LightGBM on 280 macro/onchain feats", "rejected", 20, 0.078, 1.0, "IC ~0 OOS"),
 ("H5_crypto_shock_cont", "Post-shock CONTINUATION in crypto (5d surge, 1d crash)", "candidate", 9, 0.0001, 0.001, "19 alt coins BTC-neutral +7.8%/20d; regime-dependent (BTC>MA200), dominated by 2021/2024, negative 2025; survivorship-biased"),
 ("H6_pooled_xs_model", "Kaggle-style pooled cross-sectional LightGBM, market-neutral target, purged WF, 21 coins", "rejected", 4, 0.5, 1.0, "IC~0.006, Sharpe ~0; momentum/reversal baselines ~0 too"),
]
NOTES = [
 ("lesson:data", "Setting a custom User-Agent makes FRED hang from this sandbox; use default headers. yfinance needs retry on 429 crumb errors."),
 ("lesson:method", "Per-asset models on ~2k daily rows cannot learn; Kaggle winners (G-Research crypto, Jane Street) pool assets, use market-neutral targets, heavy feature work, purged CV, noise augmentation."),
 ("lesson:method", "Daily cross-sectional crypto alpha on survivor coins from public OHLCV is ~zero after costs. Edge, if any, is regime-conditional (trend) and sparse (extreme shocks), not a general predictor."),
 ("direction", "Information advantage = data nobody backfills: option-book skew/OI, cross-venue funding/OI/basis, order-book depth, prediction-market odds, headlines. Collected hourly into data branch from now; revisit models once ~3-6 months accrue."),
 ("todo", "Need keys for real whale wallet data (Etherscan free key, Arkham). Hourly bars via Bitget candles for high-frequency pooled model (more samples = Kaggle-style edge)."),
]
con = connect()
for r in SOURCES:
    con.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?)", (*r, now() if r[-1] == "ok" else "", ""))
for h in HYP:
    con.execute("INSERT OR REPLACE INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?)", (h[0], now(), h[1], h[2], h[3], h[4], h[5], h[6], ""))
con.execute("DELETE FROM notes WHERE topic IN ('lesson:data','lesson:method','direction','todo')")
for t, b in NOTES:
    con.execute("INSERT INTO notes VALUES (?,?,?)", (now(), t, b))
led = Path("reports/ledger.csv")
if led.exists():
    L = pd.read_csv(led)
    L.to_sql("ledger_tests", con, if_exists="replace", index=False)
con.commit()
print("seeded", len(SOURCES), "sources,", len(HYP), "hypotheses")
