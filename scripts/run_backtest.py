"""Phase 2: multi-horizon FDR-corrected backtest. Usage: python scripts/run_backtest.py"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.grid import run_grid, summarize  # noqa: E402
from data_connectors.price_connector import PriceFetchError, fetch_prices  # noqa: E402
from signals.regime import build_returns, regime_score  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "basket.yaml").read_text())
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / "backtest.log"), logging.StreamHandler()],
    )
    log = logging.getLogger("run_backtest")

    tickers = cfg["risk_on"] + cfg["risk_off"]
    prices, failed = {}, []
    for t in tickers:
        try:
            prices[t] = fetch_prices(t, cfg["history_start"], str(ROOT / cfg["cache_dir"]), cfg["cache_max_age_hours"])
        except PriceFetchError as e:
            log.error("dropping %s: %s", t, e)
            failed.append(t)

    risk_on = [t for t in cfg["risk_on"] if t in prices]
    risk_off = [t for t in cfg["risk_off"] if t in prices]
    if len(risk_on) < 2 or len(risk_off) < 1:
        log.error("not enough data to build regime score; failed=%s", failed)
        sys.exit(1)

    returns = build_returns(prices)
    regime = regime_score(returns, risk_on, risk_off, cfg["regime_window_days"])
    res = run_grid(
        prices, returns, regime,
        cfg["lookback_windows"], cfg["holding_horizons"],
        cfg["residual_sigma_threshold"], cfg["fdr_alpha"],
    )

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "grid_results.csv", index=False)
    summary = summarize(res, cfg["fdr_alpha"])
    top = res.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}") if not res.empty else ""
    report = (
        f"# Phase 2 backtest report\n\nGenerated {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Assets used: {sorted(prices)}\nDropped: {failed}\n\n```\n{summary}\n```\n\n"
        f"Top 15 by raw p-value:\n\n```\n{top}\n```\n"
    )
    (out / "phase2_report.md").write_text(report)
    (out / "summary.json").write_text(json.dumps({"summary": summary, "dropped": failed}, indent=2))
    print(report)


if __name__ == "__main__":
    main()
