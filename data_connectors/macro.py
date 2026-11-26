"""Registry of free daily/weekly data connectors -> pd.Series (date index). Cached, retried.

Every series carries a `lag` (business days) so features are only usable after publication.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger("macro")
UA = {"User-Agent": "model-trading-research/1.0"}
CACHE = Path(__file__).resolve().parents[1] / "cache"


def _get(url: str, params=None, tries: int = 4, timeout: int = 40) -> requests.Response:
    last = None
    for a in range(tries):
        try:
            r = requests.get(url, params=params, timeout=(10, timeout))
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** a)
    raise RuntimeError(f"GET failed {url}: {last}")


def _cached(name: str, fn, max_age_h: int = 20) -> pd.Series:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"s_{name}.csv"
    if p.exists() and (time.time() - p.stat().st_mtime) / 3600 < max_age_h:
        return pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0].rename(name)
    s = fn().dropna().astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index().rename(name)
    s.to_frame().to_csv(p)
    return s


def fred(sid: str) -> pd.Series:
    def f():
        df = pd.read_csv(io.StringIO(_get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": sid}).text))
        return pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0]))
    return _cached(f"fred_{sid}", f)


def yf_close(tk: str) -> pd.Series:
    def f():
        import yfinance as yf
        d = yf.download(tk, start="2016-01-01", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return d["Close"]
    return _cached(f"yf_{tk}", f)


def yf_volume(tk: str) -> pd.Series:
    def f():
        import yfinance as yf
        d = yf.download(tk, start="2016-01-01", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return d["Volume"]
    return _cached(f"yfvol_{tk}", f)


def fear_greed() -> pd.Series:
    def f():
        j = _get("https://api.alternative.me/fng/", {"limit": 0}).json()["data"]
        return pd.Series({pd.Timestamp(int(x["timestamp"]), unit="s"): float(x["value"]) for x in j})
    return _cached("fear_greed", f)


def deribit_dvol(cur: str) -> pd.Series:
    def f():
        out, s, end = [], int(pd.Timestamp("2021-04-01").timestamp() * 1000), int(time.time() * 1000)
        while s < end:
            e = min(s + 500 * 86_400_000, end)
            r = _get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                     {"currency": cur, "start_timestamp": s, "end_timestamp": e, "resolution": 86400}).json()["result"]["data"]
            out += r
            s = e
        return pd.Series({pd.Timestamp(x[0], unit="ms"): x[4] for x in out})
    return _cached(f"dvol_{cur}", f)


def stablecoin_supply() -> pd.Series:
    def f():
        j = _get("https://stablecoins.llama.fi/stablecoincharts/all").json()
        return pd.Series({pd.Timestamp(int(x["date"]), unit="s"): x["totalCirculatingUSD"].get("peggedUSD") for x in j})
    return _cached("stablecoin_supply", f)


def blockchain_chart(name: str) -> pd.Series:
    def f():
        j = _get(f"https://api.blockchain.info/charts/{name}", {"timespan": "all", "format": "json", "sampled": "false"}).json()["values"]
        return pd.Series({pd.Timestamp(x["x"], unit="s"): x["y"] for x in j})
    return _cached(f"bc_{name}", f)


def coinmetrics(asset: str, metric: str) -> pd.Series:
    def f():
        rows, tok = [], None
        while True:
            prm = {"assets": asset, "metrics": metric, "frequency": "1d", "start_time": "2016-01-01", "page_size": 10000}
            if tok:
                prm["next_page_token"] = tok
            j = _get("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics", prm).json()
            rows += j["data"]
            tok = j.get("next_page_token")
            if not tok:
                break
        return pd.Series({pd.Timestamp(r["time"][:10]): float(r[metric]) for r in rows if r.get(metric) is not None})
    return _cached(f"cm_{asset}_{metric}", f)


def gpr() -> pd.Series:
    def f():
        df = pd.read_excel(io.BytesIO(_get("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls").content))
        df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
        return df.set_index("date")["GPRD"]
    return _cached("gpr_daily", f)


def cot_managed_money(market_substr: str) -> pd.Series:
    """Managed-money net positioning as % of OI (weekly, Tuesday data, released Friday)."""
    def f():
        rows, off = [], 0
        while True:
            r = _get("https://publicreporting.cftc.gov/resource/72hh-3qpy.json",
                     {"$where": f"market_and_exchange_names like '%{market_substr}%'", "$limit": 5000, "$offset": off,
                      "$order": "report_date_as_yyyy_mm_dd"}).json()
            rows += r
            if len(r) < 5000:
                break
            off += 5000
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f"no COT rows for {market_substr}")
        df["d"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        net = (pd.to_numeric(df["m_money_positions_long_all"]) - pd.to_numeric(df["m_money_positions_short_all"])) / pd.to_numeric(df["open_interest_all"])
        return pd.Series(net.values, index=df["d"] + pd.Timedelta(days=3)).groupby(level=0).mean()
    return _cached(f"cot_{market_substr.replace(' ', '_')[:20]}", f, 72)


# name -> (callable, publication lag in days)
REGISTRY: dict[str, tuple] = {
    # rates / credit / liquidity / dollar (FRED, 1d lag)
    "us10y": (lambda: fred("DGS10"), 1), "us2y": (lambda: fred("DGS2"), 1), "curve": (lambda: fred("T10Y2Y"), 1),
    "hy_spread": (lambda: fred("BAMLH0A0HYM2"), 1), "usd_broad": (lambda: fred("DTWEXBGS"), 2),
    "real10y": (lambda: fred("DFII10"), 1), "breakeven10": (lambda: fred("T10YIE"), 1),
    "fin_stress": (lambda: fred("STLFSI4"), 7), "fed_bs": (lambda: fred("WALCL"), 8), "rrp": (lambda: fred("RRPONTSYD"), 1),
    "epu_daily": (lambda: fred("USEPUINDXD"), 2), "wti": (lambda: fred("DCOILWTICO"), 2),
    # geopolitics
    "gpr": (gpr, 1),
    # market-implied
    "vix": (lambda: yf_close("^VIX"), 1), "dxy": (lambda: yf_close("DX-Y.NYB"), 1), "spx": (lambda: yf_close("^GSPC"), 1),
    "ndx": (lambda: yf_close("^NDX"), 1), "tlt": (lambda: yf_close("TLT"), 1), "hyg": (lambda: yf_close("HYG"), 1),
    "copper": (lambda: yf_close("HG=F"), 1), "oil": (lambda: yf_close("CL=F"), 1), "natgas": (lambda: yf_close("NG=F"), 1),
    "defense": (lambda: yf_close("ITA"), 1), "semis": (lambda: yf_close("SOXX"), 1), "uranium": (lambda: yf_close("URA"), 1),
    "usdjpy": (lambda: yf_close("JPY=X"), 1), "em": (lambda: yf_close("EEM"), 1), "ibit_vol": (lambda: yf_volume("IBIT"), 1),
    "gold_miners": (lambda: yf_close("GDX"), 1), "kre": (lambda: yf_close("KRE"), 1),
    # crypto
    "fear_greed": (fear_greed, 0), "dvol_btc": (lambda: deribit_dvol("BTC"), 1), "dvol_eth": (lambda: deribit_dvol("ETH"), 1),
    "stablecoins": (stablecoin_supply, 1), "btc_hashrate": (lambda: blockchain_chart("hash-rate"), 1),
    "btc_txs": (lambda: blockchain_chart("n-transactions"), 1),
    "btc_mempool_fees": (lambda: blockchain_chart("transaction-fees-usd"), 1),
    # on-chain (CoinMetrics community: exchange flows ~ whale/distribution proxy, MVRV ~ cycle valuation)
    "cm_btc_FlowInExUSD": (lambda: coinmetrics("btc", "FlowInExUSD"), 1),
    "cm_btc_FlowOutExUSD": (lambda: coinmetrics("btc", "FlowOutExUSD"), 1),
    "cm_btc_SplyExUSD": (lambda: coinmetrics("btc", "SplyExUSD"), 1),
    "cm_btc_CapMVRVCur": (lambda: coinmetrics("btc", "CapMVRVCur"), 1),
    "cm_btc_AdrActCnt": (lambda: coinmetrics("btc", "AdrActCnt"), 1),
    "cm_btc_TxCnt": (lambda: coinmetrics("btc", "TxCnt"), 1),
    "cm_btc_IssTotUSD": (lambda: coinmetrics("btc", "IssTotUSD"), 1),
    "cm_btc_FeeTotNtv": (lambda: coinmetrics("btc", "FeeTotNtv"), 1),
    "cm_btc_AdrBalCnt": (lambda: coinmetrics("btc", "AdrBalCnt"), 1),
    "cm_btc_volume_reported_spot_usd_1d": (lambda: coinmetrics("btc", "volume_reported_spot_usd_1d"), 1),
    "cm_eth_FlowInExUSD": (lambda: coinmetrics("eth", "FlowInExUSD"), 1),
    "cm_eth_FlowOutExUSD": (lambda: coinmetrics("eth", "FlowOutExUSD"), 1),
    "cm_eth_SplyExUSD": (lambda: coinmetrics("eth", "SplyExUSD"), 1),
    "cm_eth_CapMVRVCur": (lambda: coinmetrics("eth", "CapMVRVCur"), 1),
    "cm_eth_AdrActCnt": (lambda: coinmetrics("eth", "AdrActCnt"), 1),
    "cm_eth_TxCnt": (lambda: coinmetrics("eth", "TxCnt"), 1),
    "cm_eth_IssTotUSD": (lambda: coinmetrics("eth", "IssTotUSD"), 1),
    "cm_eth_FeeTotNtv": (lambda: coinmetrics("eth", "FeeTotNtv"), 1),
    "cm_eth_AdrBalCnt": (lambda: coinmetrics("eth", "AdrBalCnt"), 1),
    "cm_eth_volume_reported_spot_usd_1d": (lambda: coinmetrics("eth", "volume_reported_spot_usd_1d"), 1),
    # positioning (weekly, publication-lagged inside connector)
    "cot_gold": (lambda: cot_managed_money("GOLD - COMMODITY EXCHANGE"), 0),
    "cot_silver": (lambda: cot_managed_money("SILVER - COMMODITY EXCHANGE"), 0),
    "cot_copper": (lambda: cot_managed_money("COPPER- #1 - COMMODITY EXCHANGE"), 0),
}


def load_all() -> tuple[dict[str, pd.Series], dict[str, str]]:
    out, failed = {}, {}
    for name, (fn, lag) in REGISTRY.items():
        try:
            s = fn()
            if lag:
                s.index = s.index + pd.tseries.offsets.BDay(lag)
            s = s[~s.index.duplicated(keep='last')]
            out[name] = s
        except Exception as e:  # noqa: BLE001
            failed[name] = str(e)[:120]
            logger.warning("connector %s failed: %s", name, e)
    return out, failed
