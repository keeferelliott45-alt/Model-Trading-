"""Live collectors for data that cannot be backfilled for free (positioning, books, odds, headlines). Each returns row count."""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

from store.db import archive_raw, now, put_event, put_point, put_snapshot

T = (10, 40)


def _j(url, **kw):
    last = None
    for a in range(3):
        try:
            r = requests.get(url, timeout=T, **kw) if "json_body" not in kw else requests.post(url, timeout=T, json=kw["json_body"])
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** a)
    raise RuntimeError(f"{url}: {last}")


def hyperliquid(con) -> int:
    r = requests.post("https://api.hyperliquid.xyz/info", json={"type": "metaAndAssetCtxs"}, timeout=T)
    r.raise_for_status()
    meta, ctx = r.json()
    ts, n = now(), 0
    rows = sorted(zip(meta["universe"], ctx), key=lambda x: -float(x[1].get("dayNtlVlm", 0)))
    for u, c in rows[:60]:
        k = u["name"]
        for f, nm in (("funding", "funding"), ("openInterest", "oi"), ("markPx", "mark"), ("dayNtlVlm", "vol24"), ("premium", "premium")):
            if c.get(f) is not None:
                put_point(con, f"hl_{k}_{nm}", float(c[f]), "hyperliquid", ts); n += 1
    f = np.array([float(c["funding"]) for _, c in rows[:60] if c.get("funding")])
    put_point(con, "hl_funding_median", float(np.median(f)), "hyperliquid", ts)
    put_point(con, "hl_funding_pct_positive", float((f > 0).mean()), "hyperliquid", ts)
    archive_raw("hyperliquid", {"meta": [u["name"] for u in meta["universe"]], "ctx": ctx})
    con.commit()
    return n


def deribit_options(con) -> int:
    ts, n = now(), 0
    for cur in ("BTC", "ETH"):
        d = _j("https://www.deribit.com/api/v2/public/get_book_summary_by_currency", params={"currency": cur, "kind": "option"}).json()["result"]
        df = pd.DataFrame(d)
        df["type"] = df.instrument_name.str[-1]
        df["exp"] = df.instrument_name.str.split("-").str[1]
        df["strike"] = df.instrument_name.str.split("-").str[2].astype(float)
        px = df.underlying_price.median()
        oi_usd = df.open_interest * px
        calls, puts = oi_usd[df.type == "C"].sum(), oi_usd[df.type == "P"].sum()
        put_point(con, f"deribit_{cur}_opt_oi_usd", calls + puts, "deribit", ts)
        put_point(con, f"deribit_{cur}_opt_putcall_oi", puts / max(calls, 1), "deribit", ts)
        vol = df.volume_usd.fillna(0)
        put_point(con, f"deribit_{cur}_opt_putcall_vol", vol[df.type == "P"].sum() / max(vol[df.type == "C"].sum(), 1), "deribit", ts)
        near = df[(df.strike > px * 0.97) & (df.strike < px * 1.03) & df.mark_iv.notna()]
        if len(near):
            put_point(con, f"deribit_{cur}_atm_iv", near.mark_iv.mean(), "deribit", ts); n += 1
        # term structure + skew snapshot (cannot be reconstructed later)
        term = {}
        for e, g in df.groupby("exp"):
            gi = g[g.mark_iv.notna()]
            if len(gi) < 4:
                continue
            otm_p = gi[(gi.type == "P") & (gi.strike < px * 0.9) & (gi.strike > px * 0.8)].mark_iv.mean()
            otm_c = gi[(gi.type == "C") & (gi.strike > px * 1.1) & (gi.strike < px * 1.2)].mark_iv.mean()
            atm = gi[(gi.strike > px * 0.97) & (gi.strike < px * 1.03)].mark_iv.mean()
            # max pain-ish: strike with largest total OI
            term[e] = {"atm": atm, "skew_25ish": otm_p - otm_c, "oi_usd": float((gi.open_interest * px).sum())}
        put_snapshot(con, "deribit_option_term", cur, {"spot": px, "term": term}, ts)
        top = df.assign(oi_usd=oi_usd).sort_values("oi_usd", ascending=False).head(25)[["instrument_name", "oi_usd", "mark_iv"]]
        put_snapshot(con, "deribit_option_top_oi", cur, top.to_dict("records"), ts)  # strike magnets / dealer positioning clues
        n += 3
    con.commit()
    return n


def bitget(con) -> int:
    d = _j("https://api.bitget.com/api/v2/mix/market/tickers", params={"productType": "usdt-futures"}).json()["data"]
    df = pd.DataFrame(d)
    for c in ("fundingRate", "holdingAmount", "usdtVolume", "lastPr", "markPrice", "indexPrice"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("usdtVolume", ascending=False).head(40)
    ts = now()
    for _, r in df.iterrows():
        k = r.symbol.replace("USDT", "")
        put_point(con, f"bitget_{k}_funding", r.fundingRate, "bitget", ts)
        put_point(con, f"bitget_{k}_oi_coins", r.holdingAmount, "bitget", ts)
        put_point(con, f"bitget_{k}_basis", (r.markPrice / r.indexPrice - 1), "bitget", ts)
    put_point(con, "bitget_funding_median", df.fundingRate.median(), "bitget", ts)
    archive_raw("bitget_tickers", df[["symbol", "fundingRate", "holdingAmount", "usdtVolume", "lastPr"]].to_dict("records"))
    con.commit()
    return len(df) * 3


def okx_oi(con) -> int:
    d = _j("https://www.okx.com/api/v5/public/open-interest", params={"instType": "SWAP"}).json()["data"]
    ts = now()
    top = sorted(d, key=lambda x: -float(x["oiUsd"]))[:30]
    for r in top:
        put_point(con, f"okx_{r['instId'].split('-')[0]}_oi_usd", float(r["oiUsd"]), "okx", ts)
    put_point(con, "okx_total_oi_usd", sum(float(r["oiUsd"]) for r in d), "okx", ts)
    con.commit()
    return len(top) + 1


def orderbooks(con) -> int:
    """Depth within 1% of mid on BTC: real liquidity + bid/ask imbalance across three venues."""
    ts, n = now(), 0
    def depth(bids, asks):
        b = np.array(bids, dtype=float); a = np.array(asks, dtype=float)
        mid = (b[0, 0] + a[0, 0]) / 2
        bu = (b[b[:, 0] >= mid * .99][:, 0] * b[b[:, 0] >= mid * .99][:, 1]).sum()
        au = (a[a[:, 0] <= mid * 1.01][:, 0] * a[a[:, 0] <= mid * 1.01][:, 1]).sum()
        return bu, au
    k = _j("https://api.kraken.com/0/public/Depth", params={"pair": "XBTUSD", "count": 500}).json()["result"]
    k = next(iter(k.values()))
    cb = _j("https://api.exchange.coinbase.com/products/BTC-USD/book", params={"level": 2}).json()
    bg = _j("https://api.bitget.com/api/v2/mix/market/merge-depth", params={"symbol": "BTCUSDT", "productType": "usdt-futures", "limit": 100}).json()["data"]
    for nm, (b, a) in {"kraken": depth([x[:2] for x in k["bids"]], [x[:2] for x in k["asks"]]),
                       "coinbase": depth([x[:2] for x in cb["bids"]], [x[:2] for x in cb["asks"]]),
                       "bitget": depth(bg["bids"], bg["asks"])}.items():
        put_point(con, f"book_{nm}_btc_bid1pct_usd", b, nm, ts)
        put_point(con, f"book_{nm}_btc_ask1pct_usd", a, nm, ts)
        put_point(con, f"book_{nm}_btc_imbalance", (b - a) / (b + a), nm, ts); n += 3
    con.commit()
    return n


def polymarket(con) -> int:
    d = _j("https://gamma-api.polymarket.com/markets", params={"limit": 100, "active": "true", "closed": "false", "order": "volume24hr", "ascending": "false"}).json()
    ts, n = now(), 0
    import json as _json
    for m in d:
        try:
            prices = _json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices")
            p = float(prices[0])
        except Exception:  # noqa: BLE001
            continue
        mid = f"pm_{m['id']}"
        put_point(con, mid, p, "polymarket", ts)
        put_snapshot(con, "polymarket_market", str(m["id"]), {"q": m.get("question"), "end": m.get("endDate"), "vol24": m.get("volume24hr"), "p_yes": p}, ts)
        n += 1
    archive_raw("polymarket", [{"id": m["id"], "q": m.get("question"), "p": m.get("outcomePrices"), "v24": m.get("volume24hr")} for m in d])
    con.commit()
    return n


def mempool(con) -> int:
    ts = now()
    f = _j("https://mempool.space/api/v1/fees/recommended").json()
    for k, v in f.items():
        put_point(con, f"mempool_fee_{k}", v, "mempool.space", ts)
    blocks = _j("https://mempool.space/api/v1/blocks").json()[:10]
    put_point(con, "btc_block_median_fee_recent", float(np.mean([b["extras"]["medianFee"] for b in blocks])), "mempool.space", ts)
    put_point(con, "btc_block_txcount_recent", float(np.mean([b["tx_count"] for b in blocks])), "mempool.space", ts)
    bc = _j("https://api.blockchair.com/bitcoin/stats").json()["data"]
    for k in ("largest_transaction_24h", "mempool_transactions", "mempool_tps", "average_transaction_fee_usd_24h", "hodling_addresses", "suggested_transaction_fee_per_byte_sat"):
        v = bc.get(k)
        if isinstance(v, dict):
            v = v.get("value_usd")
        if v is not None:
            put_point(con, f"blockchair_btc_{k}", float(v), "blockchair", ts)
    con.commit()
    return 12


def news(con) -> int:
    n = 0
    for name, url in (("cointelegraph", "https://cointelegraph.com/rss"),):
        try:
            root = ET.fromstring(_j(url).content)
        except Exception:  # noqa: BLE001
            continue
        for it in root.iter("item"):
            link = (it.findtext("link") or "").strip()
            title = (it.findtext("title") or "").strip()
            pub = pd.to_datetime(it.findtext("pubDate"), errors="coerce", utc=True)
            if link and title and put_event(con, f"{name}:{link}", pub.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(pub) else now(), "news", name, title, link, {"desc": (it.findtext("description") or "")[:300]}):
                n += 1
    con.commit()
    return n


def cftc_tff(con) -> int:
    """Leveraged-funds net positioning (BTC/ES/NQ/etc.) — the equity/crypto-futures complement to the commodity COT."""
    n = 0
    for nm, like in (("btc", "BITCOIN%"), ("es", "E-MINI S&P 500%"), ("nq", "NASDAQ MINI%"), ("ust10", "UST 10Y NOTE%"), ("vix", "VIX FUTURES%")):
        r = _j("https://publicreporting.cftc.gov/resource/gpe5-46if.json", params={"$where": f"market_and_exchange_names like '{like}'", "$limit": 3000, "$order": "report_date_as_yyyy_mm_dd"}).json()
        for x in r:
            try:
                oi = float(x["open_interest_all"])
                ts = pd.Timestamp(x["report_date_as_yyyy_mm_dd"][:10]) + pd.Timedelta(days=3)  # released Friday
                put_point(con, f"cftc_tff_{nm}_levfund_net_pct_oi", (float(x["lev_money_positions_long"]) - float(x["lev_money_positions_short"])) / oi, "cftc", ts.strftime("%Y-%m-%d"))
                put_point(con, f"cftc_tff_{nm}_assetmgr_net_pct_oi", (float(x["asset_mgr_positions_long"]) - float(x["asset_mgr_positions_short"])) / oi, "cftc", ts.strftime("%Y-%m-%d"))
                n += 2
            except Exception:  # noqa: BLE001
                continue
    con.commit()
    return n


LIVE = {"hyperliquid": hyperliquid, "deribit_options": deribit_options, "bitget": bitget, "okx_oi": okx_oi, "orderbooks": orderbooks,
        "polymarket": polymarket, "mempool": mempool, "news": news}
DAILY = {"cftc_tff": cftc_tff}
