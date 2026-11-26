"""Information store: one SQLite file, many purpose-built tables. Everything timestamped, nothing overwritten silently."""
from __future__ import annotations

import gzip
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "info.db"
RAW = ROOT / "data" / "raw"

SCHEMA = """
CREATE TABLE IF NOT EXISTS series(name TEXT, ts TEXT, value REAL, source TEXT, PRIMARY KEY(name, ts));      -- every numeric time series (prices, macro, onchain, derivs, odds)
CREATE TABLE IF NOT EXISTS snapshots(source TEXT, ts TEXT, key TEXT, payload TEXT, PRIMARY KEY(source, ts, key)); -- point-in-time structured JSON that can never be backfilled
CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, ts TEXT, kind TEXT, entity TEXT, text TEXT, url TEXT, meta TEXT); -- news, geopolitics, listings, hacks, regulation
CREATE TABLE IF NOT EXISTS signals(ts TEXT, model TEXT, asset TEXT, direction TEXT, score REAL, horizon INT, why TEXT, features TEXT, outcome REAL, outcome_ts TEXT); -- every call ever made, win or loss
CREATE TABLE IF NOT EXISTS hypotheses(id TEXT PRIMARY KEY, ts TEXT, statement TEXT, status TEXT, n_tests INT, best_p REAL, best_p_fdr REAL, verdict TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS notes(ts TEXT, topic TEXT, body TEXT);                                              -- findings, decisions, lessons, dead ends
CREATE TABLE IF NOT EXISTS sources(name TEXT PRIMARY KEY, kind TEXT, url TEXT, cost TEXT, limits TEXT, backfill TEXT, status TEXT, last_ok TEXT, last_err TEXT);
CREATE TABLE IF NOT EXISTS collector_runs(ts TEXT, collector TEXT, ok INT, rows INT, secs REAL, err TEXT);
CREATE INDEX IF NOT EXISTS idx_series_ts ON series(ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts, kind);
"""


def now() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def put_series(con, name: str, s: pd.Series, source: str) -> int:
    s = s.dropna()
    rows = [(name, pd.Timestamp(i).strftime("%Y-%m-%dT%H:%M:%SZ") if getattr(i, "hour", 0) or getattr(i, "minute", 0) else pd.Timestamp(i).strftime("%Y-%m-%d"), float(v), source) for i, v in s.items()]
    con.executemany("INSERT OR REPLACE INTO series VALUES (?,?,?,?)", rows)
    con.commit()
    return len(rows)


def put_point(con, name: str, value: float, source: str, ts: str | None = None):
    con.execute("INSERT OR REPLACE INTO series VALUES (?,?,?,?)", (name, ts or now(), float(value), source))


def put_snapshot(con, source: str, key: str, payload, ts: str | None = None):
    con.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?)", (source, ts or now(), key, json.dumps(payload, separators=(",", ":"))))


def put_event(con, id_: str, ts: str, kind: str, entity: str, text: str, url: str = "", meta: dict | None = None) -> bool:
    cur = con.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?)", (id_, ts, kind, entity, text, url, json.dumps(meta or {})))
    return cur.rowcount > 0


def put_note(con, topic: str, body: str):
    con.execute("INSERT INTO notes VALUES (?,?,?)", (now(), topic, body))
    con.commit()


def archive_raw(source: str, payload) -> None:
    """Append-only compressed raw archive (day-partitioned) so nothing is ever lost even if parsing changes later."""
    d = RAW / source
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / f"{pd.Timestamp.now('UTC').strftime('%Y-%m-%d')}.jsonl.gz", "ab") as f:
        f.write((json.dumps({"ts": now(), "data": payload}, separators=(",", ":")) + "\n").encode())


def series(con, name: str) -> pd.Series:
    df = pd.read_sql_query("SELECT ts, value FROM series WHERE name=? ORDER BY ts", con, params=(name,))
    return pd.Series(df.value.values, index=pd.to_datetime(df.ts).tz_localize(None), name=name) if len(df) else pd.Series(dtype=float, name=name)
