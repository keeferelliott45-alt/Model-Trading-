"""Run collectors; each isolated so one dead source never blocks the rest. Usage: collect.py [--backfill] [--daily]"""
import sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from store.db import connect, now
from collectors.snapshots import LIVE, DAILY

def main():
    con = connect()
    jobs = dict(LIVE)
    if "--daily" in sys.argv or "--backfill" in sys.argv:
        jobs.update(DAILY)
    if "--backfill" in sys.argv:
        from collectors import backfill
        jobs["backfill"] = backfill.run
    bad = 0
    for name, fn in jobs.items():
        t = time.time()
        try:
            n = fn(con); ok, err = 1, ""
        except Exception as e:  # noqa: BLE001
            n, ok, err = 0, 0, f"{type(e).__name__}: {e}"[:300]; bad += 1
            traceback.print_exc()
        con.execute("INSERT INTO collector_runs VALUES (?,?,?,?,?,?)", (now(), name, ok, n, round(time.time() - t, 1), err))
        con.commit()
        print(f"{'OK ' if ok else 'ERR'} {name:16s} rows={n} {time.time()-t:.1f}s {err}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
