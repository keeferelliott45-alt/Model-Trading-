"""Global experiment ledger: every hypothesis ever tested; BH-correction applied over the WHOLE ledger."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

LEDGER = Path(__file__).resolve().parents[1] / "reports" / "ledger.csv"


def append(df: pd.DataFrame) -> pd.DataFrame:
    LEDGER.parent.mkdir(exist_ok=True)
    old = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
    if not old.empty and "exp" in df:
        old = old[~old["exp"].isin(df["exp"].unique())]
    led = pd.concat([old, df], ignore_index=True)
    led = led.drop(columns=[c for c in ("p_fdr_global", "p_bonf_global") if c in led])
    led["p_fdr_global"] = multipletests(led["p_value"].fillna(1), method="fdr_bh")[1]
    led["p_bonf_global"] = multipletests(led["p_value"].fillna(1), method="bonferroni")[1]
    led.to_csv(LEDGER, index=False)
    return led
