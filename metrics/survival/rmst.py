"""
metrics/survival/rmst.py

Restricted Mean Survival Time error.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def _rmst(df, duration_col, event_col, tau):
    kmf = KaplanMeierFitter()
    dur = pd.to_numeric(df[duration_col], errors="coerce").fillna(0)
    evt = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    kmf.fit(dur, evt)
    sf = kmf.survival_function_
    t = sf.index.values
    s = sf["KM_estimate"].values
    mask = t <= tau
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapezoid(s[mask], t[mask]))


def rmst_error(real_df, synth_df, duration_col,
                event_col, tau) -> float:
    try:
        r = _rmst(real_df, duration_col, event_col, tau)
        s = _rmst(synth_df, duration_col, event_col, tau)
        return float(abs(r - s))
    except Exception:
        return float("nan")


def rmst_error_multiple_taus(real_df, synth_df,
                               duration_col, event_col,
                               taus: list) -> dict:
    per = {}
    for tau in taus:
        per[str(tau)] = rmst_error(real_df, synth_df,
                                    duration_col, event_col, tau)
    vals = [v for v in per.values() if not np.isnan(v)]
    return {
        "per_tau": per,
        "mean": float(np.mean(vals)) if vals else float("nan"),
    }
