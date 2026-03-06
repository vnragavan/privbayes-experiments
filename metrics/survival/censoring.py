"""
metrics/survival/censoring.py

Joint survival-censoring fidelity — the key differentiating test.
Checks whether the (duration, event) joint structure is preserved,
not just each column's marginal.
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def censoring_rate_error(real_df, synth_df, event_col) -> float:
    try:
        r_evt = pd.to_numeric(real_df[event_col], errors="coerce").fillna(0)
        s_evt = pd.to_numeric(synth_df[event_col], errors="coerce").fillna(0)
        cr_real = 1.0 - r_evt.mean()
        cr_synth = 1.0 - s_evt.mean()
        return float(abs(cr_real - cr_synth))
    except Exception:
        return float("nan")


def joint_survival_censoring(real_df, synth_df,
                               duration_col, event_col) -> dict:
    """
    Wasserstein distance on duration distribution separately for
    censored and event subgroups.

    This is the key test for structural joint understanding of (duration, event).
    A model that treats duration and event as independent columns will produce
    the wrong joint distribution even when each marginal looks correct.

    Expected to differentiate CRNPrivBayes from SynthCity most clearly because:
    - CRNPrivBayes: survival columns are label_columns with fixed joint vocabulary
    - SynthCity: pd.cut treats duration independently of event
    """
    try:
        r_evt = pd.to_numeric(real_df[event_col], errors="coerce").fillna(0).astype(int)
        r_dur = pd.to_numeric(real_df[duration_col], errors="coerce").fillna(0)
        s_evt = pd.to_numeric(synth_df[event_col], errors="coerce").fillna(0).astype(int)
        s_dur = pd.to_numeric(synth_df[duration_col], errors="coerce").fillna(0)

        results = {}
        for label, evt_val in [("censored", 0), ("event", 1)]:
            r_sub = r_dur[r_evt == evt_val].values
            s_sub = s_dur[s_evt == evt_val].values
            if len(r_sub) < 2 or len(s_sub) < 2:
                results[f"{label}_duration_wasserstein"] = float("nan")
            else:
                results[f"{label}_duration_wasserstein"] = float(
                    wasserstein_distance(r_sub, s_sub))

        vals = [v for v in results.values() if not np.isnan(v)]
        results["mean"] = float(np.mean(vals)) if vals else float("nan")
        return results
    except Exception as e:
        return {
            "censored_duration_wasserstein": float("nan"),
            "event_duration_wasserstein": float("nan"),
            "mean": float("nan"),
            "error": str(e),
        }
