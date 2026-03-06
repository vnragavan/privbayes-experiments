"""
metrics/survival/km.py

Kaplan-Meier fidelity metrics. All use lifelines.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def _km_curve(df, duration_col, event_col, timeline):
    kmf = KaplanMeierFitter()
    dur = pd.to_numeric(df[duration_col], errors="coerce").fillna(0)
    evt = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    kmf.fit(dur, evt)
    return kmf.survival_function_at_times(timeline).values


def km_l1_distance(real_df, synth_df, duration_col,
                    event_col, n_grid=200) -> float:
    try:
        t_max = float(pd.to_numeric(
            real_df[duration_col], errors="coerce").max())
        timeline = np.linspace(0, t_max, n_grid)
        sr = _km_curve(real_df, duration_col, event_col, timeline)
        ss = _km_curve(synth_df, duration_col, event_col, timeline)
        dt = timeline[1] - timeline[0]
        return float(np.trapezoid(np.abs(sr - ss), dx=dt) / t_max)
    except Exception as e:
        return float("nan")


def km_l1_stratified(real_df, synth_df, duration_col,
                      event_col, strata_col, n_grid=200) -> dict:
    try:
        strata = real_df[strata_col].unique()
        per = {}
        for s in strata:
            r_s = real_df[real_df[strata_col] == s]
            s_s = synth_df[synth_df[strata_col] == s] \
                if strata_col in synth_df.columns else synth_df
            if len(s_s) < 5:
                continue
            per[str(s)] = km_l1_distance(r_s, s_s, duration_col, event_col, n_grid)
        vals = [v for v in per.values() if not np.isnan(v)]
        return {"per_stratum": per,
                "mean": float(np.mean(vals)) if vals else float("nan")}
    except Exception as e:
        return {"per_stratum": {}, "mean": float("nan"), "error": str(e)}


def logrank_pvalue(real_df, synth_df, duration_col, event_col) -> float:
    try:
        dur_r = pd.to_numeric(real_df[duration_col], errors="coerce").fillna(0)
        evt_r = pd.to_numeric(real_df[event_col], errors="coerce").fillna(0).astype(int)
        dur_s = pd.to_numeric(synth_df[duration_col], errors="coerce").fillna(0)
        evt_s = pd.to_numeric(synth_df[event_col], errors="coerce").fillna(0).astype(int)
        result = logrank_test(dur_r, dur_s, evt_r, evt_s)
        return float(result.p_value)
    except Exception:
        return float("nan")


def km_ci_overlap(real_df, synth_df, duration_col,
                   event_col, n_grid=200) -> float:
    """
    Fraction of time points where the synthetic KM estimate
    falls within the real KM 95% confidence interval.
    """
    try:
        kmf_r = KaplanMeierFitter()
        dur_r = pd.to_numeric(real_df[duration_col], errors="coerce").fillna(0)
        evt_r = pd.to_numeric(real_df[event_col], errors="coerce").fillna(0).astype(int)
        kmf_r.fit(dur_r, evt_r)

        t_max = float(dur_r.max())
        timeline = np.linspace(0, t_max, n_grid)

        ci = kmf_r.confidence_interval_survival_function_
        lower = np.interp(timeline, ci.index, ci.iloc[:, 0].values)
        upper = np.interp(timeline, ci.index, ci.iloc[:, 1].values)
        synth_surv = _km_curve(synth_df, duration_col, event_col, timeline)

        inside = ((synth_surv >= lower) & (synth_surv <= upper)).mean()
        return float(inside)
    except Exception:
        return float("nan")
