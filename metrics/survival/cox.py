"""
metrics/survival/cox.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter


def _fit_cox(df, duration_col, event_col, covariates):
    cols = [c for c in covariates
            if c in df.columns and c != duration_col and c != event_col]
    sub = df[cols + [duration_col, event_col]].copy()
    sub[cols] = sub[cols].apply(pd.to_numeric, errors="coerce")
    sub[duration_col] = pd.to_numeric(sub[duration_col], errors="coerce")
    sub[event_col] = pd.to_numeric(sub[event_col], errors="coerce").astype(int)
    sub = sub.dropna()
    # Drop zero-variance columns
    sub = sub[[c for c in sub.columns
               if c in [duration_col, event_col] or sub[c].std() > 0]]
    cph = CoxPHFitter()
    cph.fit(sub, duration_col=duration_col, event_col=event_col)
    return cph


def cox_coefficient_spearman(real_df, synth_df, duration_col,
                               event_col, covariates) -> float:
    try:
        cph_r = _fit_cox(real_df, duration_col, event_col, covariates)
        cph_s = _fit_cox(synth_df, duration_col, event_col, covariates)
        shared = list(set(cph_r.params_.index) & set(cph_s.params_.index))
        if len(shared) < 2:
            return float("nan")
        r_coef = cph_r.params_[shared].values
        s_coef = cph_s.params_[shared].values
        return float(stats.spearmanr(r_coef, s_coef).statistic)
    except Exception:
        return float("nan")


def cox_hr_bias(real_df, synth_df, duration_col,
                 event_col, covariates) -> dict:
    try:
        cph_r = _fit_cox(real_df, duration_col, event_col, covariates)
        cph_s = _fit_cox(synth_df, duration_col, event_col, covariates)
        shared = list(set(cph_r.params_.index) & set(cph_s.params_.index))
        hr_r = np.exp(cph_r.params_[shared])
        hr_s = np.exp(cph_s.params_[shared])
        bias = np.abs(hr_r - hr_s) / np.abs(hr_r).clip(1e-9)
        per = {k: float(v) for k, v in bias.items()}
        return {"per_covariate": per,
                "mean": float(bias.mean())}
    except Exception as e:
        return {"per_covariate": {}, "mean": float("nan"), "error": str(e)}


def tstr_cindex(synth_df, test_real_df, duration_col,
                 event_col, covariates) -> float:
    try:
        cph = _fit_cox(synth_df, duration_col, event_col, covariates)
        cols = [c for c in covariates
                if c in test_real_df.columns
                and c != duration_col and c != event_col]
        sub = test_real_df[cols + [duration_col, event_col]].copy()
        sub[cols] = sub[cols].apply(pd.to_numeric, errors="coerce")
        sub[duration_col] = pd.to_numeric(sub[duration_col], errors="coerce")
        sub[event_col] = pd.to_numeric(sub[event_col], errors="coerce").astype(int)
        sub = sub.dropna()
        return float(cph.score(sub, scoring_method="concordance_index"))
    except Exception:
        return float("nan")
