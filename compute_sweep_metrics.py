"""
compute_sweep_metrics.py

Computes all metrics from existing synthetic data CSVs in results/eps_sweep/
and saves one JSON results file per (epsilon, seed) combination.

Usage:
    python compute_sweep_metrics.py \
        --schema schemas/rossi_schema.json \
        --real-data data/rossi.csv \
        --sweep-dir results/eps_sweep \
        --out-dir results/eps_sweep

Output:
    results/eps_sweep/results_eps{e}_seed{s}.json
    for each (epsilon, seed) found in the CSV files.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# NumPy 2.0 removed np.trapz — use np.trapezoid if available
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ── Helpers ───────────────────────────────────────────────────────

def safe(fn, *args, **kwargs):
    """Run fn(*args, **kwargs); return {"error": msg, "value": float("nan")} on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": str(e), "value": float("nan")}


def nan_to_none(obj):
    """Recursively replace float nan/inf with None for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return nan_to_none(obj.tolist())
    return obj


def get_bounds(bv):
    if isinstance(bv, dict):
        return bv.get("min"), bv.get("max")
    elif isinstance(bv, (list, tuple)) and len(bv) >= 2:
        return bv[0], bv[1]
    return None, None


# ── Compliance metrics ─────────────────────────────────────────────

KNOWN_IMPLEMENTATIONS = {
    "crn": {
        "n_source": "public",
        "gap_flag": False,
        "n_gaps": 0,
        "undeclared_phases": [],
        "ledger_completeness": 0.6,
        "declared_phases": [
            "structure_learning (eps_struct, exponential mechanism)",
            "cpt_estimation (eps_cpt, Laplace mechanism)",
            "metadata_bounds (eps_disc, when require_public=False)",
        ],
    },
    "dpmm": {
        "n_source": "data_derived",
        "gap_flag": True,
        "n_gaps": 1,
        "undeclared_phases": [
            "priv_tree_binner_epsilon (numeric binning epsilon outside main epsilon total)"
        ],
        "ledger_completeness": 0.3,
        "declared_phases": [
            "structure_learning (epsilon/2, zCDP)",
            "cpt_estimation (remaining epsilon/2, Gaussian mechanism)",
        ],
    },
    "synthcity": {
        "n_source": "data_derived",
        "gap_flag": True,
        "n_gaps": 3,
        "undeclared_phases": [
            "pd.cut encoding (bounds inferred from data before DP)",
            "LabelEncoder.fit (categories inferred from data before DP)",
            "_compute_K uses len(data) — n is data-derived",
        ],
        "ledger_completeness": 0.2,
        "declared_phases": [
            "structure_learning (epsilon/2, exponential mechanism)",
            "cpt_estimation (epsilon/2, Laplace mechanism — but see gaps)",
        ],
    },
}

SYNTHCITY_EPSILON_HALVING = True   # SynthCity splits epsilon/2 each phase


def build_compliance(impl: str, epsilon: float) -> dict:
    info = KNOWN_IMPLEMENTATIONS.get(impl, {})
    declared_eps = (epsilon / 2) if (impl == "synthcity" and SYNTHCITY_EPSILON_HALVING) else epsilon
    return {
        "ledger": {
            "epsilon_total_declared": declared_eps,
            "n_source":              info.get("n_source", "unknown"),
            "gap_flag":              info.get("gap_flag", None),
            "_implementation":       impl,
        },
        "ledger_completeness": info.get("ledger_completeness", 0.0),
        "composition": {
            "declared_phases":                 info.get("declared_phases", []),
            "undeclared_data_dependent_phases": info.get("undeclared_phases", []),
            "gap_flag":                        info.get("gap_flag", None),
            "n_gaps":                          info.get("n_gaps", 0),
        },
    }


# ── Utility metrics ────────────────────────────────────────────────

def marginal_l1(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                schema: dict) -> dict:
    col_types     = schema.get("column_types", {})
    public_bounds = schema.get("public_bounds", {})
    public_cats   = schema.get("public_categories", {})
    per_col = {}

    for col in col_types:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        ctype = col_types[col]
        try:
            if ctype in ("continuous", "integer"):
                bv = public_bounds.get(col)
                if bv is None:
                    continue
                lo, hi = get_bounds(bv)
                if lo is None or hi is None:
                    continue
                lo, hi = float(lo), float(hi)
                k = 16
                bins = np.linspace(lo, hi, k + 1)
                r = pd.to_numeric(real_df[col],  errors="coerce").dropna()
                s = pd.to_numeric(synth_df[col], errors="coerce").dropna()
                rh, _ = np.histogram(r, bins=bins, density=False)
                sh, _ = np.histogram(s, bins=bins, density=False)
                rh = rh / max(rh.sum(), 1)
                sh = sh / max(sh.sum(), 1)
                per_col[col] = float(np.abs(rh - sh).sum() / 2)

            elif ctype in ("categorical", "ordinal", "binary"):
                cats = public_cats.get(col)
                if cats is None:
                    continue
                cats_str = [str(c) for c in cats]
                r = real_df[col].astype(str)
                s = synth_df[col].astype(str)
                rh = r.value_counts(normalize=True).reindex(cats_str, fill_value=0)
                sh = s.value_counts(normalize=True).reindex(cats_str, fill_value=0)
                per_col[col] = float(np.abs(rh.values - sh.values).sum() / 2)
        except Exception:
            pass

    numeric_cols = [c for c, t in col_types.items() if t in ("continuous", "integer")]
    cat_cols     = [c for c, t in col_types.items() if t in ("categorical", "ordinal", "binary")]
    num_vals     = [per_col[c] for c in numeric_cols if c in per_col]
    cat_vals     = [per_col[c] for c in cat_cols     if c in per_col]

    return {
        "per_column":    per_col,
        "mean_numeric":  float(np.mean(num_vals)) if num_vals else float("nan"),
        "mean_categorical": float(np.mean(cat_vals)) if cat_vals else float("nan"),
        "mean_overall":  float(np.mean(list(per_col.values()))) if per_col else float("nan"),
    }


def coverage(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: dict) -> dict:
    public_cats = schema.get("public_categories", {})
    per_col = {}
    for col, cats in public_cats.items():
        if col not in synth_df.columns:
            continue
        cats_str  = set(str(c) for c in cats)
        synth_str = set(synth_df[col].astype(str).unique())
        per_col[col] = len(cats_str & synth_str) / max(len(cats_str), 1)
    return {
        "per_column": per_col,
        "mean": float(np.mean(list(per_col.values()))) if per_col else float("nan"),
    }


def tvd_pairwise(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                  schema: dict) -> dict:
    """
    Mean pairwise joint total variation distance.

    For each pair of columns (i, j), computes TVD on the joint distribution
    P(col_i, col_j) using a 2D histogram:

        TVD = 0.5 * sum_{x,y} | P_real(x,y) - P_synth(x,y) |

    Continuous columns are binned to 8 bins over their public bounds.
    Categorical columns use their declared category levels.
    All pair types handled: numeric-numeric, numeric-categorical,
    categorical-categorical.
    """
    col_types = schema.get("column_types", {})
    pb        = schema.get("public_bounds", {})
    pc        = schema.get("public_categories", {})
    cols      = [c for c in col_types if c in real_df.columns and c in synth_df.columns]

    N_BINS = 8  # bins per numeric axis; 8×8 = 64 cells per pair

    def encode_col(df, col):
        """
        Returns (values_array, bin_edges_or_categories).
        Numeric: digitised to 0..N_BINS-1
        Categorical: mapped to 0..k-1
        Returns None if schema info missing.
        """
        ctype = col_types[col]
        if ctype in ("continuous", "integer"):
            bv = pb.get(col)
            if bv is None:
                return None, None
            lo, hi = get_bounds(bv)
            if lo is None or hi is None:
                return None, None
            lo, hi = float(lo), float(hi)
            vals = pd.to_numeric(df[col], errors="coerce").fillna(lo)
            vals = np.clip(vals.values, lo, hi)
            # map to integer bin indices 0..N_BINS-1
            bins = np.linspace(lo, hi, N_BINS + 1)
            idx  = np.digitize(vals, bins[1:-1])   # 0-based, length N_BINS
            return idx, N_BINS
        else:
            cats = pc.get(col)
            if cats is None:
                return None, None
            cats_str = [str(c) for c in cats]
            cat_map  = {c: i for i, c in enumerate(cats_str)}
            vals     = df[col].astype(str).map(cat_map).fillna(0).astype(int).values
            return vals, len(cats_str)

    def joint_tvd(col_a, col_b):
        """Compute joint TVD for one pair of columns across real and synth."""
        ra, ka = encode_col(real_df,  col_a)
        sa, _  = encode_col(synth_df, col_a)
        rb, kb = encode_col(real_df,  col_b)
        sb, _  = encode_col(synth_df, col_b)

        if ra is None or rb is None:
            return None

        bins = [np.arange(ka + 1) - 0.5,
                np.arange(kb + 1) - 0.5]

        hr, _ = np.histogramdd(np.column_stack([ra, rb]), bins=bins)
        hs, _ = np.histogramdd(np.column_stack([sa, sb]), bins=bins)

        hr = hr / max(hr.sum(), 1)
        hs = hs / max(hs.sum(), 1)

        return float(np.abs(hr - hs).sum() / 2)

    tvds = []
    pair_names = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = joint_tvd(cols[i], cols[j])
            if v is not None:
                tvds.append(v)
                pair_names.append(f"{cols[i]}-{cols[j]}")

    return {
        "mean":    float(np.mean(tvds)) if tvds else float("nan"),
        "max":     float(np.max(tvds))  if tvds else float("nan"),
        "n_pairs": len(tvds),
    }


def mean_wasserstein_per_column(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                                schema: dict) -> dict:
    """Per-column 1-Wasserstein between real and synth, then mean. Lower is better."""
    from scipy.stats import wasserstein_distance
    col_types = schema.get("column_types", {})
    public_cats = schema.get("public_categories", {})
    per_col = {}
    for col in col_types:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        ctype = col_types[col]
        try:
            if ctype in ("continuous", "integer"):
                r = pd.to_numeric(real_df[col], errors="coerce").dropna()
                s = pd.to_numeric(synth_df[col], errors="coerce").dropna()
                if len(r) == 0 or len(s) == 0:
                    continue
                per_col[col] = float(wasserstein_distance(r, s))
            elif ctype in ("categorical", "ordinal", "binary"):
                cats = public_cats.get(col)
                if not cats:
                    continue
                cat_list = [str(c) for c in cats]
                r = real_df[col].astype(str).map(lambda x: cat_list.index(x) if x in cat_list else -1)
                s = synth_df[col].astype(str).map(lambda x: cat_list.index(x) if x in cat_list else -1)
                r = r[r >= 0].astype(float)
                s = s[s >= 0].astype(float)
                if len(r) == 0 or len(s) == 0:
                    continue
                per_col[col] = float(wasserstein_distance(r, s))
            else:
                continue
        except Exception:
            continue
    vals = [v for v in per_col.values() if not (isinstance(v, float) and np.isnan(v))]
    return {
        "per_column": per_col,
        "mean": float(np.mean(vals)) if vals else float("nan"),
    }


def numeric_correlation(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                         schema: dict) -> dict:
    """Spearman rank correlation between numeric columns' rank-correlation matrices."""
    from scipy.stats import spearmanr
    col_types = schema.get("column_types", {})
    num_cols  = [c for c, t in col_types.items()
                 if t in ("continuous", "integer")
                 and c in real_df.columns and c in synth_df.columns]
    if len(num_cols) < 2:
        return {"numeric_spearman": float("nan"),
                "note": "fewer than 2 numeric columns"}

    r_mat = (real_df[num_cols]
             .apply(pd.to_numeric, errors="coerce")
             .corr(method="spearman").values)
    s_mat = (synth_df[num_cols]
             .apply(pd.to_numeric, errors="coerce")
             .corr(method="spearman").values)

    # Upper triangle (excluding diagonal)
    idx  = np.triu_indices(len(num_cols), k=1)
    rho, _ = spearmanr(r_mat[idx], s_mat[idx])
    return {"numeric_spearman": float(rho)}


def tstr_classification(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                         schema: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    target = schema.get("target_spec", {}).get("primary_target")
    if target is None or target not in real_df.columns:
        return {"roc_auc": float("nan"), "error": "no target"}

    kind = schema.get("target_spec", {}).get("kind", "single")
    if kind == "survival_pair":
        return {"roc_auc": float("nan"), "error": "survival_pair target, use C-index instead"}

    feature_cols = [c for c in real_df.columns if c != target]
    X_train = synth_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = synth_df[target]
    X_test  = real_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test  = real_df[target]

    holdout = int(len(real_df) * 0.2)
    X_test, y_test = X_test.iloc[:holdout], y_test.iloc[:holdout]

    if y_train.nunique() < 2:
        return {"roc_auc": float("nan"),
                "error": f"only one class in synth target: {y_train.unique()}"}

    clf = LogisticRegression(max_iter=500, multi_class="ovr", solver="lbfgs")
    clf.fit(X_train, y_train)
    classes = clf.classes_

    if len(classes) == 2:
        prob = clf.predict_proba(X_test)[:, 1]
        auc  = roc_auc_score(y_test, prob)
    else:
        prob = clf.predict_proba(X_test)
        yb   = label_binarize(y_test, classes=classes)
        auc  = roc_auc_score(yb, prob, multi_class="ovr", average="macro")

    return {
        "roc_auc":    float(auc),
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "n_features": len(feature_cols),
    }


# ── Survival metrics ───────────────────────────────────────────────

def get_survival_cols(schema: dict):
    ts = schema.get("target_spec", {})
    if ts.get("kind") != "survival_pair":
        return None, None
    primary  = ts.get("primary_target")
    targets  = ts.get("targets", [])
    duration = next((t for t in targets if t != primary), None)
    return primary, duration   # event_col, duration_col


def km_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: dict) -> dict:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    event_col, dur_col = get_survival_cols(schema)
    if event_col is None or dur_col is None:
        return {"error": "not a survival schema"}

    for df, name in [(real_df, "real"), (synth_df, "synth")]:
        if event_col not in df.columns or dur_col not in df.columns:
            return {"error": f"missing columns in {name}_df"}

    tau = schema.get("target_spec", {}).get("tau")

    r_dur   = pd.to_numeric(real_df[dur_col],    errors="coerce").dropna()
    r_evt   = pd.to_numeric(real_df[event_col],  errors="coerce")
    s_dur   = pd.to_numeric(synth_df[dur_col],   errors="coerce").dropna()
    s_evt   = pd.to_numeric(synth_df[event_col], errors="coerce")

    # Align on common index
    idx_r = r_dur.index.intersection(r_evt.dropna().index)
    idx_s = s_dur.index.intersection(s_evt.dropna().index)
    r_dur, r_evt = r_dur.loc[idx_r], r_evt.loc[idx_r]
    s_dur, s_evt = s_dur.loc[idx_s], s_evt.loc[idx_s]

    kmr = KaplanMeierFitter().fit(r_dur, r_evt, label="real")
    kms = KaplanMeierFitter().fit(s_dur, s_evt, label="synth")

    t_max = float(r_dur.max())
    t_grid = np.linspace(0, t_max, 200)

    sr = kmr.survival_function_at_times(t_grid).values
    ss = kms.survival_function_at_times(t_grid).values
    km_l1 = float(_trapz(np.abs(sr - ss), t_grid) / max(t_max, 1))

    # CI overlap
    ci = kmr.confidence_interval_survival_function_
    ci_lo = np.interp(t_grid, ci.index.values,
                      ci.iloc[:, 0].values)
    ci_hi = np.interp(t_grid, ci.index.values,
                      ci.iloc[:, 1].values)
    inside = ((ss >= ci_lo) & (ss <= ci_hi)).mean()

    # Log-rank
    lr = logrank_test(r_dur, s_dur,
                      event_observed_A=r_evt,
                      event_observed_B=s_evt)

    # RMST
    rmst_results = {}
    taus = [tau] if tau else [t_max]
    for t in taus:
        if t is None:
            continue
        mask_r = r_dur <= t
        mask_s = s_dur <= t
        if mask_r.sum() < 5 or mask_s.sum() < 5:
            rmst_results[str(t)] = float("nan")
            continue
        tg = np.linspace(0, t, 100)
        kmr_t = KaplanMeierFitter().fit(r_dur[mask_r], r_evt[mask_r])
        kms_t = KaplanMeierFitter().fit(s_dur[mask_s], s_evt[mask_s])
        sr_t  = kmr_t.survival_function_at_times(tg).values
        ss_t  = kms_t.survival_function_at_times(tg).values
        rmst_results[str(t)] = float(abs(_trapz(sr_t, tg) - _trapz(ss_t, tg)))

    return {
        "km_l1":              km_l1,
        "km_ci_overlap":      float(inside),
        "logrank_p":          float(lr.p_value),
        "rmst": {
            "per_tau": rmst_results,
            "mean":    float(np.nanmean(list(rmst_results.values())))
                       if rmst_results else float("nan"),
        },
    }


def cox_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: dict) -> dict:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from scipy.stats import spearmanr

    event_col, dur_col = get_survival_cols(schema)
    if event_col is None:
        return {"error": "not a survival schema"}

    feature_cols = [c for c in real_df.columns
                    if c not in (event_col, dur_col)]

    def prep(df, drop_constant=True):
        d = df[[dur_col, event_col] + feature_cols].copy()
        for c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna()
        d[dur_col] = d[dur_col].clip(lower=0.01)
        # Lifelines expects event in {0, 1}; SynthCity/wrappers may output float
        d[event_col] = np.clip(np.round(pd.to_numeric(d[event_col], errors="coerce").fillna(0)), 0, 1).astype(int)
        if drop_constant:
            # Drop zero-variance columns — they make the design matrix singular
            varying = [c for c in feature_cols if d[c].nunique() > 1]
            d = d[[dur_col, event_col] + varying]
        return d

    def fit_cox(df):
        d = prep(df)
        if len(d) < 5:
            raise ValueError("too few rows after prep")
        if d[event_col].nunique() < 2:
            raise ValueError("event column has only one value")
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(d, duration_col=dur_col, event_col=event_col)
        return cph, d

    # Cox Spearman: needs both real and synth to converge
    cox_spearman = float("nan")
    try:
        cph_r, dr = fit_cox(real_df)
        cph_s, ds = fit_cox(synth_df)
        # Compare coefficients on the intersection of non-constant features
        shared = sorted(set(dr.columns) & set(ds.columns) - {dur_col, event_col})
        if len(shared) >= 2:
            coefs_r = cph_r.params_.reindex(shared).fillna(0)
            coefs_s = cph_s.params_.reindex(shared).fillna(0)
            rho, _  = spearmanr(coefs_r, coefs_s)
            cox_spearman = float(rho)
    except Exception as e:
        cph_s = None
        cox_spearman_err = str(e)

    # TSTR C-index: only needs synth Cox model — independent of real Cox
    # Requires synth data to have both event types (0 and 1); otherwise Cox cannot be fitted.
    tstr_cindex = float("nan")
    tstr_cindex_note = None
    try:
        if 'cph_s' not in dir() or cph_s is None:
            _, _ = fit_cox(real_df)    # real fit not needed here; fit synth only
            cph_s, _ = fit_cox(synth_df)
        ho = prep(real_df, drop_constant=False)  # keep all cols for holdout
        # Columns the synth model was trained on (may be empty if no varying covariates)
        cols_for_risk = [c for c in cph_s.params_.index if c in ho.columns] + [dur_col, event_col]
        if len(ho) >= 5 and len(cols_for_risk) >= 2:
            risk = cph_s.predict_partial_hazard(ho[cols_for_risk])
            tstr_cindex = float(concordance_index(ho[dur_col], -risk, ho[event_col]))
    except ValueError as e:
        if "event column has only one value" in str(e):
            tstr_cindex_note = "synth event column has only one value (Cox not fitted)"
        tstr_cindex = float("nan")
    except Exception:
        tstr_cindex = float("nan")

    out = {
        "cox_spearman": cox_spearman,
        "tstr_cindex":  tstr_cindex,
    }
    if tstr_cindex_note:
        out["tstr_cindex_note"] = tstr_cindex_note
    return out


def censoring_and_joint(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                         schema: dict) -> dict:
    from scipy.stats import wasserstein_distance

    event_col, dur_col = get_survival_cols(schema)
    if event_col is None:
        return {"error": "not a survival schema"}

    r_evt = pd.to_numeric(real_df[event_col],  errors="coerce")
    s_evt = pd.to_numeric(synth_df[event_col], errors="coerce")
    r_dur = pd.to_numeric(real_df[dur_col],    errors="coerce")
    s_dur = pd.to_numeric(synth_df[dur_col],   errors="coerce")

    r_cens_rate = float((r_evt == 0).mean())
    s_cens_rate = float((s_evt == 0).mean())
    cens_err    = abs(r_cens_rate - s_cens_rate)

    # Joint survival-censoring Wasserstein
    r_cens_dur = r_dur[r_evt == 0].dropna()
    s_cens_dur = s_dur[s_evt == 0].dropna()
    r_evt_dur  = r_dur[r_evt == 1].dropna()
    s_evt_dur  = s_dur[s_evt == 1].dropna()

    w_cens = float(wasserstein_distance(r_cens_dur, s_cens_dur)) \
             if len(r_cens_dur) > 0 and len(s_cens_dur) > 0 else float("nan")
    w_evt  = float(wasserstein_distance(r_evt_dur,  s_evt_dur)) \
             if len(r_evt_dur)  > 0 and len(s_evt_dur)  > 0 else float("nan")

    mean_w = float(np.nanmean([w_cens, w_evt]))

    return {
        "censoring_rate_error":     cens_err,
        "joint_survival_censoring": {
            "censored_duration_wasserstein": w_cens,
            "event_duration_wasserstein":    w_evt,
            "mean":                          mean_w,
        },
    }


# ── Privacy metrics ────────────────────────────────────────────────

def privacy_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    num_real  = real_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    num_synth = synth_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    cols = [c for c in num_real.columns if c in num_synth.columns]
    num_real  = num_real[cols]
    num_synth = num_synth[cols]

    # Split real into train/holdout
    n = len(num_real)
    half = n // 2
    train_real   = num_real.iloc[:half]
    holdout_real = num_real.iloc[half:half + half]

    from sklearn.neighbors import NearestNeighbors

    # MIA — graceful fallback when synthetic data has collapsed to one class
    try:
        n_att = min(len(train_real), len(holdout_real), len(num_synth))
        X_mia = pd.concat([
            train_real.iloc[:n_att],
            holdout_real.iloc[:n_att],
        ])
        y_mia = np.array([1]*n_att + [0]*n_att)

        nn = NearestNeighbors(n_neighbors=1).fit(num_synth)
        dists, _ = nn.kneighbors(X_mia)
        features = dists.flatten().reshape(-1, 1)

        nn_real = NearestNeighbors(n_neighbors=2).fit(num_real)
        d_real, _ = nn_real.kneighbors(X_mia)
        d_real = d_real[:, 1].reshape(-1, 1)
        ratio_feat = features / (d_real + 1e-10)

        # Degenerate check: zero-variance features or only one label class
        degenerate = (ratio_feat.std() < 1e-10 or len(np.unique(y_mia)) < 2)
        if degenerate:
            mia = {"auc": 0.5, "advantage": 0.0,
                   "n_train_in_attack": n_att,
                   "n_holdout_in_attack": n_att,
                   "note": "degenerate features or single-class labels; MIA at chance"}
        else:
            # Train and score on the full combined set
            clf  = LogisticRegression(max_iter=200).fit(ratio_feat, y_mia)
            prob = clf.predict_proba(ratio_feat)[:, 1]
            auc  = float(roc_auc_score(y_mia, prob))
            adv  = round(abs(auc - 0.5), 4)
            mia  = {"auc": auc, "advantage": adv,
                    "n_train_in_attack": n_att,
                    "n_holdout_in_attack": n_att}
    except Exception as e:
        mia = {"auc": float("nan"), "error": str(e)}

    # NNDR — cap ratio at 100 to avoid billion-scale values when
    # synthetic records are identical (d_s ≈ 0, d_r ≈ 0 → 0/0)
    try:
        nn_s = NearestNeighbors(n_neighbors=1).fit(num_synth)
        nn_r = NearestNeighbors(n_neighbors=2).fit(num_real)
        d_s, _ = nn_s.kneighbors(num_real)
        d_r, _ = nn_r.kneighbors(num_real)
        d_s = d_s[:, 0]
        d_r = d_r[:, 1]
        # Use max(d_r, 1e-6) to avoid division by near-zero distances
        ratio    = d_s / np.maximum(d_r, 1e-6)
        ratio    = np.clip(ratio, 0, 100)   # cap at 100× for interpretability
        mean_r   = float(np.mean(ratio))
        median_r = float(np.median(ratio))
        frac_b1  = float((ratio < 1).mean())
        interp   = ("memorisation risk: synthetic records closer to real than real-to-real"
                    if frac_b1 > 0.3 else "low utility: synthetic records far from real data"
                    if mean_r > 2 else "healthy generalisation")
        nndr = {"mean_ratio": mean_r, "median_ratio": median_r,
                "fraction_below_1": frac_b1, "interpretation": interp}
    except Exception as e:
        nndr = {"mean_ratio": float("nan"), "error": str(e)}

    return {"mia": mia, "nndr": nndr}


# ── Constraint metrics ─────────────────────────────────────────────

def constraint_metrics(synth_df: pd.DataFrame, schema: dict) -> dict:
    constraints = schema.get("constraints", {})
    n = len(synth_df)
    violations = pd.Series(False, index=synth_df.index)
    col_results   = {}
    cross_results = {}

    for col, spec in constraints.get("column_constraints", {}).items():
        if col not in synth_df.columns:
            continue
        mask = pd.Series(False, index=synth_df.index)
        vals = pd.to_numeric(synth_df[col], errors="coerce")
        if "min_exclusive" in spec and spec["min_exclusive"] is not None:
            mask |= vals <= float(spec["min_exclusive"])
        if "min" in spec and spec["min"] is not None:
            mask |= vals < float(spec["min"])
        if "max" in spec and spec["max"] is not None:
            mask |= vals > float(spec["max"])
        col_results[col] = float(mask.mean())
        violations |= mask

    for c in constraints.get("cross_column_constraints", []):
        if c.get("type") == "survival_pair":
            ec      = c.get("event_col")
            tc      = c.get("time_col")
            allowed = c.get("event_allowed_values", [0, 1])
            name    = c.get("name", "survival_validity")
            if ec in synth_df.columns and tc in synth_df.columns:
                mask = (~synth_df[ec].isin(allowed) |
                        (pd.to_numeric(synth_df[tc], errors="coerce") <= 0))
                cross_results[name] = float(mask.mean())
                violations |= mask

    return {
        "overall_violation_rate":        float(violations.mean()),
        "n_records_total":               n,
        "n_records_with_any_violation":  int(violations.sum()),
        "column_constraints":            col_results,
        "cross_column_constraints":      cross_results,
    }


# ── Performance (estimated from CSV file size) ─────────────────────

def stub_performance() -> dict:
    """Performance cannot be measured post-hoc; return None stubs."""
    return {
        "fit_time_sec":           None,
        "sample_time_sec":        None,
        "total_time_sec":         None,
        "peak_memory_fit_mb":     None,
        "peak_memory_max_mb":     None,
        "note": "Performance not measured during sweep CSV generation. "
                "Re-run with run_experiment.py for timing data.",
    }


# ── Per-CSV metrics computation ────────────────────────────────────

def compute_metrics_for_csv(csv_path: Path,
                             real_df: pd.DataFrame,
                             schema: dict,
                             impl: str,
                             epsilon: float) -> dict:
    synth_df = pd.read_csv(csv_path)

    # Clip numeric columns to schema bounds (same as wrapper does)
    pb = schema.get("public_bounds", {})
    ct = schema.get("column_types", {})
    for col, bv in pb.items():
        if col not in synth_df.columns:
            continue
        if ct.get(col) not in ("continuous", "integer"):
            continue
        lo, hi = get_bounds(bv)
        if lo is not None and hi is not None:
            synth_df[col] = pd.to_numeric(
                synth_df[col], errors="coerce"
            ).clip(float(lo), float(hi))

    result = {
        "implementation": impl,
        "epsilon":        epsilon,
        "csv_path":       str(csv_path),
        "status":         "ok",
    }

    # Compliance
    result["compliance"] = build_compliance(impl, epsilon)

    # Utility
    marg   = safe(marginal_l1,        real_df, synth_df, schema)
    cov    = safe(coverage,           real_df, synth_df, schema)
    tvd    = safe(tvd_pairwise,       real_df, synth_df, schema)
    corr   = safe(numeric_correlation, real_df, synth_df, schema)
    wass   = safe(mean_wasserstein_per_column, real_df, synth_df, schema)
    tstr   = safe(tstr_classification, real_df, synth_df, schema)
    result["utility"] = {
        "marginal":    marg,
        "coverage":    cov,
        "tvd":         tvd,
        "correlation": corr,
        "wasserstein": wass,
        "tstr":        tstr,
    }

    # Survival
    km  = safe(km_metrics,          real_df, synth_df, schema)
    cox = safe(cox_metrics,         real_df, synth_df, schema)
    cen = safe(censoring_and_joint, real_df, synth_df, schema)
    result["survival"] = {**km, **cox, **cen}

    # Privacy
    result["privacy"] = safe(privacy_metrics, real_df, synth_df)
    if isinstance(result.get("privacy"), dict) and "error" not in result["privacy"]:
        from metrics.privacy.attribute_inference import (
            attribute_inference_auc,
            get_attribute_inference_target,
        )
        attr_target = get_attribute_inference_target(schema)
        result["privacy"]["attribute_inference"] = safe(
            attribute_inference_auc, real_df, synth_df, attr_target
        )

    # Constraints
    result["constraints"] = safe(constraint_metrics, synth_df, schema)

    # Performance (stub)
    result["performance"] = stub_performance()

    return result


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema",    required=True)
    parser.add_argument("--real-data", required=True)
    parser.add_argument("--sweep-dir", default="results/eps_sweep")
    parser.add_argument("--out-dir",   default="results/eps_sweep")
    args = parser.parse_args()

    with open(args.schema) as f:
        schema = json.load(f)

    real_df = pd.read_csv(args.real_data)
    print(f"Loaded real data: {real_df.shape}")

    sweep_dir = Path(args.sweep_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSVs matching pattern {impl}_eps{e}_seed{s}.csv
    csv_files = sorted(sweep_dir.glob("*.csv"))
    pattern   = re.compile(r"^(.+)_eps([\d.]+)_seed(\d+)\.csv$")

    # Group by (epsilon, seed)
    groups: Dict = {}
    for csv_path in csv_files:
        m = pattern.match(csv_path.name)
        if not m:
            continue
        impl, eps_str, seed_str = m.group(1), m.group(2), m.group(3)
        key = (float(eps_str), int(seed_str))
        groups.setdefault(key, {})[impl] = csv_path

    print(f"Found {len(csv_files)} CSVs across {len(groups)} (epsilon, seed) groups")

    total = sum(len(v) for v in groups.values())
    done  = 0

    for (epsilon, seed), impl_paths in sorted(groups.items()):
        out_path = out_dir / f"results_eps{epsilon}_seed{seed}.json"
        results  = {}

        # Preserve run-time compliance.ledger and performance from existing file when present
        existing_ledger = {}
        existing_perf = {}
        if out_path.exists():
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                for impl in impl_paths:
                    impl_data = existing.get(impl) or {}
                    comp = impl_data.get("compliance")
                    if isinstance(comp, dict) and isinstance(comp.get("ledger"), dict):
                        existing_ledger[impl] = comp["ledger"]
                    perf = impl_data.get("performance")
                    if isinstance(perf, dict) and (perf.get("fit_time_sec") is not None or perf.get("sample_time_sec") is not None):
                        existing_perf[impl] = perf
            except Exception:
                pass

        for impl, csv_path in sorted(impl_paths.items()):
            done += 1
            print(f"[{done}/{total}] {impl} eps={epsilon} seed={seed} ...",
                  end=" ", flush=True)
            t0 = time.time()
            try:
                metrics = compute_metrics_for_csv(
                    csv_path, real_df, schema, impl, epsilon)
                if impl in existing_ledger:
                    metrics["compliance"]["ledger"] = existing_ledger[impl]
                if impl in existing_perf:
                    metrics["performance"] = existing_perf[impl]
                results[impl] = metrics
                print(f"OK ({time.time()-t0:.1f}s)")
            except Exception as e:
                results[impl] = {"status": "error", "error": str(e)}
                print(f"ERROR: {e}")
                traceback.print_exc()

        with open(out_path, "w") as f:
            json.dump(nan_to_none(results), f, indent=2)
        print(f"  → saved {out_path}")

    print(f"\nDone. {len(groups)} result files written to {out_dir}/")


if __name__ == "__main__":
    main()
