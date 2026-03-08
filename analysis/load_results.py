"""
analysis/load_results.py

Loads all experiment JSON files and flattens them to DataFrames.
"""

import json
import glob
import os
import re
import numpy as np
import pandas as pd
from scipy import stats


def _safe_get(d, *keys, default=float("nan")):
    try:
        for k in keys:
            d = d[k]
        if isinstance(d, dict) and "error" in d:
            return float("nan")
        return d
    except (KeyError, TypeError):
        return default


def _flatten_result(impl_result: dict, impl: str, epsilon: float, seed: int) -> dict:
    r = impl_result
    row = {
        "implementation": impl,
        "epsilon":        epsilon,
        "seed":           seed,
        "status":         r.get("status", "ok"),
    }

    row["fit_time_sec"]    = _safe_get(r, "performance", "fit_time_sec")
    row["sample_time_sec"] = _safe_get(r, "performance", "sample_time_sec")
    row["peak_memory_mb"]  = _safe_get(r, "performance", "peak_memory_max_mb")

    row["km_l1"]           = _safe_get(r, "survival", "km_l1")
    row["km_ci_overlap"]   = _safe_get(r, "survival", "km_ci_overlap")
    row["logrank_p"]       = _safe_get(r, "survival", "logrank_p")
    row["cox_spearman"]    = _safe_get(r, "survival", "cox_spearman")
    row["tstr_cindex"]     = _safe_get(r, "survival", "tstr_cindex")
    row["censoring_err"]   = _safe_get(r, "survival", "censoring_rate_error")
    row["joint_censoring"] = _safe_get(r, "survival", "joint_survival_censoring", "mean")
    row["rmst"]            = _safe_get(r, "survival", "rmst", "mean")

    row["marginal_l1"]     = _safe_get(r, "utility", "marginal", "mean_overall")
    row["tvd_mean"]        = _safe_get(r, "utility", "tvd", "mean")
    row["corr_spearman"]   = _safe_get(r, "utility", "correlation", "numeric_spearman")
    row["tstr_auc"]        = _safe_get(r, "utility", "tstr", "roc_auc")
    row["cat_coverage"]    = _safe_get(r, "utility", "coverage", "mean")
    row["wasserstein_mean"] = _safe_get(r, "utility", "wasserstein", "mean")
    row["unk_rate"]        = _safe_get(r, "utility", "unknown_token_rate", "overall")

    row["mia_auc"]         = _safe_get(r, "privacy", "mia", "auc")
    row["mia_advantage"]   = _safe_get(r, "privacy", "mia", "advantage")
    row["nndr_mean"]       = _safe_get(r, "privacy", "nndr", "mean_ratio")
    row["nndr_median"]     = _safe_get(r, "privacy", "nndr", "median_ratio")
    row["nndr_frac_below"] = _safe_get(r, "privacy", "nndr", "fraction_below_1")
    row["attr_inference_auc"] = _safe_get(r, "privacy", "attribute_inference", "auc")

    row["constraint_viol"] = _safe_get(r, "constraints", "overall_violation_rate")
    row["constraint_viol_event"] = _safe_get(r, "constraints", "survival_pair_event_violation_rate")
    row["constraint_viol_time"] = _safe_get(r, "constraints", "survival_pair_time_violation_rate")
    row["ledger_complete"] = _safe_get(r, "compliance", "ledger_completeness")
    row["composition_gap"] = _safe_get(r, "compliance", "composition", "gap_flag")

    return row


def load_all_results(results_dir: str = "results/eps_sweep") -> pd.DataFrame:
    pattern = os.path.join(results_dir, "results_eps*_seed*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"[load_results] No result files found in {results_dir}")
        return pd.DataFrame()

    rows = []
    for path in files:
        fname = os.path.basename(path)
        m = re.search(r"eps([\d.]+)_seed(\d+)", fname)
        if not m:
            continue
        eps  = float(m.group(1))
        seed = int(m.group(2))

        with open(path) as f:
            data = json.load(f)

        for impl, impl_result in data.items():
            rows.append(_flatten_result(impl_result, impl, eps, seed))

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns
                   if c not in ("implementation", "status")]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def aggregate_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-seed rows into one row per (implementation, epsilon).

    For each metric column the following suffixed columns are produced:

      _mean   — mean across seeds
      _std    — sample standard deviation (ddof=1)
      _var    — sample variance (ddof=1)
      _se     — standard error  =  std / sqrt(n_valid_seeds)
      _ci_lo  — lower bound of 95 % t-CI  (df = n_valid_seeds − 1)
      _ci_hi  — upper bound of 95 % t-CI

    Note on seed count
    ------------------
    With n=3 seeds the t critical value is t_{0.025, df=2} = 4.303.
    This makes 95 % CI bands very wide.  Figures use _se bands (mean ± 1 SE)
    by default; tables show mean ± SE.  The _ci_lo/_ci_hi columns are
    available for supplementary material or reviewer requests.
    """
    if df.empty:
        return df

    metric_cols = [c for c in df.columns
                   if c not in ("implementation", "epsilon", "seed", "status")]
    grouped = df.groupby(["implementation", "epsilon"])

    mean_df = grouped[metric_cols].mean().add_suffix("_mean")
    std_df  = grouped[metric_cols].std(ddof=1).add_suffix("_std")
    var_df  = grouped[metric_cols].var(ddof=1).add_suffix("_var")
    n_df    = grouped[metric_cols].count()   # non-NaN count per cell

    # SE = std / sqrt(n)
    se_values = std_df.values / np.where(n_df.values > 0,
                                          np.sqrt(n_df.values),
                                          np.nan)
    se_df = pd.DataFrame(
        se_values,
        index=std_df.index,
        columns=[c.replace("_std", "_se") for c in std_df.columns],
    )

    # 95 % CI via t-distribution — t_crit per cell based on actual n
    ci_lo = np.full_like(mean_df.values, np.nan, dtype=float)
    ci_hi = np.full_like(mean_df.values, np.nan, dtype=float)

    for col_idx, base_col in enumerate(metric_cols):
        mean_col = f"{base_col}_mean"
        se_col   = f"{base_col}_se"
        if mean_col not in mean_df.columns or se_col not in se_df.columns:
            continue
        mc_idx = list(mean_df.columns).index(mean_col)
        se_idx = list(se_df.columns).index(se_col)
        for row_idx in range(len(mean_df)):
            n   = n_df.iloc[row_idx][base_col]
            mu  = mean_df.iloc[row_idx, mc_idx]
            se  = se_df.iloc[row_idx, se_idx]
            if n < 2 or np.isnan(mu) or np.isnan(se):
                continue
            t_crit = stats.t.ppf(0.975, df=int(n) - 1)
            ci_lo[row_idx, col_idx] = mu - t_crit * se
            ci_hi[row_idx, col_idx] = mu + t_crit * se

    ci_lo_df = pd.DataFrame(
        ci_lo, index=mean_df.index,
        columns=[f"{c}_ci_lo" for c in metric_cols])
    ci_hi_df = pd.DataFrame(
        ci_hi, index=mean_df.index,
        columns=[f"{c}_ci_hi" for c in metric_cols])

    return (mean_df
            .join(std_df)
            .join(var_df)
            .join(se_df)
            .join(ci_lo_df)
            .join(ci_hi_df)
            .reset_index())


def get_at_epsilon(df: pd.DataFrame, epsilon: float,
                    agg: bool = True) -> pd.DataFrame:
    sub = df[df["epsilon"] == epsilon]
    if agg:
        return aggregate_over_seeds(sub)
    return sub


def load_compliance_results(results_dir: str = "results/compliance") -> dict:
    out = {}

    static_path = os.path.join(results_dir, "static_analysis.csv")
    if os.path.exists(static_path):
        out["static_analysis"] = pd.read_csv(static_path)
    else:
        out["static_analysis"] = pd.DataFrame()

    for fname in ["cross_split_variance.json", "leave_one_out.json"]:
        path = os.path.join(results_dir, fname)
        key  = fname.replace(".json", "")
        if os.path.exists(path):
            with open(path) as f:
                out[key] = json.load(f)
        else:
            out[key] = {}

    return out
