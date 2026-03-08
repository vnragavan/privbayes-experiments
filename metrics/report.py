"""
metrics/report.py

Assembles all metric families into one result dict for a single
(implementation, epsilon, seed) run.

Every metric is wrapped in safe() so a failure in one metric
never aborts the others.
"""

from __future__ import annotations
import numpy as np

from metrics.performance.tracker import PerformanceTracker
from metrics.utility.marginal import mean_marginal_l1, pairwise_tvd, mean_wasserstein_per_column
from metrics.utility.correlation import mixed_association_similarity
from metrics.utility.tstr import tstr_classification
from metrics.utility.coverage import categorical_coverage, unknown_token_rate
from metrics.survival.km import (
    km_l1_distance, logrank_pvalue, km_ci_overlap)
from metrics.survival.cox import (
    cox_coefficient_spearman, tstr_cindex)
from metrics.survival.censoring import (
    censoring_rate_error, joint_survival_censoring)
from metrics.survival.rmst import rmst_error, rmst_error_multiple_taus
from metrics.compliance.ledger import build_ledger, ledger_completeness_score
from metrics.compliance.composition import composition_summary
from metrics.privacy.mia import mia_nearest_neighbour
from metrics.privacy.nndr import nearest_neighbour_distance_ratio
from metrics.privacy.attribute_inference import (
    attribute_inference_auc,
    get_attribute_inference_target,
)
from metrics.constraint.validator import constraint_violation_summary
from adapters.schema_normalization import normalize_to_schema_output


def safe(fn, *args, **kwargs):
    """Call fn. On any exception return {'error': msg, 'value': nan}."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": str(e), "value": float("nan")}


def compute_metrics(
    real_df,
    synth_df,
    schema: dict,
    privacy_report: dict,
    implementation: str,
    performance: dict = None,
    test_real_df=None,
    train_df=None,
    holdout_df=None,
    strata_col: str = None,
    taus: list = None,
) -> dict:
    """
    Compute all metric families.

    Parameters
    ----------
    real_df          : training split used to fit the synthesiser
    synth_df         : synthetic output from sample()
    schema           : schema-generator JSON dict
    privacy_report   : output of wrapper.privacy_report()
    implementation   : "crn" | "dpmm" | "synthcity"
    performance      : output of PerformanceTracker.summary()
    test_real_df     : held-out real data for TSTR evaluation
    train_df         : same as real_df — passed explicitly for MIA
    holdout_df       : records NOT used during fit — for MIA
    strata_col       : optional column for stratified KM
    taus             : list of RMST horizons; defaults to schema tau if set
    """
    synth_df = normalize_to_schema_output(
        synth_df,
        schema,
        fit_columns=list(real_df.columns),
    )

    # --- Extract survival columns and privacy targets from schema ---
    ts = schema.get("target_spec", {})
    event_col = ts.get("primary_target")
    targets = ts.get("targets", [])
    duration_col = next((t for t in targets if t != event_col), None)
    attr_inference_target = get_attribute_inference_target(schema)
    col_types = schema.get("column_types", {})
    exclude = {event_col, duration_col}
    covariates = [c for c in col_types if c not in exclude]

    numeric_cols = [c for c, t in col_types.items()
                    if t in ("continuous", "integer") and c not in exclude]

    # --- Ledger ---
    ledger = build_ledger(privacy_report, implementation)

    # --- RMST taus ---
    schema_tau = ts.get("tau")
    if taus is None:
        taus = [schema_tau] if schema_tau else []

    # ----------------------------------------------------------------
    result = {
        "implementation": implementation,

        "compliance": {
            "ledger": ledger,
            "ledger_completeness": safe(ledger_completeness_score, ledger),
            "composition": safe(composition_summary, ledger),
        },

        "utility": {
            "marginal": safe(mean_marginal_l1, real_df, synth_df, schema),
            "tvd": safe(pairwise_tvd, real_df, synth_df, schema),
            "correlation": safe(
                mixed_association_similarity, real_df, synth_df, schema),
            "wasserstein": safe(mean_wasserstein_per_column, real_df, synth_df, schema),
            "tstr": (safe(tstr_classification, synth_df, test_real_df,
                          event_col, covariates, schema)
                     if test_real_df is not None and event_col else None),
            "coverage": safe(categorical_coverage, synth_df, schema),
            "unknown_token_rate": safe(unknown_token_rate, synth_df),
        },

        "survival": {
            "km_l1": (safe(km_l1_distance, real_df, synth_df,
                           duration_col, event_col)
                      if duration_col and event_col else None),
            "km_ci_overlap": (safe(km_ci_overlap, real_df, synth_df,
                                   duration_col, event_col)
                               if duration_col and event_col else None),
            "logrank_p": (safe(logrank_pvalue, real_df, synth_df,
                               duration_col, event_col)
                          if duration_col and event_col else None),
            "cox_spearman": (safe(cox_coefficient_spearman,
                                  real_df, synth_df,
                                  duration_col, event_col, covariates)
                             if duration_col and event_col else None),
            "tstr_cindex": (safe(tstr_cindex, synth_df, test_real_df,
                                 duration_col, event_col, covariates)
                            if (test_real_df is not None
                                and duration_col and event_col) else None),
            "censoring_rate_error": (safe(censoring_rate_error,
                                         real_df, synth_df, event_col)
                                     if event_col else None),
            "joint_survival_censoring": (safe(joint_survival_censoring,
                                              real_df, synth_df,
                                              duration_col, event_col)
                                         if duration_col and event_col else None),
            "rmst": (safe(rmst_error_multiple_taus, real_df, synth_df,
                          duration_col, event_col, taus)
                     if (taus and duration_col and event_col) else None),
        },

        "privacy": {
            "mia": (safe(mia_nearest_neighbour, train_df,
                         holdout_df, synth_df)
                    if (train_df is not None
                        and holdout_df is not None) else None),
            "nndr": (safe(nearest_neighbour_distance_ratio,
                          real_df, synth_df, numeric_cols)
                     if numeric_cols else None),
            "attribute_inference": (safe(attribute_inference_auc,
                                         real_df, synth_df, attr_inference_target, schema)
                                    if attr_inference_target else None),
        },

        "performance": performance or {},

        "constraints": safe(constraint_violation_summary, synth_df, schema),
    }

    return result
