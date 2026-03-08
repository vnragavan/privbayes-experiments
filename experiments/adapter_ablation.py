"""
experiments/adapter_ablation.py

Runs adapted vs raw DPMM and SynthCity (no adapter/wrapper) and reports
fit-time dtype alignment, pre-normalization representation, and post-normalized
benchmark metrics. Use to show why the adapter is needed and how metrics change
after minimal representation correction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from adapters.schema_normalization import normalize_to_schema_output
from implementations.dpmm_wrapper import DPMMWrapper
from implementations.synthcity_wrapper import SynthCityWrapper
from metrics.report import compute_metrics


@dataclass
class AblationResult:
    implementation: str
    condition: str
    fit_dtype_summary: dict
    pre_normalization_summary: dict
    metrics: dict


def _load_schema(schema: dict | str) -> dict:
    if isinstance(schema, str):
        with open(schema) as f:
            return json.load(f)
    return dict(schema)


def _fit_dtype_summary(df: pd.DataFrame, schema: dict) -> dict:
    col_types = schema.get("column_types", {})
    rows = []
    mismatch_count = 0

    for col in df.columns:
        schema_type = col_types.get(col)
        dtype_str = str(df[col].dtype)

        mismatch = False
        if schema_type in ("categorical", "ordinal") and dtype_str != "object":
            mismatch = True
        elif schema_type == "binary" and dtype_str not in ("int64", "Int64", "float64", "object"):
            mismatch = True
        elif schema_type == "continuous" and dtype_str not in ("float64", "float32"):
            mismatch = True
        elif schema_type == "integer" and dtype_str not in ("int64", "Int64", "float64"):
            mismatch = True

        mismatch_count += int(mismatch)
        rows.append({
            "column": col,
            "schema_type": schema_type,
            "dtype_seen": dtype_str,
            "mismatch": mismatch,
        })

    return {
        "n_columns": len(rows),
        "n_mismatches": mismatch_count,
        "columns": rows,
    }


def _sample_representation_summary(df: pd.DataFrame, schema: dict) -> dict:
    col_types = schema.get("column_types", {})
    public_bounds = schema.get("public_bounds", {})
    public_categories = schema.get("public_categories", {})

    binary_invalid = 0
    binary_total = 0
    categorical_invalid = 0
    categorical_total = 0
    integer_float_cols = 0
    integer_cols = 0
    out_of_bounds = 0
    bounded_total = 0

    for col in df.columns:
        stype = col_types.get(col)

        if stype == "binary":
            binary_total += len(df[col])
            vals = pd.to_numeric(df[col], errors="coerce")
            binary_invalid += int((~vals.isin([0, 1])).fillna(True).sum())

        if stype in ("categorical", "ordinal") and col in public_categories:
            allowed = set(public_categories[col])
            categorical_total += len(df[col])
            categorical_invalid += int((~df[col].isin(allowed)).sum())

        if stype == "integer":
            integer_cols += 1
            if not pd.api.types.is_integer_dtype(df[col]):
                integer_float_cols += 1

        if col in public_bounds:
            raw = public_bounds[col]
            if isinstance(raw, dict):
                lo, hi = float(raw["min"]), float(raw["max"])
            else:
                lo, hi = float(raw[0]), float(raw[1])

            vals = pd.to_numeric(df[col], errors="coerce")
            mask = vals.notna()
            bounded_total += int(mask.sum())
            overflow = (vals < lo) | (vals > hi)
            out_of_bounds += int((overflow & mask).sum())

    return {
        "binary_invalid_rate": float(binary_invalid / binary_total) if binary_total else 0.0,
        "categorical_invalid_rate": float(categorical_invalid / categorical_total) if categorical_total else 0.0,
        "integer_float_column_rate": float(integer_float_cols / integer_cols) if integer_cols else 0.0,
        "out_of_bounds_rate": float(out_of_bounds / bounded_total) if bounded_total else 0.0,
        "raw_dtypes": {c: str(df[c].dtype) for c in df.columns},
    }


def run_synthcity_raw(
    train_df: pd.DataFrame,
    schema: dict,
    epsilon: float,
    n_bins: int,
    n_synth: int,
):
    from synthcity_standalone.privbayes import PrivBayes

    fit_df = train_df.copy()
    fit_summary = _fit_dtype_summary(fit_df, schema)

    model = PrivBayes(
        epsilon=max(float(epsilon), 1e-6),
        n_bins=max(int(n_bins), 2),
        target_usefulness=5,
    )
    model.fit(fit_df)

    raw_sample = model.sample(n_synth)
    raw_summary = _sample_representation_summary(raw_sample, schema)

    # Important: normalize before scoring so evaluation is controlled
    scored_sample = normalize_to_schema_output(raw_sample, schema, fit_columns=list(train_df.columns))

    return fit_summary, raw_summary, scored_sample, {
        "epsilon": epsilon,
        "schema_injection": "none",
    }


def run_synthcity_adapted(
    train_df: pd.DataFrame,
    schema: dict,
    epsilon: float,
    n_bins: int,
    n_synth: int,
):
    wrapper = SynthCityWrapper(epsilon=epsilon, n_bins=n_bins)
    wrapper.fit(train_df, schema=schema)

    # Fit summary after the minimal adapter path
    from adapters.schema_normalization import prepare_fit_df_for_synthcity
    fit_df = prepare_fit_df_for_synthcity(train_df, schema)
    fit_summary = _fit_dtype_summary(fit_df, schema)

    scored_sample = wrapper.sample(n_synth)

    # For adapted path, pre-normalization is the same object as final sample,
    # because wrapper already normalizes output representation.
    raw_summary = _sample_representation_summary(scored_sample, schema)

    return fit_summary, raw_summary, scored_sample, wrapper.privacy_report()


def run_dpmm_raw(
    train_df: pd.DataFrame,
    schema: dict,
    epsilon: float,
    delta: float,
    degree: int,
    seed: int,
    n_synth: int,
):
    from numpy.random import RandomState
    from dpmm.models.priv_bayes import PrivBayesGM
    from dpmm.processing.table_binner import TableBinner

    fit_df = train_df.copy()
    fit_summary = _fit_dtype_summary(fit_df, schema)

    binner = TableBinner(
        binner_type="uniform",
        binner_settings={"n_bins": "auto"},
        domain=None,   # raw mode, no schema adapter
        random_state=RandomState(seed),
    )
    df_encoded = binner.fit_transform(fit_df, public=True)
    int_domain = binner.bin_domain

    model = PrivBayesGM(
        epsilon=epsilon,
        delta=delta,
        degree=degree,
        domain=int_domain,
        random_state=RandomState(seed),
        n_jobs=1,
    )
    model.fit(df_encoded)

    raw_sample = binner.inverse_transform(model.generate(n_records=n_synth))
    raw_summary = _sample_representation_summary(raw_sample, schema)

    scored_sample = normalize_to_schema_output(raw_sample, schema, fit_columns=list(train_df.columns))

    return fit_summary, raw_summary, scored_sample, {
        "epsilon": epsilon,
        "delta": delta,
        "schema_injection": "none",
    }


def run_dpmm_adapted(
    train_df: pd.DataFrame,
    schema: dict,
    epsilon: float,
    delta: float,
    degree: int,
    seed: int,
    n_synth: int,
):
    wrapper = DPMMWrapper(
        epsilon=epsilon,
        delta=delta,
        degree=degree,
        seed=seed,
    )
    wrapper.fit(train_df, schema=schema)

    from adapters.schema_normalization import prepare_fit_df_for_dpmm
    fit_df = prepare_fit_df_for_dpmm(train_df[wrapper._fit_columns].copy(), schema)
    fit_summary = _fit_dtype_summary(fit_df, schema)

    scored_sample = wrapper.sample(n_synth)
    raw_summary = _sample_representation_summary(scored_sample, schema)

    return fit_summary, raw_summary, scored_sample, wrapper.privacy_report()


def summarize_metrics(metrics: dict) -> dict:
    """
    Small focused summary for the adapter ablation.
    Keys aligned with metrics returned by compute_metrics().
    """
    def _val(d, key, default=np.nan):
        if d is None or not isinstance(d, dict):
            return default
        return d.get(key, default)

    marginal = metrics.get("utility", {}) or {}
    marginal = marginal.get("marginal") if isinstance(marginal, dict) else {}
    corr = (metrics.get("utility") or {}).get("correlation")
    tstr = (metrics.get("utility") or {}).get("tstr")
    mia = (metrics.get("privacy") or {}).get("mia")
    nndr = (metrics.get("privacy") or {}).get("nndr")
    attr_inf = (metrics.get("privacy") or {}).get("attribute_inference")
    km_l1 = (metrics.get("survival") or {}).get("km_l1")
    cox = (metrics.get("survival") or {}).get("cox_spearman")
    constraints = metrics.get("constraints")
    compliance = metrics.get("compliance") or {}
    ledger_compl = compliance.get("ledger_completeness")
    composition = compliance.get("composition")

    return {
        "utility.marginal.mean_l1": _val(marginal, "mean_overall"),
        "utility.correlation.value": _val(corr, "value") if isinstance(corr, dict) else (corr if isinstance(corr, (int, float)) else np.nan),
        "utility.tstr.roc_auc": _val(tstr, "roc_auc"),
        "privacy.mia.auc": _val(mia, "auc"),
        "privacy.nndr.value": _val(nndr, "value"),
        "privacy.attribute_inference.auc": _val(attr_inf, "auc"),
        "survival.km_l1.value": _val(km_l1, "value") if isinstance(km_l1, dict) else (float(km_l1) if isinstance(km_l1, (int, float)) else np.nan),
        "survival.cox_spearman.value": _val(cox, "value") if isinstance(cox, dict) else (float(cox) if isinstance(cox, (int, float)) else np.nan),
        "constraints.violation_rate": _val(constraints, "overall_violation_rate"),
        "compliance.ledger_completeness.value": _val(ledger_compl, "value") if isinstance(ledger_compl, dict) else (ledger_compl if isinstance(ledger_compl, (int, float)) else np.nan),
        "compliance.composition.cgr": _val(composition, "composition_gap_ratio"),
    }


def run_adapter_ablation(
    train_df: pd.DataFrame,
    test_real_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    schema: dict | str,
    epsilon: float = 1.0,
    n_synth: int | None = None,
    seed: int = 0,
):
    schema = _load_schema(schema)
    if n_synth is None:
        n_synth = len(train_df)

    results = []

    runners = [
        ("synthcity", "raw", lambda: run_synthcity_raw(train_df, schema, epsilon, 100, n_synth)),
        ("synthcity", "adapted", lambda: run_synthcity_adapted(train_df, schema, epsilon, 100, n_synth)),
        ("dpmm", "raw", lambda: run_dpmm_raw(train_df, schema, epsilon, 1e-5, 2, seed, n_synth)),
        ("dpmm", "adapted", lambda: run_dpmm_adapted(train_df, schema, epsilon, 1e-5, 2, seed, n_synth)),
    ]

    for impl_name, condition, runner in runners:
        fit_summary, pre_norm_summary, synth_df, privacy_report = runner()

        metrics = compute_metrics(
            implementation=f"{impl_name}_{condition}",
            real_df=train_df,
            synth_df=synth_df,
            schema=schema,
            privacy_report=privacy_report,
            test_real_df=test_real_df,
            train_df=train_df,
            holdout_df=holdout_df,
            performance={},
        )

        results.append(AblationResult(
            implementation=impl_name,
            condition=condition,
            fit_dtype_summary=fit_summary,
            pre_normalization_summary=pre_norm_summary,
            metrics=summarize_metrics(metrics),
        ))

    return results
