"""
experiments/adapter_ablation_metrics.py

Self-contained ablation metrics: raw vs adapted (and optional wrong_schema)
for SynthCity and DPMM. Reports fit-time dtype mismatches, pre-normalization
output diagnostics, and post-normalization benchmark metrics.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from adapters.schema_normalization import (
    normalize_to_schema_output,
    prepare_fit_df_for_dpmm,
    prepare_fit_df_for_synthcity,
)
from implementations.dpmm_wrapper import DPMMWrapper
from implementations.synthcity_wrapper import SynthCityWrapper
from metrics.report import compute_metrics


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class FitDTypeMetrics:
    n_columns: int
    n_dtype_mismatches: int
    mismatch_rate: float
    binary_mismatch_count: int
    categorical_mismatch_count: int
    ordinal_mismatch_count: int
    integer_mismatch_count: int
    continuous_mismatch_count: int


@dataclass
class OutputStructureMetrics:
    binary_invalid_rate: float
    categorical_invalid_rate: float
    integer_float_column_rate: float
    out_of_bounds_rate: float


@dataclass
class BenchmarkMetrics:
    marginal: float
    correlation: float
    tstr_auc: float
    mia_auc: float
    attribute_inference_auc: float
    nndr: float
    km_l1: float
    cox_spearman: float
    constraint_violation_rate: float
    composition_gap_ratio: float


@dataclass
class AblationRow:
    implementation: str
    condition: str
    epsilon: float
    seed: int
    fit_metrics: FitDTypeMetrics
    output_metrics: OutputStructureMetrics
    benchmark_metrics: BenchmarkMetrics


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_schema(schema: dict | str) -> dict:
    if isinstance(schema, str):
        with open(schema) as f:
            return json.load(f)
    return copy.deepcopy(schema)


def _get_schema_type(schema: dict, col: str) -> str | None:
    return schema.get("column_types", {}).get(col)


def _is_dtype_mismatch(schema_type: str | None, dtype_str: str) -> bool:
    if schema_type is None:
        return False

    if schema_type == "binary":
        return dtype_str not in ("int64", "Int64", "float64", "object", "bool")

    if schema_type == "categorical":
        return dtype_str != "object"

    if schema_type == "ordinal":
        return dtype_str not in ("object", "int64", "Int64", "float64")

    if schema_type == "integer":
        return dtype_str not in ("int64", "Int64", "float64")

    if schema_type == "continuous":
        return dtype_str not in ("float64", "float32", "int64", "Int64")

    return False


def compute_fit_dtype_metrics(df_seen_by_model: pd.DataFrame, schema: dict) -> FitDTypeMetrics:
    n_columns = len(df_seen_by_model.columns)
    n_dtype_mismatches = 0

    counts = {
        "binary": 0,
        "categorical": 0,
        "ordinal": 0,
        "integer": 0,
        "continuous": 0,
    }

    for col in df_seen_by_model.columns:
        schema_type = _get_schema_type(schema, col)
        dtype_str = str(df_seen_by_model[col].dtype)
        mismatch = _is_dtype_mismatch(schema_type, dtype_str)

        if mismatch:
            n_dtype_mismatches += 1
            if schema_type in counts:
                counts[schema_type] += 1

    mismatch_rate = (n_dtype_mismatches / n_columns) if n_columns else 0.0

    return FitDTypeMetrics(
        n_columns=n_columns,
        n_dtype_mismatches=n_dtype_mismatches,
        mismatch_rate=float(mismatch_rate),
        binary_mismatch_count=counts["binary"],
        categorical_mismatch_count=counts["categorical"],
        ordinal_mismatch_count=counts["ordinal"],
        integer_mismatch_count=counts["integer"],
        continuous_mismatch_count=counts["continuous"],
    )


def _parse_bounds(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, dict):
        lo, hi = raw.get("min"), raw.get("max")
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        lo, hi = raw[0], raw[1]
    else:
        return None

    if lo is None or hi is None:
        return None
    return float(lo), float(hi)


def compute_output_structure_metrics(raw_sample_df: pd.DataFrame, schema: dict) -> OutputStructureMetrics:
    col_types = schema.get("column_types", {})
    categories = schema.get("public_categories", {})
    bounds = schema.get("public_bounds", {})

    binary_invalid = 0
    binary_total = 0

    categorical_invalid = 0
    categorical_total = 0

    integer_cols = 0
    integer_float_cols = 0

    bounded_invalid = 0
    bounded_total = 0

    for col in raw_sample_df.columns:
        stype = col_types.get(col)

        if stype == "binary":
            vals = pd.to_numeric(raw_sample_df[col], errors="coerce")
            binary_total += len(vals)
            binary_invalid += int((~vals.isin([0, 1])).fillna(True).sum())

        elif stype in ("categorical", "ordinal"):
            allowed = categories.get(col, [])
            if allowed:
                categorical_total += len(raw_sample_df[col])
                categorical_invalid += int((~raw_sample_df[col].isin(allowed)).sum())

        elif stype == "integer":
            integer_cols += 1
            if not pd.api.types.is_integer_dtype(raw_sample_df[col]):
                integer_float_cols += 1

        if col in bounds:
            parsed = _parse_bounds(bounds[col])
            if parsed is not None:
                lo, hi = parsed
                vals = pd.to_numeric(raw_sample_df[col], errors="coerce")
                mask = vals.notna()
                bounded_total += int(mask.sum())
                bounded_invalid += int((((vals < lo) | (vals > hi)) & mask).sum())

    return OutputStructureMetrics(
        binary_invalid_rate=float(binary_invalid / binary_total) if binary_total else 0.0,
        categorical_invalid_rate=float(categorical_invalid / categorical_total) if categorical_total else 0.0,
        integer_float_column_rate=float(integer_float_cols / integer_cols) if integer_cols else 0.0,
        out_of_bounds_rate=float(bounded_invalid / bounded_total) if bounded_total else 0.0,
    )


def _safe_get(dct: dict | None, *path: str, default: float = np.nan) -> float:
    """Get nested key from dict; support scalar at any step (e.g. survival.km_l1 is float)."""
    if dct is None:
        return default
    cur = dct
    for i, key in enumerate(path):
        if not isinstance(cur, dict):
            # e.g. path=("survival","km_l1","value") and km_l1 is float -> scalar is the value
            return float(cur) if isinstance(cur, (int, float)) else default
        if key not in cur:
            return default
        cur = cur[key]
    if isinstance(cur, (int, float)):
        return float(cur)
    if isinstance(cur, dict):
        return cur.get("value", cur.get("mean_overall", cur.get("roc_auc", cur.get("auc", default))))
    return default


def extract_benchmark_metrics(metrics: dict) -> BenchmarkMetrics:
    """Extract benchmark scalars from compute_metrics() result. Paths match metrics.report."""
    util = metrics.get("utility") or {}
    priv = metrics.get("privacy") or {}
    surv = metrics.get("survival") or {}
    constraints = metrics.get("constraints")
    comp = metrics.get("compliance") or {}
    composition = comp.get("composition") if isinstance(comp.get("composition"), dict) else {}

    return BenchmarkMetrics(
        marginal=_safe_get(util, "marginal", "mean_overall"),
        correlation=_safe_get(util, "correlation", "numeric_spearman"),
        tstr_auc=_safe_get(util, "tstr", "roc_auc"),
        mia_auc=_safe_get(priv, "mia", "auc"),
        attribute_inference_auc=_safe_get(priv, "attribute_inference", "auc"),
        nndr=_safe_get(priv, "nndr", "value"),
        km_l1=_safe_get(surv, "km_l1", "value"),
        cox_spearman=_safe_get(surv, "cox_spearman", "value"),
        constraint_violation_rate=_safe_get(constraints, "overall_violation_rate") if isinstance(constraints, dict) else np.nan,
        composition_gap_ratio=(float(v) if (v := composition.get("composition_gap_ratio")) is not None else np.nan) if composition else np.nan,
    )


def _perturb_schema_for_sanity_check(schema: dict) -> dict:
    """
    Deliberately corrupt a few schema types for the wrong_schema condition.
    Keep it minimal and deterministic.
    """
    s = copy.deepcopy(schema)
    col_types = s.get("column_types", {})

    changed = 0
    for col, t in list(col_types.items()):
        if t == "binary":
            col_types[col] = "continuous"
            changed += 1
        elif t == "categorical":
            col_types[col] = "integer"
            changed += 1
        elif t == "integer":
            col_types[col] = "continuous"
            changed += 1

        if changed >= 3:
            break

    s["column_types"] = col_types
    return s


# ---------------------------------------------------------------------
# Raw runners
# ---------------------------------------------------------------------

def _run_synthcity_raw(
    train_df: pd.DataFrame,
    schema: dict,
    epsilon: float,
    n_synth: int,
    n_bins: int = 100,
):
    from synthcity_standalone.privbayes import PrivBayes

    fit_df = train_df.copy()
    fit_metrics = compute_fit_dtype_metrics(fit_df, schema)

    model = PrivBayes(
        epsilon=max(float(epsilon), 1e-6),
        n_bins=max(int(n_bins), 2),
        target_usefulness=5,
    )
    model.fit(fit_df)

    raw_sample = model.sample(n_synth)
    output_metrics = compute_output_structure_metrics(raw_sample, schema)

    scored_sample = normalize_to_schema_output(
        raw_sample,
        schema,
        fit_columns=list(train_df.columns),
    )

    privacy_report = {
        "epsilon": float(epsilon),
        "schema_injection": "none",
        "bounds_source": "data_derived_inside_backend",
        "categories_source": "data_derived_inside_backend",
    }

    return fit_metrics, output_metrics, scored_sample, privacy_report


def _run_dpmm_raw(
    train_df: pd.DataFrame,
    schema: dict,
    epsilon: float,
    n_synth: int,
    delta: float = 1e-5,
    degree: int = 2,
    seed: int = 0,
):
    from numpy.random import RandomState
    from dpmm.models.priv_bayes import PrivBayesGM
    from dpmm.processing.table_binner import TableBinner

    fit_df = train_df.copy()
    fit_metrics = compute_fit_dtype_metrics(fit_df, schema)

    binner = TableBinner(
        binner_type="uniform",
        binner_settings={"n_bins": "auto"},
        domain=None,
        random_state=RandomState(seed),
    )
    encoded = binner.fit_transform(fit_df, public=True)

    model = PrivBayesGM(
        epsilon=float(epsilon),
        delta=float(delta),
        degree=int(degree),
        domain=binner.bin_domain,
        random_state=RandomState(seed),
        n_jobs=1,
    )
    model.fit(encoded)

    raw_sample = binner.inverse_transform(model.generate(n_records=n_synth))
    output_metrics = compute_output_structure_metrics(raw_sample, schema)

    scored_sample = normalize_to_schema_output(
        raw_sample,
        schema,
        fit_columns=list(train_df.columns),
    )

    privacy_report = {
        "epsilon": float(epsilon),
        "delta": float(delta),
        "schema_injection": "none",
        "bounds_source": "data_derived_inside_backend",
        "categories_source": "data_derived_inside_backend",
    }

    return fit_metrics, output_metrics, scored_sample, privacy_report


# ---------------------------------------------------------------------
# Adapted runners
# ---------------------------------------------------------------------

def _run_synthcity_adapted(
    train_df: pd.DataFrame,
    schema: dict,
    epsilon: float,
    n_synth: int,
    n_bins: int = 100,
):
    wrapper = SynthCityWrapper(
        epsilon=float(epsilon),
        n_bins=int(n_bins),
    )
    wrapper.fit(train_df, schema=schema)

    fit_df = prepare_fit_df_for_synthcity(train_df, schema)
    fit_metrics = compute_fit_dtype_metrics(fit_df, schema)

    scored_sample = wrapper.sample(n_synth)
    output_metrics = compute_output_structure_metrics(scored_sample, schema)

    return fit_metrics, output_metrics, scored_sample, wrapper.privacy_report()


def _run_dpmm_adapted(
    train_df: pd.DataFrame,
    schema: dict,
    epsilon: float,
    n_synth: int,
    delta: float = 1e-5,
    degree: int = 2,
    seed: int = 0,
):
    wrapper = DPMMWrapper(
        epsilon=float(epsilon),
        delta=float(delta),
        degree=int(degree),
        seed=int(seed),
    )
    wrapper.fit(train_df, schema=schema)

    fit_cols = wrapper._fit_columns if wrapper._fit_columns is not None else list(train_df.columns)
    fit_df = prepare_fit_df_for_dpmm(train_df[fit_cols].copy(), schema)
    fit_metrics = compute_fit_dtype_metrics(fit_df, schema)

    scored_sample = wrapper.sample(n_synth)
    output_metrics = compute_output_structure_metrics(scored_sample, schema)

    return fit_metrics, output_metrics, scored_sample, wrapper.privacy_report()


# ---------------------------------------------------------------------
# Main ablation entrypoint
# ---------------------------------------------------------------------

def run_adapter_ablation_metrics(
    train_df: pd.DataFrame,
    test_real_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    schema: dict | str,
    epsilon: float = 1.0,
    seed: int = 0,
    n_synth: int | None = None,
    include_wrong_schema: bool = False,
) -> pd.DataFrame:
    """
    Returns one flat dataframe with all ablation metrics.

    Conditions:
        - raw
        - adapted
        - wrong_schema (optional)

    Implementations:
        - synthcity
        - dpmm
    """
    schema = _load_schema(schema)

    if n_synth is None:
        n_synth = len(train_df)

    rows: list[AblationRow] = []

    experiment_plan = [
        ("synthcity", "raw", schema),
        ("synthcity", "adapted", schema),
        ("dpmm", "raw", schema),
        ("dpmm", "adapted", schema),
    ]

    if include_wrong_schema:
        wrong_schema = _perturb_schema_for_sanity_check(schema)
        experiment_plan.extend([
            ("synthcity", "wrong_schema", wrong_schema),
            ("dpmm", "wrong_schema", wrong_schema),
        ])

    for implementation, condition, condition_schema in experiment_plan:
        if implementation == "synthcity" and condition == "raw":
            fit_metrics, output_metrics, synth_df, privacy_report = _run_synthcity_raw(
                train_df=train_df,
                schema=schema,
                epsilon=epsilon,
                n_synth=n_synth,
            )

        elif implementation == "synthcity" and condition in ("adapted", "wrong_schema"):
            fit_metrics, output_metrics, synth_df, privacy_report = _run_synthcity_adapted(
                train_df=train_df,
                schema=condition_schema,
                epsilon=epsilon,
                n_synth=n_synth,
            )

        elif implementation == "dpmm" and condition == "raw":
            fit_metrics, output_metrics, synth_df, privacy_report = _run_dpmm_raw(
                train_df=train_df,
                schema=schema,
                epsilon=epsilon,
                n_synth=n_synth,
                seed=seed,
            )

        elif implementation == "dpmm" and condition in ("adapted", "wrong_schema"):
            fit_metrics, output_metrics, synth_df, privacy_report = _run_dpmm_adapted(
                train_df=train_df,
                schema=condition_schema,
                epsilon=epsilon,
                n_synth=n_synth,
                seed=seed,
            )

        else:
            raise ValueError(f"Unsupported plan entry: {(implementation, condition)}")

        metrics = compute_metrics(
            implementation=f"{implementation}_{condition}",
            real_df=train_df,
            synth_df=synth_df,
            schema=schema,
            privacy_report=privacy_report,
            test_real_df=test_real_df,
            train_df=train_df,
            holdout_df=holdout_df,
            performance={},
        )

        bench = extract_benchmark_metrics(metrics)

        rows.append(
            AblationRow(
                implementation=implementation,
                condition=condition,
                epsilon=float(epsilon),
                seed=int(seed),
                fit_metrics=fit_metrics,
                output_metrics=output_metrics,
                benchmark_metrics=bench,
            )
        )

    flat_rows = []
    for row in rows:
        flat_rows.append({
            "implementation": row.implementation,
            "condition": row.condition,
            "epsilon": row.epsilon,
            "seed": row.seed,

            "n_columns": row.fit_metrics.n_columns,
            "n_dtype_mismatches": row.fit_metrics.n_dtype_mismatches,
            "mismatch_rate": row.fit_metrics.mismatch_rate,
            "binary_mismatch_count": row.fit_metrics.binary_mismatch_count,
            "categorical_mismatch_count": row.fit_metrics.categorical_mismatch_count,
            "ordinal_mismatch_count": row.fit_metrics.ordinal_mismatch_count,
            "integer_mismatch_count": row.fit_metrics.integer_mismatch_count,
            "continuous_mismatch_count": row.fit_metrics.continuous_mismatch_count,

            "binary_invalid_rate": row.output_metrics.binary_invalid_rate,
            "categorical_invalid_rate": row.output_metrics.categorical_invalid_rate,
            "integer_float_column_rate": row.output_metrics.integer_float_column_rate,
            "out_of_bounds_rate": row.output_metrics.out_of_bounds_rate,

            "marginal": row.benchmark_metrics.marginal,
            "correlation": row.benchmark_metrics.correlation,
            "tstr_auc": row.benchmark_metrics.tstr_auc,
            "mia_auc": row.benchmark_metrics.mia_auc,
            "attribute_inference_auc": row.benchmark_metrics.attribute_inference_auc,
            "nndr": row.benchmark_metrics.nndr,
            "km_l1": row.benchmark_metrics.km_l1,
            "cox_spearman": row.benchmark_metrics.cox_spearman,
            "constraint_violation_rate": row.benchmark_metrics.constraint_violation_rate,
            "composition_gap_ratio": row.benchmark_metrics.composition_gap_ratio,
        })

    return pd.DataFrame(flat_rows)


# ---------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------

def build_table_schema_interpretation(df_metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "implementation",
        "condition",
        "n_columns",
        "n_dtype_mismatches",
        "mismatch_rate",
        "binary_mismatch_count",
        "categorical_mismatch_count",
        "ordinal_mismatch_count",
        "integer_mismatch_count",
        "continuous_mismatch_count",
    ]
    return df_metrics[cols].sort_values(["implementation", "condition"]).reset_index(drop=True)


def build_table_output_structure(df_metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "implementation",
        "condition",
        "binary_invalid_rate",
        "categorical_invalid_rate",
        "integer_float_column_rate",
        "out_of_bounds_rate",
    ]
    return df_metrics[cols].sort_values(["implementation", "condition"]).reset_index(drop=True)


def build_table_benchmark(df_metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "implementation",
        "condition",
        "marginal",
        "correlation",
        "tstr_auc",
        "mia_auc",
        "attribute_inference_auc",
        "nndr",
        "km_l1",
        "cox_spearman",
        "constraint_violation_rate",
        "composition_gap_ratio",
    ]
    return df_metrics[cols].sort_values(["implementation", "condition"]).reset_index(drop=True)


# Benchmark numeric columns used for mean/SE/CI aggregation
BENCHMARK_NUMERIC_COLS = [
    "marginal",
    "correlation",
    "tstr_auc",
    "mia_auc",
    "attribute_inference_auc",
    "nndr",
    "km_l1",
    "cox_spearman",
    "constraint_violation_rate",
    "composition_gap_ratio",
]


def build_benchmark_summary_with_ci(
    df_metrics: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Aggregate multiple runs (by implementation, condition) into mean, std, se, ci_lo, ci_hi.
    df_metrics must contain a 'seed' column (multiple runs per (impl, condition)).
    """
    cols = numeric_cols or BENCHMARK_NUMERIC_COLS
    group_cols = ["implementation", "condition"]
    available = [c for c in cols if c in df_metrics.columns]
    if not available:
        return df_metrics[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)

    out_rows = []
    for key, grp in df_metrics.groupby(group_cols, sort=False):
        row = {"implementation": key[0], "condition": key[1], "n_runs": len(grp)}
        n = len(grp)
        t_crit = float(scipy_stats.t.ppf(0.5 + confidence / 2, n - 1)) if n > 1 else 0.0
        for c in available:
            vals = grp[c].dropna()
            if len(vals) == 0:
                row[f"{c}_mean"] = np.nan
                row[f"{c}_std"] = np.nan
                row[f"{c}_se"] = np.nan
                row[f"{c}_ci_lo"] = np.nan
                row[f"{c}_ci_hi"] = np.nan
            else:
                mean = float(vals.mean())
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                se = std / (len(vals) ** 0.5) if len(vals) > 0 else np.nan
                half = t_crit * se if n > 1 else 0.0
                row[f"{c}_mean"] = mean
                row[f"{c}_std"] = std
                row[f"{c}_se"] = se
                row[f"{c}_ci_lo"] = mean - half
                row[f"{c}_ci_hi"] = mean + half
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(group_cols).reset_index(drop=True)
