"""
adapters/schema_normalization.py

Ensures all implementations see and return data in a schema-consistent way,
so comparisons reflect differences in the synthesizers rather than
differences in datatype handling.

Key functions:
  load_schema(...)           - load schema from dict or file
  parse_schema(...)          - extract column types, bounds, categories
  prepare_fit_df_for_synthcity(...) - minimal fit-time dtype correction for SynthCity
  prepare_fit_df_for_dpmm(...)      - minimal fit-time dtype correction for DPMM
  normalize_to_schema_output(...)   - shared post-sample normalization before evaluation

Change log (schema_generator v1.6 alignment):
  - prepare_fit_df_for_dpmm() now preserves missing values as an explicit
    "__NAN__" category for ordinal/categorical columns that appear in
    extensions.privbayes.missing_value_handling.columns_affected.
    Previously, NaN in these columns was silently imputed to the first
    allowed category (e.g. ph.ecog NaN → "0.0"), destroying the
    missingness signal that the schema's nan_as_extra_bin strategy is
    designed to model.  The "__NAN__" token is appended to the column's
    allowed list so TableBinner sees it as a valid declared level.
  - get_nan_bin_columns() is a new public helper that returns the set of
    columns requiring NaN-bin treatment, extracted from the schema
    extensions block.  Called by prepare_fit_df_for_dpmm() and exposed
    for use by other adapters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Token used for explicit NaN encoding in DPMM categorical columns.
# Must be consistent across fit and sample so TableBinner's encoder
# sees the same vocabulary at both stages.
_NAN_TOKEN = "__NAN__"


@dataclass
class SchemaInfo:
    column_types: dict[str, str]
    bounds: dict[str, tuple[float, float]]
    categories: dict[str, list[Any]]


def load_schema(schema: dict | str | None) -> dict:
    if schema is None:
        return {}
    if isinstance(schema, str):
        with open(schema) as f:
            return json.load(f)
    return dict(schema)


def parse_schema(schema: dict | str | None) -> SchemaInfo:
    s = load_schema(schema)

    column_types = {
        str(k): str(v)
        for k, v in s.get("column_types", {}).items()
    }

    bounds: dict[str, tuple[float, float]] = {}
    for col, raw in s.get("public_bounds", {}).items():
        if isinstance(raw, dict):
            lo, hi = raw.get("min"), raw.get("max")
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            lo, hi = raw[0], raw[1]
        else:
            continue

        if lo is None or hi is None:
            continue

        bounds[str(col)] = (float(lo), float(hi))

    categories = {
        str(col): list(vals or [])
        for col, vals in s.get("public_categories", {}).items()
    }

    return SchemaInfo(
        column_types=column_types,
        bounds=bounds,
        categories=categories,
    )


def get_nan_bin_columns(schema: dict | str | None) -> set[str]:
    """
    Return the set of columns that require explicit NaN-bin treatment.

    These are the columns listed in
    extensions.privbayes.missing_value_handling.columns_affected.
    The schema's nan_as_extra_bin strategy encodes missing values as a
    dedicated extra bin/level rather than imputing them, so DPMM's
    TableBinner must see NaN as a declared category, not as a fill value.

    Returns an empty set when the schema has no extensions block or when
    no columns are affected.
    """
    s = load_schema(schema)
    affected = (
        s.get("extensions", {})
         .get("privbayes", {})
         .get("missing_value_handling", {})
         .get("columns_affected", {})
    )
    return set(affected.keys()) if isinstance(affected, dict) else set()


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_binary(series: pd.Series) -> pd.Series:
    s = _to_numeric(series)
    return (s.fillna(0) >= 0.5).astype(np.int64)


def _repair_category_value(x: Any, allowed: list[Any]) -> Any:
    if not allowed:
        raise ValueError("allowed categories must be non-empty")

    allowed_set = set(allowed)
    if x in allowed_set:
        return x

    xs = str(x)
    for a in allowed:
        if str(a) == xs:
            return a

    try:
        xnum = pd.to_numeric(x, errors="raise")
        for a in allowed:
            try:
                if pd.to_numeric(a, errors="raise") == xnum:
                    return a
            except Exception:
                pass
    except Exception:
        pass

    return allowed[0]


def normalize_to_schema_output(
    df: pd.DataFrame,
    schema: dict | str | None,
    fit_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Shared post-sample normalization used for fair evaluation.
    This is evaluation-safe: it restores representation, not learning.
    """
    info = parse_schema(schema)
    out = df.copy()

    if fit_columns is not None:
        out = out.reindex(columns=fit_columns)

    for col in out.columns:
        stype = info.column_types.get(col)
        bounds = info.bounds.get(col)
        allowed = info.categories.get(col)

        if stype == "continuous":
            s = _to_numeric(out[col])
            if bounds is not None:
                lo, hi = bounds
                s = s.clip(lo, hi)
            out[col] = s.astype(np.float64)

        elif stype == "integer":
            s = _to_numeric(out[col])
            if bounds is not None:
                lo, hi = bounds
                s = s.clip(lo, hi)
            out[col] = s.round().fillna(0).astype(np.int64)

        elif stype == "binary":
            out[col] = _coerce_binary(out[col])

        elif stype in ("categorical", "ordinal"):
            if allowed:
                out[col] = out[col].map(lambda x: _repair_category_value(x, allowed)).astype(object)
            else:
                out[col] = out[col].astype(object)

    return out


def prepare_fit_df_for_synthcity(
    df: pd.DataFrame,
    schema: dict | str | None,
) -> pd.DataFrame:
    """
    Minimal schema-aware dtype coercion for SynthCity.
    No clipping, no filling to schema bounds, no structure hints.

    When schema declares a column as binary, we coerce to 0/1 (int64) at fit time.
    That ensures the backend sees exactly two unique values, so _encode() uses the
    categorical (LabelEncoder) path and _decode() returns discrete 0/1. Thus the
    backend's binary column handling is invoked correctly when used with schema.
    """
    info = parse_schema(schema)
    out = df.copy()

    for col in out.columns:
        stype = info.column_types.get(col)
        allowed = info.categories.get(col)

        if stype == "continuous":
            out[col] = _to_numeric(out[col]).astype(np.float64)

        elif stype == "integer":
            s = _to_numeric(out[col]).round()
            if s.isna().any():
                out[col] = s.astype(np.float64)
            else:
                out[col] = s.astype(np.int64)

        elif stype == "binary":
            s = _to_numeric(out[col])
            s = (s >= 0.5).where(~s.isna(), np.nan)
            if s.isna().any():
                out[col] = s.astype(np.float64)
            else:
                out[col] = s.astype(np.int64)

        elif stype == "categorical":
            out[col] = out[col].astype(object)

        elif stype == "ordinal":
            # If schema provides explicit categories, preserve ordinal-as-discrete.
            if allowed:
                out[col] = out[col].astype(object)
            else:
                s = _to_numeric(out[col]).round()
                if s.isna().any():
                    out[col] = s.astype(np.float64)
                else:
                    out[col] = s.astype(np.int64)

    return out


def prepare_fit_df_for_dpmm(
    df: pd.DataFrame,
    schema: dict | str | None,
) -> pd.DataFrame:
    """
    Minimal schema-aware dtype coercion for DPMM/TableBinner.
    Categorical / ordinal / binary are routed as object so TableBinner
    treats them categorically.

    NaN handling for columns in extensions.privbayes.missing_value_handling
    .columns_affected (nan_as_extra_bin strategy):
      - These columns have missing values that the schema models explicitly
        as a dedicated NaN bin/level rather than imputing them.
      - For ordinal/categorical columns in this set, NaN is replaced with
        the token "__NAN__" and the token is appended to the allowed list so
        TableBinner sees it as a declared category.
      - For numeric columns in this set, NaN is left as-is — TableBinner
        handles numeric NaN via its own missing-value strategy, and the NaN
        bin is accounted for in extensions.privbayes.discretization.per_column
        [col].n_bins_total on the CRNPrivBayes side.
      - Previously, ordinal/categorical NaN was silently filled with the
        first allowed value (e.g. ph.ecog NaN → "0.0"), which destroyed the
        missingness signal entirely.
    """
    s = load_schema(schema)
    info = parse_schema(schema)
    nan_bin_cols = get_nan_bin_columns(schema)

    out = df.copy()

    for col in out.columns:
        stype = info.column_types.get(col)
        allowed = info.categories.get(col)

        if stype in ("categorical", "ordinal", "binary"):
            out[col] = out[col].astype(object)

            # If schema categories are strings, align to strings so encoder
            # sees the declared domain type.
            if allowed and any(isinstance(a, str) for a in allowed):
                out[col] = out[col].astype(str)

            if col in nan_bin_cols and stype in ("categorical", "ordinal"):
                # Preserve NaN as explicit "__NAN__" level so TableBinner
                # models it as a genuine category rather than filling it.
                # Replace all NaN representations with the token.
                nan_representations = [np.nan, pd.NA, "nan", "NaN", float("nan")]
                out[col] = out[col].replace(nan_representations, _NAN_TOKEN)
                # The allowed list passed to TableBinner must include the token;
                # it is appended at the end (consistent with nan_bin_index being
                # the last bin in the schema).
                # Note: we do NOT modify `info.categories` in-place; the extended
                # list is only used for the fill/filter logic below.
                extended_allowed = list(allowed) + [_NAN_TOKEN] if allowed else [_NAN_TOKEN]
                # Replace any remaining out-of-vocabulary values with first allowed
                # (excluding NaN which is now the token).
                if extended_allowed:
                    first = extended_allowed[0]
                    out[col] = out[col].where(
                        out[col].isin(extended_allowed), first
                    )
            else:
                # Standard path: fill NaN with first allowed value.
                if allowed:
                    first = allowed[0]
                    out[col] = out[col].replace(
                        [np.nan, pd.NA, "nan", "NaN"], first
                    )
                    out[col] = out[col].where(out[col].isin(allowed), first)

        elif stype == "continuous":
            out[col] = _to_numeric(out[col]).astype(np.float64)

        elif stype == "integer":
            out[col] = _to_numeric(out[col]).round()

    return out
