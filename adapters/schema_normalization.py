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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


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
    Categorical / ordinal / binary are routed as object so TableBinner treats them categorically.
    """
    info = parse_schema(schema)
    out = df.copy()

    for col in out.columns:
        stype = info.column_types.get(col)
        allowed = info.categories.get(col)

        if stype in ("categorical", "ordinal", "binary"):
            out[col] = out[col].astype(object)

            # If schema categories are strings, align to strings so encoder sees the declared domain type.
            if allowed and any(isinstance(a, str) for a in allowed):
                out[col] = out[col].astype(str)
            # Fill missing so TableBinner/OrdinalEncoder never sees "nan" or NaN as a category.
            if allowed:
                first = allowed[0]
                out[col] = out[col].replace([np.nan, pd.NA, "nan", "NaN"], first)
                out[col] = out[col].where(out[col].isin(allowed), first)

        elif stype == "continuous":
            out[col] = _to_numeric(out[col]).astype(np.float64)

        elif stype == "integer":
            out[col] = _to_numeric(out[col]).round()

    return out
