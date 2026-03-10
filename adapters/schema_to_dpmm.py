"""
adapters/schema_to_dpmm.py

Translates benchmark schema into the domain format expected by TableBinner.

Change log (schema_generator v1.6 alignment):
  - schema_to_dpmm_domain() now reads n_bins per column from
    public_bounds[col]["n_bins"] (= min(n_unique, 100) — raw domain cardinality)
    instead of the removed "dpmm_n_bins" top-level field which never existed in
    any schema output and always silently fell back to 100.
  - For integer columns with few unique values (e.g. age: 42, meal.cal: 60)
    this gives TableBinner the correct natural resolution instead of 100 bins.
  - Note: public_bounds[col]["n_bins"] is the correct source for DPMM, NOT
    extensions.privbayes.discretization.per_column[col].n_bins (which is the
    Sturges-capped PrivBayes-specific value).  DPMM has its own binning logic
    via TableBinner; the raw cardinality is the right signal for it.
  - The n_bins_fallback (default 100) is retained for continuous columns that
    genuinely do not have a natural cardinality (n_bins = 100 in public_bounds).
"""

from __future__ import annotations

import json
from typing import Any

from adapters.schema_normalization import get_nan_bin_columns

_NAN_TOKEN = "__NAN__"


def _load_schema(schema: dict | str) -> dict:
    if isinstance(schema, str):
        with open(schema) as f:
            return json.load(f)
    return dict(schema)


def _normalize_bounds(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, dict):
        lo, hi = raw.get("min"), raw.get("max")
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        lo, hi = raw[0], raw[1]
    else:
        return None

    if lo is None or hi is None:
        return None

    return float(lo), float(hi)


def _col_n_bins(raw_entry: Any, fallback: int = 100) -> int:
    """
    Extract per-column bin count from a public_bounds entry.

    public_bounds[col] is a dict {"min": ..., "max": ..., "n_bins": ...}.
    "n_bins" = min(n_unique, 100) for integer columns; 100 for continuous.
    Falls back to `fallback` if the field is absent or the entry is in the
    legacy [lo, hi] list format (which has no n_bins key).
    """
    if isinstance(raw_entry, dict):
        v = raw_entry.get("n_bins")
        if v is not None:
            try:
                return max(2, int(v))
            except (TypeError, ValueError):
                pass
    return fallback


def schema_to_dpmm_domain(schema: dict | str) -> dict:
    """
    Translate benchmark schema into the domain format expected by TableBinner.

    continuous / integer:
        {"lower": lo, "upper": hi, "n_bins": N}
        N is taken from public_bounds[col]["n_bins"] (raw domain cardinality).

    categorical / ordinal / binary:
        {"categories": [...]}
    """
    s = _load_schema(schema)
    col_types = s.get("column_types", {})
    raw_bounds = s.get("public_bounds", {})
    raw_categories = s.get("public_categories", {})

    # Fallback bin count: used only when public_bounds entry has no "n_bins"
    # key (e.g. legacy list-format bounds).  100 is a safe upper bound.
    n_bins_fallback = 100

    domain: dict[str, dict] = {}

    for col, ctype in col_types.items():
        ctype = str(ctype)

        if ctype in ("continuous", "integer"):
            raw_entry = raw_bounds.get(col)
            bounds = _normalize_bounds(raw_entry)
            if bounds is None:
                raise ValueError(
                    f"Schema column '{col}' is {ctype} but has no valid public_bounds entry."
                )
            lo, hi = bounds
            # Per-column bin count from schema (raw domain cardinality).
            # For integer columns this is min(n_unique, 100); for continuous it
            # is 100 (no natural cardinality).  Both are correct inputs for
            # TableBinner — it uses this as the resolution hint, not as CPT sizing.
            col_n_bins = _col_n_bins(raw_entry, fallback=n_bins_fallback)
            domain[col] = {
                "lower": lo,
                "upper": hi,
                "n_bins": col_n_bins,
            }

        elif ctype in ("categorical", "ordinal", "binary"):
            cats = list(raw_categories.get(col, []) or [])
            if not cats:
                raise ValueError(
                    f"Schema column '{col}' is {ctype} but has no public_categories entry."
                )
            # Columns with nan_as_extra_bin get __NAN__ in prepare_fit_df_for_dpmm;
            # domain must include it so TableBinner's encoder accepts it.
            nan_bin_cols = get_nan_bin_columns(s)
            if col in nan_bin_cols and ctype in ("categorical", "ordinal"):
                cats = cats + [_NAN_TOKEN]
            domain[col] = {"categories": cats}

    return domain


def dpmm_domain_coverage_report(schema: dict | str) -> dict:
    """
    Report whether each schema column is fully covered by DPMM domain translation.
    """
    s = _load_schema(schema)
    col_types = s.get("column_types", {})
    raw_bounds = s.get("public_bounds", {})
    raw_categories = s.get("public_categories", {})

    covered: dict[str, dict] = {}

    for col, ctype in col_types.items():
        ctype = str(ctype)

        if ctype in ("continuous", "integer"):
            raw_entry = raw_bounds.get(col)
            bounds_ok = _normalize_bounds(raw_entry) is not None
            covered[col] = {
                "type": ctype,
                "has_bounds": bounds_ok,
                "n_bins": _col_n_bins(raw_entry) if bounds_ok else None,
                "has_categories": False,
                "covered": bounds_ok,
            }

        elif ctype in ("categorical", "ordinal", "binary"):
            cats = list(raw_categories.get(col, []) or [])
            covered[col] = {
                "type": ctype,
                "has_bounds": False,
                "has_categories": bool(cats),
                "covered": bool(cats),
            }

        else:
            covered[col] = {
                "type": ctype,
                "has_bounds": False,
                "has_categories": False,
                "covered": False,
            }

    fully_covered = all(v["covered"] for v in covered.values()) if covered else True

    return {
        "fully_covered": fully_covered,
        "columns": covered,
    }
