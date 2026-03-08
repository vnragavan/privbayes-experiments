"""
adapters/schema_to_dpmm.py

Translates benchmark schema into the domain format expected by TableBinner.
"""

from __future__ import annotations

import json
from typing import Any


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


def schema_to_dpmm_domain(schema: dict | str) -> dict:
    """
    Translate benchmark schema into the domain format expected by TableBinner.

    continuous / integer:
        {"lower": lo, "upper": hi, "n_bins": N}

    categorical / ordinal / binary:
        {"categories": [...]}
    """
    s = _load_schema(schema)
    col_types = s.get("column_types", {})
    raw_bounds = s.get("public_bounds", {})
    raw_categories = s.get("public_categories", {})
    n_bins_default = int(s.get("dpmm_n_bins", 100))

    domain: dict[str, dict] = {}

    for col, ctype in col_types.items():
        ctype = str(ctype)

        if ctype in ("continuous", "integer"):
            bounds = _normalize_bounds(raw_bounds.get(col))
            if bounds is None:
                raise ValueError(
                    f"Schema column '{col}' is {ctype} but has no valid public_bounds entry."
                )
            lo, hi = bounds
            domain[col] = {
                "lower": lo,
                "upper": hi,
                "n_bins": n_bins_default,
            }

        elif ctype in ("categorical", "ordinal", "binary"):
            cats = list(raw_categories.get(col, []) or [])
            if not cats:
                raise ValueError(
                    f"Schema column '{col}' is {ctype} but has no public_categories entry."
                )
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
            covered[col] = {
                "type": ctype,
                "has_bounds": _normalize_bounds(raw_bounds.get(col)) is not None,
                "has_categories": False,
                "covered": _normalize_bounds(raw_bounds.get(col)) is not None,
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
