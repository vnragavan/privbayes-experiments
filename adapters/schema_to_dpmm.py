"""
adapters/schema_to_dpmm.py

Converts schema-generator JSON to the dpmm domain dict format.
dpmm has no native schema API so this adapter is always needed.
"""

from __future__ import annotations
from typing import Dict, Any


def schema_to_dpmm_domain(schema: dict) -> dict:
    """
    Converts schema-generator JSON to dpmm domain dict.

    dpmm numeric format:
        domain[col] = {"lower": float, "upper": float}
        domain[col] = {"lower": float, "upper": float, "n_bins": int}

    dpmm categorical format:
        domain[col] = {"categories": [val1, val2, ...]}

    When a column has no entry in public_bounds or public_categories,
    a WARNING is printed — dpmm will fall back to data-derived metadata
    (compliance gap).
    """
    col_types = schema.get("column_types", {})
    raw_bounds = schema.get("public_bounds", {})
    raw_cats = schema.get("public_categories", {})
    domain = {}

    for col, ctype in col_types.items():
        if ctype in ("continuous", "integer"):
            if col in raw_bounds:
                bv = raw_bounds[col]
                if isinstance(bv, dict):
                    lo, hi = float(bv["min"]), float(bv["max"])
                else:
                    lo, hi = float(bv[0]), float(bv[1])
                entry = {"lower": lo, "upper": hi}
                # Fix 3: read n_bins directly (generator writes {"min","max","n_bins"})
                # Previously checked for "bins" (array of edges) which was never written
                if isinstance(bv, dict) and "n_bins" in bv and bv["n_bins"]:
                    entry["n_bins"] = int(bv["n_bins"])
                domain[col] = entry
            else:
                print(
                    f"WARNING [dpmm]: no bounds for col '{col}' "
                    f"(type: {ctype}) — dpmm will infer from data "
                    f"(compliance gap)"
                )
                domain[col] = {}

        elif ctype in ("categorical", "ordinal", "binary"):
            if col in raw_cats:
                domain[col] = {"categories": list(raw_cats[col])}
            else:
                print(
                    f"WARNING [dpmm]: no categories for col '{col}' "
                    f"(type: {ctype}) — dpmm will call series.unique() "
                    f"(compliance gap)"
                )
                domain[col] = {}

    return domain


def dpmm_domain_coverage_report(schema: dict) -> dict:
    """
    Returns a summary of how much of the schema is covered
    by the dpmm domain dict.
    """
    col_types = schema.get("column_types", {})
    raw_bounds = schema.get("public_bounds", {})
    raw_cats = schema.get("public_categories", {})

    numeric_cols = [c for c, t in col_types.items()
                    if t in ("continuous", "integer")]
    cat_cols = [c for c, t in col_types.items()
                if t in ("categorical", "ordinal", "binary")]

    missing_bounds = [c for c in numeric_cols if c not in raw_bounds]
    missing_cats = [c for c in cat_cols if c not in raw_cats]

    return {
        "n_numeric_cols": len(numeric_cols),
        "n_numeric_with_bounds": len(numeric_cols) - len(missing_bounds),
        "n_categorical_cols": len(cat_cols),
        "n_categorical_with_categories": len(cat_cols) - len(missing_cats),
        "missing_bounds": missing_bounds,
        "missing_categories": missing_cats,
        "compliance_gap": len(missing_bounds) > 0 or len(missing_cats) > 0,
    }


if __name__ == "__main__":
    import json, sys, pprint
    with open(sys.argv[1]) as f:
        schema = json.load(f)
    report = dpmm_domain_coverage_report(schema)
    pprint.pprint(report)
