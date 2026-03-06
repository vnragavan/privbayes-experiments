"""
metrics/utility/coverage.py
"""

import numpy as np
import pandas as pd


def categorical_coverage(synth_df, schema: dict) -> dict:
    """
    Fraction of schema-defined categories appearing at least once in synth_df.
    """
    col_types = schema.get("column_types", {})
    public_cats = schema.get("public_categories", {})
    per_col = {}

    for col, ctype in col_types.items():
        if ctype not in ("categorical", "ordinal", "binary"):
            continue
        if col not in synth_df.columns:
            continue
        cats = public_cats.get(col)
        if not cats:
            continue
        present = set(synth_df[col].astype(str).unique())
        schema_cats = set(str(c) for c in cats)
        covered = len(schema_cats & present)
        per_col[col] = covered / len(schema_cats) if schema_cats else float("nan")

    vals = [v for v in per_col.values() if not np.isnan(v)]
    return {
        "per_column": per_col,
        "mean": float(np.mean(vals)) if vals else float("nan"),
    }


def unknown_token_rate(synth_df, unknown_token="__UNK__") -> dict:
    per_col = {}
    for col in synth_df.select_dtypes(include="object").columns:
        total = len(synth_df)
        if total == 0:
            per_col[col] = float("nan")
            continue
        per_col[col] = float((synth_df[col].astype(str) == unknown_token).mean())

    overall_counts = sum(
        int((synth_df[c].astype(str) == unknown_token).sum())
        for c in per_col)
    total_cells = sum(len(synth_df) for c in per_col)

    return {
        "per_column": per_col,
        "overall": overall_counts / total_cells if total_cells > 0 else float("nan"),
    }
