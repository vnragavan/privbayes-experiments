"""
metrics/utility/marginal.py

Schema-aware 1-way and 2-way marginal distance metrics.
Bin edges always come from schema, never from data range.
"""

import numpy as np
import pandas as pd
from itertools import combinations
import random

try:
    from scipy.stats import wasserstein_distance
except ImportError:
    wasserstein_distance = None


def _hist_numeric(series, lo, hi, n_bins=50):
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(
        pd.to_numeric(series, errors="coerce").dropna(), bins=edges)
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


def _hist_categorical(series, categories):
    cats = list(categories)
    vc = series.astype(str).value_counts()
    total = len(series)
    return np.array(
        [vc.get(str(c), 0) / total if total > 0 else 0.0 for c in cats])


def marginal_l1_numeric(real_df, synth_df, col, bounds, n_bins=50) -> float:
    try:
        lo, hi = float(bounds[0]), float(bounds[1])
        p = _hist_numeric(real_df[col], lo, hi, n_bins)
        q = _hist_numeric(synth_df[col], lo, hi, n_bins)
        return float(np.sum(np.abs(p - q)) / 2.0)
    except Exception:
        return float("nan")


def marginal_l1_categorical(real_df, synth_df, col, categories) -> float:
    try:
        p = _hist_categorical(real_df[col], categories)
        q = _hist_categorical(synth_df[col], categories)
        return float(np.sum(np.abs(p - q)) / 2.0)
    except Exception:
        return float("nan")


def mean_marginal_l1(real_df, synth_df, schema: dict) -> dict:
    col_types = schema.get("column_types", {})
    public_bounds = schema.get("public_bounds", {})
    public_cats = schema.get("public_categories", {})

    per_col = {}
    numeric_vals, cat_vals = [], []

    for col, ctype in col_types.items():
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        if ctype in ("continuous", "integer"):
            bv = public_bounds.get(col)
            if bv is None:
                continue
            lo = bv["min"] if isinstance(bv, dict) else bv[0]
            hi = bv["max"] if isinstance(bv, dict) else bv[1]
            v = marginal_l1_numeric(real_df, synth_df, col, [lo, hi])
            per_col[col] = v
            if not np.isnan(v):
                numeric_vals.append(v)
        elif ctype in ("categorical", "ordinal", "binary"):
            cats = public_cats.get(col)
            if cats is None:
                continue
            v = marginal_l1_categorical(real_df, synth_df, col, cats)
            per_col[col] = v
            if not np.isnan(v):
                cat_vals.append(v)

    all_vals = [v for v in per_col.values() if not np.isnan(v)]
    return {
        "per_column": per_col,
        "mean_numeric": float(np.mean(numeric_vals)) if numeric_vals else float("nan"),
        "mean_categorical": float(np.mean(cat_vals)) if cat_vals else float("nan"),
        "mean_overall": float(np.mean(all_vals)) if all_vals else float("nan"),
    }


def mean_wasserstein_per_column(real_df, synth_df, schema: dict) -> dict:
    """
    Per-column 1-Wasserstein distance between real and synthetic marginals,
    then mean over columns. Lower is better.
    Numeric: raw values. Categorical/ordinal/binary: encoded as 0..k-1 by schema order.
    """
    if wasserstein_distance is None:
        return {"per_column": {}, "mean": float("nan"), "error": "scipy not available"}
    col_types = schema.get("column_types", {})
    public_bounds = schema.get("public_bounds", {})
    public_cats = schema.get("public_categories", {})
    per_col = {}
    for col in col_types:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        ctype = col_types[col]
        try:
            if ctype in ("continuous", "integer"):
                r = pd.to_numeric(real_df[col], errors="coerce").dropna()
                s = pd.to_numeric(synth_df[col], errors="coerce").dropna()
                if len(r) == 0 or len(s) == 0:
                    continue
                per_col[col] = float(wasserstein_distance(r, s))
            elif ctype in ("categorical", "ordinal", "binary"):
                cats = public_cats.get(col)
                if not cats:
                    continue
                cat_list = [str(c) for c in cats]
                r = real_df[col].astype(str).map(lambda x: cat_list.index(x) if x in cat_list else -1)
                s = synth_df[col].astype(str).map(lambda x: cat_list.index(x) if x in cat_list else -1)
                r = r[r >= 0].astype(float)
                s = s[s >= 0].astype(float)
                if len(r) == 0 or len(s) == 0:
                    continue
                per_col[col] = float(wasserstein_distance(r, s))
            else:
                continue
        except Exception:
            continue
    vals = [v for v in per_col.values() if not np.isnan(v)]
    return {
        "per_column": per_col,
        "mean": float(np.mean(vals)) if vals else float("nan"),
    }


def pairwise_tvd(real_df, synth_df, schema: dict, max_pairs=50) -> dict:
    """2-way TVD on randomly sampled column pairs."""
    try:
        col_types = schema.get("column_types", {})
        public_bounds = schema.get("public_bounds", {})
        public_cats = schema.get("public_categories", {})

        def discretise(df, col, ctype):
            if ctype in ("continuous", "integer"):
                bv = public_bounds.get(col)
                if bv is None:
                    return None
                lo = bv["min"] if isinstance(bv, dict) else bv[0]
                hi = bv["max"] if isinstance(bv, dict) else bv[1]
                edges = np.linspace(lo, hi, 11)
                return pd.cut(pd.to_numeric(df[col], errors="coerce"),
                              bins=edges, labels=False).astype("Int64")
            else:
                cats = public_cats.get(col)
                if cats is None:
                    return None
                return df[col].astype(str)

        all_cols = [c for c in col_types if c in real_df.columns]
        pairs = list(combinations(all_cols, 2))
        if len(pairs) > max_pairs:
            random.seed(0)
            pairs = random.sample(pairs, max_pairs)

        tvds = []
        for c1, c2 in pairs:
            try:
                r1 = discretise(real_df, c1, col_types[c1])
                r2 = discretise(real_df, c2, col_types[c2])
                s1 = discretise(synth_df, c1, col_types[c1])
                s2 = discretise(synth_df, c2, col_types[c2])
                if r1 is None or r2 is None:
                    continue
                rj = r1.astype(str) + "_" + r2.astype(str)
                sj = s1.astype(str) + "_" + s2.astype(str)
                all_vals = list(set(rj.dropna()) | set(sj.dropna()))
                p = np.array([rj.eq(v).mean() for v in all_vals])
                q = np.array([sj.eq(v).mean() for v in all_vals])
                tvds.append(float(np.sum(np.abs(p - q)) / 2.0))
            except Exception:
                continue

        return {
            "mean": float(np.mean(tvds)) if tvds else float("nan"),
            "max": float(np.max(tvds)) if tvds else float("nan"),
            "n_pairs": len(tvds),
        }
    except Exception as e:
        return {"mean": float("nan"), "max": float("nan"), "error": str(e)}
