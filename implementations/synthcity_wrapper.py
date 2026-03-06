"""
implementations/synthcity_wrapper.py

Wrapper for SynthCity PrivBayes.
No changes are made inside synthcity_standalone/; all behavior is controlled here.
- Parameter clamping (epsilon, n_bins) so the backend never sees invalid values.
- Optional schema use: pre-clip train data and post-clip samples to schema bounds
  (schema is not injected into SynthCity; wrapper applies bounds only).
"""

from __future__ import annotations

import json
import pandas as pd


def _schema_to_bounds_and_categories(schema: dict) -> tuple[dict, dict]:
    """From schema (dict or path), return (bounds_by_col, categories_by_col). Bounds: {col: (lo, hi)}. Categories: {col: list}."""
    if isinstance(schema, str):
        with open(schema) as f:
            schema = json.load(f)
    bounds = {}
    raw_bounds = schema.get("public_bounds", {})
    for col, bv in raw_bounds.items():
        if isinstance(bv, dict):
            bounds[col] = (float(bv["min"]), float(bv["max"]))
        elif isinstance(bv, (list, tuple)) and len(bv) >= 2:
            bounds[col] = (float(bv[0]), float(bv[1]))
    categories = {}
    for col, cats in schema.get("public_categories", {}).items():
        categories[col] = list(cats)
    return bounds, categories


def _clip_df_to_schema(df: pd.DataFrame, bounds: dict, categories: dict) -> pd.DataFrame:
    """Clip numeric columns to bounds; ensure categorical columns are in allowed set (map unknown to first)."""
    out = df.copy()
    for col in list(out.columns):
        if col in bounds:
            lo, hi = bounds[col]
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(lo).clip(lo, hi)
        if col in categories:
            allowed = set(categories[col])
            out[col] = out[col].apply(lambda x: x if x in allowed else categories[col][0])
    return out


class SynthCityWrapper:
    """
    Uniform interface: fit(df, schema) / sample(n) / privacy_report().
    All logic is in the wrapper; nothing is changed inside synthcity_standalone/.
    """

    def __init__(self, epsilon: float, n_bins: int = 100,
                 target_usefulness: int = 5, **kwargs):
        self.epsilon = epsilon
        self.n_bins = n_bins
        self.target_usefulness = target_usefulness
        self._extra = kwargs
        self._model = None
        self._bounds = None
        self._categories = None

    def fit(self, df, schema=None) -> "SynthCityWrapper":
        from synthcity_standalone.privbayes import PrivBayes

        if isinstance(schema, str):
            with open(schema) as f:
                schema = json.load(f)

        # Clamp so backend never sees invalid params (avoids div-by-zero / inf)
        eps_safe = max(float(self.epsilon), 1e-6)
        n_bins_safe = max(2, int(self.n_bins))

        if schema is not None:
            self._bounds, self._categories = _schema_to_bounds_and_categories(schema)
            df = _clip_df_to_schema(df, self._bounds, self._categories)
        else:
            self._bounds, self._categories = {}, {}

        self._model = PrivBayes(
            epsilon=eps_safe,
            n_bins=n_bins_safe,
            target_usefulness=self.target_usefulness,
        )
        self._model.fit(df)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        out = self._model.sample(n)
        if self._bounds or self._categories:
            out = _clip_df_to_schema(out, self._bounds or {}, self._categories or {})
        return out

    def privacy_report(self) -> dict:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return {
            "epsilon": getattr(self._model, "epsilon", self.epsilon),
            "n_source": "data_derived",
            "bounds_source": "pd.cut_on_private_data",
            "categories_source": "LabelEncoder_on_private_data",
            "schema_injection": "wrapper_pre_and_post_clip" if (self._bounds or self._categories) else "not_used",
            "compliance_gap": (
                "All metadata (bounds, categories, n) derived from "
                "private data before any DP mechanism runs; wrapper only clips to schema."
            ),
        }
