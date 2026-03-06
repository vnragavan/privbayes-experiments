"""
implementations/dpmm_wrapper.py

Wrapper for dpmm PrivBayesGM using the correct TableBinner pipeline.

Correct pipeline (from dpmm source):
  1. TableBinner.fit_transform(df)  → encodes to non-negative integers,
                                       handles NaN via indicator columns,
                                       bins numerics to consecutive ints
  2. binner.bin_domain              → {col: n_values} for PrivBayesGM
  3. PrivBayesGM.fit(encoded_df)
  4. PrivBayesGM.generate(n)        → encoded synthetic df
  5. binner.inverse_transform(...)  → original value space

schema_to_dpmm_domain() returns exactly the domain format TableBinner expects:
  categorical/ordinal/binary → {"categories": [...]}
  integer/continuous         → {"lower": lo, "upper": hi, "n_bins": N}
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _is_numeric_str(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


class DPMMWrapper:
    """Uniform interface: fit(df, schema) / sample(n) / privacy_report()"""

    def __init__(self, epsilon: float, delta: float = 1e-5,
                 degree: int = 2, seed: int = 0, **kwargs):
        self.epsilon   = epsilon
        self.delta     = delta
        self.degree    = degree
        self.seed      = seed
        self._extra    = kwargs
        self._model    = None
        self._binner   = None
        self._coverage = None
        self._schema   = None

    def fit(self, df, schema) -> "DPMMWrapper":
        from dpmm.models.priv_bayes import PrivBayesGM
        from dpmm.processing.table_binner import TableBinner
        from adapters.schema_to_dpmm import (
            schema_to_dpmm_domain, dpmm_domain_coverage_report)
        from numpy.random import RandomState

        if isinstance(schema, str):
            import json
            with open(schema) as f:
                schema = json.load(f)

        # schema_to_dpmm_domain returns the exact format TableBinner expects:
        #   categorical/ordinal/binary → {"categories": [...]}
        #   integer/continuous         → {"lower": lo, "upper": hi, "n_bins": N}
        rich_domain    = schema_to_dpmm_domain(schema)
        self._coverage = dpmm_domain_coverage_report(schema)
        self._schema   = schema

        # Only keep columns that are in both the schema domain and the dataframe
        columns  = [c for c in rich_domain if c in df.columns]
        df_work  = df[columns].copy()

        col_types = schema.get("column_types", {})
        raw_cats  = schema.get("public_categories", {})

        # Cast categorical/ordinal/binary to object first so TableBinner routes
        # them through OrdinalEncoder (not uniform binner). Do this before
        # dropna so we never pass float64/int64 for these columns.
        for col in columns:
            if col_types.get(col) in ("categorical", "ordinal", "binary"):
                df_work[col] = df_work[col].astype(object)

        # Drop rows with any NaN so we never introduce the string "nan" below.
        n_before = len(df_work)
        df_work = df_work.dropna().copy()
        if len(df_work) < n_before:
            print(f"  [dpmm] dropped {n_before - len(df_work)} rows with NaN "
                  f"({len(df_work)} rows for fit)")

        # Align with schema category type (e.g. "0","1") so OrdinalEncoder
        # only sees known categories.
        for col in columns:
            if col_types.get(col) in ("categorical", "ordinal", "binary"):
                if col in raw_cats and raw_cats[col]:
                    example = raw_cats[col][0]
                    if isinstance(example, str):
                        df_work[col] = df_work[col].astype(str)

        rng = RandomState(self.seed)
        self._binner = TableBinner(
            binner_type="uniform",
            binner_settings={"n_bins": "auto"},
            domain=rich_domain,
            random_state=rng,
        )
        df_encoded  = self._binner.fit_transform(df_work, public=True)
        int_domain  = self._binner.bin_domain

        print(f"  [dpmm] int_domain={int_domain}")
        print(f"  [dpmm] total_cells={sum(int_domain.values())}")
        print(f"  [dpmm] encoded shape={df_encoded.shape}  "
              f"dtypes={df_encoded.dtypes.value_counts().to_dict()}")

        self._model = PrivBayesGM(
            epsilon=self.epsilon,
            delta=self.delta,
            degree=self.degree,
            domain=int_domain,
            random_state=RandomState(self.seed),
            n_jobs=1,
            **self._extra,
        )
        self._model.fit(df_encoded)
        return self

    def sample(self, n: int):
        if self._model is None or self._binner is None:
            raise RuntimeError("Call fit() first.")
        df_encoded = self._model.generate(n_records=n)
        df_synth   = self._binner.inverse_transform(df_encoded)

        # Post-process dtypes to match schema categories exactly.
        # TableBinner may return ordinal/binary values as floats after
        # inverse_transform. Cast back to the type implied by the schema
        # category strings:
        #   "0", "1"        → int64   (e.g. status, sex)
        #   "0.0", "1.0"    → float64 (e.g. ph.ecog, ph.karno)
        schema    = self._schema
        col_types = schema.get("column_types", {})
        raw_cats  = schema.get("public_categories", {})
        for col in df_synth.columns:
            ctype = col_types.get(col)
            if ctype not in ("binary", "ordinal"):
                continue
            cats = raw_cats.get(col, [])
            if not cats or not all(_is_numeric_str(c) for c in cats):
                continue
            if any("." in str(c) for c in cats):
                # Float-string categories like "0.0", "1.0" — keep as float
                df_synth[col] = (
                    pd.to_numeric(df_synth[col], errors="coerce")
                    .fillna(0.0)
                )
            else:
                # Integer-string categories like "0", "1" — cast to int
                df_synth[col] = (
                    pd.to_numeric(df_synth[col], errors="coerce")
                    .fillna(0)
                    .round()
                    .astype(np.int64)
                )

        return df_synth

    def privacy_report(self) -> dict:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        g = getattr(self._model, "generator", self._model)
        return {
            "epsilon":         getattr(g, "epsilon", self.epsilon),
            "delta":           getattr(g, "delta",   self.delta),
            "rho":             getattr(g, "rho",     None),
            "sigma":           getattr(g, "sigma",   None),
            "n_source":        "data_derived",
            "schema_coverage": self._coverage,
        }

    def coverage_report(self) -> dict:
        return self._coverage or {}