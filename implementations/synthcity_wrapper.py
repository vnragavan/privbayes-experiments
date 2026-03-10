from __future__ import annotations

import json
import numpy as np
import pandas as pd

from adapters.schema_normalization import (
    normalize_to_schema_output,
    prepare_fit_df_for_synthcity,
)


class SynthCityWrapper:
    """
    Minimal schema adapter for fair comparison.

    This wrapper only:
    - aligns fit-time dtypes with the provided schema
    - normalizes sampled outputs back to schema-valid representation

    It does not modify SynthCity's internal learning algorithm.

    Change log (schema_generator v1.6 alignment):
      - n_bins is now derived from the schema when a schema is provided,
        rather than using the hardcoded constructor default of 100.
        Derivation: median of public_bounds[col]["n_bins"] across all
        integer/continuous columns.  For the lung dataset this gives 51
        (median of [100, 42, 60, 53]) vs the old fixed 100, which was
        inflating SynthCity's CPT size relative to CRNPrivBayes (n_bins=9).
      - The constructor n_bins parameter is retained as an explicit override.
        When schema is provided and n_bins is not overridden at construction
        time, the schema-derived value is used.  This is recorded in
        privacy_report() for audit traceability.
    """

    # Sentinel: constructor was not given an explicit n_bins override
    _N_BINS_AUTO = object()

    def __init__(
        self,
        epsilon: float,
        n_bins: int | None = None,
        target_usefulness: int = 5,
        **kwargs,
    ):
        """
        Parameters
        ----------
        epsilon : float
            DP epsilon passed to SynthCity PrivBayes backend.
        n_bins : int or None
            Bin count for SynthCity's internal PrivBayes discretization.
            When None (default), the value is derived from the schema at
            fit() time as the median of public_bounds[col]["n_bins"] across
            integer/continuous columns.  Pass an explicit int to override.
        target_usefulness : int
            Passed to SynthCity PrivBayes backend.
        """
        self.epsilon = float(epsilon)
        # Store None to signal "auto-derive from schema at fit time"
        self._n_bins_override = int(n_bins) if n_bins is not None else None
        self.n_bins = int(n_bins) if n_bins is not None else 100  # working value
        self.target_usefulness = int(target_usefulness)
        self._extra = dict(kwargs)

        self._model = None
        self._schema = None
        self._fit_columns: list[str] | None = None
        self._n_bins_source: str = "constructor_default"

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_n_bins_from_schema(schema: dict) -> int | None:
        """
        Derive a single n_bins value from the schema for use with SynthCity.

        Strategy: median of public_bounds[col]["n_bins"] across integer and
        continuous columns.  This gives SynthCity a bin count that is
        consistent with the raw domain cardinality of the dataset rather than
        a magic constant.

        Returns None if no numeric columns with n_bins are found (caller
        should then fall back to the default).
        """
        raw_bounds = schema.get("public_bounds", {})
        col_types = schema.get("column_types", {})
        values = []
        for col, ctype in col_types.items():
            if ctype in ("integer", "continuous"):
                entry = raw_bounds.get(col)
                if isinstance(entry, dict):
                    v = entry.get("n_bins")
                    if v is not None:
                        try:
                            values.append(int(v))
                        except (TypeError, ValueError):
                            pass
        if not values:
            return None
        return max(2, int(np.median(values)))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, schema: dict | str | None = None) -> "SynthCityWrapper":
        from synthcity_standalone.privbayes import PrivBayes

        if isinstance(schema, str):
            with open(schema) as f:
                schema = json.load(f)

        self._schema = schema
        self._fit_columns = list(df.columns)

        # Resolve n_bins: explicit override > schema-derived > constructor default
        if self._n_bins_override is not None:
            n_bins_resolved = self._n_bins_override
            self._n_bins_source = "constructor_override"
        elif schema is not None:
            derived = self._derive_n_bins_from_schema(schema)
            if derived is not None:
                n_bins_resolved = derived
                self._n_bins_source = "schema_derived_median"
            else:
                n_bins_resolved = self.n_bins  # fallback to default 100
                self._n_bins_source = "constructor_default_no_schema_bins"
        else:
            n_bins_resolved = self.n_bins
            self._n_bins_source = "constructor_default_no_schema"

        self.n_bins = n_bins_resolved  # record resolved value for privacy_report

        eps_safe = max(self.epsilon, 1e-6)
        n_bins_safe = max(n_bins_resolved, 2)

        df_fit = prepare_fit_df_for_synthcity(df, schema)

        # PrivBayes does not accept 'seed'; drop it for backend compatibility
        backend_kwargs = {k: v for k, v in self._extra.items() if k != "seed"}
        self._model = PrivBayes(
            epsilon=eps_safe,
            n_bins=n_bins_safe,
            target_usefulness=self.target_usefulness,
            **backend_kwargs,
        )
        self._model.fit(df_fit)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        out = self._model.sample(n)
        out = normalize_to_schema_output(
            out,
            self._schema,
            fit_columns=self._fit_columns,
        )
        return out

    def privacy_report(self) -> dict:
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        return {
            "epsilon": getattr(self._model, "epsilon", self.epsilon),
            "n_bins_used": self.n_bins,
            "n_bins_source": self._n_bins_source,
            "n_source": "data_derived",
            "bounds_source": "private_data_inside_backend",
            "categories_source": "private_data_inside_backend",
            "schema_injection": (
                "fit_dtype_alignment_and_output_normalization"
                if self._schema else "not_used"
            ),
            "compliance_gap": (
                "Wrapper corrects data representation and n_bins only. "
                "SynthCity remains non-schema-native internally."
            ),
        }
