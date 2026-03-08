from __future__ import annotations

import json
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
    """

    def __init__(
        self,
        epsilon: float,
        n_bins: int = 100,
        target_usefulness: int = 5,
        **kwargs,
    ):
        self.epsilon = float(epsilon)
        self.n_bins = int(n_bins)
        self.target_usefulness = int(target_usefulness)
        self._extra = dict(kwargs)

        self._model = None
        self._schema = None
        self._fit_columns: list[str] | None = None

    def fit(self, df: pd.DataFrame, schema: dict | str | None = None) -> "SynthCityWrapper":
        from synthcity_standalone.privbayes import PrivBayes

        if isinstance(schema, str):
            with open(schema) as f:
                schema = json.load(f)

        self._schema = schema
        self._fit_columns = list(df.columns)

        eps_safe = max(self.epsilon, 1e-6)
        n_bins_safe = max(self.n_bins, 2)

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
            "n_source": "data_derived",
            "bounds_source": "private_data_inside_backend",
            "categories_source": "private_data_inside_backend",
            "schema_injection": "fit_dtype_alignment_and_output_normalization" if self._schema else "not_used",
            "compliance_gap": (
                "Wrapper corrects data representation only. "
                "SynthCity remains non-schema-native internally."
            ),
        }
