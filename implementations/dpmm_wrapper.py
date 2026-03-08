from __future__ import annotations

import json
import pandas as pd

from adapters.schema_normalization import (
    normalize_to_schema_output,
    prepare_fit_df_for_dpmm,
)


class DPMMWrapper:
    """Uniform interface: fit(df, schema) / sample(n) / privacy_report()"""

    def __init__(self, epsilon: float, delta: float = 1e-5,
                 degree: int = 2, seed: int = 0, **kwargs):
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.degree = int(degree)
        self.seed = int(seed)
        self._extra = dict(kwargs)

        self._model = None
        self._binner = None
        self._coverage = None
        self._schema = None
        self._fit_columns: list[str] | None = None

    def fit(self, df: pd.DataFrame, schema: dict | str) -> "DPMMWrapper":
        from numpy.random import RandomState
        from dpmm.models.priv_bayes import PrivBayesGM
        from dpmm.processing.table_binner import TableBinner
        from adapters.schema_to_dpmm import (
            schema_to_dpmm_domain,
            dpmm_domain_coverage_report,
        )

        if isinstance(schema, str):
            with open(schema) as f:
                schema = json.load(f)

        self._schema = schema

        rich_domain = schema_to_dpmm_domain(schema)
        self._coverage = dpmm_domain_coverage_report(schema)

        columns = [c for c in rich_domain if c in df.columns]
        self._fit_columns = columns

        df_work = df[columns].copy()
        df_work = prepare_fit_df_for_dpmm(df_work, schema)

        rng = RandomState(self.seed)

        self._binner = TableBinner(
            binner_type="uniform",
            binner_settings={"n_bins": "auto"},
            domain=rich_domain,
            random_state=rng,
        )

        # Important: do not drop rows here. Let TableBinner handle missingness.
        df_encoded = self._binner.fit_transform(df_work, public=True)
        int_domain = self._binner.bin_domain

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

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None or self._binner is None:
            raise RuntimeError("Call fit() first.")

        df_encoded = self._model.generate(n_records=n)
        df_synth = self._binner.inverse_transform(df_encoded)

        df_synth = normalize_to_schema_output(
            df_synth,
            self._schema,
            fit_columns=self._fit_columns,
        )
        return df_synth

    def privacy_report(self) -> dict:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        g = getattr(self._model, "generator", self._model)
        return {
            "epsilon": getattr(g, "epsilon", self.epsilon),
            "delta": getattr(g, "delta", self.delta),
            "rho": getattr(g, "rho", None),
            "sigma": getattr(g, "sigma", None),
            "n_source": "data_derived",
            "schema_coverage": self._coverage,
        }

    def coverage_report(self) -> dict:
        return self._coverage or {}
