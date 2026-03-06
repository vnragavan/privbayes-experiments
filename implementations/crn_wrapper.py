from implementations.crn_privbayes import PrivBayesSynthesizerEnhanced

class CRNWrapper:
    def __init__(self, epsilon, delta=1e-6, seed=0,
                 max_parents=2, **kwargs):
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        self.max_parents = max_parents
        self._extra = kwargs
        self._model = None

    def fit(self, df, schema) -> "CRNWrapper":
        self._model = PrivBayesSynthesizerEnhanced(
            epsilon=self.epsilon,
            delta=self.delta,
            seed=self.seed,
            max_parents=self.max_parents,
            **self._extra,
        )
        self._model.fit(df, schema=schema)
        return self

    def sample(self, n):
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        synth_df = self._model.sample(n)

        # Clip numeric columns to schema bounds (original scale)
        schema = getattr(self._model, '_schema', None)
        if schema is not None:
            import pandas as pd
            pb = schema.get('public_bounds', {})
            ct = schema.get('column_types', {})
            for col, bv in pb.items():
                if col not in synth_df.columns:
                    continue
                if ct.get(col) not in ('continuous', 'integer'):
                    continue
                lo = bv.get('min') if isinstance(bv, dict) else bv[0]
                hi = bv.get('max') if isinstance(bv, dict) else bv[1]
                if lo is not None and hi is not None:
                    synth_df[col] = pd.to_numeric(
                        synth_df[col], errors='coerce'
                    ).clip(float(lo), float(hi))

        return synth_df

    def validate_output(self, synth_df):
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.validate_output(synth_df)

    def privacy_report(self):
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.privacy_report()
