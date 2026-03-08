"""
Thin adapter so CRN/PrivBayes can be used via the same interface as DPMM and SynthCity.

Constraint enforcement is done inside the schema-native implementation (PrivBayesSynthesizerEnhanced);
this wrapper only delegates fit/sample/validate_output/privacy_report.
"""
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
        return self._model.sample(n)

    def validate_output(self, synth_df):
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.validate_output(synth_df)

    def privacy_report(self):
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.privacy_report()
