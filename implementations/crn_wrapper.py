"""
Thin adapter so CRN/PrivBayes can be used via the same interface as DPMM and SynthCity.

Constraint enforcement is done inside the schema-native implementation (PrivBayesSynthesizerEnhanced);
this wrapper only delegates fit/sample/validate_output/privacy_report.

Change log (schema_generator v1.6 alignment):
  - Removed hardcoded max_parents=2 constructor default.
    max_parents is now set by load_schema() from extensions.privbayes.max_parents
    (sensitivity-crossover value computed by schema_generator).  Passing a fixed
    value here would silently override the schema-derived optimum.
  - The constructor still accepts max_parents as an explicit override for callers
    that do not use a schema, or that want to force a specific value.
"""
from implementations.crn_privbayes import PrivBayesSynthesizerEnhanced


class CRNWrapper:
    def __init__(self, epsilon, delta=1e-6, seed=0,
                 max_parents=None, **kwargs):
        """
        Parameters
        ----------
        epsilon : float
            Total DP budget for synthesis (ε₂).  Schema epsilon (ε₁) is
            accounted for separately by schema_generator.
        delta : float
            DP delta for (ε,δ)-DP smooth-sensitivity bounds (non-schema path).
        seed : int or None
            Sampling RNG seed (non-DP randomness only).
        max_parents : int or None
            Override max parents per node.  When None (default), the value is
            read from extensions.privbayes.max_parents in the schema JSON.
            Only set this explicitly if you want to deviate from the
            schema-derived sensitivity-crossover value.
        **kwargs
            Forwarded verbatim to PrivBayesSynthesizerEnhanced.__init__().
        """
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        self.max_parents = max_parents  # None means "let load_schema() decide"
        self._extra = kwargs
        self._model = None

    def fit(self, df, schema) -> "CRNWrapper":
        # Build init kwargs.  Only pass max_parents when the caller explicitly
        # provided a value; otherwise leave it at the class default (2) so that
        # load_schema() can override it from extensions.privbayes.max_parents.
        init_kwargs = dict(self._extra)
        if self.max_parents is not None:
            init_kwargs["max_parents"] = self.max_parents

        self._model = PrivBayesSynthesizerEnhanced(
            epsilon=self.epsilon,
            delta=self.delta,
            seed=self.seed,
            **init_kwargs,
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
