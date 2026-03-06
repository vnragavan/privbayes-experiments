import pandas as pd
from typing import Dict
from dpmm.models.base.base import GenerativeModel


class GraphicalGenerativeModel(GenerativeModel):
    def set_domain(self, domain: Dict):
        super().set_domain(domain)
        self.generator._domain = domain

    def set_structural_zeros(self, structural_zeros):
        self.generator.set_structural_zeros(structural_zeros)

    @property
    def model_size(self):
        return self.generator.model_size

    def check_fit(self, df):
        cls_name = self.__class__.__name__
        # check No Missing Values
        not_na = [col for col, series in df.items() if series.isna().any()]
        not_na_msg = (
            f"Columns {not_na} contains null values. Not supported by {cls_name}"
        )
        assert len(not_na) == 0, not_na_msg

        # Check All integers
        not_int = [col for col, series in df.items() if (series.dtype.kind not in "ui")]
        not_int_msg = (
            f"Columns {not_int} have non-int dtypes. Not Supported by {cls_name}"
        )
        assert len(not_int) == 0, not_int_msg

        # Check all Positive
        not_positive = [col for col, series in df.items() if (series < 0).any()]
        not_positive_msg = f"Columns {not_positive} have non positive values. Not Supported by {cls_name}"
        assert len(not_positive) == 0, not_positive_msg

        if getattr(self, "domain") is not None:
            upper = pd.Series(self.domain)
            real = df.max(axis=0)

            mismatch = real > upper

            if mismatch.any():
                to_high = real.loc[mismatch]
                raise ValueError(
                    f"Columns {to_high.index.tolist()} have values higher than provided domain."
                )

    def fit(self, df, public=False, marginals_only: bool = False):
        """
        Fit the model to the data.

        :param df: The dataset.
        :type df: pd.DataFrame
        :param public: Whether the data is public. Defaults to False.
        :type public: bool, optional

        **Example**::

            >>> fit(df, public=True)
        """
        self.generator.fit(df, public=public, marginals_only=marginals_only)

    def generate(self, n_records: int = None, condition_records: pd.DataFrame = None):
        """
        Generate synthetic data using the model.

        :param n_records: Number of records to generate. Defaults to None.
        :type n_records: int, optional
        :param condition_records: Conditional records. Defaults to None.
        :type condition_records: pd.DataFrame, optional
        :return: The generated synthetic data.
        :rtype: pd.DataFrame

        **Example**::

            >>> generate(n_records=100)
        """
        return self.generator.generate(
            n_records=n_records, condition_records=condition_records
        )
