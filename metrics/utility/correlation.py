"""
metrics/utility/correlation.py
"""

import numpy as np
import pandas as pd
from scipy import stats


def spearman_matrix_similarity(real_df, synth_df, numeric_cols) -> float:
    try:
        cols = [c for c in numeric_cols
                if c in real_df.columns and c in synth_df.columns]
        if len(cols) < 2:
            return float("nan")
        r = real_df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
        s = synth_df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
        n = len(cols)
        idx = np.triu_indices(n, k=1)
        rv = r.values[idx]
        sv = s.values[idx]
        mask = ~(np.isnan(rv) | np.isnan(sv))
        if mask.sum() < 2:
            return float("nan")
        return float(stats.spearmanr(rv[mask], sv[mask]).statistic)
    except Exception:
        return float("nan")


def _cramers_v(x, y):
    try:
        ct = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(ct, correction=False)[0]
        n = ct.values.sum()
        r, k = ct.shape
        phi2 = chi2 / n
        phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        rc = r - (r - 1) ** 2 / (n - 1)
        kc = k - (k - 1) ** 2 / (n - 1)
        denom = min(rc - 1, kc - 1)
        if denom <= 0:
            return float("nan")
        return float(np.sqrt(phi2c / denom))
    except Exception:
        return float("nan")


def mixed_association_similarity(real_df, synth_df, schema: dict) -> dict:
    try:
        col_types = schema.get("column_types", {})
        numeric_cols = [c for c, t in col_types.items()
                        if t in ("continuous", "integer")
                        and c in real_df.columns and c in synth_df.columns]
        cat_cols = [c for c, t in col_types.items()
                    if t in ("categorical", "ordinal", "binary")
                    and c in real_df.columns and c in synth_df.columns]

        num_sim = spearman_matrix_similarity(real_df, synth_df, numeric_cols)

        # Cramer's V per pair — compare real vs synth
        if len(cat_cols) >= 2:
            pairs = [(c1, c2) for i, c1 in enumerate(cat_cols)
                     for c2 in cat_cols[i + 1:]]
            r_vs, s_vs = [], []
            for c1, c2 in pairs:
                rv = _cramers_v(real_df[c1], real_df[c2])
                sv = _cramers_v(synth_df[c1], synth_df[c2])
                if not np.isnan(rv) and not np.isnan(sv):
                    r_vs.append(rv)
                    s_vs.append(sv)
            if len(r_vs) >= 2:
                cat_sim = float(stats.spearmanr(r_vs, s_vs).statistic)
            else:
                cat_sim = float("nan")
        else:
            cat_sim = float("nan")

        return {
            "numeric_spearman": num_sim,
            "categorical_cramersv_spearman": cat_sim,
        }
    except Exception as e:
        return {"numeric_spearman": float("nan"),
                "categorical_cramersv_spearman": float("nan"),
                "error": str(e)}
