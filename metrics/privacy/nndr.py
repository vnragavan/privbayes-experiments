"""
metrics/privacy/nndr.py

Nearest-Neighbour Distance Ratio.
ratio < 1 → memorisation risk
ratio ≈ 1 → healthy generalisation
ratio > 1 → synthetic data too far from real
"""

import numpy as np
import pandas as pd


def nearest_neighbour_distance_ratio(real_df, synth_df,
                                      numeric_cols: list,
                                      n_samples=1000) -> dict:
    try:
        cols = [c for c in numeric_cols
                if c in real_df.columns and c in synth_df.columns]
        if not cols:
            return {"mean_ratio": float("nan"), "error": "no numeric columns"}

        def mat(df):
            return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

        R = mat(real_df)
        S = mat(synth_df)

        n = min(n_samples, len(R))
        rng = np.random.RandomState(0)
        idx = rng.choice(len(R), n, replace=False)
        R_sample = R[idx]

        ratios = []
        for i, x in enumerate(R_sample):
            # distance to nearest synth
            d_synth = np.min(np.linalg.norm(S - x, axis=1))
            # distance to nearest real (excluding self)
            dists_real = np.linalg.norm(R - x, axis=1)
            dists_real[idx[i]] = np.inf
            d_real = np.min(dists_real)
            if d_real < 1e-12:
                continue
            ratios.append(d_synth / d_real)

        if not ratios:
            return {"mean_ratio": float("nan"), "error": "all zero distances"}

        mean_r = float(np.mean(ratios))
        median_r = float(np.median(ratios))
        frac_below_1 = float(np.mean(np.array(ratios) < 1.0))

        if mean_r < 0.9:
            interp = "memorisation risk: synthetic records closer to real than real-to-real"
        elif mean_r > 1.1:
            interp = "low utility: synthetic records far from real data"
        else:
            interp = "healthy generalisation: ratio near 1.0"

        return {
            "mean_ratio": mean_r,
            "median_ratio": median_r,
            "fraction_below_1": frac_below_1,
            "interpretation": interp,
        }
    except Exception as e:
        return {"mean_ratio": float("nan"), "error": str(e)}
