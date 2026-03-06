"""
metrics/privacy/mia.py

Distance-based Membership Inference Attack (Stadler et al. 2022).
Answers RQ3: does SynthCity's utility advantage come from memorisation?
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _numeric_matrix(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values


def mia_nearest_neighbour(train_df, holdout_df, synth_df,
                           n_attack_samples=500) -> dict:
    """
    For each record in the attack set (mix of train and holdout),
    feature = distance to nearest synth record / distance to nearest train record.
    Train logistic regression to distinguish membership.

    AUC = 0.5 → random — no membership leakage
    AUC > 0.5 → model leaks membership information
    """
    try:
        shared_cols = [c for c in train_df.columns
                       if c in holdout_df.columns and c in synth_df.columns]
        if not shared_cols:
            return {"auc": float("nan"), "error": "no shared columns"}

        # Sample attack set
        n_each = min(n_attack_samples // 2, len(train_df), len(holdout_df))
        rng = np.random.RandomState(42)
        train_sample = train_df.sample(n=n_each, random_state=rng).reset_index(drop=True)
        holdout_sample = holdout_df.sample(n=n_each, random_state=rng).reset_index(drop=True)

        X_train_mat = _numeric_matrix(train_sample, shared_cols)
        X_hold_mat = _numeric_matrix(holdout_sample, shared_cols)
        X_synth_mat = _numeric_matrix(synth_df, shared_cols)

        # For each attack record: ratio of distances
        def nn_ratio(query_mat, ref_mat, synth_mat):
            features = []
            for x in query_mat:
                d_synth = np.min(np.linalg.norm(synth_mat - x, axis=1))
                d_ref = np.min(np.linalg.norm(
                    np.delete(ref_mat,
                               np.argmin(np.linalg.norm(ref_mat - x, axis=1)),
                               axis=0) if len(ref_mat) > 1 else ref_mat,
                    axis=1))
                features.append(d_synth / max(d_ref, 1e-12))
            return np.array(features).reshape(-1, 1)

        feat_train = nn_ratio(X_train_mat, X_train_mat, X_synth_mat)
        feat_hold = nn_ratio(X_hold_mat, X_train_mat, X_synth_mat)

        X_attack = np.vstack([feat_train, feat_hold])
        y_attack = np.array([1] * n_each + [0] * n_each)

        clf = LogisticRegression(random_state=0)
        clf.fit(X_attack, y_attack)
        proba = clf.predict_proba(X_attack)[:, 1]
        auc = float(roc_auc_score(y_attack, proba))

        return {
            "auc": auc,
            "advantage": round(auc - 0.5, 4),
            "n_train_in_attack": n_each,
            "n_holdout_in_attack": n_each,
        }
    except Exception as e:
        return {"auc": float("nan"), "advantage": float("nan"), "error": str(e)}
