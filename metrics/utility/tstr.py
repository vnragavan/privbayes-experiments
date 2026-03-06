"""
metrics/utility/tstr.py

Train-on-Synthetic, Test-on-Real classification.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


def tstr_classification(synth_df, test_real_df, target_col,
                         feature_cols) -> dict:
    try:
        avail = [c for c in feature_cols
                 if c in synth_df.columns and c in test_real_df.columns]
        if not avail or target_col not in synth_df.columns:
            return {"roc_auc": float("nan"), "error": "missing columns"}

        X_train = synth_df[avail].apply(
            pd.to_numeric, errors="coerce").fillna(0)
        y_train = synth_df[target_col]
        X_test = test_real_df[avail].apply(
            pd.to_numeric, errors="coerce").fillna(0)
        y_test = test_real_df[target_col]

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train.astype(str))
        y_test_enc = le.transform(
            y_test.astype(str).where(
                y_test.astype(str).isin(le.classes_),
                le.classes_[0]))

        clf = LogisticRegression(max_iter=500, random_state=0)
        clf.fit(X_train, y_train_enc)
        proba = clf.predict_proba(X_test)

        n_classes = len(le.classes_)
        if n_classes == 2:
            auc = float(roc_auc_score(y_test_enc, proba[:, 1]))
        else:
            auc = float(roc_auc_score(
                y_test_enc, proba, multi_class="ovr"))

        return {
            "roc_auc": auc,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(avail),
        }
    except Exception as e:
        return {"roc_auc": float("nan"), "error": str(e)}
