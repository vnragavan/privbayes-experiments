"""
metrics/privacy/attribute_inference.py

Attribute inference risk: train a classifier on synthetic data to predict
a target (sensitive) attribute from the rest; evaluate AUC on real holdout.
High AUC = synthetic data preserves relationship → higher inference risk.

The sensitive attribute should be defined in the schema via sensitive_attributes
(e.g. ["status"], ["race"], ["income"]). If omitted, falls back to
target_spec.primary_target for backward compatibility.
"""

import numpy as np


def get_attribute_inference_target(schema: dict) -> str | None:
    """
    Resolve which column to use as the attribute-inference target (sensitive
    attribute). Prefer schema["sensitive_attributes"][0]; if absent, use
    target_spec.primary_target.
    """
    sensitive = schema.get("sensitive_attributes")
    if isinstance(sensitive, list) and len(sensitive) > 0:
        return str(sensitive[0])
    return schema.get("target_spec", {}).get("primary_target")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


def _numeric_matrix(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values


def attribute_inference_auc(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str | None,
) -> dict:
    """
    Train a classifier on synthetic data to predict target_col from other
    columns; report AUC when evaluated on real data.

    - AUC ≈ 0.5: no exploitable relationship (low inference risk).
    - AUC >> 0.5: synthetic data leaks relationship (higher risk).

    Parameters
    ----------
    real_df   : real (holdout) data
    synth_df  : synthetic data used to train the predictor
    target_col : column to infer (e.g. event/sensitive attribute); if None, skip

    Returns
    -------
    dict with keys: auc, target, n_classes, (optional) error
    """
    if not target_col or target_col not in real_df.columns or target_col not in synth_df.columns:
        return {"auc": float("nan"), "target": target_col, "n_classes": 0, "error": "no target column"}

    shared = [c for c in real_df.columns if c in synth_df.columns and c != target_col]
    if not shared:
        return {"auc": float("nan"), "target": target_col, "n_classes": 0, "error": "no feature columns"}

    X_real = _numeric_matrix(real_df, shared)
    X_synth = _numeric_matrix(synth_df, shared)
    y_real_raw = real_df[target_col].astype(str)
    y_synth_raw = synth_df[target_col].astype(str)

    le = LabelEncoder()
    all_labels = pd.concat([y_real_raw, y_synth_raw], ignore_index=True)
    le.fit(all_labels.astype(str))
    y_synth = le.transform(y_synth_raw)
    y_real = le.transform(y_real_raw)
    n_classes = len(le.classes_)

    if n_classes < 2:
        return {"auc": float("nan"), "target": target_col, "n_classes": n_classes, "error": "single class"}

    if len(np.unique(y_synth)) < 2:
        return {"auc": float("nan"), "target": target_col, "n_classes": n_classes, "error": "single class in synthetic"}

    try:
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_synth, y_synth)
        if n_classes == 2:
            proba = clf.predict_proba(X_real)[:, 1]
            auc = float(roc_auc_score(y_real, proba))
        else:
            proba = clf.predict_proba(X_real)
            auc = float(roc_auc_score(y_real, proba, multi_class="ovr", average="macro"))
    except Exception as e:
        return {"auc": float("nan"), "target": target_col, "n_classes": n_classes, "error": str(e)}

    return {"auc": auc, "target": target_col, "n_classes": n_classes}
