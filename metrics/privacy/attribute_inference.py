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
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def get_attribute_inference_target(schema: dict) -> str | None:
    sensitive = schema.get("sensitive_attributes")
    if isinstance(sensitive, list) and len(sensitive) > 0:
        return str(sensitive[0])
    return schema.get("target_spec", {}).get("primary_target")


def _split_feature_types(feature_cols, schema: dict | None):
    if not schema:
        return feature_cols, []

    col_types = schema.get("column_types", {})
    numeric = []
    categorical = []

    for c in feature_cols:
        t = col_types.get(c)
        if t in ("continuous", "integer", "binary"):
            numeric.append(c)
        else:
            categorical.append(c)

    return numeric, categorical


def attribute_inference_auc(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str | None,
    schema: dict | None = None,
) -> dict:
    if not target_col or target_col not in real_df.columns or target_col not in synth_df.columns:
        return {"auc": float("nan"), "target": target_col, "n_classes": 0, "error": "no target column"}

    shared = [c for c in real_df.columns if c in synth_df.columns and c != target_col]
    if not shared:
        return {"auc": float("nan"), "target": target_col, "n_classes": 0, "error": "no feature columns"}

    X_real = real_df[shared].copy()
    X_synth = synth_df[shared].copy()
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

    numeric_cols, categorical_cols = _split_feature_types(shared, schema)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    try:
        clf = Pipeline([
            ("pre", pre),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ])
        clf.fit(X_synth, y_synth)

        if n_classes == 2:
            proba = clf.predict_proba(X_real)[:, 1]
            auc = float(roc_auc_score(y_real, proba))
        else:
            proba = clf.predict_proba(X_real)
            auc = float(roc_auc_score(y_real, proba, multi_class="ovr", average="macro"))
    except Exception as e:
        return {"auc": float("nan"), "target": target_col, "n_classes": n_classes, "error": str(e)}

    return {
        "auc": auc,
        "target": target_col,
        "n_classes": n_classes,
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
    }
