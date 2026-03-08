import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


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


def tstr_classification(
    synth_df,
    test_real_df,
    target_col,
    feature_cols,
    schema: dict | None = None,
) -> dict:
    try:
        avail = [c for c in feature_cols if c in synth_df.columns and c in test_real_df.columns]
        if not avail or target_col not in synth_df.columns or target_col not in test_real_df.columns:
            return {"roc_auc": float("nan"), "error": "missing columns"}

        X_train = synth_df[avail].copy()
        X_test = test_real_df[avail].copy()
        y_train = synth_df[target_col]
        y_test = test_real_df[target_col]

        numeric_cols, categorical_cols = _split_feature_types(avail, schema)

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

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train.astype(str))
        y_test_aligned = y_test.astype(str).where(y_test.astype(str).isin(le.classes_), le.classes_[0])
        y_test_enc = le.transform(y_test_aligned)

        clf = Pipeline([
            ("pre", pre),
            ("model", LogisticRegression(max_iter=2000, random_state=0)),
        ])
        clf.fit(X_train, y_train_enc)
        proba = clf.predict_proba(X_test)

        n_classes = len(le.classes_)
        if n_classes == 2:
            auc = float(roc_auc_score(y_test_enc, proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_test_enc, proba, multi_class="ovr", average="macro"))

        return {
            "roc_auc": auc,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(avail),
            "n_numeric": len(numeric_cols),
            "n_categorical": len(categorical_cols),
        }
    except Exception as e:
        return {"roc_auc": float("nan"), "error": str(e)}
