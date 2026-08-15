"""
Unified ML Preprocessing
=========================
Single source of truth for preparing features and target.
ALL ML modules MUST call prepare_ml_matrix() instead of their own prepare_xy().
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

_ID_KEYWORDS = [
    "id", "uuid", "hash", "key", "identifier", "index",
    "roll", "roll_number", "rollno", "roll number",
    "application", "registration", "serial", "ticket",
    "account", "customer_id", "order_id", "transaction_id",
    "userid", "studentid", "student_id", "user_id",
]

_TIME_KEYWORDS = [
    "time", "date", "timestamp", "created", "updated",
    "datetime", "ts", "event",
]


def prepare_ml_matrix(df, target, drop_high_unique_ratio=0.95, min_non_missing=5):
    """
    Prepare clean feature matrix and target vector for ML.

    Steps:
    1. Drop rows where target is missing
    2. Drop problematic columns (empty, constant, ID-like, high-unique non-numeric, time-like text)
    3. Re-detect numeric vs categorical features
    4. Validate usable features remain

    Returns dict with: X, y, numeric_features, categorical_features, dropped_columns, warnings, usable
    """
    warnings = []

    if target not in df.columns:
        return {"X": pd.DataFrame(), "y": pd.Series(dtype="float64"),
                "numeric_features": [], "categorical_features": [],
                "dropped_columns": [], "warnings": [f"Target '{target}' not found."], "usable": False}

    data = df.dropna(subset=[target]).copy()
    rows_dropped = len(df) - len(data)
    if rows_dropped > 0:
        warnings.append(f"Dropped {rows_dropped} rows with missing target values.")

    y = data[target]
    X = data.drop(columns=[target])

    dropped = []
    rows = max(len(X), 1)

    for col in list(X.columns):
        non_missing = int(X[col].notna().sum())
        unique_count = X[col].nunique(dropna=True)
        unique_ratio = unique_count / rows
        lower = col.lower().replace("-", "_")

        if non_missing < min_non_missing:
            dropped.append({"column": col, "reason": "insufficient_non_missing"})
            continue
        if unique_count <= 1:
            dropped.append({"column": col, "reason": "constant"})
            continue
        if any(kw in lower for kw in _ID_KEYWORDS):
            dropped.append({"column": col, "reason": "id_like_keyword"})
            continue
        if any(kw in lower for kw in _TIME_KEYWORDS) and not pd.api.types.is_numeric_dtype(X[col]):
            dropped.append({"column": col, "reason": "time_like_text"})
            continue
        if unique_ratio > drop_high_unique_ratio and not pd.api.types.is_numeric_dtype(X[col]):
            dropped.append({"column": col, "reason": "high_unique_non_numeric"})
            continue

    drop_cols = [d["column"] for d in dropped]
    X = X.drop(columns=drop_cols, errors="ignore")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    total_features = len(numeric_features) + len(categorical_features)
    usable = total_features > 0 and len(y) >= 20

    if total_features == 0:
        warnings.append("No usable features remain after preprocessing.")
    if len(y) < 20:
        warnings.append(f"Only {len(y)} rows remain — too few for reliable ML.")

    return {"X": X, "y": y, "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "dropped_columns": dropped, "warnings": warnings, "usable": usable}


def _stringify_categoricals(X):
    """Force every cell in a categorical block to a string.

    Mixed-type categorical columns (e.g. a `has_partner` column that is True
    / False / "__missing__" after imputation) make OneHotEncoder fail when
    sklearn's internal `sorted()` runs on a set that contains both bools and
    strings (`'<' not supported between instances of 'str' and 'bool'`).
    Casting everything to str before the encoder fixes that without
    changing semantics.
    """
    if hasattr(X, "astype"):
        return X.astype(str)
    return np.asarray(X).astype(str)


def build_sklearn_preprocessor(numeric_features, categorical_features):
    """Build a standard sklearn ColumnTransformer.

    For categorical columns:
      1. Impute missing with a sentinel constant (avoids `most_frequent` —
         that strategy mis-coerces strings to floats in recent sklearn).
      2. Stringify so mixed bool/str/sentinel columns can be sorted.
      3. One-hot encode with `handle_unknown="ignore"` so test-time unseen
         categories don't break inference.
    """
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("stringify", FunctionTransformer(_stringify_categoricals, validate=False)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, numeric_features), ("cat", cat_pipe, categorical_features)],
        remainder="drop",
    )


def get_transformed_feature_names(preprocessor, numeric_features, categorical_features):
    """Extract feature names from a fitted ColumnTransformer."""
    names = list(numeric_features)
    if categorical_features:
        try:
            encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            names.extend(encoder.get_feature_names_out(categorical_features).tolist())
        except Exception:
            pass
    return names
