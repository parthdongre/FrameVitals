"""Unified ML preprocessing used by every FrameVitals model workflow.

The module deliberately keeps feature selection conservative: obvious IDs,
unsupported temporal columns, constants, and dangerously high-cardinality
categoricals are excluded before sklearn preprocessing. Numeric infinities are
converted to missing values so the standard median imputer can handle them.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


_CATEGORICAL_DTYPES = ["object", "string", "category", "bool"]
_IDENTIFIER_TOKENS = {"id", "uuid", "hash", "identifier", "index"}
_IDENTIFIER_EXACT = {
    "key",
    "roll",
    "roll_number",
    "rollno",
    "application",
    "registration",
    "serial",
    "ticket",
    "account",
    "userid",
    "studentid",
}
_TIME_TOKENS = {"time", "date", "timestamp", "datetime"}


def _normalise_column_name(name: object) -> tuple[str, set[str]]:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    tokens = {token for token in normalized.split("_") if token}
    return normalized, tokens


def _looks_like_identifier_name(name: object) -> bool:
    """Detect identifier-like names without unsafe substring matching.

    The old ``"id" in column_name`` approach incorrectly classified ordinary
    names such as ``paid_amount``. Token/boundary matching keeps common ID
    conventions while avoiding those false positives.
    """
    normalized, tokens = _normalise_column_name(name)
    if normalized in _IDENTIFIER_EXACT:
        return True
    if tokens & _IDENTIFIER_TOKENS:
        return True
    if normalized.endswith("_key") or normalized.endswith("_number"):
        prefix = normalized.rsplit("_", 1)[0]
        if prefix in {"account", "serial", "ticket", "roll", "registration"}:
            return True
    return False


def _looks_like_time_name(name: object) -> bool:
    normalized, tokens = _normalise_column_name(name)
    if tokens & _TIME_TOKENS:
        return True
    if normalized in {"created", "updated", "created_at", "updated_at"}:
        return True
    if normalized.endswith("_at") and tokens & {"created", "updated"}:
        return True
    return False


def _is_categorical_dtype(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    )


def prepare_ml_matrix(
    df,
    target,
    drop_high_unique_ratio=0.95,
    min_non_missing=5,
    max_categorical_levels=200,
):
    """Prepare a conservative feature matrix and target vector for ML."""
    warnings: list[str] = []

    if target not in df.columns:
        return {
            "X": pd.DataFrame(),
            "y": pd.Series(dtype="float64"),
            "numeric_features": [],
            "categorical_features": [],
            "dropped_columns": [],
            "warnings": [f"Target '{target}' not found."],
            "infinite_values_replaced": {},
            "usable": False,
        }

    if not 0 < drop_high_unique_ratio <= 1:
        raise ValueError("drop_high_unique_ratio must be in (0, 1].")
    if min_non_missing < 1:
        raise ValueError("min_non_missing must be at least 1.")
    if max_categorical_levels < 2:
        raise ValueError("max_categorical_levels must be at least 2.")

    data = df.copy()

    target_infinite_count = 0
    if pd.api.types.is_numeric_dtype(data[target]):
        target_values = pd.to_numeric(data[target], errors="coerce")
        target_array = target_values.to_numpy(dtype="float64", na_value=np.nan)
        target_infinite_count = int(np.isinf(target_array).sum())
        if target_infinite_count:
            data[target] = target_values.replace([np.inf, -np.inf], np.nan)
            warnings.append(
                f"Treated {target_infinite_count} infinite target values as missing."
            )

    data = data.dropna(subset=[target]).copy()
    rows_dropped = len(df) - len(data)
    if rows_dropped > 0:
        warnings.append(f"Dropped {rows_dropped} rows with missing/invalid target values.")

    y = data[target]
    X = data.drop(columns=[target]).copy()

    dropped: list[dict[str, str]] = []
    dropped_names: set[str] = set()
    infinite_values_replaced: dict[str, int] = {}

    def drop(column: str, reason: str) -> None:
        if column not in dropped_names:
            dropped.append({"column": column, "reason": reason})
            dropped_names.add(column)

    for column in list(X.columns):
        series = X[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            values = numeric.to_numpy(dtype="float64", na_value=np.nan)
            inf_count = int(np.isinf(values).sum())
            if inf_count:
                X[column] = numeric.replace([np.inf, -np.inf], np.nan)
                infinite_values_replaced[column] = inf_count
                warnings.append(
                    f"Replaced {inf_count} infinite values in '{column}' with missing values for imputation."
                )

    for column in list(X.columns):
        series = X[column]
        non_missing = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / max(non_missing, 1)

        if non_missing < min_non_missing:
            drop(column, "insufficient_non_missing")
            continue
        if unique_count <= 1:
            drop(column, "constant")
            continue
        if _looks_like_identifier_name(column):
            drop(column, "id_like_name")
            continue
        if _looks_like_time_name(column) and not pd.api.types.is_numeric_dtype(series):
            drop(column, "time_like_non_numeric")
            continue

        if _is_categorical_dtype(series):
            if unique_count > max_categorical_levels:
                drop(column, "high_cardinality_categorical")
                continue
            if unique_ratio > drop_high_unique_ratio:
                drop(column, "high_unique_non_numeric")
                continue

        if (
            not pd.api.types.is_numeric_dtype(series)
            and not _is_categorical_dtype(series)
        ):
            drop(column, "unsupported_dtype")

    X = X.drop(columns=list(dropped_names), errors="ignore")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        column
        for column in X.columns
        if _is_categorical_dtype(X[column])
    ]

    used_features = set(numeric_features) | set(categorical_features)
    unused_columns = [column for column in X.columns if column not in used_features]
    for column in unused_columns:
        drop(column, "unsupported_dtype")
    if unused_columns:
        X = X.drop(columns=unused_columns, errors="ignore")

    total_features = len(numeric_features) + len(categorical_features)
    usable = total_features > 0 and len(y) >= 20

    if total_features == 0:
        warnings.append("No usable features remain after preprocessing.")
    if len(y) < 20:
        warnings.append(f"Only {len(y)} rows remain — too few for reliable ML.")

    return {
        "X": X,
        "y": y,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_columns": dropped,
        "warnings": warnings,
        "infinite_values_replaced": infinite_values_replaced,
        "target_infinite_values_dropped": target_infinite_count,
        "max_categorical_levels": int(max_categorical_levels),
        "usable": usable,
    }


def _stringify_categoricals(X):
    """Force every cell in a categorical block to a string."""
    if hasattr(X, "astype"):
        return X.astype(str)
    return np.asarray(X).astype(str)


def build_sklearn_preprocessor(numeric_features, categorical_features):
    """Build the standard sklearn preprocessing graph."""
    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("stringify", FunctionTransformer(_stringify_categoricals, validate=False)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
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
