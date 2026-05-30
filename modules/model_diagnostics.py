import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def prepare_xy(df, target_column):
    data = df.dropna(subset=[target_column]).copy()

    y = data[target_column]
    X = data.drop(columns=[target_column])

    drop_columns = []

    id_like_keywords = [
        "id",
        "roll",
        "roll number",
        "number",
        "application",
        "registration",
        "serial",
        "uuid",
        "identifier",
    ]

    for column in X.columns:
        lower = column.lower()
        unique_ratio = X[column].nunique(dropna=True) / max(len(X), 1)
        non_missing_count = int(X[column].notna().sum())

        # Drop all-empty columns after target-row filtering.
        if non_missing_count == 0:
            drop_columns.append(column)
            continue

        # Drop constant columns.
        if X[column].nunique(dropna=True) <= 1:
            drop_columns.append(column)
            continue

        # Drop obvious ID-like columns.
        if any(keyword in lower for keyword in id_like_keywords):
            drop_columns.append(column)
            continue

        # Drop high-cardinality text/categorical identifiers.
        if unique_ratio > 0.95 and not pd.api.types.is_numeric_dtype(X[column]):
            drop_columns.append(column)
            continue

    X = X.drop(columns=list(set(drop_columns)), errors="ignore")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return X, y, numeric_features, categorical_features, drop_columns


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def run_regression_diagnostics(df, target_column):
    X, y, numeric_features, categorical_features, dropped = prepare_xy(df, target_column)

    if len(y) < 30:
        return {
            "available": False,
            "message": "Not enough rows for regression diagnostics.",
        }

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        max_depth=8,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        residuals = y_test - predictions

        residual_summary = {
            "mean_residual": round(float(np.mean(residuals)), 6),
            "median_residual": round(float(np.median(residuals)), 6),
            "std_residual": round(float(np.std(residuals)), 6),
            "min_residual": round(float(np.min(residuals)), 6),
            "max_residual": round(float(np.max(residuals)), 6),
        }

        abs_errors = np.abs(residuals)
        worst_indices = np.argsort(abs_errors)[-10:][::-1]

        worst_predictions = []
        y_test_values = y_test.reset_index(drop=True)

        for idx in worst_indices:
            worst_predictions.append(
                {
                    "actual": round(float(y_test_values.iloc[idx]), 6),
                    "predicted": round(float(predictions[idx]), 6),
                    "absolute_error": round(float(abs_errors.iloc[idx] if hasattr(abs_errors, "iloc") else abs_errors[idx]), 6),
                }
            )

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring="r2",
        )

        warnings = []

        if np.mean(cv_scores) > 0.98:
            warnings.append(
                "Cross-validation score is extremely high. Check target leakage or redundant target-adjacent features."
            )

        if residual_summary["std_residual"] == 0:
            warnings.append("Residual standard deviation is zero, which is suspicious for real-world regression.")

        return {
            "available": True,
            "task_type": "regression",
            "target_column": target_column,
            "residual_summary": residual_summary,
            "worst_predictions": worst_predictions,
            "cross_validation": {
                "cv": 5,
                "r2_scores": [round(float(score), 4) for score in cv_scores],
                "mean_r2": round(float(np.mean(cv_scores)), 4),
                "std_r2": round(float(np.std(cv_scores)), 4),
            },
            "warnings": warnings,
        }

    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
        }


def run_classification_diagnostics(df, target_column):
    X, y, numeric_features, categorical_features, dropped = prepare_xy(df, target_column)

    if len(y) < 30:
        return {
            "available": False,
            "message": "Not enough rows for classification diagnostics.",
        }

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        max_depth=8,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    try:
        stratify = y if y.value_counts().min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        labels = sorted([str(label) for label in pd.Series(y_test).unique().tolist()])
        report = classification_report(
            y_test.astype(str),
            pd.Series(predictions).astype(str),
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test.astype(str),
            pd.Series(predictions).astype(str),
            labels=labels,
        )

        cv_scores = cross_val_score(
            pipeline,
            X,
            y.astype(str),
            cv=5,
            scoring="f1_weighted",
        )

        warnings = []

        if np.mean(cv_scores) > 0.98:
            warnings.append(
                "Cross-validation score is extremely high. Check target leakage or duplicate-like target features."
            )

        return {
            "available": True,
            "task_type": "classification",
            "target_column": target_column,
            "labels": labels,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "cross_validation": {
                "cv": 5,
                "f1_weighted_scores": [round(float(score), 4) for score in cv_scores],
                "mean_f1_weighted": round(float(np.mean(cv_scores)), 4),
                "std_f1_weighted": round(float(np.std(cv_scores)), 4),
            },
            "warnings": warnings,
        }

    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
        }


def run_model_diagnostics(df, target_column, task_type):
    if not target_column or target_column not in df.columns:
        return {
            "available": False,
            "message": "No valid target column selected.",
        }

    if task_type == "regression":
        return run_regression_diagnostics(df, target_column)

    if task_type == "classification":
        return run_classification_diagnostics(df, target_column)

    return {
        "available": False,
        "message": "Unknown task type.",
    }
