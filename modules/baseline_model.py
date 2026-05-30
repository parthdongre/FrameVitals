import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor


def prepare_xy(df, target_column):
    data = df.copy()
    data = data.dropna(subset=[target_column])

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


def classification_metrics(y_test, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
    }


def regression_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    return {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
    }


def run_baseline_model(df, target_column, task_type):
    if not target_column or target_column not in df.columns:
        return {
            "available": False,
            "message": "No valid target column selected.",
        }

    X, y, numeric_features, categorical_features, dropped_columns = prepare_xy(df, target_column)

    if X.empty or len(y) < 30:
        return {
            "available": False,
            "message": "Not enough usable rows for baseline modelling.",
        }

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    stratify = None
    if task_type == "classification" and y.nunique() > 1:
        min_class_count = y.value_counts().min()
        if min_class_count >= 2:
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )

        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=120,
                random_state=42,
                max_depth=8,
                class_weight="balanced",
            )
            dummy = DummyClassifier(strategy="most_frequent")

        else:
            model = RandomForestRegressor(
                n_estimators=120,
                random_state=42,
                max_depth=8,
            )
            dummy = DummyRegressor(strategy="mean")

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        dummy_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", dummy),
            ]
        )

        model_pipeline.fit(X_train, y_train)
        dummy_pipeline.fit(X_train, y_train)

        model_pred = model_pipeline.predict(X_test)
        dummy_pred = dummy_pipeline.predict(X_test)

        if task_type == "classification":
            model_metrics = classification_metrics(y_test, model_pred)
            dummy_metrics = classification_metrics(y_test, dummy_pred)
            primary_metric = "f1_weighted"
        else:
            model_metrics = regression_metrics(y_test, model_pred)
            dummy_metrics = regression_metrics(y_test, dummy_pred)
            primary_metric = "r2"

        return {
            "available": True,
            "target_column": target_column,
            "task_type": task_type,
            "model": "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor",
            "baseline_model": "DummyClassifier" if task_type == "classification" else "DummyRegressor",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "model_metrics": model_metrics,
            "dummy_metrics": dummy_metrics,
            "primary_metric": primary_metric,
            "dropped_columns": dropped_columns,
            "used_numeric_features": numeric_features,
            "used_categorical_features": categorical_features,
            "warning": "This is a baseline model, not a final optimized model.",
        }

    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
        }
