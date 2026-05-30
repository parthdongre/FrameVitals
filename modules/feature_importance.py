import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def prepare_xy(df, target_column):
    data = df.copy()
    data = data.dropna(subset=[target_column])

    y = data[target_column]
    X = data.drop(columns=[target_column])

    # Drop columns that are too hard/risky for first-pass baseline.
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
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor, numeric_features, categorical_features):
    names = []

    names.extend(numeric_features)

    if categorical_features:
        encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = encoder.get_feature_names_out(categorical_features).tolist()
        names.extend(cat_names)

    return names


def collapse_onehot_importances(feature_names, importances):
    collapsed = {}

    for name, importance in zip(feature_names, importances):
        base = name.split("_")[0] if "_" in name else name
        collapsed[base] = collapsed.get(base, 0) + float(importance)

    result = [
        {
            "feature": feature,
            "importance": round(value, 6),
        }
        for feature, value in collapsed.items()
    ]

    return sorted(result, key=lambda item: item["importance"], reverse=True)


def calculate_mutual_information(X, y, numeric_features, categorical_features, task_type):
    X_simple = X.copy()

    for column in numeric_features:
        X_simple[column] = X_simple[column].fillna(X_simple[column].median())

    for column in categorical_features:
        X_simple[column] = X_simple[column].astype("category").cat.codes

    if X_simple.empty:
        return []

    try:
        if task_type == "classification":
            scores = mutual_info_classif(X_simple, y, random_state=42)
        else:
            scores = mutual_info_regression(X_simple, y, random_state=42)

        result = [
            {
                "feature": column,
                "mutual_information": round(float(score), 6),
            }
            for column, score in zip(X_simple.columns, scores)
        ]

        return sorted(result, key=lambda item: item["mutual_information"], reverse=True)

    except Exception as exc:
        return [{"error": str(exc)}]


def run_feature_importance(df, target_column, task_type):
    if not target_column or target_column not in df.columns:
        return {
            "available": False,
            "message": "No valid target column selected.",
        }

    X, y, numeric_features, categorical_features, dropped_columns = prepare_xy(df, target_column)

    if X.empty or len(y) < 20:
        return {
            "available": False,
            "message": "Not enough usable data for feature importance.",
        }

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    if task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            class_weight="balanced",
            max_depth=8,
        )
    else:
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
        pipeline.fit(X, y)

        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        fitted_model = pipeline.named_steps["model"]

        feature_names = get_feature_names(fitted_preprocessor, numeric_features, categorical_features)
        raw_importances = fitted_model.feature_importances_

        collapsed = collapse_onehot_importances(feature_names, raw_importances)
        mutual_info = calculate_mutual_information(X, y, numeric_features, categorical_features, task_type)

        return {
            "available": True,
            "target_column": target_column,
            "task_type": task_type,
            "method": "Random Forest impurity importance + mutual information",
            "top_features": collapsed[:15],
            "mutual_information": mutual_info[:15],
            "dropped_columns": dropped_columns,
            "used_numeric_features": numeric_features,
            "used_categorical_features": categorical_features,
        }

    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
        }
