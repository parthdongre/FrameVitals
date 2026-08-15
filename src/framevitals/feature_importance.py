import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.pipeline import Pipeline

from framevitals.ml_preprocessing import (
    build_sklearn_preprocessor,
    get_transformed_feature_names,
    prepare_ml_matrix,
)


def collapse_onehot_importances(
    feature_names,
    importances,
    numeric_features,
    categorical_features,
):
    collapsed = {
        column: 0.0
        for column in (
            list(numeric_features)
            + list(categorical_features)
        )
    }

    # Longest names first prevents a column such as
    # "customer" from incorrectly matching
    # "customer_segment_A".
    categorical_by_length = sorted(
        categorical_features,
        key=len,
        reverse=True,
    )

    for name, importance in zip(
        feature_names,
        importances,
    ):
        value = float(importance)

        if name in numeric_features:
            collapsed[name] = (
                collapsed.get(name, 0.0)
                + value
            )
            continue

        matched = False

        for column in categorical_by_length:
            prefix = f"{column}_"

            if name.startswith(prefix):
                collapsed[column] = (
                    collapsed.get(column, 0.0)
                    + value
                )
                matched = True
                break

        if not matched:
            collapsed[name] = (
                collapsed.get(name, 0.0)
                + value
            )

    result = [
        {
            "feature": feature,
            "importance": round(
                importance,
                6,
            ),
        }
        for feature, importance
        in collapsed.items()
    ]

    return sorted(
        result,
        key=lambda item: item["importance"],
        reverse=True,
    )


def calculate_mutual_information(
    X,
    y,
    numeric_features,
    categorical_features,
    task_type,
):
    X_simple = X.copy()

    for column in numeric_features:
        median = X_simple[column].median()

        X_simple[column] = (
            X_simple[column].fillna(median)
        )

    for column in categorical_features:
        values = (
            X_simple[column]
            .astype("string")
            .fillna("__missing__")
        )

        X_simple[column] = (
            values.astype("category").cat.codes
        )

    if X_simple.empty:
        return []

    try:
        if task_type == "classification":
            scores = mutual_info_classif(
                X_simple,
                y,
                random_state=42,
            )

        elif task_type == "regression":
            scores = mutual_info_regression(
                X_simple,
                y,
                random_state=42,
            )

        else:
            return []

        result = [
            {
                "feature": column,
                "mutual_information": round(
                    float(score),
                    6,
                ),
            }
            for column, score
            in zip(
                X_simple.columns,
                scores,
            )
        ]

        return sorted(
            result,
            key=lambda item: item[
                "mutual_information"
            ],
            reverse=True,
        )

    except Exception as exc:
        return [
            {
                "error": str(exc),
            }
        ]


def run_feature_importance(
    df,
    target_column,
    task_type,
):
    if (
        not target_column
        or target_column not in df.columns
    ):
        return {
            "available": False,
            "message": (
                "No valid target column selected."
            ),
        }

    if task_type not in {
        "classification",
        "regression",
    }:
        return {
            "available": False,
            "message": (
                f"Unsupported task type: "
                f"{task_type}"
            ),
        }

    prep = prepare_ml_matrix(
        df,
        target=target_column,
    )

    if not prep["usable"]:
        return {
            "available": False,
            "message": (
                "Not enough usable data "
                "for feature importance."
            ),
            "warnings": prep["warnings"],
        }

    X = prep["X"]
    y = prep["y"]

    numeric_features = prep[
        "numeric_features"
    ]
    categorical_features = prep[
        "categorical_features"
    ]

    preprocessor = (
        build_sklearn_preprocessor(
            numeric_features,
            categorical_features,
        )
    )

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

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            model,
        ),
    ])

    try:
        pipeline.fit(
            X,
            y,
        )

        fitted_preprocessor = (
            pipeline.named_steps[
                "preprocessor"
            ]
        )

        fitted_model = (
            pipeline.named_steps["model"]
        )

        feature_names = (
            get_transformed_feature_names(
                fitted_preprocessor,
                numeric_features,
                categorical_features,
            )
        )

        collapsed = (
            collapse_onehot_importances(
                feature_names,
                fitted_model.feature_importances_,
                numeric_features,
                categorical_features,
            )
        )

        global_importance = collapsed[:15]

        mutual_information = (
            calculate_mutual_information(
                X,
                y,
                numeric_features,
                categorical_features,
                task_type,
            )
        )

        dropped_columns = [
            item["column"]
            for item in prep[
                "dropped_columns"
            ]
        ]

        return {
            "available": True,
            "target_column": target_column,
            "task_type": task_type,
            "method": (
                "Random Forest impurity "
                "importance + mutual information"
            ),

            # New canonical frontend/API key.
            "global_importance": (
                global_importance
            ),

            # Backward-compatible legacy key.
            "top_features": (
                global_importance
            ),

            "mutual_information": (
                mutual_information[:15]
            ),
            "dropped_columns": (
                dropped_columns
            ),
            "dropped_column_details": (
                prep["dropped_columns"]
            ),
            "used_numeric_features": (
                numeric_features
            ),
            "used_categorical_features": (
                categorical_features
            ),
            "warnings": prep["warnings"],
        }

    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
        }
