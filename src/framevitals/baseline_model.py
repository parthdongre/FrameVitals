import numpy as np

from sklearn.dummy import (
    DummyClassifier,
    DummyRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from framevitals.ml_preprocessing import (
    build_sklearn_preprocessor,
    prepare_ml_matrix,
)


def classification_metrics(y_test, y_pred):
    return {
        "accuracy": round(
            float(
                accuracy_score(
                    y_test,
                    y_pred,
                )
            ),
            4,
        ),
        "precision_weighted": round(
            float(
                precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall_weighted": round(
            float(
                recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1_weighted": round(
            float(
                f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
    }


def regression_metrics(y_test, y_pred):
    mse = mean_squared_error(
        y_test,
        y_pred,
    )

    return {
        "r2": round(
            float(
                r2_score(
                    y_test,
                    y_pred,
                )
            ),
            4,
        ),
        "mae": round(
            float(
                mean_absolute_error(
                    y_test,
                    y_pred,
                )
            ),
            4,
        ),
        "mse": round(
            float(mse),
            4,
        ),
        "rmse": round(
            float(np.sqrt(mse)),
            4,
        ),
    }


def run_baseline_model(
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

    prep = prepare_ml_matrix(
        df,
        target=target_column,
    )

    if not prep["usable"] or len(prep["y"]) < 30:
        return {
            "available": False,
            "message": (
                "Not enough usable rows "
                "for baseline modelling."
            ),
        }

    X = prep["X"]
    y = prep["y"]

    numeric_features = prep[
        "numeric_features"
    ]
    categorical_features = prep[
        "categorical_features"
    ]

    dropped_columns = [
        item["column"]
        for item in prep["dropped_columns"]
    ]

    preprocessor = (
        build_sklearn_preprocessor(
            numeric_features,
            categorical_features,
        )
    )

    stratify = None

    if (
        task_type == "classification"
        and y.nunique() > 1
    ):
        min_class_count = (
            y.value_counts().min()
        )

        if min_class_count >= 2:
            stratify = y

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
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

            dummy = DummyClassifier(
                strategy="most_frequent",
            )

            primary_metric = "f1_weighted"

        elif task_type == "regression":
            model = RandomForestRegressor(
                n_estimators=120,
                random_state=42,
                max_depth=8,
            )

            dummy = DummyRegressor(
                strategy="mean",
            )

            primary_metric = "r2"

        else:
            return {
                "available": False,
                "message": (
                    f"Unsupported task type: "
                    f"{task_type}"
                ),
            }

        model_pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ])

        dummy_pipeline = Pipeline([
            (
                "preprocessor",
                build_sklearn_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                dummy,
            ),
        ])

        model_pipeline.fit(
            X_train,
            y_train,
        )

        dummy_pipeline.fit(
            X_train,
            y_train,
        )

        model_pred = model_pipeline.predict(
            X_test
        )

        dummy_pred = dummy_pipeline.predict(
            X_test
        )

        if task_type == "classification":
            model_metrics = (
                classification_metrics(
                    y_test,
                    model_pred,
                )
            )

            dummy_metrics = (
                classification_metrics(
                    y_test,
                    dummy_pred,
                )
            )

            model_name = (
                "RandomForestClassifier"
            )
            baseline_name = (
                "DummyClassifier"
            )

        else:
            model_metrics = (
                regression_metrics(
                    y_test,
                    model_pred,
                )
            )

            dummy_metrics = (
                regression_metrics(
                    y_test,
                    dummy_pred,
                )
            )

            model_name = (
                "RandomForestRegressor"
            )
            baseline_name = (
                "DummyRegressor"
            )

        return {
            "available": True,
            "target_column": target_column,
            "task_type": task_type,
            "model": model_name,
            "baseline_model": baseline_name,
            "train_rows": int(
                len(X_train)
            ),
            "test_rows": int(
                len(X_test)
            ),
            "model_metrics": model_metrics,
            "dummy_metrics": dummy_metrics,
            "primary_metric": primary_metric,
            "dropped_columns": (
                dropped_columns
            ),
            "used_numeric_features": (
                numeric_features
            ),
            "used_categorical_features": (
                categorical_features
            ),
            "warnings": prep["warnings"],
            "warning": (
                "This is a baseline model, "
                "not a final optimized model."
            ),
        }

    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
        }
