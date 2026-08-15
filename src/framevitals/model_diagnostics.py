import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from framevitals.ml_preprocessing import (
    build_sklearn_preprocessor,
    prepare_ml_matrix,
)


def _prepare_diagnostics_data(
    df,
    target_column,
):
    prep = prepare_ml_matrix(
        df,
        target=target_column,
    )

    if (
        not prep["usable"]
        or len(prep["y"]) < 30
    ):
        return None

    return prep


def _dropped_column_names(prep):
    return [
        item["column"]
        for item in prep["dropped_columns"]
    ]


def run_regression_diagnostics(
    df,
    target_column,
):
    prep = _prepare_diagnostics_data(
        df,
        target_column,
    )

    if prep is None:
        return {
            "available": False,
            "message": (
                "Not enough usable rows "
                "for regression diagnostics."
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

    pipeline = Pipeline([
        (
            "preprocessor",
            build_sklearn_preprocessor(
                numeric_features,
                categorical_features,
            ),
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=120,
                random_state=42,
                max_depth=8,
            ),
        ),
    ])

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
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(
            X_test
        )

        actual = np.asarray(
            y_test,
            dtype=float,
        )

        predicted = np.asarray(
            predictions,
            dtype=float,
        )

        residuals = actual - predicted
        absolute_errors = np.abs(
            residuals
        )

        residual_summary = {
            "mean_residual": round(
                float(np.mean(residuals)),
                6,
            ),
            "median_residual": round(
                float(np.median(residuals)),
                6,
            ),
            "std_residual": round(
                float(np.std(residuals)),
                6,
            ),
            "min_residual": round(
                float(np.min(residuals)),
                6,
            ),
            "max_residual": round(
                float(np.max(residuals)),
                6,
            ),
            "mean_abs_residual": round(
                float(
                    np.mean(
                        absolute_errors
                    )
                ),
                6,
            ),
        }

        count = min(
            10,
            len(absolute_errors),
        )

        worst_indices = np.argsort(
            absolute_errors
        )[-count:][::-1]

        worst_predictions = [
            {
                "actual": round(
                    float(actual[index]),
                    6,
                ),
                "predicted": round(
                    float(predicted[index]),
                    6,
                ),
                "absolute_error": round(
                    float(
                        absolute_errors[
                            index
                        ]
                    ),
                    6,
                ),
            }
            for index in worst_indices
        ]

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring="r2",
        )

        mean_cv = float(
            np.mean(cv_scores)
        )

        warnings = list(
            prep["warnings"]
        )

        if mean_cv > 0.98:
            warnings.append(
                "Cross-validation score is "
                "extremely high. Check target "
                "leakage or redundant "
                "target-adjacent features."
            )

        if (
            residual_summary[
                "std_residual"
            ]
            == 0
        ):
            warnings.append(
                "Residual standard deviation "
                "is zero, which is suspicious "
                "for real-world regression."
            )

        return {
            "available": True,
            "task_type": "regression",
            "target_column": (
                target_column
            ),
            "residual_summary": (
                residual_summary
            ),
            "worst_predictions": (
                worst_predictions
            ),
            "cross_validation": {
                "cv": 5,
                "r2_scores": [
                    round(
                        float(score),
                        4,
                    )
                    for score in cv_scores
                ],
                "mean_r2": round(
                    mean_cv,
                    4,
                ),
                "std_r2": round(
                    float(
                        np.std(
                            cv_scores
                        )
                    ),
                    4,
                ),
            },
            "dropped_columns": (
                _dropped_column_names(
                    prep
                )
            ),
            "dropped_column_details": (
                prep["dropped_columns"]
            ),
            "warnings": warnings,
        }

    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
        }


def run_classification_diagnostics(
    df,
    target_column,
):
    prep = _prepare_diagnostics_data(
        df,
        target_column,
    )

    if prep is None:
        return {
            "available": False,
            "message": (
                "Not enough usable rows "
                "for classification diagnostics."
            ),
        }

    X = prep["X"]
    y = prep["y"]

    if y.nunique(dropna=True) < 2:
        return {
            "available": False,
            "message": (
                "Classification target must "
                "contain at least two classes."
            ),
        }

    numeric_features = prep[
        "numeric_features"
    ]
    categorical_features = prep[
        "categorical_features"
    ]

    pipeline = Pipeline([
        (
            "preprocessor",
            build_sklearn_preprocessor(
                numeric_features,
                categorical_features,
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=120,
                random_state=42,
                max_depth=8,
                class_weight="balanced",
            ),
        ),
    ])

    try:
        min_class_count = int(
            y.value_counts().min()
        )

        stratify = (
            y
            if min_class_count >= 2
            else None
        )

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

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(
            X_test
        )

        y_test_string = (
            pd.Series(y_test)
            .astype(str)
            .reset_index(drop=True)
        )

        prediction_string = (
            pd.Series(predictions)
            .astype(str)
            .reset_index(drop=True)
        )

        labels = sorted(
            set(
                y_test_string.tolist()
            )
            | set(
                prediction_string.tolist()
            )
        )

        report = classification_report(
            y_test_string,
            prediction_string,
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test_string,
            prediction_string,
            labels=labels,
        )

        accuracy = accuracy_score(
            y_test_string,
            prediction_string,
        )

        precision = precision_score(
            y_test_string,
            prediction_string,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test_string,
            prediction_string,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test_string,
            prediction_string,
            average="weighted",
            zero_division=0,
        )

        cv_folds = min(
            5,
            min_class_count,
        )

        if cv_folds >= 2:
            cv_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv_folds,
                scoring="f1_weighted",
            )
        else:
            cv_scores = np.array([])

        warnings = list(
            prep["warnings"]
        )

        if (
            len(cv_scores)
            and np.mean(cv_scores) > 0.98
        ):
            warnings.append(
                "Cross-validation score is "
                "extremely high. Check target "
                "leakage or duplicate-like "
                "target features."
            )

        cross_validation = {
            "cv": int(cv_folds),
            "f1_weighted_scores": [
                round(
                    float(score),
                    4,
                )
                for score in cv_scores
            ],
            "mean_f1_weighted": (
                round(
                    float(
                        np.mean(
                            cv_scores
                        )
                    ),
                    4,
                )
                if len(cv_scores)
                else None
            ),
            "std_f1_weighted": (
                round(
                    float(
                        np.std(
                            cv_scores
                        )
                    ),
                    4,
                )
                if len(cv_scores)
                else None
            ),
        }

        return {
            "available": True,
            "task_type": (
                "classification"
            ),
            "target_column": (
                target_column
            ),

            # Useful directly to the frontend.
            "accuracy": round(
                float(accuracy),
                4,
            ),
            "precision_weighted": round(
                float(precision),
                4,
            ),
            "recall_weighted": round(
                float(recall),
                4,
            ),
            "f1_weighted": round(
                float(f1),
                4,
            ),

            # Detailed diagnostics retained.
            "labels": labels,
            "classification_report": (
                report
            ),
            "confusion_matrix": (
                matrix.tolist()
            ),
            "cross_validation": (
                cross_validation
            ),
            "dropped_columns": (
                _dropped_column_names(
                    prep
                )
            ),
            "dropped_column_details": (
                prep["dropped_columns"]
            ),
            "warnings": warnings,
        }

    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
        }


def run_model_diagnostics(
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

    if task_type == "regression":
        return run_regression_diagnostics(
            df,
            target_column,
        )

    if task_type == "classification":
        return (
            run_classification_diagnostics(
                df,
                target_column,
            )
        )

    return {
        "available": False,
        "message": (
            f"Unknown task type: {task_type}"
        ),
    }
