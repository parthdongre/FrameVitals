"""Example third-party checks discovered through FrameVitals entry points."""

from __future__ import annotations

import framevitals as fv


@fv.check(
    "positive revenue",
    severity="error",
    description="Revenue cannot be negative.",
)
def positive_revenue(dataframe):
    if "revenue" not in dataframe.columns:
        return {
            "passed": False,
            "message": "Required revenue column is missing.",
            "details": {"required_column": "revenue"},
        }

    minimum = float(dataframe["revenue"].min())
    return {
        "passed": minimum >= 0,
        "message": (
            "Revenue is non-negative."
            if minimum >= 0
            else "Negative revenue records were found."
        ),
        "details": {"minimum": minimum},
    }


@fv.check(
    "preferred plan domain",
    severity="warning",
    description="Plans should use the preferred product vocabulary.",
)
def preferred_plan(dataframe):
    if "plan" not in dataframe.columns:
        return {
            "passed": False,
            "message": "Plan column is missing.",
            "details": {"required_column": "plan"},
        }

    allowed = {"basic", "pro", "enterprise"}
    observed = set(dataframe["plan"].dropna().astype(str))
    unexpected = sorted(observed - allowed)
    return {
        "passed": not unexpected,
        "message": (
            "Plan values use the preferred domain."
            if not unexpected
            else "Unexpected plan labels were found."
        ),
        "details": {"unexpected": unexpected},
    }
