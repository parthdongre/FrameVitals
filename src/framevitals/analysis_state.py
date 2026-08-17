"""Mergeable analysis-state primitives.

These classes are the reference semantics for FrameVitals' Rust/Arrow streaming
engine. A partition can be summarized independently and merged with another
partition without concatenating raw rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from framevitals.backends import numeric_state


@dataclass(slots=True)
class NumericColumnState:
    """Mergeable exact central-moment state through fourth order."""

    count: int = 0
    missing: int = 0
    mean: float = 0.0
    m2: float = 0.0
    m3: float = 0.0
    m4: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    infinite: int = 0

    @classmethod
    def from_series(cls, series: pd.Series) -> "NumericColumnState":
        payload = numeric_state(series)
        count = int(payload["count"])
        variance = payload.get("variance")
        return cls(
            count=count,
            missing=int(payload["missing"]),
            mean=float(payload["mean"]) if count else 0.0,
            m2=float(
                payload.get(
                    "m2",
                    float(variance) * (count - 1)
                    if variance is not None and count >= 2
                    else 0.0,
                )
            ),
            m3=float(payload.get("m3", 0.0)),
            m4=float(payload.get("m4", 0.0)),
            minimum=(
                float(payload["minimum"])
                if payload.get("minimum") is not None
                else None
            ),
            maximum=(
                float(payload["maximum"])
                if payload.get("maximum") is not None
                else None
            ),
            infinite=int(payload["infinite"]),
        )

    def merge(self, other: "NumericColumnState") -> "NumericColumnState":
        """Merge another partition through fourth central moment."""
        if other.count == 0:
            return NumericColumnState(
                count=self.count,
                missing=self.missing + other.missing,
                mean=self.mean,
                m2=self.m2,
                m3=self.m3,
                m4=self.m4,
                minimum=self.minimum,
                maximum=self.maximum,
                infinite=self.infinite + other.infinite,
            )
        if self.count == 0:
            return NumericColumnState(
                count=other.count,
                missing=self.missing + other.missing,
                mean=other.mean,
                m2=other.m2,
                m3=other.m3,
                m4=other.m4,
                minimum=other.minimum,
                maximum=other.maximum,
                infinite=self.infinite + other.infinite,
            )

        total = self.count + other.count
        left = float(self.count)
        right = float(other.count)
        total_f = float(total)
        delta = other.mean - self.mean
        delta2 = delta * delta
        delta3 = delta2 * delta
        delta4 = delta2 * delta2
        total2 = total_f * total_f
        total3 = total2 * total_f

        mean = self.mean + delta * right / total_f
        m2 = self.m2 + other.m2 + delta2 * left * right / total_f
        m3 = (
            self.m3
            + other.m3
            + delta3 * left * right * (left - right) / total2
            + 3.0 * delta * (left * other.m2 - right * self.m2) / total_f
        )
        m4 = (
            self.m4
            + other.m4
            + delta4
            * left
            * right
            * (left * left - left * right + right * right)
            / total3
            + 6.0
            * delta2
            * (left * left * other.m2 + right * right * self.m2)
            / total2
            + 4.0 * delta * (left * other.m3 - right * self.m3) / total_f
        )
        minimum = min(
            value for value in (self.minimum, other.minimum) if value is not None
        )
        maximum = max(
            value for value in (self.maximum, other.maximum) if value is not None
        )
        return NumericColumnState(
            count=total,
            missing=self.missing + other.missing,
            mean=float(mean),
            m2=float(m2),
            m3=float(m3),
            m4=float(m4),
            minimum=float(minimum),
            maximum=float(maximum),
            infinite=self.infinite + other.infinite,
        )

    @property
    def variance(self) -> float | None:
        if self.count < 2:
            return None
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float | None:
        variance = self.variance
        return sqrt(variance) if variance is not None and variance >= 0 else None

    @property
    def skewness(self) -> float | None:
        """Bias-corrected Fisher-Pearson sample skewness."""
        if self.count < 3 or self.m2 <= 0:
            return None
        n = float(self.count)
        population_skew = sqrt(n) * self.m3 / (self.m2 ** 1.5)
        return sqrt(n * (n - 1.0)) / (n - 2.0) * population_skew

    @property
    def kurtosis(self) -> float | None:
        """Bias-corrected Fisher excess kurtosis."""
        if self.count < 4 or self.m2 <= 0:
            return None
        n = float(self.count)
        population_excess = n * self.m4 / (self.m2 * self.m2) - 3.0
        return (
            (n - 1.0)
            / ((n - 2.0) * (n - 3.0))
            * ((n + 1.0) * population_excess + 6.0)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "missing": self.missing,
            "infinite": self.infinite,
            "mean": self.mean if self.count else None,
            "variance": self.variance,
            "std": self.std,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "m2": self.m2,
            "m3": self.m3,
            "m4": self.m4,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(slots=True)
class AnalysisState:
    """Compact mergeable state for a dataset partition."""

    rows: int = 0
    columns: int = 0
    numeric: dict[str, NumericColumnState] = field(default_factory=dict)
    schema: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, dataframe: pd.DataFrame) -> "AnalysisState":
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
        return cls(
            rows=int(len(dataframe)),
            columns=int(len(dataframe.columns)),
            numeric={
                str(column): NumericColumnState.from_series(dataframe[column])
                for column in numeric_columns
            },
            schema={str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
        )

    def merge(self, other: "AnalysisState") -> "AnalysisState":
        """Merge states without access to either partition's raw rows."""
        if self.schema and other.schema and self.schema != other.schema:
            raise ValueError("Cannot merge AnalysisState objects with different schemas.")

        names = set(self.numeric) | set(other.numeric)
        merged_numeric: dict[str, NumericColumnState] = {}
        for name in names:
            left = self.numeric.get(name, NumericColumnState())
            right = other.numeric.get(name, NumericColumnState())
            merged_numeric[name] = left.merge(right)

        return AnalysisState(
            rows=self.rows + other.rows,
            columns=max(self.columns, other.columns),
            numeric=merged_numeric,
            schema=dict(self.schema or other.schema),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "schema": dict(self.schema),
            "numeric": {
                name: state.to_dict()
                for name, state in sorted(self.numeric.items())
            },
        }
