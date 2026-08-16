"""Mergeable analysis-state primitives.

These classes are the reference semantics for FrameVitals' future Rust/Arrow
streaming engine. A partition can be summarized independently and merged with
another partition without concatenating raw rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class NumericColumnState:
    """Mergeable exact first/second-moment state for one numeric column."""

    count: int = 0
    missing: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    infinite: int = 0

    @classmethod
    def from_series(cls, series: pd.Series) -> "NumericColumnState":
        values = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype="float64",
            na_value=np.nan,
        )
        missing = int(np.isnan(values).sum())
        infinite = int(np.isinf(values).sum())
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return cls(missing=missing, infinite=infinite)

        mean = float(finite.mean())
        centered = finite - mean
        return cls(
            count=int(finite.size),
            missing=missing,
            mean=mean,
            m2=float(np.dot(centered, centered)),
            minimum=float(finite.min()),
            maximum=float(finite.max()),
            infinite=infinite,
        )

    def merge(self, other: "NumericColumnState") -> "NumericColumnState":
        """Merge another partition using the parallel-variance formula."""
        if other.count == 0:
            return NumericColumnState(
                count=self.count,
                missing=self.missing + other.missing,
                mean=self.mean,
                m2=self.m2,
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
                minimum=other.minimum,
                maximum=other.maximum,
                infinite=self.infinite + other.infinite,
            )

        total = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / total
        m2 = (
            self.m2
            + other.m2
            + delta * delta * self.count * other.count / total
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "missing": self.missing,
            "infinite": self.infinite,
            "mean": self.mean if self.count else None,
            "variance": self.variance,
            "std": self.std,
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
