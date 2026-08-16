"""Dataset source abstraction for FrameVitals.

Today the execution pipeline still materializes supported files through pandas.
This module establishes the source contract needed for future Arrow/DataFusion
streaming without changing the public ``fv.analyze(data)`` API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from framevitals.loader import load_dataset


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Cheap source metadata available before deep analysis."""

    name: str
    kind: str
    format: str
    rows: int | None
    columns: int | None
    size_bytes: int | None
    materialized: bool
    supports_projection: bool
    supports_streaming: bool

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class DatasetSource(Protocol):
    """Minimal contract implemented by FrameVitals data sources."""

    def inspect(self) -> DatasetMetadata: ...

    def load(self) -> pd.DataFrame: ...


@dataclass(slots=True)
class PandasSource:
    dataframe: pd.DataFrame
    name: str = "<dataframe>"

    def inspect(self) -> DatasetMetadata:
        rows, columns = self.dataframe.shape
        size_bytes = int(self.dataframe.memory_usage(index=True, deep=True).sum())
        return DatasetMetadata(
            name=self.name,
            kind="memory",
            format="pandas",
            rows=int(rows),
            columns=int(columns),
            size_bytes=size_bytes,
            materialized=True,
            supports_projection=True,
            supports_streaming=False,
        )

    def load(self) -> pd.DataFrame:
        if self.dataframe.empty:
            raise ValueError("Dataset DataFrame is empty.")
        return self.dataframe.copy()


@dataclass(slots=True)
class FileSource:
    path: Path

    def inspect(self) -> DatasetMetadata:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Expected a dataset file, got: {self.path}")

        suffix = self.path.suffix.lower().lstrip(".") or "unknown"
        # The current loader materializes every supported file. These capability
        # flags describe what FrameVitals can do *today*, not what the format
        # could theoretically support. Arrow/Parquet sources will override them.
        return DatasetMetadata(
            name=self.path.name,
            kind="file",
            format=suffix,
            rows=None,
            columns=None,
            size_bytes=int(self.path.stat().st_size),
            materialized=False,
            supports_projection=False,
            supports_streaming=False,
        )

    def load(self) -> pd.DataFrame:
        dataframe = load_dataset(self.path)
        if dataframe.empty:
            raise ValueError(f"Dataset is empty: {self.path}")
        return dataframe


def resolve_source(data: str | Path | pd.DataFrame | DatasetSource) -> DatasetSource:
    """Normalize supported user inputs into a DatasetSource implementation."""
    if isinstance(data, pd.DataFrame):
        return PandasSource(data)
    if isinstance(data, (str, Path)):
        return FileSource(Path(data))
    if isinstance(data, DatasetSource):
        return data
    raise TypeError("data must be a pandas DataFrame, dataset path, or DatasetSource.")
