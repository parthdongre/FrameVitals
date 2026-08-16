"""Dataset source abstractions for FrameVitals.

Sources expose cheap metadata before analysis. Streaming-capable sources can
additionally yield bounded record batches, allowing focused operations to avoid
materializing the complete dataset in pandas.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class StreamingDatasetSource(DatasetSource, Protocol):
    """Optional extension for sources that can yield bounded record batches."""

    def iter_batches(
        self,
        *,
        batch_size: int = 65_536,
        columns: Sequence[str] | None = None,
    ) -> Iterator[Any]: ...


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
        _validate_file_path(self.path)
        suffix = self.path.suffix.lower().lstrip(".") or "unknown"
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


@dataclass(slots=True)
class ParquetSource:
    """Projection-aware, streaming Parquet source backed by optional PyArrow."""

    path: Path

    def _parquet_file(self):
        _validate_file_path(self.path)
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Parquet streaming requires the optional Arrow capability. "
                'Install it with: pip install "framevitals[arrow]"'
            ) from exc
        return pq.ParquetFile(self.path)

    def inspect(self) -> DatasetMetadata:
        parquet_file = self._parquet_file()
        metadata = parquet_file.metadata
        return DatasetMetadata(
            name=self.path.name,
            kind="file",
            format="parquet",
            rows=int(metadata.num_rows),
            columns=int(metadata.num_columns),
            size_bytes=int(self.path.stat().st_size),
            materialized=False,
            supports_projection=True,
            supports_streaming=True,
        )

    def schema(self):
        """Return the Arrow schema without reading all row data."""
        return self._parquet_file().schema_arrow

    def iter_batches(
        self,
        *,
        batch_size: int = 65_536,
        columns: Sequence[str] | None = None,
    ) -> Iterator[Any]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        parquet_file = self._parquet_file()
        yield from parquet_file.iter_batches(
            batch_size=int(batch_size),
            columns=list(columns) if columns is not None else None,
            use_threads=True,
        )

    def load(self) -> pd.DataFrame:
        parquet_file = self._parquet_file()
        table = parquet_file.read(use_threads=True)
        dataframe = table.to_pandas()
        if dataframe.empty:
            raise ValueError(f"Dataset is empty: {self.path}")
        return dataframe


def _validate_file_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a dataset file, got: {path}")


def resolve_source(data: str | Path | pd.DataFrame | DatasetSource) -> DatasetSource:
    """Normalize supported user inputs into a DatasetSource implementation."""
    if isinstance(data, pd.DataFrame):
        return PandasSource(data)
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix.lower() == ".parquet":
            return ParquetSource(path)
        return FileSource(path)
    if isinstance(data, DatasetSource):
        return data
    raise TypeError("data must be a pandas DataFrame, dataset path, or DatasetSource.")
