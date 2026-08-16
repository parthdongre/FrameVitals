"""Dataset source abstractions for FrameVitals.

Sources expose cheap metadata before analysis. Streaming-capable sources can
additionally yield bounded record batches, allowing focused operations to avoid
materializing the complete dataset in pandas.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
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
class DelimitedTextSource(FileSource):
    """CSV/TSV file source with optional Arrow streaming acceleration.

    This remains a :class:`FileSource` for compatibility. Without the Arrow
    extra it advertises the same non-streaming behaviour as a normal file
    source and falls back to the existing pandas loader. With Arrow it performs
    one bounded-memory metadata scan to obtain an exact row count, caches that
    metadata, and then yields projected Arrow record batches.
    """

    delimiter: str = ","
    _metadata_cache: DatasetMetadata | None = field(default=None, init=False, repr=False)
    _schema_cache: Any = field(default=None, init=False, repr=False)
    _arrow_compatible: bool | None = field(default=None, init=False, repr=False)

    @property
    def format(self) -> str:
        return "tsv" if self.delimiter == "\t" else "csv"

    def _pyarrow_csv(self):
        try:
            import pyarrow.csv as pacsv
        except ImportError:
            return None
        return pacsv

    def _arrow_reader(self, *, columns: Sequence[str] | None = None):
        pacsv = self._pyarrow_csv()
        if pacsv is None:
            raise ImportError(
                "CSV/TSV streaming requires the optional Arrow capability. "
                'Install it with: pip install "framevitals[arrow]"'
            )

        read_options = pacsv.ReadOptions(use_threads=True)
        parse_options = pacsv.ParseOptions(delimiter=self.delimiter)
        convert_options = pacsv.ConvertOptions(
            include_columns=list(columns) if columns is not None else None,
        )
        return pacsv.open_csv(
            str(self.path),
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )

    def inspect(self) -> DatasetMetadata:
        _validate_file_path(self.path)
        if self._metadata_cache is not None:
            return self._metadata_cache

        if self._pyarrow_csv() is None:
            self._arrow_compatible = False
            # Avoid zero-argument super() here: @dataclass(slots=True) may
            # replace the class object, which breaks the implicit __class__
            # cell on Python versions where this fallback path is exercised.
            self._metadata_cache = FileSource.inspect(self)
            return self._metadata_cache

        try:
            reader = self._arrow_reader()
            schema = reader.schema
            rows = 0
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                rows += int(batch.num_rows)
        except Exception:
            # Preserve existing CSV/TSV compatibility if Arrow cannot parse a
            # file that the pandas loader may still understand.
            self._arrow_compatible = False
            self._metadata_cache = FileSource.inspect(self)
            return self._metadata_cache

        self._arrow_compatible = True
        self._schema_cache = schema
        self._metadata_cache = DatasetMetadata(
            name=self.path.name,
            kind="file",
            format=self.format,
            rows=int(rows),
            columns=int(len(schema)),
            size_bytes=int(self.path.stat().st_size),
            materialized=False,
            supports_projection=True,
            supports_streaming=True,
        )
        return self._metadata_cache

    def schema(self):
        """Return the inferred Arrow schema without materializing row data."""
        metadata = self.inspect()
        if not metadata.supports_streaming:
            raise TypeError(
                f"{self.format.upper()} source is not Arrow-streamable: {self.path}"
            )
        if self._schema_cache is None:
            self._schema_cache = self._arrow_reader().schema
        return self._schema_cache

    def iter_batches(
        self,
        *,
        batch_size: int = 65_536,
        columns: Sequence[str] | None = None,
    ) -> Iterator[Any]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        metadata = self.inspect()
        if not metadata.supports_streaming:
            raise TypeError(
                f"{self.format.upper()} source cannot stream without a compatible Arrow reader."
            )

        reader = self._arrow_reader(columns=columns)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            for offset in range(0, int(batch.num_rows), int(batch_size)):
                yield batch.slice(offset, min(int(batch_size), int(batch.num_rows) - offset))

    def load(self) -> pd.DataFrame:
        return FileSource.load(self)


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
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return ParquetSource(path)
        if suffix == ".csv":
            return DelimitedTextSource(path, delimiter=",")
        if suffix == ".tsv":
            return DelimitedTextSource(path, delimiter="\t")
        return FileSource(path)
    if isinstance(data, DatasetSource):
        return data
    raise TypeError("data must be a pandas DataFrame, dataset path, or DatasetSource.")
