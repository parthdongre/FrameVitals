"""Optional DuckDB relation adapter for FrameVitals source-aware execution."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from framevitals.sources import DatasetMetadata


def _quoted_identifier(name: str) -> str:
    """Quote a DuckDB identifier without treating column names as expressions."""
    return '"' + str(name).replace('"', '""') + '"'


@dataclass(slots=True)
class DuckDBRelationSource:
    """Stream a lazy DuckDB relation through Arrow record batches.

    ``inspect()`` executes an exact ``count(*)`` once and caches the result.
    This can scan the underlying relation, but it does not materialize all rows
    in Python memory. Row data is consumed later through ``to_arrow_reader``.
    """

    relation: Any
    name: str = "<duckdb_relation>"
    _metadata_cache: DatasetMetadata | None = field(default=None, init=False, repr=False)
    _schema_cache: Any = field(default=None, init=False, repr=False)

    def inspect(self) -> DatasetMetadata:
        if self._metadata_cache is not None:
            return self._metadata_cache

        columns = list(self.relation.columns)
        count_row = self.relation.aggregate(
            "count(*) AS __framevitals_rows"
        ).fetchone()
        if not count_row:
            raise ValueError("DuckDB relation did not return a row count.")

        rows = int(count_row[0])
        schema = self.schema()
        self._metadata_cache = DatasetMetadata(
            name=self.name,
            kind="relation",
            format="duckdb",
            rows=rows,
            columns=len(columns),
            size_bytes=None,
            materialized=False,
            supports_projection=True,
            supports_streaming=True,
        )
        if len(schema) != len(columns):
            raise ValueError(
                "DuckDB relation schema changed while FrameVitals inspected it."
            )
        return self._metadata_cache

    def schema(self):
        """Return an Arrow schema without fetching relation rows."""
        if self._schema_cache is None:
            self._schema_cache = self.relation.limit(0).to_arrow_table().schema
        return self._schema_cache

    def iter_batches(
        self,
        *,
        batch_size: int = 65_536,
        columns: Sequence[str] | None = None,
    ) -> Iterator[Any]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        projected = self.relation
        if columns is not None:
            available = set(self.relation.columns)
            missing = [column for column in columns if column not in available]
            if missing:
                raise KeyError(
                    "DuckDB relation does not contain requested column(s): "
                    + ", ".join(map(str, missing))
                )
            projected = self.relation.select(
                *[_quoted_identifier(column) for column in columns]
            )

        reader = projected.to_arrow_reader(int(batch_size))
        yield from reader

    def load(self) -> pd.DataFrame:
        """Materialize the complete relation only for exact/full-row APIs."""
        dataframe = self.relation.df()
        if dataframe.empty:
            raise ValueError("Dataset is empty: <duckdb_relation>")
        return dataframe


def resolve_duckdb_source(data: Any) -> DuckDBRelationSource | None:
    """Recognize DuckDB relations without importing DuckDB for normal inputs."""
    data_type = type(data)
    if data_type.__name__ != "DuckDBPyRelation":
        return None
    if not data_type.__module__.startswith(("duckdb", "_duckdb")):
        return None

    try:
        import duckdb
    except ImportError:
        return None

    relation_type = getattr(duckdb, "DuckDBPyRelation", None)
    if relation_type is None or not isinstance(data, relation_type):
        return None
    return DuckDBRelationSource(data)
