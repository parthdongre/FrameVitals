"""Per-run reusable execution context for FrameVitals analyses.

The context is intentionally backend-agnostic and stores references only for the
lifetime of one analysis run. It provides a thread-safe cache for intermediates,
named reusable samples, and structured metadata that can be surfaced without
serializing raw dataset values.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Mapping

from framevitals.config import AnalysisConfig
from framevitals.execution import ExecutionPolicy


CONTEXT_SCHEMA_VERSION = "1"
DEFAULT_CONTEXT_SEED = 0x9E3779B97F4A7C15
_MISSING = object()


@dataclass(slots=True)
class AnalysisContext:
    """Mutable per-run state shared by planning and execution stages.

    The object itself is not part of the public result schema. ``metadata()`` is
    safe to expose because it reports names/counts/provenance only and never raw
    cached values or sample contents.
    """

    dataset_name: str
    source: Mapping[str, Any]
    config: AnalysisConfig
    execution_policy: ExecutionPolicy
    rows: int
    columns: int
    seed: int = DEFAULT_CONTEXT_SEED
    _facts: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _samples: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _sample_metadata: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _cache_hits: int = field(default=0, init=False, repr=False)
    _cache_misses: int = field(default=0, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dataset_name = str(self.dataset_name or "<unknown>")
        self.source = dict(self.source)
        self.rows = int(self.rows)
        self.columns = int(self.columns)
        self.seed = int(self.seed)
        if self.rows < 0 or self.columns < 0:
            raise ValueError("AnalysisContext rows and columns must be non-negative.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.columns

    @property
    def fact_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._facts))

    @property
    def sample_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._samples))

    def set_fact(self, name: str, value: Any, *, overwrite: bool = False) -> Any:
        """Store an authoritative run fact and return ``value`` for fluent use."""
        key = str(name).strip()
        if not key:
            raise ValueError("fact name must not be empty")
        with self._lock:
            if key in self._facts and not overwrite:
                raise KeyError(f"AnalysisContext fact already exists: {key}")
            self._facts[key] = value
        return value

    def fact(self, name: str, default: Any = None) -> Any:
        """Return a stored fact without copying its potentially large value."""
        with self._lock:
            return self._facts.get(name, default)

    def require_fact(self, name: str) -> Any:
        """Return a stored fact or fail with a descriptive error."""
        with self._lock:
            value = self._facts.get(name, _MISSING)
        if value is _MISSING:
            raise KeyError(f"AnalysisContext fact is not available: {name}")
        return value

    def get_or_compute(self, key: str, factory: Callable[[], Any]) -> Any:
        """Return a cached intermediate, computing it at most once per context.

        The factory executes while the re-entrant context lock is held. That makes
        duplicate work impossible when independent scheduler threads request the
        same reusable intermediate concurrently. Context factories should remain
        local computations and should not wait on other analysis threads.
        """
        cache_key = str(key).strip()
        if not cache_key:
            raise ValueError("cache key must not be empty")
        if not callable(factory):
            raise TypeError("factory must be callable")

        with self._lock:
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
            value = factory()
            self._cache[cache_key] = value
            self._cache_misses += 1
            return value

    def cache_value(self, key: str, value: Any, *, overwrite: bool = False) -> Any:
        """Insert a known intermediate without invoking a factory."""
        cache_key = str(key).strip()
        if not cache_key:
            raise ValueError("cache key must not be empty")
        with self._lock:
            if cache_key in self._cache and not overwrite:
                raise KeyError(f"AnalysisContext cache entry already exists: {cache_key}")
            self._cache[cache_key] = value
        return value

    def store_sample(
        self,
        name: str,
        sample: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Any:
        """Retain a named bounded sample for reuse without serializing its values."""
        key = str(name).strip()
        if not key:
            raise ValueError("sample name must not be empty")

        sample_metadata = dict(metadata or {})
        if "rows" not in sample_metadata:
            try:
                sample_metadata["rows"] = int(len(sample))
            except (TypeError, AttributeError):
                pass
        if "columns" not in sample_metadata:
            columns = getattr(sample, "columns", None)
            if columns is not None:
                try:
                    sample_metadata["columns"] = int(len(columns))
                except TypeError:
                    pass

        with self._lock:
            if key in self._samples and not overwrite:
                raise KeyError(f"AnalysisContext sample already exists: {key}")
            self._samples[key] = sample
            self._sample_metadata[key] = sample_metadata
        return sample

    def sample(self, name: str, default: Any = None) -> Any:
        """Return a retained sample by name."""
        with self._lock:
            return self._samples.get(name, default)

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe-ish context provenance without raw cached/sample data."""
        with self._lock:
            samples = deepcopy(self._sample_metadata)
            facts = sorted(self._facts)
            cache_entries = sorted(self._cache)
            hits = int(self._cache_hits)
            misses = int(self._cache_misses)

        return {
            "context_schema_version": CONTEXT_SCHEMA_VERSION,
            "dataset_name": self.dataset_name,
            "shape": {"rows": self.rows, "columns": self.columns},
            "source": deepcopy(dict(self.source)),
            "config": self.config.to_dict(),
            "resource_policy": self.execution_policy.to_dict(),
            "seed": self.seed,
            "facts": facts,
            "cache": {
                "entries": cache_entries,
                "entry_count": len(cache_entries),
                "hits": hits,
                "misses": misses,
            },
            "samples": samples,
        }
