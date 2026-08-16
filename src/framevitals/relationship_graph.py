"""Sparse feature-relationship discovery for wide numeric datasets.

The engine deliberately avoids allocating a dense ``columns x columns`` matrix.
It builds compact SimHash-style signatures from a bounded row view, uses
locality-sensitive hash bands to generate a bounded candidate set, and verifies
only those candidates with Pearson correlation on the sampled observations.

This Python implementation defines the semantics for the future Rust engine.
The native implementation can replace the scanning/signature/candidate kernels
without changing the result contract.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import heapq
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class _UnionFind:
    parent: dict[int, int]
    size: dict[int, int]

    @classmethod
    def create(cls) -> "_UnionFind":
        return cls(parent={}, size={})

    def _ensure(self, item: int) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: int) -> int:
        self._ensure(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _sample_positions(rows: int, max_rows: int) -> np.ndarray:
    if rows <= max_rows:
        return np.arange(rows, dtype=np.int64)
    positions = np.linspace(0, rows - 1, num=max_rows, dtype=np.int64)
    return np.unique(positions)


def _numeric_sample(
    dataframe: pd.DataFrame,
    column: Any,
    positions: np.ndarray,
) -> np.ndarray:
    series = pd.to_numeric(dataframe[column].iloc[positions], errors="coerce")
    return series.to_numpy(dtype="float64", na_value=np.nan)


def _standardized_vector(values: np.ndarray, *, min_observations: int) -> np.ndarray | None:
    finite = np.isfinite(values)
    if int(finite.sum()) < min_observations:
        return None

    clean = values[finite]
    median = float(np.median(clean))
    work = np.where(finite, values, median).astype(np.float32, copy=False)
    mean = float(work.mean())
    std = float(work.std())
    if not math.isfinite(std) or std <= 1e-12:
        return None
    return ((work - mean) / std).astype(np.float32, copy=False)


def _signature(vector: np.ndarray, hyperplanes: np.ndarray) -> int:
    bits = np.asarray(vector @ hyperplanes >= 0, dtype=np.uint8)
    packed = np.packbits(bits, bitorder="little").tobytes()
    return int.from_bytes(packed, byteorder="little", signed=False)


def _absolute_band(signature: int, shift: int, band_bits: int) -> int:
    """Canonicalize a signature band so positive/negative correlation can meet."""
    mask = (1 << band_bits) - 1
    value = (signature >> shift) & mask
    complement = mask ^ value
    return min(value, complement)


def _pearson_from_sample(left: np.ndarray, right: np.ndarray) -> tuple[float | None, int]:
    mask = np.isfinite(left) & np.isfinite(right)
    overlap = int(mask.sum())
    if overlap < 10:
        return None, overlap

    x = left[mask].astype(np.float64, copy=False)
    y = right[mask].astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-12:
        return None, overlap
    correlation = float(np.dot(x, y) / denominator)
    if not math.isfinite(correlation):
        return None, overlap
    return max(-1.0, min(1.0, correlation)), overlap


class _SampleCache:
    """Tiny LRU cache so candidate verification does not retain the wide sample."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        columns: list[Any],
        positions: np.ndarray,
        *,
        max_columns: int,
    ) -> None:
        self.dataframe = dataframe
        self.columns = columns
        self.positions = positions
        self.max_columns = max_columns
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, index: int) -> np.ndarray:
        if index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value

        value = _numeric_sample(
            self.dataframe,
            self.columns[index],
            self.positions,
        )
        self.cache[index] = value
        if len(self.cache) > self.max_columns:
            self.cache.popitem(last=False)
        return value


def build_numeric_relationship_graph(
    dataframe: pd.DataFrame,
    *,
    max_sample_rows: int = 512,
    projections: int = 64,
    band_bits: int = 16,
    neighbors_per_bucket: int = 4,
    max_bucket_members: int = 64,
    max_candidate_pairs: int = 250_000,
    min_abs_correlation: float = 0.80,
    max_edges_returned: int = 5_000,
    sample_cache_columns: int = 128,
    random_state: int = 42,
) -> dict[str, Any]:
    """Discover strong numeric relationships without a dense correlation matrix.

    Candidate generation is approximate; candidate verification is a normal
    Pearson calculation on the bounded row view. The result always reports the
    candidate/row budgets and whether either was truncated.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if max_sample_rows < 20:
        raise ValueError("max_sample_rows must be at least 20.")
    if projections < 16 or projections > 256 or projections % 8:
        raise ValueError("projections must be a multiple of 8 between 16 and 256.")
    if band_bits < 4 or projections % band_bits:
        raise ValueError("band_bits must divide projections and be at least 4.")
    if neighbors_per_bucket < 1:
        raise ValueError("neighbors_per_bucket must be at least 1.")
    if max_bucket_members < neighbors_per_bucket:
        raise ValueError("max_bucket_members must be >= neighbors_per_bucket.")
    if max_candidate_pairs < 1:
        raise ValueError("max_candidate_pairs must be at least 1.")
    if not 0 < min_abs_correlation <= 1:
        raise ValueError("min_abs_correlation must be in (0, 1].")
    if max_edges_returned < 1:
        raise ValueError("max_edges_returned must be at least 1.")
    if sample_cache_columns < 2:
        raise ValueError("sample_cache_columns must be at least 2.")

    columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    node_count = len(columns)
    if node_count < 2:
        return {
            "available": False,
            "reason": "Need at least two numeric columns.",
            "nodes": node_count,
            "edges": [],
        }

    positions = _sample_positions(len(dataframe), max_sample_rows)
    sample_rows = int(len(positions))
    min_observations = min(20, max(10, sample_rows // 4))

    rng = np.random.default_rng(random_state)
    hyperplanes = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(sample_rows, projections),
    )
    hyperplanes /= math.sqrt(max(sample_rows, 1))

    signatures: dict[int, int] = {}
    skipped_columns: list[str] = []
    for index, column in enumerate(columns):
        raw = _numeric_sample(dataframe, column, positions)
        vector = _standardized_vector(raw, min_observations=min_observations)
        if vector is None:
            skipped_columns.append(str(column))
            continue
        signatures[index] = _signature(vector, hyperplanes)

    band_count = projections // band_bits
    buckets: dict[tuple[int, int], list[int]] = {}
    candidates: set[tuple[int, int]] = set()
    candidate_truncated = False

    for index in sorted(signatures):
        signature = signatures[index]
        for band in range(band_count):
            key = (
                band,
                _absolute_band(signature, band * band_bits, band_bits),
            )
            members = buckets.setdefault(key, [])
            for other in members[-neighbors_per_bucket:]:
                pair = (other, index) if other < index else (index, other)
                candidates.add(pair)
                if len(candidates) >= max_candidate_pairs:
                    candidate_truncated = True
                    break
            if len(members) >= max_bucket_members:
                del members[: len(members) - max_bucket_members + 1]
            members.append(index)
            if candidate_truncated:
                break
        if candidate_truncated:
            break

    total_possible_pairs = node_count * (node_count - 1) // 2
    cache = _SampleCache(
        dataframe,
        columns,
        positions,
        max_columns=sample_cache_columns,
    )

    union_find = _UnionFind.create()
    degrees: Counter[int] = Counter()
    verified_edges = 0
    top_edge_heap: list[tuple[float, int, int, float, int]] = []

    for left, right in sorted(candidates):
        correlation, overlap = _pearson_from_sample(cache.get(left), cache.get(right))
        if correlation is None or abs(correlation) < min_abs_correlation:
            continue

        verified_edges += 1
        union_find.union(left, right)
        degrees[left] += 1
        degrees[right] += 1
        item = (abs(correlation), left, right, correlation, overlap)
        if len(top_edge_heap) < max_edges_returned:
            heapq.heappush(top_edge_heap, item)
        elif item[0] > top_edge_heap[0][0]:
            heapq.heapreplace(top_edge_heap, item)

    edges = [
        {
            "source": str(columns[left]),
            "target": str(columns[right]),
            "correlation": round(float(correlation), 6),
            "abs_correlation": round(float(score), 6),
            "overlap": int(overlap),
        }
        for score, left, right, correlation, overlap in sorted(
            top_edge_heap,
            key=lambda item: (-item[0], str(columns[item[1]]), str(columns[item[2]])),
        )
    ]

    component_members: dict[int, list[int]] = {}
    for index in union_find.parent:
        root = union_find.find(index)
        component_members.setdefault(root, []).append(index)

    components = sorted(
        component_members.values(),
        key=lambda members: (-len(members), [str(columns[index]) for index in members]),
    )
    component_summary = [
        {
            "size": len(members),
            "members": [str(columns[index]) for index in sorted(members)[:20]],
            "members_truncated": len(members) > 20,
        }
        for members in components[:50]
        if len(members) >= 2
    ]

    connected_nodes = len({index for pair in candidates for index in pair if degrees[index]})
    high_degree = sorted(
        (
            {"column": str(columns[index]), "degree": int(degree)}
            for index, degree in degrees.items()
        ),
        key=lambda item: (-item["degree"], item["column"]),
    )[:25]

    return {
        "available": True,
        "method": "bounded_simhash_lsh_then_pearson",
        "nodes": node_count,
        "usable_signature_nodes": len(signatures),
        "skipped_nodes": len(skipped_columns),
        "skipped_columns": skipped_columns[:25],
        "sample": {
            "source_rows": int(len(dataframe)),
            "sample_rows": sample_rows,
            "sampled": len(dataframe) > sample_rows,
            "strategy": "deterministic_evenly_spaced",
        },
        "candidate_generation": {
            "projections": projections,
            "band_bits": band_bits,
            "bands": band_count,
            "neighbors_per_bucket": neighbors_per_bucket,
            "candidate_pairs": len(candidates),
            "max_candidate_pairs": max_candidate_pairs,
            "truncated": candidate_truncated,
            "dense_pairs_avoided": max(total_possible_pairs - len(candidates), 0),
            "total_possible_dense_pairs": total_possible_pairs,
        },
        "verification": {
            "method": "sample_pearson",
            "min_abs_correlation": float(min_abs_correlation),
            "verified_relationships": verified_edges,
            "edges_returned": len(edges),
            "edges_truncated": verified_edges > len(edges),
        },
        "graph": {
            "connected_nodes": connected_nodes,
            "isolated_or_unverified_nodes": max(node_count - connected_nodes, 0),
            "component_count": len(component_summary),
            "components": component_summary,
            "high_degree_nodes": high_degree,
        },
        "edges": edges,
    }
