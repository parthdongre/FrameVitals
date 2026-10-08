"""Metadata-first diagnostics for local Safetensors checkpoint files."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from framevitals.core.beacons import beacon
from framevitals.quality_results import DriftResult
from framevitals.result import AnalysisResult


_MAX_HEADER_BYTES = 64 * 1024 * 1024


def _architecture(names: list[str], shapes: list[list[int]]) -> str:
    lowered = [name.lower() for name in names]
    if any(
        marker in name
        for name in lowered
        for marker in ("q_proj", "k_proj", "v_proj", "self_attn", "attention.")
    ):
        return "transformer"
    if any("conv" in name for name in lowered) and any(len(shape) in (3, 4, 5) for shape in shapes):
        return "cnn"
    if any(marker in name for name in lowered for marker in ("lstm", "gru", "rnn")):
        return "recurrent"
    if any(marker in name for name in lowered for marker in ("embed", "embedding")):
        return "neural_network"
    return "checkpoint"


def _read_header(path: str | Path) -> tuple[dict[str, Any], int, int]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    file_size = checkpoint.stat().st_size
    if file_size < 8:
        raise ValueError("Safetensors file is too small to contain a valid header.")

    with checkpoint.open("rb") as handle:
        raw_length = handle.read(8)
        header_length = int.from_bytes(raw_length, byteorder="little", signed=False)
        if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
            raise ValueError(
                f"Safetensors header length {header_length} is outside the supported bound."
            )
        if 8 + header_length > file_size:
            raise ValueError("Safetensors header extends beyond the end of the file.")
        raw_header = handle.read(header_length)

    try:
        header = json.loads(raw_header.decode("utf-8").strip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Safetensors header is not valid UTF-8 JSON.") from exc

    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be a JSON object.")

    return header, header_length, file_size


def inspect_safetensors(path: str | Path) -> dict[str, Any]:
    """Inspect Safetensors metadata without loading tensor payloads."""
    header, header_length, file_size = _read_header(path)
    checkpoint = Path(path)
    data_start = 8 + header_length
    data_bytes = max(0, file_size - data_start)

    user_metadata = header.get("__metadata__")
    if not isinstance(user_metadata, dict):
        user_metadata = {}

    tensors: list[dict[str, Any]] = []
    dtype_counts: Counter[str] = Counter()
    total_elements = 0
    total_tensor_bytes = 0
    zero_sized: list[str] = []

    for name, payload in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"Tensor entry {name!r} is not an object.")

        dtype = str(payload.get("dtype", "unknown"))
        shape_raw = payload.get("shape", [])
        offsets = payload.get("data_offsets", [])

        if not isinstance(shape_raw, list) or not all(
            isinstance(v, int) and v >= 0 for v in shape_raw
        ):
            raise ValueError(f"Tensor {name!r} has an invalid shape.")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(v, int) and v >= 0 for v in offsets)
        ):
            raise ValueError(f"Tensor {name!r} has invalid data offsets.")

        start, end = int(offsets[0]), int(offsets[1])
        if end < start or end > data_bytes:
            raise ValueError(f"Tensor {name!r} points outside the checkpoint data section.")

        shape = [int(v) for v in shape_raw]
        elements = int(math.prod(shape)) if shape else 1
        tensor_bytes = end - start
        if elements == 0:
            zero_sized.append(str(name))

        total_elements += elements
        total_tensor_bytes += tensor_bytes
        dtype_counts[dtype] += elements

        tensors.append({
            "name": str(name),
            "dtype": dtype,
            "shape": shape,
            "elements": elements,
            "bytes": tensor_bytes,
            "data_offsets": [start, end],
        })

    tensors.sort(key=lambda item: (-int(item["bytes"]), item["name"]))
    architecture = _architecture(
        [item["name"] for item in tensors],
        [item["shape"] for item in tensors],
    )

    return {
        "format": "safetensors",
        "path": str(checkpoint),
        "filename": checkpoint.name,
        "file_size": file_size,
        "header_bytes": header_length,
        "data_bytes": data_bytes,
        "tensor_count": len(tensors),
        "parameters": total_elements,
        "tensor_bytes": total_tensor_bytes,
        "architecture": architecture,
        "dtype_parameter_counts": dict(dtype_counts),
        "metadata": dict(user_metadata),
        "zero_sized_tensors": zero_sized,
        "largest_tensors": tensors[:25],
        "tensors": tensors,
    }


def _health_label(score: float) -> str:
    if score >= 90:
        return "healthy"
    if score >= 75:
        return "good"
    if score >= 55:
        return "attention"
    return "critical"


def analyze_safetensors(
    path: str | Path,
    *,
    depth: str | None = None,
) -> AnalysisResult:
    """Run metadata-first Prism diagnostics on a local Safetensors file."""
    summary = inspect_safetensors(path)
    findings: list[dict[str, Any]] = []

    if not summary["tensor_count"]:
        findings.append(beacon(
            "model.safetensors.empty",
            "The checkpoint contains no tensors",
            severity="critical",
            confidence=1.0,
            recommendation="Verify the checkpoint export and source file.",
        ))

    if summary["zero_sized_tensors"]:
        findings.append(beacon(
            "model.safetensors.zero_sized",
            "Zero-sized tensors were found in the checkpoint",
            severity="medium",
            confidence=1.0,
            summary=f"{len(summary['zero_sized_tensors'])} tensors contain zero elements.",
            recommendation="Confirm whether empty tensors are intentional for this architecture.",
            evidence={"tensors": summary["zero_sized_tensors"][:30]},
        ))

    dtype_counts = summary["dtype_parameter_counts"]
    if len(dtype_counts) >= 3:
        findings.append(beacon(
            "model.safetensors.mixed_precision",
            "The checkpoint uses several parameter dtypes",
            severity="info",
            confidence=1.0,
            summary=f"{len(dtype_counts)} tensor dtypes are present.",
            recommendation="Confirm that the precision mix matches the intended export/runtime policy.",
            evidence={"dtype_parameter_counts": dtype_counts},
        ))

    score = 100.0
    score -= min(60.0, len(summary["zero_sized_tensors"]) * 5.0)
    if not summary["tensor_count"]:
        score = 0.0
    score = round(max(0.0, min(100.0, score)), 2)

    model_summary = {
        **summary,
        "framework": "safetensors",
        "class_name": "SafetensorsCheckpoint",
        "trainable_parameters": None,
        "frozen_parameters": None,
        "runtime": None,
    }

    return AnalysisResult({
        "dataset_id": None,
        "filename": summary["filename"],
        "analysis_mode": str(depth or "standard"),
        "source_kind": "model",
        "profile": {
            "shape": {
                "rows": int(summary["parameters"]),
                "columns": int(summary["tensor_count"]),
            },
            "structure": model_summary,
        },
        "model": model_summary,
        "health": {"overall_score": score, "label": _health_label(score)},
        "ml_readiness": {"score": None, "label": "not_applicable"},
        "findings": findings,
        "artifacts_enabled": False,
        "execution": {
            "method": "safetensors_metadata",
            "sampled": False,
            "resource_bounded": True,
            "tensor_payloads_loaded": False,
        },
    })


def compare_safetensors(reference: str | Path, current: str | Path) -> DriftResult:
    """Compare checkpoint structure using Safetensors metadata only."""
    ref = inspect_safetensors(reference)
    cur = inspect_safetensors(current)

    ref_tensors = {item["name"]: item for item in ref["tensors"]}
    cur_tensors = {item["name"]: item for item in cur["tensors"]}
    ref_names = set(ref_tensors)
    cur_names = set(cur_tensors)

    added = sorted(cur_names - ref_names)
    removed = sorted(ref_names - cur_names)
    shape_changes: list[dict[str, Any]] = []
    dtype_changes: list[dict[str, Any]] = []

    for name in sorted(ref_names & cur_names):
        left = ref_tensors[name]
        right = cur_tensors[name]
        if left["shape"] != right["shape"]:
            shape_changes.append({
                "name": name,
                "reference_shape": left["shape"],
                "current_shape": right["shape"],
            })
        if left["dtype"] != right["dtype"]:
            dtype_changes.append({
                "name": name,
                "reference_dtype": left["dtype"],
                "current_dtype": right["dtype"],
            })

    union_count = max(1, len(ref_names | cur_names))
    structural_churn = (
        len(added) + len(removed) + len(shape_changes)
    ) / union_count
    dtype_churn = len(dtype_changes) / max(1, len(ref_names & cur_names))
    parameter_change = abs(cur["parameters"] - ref["parameters"]) / max(1, ref["parameters"])
    score = max(
        min(1.0, structural_churn * 2.0),
        min(1.0, dtype_churn),
        min(1.0, parameter_change),
    )

    if score >= 0.75:
        severity = "severe"
    elif score >= 0.40:
        severity = "moderate"
    elif score >= 0.15:
        severity = "minor"
    else:
        severity = "stable"

    status = "fail" if severity == "severe" else "warn" if severity != "stable" else "pass"

    return DriftResult({
        "available": True,
        "source_kind": "model",
        "gate": {"status": status, "severity": severity},
        "summary": {
            "overall_verdict": severity,
            "change_score": round(score, 6),
            "parameters_compared": len(ref_names & cur_names),
            "parameters_added": len(added),
            "parameters_removed": len(removed),
            "shape_changes": len(shape_changes),
            "dtype_changes": len(dtype_changes),
        },
        "model": {
            "reference_class": "SafetensorsCheckpoint",
            "current_class": "SafetensorsCheckpoint",
            "reference_parameters": ref["parameters"],
            "current_parameters": cur["parameters"],
            "parameters_compared": len(ref_names & cur_names),
            "added_parameters": added,
            "removed_parameters": removed,
            "shape_changes": shape_changes,
            "dtype_changes": dtype_changes,
            "median_relative_l2": None,
            "p95_relative_l2": None,
            "max_relative_l2": None,
            "most_changed": [],
            "comparison_scope": "metadata_only",
        },
        "columns": [],
    })
