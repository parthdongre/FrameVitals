"""Tensor diagnostics shared by NumPy arrays and framework tensor adapters."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from framevitals.core.beacons import beacon
from framevitals.result import AnalysisResult


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in getattr(tensor, "shape", ()))


def _numel(tensor: Any) -> int:
    method = getattr(tensor, "numel", None)
    if callable(method):
        return int(method())
    return int(np.asarray(tensor).size)


def _nbytes(tensor: Any) -> int | None:
    value = getattr(tensor, "nbytes", None)
    if isinstance(value, (int, np.integer)):
        return int(value)
    element_size = getattr(tensor, "element_size", None)
    if callable(element_size):
        try:
            return int(element_size()) * _numel(tensor)
        except Exception:
            return None
    return None


def _numpy_sample(tensor: Any, limit: int) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        flat = tensor.reshape(-1)
        if flat.size > limit:
            step = max(1, flat.size // limit)
            flat = flat[::step][:limit]
        return np.asarray(flat)

    detached = tensor.detach()
    flat = detached.reshape(-1)
    count = int(flat.numel())
    if count > limit:
        step = max(1, count // limit)
        flat = flat[::step][:limit]
    try:
        flat = flat.float()
    except Exception:
        pass
    return np.asarray(flat.cpu().numpy())


def _full_matrix(tensor: Any) -> np.ndarray | None:
    shape = _shape(tensor)
    if len(shape) != 2:
        return None
    if not shape or max(shape) > 2048 or math.prod(shape) > 2_000_000:
        return None
    if isinstance(tensor, np.ndarray):
        return np.asarray(tensor, dtype=float)
    try:
        return np.asarray(tensor.detach().float().cpu().numpy(), dtype=float)
    except Exception:
        return None


def _health_label(score: float) -> str:
    if score >= 90:
        return "healthy"
    if score >= 75:
        return "good"
    if score >= 55:
        return "attention"
    return "critical"


def tensor_metrics(
    tensor: Any,
    *,
    sample_values: int = 250_000,
) -> dict[str, Any]:
    """Return bounded numeric diagnostics for a tensor-like object."""
    shape = _shape(tensor)
    values = _numpy_sample(tensor, max(1, int(sample_values)))
    total_values = _numel(tensor)

    numeric = values
    try:
        numeric = values.astype(float, copy=False)
    except (TypeError, ValueError):
        return {
            "shape": list(shape),
            "ndim": len(shape),
            "size": total_values,
            "nbytes": _nbytes(tensor),
            "dtype": str(getattr(tensor, "dtype", values.dtype)),
            "numeric": False,
            "sampled_values": int(values.size),
        }

    finite = np.isfinite(numeric)
    nan_count_sample = int(np.isnan(numeric).sum())
    inf_count_sample = int(np.isinf(numeric).sum())
    finite_values = numeric[finite]

    zeros_sample = int((numeric == 0).sum())
    near_zero_sample = int((np.abs(numeric) <= 1.0e-8).sum())
    denom = max(1, numeric.size)

    result: dict[str, Any] = {
        "shape": list(shape),
        "ndim": len(shape),
        "size": total_values,
        "nbytes": _nbytes(tensor),
        "dtype": str(getattr(tensor, "dtype", values.dtype)),
        "numeric": True,
        "sampled_values": int(numeric.size),
        "sample_fraction": round(min(1.0, numeric.size / max(1, total_values)), 8),
        "nan_fraction_sample": round(nan_count_sample / denom, 8),
        "inf_fraction_sample": round(inf_count_sample / denom, 8),
        "zero_fraction_sample": round(zeros_sample / denom, 8),
        "near_zero_fraction_sample": round(near_zero_sample / denom, 8),
    }

    if finite_values.size:
        abs_values = np.abs(finite_values)
        result["distribution"] = {
            "mean": round(float(np.mean(finite_values)), 8),
            "std": round(float(np.std(finite_values)), 8),
            "min": round(float(np.min(finite_values)), 8),
            "max": round(float(np.max(finite_values)), 8),
            "median": round(float(np.median(finite_values)), 8),
            "p01": round(float(np.quantile(finite_values, 0.01)), 8),
            "p99": round(float(np.quantile(finite_values, 0.99)), 8),
            "l1_sample": round(float(np.sum(abs_values)), 8),
            "l2_sample": round(float(np.linalg.norm(finite_values)), 8),
        }

    matrix = _full_matrix(tensor)
    if matrix is not None and np.isfinite(matrix).all():
        try:
            singular = np.linalg.svd(matrix, compute_uv=False)
            if singular.size:
                tol = max(matrix.shape) * np.finfo(float).eps * float(singular[0])
                rank = int(np.sum(singular > tol))
                nonzero = singular[singular > tol]
                condition = (
                    float(singular[0] / nonzero[-1])
                    if nonzero.size
                    else math.inf
                )
                energy = singular**2
                total_energy = float(np.sum(energy))
                effective_rank = 0.0
                if total_energy > 0:
                    probs = energy / total_energy
                    probs = probs[probs > 0]
                    entropy = -float(np.sum(probs * np.log(probs)))
                    effective_rank = float(np.exp(entropy))
                result["matrix"] = {
                    "rank": rank,
                    "rank_ratio": round(rank / max(1, min(matrix.shape)), 8),
                    "effective_rank": round(effective_rank, 6),
                    "condition_number": (
                        round(condition, 6) if math.isfinite(condition) else None
                    ),
                    "largest_singular_value": round(float(singular[0]), 8),
                    "smallest_nonzero_singular_value": (
                        round(float(nonzero[-1]), 8) if nonzero.size else None
                    ),
                }
        except np.linalg.LinAlgError:
            result["matrix"] = {"available": False, "reason": "svd_did_not_converge"}

    return result


def analyze_tensor(
    tensor: Any,
    *,
    depth: str | None = None,
) -> AnalysisResult:
    """Run Prism-compatible diagnostics on a tensor or ndarray."""
    sample_limit = {
        "quick": 50_000,
        "standard": 250_000,
        "deep": 750_000,
        "research": 1_500_000,
    }.get(str(depth or "standard").lower(), 250_000)

    metrics = tensor_metrics(tensor, sample_values=sample_limit)
    findings: list[dict[str, Any]] = []

    nan_fraction = float(metrics.get("nan_fraction_sample") or 0.0)
    inf_fraction = float(metrics.get("inf_fraction_sample") or 0.0)
    zero_fraction = float(metrics.get("zero_fraction_sample") or 0.0)
    near_zero_fraction = float(metrics.get("near_zero_fraction_sample") or 0.0)

    if nan_fraction > 0 or inf_fraction > 0:
        findings.append(
            beacon(
                "tensor.non_finite",
                "Non-finite tensor values were detected",
                severity="critical",
                confidence=1.0,
                summary=(
                    f"Sampled NaN fraction {nan_fraction:.3%}; "
                    f"Inf fraction {inf_fraction:.3%}."
                ),
                recommendation="Trace the first operation that introduces NaN/Inf values.",
                evidence={
                    "nan_fraction_sample": nan_fraction,
                    "inf_fraction_sample": inf_fraction,
                },
            )
        )

    if zero_fraction >= 0.95 and int(metrics.get("size") or 0):
        findings.append(
            beacon(
                "tensor.near_empty",
                "The tensor is almost entirely zero",
                severity="high",
                summary=f"{zero_fraction:.1%} of sampled values are exactly zero.",
                recommendation="Confirm that initialization, masking, pruning, or serialization is correct.",
                evidence={"zero_fraction_sample": zero_fraction},
            )
        )
    elif near_zero_fraction >= 0.80:
        findings.append(
            beacon(
                "tensor.low_magnitude",
                "Most tensor values are near zero",
                severity="medium",
                confidence=0.9,
                summary=f"{near_zero_fraction:.1%} of sampled values have magnitude <= 1e-8.",
                recommendation="Check for collapse, over-pruning, or unexpectedly weak updates.",
                evidence={"near_zero_fraction_sample": near_zero_fraction},
            )
        )

    matrix = metrics.get("matrix") or {}
    rank_ratio = matrix.get("rank_ratio")
    if isinstance(rank_ratio, (float, int)) and rank_ratio < 0.50:
        findings.append(
            beacon(
                "tensor.rank_deficiency",
                "The matrix is strongly rank-deficient",
                severity="high",
                confidence=0.98,
                summary=f"Observed rank uses only {float(rank_ratio):.1%} of available dimensions.",
                recommendation="Inspect redundant dimensions, collapsed features, or degenerate weights.",
                evidence=dict(matrix),
            )
        )

    condition = matrix.get("condition_number")
    if isinstance(condition, (float, int)) and condition >= 1.0e8:
        findings.append(
            beacon(
                "tensor.conditioning",
                "The matrix is poorly conditioned",
                severity="medium",
                confidence=0.95,
                summary=f"Estimated condition number is {float(condition):.3g}.",
                recommendation="Review scaling, redundant dimensions, and numerical stability.",
                evidence=dict(matrix),
            )
        )

    score = 100.0
    score -= min(60.0, (nan_fraction + inf_fraction) * 1000.0)
    score -= min(20.0, max(0.0, zero_fraction - 0.80) * 100.0)
    if isinstance(rank_ratio, (float, int)):
        score -= min(25.0, max(0.0, 0.75 - float(rank_ratio)) * 50.0)
    if isinstance(condition, (float, int)) and condition >= 1.0e8:
        score -= 10.0
    score = round(max(0.0, min(100.0, score)), 2)

    shape = metrics.get("shape") or []
    rows = int(shape[0]) if shape else int(metrics.get("size") or 0)
    columns = int(math.prod(shape[1:])) if len(shape) > 1 else 1

    return AnalysisResult(
        {
            "dataset_id": None,
            "filename": type(tensor).__name__,
            "analysis_mode": str(depth or "standard"),
            "source_kind": "tensor",
            "profile": {
                "shape": {"rows": rows, "columns": columns},
                "structure": metrics,
            },
            "tensor": metrics,
            "health": {"overall_score": score, "label": _health_label(score)},
            "ml_readiness": {"score": None, "label": "not_applicable"},
            "findings": findings,
            "artifacts_enabled": False,
            "execution": {
                "method": "bounded_tensor_diagnostics",
                "sampled": metrics.get("sample_fraction", 1.0) < 1.0,
                "resource_bounded": True,
            },
        }
    )
