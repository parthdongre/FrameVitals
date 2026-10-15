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
    """Uniform seeded positions; avoid stride aliasing and full NumPy copies."""
    count = _numel(tensor)
    positions = None
    if count > limit:
        positions = np.sort(np.random.default_rng(42).choice(count, limit, replace=False))
    if isinstance(tensor, np.ndarray):
        return np.asarray(tensor.flat[positions] if positions is not None else tensor.ravel())
    value = tensor.detach()
    if positions is not None:
        import torch

        coordinates = np.unravel_index(positions, _shape(tensor))
        value = value[tuple(torch.as_tensor(c, device=value.device) for c in coordinates)]
    if str(value.dtype) == "torch.bfloat16":
        value = value.float()
    return np.asarray(value.cpu().numpy()).reshape(-1)


def _nonfinite_summary(tensor: Any, sample: np.ndarray, total: int) -> dict[str, Any]:
    cap = 10_000_000 if isinstance(tensor, np.ndarray) else 1_000_000
    nan = inf = checked = 0
    if total <= cap and isinstance(tensor, np.ndarray):
        for start in range(0, total, 65536):
            chunk = np.asarray(tensor.flat[start : start + 65536], dtype=float)
            nan += int(np.isnan(chunk).sum())
            inf += int(np.isinf(chunk).sum())
            checked += chunk.size
    elif total <= cap:
        import torch

        value = tensor.detach()
        nan = int(torch.isnan(value).sum().item())
        inf = int(torch.isinf(value).sum().item())
        checked = total
    else:
        nan, inf, checked = int(np.isnan(sample).sum()), int(np.isinf(sample).sum()), sample.size
    return {
        "coverage": "full" if checked == total else "sampled",
        "sampled": checked != total,
        "approximate": False,
        "population": "checked_values",
        "checked_values": int(checked),
        "nan_count": nan,
        "inf_count": inf,
        "scan_cap": cap,
        "all_finite": (nan + inf == 0) if checked == total else None,
        "interpretation": "Counts are exact for checked values; sample absence does not certify population finiteness.",
    }


def _matrix_sketch(tensor: Any, *, max_rows: int = 768, max_columns: int = 768):
    shape = _shape(tensor)
    if len(shape) != 2 or min(shape) == 0:
        return None, False, shape
    exact = max(shape) <= 2048 and math.prod(shape) <= 2_000_000
    try:
        if exact:
            return (
                _numpy_sample(tensor, math.prod(shape)).reshape(shape).astype(float),
                False,
                shape,
            )
        rng = np.random.default_rng(42)
        rows = np.sort(rng.choice(shape[0], min(max_rows, shape[0]), replace=False))
        cols = np.sort(rng.choice(shape[1], min(max_columns, shape[1]), replace=False))
        if isinstance(tensor, np.ndarray):
            return np.asarray(tensor[np.ix_(rows, cols)], dtype=float), True, shape
        import torch

        value = tensor.detach()
        rr = torch.as_tensor(rows, device=value.device)
        cc = torch.as_tensor(cols, device=value.device)
        sketch = value[rr[:, None], cc[None, :]]
        if str(sketch.dtype) == "torch.bfloat16":
            sketch = sketch.float()
        return np.asarray(sketch.cpu().numpy(), dtype=float), True, shape
    except (TypeError, ValueError, RuntimeError):
        return None, False, shape


def _spectral_metrics(matrix: np.ndarray, sampled: bool, original_shape) -> dict[str, Any]:
    """Scale before squaring; spectral summaries always name their population."""
    min_dim = min(matrix.shape)
    scale = float(np.max(np.abs(matrix)))
    a = matrix / scale if scale else matrix
    frobenius_sq = float(np.sum(a * a))
    full = not sampled and (a.size <= 250_000 or min_dim <= 64)
    if full:
        singular = np.linalg.svd(a, compute_uv=False)
        components = min_dim
    else:
        from sklearn.utils.extmath import randomized_svd

        components = min(64, min_dim)
        _, singular, _ = randomized_svd(a, n_components=components, n_iter=3, random_state=42)
    energy = singular**2
    captured = float(energy.sum())
    probs = energy[energy > 0] / captured if captured else np.array([])
    effective = float(np.exp(-np.sum(probs * np.log(probs)))) if captured else 0.0
    stable = frobenius_sq / float(singular[0] ** 2) if singular[0] > 0 else 0.0
    result = {
        "method": "exact_svd"
        if full
        else "sampled_submatrix_randomized_svd"
        if sampled
        else "randomized_svd",
        "method_version": "2",
        "available": True,
        "approximate": not full,
        "sampled": sampled,
        "population": "sampled_submatrix" if sampled else "full_matrix",
        "original_shape": list(original_shape),
        "analyzed_shape": list(matrix.shape),
        "singular_values": [float(v * scale) for v in singular[:64]],
        "singular_values_truncated": len(singular) > 64 or components < min_dim,
        "largest_singular_value": float(singular[0] * scale),
        "stable_rank": None if sampled else stable,
        "stable_rank_ratio": None if sampled else stable / min(original_shape),
        "effective_rank_definition": "exp(entropy(squared_singular_value_proportions))",
        "numerical_scaling": "max_absolute_entry",
        "scale": scale,
    }
    if full:
        tol = max(a.shape) * np.finfo(float).eps * float(singular[0])
        rank = int(np.sum(singular > tol))
        condition = float(singular[0] / singular[-1]) if rank == min_dim else None
        result.update(
            rank=rank,
            rank_ratio=rank / min_dim,
            effective_rank=effective,
            rank_tolerance=float(tol * scale),
            condition_number=condition,
            condition_status="finite" if condition is not None else "singular_under_tolerance",
            smallest_nonzero_singular_value=float(singular[rank - 1] * scale) if rank else None,
        )
    else:
        result.update(
            components=components,
            effective_rank_topk=effective,
            captured_energy_ratio=min(1.0, captured / frobenius_sq) if frobenius_sq else 0.0,
            relative_frobenius_residual=math.sqrt(max(0.0, 1 - captured / frobenius_sq))
            if frobenius_sq
            else 0.0,
            residual_population=result["population"],
            random_seed=42,
            power_iterations=3,
            stable_rank_interpretation="Upper estimate: randomized spectral norm can be underestimated; no rank estimate or probabilistic confidence bound.",
        )
        if sampled:
            result.update(
                submatrix_stable_rank=stable,
                submatrix_stable_rank_ratio=stable / min_dim,
                sampling_method="uniform_without_replacement",
                interpretation="No unbiased full-matrix rank or spectrum claim; residual describes the sampled submatrix only.",
            )
    return result


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

    if np.iscomplexobj(values):
        return {
            "shape": list(shape),
            "size": total_values,
            "numeric": False,
            "reason": "complex_distribution_unsupported",
            "sampled_values": int(values.size),
        }
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

    result["nonfinite"] = _nonfinite_summary(tensor, numeric, total_values)
    result["sampling"] = {
        "method": "uniform_without_replacement",
        "seed": 42,
        "population": "tensor_entries",
        "sampled": numeric.size < total_values,
    }

    if finite_values.size:
        abs_values = np.abs(finite_values)
        scale = float(abs_values.max())
        normalized = finite_values / scale if scale else finite_values
        result["distribution"] = {
            "mean": float(np.mean(normalized) * scale),
            "std": float(np.std(normalized) * scale),
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
            "median": float(np.median(normalized) * scale),
            "p01": float(np.quantile(normalized, 0.01) * scale),
            "p99": float(np.quantile(normalized, 0.99) * scale),
            "l1_sample": float(np.sum(np.abs(normalized)) * scale),
            "l2_sample": float(np.linalg.norm(normalized) * scale),
        }
    matrix, sampled, original_shape = _matrix_sketch(tensor)
    if matrix is not None:
        if not np.isfinite(matrix).all():
            result["matrix"] = {"available": False, "reason": "nonfinite_matrix"}
        else:
            try:
                result["matrix"] = _spectral_metrics(matrix, sampled, original_shape)
            except (np.linalg.LinAlgError, ValueError):
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

    nonfinite = metrics.get("nonfinite", {})
    if nonfinite.get("nan_count", 0) or nonfinite.get("inf_count", 0):
        findings.append(
            beacon(
                "tensor.non_finite",
                "Non-finite tensor values were detected",
                severity="critical",
                confidence=1.0,
                summary=(
                    f"Checked {nonfinite.get('checked_values', 0):,} values: "
                    f"{nonfinite.get('nan_count', 0)} NaN and {nonfinite.get('inf_count', 0)} infinity values."
                ),
                recommendation="Trace the first operation that introduces NaN/Inf values.",
                evidence={
                    **nonfinite,
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
    stable_rank_ratio = matrix.get("stable_rank_ratio")
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
    elif (
        bool(matrix.get("approximate"))
        and matrix.get("population") == "full_matrix"
        and isinstance(stable_rank_ratio, (float, int))
        and float(stable_rank_ratio) < 0.10
    ):
        findings.append(
            beacon(
                "tensor.low_effective_dimension",
                "The matrix has very low stable rank",
                severity="medium",
                confidence=0.9,
                summary=(
                    f"Stable rank uses about {float(stable_rank_ratio):.1%} "
                    "of available dimensions."
                ),
                recommendation="Inspect redundant or collapsed representation dimensions.",
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
    elif (
        isinstance(stable_rank_ratio, (float, int))
        and bool(matrix.get("approximate"))
        and matrix.get("population") == "full_matrix"
    ):
        score -= min(
            15.0,
            max(0.0, 0.20 - float(stable_rank_ratio)) * 50.0,
        )
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
            "health": {
                "overall_score": score if metrics.get("numeric") else None,
                "label": _health_label(score) if metrics.get("numeric") else "unavailable",
            },
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
