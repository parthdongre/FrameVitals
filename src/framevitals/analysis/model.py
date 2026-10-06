"""Static and lightweight runtime diagnostics for PyTorch-style models."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from framevitals.analysis.tensor import tensor_metrics
from framevitals.core.beacons import beacon
from framevitals.result import AnalysisResult


def _sample_tensor(tensor: Any, limit: int) -> np.ndarray:
    detached = tensor.detach().reshape(-1)
    count = int(detached.numel())
    if count > limit:
        step = max(1, count // limit)
        detached = detached[::step][:limit]
    try:
        detached = detached.float()
    except Exception:
        pass
    return np.asarray(detached.cpu().numpy(), dtype=float)


def _architecture(module_counts: Counter[str]) -> str:
    names = set(module_counts)
    if any(name.startswith("Conv") for name in names):
        return "cnn"
    if any(
        marker in name
        for name in names
        for marker in ("MultiheadAttention", "TransformerEncoder", "TransformerDecoder")
    ):
        return "transformer"
    if any(name in {"LSTM", "GRU", "RNN"} for name in names):
        return "recurrent"
    if any("GraphConv" in name or "GCN" in name or "GAT" in name for name in names):
        return "gnn"
    return "neural_network"


def _health_label(score: float) -> str:
    if score >= 90:
        return "healthy"
    if score >= 75:
        return "good"
    if score >= 55:
        return "attention"
    return "critical"


def _parameter_summary(name: str, parameter: Any, sample_limit: int) -> dict[str, Any]:
    shape = [int(v) for v in parameter.shape]
    size = int(parameter.numel())
    values = _sample_tensor(parameter, sample_limit)
    finite = np.isfinite(values)
    finite_values = values[finite]
    result: dict[str, Any] = {
        "name": name,
        "shape": shape,
        "size": size,
        "dtype": str(parameter.dtype),
        "requires_grad": bool(getattr(parameter, "requires_grad", False)),
        "sampled_values": int(values.size),
        "nan_fraction_sample": round(float(np.isnan(values).mean()), 8) if values.size else 0.0,
        "inf_fraction_sample": round(float(np.isinf(values).mean()), 8) if values.size else 0.0,
        "zero_fraction_sample": round(float((values == 0).mean()), 8) if values.size else 0.0,
        "near_zero_fraction_sample": (
            round(float((np.abs(values) <= 1.0e-8).mean()), 8) if values.size else 0.0
        ),
    }
    element_size = getattr(parameter, "element_size", None)
    if callable(element_size):
        result["nbytes"] = int(element_size()) * size

    if finite_values.size:
        result.update(
            {
                "mean": round(float(np.mean(finite_values)), 8),
                "std": round(float(np.std(finite_values)), 8),
                "min": round(float(np.min(finite_values)), 8),
                "max": round(float(np.max(finite_values)), 8),
                "l2_norm_sample": round(float(np.linalg.norm(finite_values)), 8),
            }
        )
    return result


def _gradient_summary(named_parameters: list[tuple[str, Any]], sample_limit: int) -> dict[str, Any]:
    present = 0
    nonfinite: list[str] = []
    norms: list[tuple[str, float]] = []
    zeros: list[str] = []

    for name, parameter in named_parameters:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        present += 1
        try:
            values = _sample_tensor(grad, sample_limit)
        except Exception:
            continue
        if values.size == 0:
            continue
        if not np.isfinite(values).all():
            nonfinite.append(name)
            continue
        norm = float(np.linalg.norm(values))
        norms.append((name, norm))
        if norm <= 1.0e-12:
            zeros.append(name)

    norms.sort(key=lambda item: item[1], reverse=True)
    return {
        "available": present > 0,
        "parameters_with_gradients": present,
        "nonfinite_parameters": nonfinite[:50],
        "zero_gradient_parameters": zeros[:50],
        "largest_sampled_gradient_norms": [
            {"name": name, "norm": round(value, 8)} for name, value in norms[:20]
        ],
    }


def _cnn_filter_summary(
    named_parameters: list[tuple[str, Any]],
    modules: dict[str, Any],
) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    total_dead = 0

    for name, parameter in named_parameters:
        if not name.endswith(".weight"):
            continue
        prefix = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules.get(prefix)
        module_name = type(module).__name__ if module is not None else ""
        if not module_name.startswith("Conv"):
            continue
        if len(parameter.shape) not in (3, 4, 5):
            continue

        try:
            matrix = parameter.detach().float().reshape(int(parameter.shape[0]), -1)
            norms = np.asarray(matrix.norm(dim=1).cpu().numpy(), dtype=float)
        except Exception:
            continue

        if norms.size == 0:
            continue
        median = float(np.median(norms))
        threshold = max(1.0e-10, median * 1.0e-3)
        dead = int(np.sum(norms <= threshold))
        total_dead += dead
        layers.append(
            {
                "name": prefix,
                "module": module_name,
                "filters": int(norms.size),
                "dead_filters": dead,
                "dead_fraction": round(dead / norms.size, 8),
                "median_filter_norm": round(median, 8),
                "max_filter_norm": round(float(np.max(norms)), 8),
            }
        )

    return {
        "available": bool(layers),
        "layers": layers,
        "dead_filters": total_dead,
        "affected_layers": sum(1 for layer in layers if layer["dead_filters"] > 0),
    }


def _rank_diagnostics(
    named_parameters: list[tuple[str, Any]],
    *,
    max_matrices: int,
) -> list[dict[str, Any]]:
    candidates = [
        (name, parameter)
        for name, parameter in named_parameters
        if len(parameter.shape) == 2
        and int(parameter.numel()) <= 2_000_000
        and min(int(v) for v in parameter.shape) >= 2
    ]
    candidates.sort(key=lambda item: int(item[1].numel()), reverse=True)
    output: list[dict[str, Any]] = []
    for name, parameter in candidates[:max_matrices]:
        metrics = tensor_metrics(parameter, sample_values=50_000)
        matrix = metrics.get("matrix")
        if isinstance(matrix, dict):
            output.append({"name": name, "shape": list(parameter.shape), **matrix})
    return output


def analyze_model(
    model: Any,
    *,
    depth: str | None = None,
    sample_batch: Any = None,
    targets: Any = None,
    loss_fn: Any = None,
    backward: bool = False,
    max_runtime_modules: int | None = None,
    optimizer: Any = None,
) -> AnalysisResult:
    """Inspect a PyTorch-style nn.Module without requiring PyTorch in FrameVitals."""
    if not (
        callable(getattr(model, "named_parameters", None))
        and callable(getattr(model, "named_modules", None))
    ):
        raise TypeError("Expected a PyTorch-style model with named_parameters/named_modules.")

    mode = str(depth or "standard").lower()
    sample_limit = {
        "quick": 2_048,
        "standard": 8_192,
        "deep": 16_384,
        "research": 32_768,
    }.get(mode, 8_192)
    rank_limit = {"quick": 2, "standard": 4, "deep": 8, "research": 16}.get(mode, 4)

    named_parameters = list(model.named_parameters())
    modules = dict(model.named_modules())
    module_counts = Counter(type(module).__name__ for module in modules.values())
    architecture = _architecture(module_counts)

    parameter_summaries = [
        _parameter_summary(name, parameter, sample_limit)
        for name, parameter in named_parameters
    ]

    total_parameters = sum(int(parameter.numel()) for _, parameter in named_parameters)
    trainable_parameters = sum(
        int(parameter.numel())
        for _, parameter in named_parameters
        if bool(getattr(parameter, "requires_grad", False))
    )
    total_bytes = sum(int(item.get("nbytes") or 0) for item in parameter_summaries)
    dtype_counts: Counter[str] = Counter()
    for item in parameter_summaries:
        dtype_counts[item["dtype"]] += int(item["size"])

    nonfinite = [
        item
        for item in parameter_summaries
        if float(item.get("nan_fraction_sample") or 0.0) > 0
        or float(item.get("inf_fraction_sample") or 0.0) > 0
    ]
    highly_sparse = [
        item
        for item in parameter_summaries
        if int(item.get("size") or 0) > 0
        and float(item.get("near_zero_fraction_sample") or 0.0) >= 0.95
    ]

    norm_items = [
        (item["name"], float(item["l2_norm_sample"]))
        for item in parameter_summaries
        if isinstance(item.get("l2_norm_sample"), (int, float))
        and float(item["l2_norm_sample"]) > 0
    ]
    norm_outliers: list[dict[str, Any]] = []
    if norm_items:
        logs = np.asarray([math.log10(value) for _, value in norm_items], dtype=float)
        median = float(np.median(logs))
        mad = float(np.median(np.abs(logs - median)))
        threshold = median + 6.0 * max(mad, 0.15)
        for (name, norm), log_norm in zip(norm_items, logs, strict=True):
            if float(log_norm) > threshold:
                norm_outliers.append({"name": name, "sampled_l2_norm": round(norm, 8)})

    rank = _rank_diagnostics(named_parameters, max_matrices=rank_limit)
    low_rank = [
        item
        for item in rank
        if isinstance(item.get("rank_ratio"), (float, int))
        and float(item["rank_ratio"]) < 0.50
    ]
    gradients = _gradient_summary(named_parameters, max(1024, sample_limit // 2))
    cnn = _cnn_filter_summary(named_parameters, modules) if architecture == "cnn" else {
        "available": False,
        "layers": [],
        "dead_filters": 0,
        "affected_layers": 0,
    }

    from framevitals.analysis.architectures import analyze_architecture

    architecture_details, architecture_findings = analyze_architecture(
        architecture,
        named_parameters,
        modules,
    )

    findings: list[dict[str, Any]] = list(architecture_findings)
    if nonfinite:
        findings.append(
            beacon(
                "model.non_finite_weights",
                "Non-finite model parameters were detected",
                severity="critical",
                confidence=1.0,
                summary=f"{len(nonfinite)} parameter tensors contain sampled NaN/Inf values.",
                recommendation="Stop training/inference and trace the first unstable layer or update.",
                evidence={"parameters": [item["name"] for item in nonfinite[:50]]},
            )
        )

    if gradients.get("nonfinite_parameters"):
        findings.append(
            beacon(
                "model.non_finite_gradients",
                "Non-finite gradients were detected",
                severity="critical",
                confidence=1.0,
                summary=(
                    f"{len(gradients['nonfinite_parameters'])} parameter tensors "
                    "contain non-finite sampled gradients."
                ),
                recommendation="Inspect loss scaling, learning rate, normalization, and gradient clipping.",
                evidence={"parameters": gradients["nonfinite_parameters"]},
            )
        )

    if norm_outliers:
        findings.append(
            beacon(
                "model.weight_norm_outliers",
                "Extreme parameter-norm outliers were detected",
                severity="high",
                confidence=0.9,
                summary=f"{len(norm_outliers)} tensors have unusually large sampled L2 norms.",
                recommendation="Inspect these layers for unstable updates or incompatible initialization.",
                evidence={"parameters": norm_outliers[:20]},
            )
        )

    if low_rank:
        findings.append(
            beacon(
                "model.rank_collapse",
                "Some weight matrices are strongly rank-deficient",
                severity="high",
                confidence=0.95,
                summary=f"{len(low_rank)} bounded matrices use less than half their available rank.",
                recommendation="Inspect representation collapse, redundant projections, or over-compression.",
                evidence={"matrices": low_rank[:20]},
            )
        )

    if cnn.get("dead_filters", 0):
        findings.append(
            beacon(
                "model.cnn.dead_filters",
                "Inactive convolution filters were detected",
                severity="high" if cnn["dead_filters"] >= 8 else "medium",
                confidence=0.95,
                summary=(
                    f"{cnn['dead_filters']} near-zero filters were found across "
                    f"{cnn['affected_layers']} convolution layers."
                ),
                recommendation="Inspect initialization, pruning, regularization, and activation flow.",
                evidence={
                    "dead_filters": cnn["dead_filters"],
                    "layers": [
                        item for item in cnn["layers"] if item["dead_filters"] > 0
                    ][:20],
                },
            )
        )

    if highly_sparse and len(highly_sparse) >= max(2, len(parameter_summaries) // 10):
        findings.append(
            beacon(
                "model.extreme_sparsity",
                "Many parameter tensors are almost entirely near zero",
                severity="medium",
                confidence=0.9,
                summary=f"{len(highly_sparse)} tensors are at least 95% near zero in sampled values.",
                recommendation="Confirm whether this sparsity is intentional pruning or model collapse.",
                evidence={"parameters": [item["name"] for item in highly_sparse[:50]]},
            )
        )

    optimizer_report: dict[str, Any] | None = None
    optimizer_findings: list[dict[str, Any]] = []
    if optimizer is not None:
        from framevitals.analysis.optimizer import inspect_optimizer

        optimizer_report = inspect_optimizer(model, optimizer)
        raw_optimizer_findings = optimizer_report.get("findings", [])
        if isinstance(raw_optimizer_findings, list):
            optimizer_findings = [
                item for item in raw_optimizer_findings
                if isinstance(item, dict)
            ]
            findings.extend(optimizer_findings)

    runtime: dict[str, Any] | None = None
    if sample_batch is not None:
        from framevitals.analysis.runtime_model import observe_model_runtime

        runtime_module_limit = (
            int(max_runtime_modules)
            if max_runtime_modules is not None
            else {"quick": 48, "standard": 128, "deep": 256, "research": 512}.get(
                mode,
                128,
            )
        )
        runtime = observe_model_runtime(
            model,
            sample_batch,
            targets=targets,
            loss_fn=loss_fn,
            backward=backward,
            max_modules=runtime_module_limit,
            sample_values=sample_limit,
        )
        runtime_findings = runtime.get("findings", [])
        if isinstance(runtime_findings, list):
            findings.extend(runtime_findings)

    score = 100.0
    score -= min(60.0, len(nonfinite) * 15.0)
    score -= min(50.0, len(gradients.get("nonfinite_parameters") or []) * 10.0)
    score -= min(20.0, len(norm_outliers) * 4.0)
    score -= min(20.0, len(low_rank) * 5.0)
    score -= min(20.0, int(cnn.get("dead_filters") or 0) * 1.5)
    if architecture_findings:
        architecture_cost = {
            "critical": 18.0,
            "high": 8.0,
            "medium": 3.0,
            "info": 0.0,
        }
        score -= min(
            35.0,
            sum(
                architecture_cost.get(
                    str(item.get("severity", "info")).lower(),
                    1.0,
                )
                for item in architecture_findings
            ),
        )

    if optimizer_findings:
        severity_cost = {
            "critical": 20.0,
            "high": 10.0,
            "medium": 4.0,
            "info": 0.0,
        }
        score -= min(
            40.0,
            sum(
                severity_cost.get(str(item.get("severity", "info")).lower(), 2.0)
                for item in optimizer_findings
            ),
        )

    if runtime is not None:
        runtime_summary = runtime.get("summary", {})
        if isinstance(runtime_summary, dict):
            score -= min(
                60.0,
                float(runtime_summary.get("nonfinite_activations", 0)) * 15.0
                + float(runtime_summary.get("nonfinite_gradients", 0)) * 15.0
                + float(runtime_summary.get("gradient_explosions", 0)) * 10.0
                + float(runtime_summary.get("gradient_vanishing", 0)) * 6.0
                + float(runtime_summary.get("dead_activations", 0)) * 4.0
                + float(runtime_summary.get("saturated_activations", 0)) * 4.0,
            )

    score = round(max(0.0, min(100.0, score)), 2)

    model_summary = {
        "framework": "pytorch",
        "architecture": architecture,
        "class_name": type(model).__name__,
        "parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": total_parameters - trainable_parameters,
        "estimated_parameter_bytes": total_bytes,
        "modules": len(modules),
        "module_types": dict(module_counts.most_common()),
        "dtype_parameter_counts": dict(dtype_counts),
        "parameter_diagnostics": parameter_summaries,
        "rank_diagnostics": rank,
        "gradients": gradients,
        "cnn": cnn,
        "architecture_diagnostics": architecture_details,
        "optimizer": optimizer_report,
        "runtime": runtime,
    }

    return AnalysisResult(
        {
            "dataset_id": None,
            "filename": type(model).__name__,
            "analysis_mode": mode,
            "source_kind": "model",
            "profile": {
                "shape": {
                    "rows": total_parameters,
                    "columns": len(modules),
                },
                "structure": model_summary,
            },
            "model": model_summary,
            "health": {"overall_score": score, "label": _health_label(score)},
            "ml_readiness": {"score": None, "label": "not_applicable"},
            "findings": findings,
            "artifacts_enabled": False,
            "execution": {
                "method": "bounded_model_diagnostics",
                "sampled": True,
                "resource_bounded": True,
            },
        }
    )
