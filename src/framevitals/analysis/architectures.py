"""Architecture-aware diagnostics for deep-learning models."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from framevitals.core.beacons import beacon


def _sample_numpy(tensor: Any, limit: int = 100_000) -> np.ndarray:
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
    return np.asarray(flat.cpu().numpy(), dtype=float)


def _row_norms(tensor: Any, max_rows: int = 4096) -> np.ndarray:
    detached = tensor.detach().float()
    if detached.ndim < 2:
        return np.array([], dtype=float)
    rows = int(detached.shape[0])
    if rows > max_rows:
        step = max(1, rows // max_rows)
        detached = detached[::step][:max_rows]
    matrix = detached.reshape(int(detached.shape[0]), -1)
    return np.asarray(matrix.norm(dim=1).cpu().numpy(), dtype=float)


def _coefficient_of_variation(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    mean = float(np.mean(np.abs(finite)))
    if mean <= 1.0e-12:
        return None
    return float(np.std(finite) / mean)


def _qkv_parameter_groups(named_parameters: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[str, dict[str, tuple[str, Any]]] = defaultdict(dict)
    suffixes = {
        "q_proj.weight": "q",
        "k_proj.weight": "k",
        "v_proj.weight": "v",
        "query.weight": "q",
        "key.weight": "k",
        "value.weight": "v",
    }
    for name, parameter in named_parameters:
        lowered = name.lower()
        for suffix, key in suffixes.items():
            if lowered.endswith(suffix):
                prefix = name[: -len(suffix)].rstrip(".")
                groups[prefix][key] = (name, parameter)
                break

    output: list[dict[str, Any]] = []
    for prefix, entries in sorted(groups.items()):
        if not {"q", "k", "v"}.issubset(entries):
            continue
        norms: dict[str, float] = {}
        for key in ("q", "k", "v"):
            values = _sample_numpy(entries[key][1], limit=50_000)
            finite = values[np.isfinite(values)]
            norms[key] = float(np.linalg.norm(finite)) if finite.size else 0.0
        nonzero = [value for value in norms.values() if value > 0]
        ratio = max(nonzero) / min(nonzero) if len(nonzero) >= 2 else None
        output.append({
            "prefix": prefix,
            "q_norm": round(norms["q"], 8),
            "k_norm": round(norms["k"], 8),
            "v_norm": round(norms["v"], 8),
            "max_min_norm_ratio": round(float(ratio), 6) if ratio is not None else None,
        })
    return output


def _multihead_attention_summary(modules: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, module in modules.items():
        if type(module).__name__ != "MultiheadAttention":
            continue
        heads = int(getattr(module, "num_heads", 0) or 0)
        embed_dim = int(getattr(module, "embed_dim", 0) or 0)
        item: dict[str, Any] = {
            "name": name or "<root>",
            "heads": heads,
            "embed_dim": embed_dim,
            "head_dim": int(embed_dim // heads) if heads else None,
        }

        weight = getattr(module, "in_proj_weight", None)
        if (
            weight is not None
            and heads > 0
            and embed_dim > 0
            and tuple(weight.shape) == (3 * embed_dim, embed_dim)
        ):
            try:
                matrix = weight.detach().float()
                q, k, v = matrix.chunk(3, dim=0)
                head_dim = embed_dim // heads
                head_stats = {}
                for label, block in (("q", q), ("k", k), ("v", v)):
                    reshaped = block.reshape(heads, head_dim, embed_dim)
                    norms = np.asarray(
                        reshaped.reshape(heads, -1).norm(dim=1).cpu().numpy(),
                        dtype=float,
                    )
                    head_stats[label] = {
                        "min_norm": round(float(np.min(norms)), 8),
                        "median_norm": round(float(np.median(norms)), 8),
                        "max_norm": round(float(np.max(norms)), 8),
                        "cv": (
                            round(float(_coefficient_of_variation(norms)), 6)
                            if _coefficient_of_variation(norms) is not None
                            else None
                        ),
                    }
                item["head_projection_norms"] = head_stats
            except Exception:
                pass
        results.append(item)
    return results


def _embedding_summary(modules: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, module in modules.items():
        if type(module).__name__ != "Embedding":
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        try:
            norms = _row_norms(weight)
        except Exception:
            continue
        if not norms.size:
            continue
        median = float(np.median(norms))
        near_zero = int(np.sum(norms <= max(1.0e-10, median * 1.0e-3)))
        results.append({
            "name": name or "<root>",
            "rows_sampled": int(norms.size),
            "embedding_dim": int(weight.shape[-1]) if len(weight.shape) >= 2 else None,
            "median_row_norm": round(median, 8),
            "min_row_norm": round(float(np.min(norms)), 8),
            "max_row_norm": round(float(np.max(norms)), 8),
            "near_zero_rows": near_zero,
            "row_norm_cv": (
                round(float(_coefficient_of_variation(norms)), 6)
                if _coefficient_of_variation(norms) is not None
                else None
            ),
        })
    return results


def _normalization_summary(modules: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name, module in modules.items():
        module_type = type(module).__name__
        if module_type not in {"LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        try:
            values = _sample_numpy(weight, limit=20_000)
        except Exception:
            continue
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        output.append({
            "name": name or "<root>",
            "module": module_type,
            "mean_scale": round(float(np.mean(finite)), 8),
            "std_scale": round(float(np.std(finite)), 8),
            "min_scale": round(float(np.min(finite)), 8),
            "max_scale": round(float(np.max(finite)), 8),
            "near_zero_fraction": round(float(np.mean(np.abs(finite) <= 1.0e-4)), 8),
        })
    return output


def analyze_transformer_architecture(
    named_parameters: list[tuple[str, Any]],
    modules: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inspect attention projections, heads, embeddings, and normalization."""
    qkv = _qkv_parameter_groups(named_parameters)
    attention = _multihead_attention_summary(modules)
    embeddings = _embedding_summary(modules)
    normalization = _normalization_summary(modules)
    findings: list[dict[str, Any]] = []

    imbalanced_qkv = [
        item for item in qkv
        if isinstance(item.get("max_min_norm_ratio"), (int, float))
        and float(item["max_min_norm_ratio"]) >= 4.0
    ]
    if imbalanced_qkv:
        findings.append(beacon(
            "model.transformer.qkv_imbalance",
            "Q/K/V projection magnitudes are strongly imbalanced",
            severity="high",
            confidence=0.9,
            summary=f"{len(imbalanced_qkv)} attention projection groups exceed a 4x norm ratio.",
            recommendation="Inspect attention initialization, fine-tuning updates, and projection scaling.",
            evidence={"groups": imbalanced_qkv[:20]},
        ))

    imbalanced_heads = []
    for item in attention:
        stats = item.get("head_projection_norms", {})
        if not isinstance(stats, dict):
            continue
        max_cv = max(
            (
                float(block.get("cv"))
                for block in stats.values()
                if isinstance(block, dict)
                and isinstance(block.get("cv"), (int, float))
            ),
            default=0.0,
        )
        if max_cv >= 0.75:
            imbalanced_heads.append({**item, "max_head_norm_cv": round(max_cv, 6)})
    if imbalanced_heads:
        findings.append(beacon(
            "model.transformer.head_imbalance",
            "Attention-head projection magnitudes are highly uneven",
            severity="medium",
            confidence=0.85,
            summary=f"{len(imbalanced_heads)} attention modules have high head-to-head norm variation.",
            recommendation="Inspect head collapse, pruning effects, or unstable fine-tuning.",
            evidence={"modules": imbalanced_heads[:20]},
        ))

    dead_embeddings = [
        item for item in embeddings
        if int(item.get("near_zero_rows") or 0) > 0
        and int(item.get("rows_sampled") or 0) > 0
        and int(item["near_zero_rows"]) / int(item["rows_sampled"]) >= 0.05
    ]
    if dead_embeddings:
        findings.append(beacon(
            "model.transformer.embedding_collapse",
            "A meaningful share of sampled embedding rows are near zero",
            severity="medium",
            confidence=0.9,
            summary=f"{len(dead_embeddings)} embedding tables show at least 5% near-zero sampled rows.",
            recommendation="Check padding policy, frozen embeddings, pruning, and representation collapse.",
            evidence={"embeddings": dead_embeddings[:20]},
        ))

    collapsed_norms = [
        item for item in normalization
        if float(item.get("near_zero_fraction") or 0.0) >= 0.10
    ]
    if collapsed_norms:
        findings.append(beacon(
            "model.transformer.normalization_collapse",
            "Normalization scale parameters contain many near-zero values",
            severity="medium",
            confidence=0.9,
            summary=f"{len(collapsed_norms)} normalization modules have at least 10% near-zero scales.",
            recommendation="Inspect over-regularization, pruning, and fine-tuning stability.",
            evidence={"modules": collapsed_norms[:20]},
        ))

    return {
        "available": bool(qkv or attention or embeddings or normalization),
        "qkv_groups": qkv,
        "attention_modules": attention,
        "embeddings": embeddings,
        "normalization": normalization,
    }, findings


def _spectral_radius(matrix: np.ndarray) -> float | None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.shape[0] > 512:
        return None
    try:
        eigvals = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None
    if not eigvals.size:
        return None
    return float(np.max(np.abs(eigvals)))


def analyze_recurrent_architecture(
    modules: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inspect recurrent gate matrices and bounded spectral behavior."""
    recurrent: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    gate_counts = {"LSTM": 4, "GRU": 3, "RNN": 1}
    for name, module in modules.items():
        module_type = type(module).__name__
        if module_type not in gate_counts:
            continue
        hidden_size = int(getattr(module, "hidden_size", 0) or 0)
        num_layers = int(getattr(module, "num_layers", 1) or 1)
        bidirectional = bool(getattr(module, "bidirectional", False))
        item: dict[str, Any] = {
            "name": name or "<root>",
            "module": module_type,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "layers": [],
        }

        directions = 2 if bidirectional else 1
        gate_count = gate_counts[module_type]
        for layer in range(num_layers):
            for direction in range(directions):
                suffix = f"_l{layer}" + ("_reverse" if direction else "")
                weight = getattr(module, f"weight_hh{suffix}", None)
                if weight is None or hidden_size <= 0:
                    continue
                try:
                    matrix = np.asarray(weight.detach().float().cpu().numpy(), dtype=float)
                except Exception:
                    continue
                if matrix.shape[0] != gate_count * hidden_size:
                    continue
                gates = np.split(matrix, gate_count, axis=0)
                gate_norms = [float(np.linalg.norm(gate)) for gate in gates]
                radii = [_spectral_radius(gate) for gate in gates]
                finite_radii = [value for value in radii if value is not None and math.isfinite(value)]
                item["layers"].append({
                    "layer": layer,
                    "direction": "reverse" if direction else "forward",
                    "gate_norms": [round(value, 8) for value in gate_norms],
                    "gate_norm_ratio": (
                        round(max(gate_norms) / max(min(gate_norms), 1.0e-12), 6)
                        if gate_norms else None
                    ),
                    "spectral_radii": [
                        round(value, 8) if value is not None else None
                        for value in radii
                    ],
                    "max_spectral_radius": (
                        round(max(finite_radii), 8) if finite_radii else None
                    ),
                })
        recurrent.append(item)

    unstable = []
    collapsed = []
    imbalanced = []
    for module in recurrent:
        for layer in module["layers"]:
            radius = layer.get("max_spectral_radius")
            ratio = layer.get("gate_norm_ratio")
            if isinstance(radius, (int, float)) and float(radius) >= 2.0:
                unstable.append({"module": module["name"], **layer})
            if isinstance(radius, (int, float)) and float(radius) <= 0.05:
                collapsed.append({"module": module["name"], **layer})
            if isinstance(ratio, (int, float)) and float(ratio) >= 5.0:
                imbalanced.append({"module": module["name"], **layer})

    if unstable:
        findings.append(beacon(
            "model.recurrent.spectral_instability",
            "Recurrent gate matrices have very large spectral radii",
            severity="high",
            confidence=0.9,
            summary=f"{len(unstable)} recurrent layers exceed a spectral radius of 2.",
            recommendation="Inspect recurrent initialization, gradient clipping, and training stability.",
            evidence={"layers": unstable[:20]},
        ))
    if collapsed:
        findings.append(beacon(
            "model.recurrent.spectral_collapse",
            "Recurrent gate matrices have near-zero spectral radii",
            severity="medium",
            confidence=0.85,
            summary=f"{len(collapsed)} recurrent layers have spectral radius <= 0.05.",
            recommendation="Inspect vanishing recurrent dynamics or excessive regularization.",
            evidence={"layers": collapsed[:20]},
        ))
    if imbalanced:
        findings.append(beacon(
            "model.recurrent.gate_imbalance",
            "Recurrent gate weight magnitudes are strongly imbalanced",
            severity="medium",
            confidence=0.85,
            summary=f"{len(imbalanced)} recurrent layers exceed a 5x gate-norm ratio.",
            recommendation="Inspect gate initialization and training updates.",
            evidence={"layers": imbalanced[:20]},
        ))

    return {"available": bool(recurrent), "modules": recurrent}, findings


def analyze_architecture(
    architecture: str,
    named_parameters: list[tuple[str, Any]],
    modules: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Dispatch architecture-specific static diagnostics."""
    if architecture == "transformer":
        report, findings = analyze_transformer_architecture(named_parameters, modules)
        return {"transformer": report}, findings
    if architecture == "recurrent":
        report, findings = analyze_recurrent_architecture(modules)
        return {"recurrent": report}, findings
    return {}, []
