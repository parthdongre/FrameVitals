"""Optimizer-state diagnostics for PyTorch-style optimizers."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from framevitals.core.beacons import beacon


def _sample_tensor(value: Any, limit: int = 4096) -> np.ndarray | None:
    if not (
        hasattr(value, "detach")
        and callable(getattr(value, "numel", None))
    ):
        return None
    try:
        flat = value.detach().reshape(-1)
        count = int(flat.numel())
        if count > limit:
            step = max(1, count // limit)
            flat = flat[::step][:limit]
        try:
            flat = flat.float()
        except Exception:
            pass
        return np.asarray(flat.cpu().numpy(), dtype=float)
    except Exception:
        return None


def inspect_optimizer(model: Any, optimizer: Any) -> dict[str, Any]:
    """Inspect optimizer coverage, group policy, and state tensors."""
    if optimizer is None:
        return {"available": False, "reason": "not_provided"}

    param_groups = getattr(optimizer, "param_groups", None)
    state = getattr(optimizer, "state", None)
    if not isinstance(param_groups, list) or state is None:
        raise TypeError("Expected a PyTorch-style optimizer with param_groups and state.")

    named_parameters = list(model.named_parameters())
    id_to_name = {id(parameter): name for name, parameter in named_parameters}
    trainable_ids = {
        id(parameter)
        for _, parameter in named_parameters
        if bool(getattr(parameter, "requires_grad", False))
    }

    assigned_ids: list[int] = []
    groups: list[dict[str, Any]] = []
    learning_rates: list[float] = []

    for index, group in enumerate(param_groups):
        params = list(group.get("params", []) or [])
        ids = [id(parameter) for parameter in params]
        assigned_ids.extend(ids)

        lr = group.get("lr")
        lr_value = None
        try:
            if lr is not None:
                lr_value = float(lr)
                if math.isfinite(lr_value):
                    learning_rates.append(lr_value)
        except (TypeError, ValueError):
            lr_value = None

        hyperparameters = {}
        for key in (
            "lr",
            "weight_decay",
            "momentum",
            "dampening",
            "eps",
            "alpha",
            "maximize",
            "amsgrad",
            "foreach",
            "capturable",
            "fused",
        ):
            if key in group and key != "params":
                value = group.get(key)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    hyperparameters[key] = value
        if "betas" in group:
            try:
                hyperparameters["betas"] = [
                    float(group["betas"][0]),
                    float(group["betas"][1]),
                ]
            except Exception:
                pass

        groups.append({
            "index": index,
            "parameter_count": len(params),
            "parameter_names": [
                id_to_name.get(id(parameter), f"<unmapped:{id(parameter)}>")
                for parameter in params[:100]
            ],
            "learning_rate": lr_value,
            "hyperparameters": hyperparameters,
        })

    assignment_counts = Counter(assigned_ids)
    duplicated = [
        id_to_name.get(parameter_id, f"<unmapped:{parameter_id}>")
        for parameter_id, count in assignment_counts.items()
        if count > 1
    ]
    assigned_set = set(assigned_ids)
    untracked = [
        id_to_name[parameter_id]
        for parameter_id in trainable_ids - assigned_set
        if parameter_id in id_to_name
    ]
    nontrainable_assigned = [
        id_to_name.get(parameter_id, f"<unmapped:{parameter_id}>")
        for parameter_id in assigned_set - trainable_ids
    ]

    state_records: list[dict[str, Any]] = []
    nonfinite_state: list[dict[str, Any]] = []
    state_bytes = 0
    step_values: list[float] = []

    try:
        state_items = list(state.items())
    except Exception:
        state_items = []

    for parameter, payload in state_items:
        parameter_name = id_to_name.get(id(parameter), f"<unmapped:{id(parameter)}>")
        if not isinstance(payload, dict):
            continue

        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    numeric = float(value)
                    if key == "step" and math.isfinite(numeric):
                        step_values.append(numeric)
                except (TypeError, ValueError):
                    pass
                continue

            sampled = _sample_tensor(value)
            if sampled is None:
                continue

            element_size = getattr(value, "element_size", None)
            numel = getattr(value, "numel", None)
            nbytes = None
            if callable(element_size) and callable(numel):
                try:
                    nbytes = int(element_size()) * int(numel())
                    state_bytes += nbytes
                except Exception:
                    nbytes = None

            finite = np.isfinite(sampled)
            record = {
                "parameter": parameter_name,
                "state": str(key),
                "shape": [int(v) for v in getattr(value, "shape", ())],
                "dtype": str(getattr(value, "dtype", "unknown")),
                "sampled_values": int(sampled.size),
                "nbytes": nbytes,
                "nan_fraction": (
                    round(float(np.isnan(sampled).mean()), 8)
                    if sampled.size else 0.0
                ),
                "inf_fraction": (
                    round(float(np.isinf(sampled).mean()), 8)
                    if sampled.size else 0.0
                ),
            }
            if finite.any():
                values = sampled[finite]
                record["mean"] = round(float(np.mean(values)), 8)
                record["std"] = round(float(np.std(values)), 8)
                record["l2_norm_sample"] = round(float(np.linalg.norm(values)), 8)

            state_records.append(record)
            if (
                float(record["nan_fraction"]) > 0
                or float(record["inf_fraction"]) > 0
            ):
                nonfinite_state.append(record)

    lr_spread = None
    positive_lrs = [value for value in learning_rates if value > 0]
    if positive_lrs:
        lr_spread = max(positive_lrs) / min(positive_lrs)

    step_range = None
    if step_values:
        step_range = {
            "min": min(step_values),
            "max": max(step_values),
            "spread": max(step_values) - min(step_values),
        }

    findings: list[dict[str, Any]] = []

    if nonfinite_state:
        findings.append(beacon(
            "model.optimizer.non_finite_state",
            "Optimizer state contains non-finite values",
            severity="critical",
            confidence=1.0,
            summary=f"{len(nonfinite_state)} optimizer-state tensors contain sampled NaN/Inf values.",
            recommendation="Stop the next update and inspect gradient scaling, learning rate, and optimizer state.",
            evidence={"state": nonfinite_state[:30]},
        ))

    if untracked:
        findings.append(beacon(
            "model.optimizer.untracked_parameters",
            "Trainable parameters are missing from optimizer groups",
            severity="high",
            confidence=1.0,
            summary=f"{len(untracked)} trainable parameters are not assigned to the optimizer.",
            recommendation="Confirm these parameters are intentionally excluded from optimization.",
            evidence={"parameters": untracked[:50]},
        ))

    if duplicated:
        findings.append(beacon(
            "model.optimizer.duplicate_parameters",
            "Parameters appear in multiple optimizer groups",
            severity="high",
            confidence=1.0,
            summary=f"{len(duplicated)} parameters are assigned more than once.",
            recommendation="Remove duplicate optimizer assignments to avoid ambiguous update policy.",
            evidence={"parameters": duplicated[:50]},
        ))

    if lr_spread is not None and lr_spread >= 100.0:
        findings.append(beacon(
            "model.optimizer.learning_rate_spread",
            "Optimizer groups use extremely different learning rates",
            severity="medium",
            confidence=0.9,
            summary=f"Maximum learning-rate ratio across groups is {lr_spread:.1f}x.",
            recommendation="Confirm the learning-rate policy is intentional for all parameter groups.",
            evidence={"learning_rates": learning_rates},
        ))

    return {
        "available": True,
        "class_name": type(optimizer).__name__,
        "parameter_groups": groups,
        "group_count": len(groups),
        "tracked_parameter_count": len(assigned_set),
        "trainable_parameter_count": len(trainable_ids),
        "untracked_trainable_parameters": untracked,
        "duplicate_parameter_assignments": duplicated,
        "nontrainable_assigned_parameters": nontrainable_assigned,
        "learning_rates": learning_rates,
        "learning_rate_spread": (
            round(float(lr_spread), 8) if lr_spread is not None else None
        ),
        "state_entry_count": len(state_items),
        "state_tensor_count": len(state_records),
        "estimated_state_bytes": state_bytes,
        "step_range": step_range,
        "state_tensors": state_records[:200],
        "findings": findings,
    }
