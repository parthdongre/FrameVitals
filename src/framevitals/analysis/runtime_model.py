"""Bounded runtime observation for PyTorch-style neural networks.

This module observes a single caller-supplied forward pass and, optionally, an
autograd pass. Hooks are temporary and always removed. The implementation never
requires PyTorch at FrameVitals import time.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import Any

import numpy as np

from framevitals.core.beacons import beacon


def _iter_tensors(value: Any):
    if hasattr(value, "detach") and hasattr(value, "numel"):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _sample_numpy(tensor: Any, limit: int) -> np.ndarray:
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


def _tensor_observation(tensor: Any, *, sample_limit: int) -> dict[str, Any]:
    values = _sample_numpy(tensor, sample_limit)
    size = int(getattr(tensor, "numel")())
    result: dict[str, Any] = {
        "shape": [int(v) for v in getattr(tensor, "shape", ())],
        "size": size,
        "sampled_values": int(values.size),
        "dtype": str(getattr(tensor, "dtype", "unknown")),
    }
    if not values.size:
        return result

    finite = np.isfinite(values)
    finite_values = values[finite]
    result.update({
        "nan_fraction": round(float(np.isnan(values).mean()), 8),
        "inf_fraction": round(float(np.isinf(values).mean()), 8),
        "zero_fraction": round(float((values == 0).mean()), 8),
        "near_zero_fraction": round(float((np.abs(values) <= 1.0e-8).mean()), 8),
    })
    if finite_values.size:
        result.update({
            "mean": round(float(np.mean(finite_values)), 8),
            "std": round(float(np.std(finite_values)), 8),
            "min": round(float(np.min(finite_values)), 8),
            "max": round(float(np.max(finite_values)), 8),
            "abs_p99": round(float(np.quantile(np.abs(finite_values), 0.99)), 8),
            "l2_norm_sample": round(float(np.linalg.norm(finite_values)), 8),
            "rms_sample": round(
                float(np.linalg.norm(finite_values) / math.sqrt(finite_values.size)),
                8,
            ),
        })
    return result


def _module_saturation(module_name: str, values: np.ndarray) -> float | None:
    if not values.size:
        return None
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    lower = module_name.lower()
    if "sigmoid" in lower:
        return float(np.mean((finite <= 0.01) | (finite >= 0.99)))
    if "tanh" in lower:
        return float(np.mean(np.abs(finite) >= 0.99))
    return None


def _leaf_modules(model: Any) -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        try:
            children = list(module.children())
        except Exception:
            children = []
        if not children:
            output.append((name, module))

    # A model can itself be a single leaf module (for example nn.Linear).
    # named_modules then exposes only the unnamed root.
    if not output:
        output.append(("<root>", model))
    return output


def _bounded_modules(model: Any, limit: int) -> list[tuple[str, Any]]:
    modules = _leaf_modules(model)
    if len(modules) <= limit:
        return modules
    positions = np.linspace(0, len(modules) - 1, num=limit, dtype=int)
    return [modules[int(i)] for i in positions]


def _invoke_model(model: Any, inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, list):
        return model(*inputs)
    return model(inputs)


def _cuda_observation_devices(model: Any, inputs: Any) -> list[int]:
    """Only fork RNG on CUDA devices already holding input/model tensors."""
    devices: set[int] = set()
    for value in (
        list(model.parameters())
        + list(model.buffers())
        + list(_iter_tensors(inputs))
    ):
        device = getattr(value, "device", None)
        if getattr(device, "type", None) == "cuda":
            devices.add(int(device.index) if device.index is not None else 0)
    return sorted(devices)


@contextmanager
def _isolated_model_observation(model: Any, inputs: Any, torch: Any):
    """Run an observational pass in evaluation mode and restore module flags.

    Training-mode BatchNorm changes running statistics and dropout consumes
    randomness. Eval mode plus torch.random.fork_rng avoids both common side
    effects. Arbitrary custom forward implementations may still mutate other
    Python state or user-defined buffers; do not advertise full purity.
    """
    states = [(module, bool(module.training)) for module in model.modules()]
    cuda_devices = _cuda_observation_devices(model, inputs)

    with torch.random.fork_rng(devices=cuda_devices):
        try:
            model.eval()
            yield
        finally:
            # Directly restore every module flag. Calling model.train() would
            # overwrite intentionally mixed per-module training/eval states.
            for module, training in states:
                module.training = training


def _loss_value(output: Any, *, targets: Any, loss_fn: Any) -> Any:
    if loss_fn is not None:
        return loss_fn(output, targets) if targets is not None else loss_fn(output)

    tensors = list(_iter_tensors(output))
    if len(tensors) == 1 and int(tensors[0].numel()) == 1:
        return tensors[0]
    raise ValueError(
        "backward=True requires loss_fn= unless the model output is a scalar tensor."
    )


def _robust_norm_outliers(
    records: list[dict[str, Any]],
    *,
    key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable = [
        (record, float(record[key]))
        for record in records
        if isinstance(record.get(key), (int, float)) and float(record[key]) > 0
    ]
    if len(usable) < 4:
        return [], []

    logs = np.asarray([math.log10(value) for _, value in usable], dtype=float)
    median = float(np.median(logs))
    mad = float(np.median(np.abs(logs - median)))
    spread = max(mad, 0.20)
    high = median + 6.0 * spread
    low = median - 6.0 * spread

    exploding = [
        record
        for (record, _), value in zip(usable, logs, strict=True)
        if value > high
    ]
    vanishing = [
        record
        for (record, _), value in zip(usable, logs, strict=True)
        if value < low
    ]
    return exploding, vanishing


def observe_model_runtime(
    model: Any,
    inputs: Any,
    *,
    targets: Any = None,
    loss_fn: Any = None,
    backward: bool = False,
    max_modules: int = 128,
    sample_values: int = 8_192,
) -> dict[str, Any]:
    """Observe one bounded forward/backward pass without retaining hooks.

    When backward is true, gradients are obtained with torch.autograd.grad
    instead of loss.backward, so existing parameter.grad buffers are not
    accumulated or zeroed by FrameVitals.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Runtime model diagnostics require PyTorch in the caller environment."
        ) from exc

    if inputs is None:
        raise ValueError("Runtime model diagnostics require sample_batch=/inputs.")

    selected = _bounded_modules(model, max(1, int(max_modules)))
    total_leaf_modules = len(_leaf_modules(model))
    activations: dict[str, dict[str, Any]] = {}
    module_gradients: dict[str, dict[str, Any]] = {}
    handles: list[Any] = []

    def forward_hook(name: str, module_type: str):
        def hook(_module, _args, output):
            tensors = list(_iter_tensors(output))
            if not tensors:
                return
            tensor = tensors[0]
            try:
                observation = _tensor_observation(tensor, sample_limit=sample_values)
                sampled = _sample_numpy(tensor, sample_values)
            except Exception:
                return
            observation["module_type"] = module_type
            saturation = _module_saturation(module_type, sampled)
            if saturation is not None:
                observation["saturation_fraction"] = round(saturation, 8)
            activations[name] = observation

            if backward and bool(getattr(tensor, "requires_grad", False)):
                register_hook = getattr(tensor, "register_hook", None)
                if callable(register_hook):
                    def capture_gradient(gradient):
                        try:
                            grad_observation = _tensor_observation(
                                gradient,
                                sample_limit=max(1_024, sample_values // 2),
                            )
                        except Exception:
                            return gradient
                        grad_observation["module_type"] = module_type
                        module_gradients[name] = grad_observation
                        return gradient

                    handles.append(register_hook(capture_gradient))
        return hook

    parameter_gradients: list[dict[str, Any]] = []
    loss_scalar: float | None = None
    backward_ms: float | None = None

    try:
        with _isolated_model_observation(model, inputs, torch):
            for name, module in selected:
                module_type = type(module).__name__
                handles.append(module.register_forward_hook(forward_hook(name, module_type)))
    
            started = perf_counter()
            context = nullcontext() if backward else torch.no_grad()
            with context:
                output = _invoke_model(model, inputs)
            forward_ms = (perf_counter() - started) * 1000.0
    
            if backward:
                loss = _loss_value(output, targets=targets, loss_fn=loss_fn)
                if not hasattr(loss, "numel") or int(loss.numel()) != 1:
                    raise ValueError("loss_fn must return a scalar tensor for backward diagnostics.")
                loss_scalar = float(loss.detach().cpu().item())
    
                named_trainable = [
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if bool(getattr(parameter, "requires_grad", False))
                ]
                params = [parameter for _, parameter in named_trainable]
    
                started = perf_counter()
                grads = torch.autograd.grad(
                    loss,
                    params,
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )
                backward_ms = (perf_counter() - started) * 1000.0
    
                for (name, _parameter), grad in zip(named_trainable, grads, strict=True):
                    if grad is None:
                        parameter_gradients.append({
                            "name": name,
                            "available": False,
                            "reason": "unused",
                        })
                        continue
                    try:
                        observation = _tensor_observation(
                            grad,
                            sample_limit=max(1_024, sample_values // 2),
                        )
                    except Exception:
                        continue
                    parameter_gradients.append({
                        "name": name,
                        "available": True,
                        **observation,
                    })
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    activation_records = [
        {"name": name, **payload}
        for name, payload in activations.items()
    ]
    module_gradient_records = [
        {"name": name, **payload}
        for name, payload in module_gradients.items()
    ]

    activation_exploding, activation_vanishing = _robust_norm_outliers(
        activation_records,
        key="rms_sample",
    )
    gradient_exploding, gradient_vanishing = _robust_norm_outliers(
        module_gradient_records,
        key="rms_sample",
    )

    dead_activations = [
        record
        for record in activation_records
        if (
            "relu" in str(record.get("module_type", "")).lower()
            and float(record.get("zero_fraction") or 0.0) >= 0.95
        )
        or (
            float(record.get("near_zero_fraction") or 0.0) >= 0.95
            and int(record.get("size") or 0) >= 16
        )
    ]
    saturated_activations = [
        record
        for record in activation_records
        if float(record.get("saturation_fraction") or 0.0) >= 0.80
    ]
    nonfinite_activations = [
        record
        for record in activation_records
        if float(record.get("nan_fraction") or 0.0) > 0
        or float(record.get("inf_fraction") or 0.0) > 0
    ]
    nonfinite_gradients = [
        record
        for record in module_gradient_records
        if float(record.get("nan_fraction") or 0.0) > 0
        or float(record.get("inf_fraction") or 0.0) > 0
    ]

    findings: list[dict[str, Any]] = []

    if nonfinite_activations:
        findings.append(beacon(
            "model.runtime.non_finite_activations",
            "Non-finite activations appeared during the observed forward pass",
            severity="critical",
            confidence=1.0,
            summary=f"{len(nonfinite_activations)} observed modules emitted NaN/Inf values.",
            recommendation="Trace the first affected layer and inspect inputs, normalization, and numerics.",
            evidence={"modules": [r["name"] for r in nonfinite_activations[:30]]},
        ))

    if dead_activations:
        findings.append(beacon(
            "model.runtime.dead_activations",
            "Dead or collapsed activations were detected",
            severity="high",
            confidence=0.95,
            summary=f"{len(dead_activations)} observed modules produced almost no activation variation.",
            recommendation="Inspect initialization, ReLU death, normalization, and upstream signal flow.",
            evidence={"modules": [r["name"] for r in dead_activations[:30]]},
        ))

    if saturated_activations:
        findings.append(beacon(
            "model.runtime.saturation",
            "Activation saturation was detected",
            severity="high",
            confidence=0.95,
            summary=f"{len(saturated_activations)} sigmoid/tanh modules were at least 80% saturated.",
            recommendation="Review input scaling, initialization, and recurrent/activation dynamics.",
            evidence={"modules": [r["name"] for r in saturated_activations[:30]]},
        ))

    if activation_exploding:
        findings.append(beacon(
            "model.runtime.activation_explosion",
            "Activation magnitude outliers were detected",
            severity="high",
            confidence=0.9,
            summary=f"{len(activation_exploding)} modules have robustly extreme activation norms.",
            recommendation="Inspect residual scaling, normalization, initialization, and unstable upstream layers.",
            evidence={"modules": [r["name"] for r in activation_exploding[:30]]},
        ))

    if activation_vanishing:
        findings.append(beacon(
            "model.runtime.activation_vanishing",
            "Very weak activation flow was detected",
            severity="medium",
            confidence=0.85,
            summary=f"{len(activation_vanishing)} modules have robustly tiny activation norms.",
            recommendation="Inspect depth, normalization, initialization, and activation choice.",
            evidence={"modules": [r["name"] for r in activation_vanishing[:30]]},
        ))

    if nonfinite_gradients:
        findings.append(beacon(
            "model.runtime.non_finite_gradients",
            "Non-finite gradients appeared during the observed backward pass",
            severity="critical",
            confidence=1.0,
            summary=f"{len(nonfinite_gradients)} observed modules received NaN/Inf gradients.",
            recommendation="Inspect loss scaling, learning rate, clipping, and the earliest unstable layer.",
            evidence={"modules": [r["name"] for r in nonfinite_gradients[:30]]},
        ))

    if gradient_exploding:
        findings.append(beacon(
            "model.runtime.gradient_explosion",
            "Exploding gradient flow was detected",
            severity="critical",
            confidence=0.95,
            summary=f"{len(gradient_exploding)} modules have robustly extreme gradient norms.",
            recommendation="Inspect learning rate, normalization, residual scaling, and gradient clipping.",
            evidence={"modules": [r["name"] for r in gradient_exploding[:30]]},
        ))

    if gradient_vanishing:
        findings.append(beacon(
            "model.runtime.gradient_vanishing",
            "Vanishing gradient flow was detected",
            severity="high",
            confidence=0.9,
            summary=f"{len(gradient_vanishing)} modules have robustly tiny gradient norms.",
            recommendation="Inspect depth, activations, normalization, skip connections, and initialization.",
            evidence={"modules": [r["name"] for r in gradient_vanishing[:30]]},
        ))

    parameter_nonfinite = [
        record
        for record in parameter_gradients
        if record.get("available")
        and (
            float(record.get("nan_fraction") or 0.0) > 0
            or float(record.get("inf_fraction") or 0.0) > 0
        )
    ]
    parameter_zero = [
        record
        for record in parameter_gradients
        if record.get("available")
        and float(record.get("near_zero_fraction") or 0.0) >= 0.999
    ]

    if parameter_nonfinite:
        findings.append(beacon(
            "model.runtime.parameter_gradient_non_finite",
            "Parameter gradients contain non-finite values",
            severity="critical",
            confidence=1.0,
            summary=f"{len(parameter_nonfinite)} trainable parameter tensors contain NaN/Inf gradients.",
            recommendation="Stop the update and inspect loss scaling, numerics, and the first affected parameter group.",
            evidence={"parameters": [r["name"] for r in parameter_nonfinite[:30]]},
        ))

    if parameter_zero:
        findings.append(beacon(
            "model.runtime.zero_parameter_gradients",
            "Some trainable parameters received effectively zero gradients",
            severity="medium",
            confidence=0.9,
            summary=f"{len(parameter_zero)} parameter tensors were effectively zero-gradient in the observed pass.",
            recommendation="Confirm the computation path, frozen logic, detached tensors, and loss connectivity.",
            evidence={"parameters": [r["name"] for r in parameter_zero[:30]]},
        ))

    return {
        "available": True,
        "observation_mode": "temporary_eval",
        "training_flags_restored": True,
        "torch_rng_restored": True,
        "model_buffer_purity_guaranteed": False,
        "observed_modules": len(selected),
        "total_leaf_modules": total_leaf_modules,
        "module_sampling": len(selected) < total_leaf_modules,
        "forward_ms": round(forward_ms, 3),
        "backward_ms": round(backward_ms, 3) if backward_ms is not None else None,
        "loss": loss_scalar,
        "backward": bool(backward),
        "activations": activation_records,
        "module_gradients": module_gradient_records,
        "parameter_gradients": parameter_gradients,
        "summary": {
            "dead_activations": len(dead_activations),
            "saturated_activations": len(saturated_activations),
            "activation_explosions": len(activation_exploding),
            "activation_vanishing": len(activation_vanishing),
            "nonfinite_activations": len(nonfinite_activations),
            "gradient_explosions": len(gradient_exploding),
            "gradient_vanishing": len(gradient_vanishing),
            "nonfinite_gradients": len(nonfinite_gradients) + len(parameter_nonfinite),
            "zero_parameter_gradients": len(parameter_zero),
        },
        "findings": findings,
    }
