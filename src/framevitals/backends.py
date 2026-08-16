"""Backend routing for FrameVitals native and NumPy kernels.

The router keeps optional native/GPU dependencies behind lazy imports. Normal
installations remain NumPy-first, while a compiled ``framevitals._native``
extension can transparently accelerate compatible streaming scans.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
import os
from typing import Any, Literal

import numpy as np
import pandas as pd


BackendName = Literal["auto", "numpy", "rust"]
_VALID_BACKENDS = {"auto", "numpy", "rust"}


def native_available() -> bool:
    """Return whether the optional FrameVitals native extension is importable."""
    return find_spec("framevitals._native") is not None


def resolve_numeric_backend(requested: str | None = None) -> Literal["numpy", "rust"]:
    """Resolve the backend for exact/streaming primitives.

    ``FRAMEVITALS_BACKEND`` may set ``auto``, ``numpy`` or ``rust``. Explicit
    function arguments take precedence. Requesting Rust without the extension
    installed is an error; ``auto`` always falls back safely to NumPy.
    """
    value = (requested or os.getenv("FRAMEVITALS_BACKEND", "auto")).strip().lower()
    if value not in _VALID_BACKENDS:
        raise ValueError(
            "Unknown FrameVitals numeric backend "
            f"'{value}'. Choose from: auto, numpy, rust."
        )
    if value == "numpy":
        return "numpy"
    if value == "rust":
        if not native_available():
            raise RuntimeError(
                "The Rust backend was requested but framevitals._native is not installed."
            )
        return "rust"
    return "rust" if native_available() else "numpy"


def _float64_array(values: pd.Series | np.ndarray | list[Any]) -> np.ndarray:
    if isinstance(values, pd.Series):
        array = pd.to_numeric(values, errors="coerce").to_numpy(
            dtype="float64",
            na_value=np.nan,
        )
    else:
        array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("FrameVitals numeric kernels require one-dimensional input.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _numpy_numeric_state(array: np.ndarray) -> dict[str, Any]:
    missing = int(np.isnan(array).sum())
    infinite = int(np.isinf(array).sum())
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "backend": "numpy",
            "observations": int(array.size),
            "count": 0,
            "missing": missing,
            "infinite": infinite,
            "mean": None,
            "variance": None,
            "std": None,
            "minimum": None,
            "maximum": None,
        }

    mean = float(finite.mean())
    variance = float(finite.var(ddof=1)) if finite.size >= 2 else None
    return {
        "backend": "numpy",
        "observations": int(array.size),
        "count": int(finite.size),
        "missing": missing,
        "infinite": infinite,
        "mean": mean,
        "variance": variance,
        "std": float(np.sqrt(variance)) if variance is not None else None,
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def numeric_state(
    values: pd.Series | np.ndarray | list[Any],
    *,
    backend: BackendName | str | None = None,
) -> dict[str, Any]:
    """Calculate exact first/second-moment state through the selected backend."""
    array = _float64_array(values)
    selected = resolve_numeric_backend(backend)
    if selected == "rust":
        native = import_module("framevitals._native")
        payload = native.numeric_state_f64(array)
        return dict(payload)
    return _numpy_numeric_state(array)


def numeric_profile(
    values: pd.Series | np.ndarray | list[Any],
    *,
    backend: BackendName | str | None = None,
    stream_id: int = 0,
) -> dict[str, Any]:
    """Run the fused native numeric profile when available."""
    array = _float64_array(values)
    selected = resolve_numeric_backend(backend)
    if selected == "rust":
        native = import_module("framevitals._native")
        payload = dict(native.numeric_profile_f64(array, stream_id=int(stream_id)))
        payload["sketches_available"] = True
        return payload

    payload = _numpy_numeric_state(array)
    payload["sketches_available"] = False
    return payload


def create_numeric_accumulator(*, stream_id: int = 0):
    """Create a persistent native numeric accumulator for multi-batch scans."""
    if resolve_numeric_backend() != "rust":
        return None
    native = import_module("framevitals._native")
    return native.NumericAccumulator(stream_id=int(stream_id))


def create_string_accumulator():
    """Create a persistent native UTF-8 sketch accumulator when available."""
    if resolve_numeric_backend() != "rust":
        return None
    native = import_module("framevitals._native")
    accumulator = getattr(native, "StringAccumulator", None)
    return accumulator() if accumulator is not None else None


def backend_status() -> dict[str, Any]:
    """Return lightweight backend availability without importing native code."""
    has_native = native_available()
    eligible = ["numpy"]
    if has_native:
        eligible.append("rust")
    return {
        "selected": resolve_numeric_backend("auto"),
        "native_available": has_native,
        "environment_override": os.getenv("FRAMEVITALS_BACKEND"),
        "eligible": eligible,
    }
