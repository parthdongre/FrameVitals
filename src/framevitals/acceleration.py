"""Hardware and optional acceleration discovery for FrameVitals.

Discovery is intentionally read-only: importing or calling this module never
installs packages and never makes CUDA a hard dependency. The future planner can
consume this structured capability report when selecting per-operation backends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
import os
import platform
import re
import shutil
import subprocess
from typing import Any


_CUDA_VERSION_RE = re.compile(r"CUDA Version:\s*([0-9]+)(?:\.([0-9]+))?")


@dataclass(frozen=True, slots=True)
class GpuDevice:
    name: str
    memory_total_mb: int | None = None
    driver_version: str | None = None


@dataclass(frozen=True, slots=True)
class SystemCapabilities:
    platform: str
    architecture: str
    cpu_count: int | None
    memory_total_bytes: int | None
    native_module_available: bool
    nvidia_driver_available: bool
    cuda_compatibility: str | None
    gpu_devices: tuple[GpuDevice, ...] = field(default_factory=tuple)
    cupy_installed: bool = False
    cupy_usable: bool = False
    cupy_error: str | None = None
    recommended_cupy_package: str | None = None
    default_cpu_backend: str = "numpy"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpu_devices"] = [asdict(device) for device in self.gpu_devices]
        payload["eligible_backends"] = [self.default_cpu_backend]
        if self.cupy_usable:
            payload["eligible_backends"].append("cupy")
        return payload


def _total_memory_bytes() -> int | None:
    """Best-effort physical-memory detection using only the standard library."""
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
            return None

        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * physical_pages)
    except (AttributeError, OSError, ValueError):
        return None


def _run_nvidia_smi(arguments: list[str]) -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _detect_nvidia() -> tuple[tuple[GpuDevice, ...], str | None]:
    query = _run_nvidia_smi([
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not query:
        return (), None

    devices: list[GpuDevice] = []
    for line in query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        memory = None
        if len(parts) >= 2:
            try:
                memory = int(float(parts[1]))
            except ValueError:
                memory = None
        devices.append(
            GpuDevice(
                name=parts[0],
                memory_total_mb=memory,
                driver_version=parts[2] if len(parts) >= 3 and parts[2] else None,
            )
        )

    banner = _run_nvidia_smi([])
    cuda_compatibility = None
    if banner:
        match = _CUDA_VERSION_RE.search(banner)
        if match:
            major = match.group(1)
            minor = match.group(2) or "0"
            cuda_compatibility = f"{major}.{minor}"

    return tuple(devices), cuda_compatibility


def _recommend_cupy_package(
    system: str,
    cuda_compatibility: str | None,
    *,
    has_nvidia: bool,
) -> str | None:
    """Return the current CuPy wheel family when the environment is eligible."""
    if not has_nvidia or system not in {"Linux", "Windows"}:
        return None
    if not cuda_compatibility:
        return None

    try:
        major = int(cuda_compatibility.split(".", 1)[0])
    except ValueError:
        return None

    if major >= 13:
        return "cupy-cuda13x[ctk]"
    if major == 12:
        return "cupy-cuda12x[ctk]"
    return None


def _probe_cupy() -> tuple[bool, bool, str | None]:
    installed = find_spec("cupy") is not None
    if not installed:
        return False, False, None

    try:
        import cupy

        device_count = int(cupy.cuda.runtime.getDeviceCount())
        return True, device_count > 0, None if device_count > 0 else "No CUDA devices found."
    except Exception as exc:  # noqa: BLE001 - capability probe must fail soft
        return True, False, f"{type(exc).__name__}: {exc}"


def detect_system_capabilities(*, probe_gpu: bool = True) -> SystemCapabilities:
    """Inspect CPU/native/CUDA capabilities without installing anything."""
    system = platform.system() or "Unknown"
    architecture = platform.machine() or "unknown"
    native_available = find_spec("framevitals._native") is not None

    devices: tuple[GpuDevice, ...] = ()
    cuda_compatibility = None
    if probe_gpu and system != "Darwin":
        devices, cuda_compatibility = _detect_nvidia()

    if probe_gpu:
        cupy_installed, cupy_usable, cupy_error = _probe_cupy()
    else:
        cupy_installed = find_spec("cupy") is not None
        cupy_usable = False
        cupy_error = None

    recommendation = _recommend_cupy_package(
        system,
        cuda_compatibility,
        has_nvidia=bool(devices),
    )

    return SystemCapabilities(
        platform=system,
        architecture=architecture,
        cpu_count=os.cpu_count(),
        memory_total_bytes=_total_memory_bytes(),
        native_module_available=native_available,
        nvidia_driver_available=bool(devices),
        cuda_compatibility=cuda_compatibility,
        gpu_devices=devices,
        cupy_installed=cupy_installed,
        cupy_usable=cupy_usable,
        cupy_error=cupy_error,
        recommended_cupy_package=recommendation,
        default_cpu_backend="rust" if native_available else "numpy",
    )


def system_info(*, probe_gpu: bool = True) -> dict[str, Any]:
    """Return a JSON-safe report suitable for API/CLI/TUI diagnostics."""
    capabilities = detect_system_capabilities(probe_gpu=probe_gpu)
    payload = capabilities.to_dict()
    payload["gpu_acceleration"] = {
        "available": capabilities.cupy_usable,
        "installable": bool(
            capabilities.nvidia_driver_available
            and capabilities.recommended_cupy_package
            and not capabilities.cupy_installed
        ),
        "recommended_package": capabilities.recommended_cupy_package,
        "automatic_install_performed": False,
    }
    return payload
