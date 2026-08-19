import framevitals as fv
import framevitals.acceleration as acceleration


def test_cupy_recommendation_is_platform_and_cuda_aware():
    assert acceleration._recommend_cupy_package(
        "Linux", "13.0", has_nvidia=True
    ) == "cupy-cuda13x[ctk]"
    assert acceleration._recommend_cupy_package(
        "Windows", "12.8", has_nvidia=True
    ) == "cupy-cuda12x[ctk]"
    assert acceleration._recommend_cupy_package(
        "Darwin", "13.0", has_nvidia=True
    ) is None
    assert acceleration._recommend_cupy_package(
        "Linux", None, has_nvidia=True
    ) is None
    assert acceleration._recommend_cupy_package(
        "Linux", "13.0", has_nvidia=False
    ) is None


def test_nvidia_smi_parser_is_best_effort(monkeypatch):
    def fake_smi(arguments):
        if arguments:
            return "NVIDIA Test GPU, 24576, 999.1"
        return "NVIDIA-SMI ... CUDA Version: 13.0"

    monkeypatch.setattr(acceleration, "_run_nvidia_smi", fake_smi)

    devices, cuda = acceleration._detect_nvidia()

    assert cuda == "13.0"
    assert len(devices) == 1
    assert devices[0].name == "NVIDIA Test GPU"
    assert devices[0].memory_total_mb == 24576
    assert devices[0].driver_version == "999.1"


def test_system_info_probe_can_run_without_gpu_initialization():
    result = fv.system_info(probe_gpu=False)

    assert result["platform"]
    assert result["architecture"]
    assert result["default_cpu_backend"] in {"numpy", "rust"}
    assert result["gpu_acceleration"]["automatic_install_performed"] is False
    assert isinstance(result["eligible_backends"], list)


def test_gpu_install_is_only_recommended_not_performed(monkeypatch):
    monkeypatch.setattr(acceleration.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        acceleration,
        "_detect_nvidia",
        lambda: ((acceleration.GpuDevice("GPU", 16384, "1"),), "13.0"),
    )
    monkeypatch.setattr(
        acceleration,
        "_probe_cupy",
        lambda: (False, False, None),
    )

    result = acceleration.system_info(probe_gpu=True)

    assert result["gpu_acceleration"]["available"] is False
    assert result["gpu_acceleration"]["installable"] is True
    assert result["gpu_acceleration"]["recommended_package"] == "cupy-cuda13x[ctk]"
    assert result["gpu_acceleration"]["automatic_install_performed"] is False
