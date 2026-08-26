"""Runtime configuration for FrameVitals analysis.

The configuration layer controls both analysis depth/resources and optional
pipeline modules. Defaults preserve historical behaviour; users can explicitly
disable expensive or irrelevant modules and cap expensive execution work without
maintaining a second configuration system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import tomllib
from typing import Any, Mapping


VALID_MODES = {"quick", "standard", "deep", "research"}
VALID_MODULES = {
    "quality_diagnostics",
    "deep_statistics",
    "anomaly_detection",
    "time_series",
    "text_profile",
    "target_intelligence",
    "modeling",
    "explainability",
    "cleaning",
    "charts",
    "ai",
}

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {"mode": "quick", "workers": 2, "artifacts": False},
    "standard": {"mode": "standard", "workers": 4, "artifacts": False},
    "deep": {"mode": "deep", "workers": 4, "artifacts": False},
    "research": {"mode": "research", "workers": 4, "artifacts": False},
    # Exhaustive is the forward-looking name for the deepest built-in policy.
    # Keep ``research`` as a compatibility preset/mode throughout the 0.x series.
    "exhaustive": {"mode": "research", "workers": 4, "artifacts": False},
    "ci": {
        "mode": "standard",
        "workers": 2,
        "artifacts": False,
        "disabled_modules": ("modeling", "explainability", "charts", "ai"),
    },
}

_RESOURCE_KEYS = (
    "max_sample_rows",
    "max_relationship_pairs",
    "max_memory_heavy_parallelism",
    "max_streaming_profile_columns",
)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Resolved configuration consumed by the analysis pipeline.

    Resource fields are hard upper bounds. They may reduce a mode's adaptive
    execution budget but never force FrameVitals to perform more work than the
    mode would normally allow.
    """

    mode: str = "standard"
    target: str | None = None
    artifacts: bool = False
    workers: int = 4
    disabled_modules: tuple[str, ...] = ()
    max_sample_rows: int | None = None
    max_relationship_pairs: int | None = None
    max_memory_heavy_parallelism: int | None = None
    max_streaming_profile_columns: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid analysis mode '{self.mode}'. "
                f"Choose from: {', '.join(sorted(VALID_MODES))}"
            )
        if self.workers < 1:
            raise ValueError("workers must be at least 1.")

        for name in _RESOURCE_KEYS:
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1 when provided.")

        modules = tuple(dict.fromkeys(self.disabled_modules))
        unknown = sorted(set(modules) - VALID_MODULES)
        if unknown:
            raise ValueError(
                "Unknown FrameVitals module(s): "
                f"{', '.join(unknown)}. Choose from: {', '.join(sorted(VALID_MODULES))}"
            )
        object.__setattr__(self, "disabled_modules", modules)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Preserve the 0.2.x serialized config shape unless a 0.3 resource cap
        # is explicitly configured. This keeps existing integrations stable.
        for name in _RESOURCE_KEYS:
            if payload.get(name) is None:
                payload.pop(name, None)
        return payload

    def module_enabled(self, name: str) -> bool:
        if name not in VALID_MODULES:
            raise ValueError(f"Unknown FrameVitals module: {name}")
        return name not in self.disabled_modules

    def execution_policy(self) -> dict[str, int | None]:
        """Return the resource caps understood by the adaptive execution layer."""
        return {name: getattr(self, name) for name in _RESOURCE_KEYS}


ConfigInput = AnalysisConfig | Mapping[str, Any] | str | Path | None


_ENV_VALUE_KEYS = {
    "FRAMEVITALS_MODE": "mode",
    "FRAMEVITALS_TARGET": "target",
    "FRAMEVITALS_ARTIFACTS": "artifacts",
    "FRAMEVITALS_WORKERS": "workers",
    "FRAMEVITALS_DISABLED_MODULES": "disabled_modules",
    "FRAMEVITALS_MAX_SAMPLE_ROWS": "max_sample_rows",
    "FRAMEVITALS_MAX_RELATIONSHIP_PAIRS": "max_relationship_pairs",
    "FRAMEVITALS_MAX_MEMORY_HEAVY_PARALLELISM": "max_memory_heavy_parallelism",
    "FRAMEVITALS_MAX_STREAMING_PROFILE_COLUMNS": "max_streaming_profile_columns",
}


def available_presets() -> tuple[str, ...]:
    """Return built-in preset names in deterministic order."""
    return tuple(PRESETS)


def available_modules() -> tuple[str, ...]:
    """Return configurable execution module names in deterministic order."""
    return tuple(sorted(VALID_MODULES))


def _read_toml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"FrameVitals config not found: {source}")
    if not source.is_file():
        raise ValueError(f"Expected a config file, got: {source}")

    try:
        with source.open("rb") as handle:
            payload = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in FrameVitals config: {source}") from exc

    if not isinstance(payload, dict):
        raise ValueError("FrameVitals config must contain a TOML table.")
    return payload


def _coerce_disabled_modules(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple, set)):
        raise ValueError("disabled_modules must be a list/tuple of module names.")
    return tuple(str(item) for item in value)


def _extract_values(
    mapping: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any], dict[str, bool]]:
    """Extract scalar values and module overrides from config mappings."""
    analysis = mapping.get("analysis", {})
    resources = mapping.get("resources", {})
    modules = mapping.get("modules", {})

    if not isinstance(analysis, Mapping):
        raise ValueError("[analysis] must be a TOML table/object.")
    if not isinstance(resources, Mapping):
        raise ValueError("[resources] must be a TOML table/object.")
    if not isinstance(modules, Mapping):
        raise ValueError("[modules] must be a TOML table/object.")

    preset = analysis.get("preset", mapping.get("preset"))
    values: dict[str, Any] = {}

    for key in ("mode", "target", "artifacts"):
        if key in analysis:
            values[key] = analysis[key]
        elif key in mapping:
            values[key] = mapping[key]

    for key in ("workers", *_RESOURCE_KEYS):
        if key in resources:
            values[key] = resources[key]
        elif key in mapping:
            values[key] = mapping[key]

    if "disabled_modules" in analysis:
        values["disabled_modules"] = _coerce_disabled_modules(
            analysis["disabled_modules"]
        )
    elif "disabled_modules" in mapping:
        values["disabled_modules"] = _coerce_disabled_modules(
            mapping["disabled_modules"]
        )

    module_overrides: dict[str, bool] = {}
    for name, enabled in modules.items():
        if name not in VALID_MODULES:
            raise ValueError(f"Unknown FrameVitals module in [modules]: {name}")
        if not isinstance(enabled, bool):
            raise ValueError(f"[modules].{name} must be true or false.")
        module_overrides[name] = enabled

    return str(preset) if preset is not None else None, values, module_overrides


def _coerce_environment_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of: true, false, 1, 0, yes, no, on, off."
    )


def _environment_values(
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Read deterministic FrameVitals runtime overrides from the environment."""
    source = os.environ if environ is None else environ
    preset = source.get("FRAMEVITALS_PRESET")
    if preset is not None:
        preset = preset.strip() or None

    values: dict[str, Any] = {}
    for env_name, config_name in _ENV_VALUE_KEYS.items():
        if env_name not in source:
            continue
        raw = str(source[env_name])
        if config_name == "artifacts":
            values[config_name] = _coerce_environment_bool(env_name, raw)
        elif config_name == "disabled_modules":
            values[config_name] = tuple(
                item.strip() for item in raw.split(",") if item.strip()
            )
        elif config_name == "target":
            values[config_name] = raw.strip() or None
        else:
            values[config_name] = raw.strip()
    return preset, values


def _preset_values(name: str | None) -> dict[str, Any]:
    if name is None:
        return {}
    if name not in PRESETS:
        raise ValueError(
            f"Unknown FrameVitals preset '{name}'. "
            f"Choose from: {', '.join(available_presets())}"
        )
    return dict(PRESETS[name])


def _apply_module_overrides(
    values: dict[str, Any],
    overrides: Mapping[str, bool],
) -> None:
    disabled = list(_coerce_disabled_modules(values.get("disabled_modules", ())))
    for name, enabled in overrides.items():
        if enabled:
            disabled = [item for item in disabled if item != name]
        elif name not in disabled:
            disabled.append(name)
    values["disabled_modules"] = tuple(disabled)


def _coerce_optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer when provided.")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer when provided.") from exc
    if converted < 1:
        raise ValueError(f"{name} must be at least 1 when provided.")
    return converted


def resolve_config(
    config: ConfigInput = None,
    *,
    preset: str | None = None,
    mode: str | None = None,
    target: str | None = None,
    artifacts: bool | None = None,
    workers: int | None = None,
    disabled_modules: list[str] | tuple[str, ...] | None = None,
    max_sample_rows: int | None = None,
    max_relationship_pairs: int | None = None,
    max_memory_heavy_parallelism: int | None = None,
    max_streaming_profile_columns: int | None = None,
) -> AnalysisConfig:
    """Resolve defaults, preset, environment, config, then explicit overrides.

    Precedence, from lowest to highest, is:

    1. FrameVitals defaults
    2. preset defaults
    3. ``FRAMEVITALS_*`` environment overrides
    4. configuration file/mapping/object (including ``[modules]`` booleans)
    5. explicit function/CLI arguments

    Resource caps are strict upper bounds, not requests to increase work above
    the selected mode's adaptive defaults.
    """
    values: dict[str, Any] = AnalysisConfig().to_dict()
    values.update(_preset_values(preset))

    environment_preset, environment_values = _environment_values()
    if environment_preset is not None:
        values.update(_preset_values(environment_preset))
    values.update(environment_values)

    if isinstance(config, AnalysisConfig):
        values.update(config.to_dict())
    elif isinstance(config, (str, Path)):
        config_preset, config_values, module_overrides = _extract_values(
            _read_toml(config)
        )
        if config_preset is not None:
            values.update(_preset_values(config_preset))
        values.update(config_values)
        _apply_module_overrides(values, module_overrides)
    elif isinstance(config, Mapping):
        config_preset, config_values, module_overrides = _extract_values(config)
        if config_preset is not None:
            values.update(_preset_values(config_preset))
        values.update(config_values)
        _apply_module_overrides(values, module_overrides)
    elif config is not None:
        raise TypeError(
            "config must be an AnalysisConfig, mapping, TOML path, or None."
        )

    explicit = {
        "mode": mode,
        "target": target,
        "artifacts": artifacts,
        "workers": workers,
        "disabled_modules": (
            tuple(disabled_modules) if disabled_modules is not None else None
        ),
        "max_sample_rows": max_sample_rows,
        "max_relationship_pairs": max_relationship_pairs,
        "max_memory_heavy_parallelism": max_memory_heavy_parallelism,
        "max_streaming_profile_columns": max_streaming_profile_columns,
    }
    values.update({key: value for key, value in explicit.items() if value is not None})

    try:
        values["workers"] = int(values["workers"])
    except (TypeError, ValueError) as exc:
        raise ValueError("workers must be an integer.") from exc

    if not isinstance(values["artifacts"], bool):
        raise ValueError("artifacts must be true or false.")
    if values["target"] is not None and not isinstance(values["target"], str):
        raise ValueError("target must be a column name string or null.")

    values["disabled_modules"] = _coerce_disabled_modules(
        values.get("disabled_modules", ())
    )
    for name in _RESOURCE_KEYS:
        values[name] = _coerce_optional_positive_int(name, values.get(name))

    return AnalysisConfig(**values)


def with_overrides(config: AnalysisConfig, **changes: Any) -> AnalysisConfig:
    """Return a validated copy of an existing resolved configuration."""
    return replace(config, **changes)
