"""Runtime configuration for FrameVitals analysis.

Configuration deliberately starts with options the current pipeline can honor
fully.  Future category/model/backend settings can extend the same schema
without introducing a second configuration system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping
import tomllib


VALID_MODES = {"quick", "standard", "deep", "research"}

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {"mode": "quick", "workers": 2, "artifacts": False},
    "standard": {"mode": "standard", "workers": 4, "artifacts": False},
    "deep": {"mode": "deep", "workers": 4, "artifacts": False},
    "research": {"mode": "research", "workers": 4, "artifacts": False},
    "ci": {"mode": "standard", "workers": 2, "artifacts": False},
}


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Resolved configuration consumed by the analysis pipeline."""

    mode: str = "standard"
    target: str | None = None
    artifacts: bool = False
    workers: int = 4

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid analysis mode '{self.mode}'. "
                f"Choose from: {', '.join(sorted(VALID_MODES))}"
            )
        if self.workers < 1:
            raise ValueError("workers must be at least 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ConfigInput = AnalysisConfig | Mapping[str, Any] | str | Path | None


def available_presets() -> tuple[str, ...]:
    """Return built-in preset names in deterministic order."""
    return tuple(PRESETS)


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


def _extract_values(mapping: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Extract supported values from nested or flat configuration mappings."""
    analysis = mapping.get("analysis", {})
    resources = mapping.get("resources", {})

    if not isinstance(analysis, Mapping):
        raise ValueError("[analysis] must be a TOML table/object.")
    if not isinstance(resources, Mapping):
        raise ValueError("[resources] must be a TOML table/object.")

    preset = analysis.get("preset", mapping.get("preset"))
    values: dict[str, Any] = {}

    for key in ("mode", "target", "artifacts"):
        if key in analysis:
            values[key] = analysis[key]
        elif key in mapping:
            values[key] = mapping[key]

    if "workers" in resources:
        values["workers"] = resources["workers"]
    elif "workers" in mapping:
        values["workers"] = mapping["workers"]

    return str(preset) if preset is not None else None, values


def _preset_values(name: str | None) -> dict[str, Any]:
    if name is None:
        return {}
    if name not in PRESETS:
        raise ValueError(
            f"Unknown FrameVitals preset '{name}'. "
            f"Choose from: {', '.join(available_presets())}"
        )
    return dict(PRESETS[name])


def resolve_config(
    config: ConfigInput = None,
    *,
    preset: str | None = None,
    mode: str | None = None,
    target: str | None = None,
    artifacts: bool | None = None,
    workers: int | None = None,
) -> AnalysisConfig:
    """Resolve defaults, preset, config file/object, then explicit overrides.

    Precedence, from lowest to highest, is:

    1. FrameVitals defaults
    2. explicit ``preset=`` argument
    3. configuration file/mapping/object
    4. explicit function/CLI arguments
    """
    values: dict[str, Any] = AnalysisConfig().to_dict()
    values.update(_preset_values(preset))

    if isinstance(config, AnalysisConfig):
        values.update(config.to_dict())
    elif isinstance(config, (str, Path)):
        config_preset, config_values = _extract_values(_read_toml(config))
        if config_preset is not None:
            values.update(_preset_values(config_preset))
        values.update(config_values)
    elif isinstance(config, Mapping):
        config_preset, config_values = _extract_values(config)
        if config_preset is not None:
            values.update(_preset_values(config_preset))
        values.update(config_values)
    elif config is not None:
        raise TypeError(
            "config must be an AnalysisConfig, mapping, TOML path, or None."
        )

    explicit = {
        "mode": mode,
        "target": target,
        "artifacts": artifacts,
        "workers": workers,
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

    return AnalysisConfig(**values)


def with_overrides(config: AnalysisConfig, **changes: Any) -> AnalysisConfig:
    """Return a validated copy of an existing resolved configuration."""
    return replace(config, **changes)
