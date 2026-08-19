"""Opt-in discovery for third-party FrameVitals extensions.

FrameVitals never imports installed plugins automatically. Applications that
want extension discovery must call :func:`discover_checks` explicitly, then
pass the returned definitions to ``framevitals.run_checks`` or
``framevitals.gate``.

Third-party packages register one check per Python entry point under the
``framevitals.checks`` group::

    [project.entry-points."framevitals.checks"]
    positive_revenue = "acme_data_checks:positive_revenue"

The referenced object must be either a :class:`framevitals.checks.DataCheck`
or a callable accepting one pandas DataFrame.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any

from framevitals.checks import DataCheck


CHECK_ENTRYPOINT_GROUP = "framevitals.checks"


def _entry_points_for(group: str):
    discovered = importlib_metadata.entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=group))
    # Compatibility with older importlib.metadata collection objects.
    return list(discovered.get(group, ()))


def _load_entry_point(entry: Any) -> DataCheck:
    try:
        loaded = entry.load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load FrameVitals check plugin {entry.name!r} from {entry.value!r}."
        ) from exc

    if isinstance(loaded, DataCheck):
        return loaded
    if callable(loaded):
        return DataCheck(name=str(entry.name), function=loaded)
    raise TypeError(
        "FrameVitals check plugin "
        f"{entry.name!r} must expose a DataCheck or DataFrame callable; "
        f"got {type(loaded).__name__}."
    )


def discover_checks(
    *,
    group: str = CHECK_ENTRYPOINT_GROUP,
) -> list[DataCheck]:
    """Load explicitly installed check plugins from a Python entry-point group.

    Discovery is opt-in because loading an entry point executes code from the
    installed provider package. Returned checks are sorted by registration
    name for deterministic behavior. Duplicate public check names are rejected
    so gate behavior cannot depend on package discovery order.
    """
    entries = sorted(
        _entry_points_for(group),
        key=lambda entry: (str(entry.name), str(entry.value)),
    )
    checks: list[DataCheck] = []
    seen_names: set[str] = set()

    for entry in entries:
        definition = _load_entry_point(entry)
        if definition.name in seen_names:
            raise ValueError(
                f"Duplicate FrameVitals check plugin name: {definition.name!r}."
            )
        seen_names.add(definition.name)
        checks.append(definition)

    return checks


__all__ = ["CHECK_ENTRYPOINT_GROUP", "discover_checks"]
