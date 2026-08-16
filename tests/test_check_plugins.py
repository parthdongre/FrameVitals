from __future__ import annotations

import pandas as pd
import pytest

from framevitals.checks import DataCheck
from framevitals.plugins import CHECK_ENTRYPOINT_GROUP, discover_checks


class _FakeEntryPoint:
    def __init__(self, name, value, loaded=None, error=None):
        self.name = name
        self.value = value
        self._loaded = loaded
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._loaded


class _FakeEntryPoints(list):
    def select(self, *, group):
        assert group == CHECK_ENTRYPOINT_GROUP
        return self


def test_discover_checks_loads_callables_and_datachecks(monkeypatch):
    def positive_values(df: pd.DataFrame):
        return bool((df["value"] > 0).all())

    explicit = DataCheck(
        name="explicit check",
        function=lambda df: True,
        severity="warning",
    )
    entries = _FakeEntryPoints([
        _FakeEntryPoint("z_plugin", "pkg.z:check", loaded=explicit),
        _FakeEntryPoint("a_plugin", "pkg.a:positive_values", loaded=positive_values),
    ])
    monkeypatch.setattr("framevitals.plugins.importlib_metadata.entry_points", lambda: entries)

    checks = discover_checks()

    assert [item.name for item in checks] == ["a_plugin", "explicit check"]
    assert checks[0](pd.DataFrame({"value": [1, 2]})) is True
    assert checks[1].severity == "warning"


def test_discover_checks_rejects_duplicate_public_names(monkeypatch):
    first = DataCheck(name="same", function=lambda df: True)
    second = DataCheck(name="same", function=lambda df: True)
    entries = _FakeEntryPoints([
        _FakeEntryPoint("one", "pkg.one:check", loaded=first),
        _FakeEntryPoint("two", "pkg.two:check", loaded=second),
    ])
    monkeypatch.setattr("framevitals.plugins.importlib_metadata.entry_points", lambda: entries)

    with pytest.raises(ValueError, match="Duplicate FrameVitals check plugin name"):
        discover_checks()


def test_discover_checks_surfaces_plugin_load_failures(monkeypatch):
    entries = _FakeEntryPoints([
        _FakeEntryPoint(
            "broken",
            "broken_pkg:check",
            error=ImportError("provider dependency missing"),
        )
    ])
    monkeypatch.setattr("framevitals.plugins.importlib_metadata.entry_points", lambda: entries)

    with pytest.raises(RuntimeError, match="broken"):
        discover_checks()


def test_discover_checks_rejects_unsupported_exports(monkeypatch):
    entries = _FakeEntryPoints([
        _FakeEntryPoint("bad", "pkg.bad:value", loaded=42),
    ])
    monkeypatch.setattr("framevitals.plugins.importlib_metadata.entry_points", lambda: entries)

    with pytest.raises(TypeError, match="DataCheck or DataFrame callable"):
        discover_checks()


def test_discover_checks_can_target_an_explicit_group(monkeypatch):
    class _Groups:
        def select(self, *, group):
            assert group == "acme.framevitals.checks"
            return []

    monkeypatch.setattr("framevitals.plugins.importlib_metadata.entry_points", _Groups)

    assert discover_checks(group="acme.framevitals.checks") == []
