"""Installed CLI entrypoint with source-aware analysis routing.

The legacy CLI parser remains the single command implementation. Before it
runs, this shim replaces only ``framevitals.api.analyze`` with the newer source
dispatcher so ``framevitals analyze`` and ``framevitals.analyze`` share the same
streaming behavior without duplicating CLI parsing or output code.
"""

from __future__ import annotations


def main() -> int:
    from framevitals import api as legacy_api
    from framevitals.analysis_api import analyze
    from framevitals.cli import main as legacy_main

    legacy_api.analyze = analyze
    return legacy_main()
