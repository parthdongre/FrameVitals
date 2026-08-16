"""Backward-compatible CLI entrypoint.

The installed console script now points directly to :mod:`framevitals.cli`.
This module remains as a lightweight compatibility alias for callers that
imported ``framevitals.cli_entry.main`` during the 0.x series.
"""

from __future__ import annotations


def main() -> int:
    from framevitals.cli import main as cli_main

    return cli_main()
