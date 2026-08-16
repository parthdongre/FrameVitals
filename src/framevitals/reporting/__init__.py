"""Human-facing renderers for FrameVitals results.

Renderers are intentionally dependency-light and consume the stable result
shape rather than re-running analysis.
"""

from framevitals.reporting.html import render_html_report, render_notebook_summary
from framevitals.reporting.terminal import render_terminal_summary

__all__ = [
    "render_html_report",
    "render_notebook_summary",
    "render_terminal_summary",
]
