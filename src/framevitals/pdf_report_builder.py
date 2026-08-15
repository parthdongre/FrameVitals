"""
PDF Report Builder (v3 — editorial dark)
========================================
Generates the DataLens AI dataset report PDF.

Visual identity matches the dashboard: cream + teal on near-black, heavy
display weights, mono eyebrows, generous whitespace. Pages flow as:

    1.  Cover            (hero band, dataset name, KPI tiles)
    2.  Contents         (numbered section list)
    3+. Content sections (executive summary, quality, signals, ML, etc.)
    n.  Visual Evidence  (2 charts per page, headlined + captioned)

Built on matplotlib's PdfPages so we don't pull a new heavyweight dep.
"""

from __future__ import annotations

import datetime as _dt
import textwrap
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Page geometry  (A4 in inches)
# ---------------------------------------------------------------------------

PAGE_W = 8.27
PAGE_H = 11.69

MARGIN_L = 0.70
MARGIN_R = 0.70
MARGIN_T = 0.70
MARGIN_B = 0.70

CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R


# ---------------------------------------------------------------------------
# Editorial palette  (mirror of frontend tokens + the new visualizer.py)
# ---------------------------------------------------------------------------

BG_PAGE   = "#0a0a0a"
BG_PANEL  = "#141414"
BG_PANEL_2 = "#1a1a1a"
LINE      = "#272727"
LINE_2    = "#3a3a3a"

INK_1     = "#f5efe6"
INK_2     = "#c9c1b4"
INK_3     = "#8a8478"
INK_4     = "#5a5650"

ACCENT    = "#5eead4"
ACCENT_2  = "#84d8b8"
ACCENT_SOFT = "#1f4540"  # deep teal, used for fills

WARN      = "#f5b14a"
BAD       = "#f08080"
OK        = "#84d8b8"

# Section-divider color codes (kept inside the family — accent on category)
SECTION_COLORS = {
    "summary":  ACCENT,
    "quality":  OK,
    "signals":  WARN,
    "cleaning": ACCENT_2,
    "deep":     ACCENT,
    "ml":       WARN,
    "time":     ACCENT_2,
    "text":     INK_2,
    "ai":       ACCENT,
    "charts":   ACCENT,
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def safe_text(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    replacements = {
        "—": "-", "–": "-",
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "•": "-",
        "×": "x",
        "↔": "<->",
        "→": "->",
        "←": "<-",
        "·": "-",
        "≥": ">=",
        "≤": "<=",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = unicodedata.normalize("NFKD", value)
    return value.encode("ascii", "ignore").decode("ascii")


def fmt_number(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != value:  # NaN
            return "N/A"
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return safe_text(value)


def _wrap(text: str, width: int) -> list[str]:
    cleaned = safe_text(text).strip()
    if not cleaned:
        return []
    out: list[str] = []
    for paragraph in cleaned.split("\n"):
        if not paragraph.strip():
            out.append("")
            continue
        wrapped = textwrap.wrap(
            paragraph, width=max(24, width),
            break_long_words=False, break_on_hyphens=False,
        )
        out.extend(wrapped or [""])
    return out


def resolve_chart_path(chart_path: str) -> Path | None:
    if not chart_path:
        return None
    raw = Path(chart_path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([
            Path("static") / raw,
        ])
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

class Composer:
    """
    Single-fig-per-page composer.

    A "layout axis" spans the full page using inches as its coord system, so
    every position is just (x_inches_from_left, y_inches_from_bottom). This is
    the simplest mental model for editorial layout, and it matches how a print
    designer would lay things out.
    """

    def __init__(self, output_path: Path, *, title: str, dataset_label: str):
        self.output_path = output_path
        self.title = title
        self.dataset_label = dataset_label
        self.pdf = PdfPages(str(output_path))

        self.page_number = 0
        self.total_pages_estimate: int | None = None
        self.fig: plt.Figure | None = None
        self.lay = None  # layout axis
        self.y = 0.0
        self.is_cover = False
        self.show_chrome = True  # header + footer

    # -------------------------------------------------------------- lifecycle

    def close(self):
        if self.fig is not None:
            self._end_page()
        self.pdf.close()

    def _begin_page(self, *, cover: bool = False, chrome: bool = True):
        self.page_number += 1
        self.is_cover = cover
        self.show_chrome = chrome
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.fig.patch.set_facecolor(BG_PAGE)

        # Layout axis — full-bleed, inches as units
        self.lay = self.fig.add_axes([0, 0, 1, 1])
        self.lay.set_xlim(0, PAGE_W)
        self.lay.set_ylim(0, PAGE_H)
        self.lay.set_facecolor(BG_PAGE)
        for spine in self.lay.spines.values():
            spine.set_visible(False)
        self.lay.set_xticks([])
        self.lay.set_yticks([])

        # Body cursor starts below top margin
        self.y = PAGE_H - MARGIN_T

        if chrome and not cover:
            self._draw_header()
            self._draw_footer()

    def _end_page(self):
        if self.fig is None:
            return
        self.pdf.savefig(self.fig, facecolor=BG_PAGE)
        plt.close(self.fig)
        self.fig = None
        self.lay = None

    def new_page(self, *, cover: bool = False, chrome: bool = True):
        if self.fig is not None:
            self._end_page()
        self._begin_page(cover=cover, chrome=chrome)

    # -------------------------------------------------------------- chrome

    def _draw_header(self):
        # Eyebrow brand
        self.lay.text(
            MARGIN_L, PAGE_H - 0.42,
            "DATALENS  ·  AI DATASET REPORT",
            fontsize=8, color=INK_3, fontweight="bold",
            family="monospace", va="center", ha="left",
        )
        # Dataset label, right-aligned, ink-2
        label = safe_text(self.dataset_label)
        if label:
            self.lay.text(
                PAGE_W - MARGIN_R, PAGE_H - 0.42,
                label[:48],
                fontsize=8, color=INK_2, fontweight="700",
                va="center", ha="right",
            )
        # Hairline rule
        self.lay.add_patch(patches.Rectangle(
            (MARGIN_L, PAGE_H - 0.55),
            CONTENT_W, 0.005,
            facecolor=LINE_2, edgecolor="none", linewidth=0,
        ))

    def _draw_footer(self):
        # Hairline rule
        self.lay.add_patch(patches.Rectangle(
            (MARGIN_L, MARGIN_B - 0.18),
            CONTENT_W, 0.005,
            facecolor=LINE, edgecolor="none", linewidth=0,
        ))
        self.lay.text(
            MARGIN_L, MARGIN_B - 0.32,
            "DATALENS · AI",
            fontsize=8, color=INK_3, fontweight="bold",
            family="monospace", va="center", ha="left",
        )
        page_label = f"{self.page_number:02d}"
        if self.total_pages_estimate:
            page_label = f"{self.page_number:02d}  /  {self.total_pages_estimate:02d}"
        self.lay.text(
            PAGE_W - MARGIN_R, MARGIN_B - 0.32,
            page_label,
            fontsize=8, color=INK_2, fontweight="bold",
            family="monospace", va="center", ha="right",
        )

    # -------------------------------------------------------------- primitives

    def _ensure_space(self, height_inches: float):
        if self.y - height_inches < MARGIN_B + 0.1:
            self.new_page()

    def rect(self, x, y, w, h, color, *, alpha=1.0, radius=0.0):
        if radius and radius > 0:
            r = patches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=color, edgecolor="none", alpha=alpha,
            )
        else:
            r = patches.Rectangle((x, y), w, h, facecolor=color,
                                   edgecolor="none", alpha=alpha)
        self.lay.add_patch(r)

    def stroke_rect(self, x, y, w, h, color, *, lw=0.8, alpha=1.0, radius=0.0):
        if radius and radius > 0:
            r = patches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor="none", edgecolor=color, linewidth=lw, alpha=alpha,
            )
        else:
            r = patches.Rectangle((x, y), w, h, facecolor="none",
                                   edgecolor=color, linewidth=lw, alpha=alpha)
        self.lay.add_patch(r)

    def hline(self, x1, x2, y, color=LINE, *, lw=0.8):
        self.lay.add_patch(patches.Rectangle(
            (x1, y), x2 - x1, max(lw / 72.0, 0.005),
            facecolor=color, edgecolor="none",
        ))

    def text(self, x, y, text, *, size=10, color=INK_2, weight="normal",
             ha="left", va="baseline", family=None, alpha=1.0):
        kwargs = dict(fontsize=size, color=color, fontweight=weight,
                      ha=ha, va=va, alpha=alpha)
        if family:
            kwargs["family"] = family
        self.lay.text(x, y, safe_text(text), **kwargs)

    # -------------------------------------------------------------- typography

    def text_block(self, text, *, x=None, size=10, color=INK_2, weight="normal",
                   line_height=None, family=None, max_width_in=None):
        """
        Flow a paragraph at the current y cursor. Wraps to fit content width.
        Advances self.y down. Auto-paginates.
        """
        if x is None:
            x = MARGIN_L
        if max_width_in is None:
            max_width_in = (MARGIN_L + CONTENT_W) - x
        line_h = line_height or max(0.16, size * 0.018)

        # rough char-per-line estimate based on Inter/sans — tuned empirically
        chars_per_line = max(28, int(max_width_in / (size * 0.0070)))
        lines = _wrap(text, chars_per_line)
        if not lines:
            self.y -= 0.04
            return

        for line in lines:
            self._ensure_space(line_h + 0.02)
            self.lay.text(
                x, self.y, line,
                fontsize=size, color=color, fontweight=weight,
                ha="left", va="top",
                family=family or "sans-serif",
            )
            self.y -= line_h

    def heading(self, text, *, size=16, color=INK_1, weight="bold",
                family=None, after=0.10):
        self._ensure_space(size * 0.025 + 0.1)
        self.lay.text(
            MARGIN_L, self.y, safe_text(text),
            fontsize=size, color=color, fontweight=weight,
            ha="left", va="top",
            family=family or "sans-serif",
        )
        self.y -= size * 0.022
        self.y -= after

    def eyebrow(self, text, *, color=ACCENT, after=0.14):
        self._ensure_space(0.22)
        self.lay.text(
            MARGIN_L, self.y, safe_text(str(text).upper()),
            fontsize=8, color=color, fontweight="bold",
            ha="left", va="top",
            family="monospace",
        )
        self.y -= 0.18
        self.y -= after

    def section(self, title: str, *, kind: str = "summary", subtitle: str | None = None):
        """Section divider — eyebrow + big title + colored hairline."""
        self._ensure_space(1.0)
        color = SECTION_COLORS.get(kind, ACCENT)
        self.y -= 0.12
        self.eyebrow(title.upper().split(" ")[0] if " " not in title else title.split(" ")[0].upper(),
                     color=color, after=0.06)
        # Restore the actual title (eyebrow used a one-word teaser)
        self._ensure_space(0.6)
        self.lay.text(
            MARGIN_L, self.y, safe_text(title),
            fontsize=22, color=INK_1, fontweight="bold",
            ha="left", va="top",
        )
        self.y -= 0.42
        # Subtitle
        if subtitle:
            self.lay.text(
                MARGIN_L, self.y, safe_text(subtitle),
                fontsize=10, color=INK_3, fontweight="700",
                ha="left", va="top",
            )
            self.y -= 0.20
        # Color rule
        self.hline(MARGIN_L, MARGIN_L + 1.3, self.y, color=color, lw=2.2)
        self.y -= 0.22

    def subheading(self, text, *, color=INK_1):
        self._ensure_space(0.4)
        self.y -= 0.04
        self.lay.text(
            MARGIN_L, self.y, safe_text(text),
            fontsize=12, color=color, fontweight="bold",
            ha="left", va="top",
        )
        self.y -= 0.22

    def callout(self, text, *, color=ACCENT, height=None):
        lines = _wrap(text, 96)
        h = height or max(0.34, 0.14 + 0.14 * len(lines))
        self._ensure_space(h + 0.08)
        # Left bar
        self.rect(MARGIN_L, self.y - h, 0.06, h, color)
        # Body
        cursor_y = self.y - 0.06
        for line in lines:
            self.lay.text(
                MARGIN_L + 0.16, cursor_y, line,
                fontsize=10, color=INK_2, fontweight="700",
                ha="left", va="top",
            )
            cursor_y -= 0.18
        self.y -= h + 0.10

    def bullets(self, items, *, color=INK_2, indent=0.18, size=9.5,
                marker_color=ACCENT):
        for item in items:
            line_h = 0.18
            text = safe_text(item)
            chars_per_line = max(28, int((CONTENT_W - indent) / (size * 0.0070)))
            wrapped = _wrap(text, chars_per_line)
            if not wrapped:
                continue
            self._ensure_space(line_h + 0.02)
            # marker (small dot)
            self.lay.add_patch(patches.Circle(
                (MARGIN_L + indent - 0.08, self.y - 0.07),
                0.025, facecolor=marker_color, edgecolor="none",
            ))
            self.lay.text(
                MARGIN_L + indent, self.y, wrapped[0],
                fontsize=size, color=color, fontweight="700",
                ha="left", va="top",
            )
            self.y -= line_h
            for cont in wrapped[1:]:
                self._ensure_space(line_h + 0.02)
                self.lay.text(
                    MARGIN_L + indent, self.y, cont,
                    fontsize=size, color=color, fontweight="700",
                    ha="left", va="top",
                )
                self.y -= line_h
            self.y -= 0.02

    # -------------------------------------------------------------- KPI tiles

    def kpi_grid(self, tiles: list[dict], *, columns: int = 4, row_height: float = 1.30):
        """tiles = [{label, value, hint?, accent?}]"""
        if not tiles:
            return
        n = len(tiles)
        per_row = columns
        gap = 0.16
        total_w = CONTENT_W
        tile_w = (total_w - gap * (per_row - 1)) / per_row
        rows = (n + per_row - 1) // per_row

        self._ensure_space(rows * (row_height + gap) + 0.1)

        for r in range(rows):
            row_y_top = self.y
            for c in range(per_row):
                idx = r * per_row + c
                if idx >= n:
                    break
                t = tiles[idx]
                x = MARGIN_L + c * (tile_w + gap)
                y = row_y_top - row_height
                # Tile background
                self.rect(x, y, tile_w, row_height, BG_PANEL, radius=0.08)
                # Accent slim bar at top
                self.rect(x, y + row_height - 0.06,
                          tile_w, 0.06,
                          t.get("accent", ACCENT), radius=0.0)
                # Label
                self.lay.text(
                    x + 0.16, y + row_height - 0.32,
                    safe_text(str(t.get("label", "")).upper()),
                    fontsize=8, color=INK_3, fontweight="bold",
                    family="monospace", ha="left", va="top",
                )
                # Value
                self.lay.text(
                    x + 0.16, y + row_height - 0.56,
                    safe_text(t.get("value", "")),
                    fontsize=22, color=INK_1, fontweight="bold",
                    ha="left", va="top",
                )
                # Hint
                hint = t.get("hint")
                if hint:
                    self.lay.text(
                        x + 0.16, y + 0.14,
                        safe_text(hint),
                        fontsize=8, color=INK_3, fontweight="700",
                        ha="left", va="bottom",
                    )
            self.y = row_y_top - row_height - gap

        self.y -= 0.05

    # -------------------------------------------------------------- table

    def kv_table(self, rows: list[tuple[str, str]], *, label_w: float = 2.4):
        """Two-column key/value list with subtle row dividers."""
        if not rows:
            return
        line_h = 0.22
        self._ensure_space(len(rows) * line_h + 0.1)
        for label, value in rows:
            self._ensure_space(line_h + 0.02)
            self.lay.text(
                MARGIN_L, self.y, safe_text(str(label).upper()),
                fontsize=8, color=INK_3, fontweight="bold",
                family="monospace", ha="left", va="top",
            )
            self.lay.text(
                MARGIN_L + label_w, self.y, safe_text(value),
                fontsize=10, color=INK_1, fontweight="700",
                ha="left", va="top",
            )
            self.y -= 0.05
            self.hline(MARGIN_L, MARGIN_L + CONTENT_W, self.y, color=LINE, lw=0.5)
            self.y -= line_h - 0.05

    # -------------------------------------------------------------- chart embed

    def add_chart_image(self, chart: dict, *, x: float, y_top: float,
                        width: float, height: float):
        """
        Embed a chart PNG inside the layout at the given inches box.
        Returns False if the chart was missing.
        """
        path = resolve_chart_path(chart.get("path", ""))

        # Caption header band
        self.rect(x, y_top - 0.36, width, 0.36, BG_PANEL_2, radius=0.06)
        self.lay.text(
            x + 0.14, y_top - 0.13,
            safe_text(str(chart.get("type", "chart")).upper()),
            fontsize=7.5, color=INK_3, fontweight="bold",
            family="monospace", ha="left", va="top",
        )
        self.lay.text(
            x + 0.14, y_top - 0.27,
            safe_text(chart.get("title", "Untitled")),
            fontsize=10, color=INK_1, fontweight="bold",
            ha="left", va="top",
        )
        # Image area
        img_top = y_top - 0.40
        img_bottom = y_top - height + 0.35  # leave room for caption
        img_height = img_top - img_bottom
        img_x_frac = x / PAGE_W
        img_y_frac = img_bottom / PAGE_H
        img_w_frac = width / PAGE_W
        img_h_frac = img_height / PAGE_H

        # Frame
        self.rect(x, img_bottom, width, img_height, BG_PANEL_2, radius=0.04)

        if path is not None:
            try:
                image = plt.imread(str(path))
                ax = self.fig.add_axes([img_x_frac + 0.005,
                                        img_y_frac + 0.005,
                                        img_w_frac - 0.01,
                                        img_h_frac - 0.01])
                ax.imshow(image)
                ax.axis("off")
                ax.set_anchor("C")
            except Exception as exc:
                self.lay.text(
                    x + width / 2, img_bottom + img_height / 2,
                    f"(chart embed failed: {safe_text(exc)})",
                    fontsize=8, color=INK_3, fontweight="700",
                    ha="center", va="center",
                )
        else:
            self.lay.text(
                x + width / 2, img_bottom + img_height / 2,
                "(chart image not found on disk)",
                fontsize=8, color=INK_3, fontweight="700",
                ha="center", va="center",
            )

        # Caption
        desc = chart.get("description", "")
        if desc:
            self.lay.text(
                x + 0.14, img_bottom - 0.05,
                safe_text(desc[:140]),
                fontsize=8, color=INK_3, fontweight="700",
                ha="left", va="top",
            )
        return True


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _cover_page(c: Composer, result: dict):
    c.new_page(cover=True, chrome=False)
    f = c.fig
    lay = c.lay

    # Hero band — gradient from teal-soft → near-black using a stack of rects
    band_top = PAGE_H - 0.0
    band_bottom = PAGE_H - 4.6
    # outer teal halo (very dark)
    c.rect(0, band_bottom, PAGE_W, band_top - band_bottom, "#0f1f1d")
    # accent lines
    c.hline(0, PAGE_W, band_bottom + 0.04, color=LINE_2, lw=0.6)
    c.hline(0, PAGE_W, band_top - 0.04, color=LINE, lw=0.4)

    # Brand eyebrow
    lay.text(
        MARGIN_L, PAGE_H - 1.0,
        "DATALENS · AI",
        fontsize=10, color=ACCENT, fontweight="bold",
        family="monospace", ha="left", va="top",
    )
    # divider dots
    lay.text(
        MARGIN_L + 1.6, PAGE_H - 1.0,
        "DATASET REPORT",
        fontsize=10, color=INK_3, fontweight="bold",
        family="monospace", ha="left", va="top",
    )

    # Big title
    lay.text(
        MARGIN_L, PAGE_H - 1.85,
        "DATASET",
        fontsize=64, color=INK_1, fontweight="bold",
        ha="left", va="top",
    )
    lay.text(
        MARGIN_L, PAGE_H - 2.85,
        "REPORT.",
        fontsize=64, color=ACCENT, fontweight="bold",
        ha="left", va="top",
    )

    # Filename
    fname = safe_text(result.get("filename", "Untitled"))
    lay.text(
        MARGIN_L, PAGE_H - 3.45,
        fname,
        fontsize=14, color=INK_2, fontweight="700",
        ha="left", va="top",
    )

    # Meta line
    profile = result.get("profile", {}) or {}
    shape = profile.get("shape", {}) or {}
    rows_v = shape.get("rows", 0)
    cols_v = shape.get("columns", 0)
    mode = result.get("analysis_mode", "standard")
    today = _dt.datetime.now().strftime("%b %d, %Y").upper()
    meta = f"{fmt_number(rows_v)} ROWS  ·  {fmt_number(cols_v)} COLS  ·  MODE {str(mode).upper()}  ·  {today}"
    lay.text(
        MARGIN_L, PAGE_H - 3.85,
        meta,
        fontsize=10, color=INK_3, fontweight="bold",
        family="monospace", ha="left", va="top",
    )

    # KPI grid below the band — 4 tiles, each 1.55in tall
    c.y = PAGE_H - 5.20
    health = result.get("health", {}) or {}
    ml = result.get("ml_readiness", {}) or {}
    charts = result.get("charts") or []
    signals = result.get("signals") or []

    def _color_for_score(v):
        try:
            v = float(v)
        except Exception:
            return INK_3
        if v >= 80:
            return OK
        if v >= 60:
            return WARN
        return BAD

    health_score = health.get("overall_score", 0) or 0
    ml_score = ml.get("score", 0) or 0
    tiles = [
        {"label": "Health Score",
         "value": f"{fmt_number(health_score)}",
         "hint": str(health.get("label", "")).title() or "—",
         "accent": _color_for_score(health_score)},
        {"label": "ML Readiness",
         "value": f"{fmt_number(ml_score)}",
         "hint": str(ml.get("label", "")).title() or "—",
         "accent": _color_for_score(ml_score)},
        {"label": "Charts",
         "value": f"{len(charts)}",
         "hint": "Generated",
         "accent": ACCENT},
        {"label": "Signals",
         "value": f"{len(signals)}",
         "hint": "Detected",
         "accent": ACCENT_2 if len(signals) <= 5 else WARN},
    ]
    c.kpi_grid(tiles, columns=4, row_height=1.50)

    # Tagline at bottom
    c.y = MARGIN_B + 1.6
    lay.text(
        MARGIN_L, c.y,
        "Generated by DataLens AI's analytical pipeline.",
        fontsize=10, color=INK_3, fontweight="700",
        ha="left", va="top",
    )
    lay.text(
        MARGIN_L, c.y - 0.18,
        "This report includes data quality, modeling, and visual evidence drawn from the uploaded dataset.",
        fontsize=10, color=INK_3, fontweight="700",
        ha="left", va="top",
    )

    # Footer hairline + brand
    c.hline(MARGIN_L, MARGIN_L + CONTENT_W, MARGIN_B + 0.4, color=LINE, lw=0.6)
    lay.text(
        MARGIN_L, MARGIN_B + 0.18,
        "DATALENS · AI",
        fontsize=8, color=INK_3, fontweight="bold",
        family="monospace", ha="left", va="top",
    )
    lay.text(
        PAGE_W - MARGIN_R, MARGIN_B + 0.18,
        "01",
        fontsize=8, color=INK_2, fontweight="bold",
        family="monospace", ha="right", va="top",
    )


def _toc_page(c: Composer, sections: list[tuple[str, str, str]]):
    """sections = list of (number, title, blurb)"""
    c.new_page()
    c.eyebrow("CONTENTS", color=ACCENT)
    c.heading("Inside this report", size=28, after=0.30)
    c.text_block(
        "A guided walkthrough of every analytical layer applied to the uploaded dataset, "
        "from raw quality checks all the way through to modeling and visual evidence.",
        size=10, color=INK_2, weight="700",
    )
    c.y -= 0.20
    c.hline(MARGIN_L, MARGIN_L + CONTENT_W, c.y, color=LINE_2, lw=0.6)
    c.y -= 0.22

    for num, title, blurb in sections:
        c._ensure_space(0.65)
        # number
        c.lay.text(
            MARGIN_L, c.y,
            num,
            fontsize=22, color=ACCENT, fontweight="bold",
            family="monospace", ha="left", va="top",
        )
        # title
        c.lay.text(
            MARGIN_L + 0.9, c.y,
            safe_text(title),
            fontsize=15, color=INK_1, fontweight="bold",
            ha="left", va="top",
        )
        # blurb on next line
        c.lay.text(
            MARGIN_L + 0.9, c.y - 0.28,
            safe_text(blurb),
            fontsize=9.5, color=INK_3, fontweight="700",
            ha="left", va="top",
        )
        c.y -= 0.62
        c.hline(MARGIN_L + 0.9, MARGIN_L + CONTENT_W, c.y, color=LINE, lw=0.5)
        c.y -= 0.18


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_executive_summary(c: Composer, result: dict):
    profile = result.get("profile", {}) or {}
    shape = profile.get("shape", {}) or {}
    health = result.get("health", {}) or {}
    ml = result.get("ml_readiness", {}) or {}
    sigs = result.get("signals") or []
    sel = (result.get("analysis_selection") or {}).get("summary") or {}
    mode = result.get("analysis_mode", "standard")

    c.new_page()
    c.section("Executive Summary", kind="summary",
              subtitle="Top-line read of what the pipeline found.")

    # Lead paragraph
    rows_v = fmt_number(shape.get("rows", 0))
    cols_v = fmt_number(shape.get("columns", 0))
    health_v = fmt_number(health.get("overall_score", 0))
    ml_v = fmt_number(ml.get("score", 0))
    health_label = (health.get("label") or "Unknown").title()
    ml_label = (ml.get("label") or "Unknown").title()

    lead = (
        f"This dataset contains {rows_v} rows across {cols_v} columns. "
        f"It scores {health_v}/100 on quality ({health_label}) and {ml_v}/100 on ML readiness ({ml_label}). "
        f"DataLens AI ran in {str(mode).upper()} mode and selected "
        f"{sel.get('selected_count', 0)} analyses, recommended {sel.get('recommended_count', 0)} more, "
        f"and skipped {sel.get('skipped_count', 0)}."
    )
    c.text_block(lead, size=11, color=INK_2, weight="700", line_height=0.20)
    c.y -= 0.10

    # Quick KPI tiles
    c.kpi_grid([
        {"label": "Quality", "value": f"{health_v}", "hint": health_label, "accent": ACCENT},
        {"label": "ML Score", "value": f"{ml_v}", "hint": ml_label, "accent": ACCENT_2},
        {"label": "Signals", "value": f"{len(sigs)}", "hint": "Detected", "accent": WARN if len(sigs) > 5 else ACCENT},
        {"label": "Charts", "value": f"{len(result.get('charts') or [])}", "hint": "Generated", "accent": ACCENT},
    ], columns=4, row_height=1.20)

    # High-severity signals callout
    high = [s for s in sigs if str(s.get("severity", "")).lower() in {"high", "critical"}]
    if high:
        c.subheading("Critical signals to act on")
        for s in high[:5]:
            line = f"{s.get('name', 'Signal')} — {s.get('recommendation', s.get('evidence', ''))}"
            c.callout(line, color=BAD)

    # Headline ML winner
    lb = result.get("model_leaderboard") or {}
    if isinstance(lb, dict) and lb.get("available") and lb.get("winner"):
        winner = lb["winner"]
        score = winner.get("primary_score")
        c.subheading("Headline model")
        c.callout(
            f"{winner.get('model', '?')} leads on {lb.get('task_type', '?')} (target: {lb.get('target_column', '?')}) "
            f"with cross-validated primary score {fmt_number(score)}.",
            color=ACCENT,
        )


def _render_quality(c: Composer, result: dict):
    health = result.get("health", {}) or {}
    components = health.get("components", {}) or {}
    if not components:
        return
    c.new_page()
    c.section("Health & Quality", kind="quality",
              subtitle="Per-component breakdown of the overall quality score.")

    rows = []
    for k, v in components.items():
        rows.append((k.replace("_", " "), f"{fmt_number(v)} / 100"))
    rows.append(("Overall Score", f"{fmt_number(health.get('overall_score', 0))} / 100"))
    rows.append(("Label", str(health.get("label", "Unknown")).title()))
    c.kv_table(rows)

    c.y -= 0.05
    c.subheading("How to read these scores")
    c.bullets([
        "Completeness: how full of values your columns are.",
        "Consistency: same kind of value in each column, no mixed types.",
        "Uniqueness: deduplication signal, lower if many repeated rows.",
        "Outlier safety: how many statistically extreme values are present.",
    ])


def _render_signals(c: Composer, result: dict):
    sigs = result.get("signals") or []
    if not sigs:
        return
    c.new_page()
    c.section("Signals & Recommendations", kind="signals",
              subtitle="Auto-detected issues with severity and a recommended action.")

    by_sev: dict[str, list[dict]] = {}
    for s in sigs:
        sev = str(s.get("severity", "low")).lower()
        by_sev.setdefault(sev, []).append(s)

    order = ["critical", "high", "medium", "low", "info"]
    palette = {
        "critical": BAD, "high": BAD,
        "medium": WARN, "low": ACCENT, "info": INK_3,
    }
    for sev in order:
        items = by_sev.get(sev) or []
        if not items:
            continue
        c.subheading(f"{sev.title()}  ·  {len(items)}")
        for s in items[:8]:
            text = (
                f"{s.get('name', 'Signal')}.  "
                f"{s.get('evidence', '')}  →  {s.get('recommendation', '')}"
            )
            c.callout(text, color=palette.get(sev, ACCENT))


def _render_cleaning(c: Composer, result: dict):
    cl = result.get("cleaning") or {}
    if not cl:
        return
    c.new_page()
    c.section("Cleaning Summary", kind="cleaning",
              subtitle="What changed when DataLens auto-cleaned the dataset.")

    before = (cl.get("before_health") or {}).get("overall_score")
    after = (cl.get("after_health") or {}).get("overall_score")
    delta = None
    try:
        delta = (after or 0) - (before or 0)
    except Exception:
        pass
    delta_color = OK if (delta or 0) >= 0 else BAD

    c.kpi_grid([
        {"label": "Before", "value": f"{fmt_number(before)}", "hint": "Health score", "accent": INK_3},
        {"label": "After", "value": f"{fmt_number(after)}", "hint": "Health score", "accent": ACCENT},
        {"label": "Delta", "value": f"{'+' if (delta or 0) >= 0 else ''}{fmt_number(delta)}", "hint": "Improvement", "accent": delta_color},
        {"label": "Duplicates", "value": f"{fmt_number(cl.get('duplicates_before', 0))} → {fmt_number(cl.get('duplicates_after', 0))}", "hint": "Rows removed", "accent": ACCENT_2},
    ], columns=4, row_height=1.20)

    actions = cl.get("actions") or []
    if actions:
        c.subheading("Cleaning actions")
        c.bullets([
            f"{a.get('action', 'Action')}: {a.get('details', '')} (risk: {a.get('risk', 'low')})"
            for a in actions[:14]
        ])


def _render_deep_stats(c: Composer, result: dict):
    deep = result.get("deep_statistics_v2") or {}
    if not deep:
        return
    summary = deep.get("summary") or {}
    biv = deep.get("bivariate") or {}
    if not summary and not biv:
        return

    c.new_page()
    c.section("Deep Statistics", kind="deep",
              subtitle="Pairwise relationships and group differences across columns.")

    if summary:
        c.kv_table([
            ("Numeric columns", fmt_number(summary.get("numeric_count", 0))),
            ("Categorical columns", fmt_number(summary.get("categorical_count", 0))),
            ("Numeric pairs tested", fmt_number(summary.get("numeric_pairs_tested", 0))),
            ("Categorical pairs tested", fmt_number(summary.get("categorical_pairs_tested", 0))),
            ("Group difference tests", fmt_number(summary.get("group_difference_tests_run", 0))),
        ])
        c.y -= 0.05

    nums = biv.get("numeric_pairs") or []
    if nums:
        c.subheading("Top numeric correlations")
        items = []
        for p in nums[:8]:
            r = ((p.get("pearson") or {}).get("r"))
            if r is None:
                continue
            items.append(f"{p.get('column_a')} ↔ {p.get('column_b')}: r = {fmt_number(r)}")
        c.bullets(items)

    cats = biv.get("categorical_pairs") or []
    if cats:
        c.subheading("Top categorical associations")
        items = []
        for p in cats[:6]:
            v = p.get("cramers_v")
            if v is None:
                continue
            items.append(f"{p.get('column_a')} ↔ {p.get('column_b')}: Cramér's V = {fmt_number(v)}")
        c.bullets(items)


def _render_modeling(c: Composer, result: dict):
    lb = result.get("model_leaderboard") or {}
    ex = result.get("explainability") or {}
    if not (isinstance(lb, dict) and lb.get("available")):
        return

    c.new_page()
    c.section("Modeling", kind="ml",
              subtitle="Cross-validated leaderboard and feature importance from the winner.")

    winner = lb.get("winner") or {}
    rows = lb.get("leaderboard") or []
    scored = [r for r in rows if r.get("primary_score") is not None]
    rows_for_table = [(r.get("model", "?"), fmt_number(r.get("primary_score")))
                       for r in scored[:8]]
    c.kpi_grid([
        {"label": "Task", "value": str(lb.get("task_type", "—")).title(), "hint": f"target: {lb.get('target_column', '—')}", "accent": ACCENT},
        {"label": "Winner", "value": str(winner.get("model", "—"))[:20], "hint": "Best CV model", "accent": OK},
        {"label": "Score", "value": f"{fmt_number(winner.get('primary_score'))}", "hint": "Primary metric", "accent": ACCENT_2},
        {"label": "Models", "value": f"{len(scored)}", "hint": "On the board", "accent": INK_3},
    ], columns=4, row_height=1.20)

    if rows_for_table:
        c.subheading("Leaderboard")
        c.kv_table(rows_for_table, label_w=2.4)

    # Explainability
    if isinstance(ex, dict) and ex.get("available"):
        c.subheading("Top features by importance")
        gi = ex.get("global_importance") or []
        items = [
            f"{r.get('feature', '?')}: {fmt_number(r.get('importance'))}"
            for r in gi[:10]
        ]
        c.bullets(items)


def _render_time_text(c: Composer, result: dict):
    ts = result.get("time_series") or {}
    tp = result.get("text_profile") or {}
    if not ((isinstance(ts, dict) and ts.get("available")) or (isinstance(tp, dict) and tp.get("available"))):
        return

    c.new_page()
    c.section("Time-series & Text", kind="time",
              subtitle="Detected temporal structure and textual columns, when present.")

    if isinstance(ts, dict) and ts.get("available"):
        c.subheading("Time series")
        c.kv_table([
            ("Date column", str(ts.get("detected_date_column") or "—")),
            ("Numeric column", str(ts.get("numeric_column") or "—")),
            ("Frequency", str((ts.get("frequency") or {}).get("label") or "—")),
            ("Stationarity", str((ts.get("stationarity") or {}).get("verdict") or "—")),
        ])

    if isinstance(tp, dict) and tp.get("available"):
        c.subheading("Text profile")
        cols = tp.get("profiled_columns") or []
        profiles = tp.get("profiles") or {}
        items = []
        for col in cols[:5]:
            # `profiled_columns` is a list of column-name strings; the per-column
            # stats live in `profiles[col]`. Be tolerant of both shapes just in case.
            if isinstance(col, str):
                name = col
                prof = profiles.get(col) or {}
            elif isinstance(col, dict):
                name = col.get("column", "?")
                prof = col
            else:
                continue
            n_rows = prof.get("n_rows") or prof.get("n_documents") or 0
            len_stats = prof.get("length_stats") or {}
            avg = len_stats.get("avg") if isinstance(len_stats, dict) else None
            if avg is None:
                avg = prof.get("avg_chars", 0)
            items.append(
                f"{name}: {fmt_number(n_rows)} docs · avg {fmt_number(avg)} chars"
            )
        if items:
            c.bullets(items)


def _render_ai_report(c: Composer, result: dict):
    ai = result.get("ai_report") or {}
    text = ai.get("text") or ""
    if not text or len(text.strip()) < 20:
        return

    c.new_page()
    c.section("AI Analyst Report", kind="ai",
              subtitle="Narrative interpretation of the dataset by DataLens's analyst model.")
    c.callout(f"source: {ai.get('source', 'unknown')}", color=ACCENT_2)
    c.text_block(text[:8000], size=10, color=INK_2, weight="700", line_height=0.20)


def _render_charts(c: Composer, result: dict):
    """Visual evidence — 2 charts per page."""
    charts = result.get("charts") or []
    if not charts:
        return

    # Section opener page
    c.new_page()
    c.section("Visual Evidence", kind="charts",
              subtitle=f"{len(charts)} planner-driven charts. Two per page from here on.")
    c.text_block(
        "Each figure below is generated only when the underlying analytical phase produced data. "
        "Captions above each image describe what to look for.",
        size=10, color=INK_3, weight="700",
    )

    # Pages of two charts each
    chart_w = CONTENT_W
    chart_h = (PAGE_H - MARGIN_T - MARGIN_B - 0.6) / 2  # two rows

    for i in range(0, len(charts), 2):
        c.new_page()
        c.eyebrow("VISUAL EVIDENCE", color=ACCENT)
        c.heading("Charts", size=22, after=0.10)
        c.hline(MARGIN_L, MARGIN_L + 1.3, c.y + 0.05, color=ACCENT, lw=2.0)
        c.y -= 0.10

        top_y = c.y
        # First chart
        c.add_chart_image(
            charts[i],
            x=MARGIN_L,
            y_top=top_y,
            width=chart_w,
            height=chart_h,
        )
        # Second chart, if any
        if i + 1 < len(charts):
            second_top = top_y - chart_h - 0.15
            c.add_chart_image(
                charts[i + 1],
                x=MARGIN_L,
                y_top=second_top,
                width=chart_w,
                height=chart_h,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_toc_sections(result: dict) -> list[tuple[str, str, str]]:
    sections = [("01", "Executive Summary",
                  "Top-line scores, headline model, and the most pressing signals.")]
    if (result.get("health") or {}).get("components"):
        sections.append(("02", "Health & Quality",
                          "Per-component breakdown of the overall quality score."))
    if result.get("signals"):
        sections.append((f"{len(sections)+1:02d}", "Signals & Recommendations",
                          "Auto-detected issues, sorted by severity."))
    if result.get("cleaning"):
        sections.append((f"{len(sections)+1:02d}", "Cleaning Summary",
                          "Before / after health and the actions DataLens took."))
    if result.get("deep_statistics_v2"):
        sections.append((f"{len(sections)+1:02d}", "Deep Statistics",
                          "Pairwise relationships and group differences."))
    lb = result.get("model_leaderboard") or {}
    if isinstance(lb, dict) and lb.get("available"):
        sections.append((f"{len(sections)+1:02d}", "Modeling",
                          "Cross-validated leaderboard and feature importance."))
    ts = result.get("time_series") or {}
    tp = result.get("text_profile") or {}
    if (isinstance(ts, dict) and ts.get("available")) or (isinstance(tp, dict) and tp.get("available")):
        sections.append((f"{len(sections)+1:02d}", "Time-series & Text",
                          "Temporal structure and textual columns where present."))
    ai = result.get("ai_report") or {}
    if ai.get("text"):
        sections.append((f"{len(sections)+1:02d}", "AI Analyst Report",
                          "Narrative interpretation by the DataLens analyst model."))
    if result.get("charts"):
        sections.append((f"{len(sections)+1:02d}", "Visual Evidence",
                          f"{len(result.get('charts') or [])} planner-driven charts, two per page."))
    return sections


def generate_pdf_report(
    result: dict,
    output_dir: Path | str = "reports",
    report_title: str = "DataLens AI Dataset Report",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = safe_text(result.get("dataset_id", "report")).strip() or "report"
    dataset_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in dataset_id)
    output_path = output_dir / f"{dataset_id}_report.pdf"

    dataset_label = safe_text(result.get("filename", "Untitled"))
    composer = Composer(output_path, title=report_title, dataset_label=dataset_label)

    try:
        # 1. Cover
        _cover_page(composer, result)

        # 2. Contents
        _toc_page(composer, _build_toc_sections(result))

        # 3+. Content sections
        _render_executive_summary(composer, result)
        _render_quality(composer, result)
        _render_signals(composer, result)
        _render_cleaning(composer, result)
        _render_deep_stats(composer, result)
        _render_modeling(composer, result)
        _render_time_text(composer, result)
        _render_ai_report(composer, result)

        # n. Visual evidence
        _render_charts(composer, result)
    finally:
        composer.close()

    return output_path
