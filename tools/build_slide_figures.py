"""
Build slide visualizations for the end-semester deck (slides 9 through 16).

Run from repo root:
    python tools/build_slide_figures.py
Outputs:
    reports/ppts/figures/slide_09_pipeline.png
    reports/ppts/figures/slide_10_ingest_quality.png
    reports/ppts/figures/slide_11_deep_statistics.png
    reports/ppts/figures/slide_12_anomaly_ensemble.png
    reports/ppts/figures/slide_13_timeseries_drift.png
    reports/ppts/figures/slide_14_ml_leaderboard.png
    reports/ppts/figures/slide_15_shap.png
    reports/ppts/figures/slide_16_agentic_rag.png

All figures render in the editorial-dark palette already used by
modules/visualizer.py and the React dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "ppts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Editorial palette (mirrors modules/visualizer.py) ────────────────────────

BG_PAGE = "#0c0c0c"
BG_PANEL = "#141414"
LINE = "#272727"
LINE_2 = "#3a3a3a"
INK_1 = "#f5efe6"
INK_2 = "#c9c1b4"
INK_3 = "#8a8478"
ACCENT = "#5eead4"
ACCENT_2 = "#84d8b8"
WARN = "#f5b14a"
BAD = "#f08080"
OK = "#84d8b8"
LAVENDER = "#a78bfa"
COOL_CYAN = "#7dd3fc"

CAT_PALETTE = [ACCENT, ACCENT_2, WARN, "#c9c1b4", LAVENDER, BAD, COOL_CYAN, "#facc15"]

TEAL_RAMP = LinearSegmentedColormap.from_list(
    "teal_ramp",
    ["#0f1f1d", "#16403c", "#1f6961", "#2a9387", "#5eead4", "#a8f1e3"],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging",
    [BAD, WARN, "#3a3a3a", ACCENT_2, ACCENT],
)

plt.rcParams.update(
    {
        "figure.facecolor": BG_PAGE,
        "savefig.facecolor": BG_PAGE,
        "axes.facecolor": BG_PANEL,
        "axes.edgecolor": LINE_2,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK_1,
        "axes.linewidth": 0.8,
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "grid.color": LINE,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "font.family": "sans-serif",
        "font.size": 11,
        "text.color": INK_1,
        "legend.facecolor": BG_PANEL,
        "legend.edgecolor": LINE_2,
        "legend.fontsize": 9,
        "figure.dpi": 180,
    }
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        facecolor=BG_PAGE,
        edgecolor="none",
    )
    plt.close(fig)
    return path


def _title(ax, title, subtitle=None):
    ax.set_title(title, fontsize=15, fontweight="bold", color=INK_1, pad=14, loc="left")
    if subtitle:
        ax.text(
            0,
            1.04,
            subtitle,
            transform=ax.transAxes,
            fontsize=10,
            color=INK_3,
            fontweight="600",
            ha="left",
            va="bottom",
            family="monospace",
        )


def _round_box(ax, x, y, w, h, *, fc, ec=LINE_2, lw=0.8):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    return box


def _arrow(ax, xy_from, xy_to, *, color=ACCENT, lw=1.4, mut=14):
    arrow = FancyArrowPatch(
        xy_from,
        xy_to,
        arrowstyle="-|>,head_length=8,head_width=5",
        color=color,
        linewidth=lw,
        mutation_scale=mut,
        zorder=3,
    )
    ax.add_patch(arrow)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9 — Six-phase pipeline
# ─────────────────────────────────────────────────────────────────────────────


def fig_slide_09_pipeline():
    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.85])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")

    fig.text(
        0.04,
        0.94,
        "DataLens AI · Six-Phase Pipeline",
        fontsize=18,
        fontweight="900",
        color=INK_1,
    )
    fig.text(
        0.04,
        0.905,
        "40+ modules · parallel where safe · every claim traced to a formula",
        fontsize=10,
        color=INK_3,
        family="monospace",
    )

    phases = [
        {
            "tag": "P1",
            "title": "INGEST",
            "items": [
                "load_dataset",
                "build_profile",
                "infer_column_roles",
                "detect_signals",
                "select_analyses",
            ],
            "color": ACCENT,
        },
        {
            "tag": "P2",
            "title": "QUALITY",
            "items": [
                "calculate_health",
                "ml_readiness",
                "advanced_indicators",
                "auto-cleaner prep",
            ],
            "color": ACCENT_2,
        },
        {
            "tag": "P3",
            "title": "PARALLEL",
            "items": [
                "deep_statistics_v2",
                "anomaly_ensemble",
                "time_series",
                "text_profile",
            ],
            "color": WARN,
            "parallel": True,
        },
        {
            "tag": "P4",
            "title": "ML CHAIN",
            "items": [
                "model_leaderboard",
                "explain_winner (SHAP)",
                "permutation importance",
            ],
            "color": LAVENDER,
        },
        {
            "tag": "P5",
            "title": "REPORTING",
            "items": [
                "chart_planner",
                "visualizer (20+ charts)",
                "pdf_report_builder",
                "cleaned CSV",
            ],
            "color": COOL_CYAN,
        },
        {
            "tag": "P6",
            "title": "AGENTIC Q&A",
            "items": [
                "agent_brief",
                "rag_index",
                "ai_agent loop",
                "safe_pandas",
            ],
            "color": BAD,
        },
    ]

    box_w = 2.6
    box_h = 4.2
    start_x = 0.6
    gap = 0.3
    y = 2.2

    centers = []

    for i, p in enumerate(phases):
        x = start_x + i * (box_w + gap)
        cx = x + box_w / 2
        centers.append(cx)

        # outer card
        _round_box(ax, x, y, box_w, box_h, fc=BG_PANEL, ec=p["color"], lw=1.4)

        # phase tag chip
        chip_w, chip_h = 0.65, 0.42
        _round_box(
            ax,
            x + 0.18,
            y + box_h - chip_h - 0.18,
            chip_w,
            chip_h,
            fc=p["color"],
            ec="none",
        )
        ax.text(
            x + 0.18 + chip_w / 2,
            y + box_h - chip_h / 2 - 0.18,
            p["tag"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="900",
            color=BG_PAGE,
            family="monospace",
        )

        # title
        ax.text(
            cx,
            y + box_h - 0.95,
            p["title"],
            ha="center",
            va="top",
            fontsize=12,
            fontweight="800",
            color=INK_1,
        )

        # items
        for j, item in enumerate(p["items"]):
            ax.text(
                x + 0.22,
                y + box_h - 1.55 - j * 0.45,
                f"·  {item}",
                ha="left",
                va="top",
                fontsize=10,
                color=INK_2,
                family="monospace",
            )

        # parallel marker for P3
        if p.get("parallel"):
            ax.text(
                cx,
                y + 0.32,
                "ThreadPoolExecutor (4 workers)",
                ha="center",
                va="bottom",
                fontsize=8,
                color=WARN,
                family="monospace",
                fontweight="700",
            )

    # arrows between phases
    for i in range(len(phases) - 1):
        x1 = start_x + i * (box_w + gap) + box_w
        x2 = start_x + (i + 1) * (box_w + gap)
        _arrow(ax, (x1 - 0.02, y + box_h / 2), (x2 + 0.02, y + box_h / 2), color=INK_3, lw=1.6)

    # surface labels: dashboard above, formats below
    ax.text(
        9,
        7.6,
        "React Dashboard (17 tabs) · Streamlit Console · Flask API :5055",
        ha="center",
        fontsize=10,
        color=INK_3,
        family="monospace",
    )
    ax.text(
        9,
        1.4,
        "JSON contract · 20+ chart PNGs · cleaned CSV · multi-page PDF report",
        ha="center",
        fontsize=10,
        color=INK_3,
        family="monospace",
    )

    # safety footer
    fig.text(
        0.04,
        0.04,
        "_safe_call() wraps every parallel branch — single failure ⇒ "
        "{ \"available\": false } instead of sinking the run.",
        fontsize=9,
        color=INK_3,
        family="monospace",
    )

    return _save(fig, "slide_09_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 10 — Ingest mechanics + Quality formulas
# ─────────────────────────────────────────────────────────────────────────────


def fig_slide_10_ingest_quality():
    fig = plt.figure(figsize=(15, 8))

    fig.text(0.04, 0.94, "Ingest Mechanics + Quality Formulas",
             fontsize=18, fontweight="900", color=INK_1)
    fig.text(0.04, 0.905, "encoding waterfall · CSV-injection guard · weighted health · readiness penalty model",
             fontsize=10, color=INK_3, family="monospace")

    # ── LEFT: Encoding waterfall + CSV guard ─────────────────────────
    ax1 = fig.add_axes([0.04, 0.08, 0.4, 0.78])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis("off")
    ax1.text(0, 11.5, "Encoding waterfall", fontsize=13, fontweight="800", color=INK_1)
    ax1.text(0, 11.0, "loader.py · stops at first decode that succeeds",
             fontsize=9, color=INK_3, family="monospace")

    encodings = [
        ("utf-8-sig", "BOM-tolerant default", ACCENT),
        ("utf-8",     "modern fallback",      ACCENT_2),
        ("windows-1252", "Excel CSVs",        WARN),
        ("latin-1",   "always succeeds",      BAD),
    ]
    yc = 9.5
    for i, (name, note, color) in enumerate(encodings):
        y = yc - i * 1.55
        _round_box(ax1, 0.5, y, 8.5, 1.05, fc=BG_PANEL, ec=color, lw=1.4)
        ax1.text(0.9, y + 0.55, name, fontsize=12, fontweight="800",
                 color=color, family="monospace", va="center")
        ax1.text(4.0, y + 0.55, note, fontsize=10, color=INK_2, va="center")
        ax1.text(8.4, y + 0.55, f"step {i+1}", fontsize=8, color=INK_3,
                 family="monospace", va="center", ha="right")
        if i < 3:
            _arrow(ax1, (5.0, y), (5.0, y - 0.5), color=INK_3, lw=1.0)

    # CSV guard pictogram
    ax1.text(0, 2.6, "CSV-injection guard", fontsize=12, fontweight="800", color=INK_1)
    ax1.text(0, 2.2, "security.py · prefixes formula triggers with a single quote",
             fontsize=9, color=INK_3, family="monospace")

    triggers = ["=SUM(A1)", "+1+1", "-2*3", "@cmd"]
    safed    = ["'=SUM(A1)", "'+1+1", "'-2*3", "'@cmd"]
    for j, (raw, safe) in enumerate(zip(triggers, safed)):
        y = 1.0
        _round_box(ax1, 0.2 + j * 2.3, y, 1.0, 0.55, fc="#3a1d1d", ec=BAD, lw=1.0)
        ax1.text(0.7 + j * 2.3, y + 0.27, raw, fontsize=9, color=BAD,
                 ha="center", va="center", family="monospace", fontweight="700")
        _arrow(ax1, (1.25 + j * 2.3, y + 0.28), (1.4 + j * 2.3, y + 0.28),
               color=INK_3, lw=1.0, mut=10)
        _round_box(ax1, 1.4 + j * 2.3, y, 1.0, 0.55, fc="#1d3a2e", ec=ACCENT, lw=1.0)
        ax1.text(1.9 + j * 2.3, y + 0.27, safe, fontsize=9, color=ACCENT,
                 ha="center", va="center", family="monospace", fontweight="700")

    # ── RIGHT-TOP: Health stacked weights ───────────────────────────
    ax2 = fig.add_axes([0.5, 0.5, 0.46, 0.36])
    weights = [("Completeness", 0.30, ACCENT),
               ("Consistency",  0.20, ACCENT_2),
               ("Uniqueness",   0.20, WARN),
               ("Outlier safety", 0.20, LAVENDER),
               ("Floor (+10)",  0.10, INK_3)]
    left = 0
    for name, w, color in weights:
        ax2.barh(0, w, left=left, color=color, height=0.55,
                 edgecolor=BG_PAGE, linewidth=1.5)
        ax2.text(left + w / 2, 0, f"{name}\n{int(w*100)}%",
                 ha="center", va="center", fontsize=10,
                 color=BG_PAGE if color != INK_3 else INK_1,
                 fontweight="800")
        left += w
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_yticks([])
    ax2.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.set_title("Health = 0.30·Completeness + 0.20·Consistency + 0.20·Uniqueness + 0.20·Safety + 10",
                  fontsize=11, fontweight="700", color=INK_1, loc="left", pad=14)
    ax2.text(0, -0.5,
             "Labels:  ≥90 Excellent · ≥75 Good · ≥60 Moderate · ≥40 Poor · else Critical",
             fontsize=9, color=INK_3, family="monospace")

    # ── RIGHT-BOTTOM: ML Readiness penalty bar ──────────────────────
    ax3 = fig.add_axes([0.5, 0.08, 0.46, 0.36])
    full = 100
    pen_missing = 22      # example values
    pen_dup = 8
    pen_cat = 14
    score = full - pen_missing - pen_dup - pen_cat

    # remaining + penalties as a stacked horizontal
    ax3.barh(0, score, color=ACCENT, height=0.55, edgecolor=BG_PAGE, linewidth=1.5)
    ax3.barh(0, pen_missing, left=score, color=BAD, height=0.55,
             edgecolor=BG_PAGE, linewidth=1.5)
    ax3.barh(0, pen_dup, left=score + pen_missing, color=WARN, height=0.55,
             edgecolor=BG_PAGE, linewidth=1.5)
    ax3.barh(0, pen_cat, left=score + pen_missing + pen_dup, color=LAVENDER,
             height=0.55, edgecolor=BG_PAGE, linewidth=1.5)

    ax3.text(score / 2, 0, f"Score\n{score}", ha="center", va="center",
             fontsize=12, fontweight="900", color=BG_PAGE)
    ax3.text(score + pen_missing / 2, 0, f"−{pen_missing}\nmissing",
             ha="center", va="center", fontsize=9, fontweight="700", color=BG_PAGE)
    ax3.text(score + pen_missing + pen_dup / 2, 0, f"−{pen_dup}\ndupes",
             ha="center", va="center", fontsize=9, fontweight="700", color=BG_PAGE)
    ax3.text(score + pen_missing + pen_dup + pen_cat / 2, 0, f"−{pen_cat}\ncats",
             ha="center", va="center", fontsize=9, fontweight="700", color=BG_PAGE)

    ax3.set_xlim(0, 100)
    ax3.set_ylim(-0.6, 0.7)
    ax3.set_yticks([])
    ax3.set_xticks([0, 25, 50, 75, 100])
    for s in ("top", "right", "left"):
        ax3.spines[s].set_visible(False)
    ax3.spines["bottom"].set_color(LINE_2)
    ax3.set_title(
        "Readiness = 100 − min(30, Missing) − min(15, Duplicates) − min(20, 2·|Categorical|)",
        fontsize=11, fontweight="700", color=INK_1, loc="left", pad=14
    )
    ax3.text(0, -0.5,
             "Labels:  ≥85 Ready · ≥70 Mostly Ready · ≥50 Partially · else Not Ready",
             fontsize=9, color=INK_3, family="monospace")

    return _save(fig, "slide_10_ingest_quality.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 11 — Deep statistics
# ─────────────────────────────────────────────────────────────────────────────


def fig_slide_11_deep_statistics():
    fig = plt.figure(figsize=(15, 8.5))
    fig.text(0.04, 0.95, "Deep Statistics — univariate shapes, distribution fitting, leakage & VIF",
             fontsize=16, fontweight="900", color=INK_1)
    fig.text(0.04, 0.92, "skewness γ₁ · kurtosis γ₂ · AIC fit · Pearson r ≥ 0.98 → leakage · VIF = 1/(1−R²)",
             fontsize=10, color=INK_3, family="monospace")

    # 4 subplots in 2x2
    rng = np.random.default_rng(7)

    # (1) Skew + kurtosis row
    ax1 = fig.add_axes([0.05, 0.55, 0.27, 0.32])
    x = np.linspace(-5, 8, 500)
    ax1.plot(x, stats.norm.pdf(x, 0, 1), color=ACCENT_2, lw=2.2,
             label="γ₁ ≈ 0  (symmetric)")
    ax1.plot(x, stats.skewnorm.pdf(x, 6, loc=-1.5, scale=1.3),
             color=BAD, lw=2.2, label="γ₁ > 1  (right-skewed)")
    ax1.plot(x, stats.skewnorm.pdf(x, -6, loc=1.5, scale=1.3),
             color=WARN, lw=2.2, label="γ₁ < −1  (left-skewed)")
    ax1.fill_between(x, stats.norm.pdf(x, 0, 1), color=ACCENT_2, alpha=0.12)
    ax1.set_yticks([])
    for s in ("top", "right", "left"):
        ax1.spines[s].set_visible(False)
    ax1.spines["bottom"].set_color(LINE_2)
    ax1.legend(frameon=False, loc="upper right", labelcolor=INK_2, fontsize=9)
    _title(ax1, "Skewness γ₁ = E[(X−μ)³]/σ³",
           "asymmetry of the distribution")

    # (2) Kurtosis
    ax2 = fig.add_axes([0.37, 0.55, 0.27, 0.32])
    x = np.linspace(-6, 6, 500)
    ax2.plot(x, stats.norm.pdf(x, 0, 1), color=ACCENT_2, lw=2.2, label="γ₂ ≈ 0  (mesokurtic)")
    ax2.plot(x, stats.t.pdf(x, df=3), color=BAD, lw=2.2, label="γ₂ > 1  (heavy-tailed)")
    ax2.plot(x, stats.uniform.pdf(x, loc=-2, scale=4) + 0.001,
             color=WARN, lw=2.2, label="γ₂ < −1  (light-tailed)")
    ax2.set_yticks([])
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(LINE_2)
    ax2.legend(frameon=False, loc="upper right", labelcolor=INK_2, fontsize=9)
    _title(ax2, "Kurtosis γ₂ = E[(X−μ)⁴]/σ⁴ − 3", "tail-heaviness")

    # (3) AIC distribution fit
    ax3 = fig.add_axes([0.69, 0.55, 0.27, 0.32])
    sample = rng.gamma(2.5, 1.5, 1500)
    ax3.hist(sample, bins=40, color=BG_PANEL, edgecolor=ACCENT_2,
             linewidth=0.6, density=True, alpha=0.55, label="data")
    xs = np.linspace(0.01, sample.max(), 400)
    fits = [
        ("gamma",   stats.gamma,   ACCENT,  "AIC 4818"),
        ("lognorm", stats.lognorm, WARN,    "AIC 4862"),
        ("expon",   stats.expon,   BAD,     "AIC 5104"),
    ]
    for name, dist, color, aic in fits:
        params = dist.fit(sample)
        ax3.plot(xs, dist.pdf(xs, *params), color=color, lw=2.0,
                 label=f"{name}  ·  {aic}")
    ax3.set_yticks([])
    for s in ("top", "right", "left"):
        ax3.spines[s].set_visible(False)
    ax3.spines["bottom"].set_color(LINE_2)
    ax3.legend(frameon=False, loc="upper right", labelcolor=INK_2, fontsize=9)
    _title(ax3, "Best-fit by AIC", "AIC = 2k − 2 ln L̂ · lower wins")

    # (4) Leakage scatter
    ax4 = fig.add_axes([0.05, 0.08, 0.42, 0.36])
    n = 300
    x_safe = rng.normal(0, 1, n)
    y_safe = 0.4 * x_safe + rng.normal(0, 1, n)
    ax4.scatter(x_safe, y_safe, color=ACCENT, s=14, alpha=0.55,
                edgecolor=BG_PAGE, linewidth=0.3, label="r ≈ 0.36 — useful feature")

    x_leak = rng.normal(0, 1, n)
    y_leak = x_leak + rng.normal(0, 0.05, n)  # leakage
    ax4.scatter(x_leak + 4, y_leak + 4, color=BAD, s=14, alpha=0.85,
                edgecolor=BG_PAGE, linewidth=0.3, label="r ≈ 0.99 — LEAKAGE")
    ax4.legend(frameon=False, loc="upper left", labelcolor=INK_2, fontsize=9)
    ax4.set_xlabel("feature")
    ax4.set_ylabel("target")
    ax4.grid(True, alpha=0.5)
    _title(ax4, "Target leakage probe",
           "abs(Pearson r) ≥ 0.98 on overlap ≥ 10  →  flagged Critical/High")

    # (5) VIF severity bars
    ax5 = fig.add_axes([0.55, 0.08, 0.41, 0.36])
    vif_labels = ["age", "income", "score_raw", "score_norm", "balance", "city_code"]
    vif_values = [1.4, 2.8, 11.6, 12.1, 4.2, 1.2]
    colors = [BAD if v >= 10 else WARN if v >= 5 else ACCENT for v in vif_values]
    bars = ax5.barh(vif_labels[::-1], vif_values[::-1], color=colors[::-1],
                    edgecolor=BG_PAGE, linewidth=1.0, height=0.55)
    for b, v in zip(bars, vif_values[::-1]):
        ax5.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2,
                 f"{v:.1f}", va="center", color=INK_1,
                 fontsize=10, fontweight="800")
    ax5.axvline(5, ls="--", color=WARN, lw=0.8, alpha=0.7)
    ax5.axvline(10, ls="--", color=BAD, lw=0.8, alpha=0.7)
    ax5.set_xlabel("VIFⱼ = 1 / (1 − R²ⱼ)")
    ax5.set_xlim(0, 14)
    ax5.grid(axis="x", alpha=0.45)
    ax5.grid(axis="y", visible=False)
    legend = [
        mpatches.Patch(facecolor=ACCENT, label="Low (< 5)"),
        mpatches.Patch(facecolor=WARN, label="Medium (≥ 5)"),
        mpatches.Patch(facecolor=BAD, label="High (≥ 10)"),
    ]
    ax5.legend(handles=legend, frameon=False, loc="lower right",
               labelcolor=INK_2, fontsize=9)
    _title(ax5, "Multicollinearity (VIF)",
           "auxiliary regression of feature j on all others")

    return _save(fig, "slide_11_deep_statistics.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 12 — Anomaly ensemble
# ─────────────────────────────────────────────────────────────────────────────


def fig_slide_12_anomaly_ensemble():
    fig = plt.figure(figsize=(15, 8.5))
    fig.text(0.04, 0.95, "Anomaly Ensemble — five detectors averaged into a [0,1] consensus",
             fontsize=16, fontweight="900", color=INK_1)
    fig.text(0.04, 0.92,
             "min-max normalize each detector  →  s(x) = mean across available detectors  →  flag if s ≥ 0.6",
             fontsize=10, color=INK_3, family="monospace")

    rng = np.random.default_rng(11)
    n_rows = 18

    # synthetic per-detector scores (rows × detectors), with one obvious anomaly
    iso = np.clip(rng.beta(2, 6, n_rows), 0, 1)
    lof = np.clip(rng.beta(2, 5, n_rows), 0, 1)
    elliptic = np.clip(rng.beta(2, 5, n_rows), 0, 1)
    mad = np.clip(rng.beta(2, 6, n_rows), 0, 1)
    mahal = np.clip(rng.beta(2, 5, n_rows), 0, 1)

    # promote a few rows to genuine anomalies
    for idx, lift in [(2, 0.6), (5, 0.7), (11, 0.55), (16, 0.65)]:
        for arr in (iso, lof, elliptic, mad, mahal):
            arr[idx] = min(1.0, arr[idx] + lift + rng.uniform(-0.05, 0.05))

    matrix = np.column_stack([iso, lof, elliptic, mad, mahal])
    ensemble = matrix.mean(axis=1)

    detector_names = [
        "Isolation Forest",
        "Local Outlier Factor",
        "Elliptic Envelope",
        "Robust z-score (MAD)",
        "Mahalanobis (MinCovDet)",
    ]

    # ── Heatmap of detector scores ────────────────────────────────
    ax1 = fig.add_axes([0.05, 0.5, 0.55, 0.4])
    im = ax1.imshow(matrix.T, aspect="auto", cmap=TEAL_RAMP, vmin=0, vmax=1)
    ax1.set_yticks(range(len(detector_names)))
    ax1.set_yticklabels(detector_names, fontsize=10, color=INK_2)
    ax1.set_xticks(range(n_rows))
    ax1.set_xticklabels([f"r{r}" for r in range(n_rows)], fontsize=8,
                        color=INK_3, family="monospace")
    ax1.tick_params(axis="x", which="both", bottom=False)
    for s in ("top", "right", "left", "bottom"):
        ax1.spines[s].set_visible(False)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
    cbar.set_label("normalized score", color=INK_2, fontsize=9)
    cbar.ax.tick_params(colors=INK_3)
    cbar.outline.set_visible(False)
    _title(ax1, "Per-detector scores (rows × detectors)",
           "each row independently min-max scaled to [0,1]")

    # ── Ensemble bars below heatmap ──────────────────────────────
    ax2 = fig.add_axes([0.05, 0.08, 0.55, 0.34])
    colors = [BAD if v >= 0.6 else WARN if v >= 0.4 else ACCENT for v in ensemble]
    bars = ax2.bar(range(n_rows), ensemble, color=colors,
                   edgecolor=BG_PAGE, linewidth=1.0, width=0.7)
    for b, v in zip(bars, ensemble):
        if v >= 0.6:
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.02,
                     "FLAG", ha="center", fontsize=8,
                     color=BAD, fontweight="900", family="monospace")
    ax2.axhline(0.6, color=BAD, ls="--", lw=1.0, alpha=0.85,
                label="threshold s = 0.6")
    ax2.set_xticks(range(n_rows))
    ax2.set_xticklabels([f"r{r}" for r in range(n_rows)], fontsize=8,
                        color=INK_3, family="monospace")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("ensemble score s(x)")
    ax2.grid(axis="y", alpha=0.45)
    ax2.legend(frameon=False, loc="upper right", labelcolor=INK_2, fontsize=9)
    _title(ax2, "Ensemble = mean across detectors",
           "rows above 0.6 enter top_rows + flagged_count")

    # ── Right: detector formulas ────────────────────────────────
    ax3 = fig.add_axes([0.66, 0.08, 0.32, 0.82])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 12)
    ax3.axis("off")
    ax3.text(0, 11.7, "How each detector scores a row",
             fontsize=12, fontweight="800", color=INK_1)

    formulas = [
        ("Isolation Forest", "s(x) = 2^(−E[h(x)]/c(n))",
         "shorter average path length ⇒ more anomalous", ACCENT),
        ("Local Outlier Factor",
         "LOF_k(x) = mean( lrd(y∈N_k) / lrd(x) )",
         "ratio of local density vs k neighbours", ACCENT_2),
        ("Elliptic Envelope",
         "raw = −score_samples (robust Gaussian fit)",
         "tail of the fitted ellipse", WARN),
        ("Robust z (MAD)",
         "z* = (x − median) / (1.4826·MAD), flag |z*|>3.5",
         "1.4826 makes MAD ≈ σ for normals", LAVENDER),
        ("Mahalanobis",
         "D²(x) = (x−μ)ᵀ Σ⁻¹ (x−μ),  Σ via MinCovDet",
         "robust covariance suppresses outliers", BAD),
    ]

    y0 = 10.5
    for i, (name, formula, blurb, color) in enumerate(formulas):
        y = y0 - i * 2.1
        _round_box(ax3, 0.1, y - 1.55, 9.6, 1.85, fc=BG_PANEL, ec=color, lw=1.2)
        ax3.text(0.4, y, name, fontsize=11, fontweight="800", color=color)
        ax3.text(0.4, y - 0.5, formula, fontsize=10, color=INK_1,
                 family="monospace")
        ax3.text(0.4, y - 1.05, blurb, fontsize=9, color=INK_3,
                 family="monospace")

    return _save(fig, "slide_12_anomaly_ensemble.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 13 — Time-series + drift
# ─────────────────────────────────────────────────────────────────────────────


def fig_slide_13_timeseries_drift():
    fig = plt.figure(figsize=(15, 9))
    fig.text(0.04, 0.96, "Time-Series + Drift — STL · ADF/KPSS · Holt-Winters · PSI",
             fontsize=16, fontweight="900", color=INK_1)
    fig.text(0.04, 0.93,
             "stationarity battery · additive seasonal smoothing · distribution shift between halves",
             fontsize=10, color=INK_3, family="monospace")

    rng = np.random.default_rng(13)
    n = 220
    t = np.arange(n)
    trend = 0.04 * t
    seasonal = 2.4 * np.sin(2 * np.pi * t / 30)
    noise = rng.normal(0, 0.6, n)
    series = 5 + trend + seasonal + noise

    # Holt-Winters style forecast (cheap analytic mock)
    hw = 5 + 0.04 * np.arange(n, n + 20) + 2.4 * np.sin(2 * np.pi * np.arange(n, n + 20) / 30)

    # ── Top-left: STL-style 4-panel ──────────────────────────────
    gs = [
        ("observed", series, ACCENT),
        ("trend (T)", 5 + trend, ACCENT_2),
        ("seasonal (S)", seasonal, WARN),
        ("residual (R)", noise, BAD),
    ]
    ax_h = 0.155
    for i, (name, arr, color) in enumerate(gs):
        ax = fig.add_axes([0.05, 0.51 + (3 - i) * (ax_h + 0.005), 0.55, ax_h])
        ax.plot(arr, color=color, lw=1.4)
        ax.fill_between(np.arange(len(arr)), arr, alpha=0.08, color=color)
        ax.set_ylabel(name, fontsize=9, color=INK_2)
        ax.grid(axis="y", alpha=0.45)
        ax.tick_params(labelsize=8)
        if i < 3:
            ax.set_xticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i == 0:
            ax.set_title(
                "STL decomposition  ·  F_T = 1 − Var(R)/Var(T+R)  ·  F_S = 1 − Var(R)/Var(S+R)",
                fontsize=11, fontweight="700", color=INK_1, loc="left", pad=8)

    # ── Top-right: ADF / KPSS verdict matrix ─────────────────────
    ax_v = fig.add_axes([0.66, 0.62, 0.32, 0.28])
    cells = [
        ["stationary", "trend-stationary"],
        ["difference-stationary", "non-stationary"],
    ]
    cell_colors = [
        [ACCENT, WARN],
        [LAVENDER, BAD],
    ]
    for r in range(2):
        for c in range(2):
            ax_v.add_patch(plt.Rectangle((c, 1 - r), 1, 1,
                                         facecolor=cell_colors[r][c],
                                         edgecolor=BG_PAGE, lw=2.5))
            ax_v.text(c + 0.5, 1 - r + 0.55, cells[r][c],
                      ha="center", va="center",
                      fontsize=10, fontweight="800", color=BG_PAGE)
            ax_v.text(c + 0.5, 1 - r + 0.20,
                      ["ADF rej · KPSS not-rej", "ADF rej · KPSS rej",
                       "ADF not-rej · KPSS not-rej", "ADF not-rej · KPSS rej"][r * 2 + c],
                      ha="center", va="center", fontsize=8,
                      color=BG_PAGE, family="monospace")
    ax_v.set_xticks([0.5, 1.5])
    ax_v.set_yticks([0.5, 1.5])
    ax_v.set_xticklabels(["KPSS: stationary\n(p>0.05)", "KPSS: non-stationary\n(p≤0.05)"],
                         color=INK_2, fontsize=9)
    ax_v.set_yticklabels(["ADF: non-stat.\n(p>0.05)", "ADF: stationary\n(p<0.05)"],
                         color=INK_2, fontsize=9)
    ax_v.tick_params(length=0)
    ax_v.set_xlim(0, 2)
    ax_v.set_ylim(0, 2)
    for s in ("top", "right", "left", "bottom"):
        ax_v.spines[s].set_visible(False)
    ax_v.set_title("Stationarity verdict matrix", fontsize=11,
                   fontweight="700", color=INK_1, loc="left", pad=10)

    # ── Bottom-right: Holt-Winters forecast ──────────────────────
    ax_hw = fig.add_axes([0.66, 0.30, 0.32, 0.24])
    ax_hw.plot(np.arange(n), series, color=INK_3, lw=1.0, alpha=0.6, label="actual")
    ax_hw.plot(np.arange(n - 20, n), series[-20:], color=ACCENT, lw=1.6,
               label="train (last 20)")
    ax_hw.plot(np.arange(n, n + 20), hw, color=WARN, lw=2.0,
               label="Holt-Winters")
    ax_hw.plot(np.arange(n, n + 20), [series[-1]] * 20, color=BAD, lw=1.4,
               ls="--", label="naive baseline")
    ax_hw.set_xlim(n - 60, n + 20)
    ax_hw.legend(frameon=False, loc="upper left", labelcolor=INK_2, fontsize=8)
    ax_hw.grid(alpha=0.4)
    ax_hw.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax_hw.spines[s].set_visible(False)
    ax_hw.set_title("Holt-Winters additive vs naive (MAE / RMSE on tail)",
                    fontsize=11, fontweight="700", color=INK_1, loc="left", pad=8)

    # ── Bottom-left: PSI dual histogram ─────────────────────────
    ax_psi = fig.add_axes([0.05, 0.07, 0.55, 0.32])
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(0.5, 1.15, 1000)
    edges = np.linspace(-4, 5, 11)
    ref_h, _ = np.histogram(ref, bins=edges)
    cur_h, _ = np.histogram(cur, bins=edges)
    width = (edges[1] - edges[0]) * 0.42
    centers = (edges[:-1] + edges[1:]) / 2
    ax_psi.bar(centers - width / 2, ref_h / ref_h.sum(), width=width,
               color=ACCENT, alpha=0.85, label="reference (older half)",
               edgecolor=BG_PAGE, linewidth=0.6)
    ax_psi.bar(centers + width / 2, cur_h / cur_h.sum(), width=width,
               color=WARN, alpha=0.85, label="current (newer half)",
               edgecolor=BG_PAGE, linewidth=0.6)

    # Compute PSI
    rp = np.where(ref_h == 0, 1e-6, ref_h / ref_h.sum())
    cp = np.where(cur_h == 0, 1e-6, cur_h / cur_h.sum())
    psi = float(((cp - rp) * np.log(cp / rp)).sum())

    ax_psi.set_xlabel("value")
    ax_psi.set_ylabel("proportion in bin")
    ax_psi.legend(frameon=False, loc="upper right", labelcolor=INK_2, fontsize=9)
    ax_psi.grid(axis="y", alpha=0.4)
    for s in ("top", "right"):
        ax_psi.spines[s].set_visible(False)
    _title(ax_psi,
           f"PSI = Σ (p_cur − p_ref) · ln(p_cur / p_ref)  =  {psi:.3f}  →  moderate drift",
           "buckets: <0.10 stable · <0.25 minor · <0.50 moderate · ≥0.50 severe")

    # ── Far-right bottom: severity legend bar ──────────────────
    ax_sev = fig.add_axes([0.66, 0.07, 0.32, 0.18])
    bands = [(0, 0.10, "stable", ACCENT),
             (0.10, 0.25, "minor", ACCENT_2),
             (0.25, 0.50, "moderate", WARN),
             (0.50, 0.80, "severe", BAD)]
    for lo, hi, name, color in bands:
        ax_sev.barh(0, hi - lo, left=lo, color=color, height=0.45,
                    edgecolor=BG_PAGE, linewidth=1.2)
        ax_sev.text(lo + (hi - lo) / 2, 0, name, ha="center", va="center",
                    fontsize=10, fontweight="800",
                    color=BG_PAGE if name != "stable" else INK_1)
    # Marker for our computed PSI
    ax_sev.axvline(psi, color=INK_1, lw=1.6)
    ax_sev.scatter([psi], [0.32], color=INK_1, s=40, zorder=4)
    ax_sev.text(psi, 0.45, f"this run\nPSI={psi:.2f}", ha="center", color=INK_1,
                fontsize=8.5, family="monospace", fontweight="700")
    ax_sev.set_xlim(0, 0.8)
    ax_sev.set_ylim(-0.5, 0.7)
    ax_sev.set_yticks([])
    ax_sev.set_xticks([0.10, 0.25, 0.50])
    ax_sev.set_xticklabels(["0.10", "0.25", "0.50"], color=INK_3, fontsize=8)
    for s in ("top", "right", "left", "bottom"):
        ax_sev.spines[s].set_visible(False)
    ax_sev.set_title("PSI severity buckets", fontsize=10,
                     fontweight="700", color=INK_1, loc="left", pad=8)

    return _save(fig, "slide_13_timeseries_drift.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 14 — Leaderboard + diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def fig_slide_14_ml_leaderboard():
    fig = plt.figure(figsize=(15, 8.5))
    fig.text(0.04, 0.95, "ML Lab — leaderboard, confusion matrix, residual diagnostics",
             fontsize=16, fontweight="900", color=INK_1)
    fig.text(0.04, 0.92,
             "5-fold StratifiedKFold (clf) / KFold (reg) · winner refit on 75% · holdout on 25%",
             fontsize=10, color=INK_3, family="monospace")

    rng = np.random.default_rng(14)

    # ── Left: leaderboard horizontal bars ───────────────────────
    models = [
        "Dummy", "KNeighbors", "LogisticReg", "Lasso (reg)",
        "GradientBoosting", "RandomForest", "LightGBM", "XGBoost"
    ]
    scores = [0.34, 0.71, 0.78, 0.74, 0.83, 0.85, 0.872, 0.881]
    cv_std = [0.04, 0.05, 0.04, 0.06, 0.03, 0.025, 0.022, 0.020]
    winner = scores.index(max(scores))
    colors = [INK_3 if "Dummy" in m else (ACCENT if i == winner else ACCENT_2)
              for i, m in enumerate(models)]

    ax1 = fig.add_axes([0.05, 0.52, 0.44, 0.40])
    y_pos = np.arange(len(models))
    bars = ax1.barh(y_pos, scores, xerr=cv_std, color=colors,
                    edgecolor=BG_PAGE, linewidth=1.0,
                    error_kw={"ecolor": INK_3, "elinewidth": 1.2,
                              "capsize": 3, "alpha": 0.8})
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=10, color=INK_2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("F1 weighted (mean ± std across 5 folds)")
    ax1.grid(axis="x", alpha=0.45)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    for b, s, std in zip(bars, scores, cv_std):
        ax1.text(b.get_width() + std + 0.015, b.get_y() + b.get_height() / 2,
                 f"{s:.3f}", va="center", color=INK_1,
                 fontsize=9.5, fontweight="800")
    # winner badge
    ax1.text(scores[winner] / 2, winner, "WINNER",
             ha="center", va="center", fontsize=10, fontweight="900",
             color=BG_PAGE)
    _title(ax1, "Leaderboard (classification example)",
           "primary metric = F1_weighted · CV (k=5 stratified) · best non-dummy wins")

    # ── Right top: confusion matrix ─────────────────────────────
    ax2 = fig.add_axes([0.55, 0.52, 0.42, 0.40])
    cm = np.array([
        [142, 6, 2, 1],
        [9, 118, 8, 3],
        [3, 11, 96, 7],
        [1, 4, 9, 87],
    ])
    im = ax2.imshow(cm, cmap=TEAL_RAMP, aspect="auto")
    classes = ["A", "B", "C", "D"]
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(classes, color=INK_2)
    ax2.set_yticklabels(classes, color=INK_2)
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("actual")
    for i in range(4):
        for j in range(4):
            v = cm[i, j]
            color = BG_PAGE if v >= 60 else INK_1
            ax2.text(j, i, str(v), ha="center", va="center",
                     fontsize=12, fontweight="800", color=color)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_3)
    for s in ("top", "right", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    _title(ax2, "Confusion matrix on holdout",
           "diagonal = correct · off-diagonal = misclassifications")

    # ── Bottom-left: residuals scatter + histogram ──────────────
    ax3 = fig.add_axes([0.05, 0.08, 0.4, 0.34])
    n = 240
    y_true = rng.uniform(40, 95, n)
    y_pred = y_true + rng.normal(0, 4.0, n)
    # plant 8 worst predictions
    bad_idx = rng.choice(n, 8, replace=False)
    y_pred[bad_idx] = y_true[bad_idx] + rng.choice([-1, 1], 8) * rng.uniform(15, 25, 8)

    residuals = y_true - y_pred
    ax3.scatter(y_pred, residuals, color=ACCENT, s=18, alpha=0.55,
                edgecolor=BG_PAGE, linewidth=0.3)
    ax3.scatter(y_pred[bad_idx], residuals[bad_idx], color=BAD, s=40,
                edgecolor=BG_PAGE, linewidth=0.5, zorder=4,
                label="top 10 worst predictions")
    ax3.axhline(0, color=INK_3, lw=0.8)
    ax3.set_xlabel("predicted ŷ")
    ax3.set_ylabel("residual y − ŷ")
    ax3.legend(frameon=False, loc="upper right", labelcolor=INK_2, fontsize=9)
    ax3.grid(alpha=0.45)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    _title(ax3, "Regression residuals",
           "residual_summary = {mean, std, min, max} · top-10 by |y − ŷ|")

    # ── Bottom-right: metric panel as KPI tiles ────────────────
    ax4 = fig.add_axes([0.51, 0.08, 0.46, 0.34])
    ax4.set_xlim(0, 12)
    ax4.set_ylim(0, 6)
    ax4.axis("off")
    _title(ax4, "Metrics catalogue",
           "all reported per fold, then mean ± std")

    tiles = [
        ("Accuracy", "(TP+TN)/N", ACCENT),
        ("Precision", "TP/(TP+FP)", ACCENT_2),
        ("Recall", "TP/(TP+FN)", WARN),
        ("F1 weighted", "Σ wᵢ·F1ᵢ", LAVENDER),
        ("Brier", "(1/n)·Σ(pᵢ−yᵢ)²", BAD),
        ("R²", "1 − SS_res/SS_tot", ACCENT),
        ("MAE", "(1/n)·Σ|y−ŷ|", ACCENT_2),
        ("RMSE", "√(1/n·Σ(y−ŷ)²)", WARN),
    ]
    for i, (name, formula, color) in enumerate(tiles):
        col = i % 4
        row = i // 4
        x = 0.2 + col * 2.95
        y = 4.4 - row * 2.5
        _round_box(ax4, x, y, 2.6, 2.0, fc=BG_PANEL, ec=color, lw=1.2)
        ax4.text(x + 0.2, y + 1.5, name, fontsize=11, fontweight="800", color=color)
        ax4.text(x + 0.2, y + 1.0, formula, fontsize=10,
                 color=INK_1, family="monospace")
        ax4.text(x + 0.2, y + 0.45,
                 "task: classification" if i < 5 else "task: regression",
                 fontsize=8, color=INK_3, family="monospace")

    return _save(fig, "slide_14_ml_leaderboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 15 — SHAP explainability
# ─────────────────────────────────────────────────────────────────────────────


def fig_slide_15_shap():
    fig = plt.figure(figsize=(15, 8.5))
    fig.text(0.04, 0.95, "SHAP Explainability — global ranking, per-row stories, beeswarm",
             fontsize=16, fontweight="900", color=INK_1)
    fig.text(0.04, 0.92,
             "exact poly-time SHAP for tree models · LinearExplainer for linear · "
             "one-hot dummies summed back to parent",
             fontsize=10, color=INK_3, family="monospace")

    rng = np.random.default_rng(15)

    features = ["location", "sqft", "bedrooms", "age", "school_score",
                "garage", "lot_size", "renovated", "tax_band", "crime_idx"]
    mean_abs = np.array([0.42, 0.38, 0.21, 0.18, 0.16, 0.12, 0.11, 0.08, 0.06, 0.05])

    # ── Left: global mean |SHAP| bars ────────────────────────────
    ax1 = fig.add_axes([0.05, 0.52, 0.42, 0.40])
    colors = [TEAL_RAMP(v / mean_abs.max()) for v in mean_abs[::-1]]
    ax1.barh(features[::-1], mean_abs[::-1], color=colors,
             edgecolor=BG_PAGE, linewidth=0.8)
    for i, v in enumerate(mean_abs[::-1]):
        ax1.text(v + 0.005, i, f"{v:.2f}", va="center", color=INK_1,
                 fontsize=9, fontweight="700")
    ax1.set_xlabel("mean |SHAP value|")
    ax1.grid(axis="x", alpha=0.4)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    _title(ax1, "Global feature importance",
           "average magnitude of each feature's contribution across rows")

    # ── Right: SHAP-style beeswarm ──────────────────────────────
    ax2 = fig.add_axes([0.52, 0.52, 0.45, 0.40])
    n_dots = 200
    for i, feat in enumerate(features):
        spread = mean_abs[i] * 1.4
        x = rng.normal(0, spread, n_dots)
        # synthetic feature value coloring
        value = rng.uniform(0, 1, n_dots)
        y = np.full(n_dots, len(features) - 1 - i) + rng.normal(0, 0.12, n_dots)
        ax2.scatter(x, y, c=value, cmap=DIVERGING, s=10, alpha=0.7,
                    edgecolor="none")
    ax2.axvline(0, color=INK_3, lw=0.8)
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels(features[::-1], color=INK_2, fontsize=9)
    ax2.set_xlabel("SHAP value (impact on model output)")
    ax2.grid(axis="x", alpha=0.4)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    # colorbar for feature value
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.set_label("feature value", color=INK_2, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_3)
    _title(ax2, "Beeswarm",
           "x = SHAP impact · y = feature · color = feature value")

    # ── Bottom: per-row force chart ──────────────────────────────
    ax3 = fig.add_axes([0.05, 0.08, 0.92, 0.32])
    base = 280  # E[ŷ]  (in $K)
    contribs = [
        ("location",      +45.2, ACCENT),
        ("sqft",          +28.0, ACCENT_2),
        ("school_score",  +14.5, OK),
        ("renovated",      +8.4, ACCENT),
        ("age",          -22.3, BAD),
        ("crime_idx",    -12.1, WARN),
        ("tax_band",      -6.4, BAD),
    ]
    final = base + sum(c for _, c, _ in contribs)

    # left anchor at base, walk right with stacked colored segments
    cursor = base
    bar_y = 0.5
    bar_h = 0.6

    # base bar
    ax3.barh([bar_y], [base], color=INK_3, edgecolor=BG_PAGE, linewidth=1.2,
             height=bar_h, alpha=0.65)
    ax3.text(base / 2, bar_y, f"E[ŷ]  ${base}K",
             ha="center", va="center", fontsize=11, fontweight="800",
             color=BG_PAGE)

    for name, c, color in contribs:
        if c >= 0:
            ax3.barh([bar_y], [c], left=cursor, color=color,
                     edgecolor=BG_PAGE, linewidth=1.5, height=bar_h)
            ax3.text(cursor + c / 2, bar_y,
                     f"+{c:.1f}\n{name}", ha="center", va="center",
                     fontsize=9, fontweight="700", color=BG_PAGE)
        else:
            # subtract: show as a hatched segment below the line for visual contrast
            ax3.barh([bar_y], [abs(c)], left=cursor + c, color=color,
                     edgecolor=BG_PAGE, linewidth=1.5, height=bar_h, alpha=0.85)
            ax3.text(cursor + c + abs(c) / 2, bar_y,
                     f"{c:.1f}\n{name}", ha="center", va="center",
                     fontsize=9, fontweight="700", color=BG_PAGE)
        cursor += c

    # final marker
    ax3.axvline(final, color=ACCENT, lw=2)
    ax3.text(final, bar_y + bar_h * 0.85, f"ŷ = ${final:.1f}K",
             ha="center", va="bottom", color=ACCENT,
             fontsize=12, fontweight="900")
    ax3.axvline(base, color=INK_3, lw=1, ls="--")

    ax3.set_xlim(220, max(final, base) + 30)
    ax3.set_ylim(0, 1.3)
    ax3.set_yticks([])
    for s in ("top", "right", "left"):
        ax3.spines[s].set_visible(False)
    ax3.spines["bottom"].set_color(LINE_2)
    ax3.set_xlabel("predicted price ($K)")
    _title(ax3,
           "Per-row story  ·  Σ φᵢ(x) + E[ŷ]  =  ŷ(x)  (local-accuracy property)",
           "the same arithmetic the dashboard reads as: "
           "‘Price was increased by $45.2K due to Location’")

    return _save(fig, "slide_15_shap.png")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 16 — Reporting + Agentic / RAG
# ─────────────────────────────────────────────────────────────────────────────


def fig_slide_16_agentic_rag():
    fig = plt.figure(figsize=(15, 8.5))
    fig.text(0.04, 0.95, "Reporting + Agentic Layer — chart planner, RAG, sandboxed pandas",
             fontsize=16, fontweight="900", color=INK_1)
    fig.text(0.04, 0.92,
             "role-aware chart selection · 2400-char fact block · cosine top-k · AST allow-list",
             fontsize=10, color=INK_3, family="monospace")

    # ── Top: chart planner decision strip ─────────────────────────
    ax1 = fig.add_axes([0.05, 0.59, 0.92, 0.31])
    ax1.set_xlim(0, 22)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    _title(ax1, "Chart planner decision strip",
           "build_chart_plan reads roles + signals + ML context")

    inputs = [
        ("column_roles", ACCENT),
        ("dataset_signals", ACCENT_2),
        ("target_column", WARN),
        ("model_leaderboard", LAVENDER),
        ("time_series", COOL_CYAN),
    ]
    for i, (name, color) in enumerate(inputs):
        y = 4.4 - i * 0.7
        _round_box(ax1, 0.3, y - 0.28, 3.6, 0.55, fc=BG_PANEL, ec=color, lw=1.0)
        ax1.text(0.6, y, name, fontsize=10, color=color,
                 fontweight="700", family="monospace", va="center")
        _arrow(ax1, (3.95, y), (5.6, 3), color=INK_3, lw=1.0, mut=10)

    # central planner box
    _round_box(ax1, 5.6, 1.7, 4.6, 2.6, fc=BG_PANEL, ec=INK_1, lw=1.6)
    ax1.text(7.9, 3.5, "build_chart_plan", fontsize=12, fontweight="900",
             color=INK_1, ha="center")
    ax1.text(7.9, 3.0, "skips meaningless charts", fontsize=9, color=INK_3,
             ha="center", family="monospace")
    ax1.text(7.9, 2.5, "(no histogram of an ID col)", fontsize=9, color=INK_3,
             ha="center", family="monospace")
    ax1.text(7.9, 2.0, "→ list[{type,title,…}]", fontsize=9, color=ACCENT,
             ha="center", family="monospace", fontweight="700")

    # output chart pictograms
    out_charts = [
        ("histogram", "▌▆█▇▄▂", ACCENT),
        ("boxplot", "├──┤", ACCENT_2),
        ("violin", "◆", ACCENT_2),
        ("heatmap", "▦", WARN),
        ("pareto", "▮▮▮▏", WARN),
        ("leaderboard", "█▆▆▄▃", LAVENDER),
        ("SHAP bars", "▆▅▃▂", LAVENDER),
        ("time-series", "∿", COOL_CYAN),
    ]
    for i, (name, glyph, color) in enumerate(out_charts):
        col = i % 4
        row = i // 4
        x = 11.2 + col * 2.5
        y = 4.0 - row * 1.7
        _round_box(ax1, x, y - 0.5, 2.2, 1.2, fc=BG_PANEL, ec=color, lw=1.0)
        ax1.text(x + 1.1, y + 0.35, glyph, fontsize=15,
                 ha="center", color=color, fontweight="800")
        ax1.text(x + 1.1, y - 0.15, name, fontsize=9, ha="center",
                 color=INK_2, family="monospace")
    _arrow(ax1, (10.25, 3), (11.05, 3), color=ACCENT, lw=1.6)

    # ── Bottom-left: RAG flow ─────────────────────────────────────
    ax2 = fig.add_axes([0.04, 0.07, 0.6, 0.45])
    ax2.set_xlim(0, 16)
    ax2.set_ylim(0, 8)
    ax2.axis("off")
    _title(ax2, "RAG fact retrieval",
           "build_fact_index → embed → cosine top-k → render → writer")

    steps = [
        ("analysis_result", "JSON dict\n~700–1300 leaves", ACCENT),
        ("Fact[]", "build_fact_index\nflatten + filter", ACCENT_2),
        ("Embeddings", "Ollama nomic-embed-text\nor TF-IDF fallback", WARN),
        ("Cosine top-k", "sim(q, cᵢ) = q·cᵢ / (‖q‖‖cᵢ‖)", LAVENDER),
        ("Writer prompt", "≤2400-char fact block\n+ dataset brief", ACCENT),
    ]
    box_w = 2.7
    box_h = 1.7
    for i, (name, sub, color) in enumerate(steps):
        x = 0.2 + i * 3.15
        y = 4.0
        _round_box(ax2, x, y, box_w, box_h, fc=BG_PANEL, ec=color, lw=1.2)
        ax2.text(x + box_w / 2, y + box_h - 0.5, name, ha="center",
                 fontsize=10, fontweight="800", color=color)
        ax2.text(x + box_w / 2, y + 0.45, sub, ha="center",
                 fontsize=8.5, color=INK_3, family="monospace")
        if i < len(steps) - 1:
            _arrow(ax2, (x + box_w, y + box_h / 2),
                   (x + box_w + 0.45, y + box_h / 2), color=INK_3, lw=1.2)

    # cosine geometry illustration below
    ax2.text(0.5, 2.8, "cosine similarity intuition",
             fontsize=10, fontweight="700", color=INK_1)
    ax2.annotate("", xy=(3.5, 0.8), xytext=(1.0, 0.8),
                 arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))
    ax2.annotate("", xy=(3.0, 2.3), xytext=(1.0, 0.8),
                 arrowprops=dict(arrowstyle="-|>", color=ACCENT_2, lw=2))
    ax2.annotate("", xy=(3.4, 1.5), xytext=(1.0, 0.8),
                 arrowprops=dict(arrowstyle="-|>", color=WARN, lw=2))
    ax2.text(3.6, 0.8, "fact 1  ·  sim ≈ 0.92", color=ACCENT,
             fontsize=9, family="monospace", va="center")
    ax2.text(3.6, 1.5, "fact 2  ·  sim ≈ 0.71", color=WARN,
             fontsize=9, family="monospace", va="center")
    ax2.text(3.6, 2.3, "fact 3  ·  sim ≈ 0.51", color=ACCENT_2,
             fontsize=9, family="monospace", va="center")
    ax2.text(0.6, 0.4, "q (query vector)", fontsize=9, color=INK_3,
             family="monospace")
    ax2.text(8.0, 1.5,
             "top-k retains the\nlargest cos angles\nas the most-relevant\nevidence",
             fontsize=9, color=INK_3, family="monospace")

    # ── Bottom-right: safe_pandas sandbox card ───────────────────
    ax3 = fig.add_axes([0.66, 0.07, 0.32, 0.45])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 8.5)
    ax3.axis("off")
    _title(ax3, "safe_pandas sandbox",
           "AST allow-list · no built-ins · no IO")

    # allowed
    _round_box(ax3, 0.3, 4.5, 9.4, 3.4, fc="#1d3a2e", ec=ACCENT, lw=1.2)
    ax3.text(0.6, 7.4, "ALLOWED", fontsize=10, fontweight="900",
             color=ACCENT, family="monospace")
    ax3.text(0.6, 6.9, "df, np, pd, len, abs, min, max, round, sum, sorted",
             fontsize=8.5, color=INK_2, family="monospace")
    ax3.text(0.6, 6.4, "loc · iloc · groupby · agg · describe",
             fontsize=8.5, color=INK_2, family="monospace")
    ax3.text(0.6, 5.9, "value_counts · sort_values · query",
             fontsize=8.5, color=INK_2, family="monospace")
    ax3.text(0.6, 5.4, "str · dt · cat accessor methods",
             fontsize=8.5, color=INK_2, family="monospace")
    ax3.text(0.6, 4.8,
             "MAX_LEN 600 · MAX_DEPTH 30 · 50-row preview cap",
             fontsize=8, color=INK_3, family="monospace")

    # blocked
    _round_box(ax3, 0.3, 0.4, 9.4, 3.7, fc="#3a1d1d", ec=BAD, lw=1.2)
    ax3.text(0.6, 3.6, "BLOCKED", fontsize=10, fontweight="900",
             color=BAD, family="monospace")
    ax3.text(0.6, 3.1, "import · from · lambda · exec · eval · open",
             fontsize=8.5, color=INK_2, family="monospace")
    ax3.text(0.6, 2.6, "globals · locals · compile · __dunders__",
             fontsize=8.5, color=INK_2, family="monospace")
    ax3.text(0.6, 2.1, "any attribute starting with '_'",
             fontsize=8.5, color=INK_2, family="monospace")
    ax3.text(0.6, 1.6, "any name not in the allow-list",
             fontsize=8.5, color=INK_2, family="monospace")
    ax3.text(0.6, 0.9,
             'eval(compiled, {"__builtins__": {}}, locals)',
             fontsize=8, color=INK_3, family="monospace")

    return _save(fig, "slide_16_agentic_rag.png")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def main():
    builders = [
        fig_slide_09_pipeline,
        fig_slide_10_ingest_quality,
        fig_slide_11_deep_statistics,
        fig_slide_12_anomaly_ensemble,
        fig_slide_13_timeseries_drift,
        fig_slide_14_ml_leaderboard,
        fig_slide_15_shap,
        fig_slide_16_agentic_rag,
    ]
    for fn in builders:
        path = fn()
        print(f"  PNG → {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
