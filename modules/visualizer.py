"""
Visualizer (v3 — editorial dark)
================================
Role-aware chart renderer driven by the chart planner.
Uses matplotlib + seaborn with the same cream-on-near-black + teal palette
the React dashboard uses. PNGs are saved with a dark background so they
sit cleanly inside both the dashboard cards and the modernized PDF report.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import seaborn as sns
import numpy as np
import pandas as pd

from modules.column_roles import infer_column_roles
from modules.chart_planner import build_chart_plan

CHART_DIR = Path("static/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Editorial palette  (mirror of frontend tokens in styles/globals.css)
# ---------------------------------------------------------------------------

BG_PAGE   = "#0c0c0c"  # page bg behind the chart
BG_PANEL  = "#141414"  # axes facecolor
LINE      = "#272727"  # subtle grid
LINE_2    = "#3a3a3a"  # axis spines
INK_1     = "#f5efe6"  # primary ink (titles)
INK_2     = "#c9c1b4"  # secondary ink (labels)
INK_3     = "#8a8478"  # tertiary ink (ticks)

ACCENT    = "#5eead4"  # teal accent (single accent rule)
ACCENT_2  = "#84d8b8"  # softer teal
WARN      = "#f5b14a"
BAD       = "#f08080"
OK        = "#84d8b8"

# Categorical sequence — editorial-but-readable. Avoids fully-saturated rainbow.
CAT_PALETTE = [
    "#5eead4",  # teal accent
    "#84d8b8",  # mint
    "#f5b14a",  # warm amber
    "#c9c1b4",  # warm grey
    "#a78bfa",  # lavender (use sparingly)
    "#f08080",  # warm red
    "#7dd3fc",  # cool cyan
    "#facc15",  # gold
    "#fb923c",  # orange
    "#94a3b8",  # cool grey
]

# Sequential teal ramp for heatmaps / continuous bars
TEAL_RAMP = ["#0f1f1d", "#16403c", "#1f6961", "#2a9387", "#5eead4", "#a8f1e3"]
DIVERGING_RAMP = ["#f08080", "#f5b14a", "#3a3a3a", "#84d8b8", "#5eead4"]


sns.set_theme(style="white", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor":   BG_PAGE,
    "savefig.facecolor":  BG_PAGE,
    "axes.facecolor":     BG_PANEL,
    "axes.edgecolor":     LINE_2,
    "axes.linewidth":     0.8,
    "axes.labelcolor":    INK_2,
    "axes.titlecolor":    INK_1,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.labelweight":   "600",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.bottom": True,
    "axes.spines.left":   True,
    "xtick.color":        INK_3,
    "ytick.color":        INK_3,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "grid.color":         LINE,
    "grid.linewidth":     0.6,
    "grid.alpha":         0.6,
    "font.family":        "sans-serif",
    "font.size":          11,
    "text.color":         INK_1,
    "legend.facecolor":   BG_PANEL,
    "legend.edgecolor":   LINE_2,
    "legend.fontsize":    10,
    "figure.dpi":         180,
})


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _save(dataset_id, name):
    safe = str(name).replace("/", "_").replace(" ", "_")
    path = CHART_DIR / f"{dataset_id}_{safe}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=BG_PAGE, edgecolor="none")
    plt.close()
    return f"charts/{dataset_id}_{safe}.png"


def _title(ax, title, subtitle=None):
    ax.set_title(title, fontsize=14, fontweight="bold",
                 color=INK_1, pad=14, loc="left")
    if subtitle:
        ax.text(0, 1.04, subtitle, transform=ax.transAxes,
                fontsize=10, color=INK_3, fontweight="600",
                ha="left", va="bottom",
                family="monospace")


def _grid_y_only(ax):
    ax.grid(axis="y", color=LINE, linewidth=0.6, alpha=0.55, zorder=0)
    ax.grid(axis="x", visible=False)


def _value_labels(ax, bars, fmt=lambda v: f"{v:,.0f}", color=INK_1):
    """Place small data labels above each bar."""
    for b in bars:
        h = b.get_height()
        if h is None or (isinstance(h, float) and (np.isnan(h) or np.isinf(h))):
            continue
        ax.text(b.get_x() + b.get_width() / 2, h, fmt(h),
                ha="center", va="bottom",
                fontsize=9, fontweight="700",
                color=color)


def _safe_numeric_series(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    return s


# ---------------------------------------------------------------------------
# Quality / structure
# ---------------------------------------------------------------------------

def _chart_health(did, health):
    comps = health.get("components", {}) or {}
    if not comps:
        return None
    labels = [k.replace("_", " ").title() for k in comps]
    vals = [float(v) for v in comps.values()]
    colors = [OK if v >= 80 else WARN if v >= 60 else BAD for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor=BG_PAGE,
                  linewidth=1.5, width=0.55, zorder=3)
    _value_labels(ax, bars, fmt=lambda v: f"{v:.0f}")
    _title(ax, "Dataset Health Components",
           "completeness · consistency · uniqueness · outliers")
    _grid_y_only(ax)
    ax.set_ylabel("Score", color=INK_2, fontweight="700")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=15)
    return {"title": "Dataset Health Components", "type": "health_components",
            "description": "Completeness, consistency, uniqueness, and outlier safety.",
            "path": _save(did, "health_components")}


def _chart_dtype_breakdown(did, df):
    if df.empty or len(df.columns) == 0:
        return None
    dtype_buckets: dict[str, int] = {}
    for c in df.columns:
        dt = df[c].dtype
        if pd.api.types.is_bool_dtype(dt):
            label = "boolean"
        elif pd.api.types.is_integer_dtype(dt):
            label = "integer"
        elif pd.api.types.is_float_dtype(dt):
            label = "float"
        elif pd.api.types.is_datetime64_any_dtype(dt):
            label = "datetime"
        elif pd.api.types.is_categorical_dtype(dt):
            label = "categorical"
        else:
            label = "object/text"
        dtype_buckets[label] = dtype_buckets.get(label, 0) + 1

    items = sorted(dtype_buckets.items(), key=lambda kv: kv[1], reverse=True)
    labels = [f"{k}  ·  {v}" for k, v in items]
    sizes = [v for _, v in items]
    colors = CAT_PALETTE[: len(sizes)]

    fig, ax = plt.subplots(figsize=(8, 5))
    wedges, _texts = ax.pie(
        sizes, colors=colors, startangle=90, counterclock=False,
        wedgeprops={"width": 0.36, "edgecolor": BG_PAGE, "linewidth": 2},
    )
    ax.text(0, 0.05, str(sum(sizes)), ha="center", va="center",
            fontsize=30, fontweight="900", color=INK_1)
    ax.text(0, -0.18, "columns", ha="center", va="center",
            fontsize=10, color=INK_3, fontweight="700",
            family="monospace")
    ax.legend(wedges, labels, loc="center left",
              bbox_to_anchor=(1.05, 0.5), frameon=False,
              labelcolor=INK_2, fontsize=10)
    ax.set_title("Schema Composition", fontsize=14, fontweight="bold",
                 color=INK_1, pad=14, loc="left")
    return {"title": "Schema Composition", "type": "dtype_breakdown",
            "description": "Distribution of column dtypes in the dataset.",
            "path": _save(did, "dtype_breakdown")}


def _chart_missing(did, df):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(20)
    if missing.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(missing.index.astype(str), missing.values,
                  color=ACCENT, edgecolor=BG_PAGE, linewidth=1.0, zorder=3)
    _value_labels(ax, bars, fmt=lambda v: f"{int(v):,}")
    _title(ax, "Missing Values by Column",
           "raw missing-cell count, top 20")
    _grid_y_only(ax)
    ax.set_ylabel("Missing count", color=INK_2, fontweight="700")
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")
    return {"title": "Missing Values by Column", "type": "missing_values",
            "description": "Columns with missing data and severity.",
            "path": _save(did, "missing_values")}


def _chart_cleaning_risk(did, df):
    pct = df.isna().mean() * 100
    pct = pct[pct > 0].sort_values(ascending=False).head(20)
    if pct.empty:
        return None
    colors = [BAD if v >= 50 else WARN if v >= 20 else ACCENT for v in pct.values]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(pct.index.astype(str), pct.values, color=colors,
                  edgecolor=BG_PAGE, linewidth=1.0, zorder=3)
    _value_labels(ax, bars, fmt=lambda v: f"{v:.0f}%")
    _title(ax, "Cleaning Risk by Missingness %",
           ">20% caution · >50% manual review")
    _grid_y_only(ax)
    ax.axhline(20, ls="--", lw=0.8, color=WARN, alpha=0.7)
    ax.axhline(50, ls="--", lw=0.8, color=BAD, alpha=0.7)
    ax.set_ylabel("Missing %", color=INK_2, fontweight="700")
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")
    return {"title": "Cleaning Risk", "type": "cleaning_risk",
            "description": "Columns above 20% require caution; above 50% need manual review.",
            "path": _save(did, "cleaning_risk")}


def _chart_cardinality_strip(did, df):
    if df.empty or len(df.columns) == 0:
        return None
    n_rows = max(1, len(df))
    pairs: list[tuple[str, int, float]] = []
    for c in df.columns:
        try:
            nun = int(df[c].nunique(dropna=True))
        except Exception:
            continue
        pairs.append((c, nun, nun / n_rows))
    if not pairs:
        return None
    pairs.sort(key=lambda t: t[1], reverse=True)
    pairs = pairs[:18]

    cols = [p[0] for p in pairs]
    n_unique = [p[1] for p in pairs]
    ratio = [p[2] for p in pairs]

    def _color(r, n):
        if n <= 1:
            return BAD
        if r >= 0.95:
            return WARN  # ID-like / near unique → not useful for grouping
        if n <= 5:
            return ACCENT
        return ACCENT_2

    colors = [_color(r, n) for r, n in zip(ratio, n_unique)]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.barh(cols[::-1], n_unique[::-1], color=colors[::-1],
                   edgecolor=BG_PAGE, linewidth=0.8, zorder=3)
    for b, n in zip(bars, n_unique[::-1]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f"  {n:,}", ha="left", va="center",
                fontsize=9, color=INK_1, fontweight="700")
    _title(ax, "Column Cardinality",
           "amber = ID-like / near-unique · red = constant")
    ax.set_xlabel("Unique values", color=INK_2, fontweight="700")
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.55, zorder=0)
    ax.grid(axis="y", visible=False)

    legend = [
        Patch(facecolor=ACCENT,   edgecolor="none", label="useful (low/mid card.)"),
        Patch(facecolor=ACCENT_2, edgecolor="none", label="moderate"),
        Patch(facecolor=WARN,     edgecolor="none", label="ID-like (≥95% unique)"),
        Patch(facecolor=BAD,      edgecolor="none", label="constant"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=9)
    return {"title": "Column Cardinality", "type": "cardinality_strip",
            "description": "Number of unique values per column. Flags ID-like and constant columns.",
            "path": _save(did, "cardinality_strip")}


# ---------------------------------------------------------------------------
# Numeric distributions
# ---------------------------------------------------------------------------

def _chart_distribution(did, df, col):
    s = _safe_numeric_series(df[col])
    if s is None or s.nunique() < 2:
        return None
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.histplot(s, kde=True, bins=35, color=ACCENT, edgecolor=BG_PAGE,
                 linewidth=0.6, alpha=0.85, ax=ax,
                 line_kws={"color": ACCENT_2, "linewidth": 2.4})
    mean_v, med_v = float(s.mean()), float(s.median())
    ax.axvline(med_v, ls="--", lw=1, color=INK_2, alpha=0.7,
               label=f"median {med_v:,.2f}")
    ax.axvline(mean_v, ls=":", lw=1, color=WARN, alpha=0.85,
               label=f"mean {mean_v:,.2f}")
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_2)
    _title(ax, f"Distribution of {col}",
           f"n = {len(s):,} · σ = {float(s.std()):,.3f}")
    _grid_y_only(ax)
    ax.set_xlabel(col, color=INK_2, fontweight="700")
    ax.set_ylabel("Count", color=INK_2, fontweight="700")
    return {"title": f"Distribution of {col}", "type": "numeric_distribution",
            "column": col,
            "description": f"Distribution pattern of '{col}'.",
            "path": _save(did, f"dist_{col}")}


def _chart_boxplot(did, df, col):
    s = _safe_numeric_series(df[col])
    if s is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(y=s, color=ACCENT, width=0.4, ax=ax,
                boxprops={"edgecolor": INK_2, "linewidth": 1.0},
                whiskerprops={"color": INK_2, "linewidth": 1.0},
                capprops={"color": INK_2, "linewidth": 1.0},
                medianprops={"color": INK_1, "linewidth": 1.6},
                flierprops={"marker": "o", "markerfacecolor": BAD,
                            "markeredgecolor": "none", "markersize": 5,
                            "alpha": 0.85})
    _title(ax, f"Outlier Review: {col}",
           "box = IQR · whiskers = 1.5×IQR")
    ax.set_ylabel(col, color=INK_2, fontweight="700")
    ax.grid(axis="y", color=LINE, linewidth=0.6, alpha=0.55, zorder=0)
    return {"title": f"Outlier Review: {col}", "type": "boxplot",
            "column": col,
            "description": f"Spread and outliers for '{col}'.",
            "path": _save(did, f"box_{col}")}


def _chart_violin(did, df, col):
    s = _safe_numeric_series(df[col])
    if s is None or s.nunique() < 5:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot([s.values], showmeans=False, showmedians=True,
                          widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(ACCENT)
        pc.set_edgecolor(ACCENT_2)
        pc.set_alpha(0.55)
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        if k in parts:
            parts[k].set_color(INK_2)
            parts[k].set_linewidth(1.2)
    ax.scatter([1], [float(s.mean())], color=WARN, zorder=3, s=22,
               label=f"mean {float(s.mean()):,.2f}")
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_2)
    ax.set_xticks([1])
    ax.set_xticklabels([col])
    _title(ax, f"Density Violin: {col}",
           "wider = more rows at that value")
    ax.grid(axis="y", color=LINE, linewidth=0.6, alpha=0.55, zorder=0)
    return {"title": f"Density Violin: {col}", "type": "violin",
            "column": col,
            "description": f"Density-aware shape of '{col}', median + mean overlay.",
            "path": _save(did, f"violin_{col}")}


def _chart_numeric_overview(did, df, columns):
    cols = [c for c in columns if c in df.columns][:6]
    series = [(c, _safe_numeric_series(df[c])) for c in cols]
    series = [(c, s) for c, s in series if s is not None and s.nunique() >= 2]
    if len(series) < 2:
        return None

    n = len(series)
    rows = 2 if n > 3 else 1
    cols_ = (n + 1) // 2 if rows == 2 else n
    fig, axes = plt.subplots(rows, cols_, figsize=(4.0 * cols_, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[len(series):]:
        ax.set_visible(False)
    for ax, (name, s) in zip(axes, series):
        ax.hist(s.values, bins=24, color=ACCENT, edgecolor=BG_PAGE,
                linewidth=0.5, alpha=0.85)
        ax.set_title(name, fontsize=11, color=INK_1, fontweight="700",
                     loc="left", pad=6)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color=LINE, linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_facecolor(BG_PANEL)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Numeric Columns at a Glance", fontsize=14, fontweight="bold",
                 color=INK_1, ha="left", x=0.06, y=0.995)
    return {"title": "Numeric Columns at a Glance", "type": "numeric_overview",
            "description": "Compact 6-up overview of the most analytically useful numerics.",
            "path": _save(did, "numeric_overview")}


def _chart_bid_ask(did, df, bid, ask):
    if bid not in df.columns or ask not in df.columns:
        return None
    spread = (df[ask] - df[bid]).dropna()
    if spread.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(spread, kde=True, bins=30, color=ACCENT, ax=ax,
                 alpha=0.85, edgecolor=BG_PAGE, linewidth=0.6,
                 line_kws={"color": ACCENT_2, "linewidth": 2.4})
    _title(ax, "Bid-Ask Spread", "ask − bid distribution")
    _grid_y_only(ax)
    ax.set_xlabel("Ask − Bid", color=INK_2, fontweight="700")
    return {"title": "Bid-Ask Spread", "type": "bid_ask_spread",
            "description": "Spread between ask and bid prices.",
            "path": _save(did, "bid_ask_spread")}


# ---------------------------------------------------------------------------
# Categoricals
# ---------------------------------------------------------------------------

def _chart_categorical(did, df, col):
    counts = df[col].astype(object).value_counts(dropna=False)
    if len(counts) < 2:
        return None
    counts = counts.head(12)
    n = len(counts)
    colors = (CAT_PALETTE * 3)[:n]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    labels = [str(x)[:22] for x in counts.index]
    bars = ax.bar(range(n), counts.values, color=colors,
                  edgecolor=BG_PAGE, linewidth=1.0, width=0.65, zorder=3)
    _value_labels(ax, bars, fmt=lambda v: f"{int(v):,}")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _title(ax, f"Categories in {col}",
           f"top {n} categories")
    _grid_y_only(ax)
    ax.set_ylabel("Count", color=INK_2, fontweight="700")
    return {"title": f"Categories in {col}", "type": "categorical_count",
            "column": col,
            "description": f"Category frequency for '{col}'.",
            "path": _save(did, f"cat_{col}")}


def _chart_pareto_categorical(did, df, col):
    counts = df[col].astype(object).value_counts(dropna=False)
    if len(counts) < 3:
        return None
    counts = counts.head(15)
    cum = counts.cumsum() / counts.sum() * 100

    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    bars = ax1.bar(range(len(counts)), counts.values, color=ACCENT,
                   edgecolor=BG_PAGE, linewidth=1.0, width=0.65, zorder=3)
    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels([str(x)[:18] for x in counts.index],
                        rotation=25, ha="right")
    ax1.set_ylabel("Count", color=INK_2, fontweight="700")
    _grid_y_only(ax1)

    ax2 = ax1.twinx()
    ax2.plot(range(len(counts)), cum.values, color=WARN, marker="o",
             markersize=5, linewidth=2.0, zorder=4,
             markerfacecolor=WARN, markeredgecolor=BG_PAGE)
    ax2.axhline(80, ls="--", lw=0.8, color=INK_3, alpha=0.7)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative %", color=WARN, fontweight="700")
    ax2.tick_params(axis="y", colors=WARN)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    _title(ax1, f"Pareto: {col}",
           "bars = frequency · line = cumulative coverage")
    return {"title": f"Pareto: {col}", "type": "pareto_categorical",
            "column": col,
            "description": f"Cumulative share of top categories in '{col}'.",
            "path": _save(did, f"pareto_{col}")}


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

def _chart_correlation(did, df, numeric_cols):
    if len(numeric_cols) < 2:
        return None
    cols = [c for c in numeric_cols if c in df.columns][:14]
    corr = df[cols].corr(numeric_only=True).dropna(how="all").dropna(axis=1, how="all")
    if corr.shape[0] < 2:
        return None
    cmap = sns.diverging_palette(15, 175, s=85, l=58, as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, fmt=".2f", center=0,
                square=True, linewidths=1, linecolor=BG_PAGE,
                cbar_kws={"shrink": 0.7, "label": "correlation"},
                annot_kws={"fontsize": 9, "fontweight": "bold", "color": INK_1},
                ax=ax)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.yaxis.set_tick_params(color=INK_3)
        cbar.set_label("correlation", color=INK_2, fontweight="700")
        plt.setp(cbar.ax.get_yticklabels(), color=INK_2)
    ax.tick_params(colors=INK_2)
    _title(ax, "Correlation Heatmap", "lower triangle, Pearson")
    return {"title": "Correlation Heatmap", "type": "correlation_heatmap",
            "description": "Pairwise relationships between numeric columns.",
            "path": _save(did, "correlation_heatmap")}


def _chart_top_correlations(did, df, numeric_cols):
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return None
    corr = df[cols].corr(numeric_only=True)
    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            v = corr.loc[a, b]
            if v is None or np.isnan(v):
                continue
            pairs.append((a, b, float(v)))
    if not pairs:
        return None
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    pairs = pairs[:10]

    labels = [f"{a} ↔ {b}" for a, b, _ in pairs]
    vals = [v for _, _, v in pairs]
    colors = [ACCENT if v >= 0 else BAD for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1],
                   edgecolor=BG_PAGE, linewidth=0.8, zorder=3)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2,
                f"  {v:+.2f}",
                ha="left" if v >= 0 else "right",
                va="center", fontsize=9, fontweight="700", color=INK_1)
    ax.axvline(0, color=LINE_2, lw=0.8)
    ax.set_xlim(-1.1, 1.1)
    _title(ax, "Top Correlations",
           "absolute strength · teal = positive · red = negative")
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.55, zorder=0)
    return {"title": "Top Correlations", "type": "top_correlations",
            "description": "Strongest pairwise relationships at a glance.",
            "path": _save(did, "top_correlations")}


def _chart_utility(did, advanced):
    utility = (advanced or {}).get("column_utility", []) or []
    if not utility:
        return None
    utility = sorted(utility, key=lambda x: x.get("score", 0), reverse=True)[:10]
    cols = [u["column"] for u in utility]
    scores = [float(u["score"]) for u in utility]
    colors = [ACCENT if s >= 80 else ACCENT_2 if s >= 50 else BAD for s in scores]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cols[::-1], scores[::-1], color=colors[::-1],
                   edgecolor=BG_PAGE, linewidth=0.8, zorder=3)
    for b, s in zip(bars, scores[::-1]):
        ax.text(s, b.get_y() + b.get_height() / 2,
                f"  {s:.0f}", ha="left", va="center",
                fontsize=9, fontweight="700", color=INK_1)
    ax.set_xlim(0, 110)
    _title(ax, "Column Utility Scores",
           "ranks columns by analytical usefulness")
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.55, zorder=0)
    return {"title": "Column Utility Scores", "type": "column_utility",
            "description": "Higher = more useful for downstream analysis / ML.",
            "path": _save(did, "column_utility")}


def _chart_cleaning_impact(did, cleaning):
    if not cleaning:
        return None
    before = float((cleaning.get("before_health") or {}).get("overall_score") or 0)
    after = float((cleaning.get("after_health") or {}).get("overall_score") or 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Before", "After"], [before, after],
                  color=[INK_3, ACCENT],
                  edgecolor=BG_PAGE, linewidth=1.2, width=0.5, zorder=3)
    _value_labels(ax, bars, fmt=lambda v: f"{v:.0f}")
    delta = after - before
    sign = "+" if delta >= 0 else ""
    ax.text(0.5, max(before, after) * 1.06,
            f"Δ {sign}{delta:.1f}",
            ha="center", color=ACCENT if delta >= 0 else BAD,
            fontsize=11, fontweight="800", transform=ax.transData)
    _title(ax, "Cleaning Impact", "health score before vs after")
    _grid_y_only(ax)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Health Score", color=INK_2, fontweight="700")
    return {"title": "Cleaning Impact", "type": "cleaning_impact",
            "description": "Quality score before and after cleaning.",
            "path": _save(did, "cleaning_impact")}


def _chart_anomaly(did, advanced):
    rows = (advanced or {}).get("anomalies", {}).get("top_rows", []) or []
    if not rows:
        return None
    rows = rows[:15]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([str(r.get("row_index", i)) for i, r in enumerate(rows)],
                  [float(r.get("score", 0)) for r in rows],
                  color=BAD, edgecolor=BG_PAGE, linewidth=0.8, zorder=3,
                  alpha=0.9)
    _value_labels(ax, bars, fmt=lambda v: f"{v:.2f}", color=INK_1)
    _title(ax, "Top Row Anomaly Scores", "row index → score (0..1)")
    _grid_y_only(ax)
    ax.set_xlabel("Row Index", color=INK_2, fontweight="700")
    ax.set_ylabel("Anomaly Score", color=INK_2, fontweight="700")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    return {"title": "Top Anomaly Scores", "type": "anomaly_scores",
            "description": "Rows with highest anomaly scores.",
            "path": _save(did, "anomaly_scores")}


# ---------------------------------------------------------------------------
# Target-aware
# ---------------------------------------------------------------------------

def _chart_target_distribution(did, df, target):
    if target not in df.columns:
        return None
    s = df[target]
    numeric = pd.to_numeric(s, errors="coerce")
    is_numeric = numeric.notna().sum() / max(1, len(s)) >= 0.85 and numeric.nunique() > 10

    fig, ax = plt.subplots(figsize=(10, 5.2))
    if is_numeric:
        s_num = numeric.dropna()
        sns.histplot(s_num, kde=True, bins=35, color=ACCENT, ax=ax,
                     alpha=0.85, edgecolor=BG_PAGE, linewidth=0.6,
                     line_kws={"color": ACCENT_2, "linewidth": 2.4})
        med = float(s_num.median())
        ax.axvline(med, ls="--", lw=1, color=INK_2, alpha=0.7,
                   label=f"median {med:,.2f}")
        ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_2)
        _title(ax, f"Target Distribution: {target}",
               f"regression target · n = {len(s_num):,}")
        ax.set_xlabel(target, color=INK_2, fontweight="700")
        ax.set_ylabel("Count", color=INK_2, fontweight="700")
    else:
        counts = s.astype(object).value_counts(dropna=False).head(12)
        n = len(counts)
        if n < 2:
            plt.close(fig)
            return None
        colors = (CAT_PALETTE * 3)[:n]
        bars = ax.bar(range(n), counts.values, color=colors,
                      edgecolor=BG_PAGE, linewidth=1.0, width=0.65, zorder=3)
        _value_labels(ax, bars, fmt=lambda v: f"{int(v):,}")
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(x)[:18] for x in counts.index],
                           rotation=20, ha="right")
        balance_ratio = counts.values.min() / counts.values.max()
        sub = f"classification target · balance {balance_ratio:.2f}"
        _title(ax, f"Target Distribution: {target}", sub)
        ax.set_ylabel("Count", color=INK_2, fontweight="700")
    _grid_y_only(ax)
    return {"title": f"Target Distribution: {target}", "type": "target_distribution",
            "column": target,
            "description": f"Shape and balance of the target '{target}'.",
            "path": _save(did, f"target_dist_{target}")}


def _chart_target_vs_feature(did, df, feature, target):
    if feature not in df.columns or target not in df.columns:
        return None
    f = pd.to_numeric(df[feature], errors="coerce")
    t = df[target]
    t_num = pd.to_numeric(t, errors="coerce")
    is_reg = t_num.notna().sum() / max(1, len(t)) >= 0.85 and t_num.nunique() > 10

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if is_reg:
        good = pd.DataFrame({"f": f, "t": t_num}).dropna()
        if len(good) < 5:
            plt.close(fig); return None
        ax.scatter(good["f"], good["t"], color=ACCENT,
                   edgecolor=BG_PAGE, linewidth=0.4, alpha=0.55, s=18)
        # quick linear fit for a guide
        try:
            m, b = np.polyfit(good["f"].values, good["t"].values, 1)
            xs = np.linspace(good["f"].min(), good["f"].max(), 100)
            ax.plot(xs, m * xs + b, color=WARN, lw=1.6, alpha=0.85,
                    label=f"y ≈ {m:.3f}x + {b:.2f}")
            ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_2)
        except Exception:
            pass
        ax.set_xlabel(feature, color=INK_2, fontweight="700")
        ax.set_ylabel(target, color=INK_2, fontweight="700")
        sub = f"scatter · n = {len(good):,}"
    else:
        # Numeric feature × categorical target → grouped distributions
        good = pd.DataFrame({"f": f, "t": t.astype(object)}).dropna(subset=["f"])
        groups = good["t"].value_counts().head(6).index.tolist()
        good = good[good["t"].isin(groups)]
        if len(good) < 10 or len(groups) < 2:
            plt.close(fig); return None
        data = [good.loc[good["t"] == g, "f"].values for g in groups]
        parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(CAT_PALETTE[i % len(CAT_PALETTE)])
            pc.set_edgecolor(LINE_2)
            pc.set_alpha(0.6)
        for k in ("cmedians", "cbars", "cmins", "cmaxes"):
            if k in parts:
                parts[k].set_color(INK_2); parts[k].set_linewidth(1.0)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([str(g)[:14] for g in groups], rotation=10)
        ax.set_xlabel(target, color=INK_2, fontweight="700")
        ax.set_ylabel(feature, color=INK_2, fontweight="700")
        sub = f"density per class · n = {len(good):,}"
    _title(ax, f"{feature} vs {target}", sub)
    _grid_y_only(ax)
    return {"title": f"{feature} vs {target}", "type": "target_vs_feature",
            "feature": feature, "target": target,
            "description": f"How '{feature}' moves with the target '{target}'.",
            "path": _save(did, f"tvf_{feature}_{target}")}


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

def _chart_leaderboard_bars(did, lb):
    rows = (lb or {}).get("leaderboard") or []
    rows = [r for r in rows if r.get("primary_score") is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: float(r["primary_score"]), reverse=True)[:8]
    names = [r.get("model", "?") for r in rows]
    scores = [float(r["primary_score"]) for r in rows]

    winner_name = ((lb.get("winner") or {}).get("model")) if lb.get("winner") else names[0]
    colors = [ACCENT if n == winner_name else INK_3 for n in names]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1],
                   edgecolor=BG_PAGE, linewidth=0.8, zorder=3)
    for b, v in zip(bars, scores[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2,
                f"  {v:.4f}", ha="left", va="center",
                fontsize=9, fontweight="700", color=INK_1)
    task = lb.get("task_type", "")
    target = lb.get("target_column", "")
    _title(ax, "Model Leaderboard",
           f"{task} · target = {target} · cross-validated primary score")
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.55, zorder=0)
    return {"title": "Model Leaderboard", "type": "leaderboard_bars",
            "description": "Cross-validated model scores. Winner highlighted in teal.",
            "path": _save(did, "leaderboard_bars")}


def _chart_feature_importance_bars(did, explain):
    items = (explain or {}).get("global_importance") or []
    if not items:
        return None
    items = sorted(items, key=lambda r: abs(float(r.get("importance", 0))),
                   reverse=True)[:12]
    feats = [r.get("feature", "?") for r in items]
    vals = [float(r.get("importance", 0)) for r in items]
    colors = [ACCENT if v >= 0 else BAD for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(feats[::-1], vals[::-1], color=colors[::-1],
                   edgecolor=BG_PAGE, linewidth=0.8, zorder=3)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2,
                f"  {v:+.4f}",
                ha="left" if v >= 0 else "right",
                va="center", fontsize=9, fontweight="700", color=INK_1)
    ax.axvline(0, color=LINE_2, lw=0.8)
    method = (explain.get("method") or "shap")
    model = (explain.get("model") or "")
    _title(ax, "Feature Importance",
           f"{method} on winner ({model})")
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.55, zorder=0)
    return {"title": "Feature Importance", "type": "feature_importance_bars",
            "description": "Mean |SHAP| or permutation importance from the leaderboard winner.",
            "path": _save(did, "feature_importance_bars")}


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def _chart_time_series_trend(did, df, ts):
    date_col = (ts or {}).get("detected_date_column")
    num_col = (ts or {}).get("numeric_column")
    if not date_col or not num_col or date_col not in df.columns or num_col not in df.columns:
        return None
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    pair = pd.DataFrame({"d": parsed, "v": pd.to_numeric(df[num_col], errors="coerce")}).dropna()
    if len(pair) < 30:
        return None
    pair = pair.sort_values("d")
    # Smooth via rolling window relative to series length
    win = max(7, len(pair) // 30)
    pair["smooth"] = pair["v"].rolling(win, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(pair["d"], pair["v"], color=ACCENT, alpha=0.35, lw=0.9, label="raw")
    ax.plot(pair["d"], pair["smooth"], color=ACCENT, lw=2.0,
            label=f"rolling mean (window {win})")
    # Overlay STL trend if available
    stl = (ts or {}).get("stl_decomposition") or {}
    trend = stl.get("trend_preview") or []
    if isinstance(trend, list) and len(trend) >= 5:
        n = len(trend)
        # align tail of dates
        d_tail = pair["d"].tail(n).values
        ax.plot(d_tail, trend, color=WARN, lw=1.6, alpha=0.95,
                label="STL trend")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_2)
    granularity = (ts.get("frequency") or {}).get("label") or "auto"
    _title(ax, "Time-series Trend",
           f"{num_col} over {date_col} · granularity {granularity}")
    _grid_y_only(ax)
    ax.set_xlabel(date_col, color=INK_2, fontweight="700")
    ax.set_ylabel(num_col, color=INK_2, fontweight="700")
    fig.autofmt_xdate()
    return {"title": "Time-series Trend", "type": "time_series_trend",
            "description": f"Trend of '{num_col}' over '{date_col}'.",
            "path": _save(did, "time_series_trend")}


# ---------------------------------------------------------------------------
# Bivariate / deep stats highlights
# ---------------------------------------------------------------------------

def _chart_bivariate_highlights(did, deep):
    biv = (deep or {}).get("bivariate") or {}
    nums = biv.get("numeric_pairs") or []
    cats = biv.get("categorical_pairs") or []
    diffs = biv.get("group_difference_tests") or []
    if not (nums or cats or diffs):
        return None

    items: list[tuple[str, float, str]] = []
    for p in nums[:5]:
        r = ((p.get("pearson") or {}).get("r"))
        if r is None:
            continue
        items.append((f"{p['column_a']} ↔ {p['column_b']} (numeric)",
                      float(r),
                      "numeric"))
    for p in cats[:5]:
        v = p.get("cramers_v")
        if v is None:
            continue
        items.append((f"{p['column_a']} ↔ {p['column_b']} (categorical)",
                      float(v),
                      "categorical"))
    for p in diffs[:5]:
        eff = (p.get("effect_size") or {}).get("value")
        if eff is None:
            continue
        items.append((f"{p['numeric_column']} ~ {p['group_column']} (group diff)",
                      float(eff),
                      "group"))
    if not items:
        return None
    items.sort(key=lambda t: abs(t[1]), reverse=True)
    items = items[:12]

    color_map = {"numeric": ACCENT, "categorical": ACCENT_2, "group": WARN}
    labels = [t[0] for t in items]
    vals = [t[1] for t in items]
    colors = [color_map[t[2]] for t in items]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1],
                   edgecolor=BG_PAGE, linewidth=0.8, zorder=3)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2,
                f"  {v:+.2f}",
                ha="left" if v >= 0 else "right",
                va="center", fontsize=9, fontweight="700", color=INK_1)
    ax.axvline(0, color=LINE_2, lw=0.8)
    _title(ax, "Bivariate Highlights",
           "strongest numeric / categorical / group-difference relationships")
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.55, zorder=0)
    legend = [
        Patch(facecolor=ACCENT,   edgecolor="none", label="numeric (r)"),
        Patch(facecolor=ACCENT_2, edgecolor="none", label="categorical (Cramér's V)"),
        Patch(facecolor=WARN,     edgecolor="none", label="group diff (effect size)"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=9)
    return {"title": "Bivariate Highlights", "type": "bivariate_highlights",
            "description": "Strongest relationships from deep statistics.",
            "path": _save(did, "bivariate_highlights")}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_charts(
    dataset_id,
    df,
    health,
    advanced,
    cleaning,
    *,
    target_column=None,
    model_leaderboard=None,
    explainability=None,
    time_series=None,
    deep_statistics_v2=None,
):
    """Generate all role-aware charts using the chart planner.

    The keyword args carry upstream phases (target column, ML leaderboard,
    SHAP, time-series, deep stats) so the planner can light up the
    target-aware / modeling / time-series charts when those phases produced
    data. Renderers gracefully no-op when their input is missing.
    """
    column_roles = infer_column_roles(df)
    from modules.profiler import build_profile
    profile = build_profile(df)

    context = {
        "target_column": target_column,
        "model_leaderboard": model_leaderboard,
        "explainability": explainability,
        "time_series": time_series,
        "deep_statistics_v2": deep_statistics_v2,
    }

    chart_plan = build_chart_plan(
        df, profile, health, advanced, cleaning, column_roles, context=context,
    )

    charts: list[dict] = []
    for item in chart_plan:
        chart = None
        t = item.get("type")
        try:
            if t == "health_components":
                chart = _chart_health(dataset_id, health)
            elif t == "dtype_breakdown":
                chart = _chart_dtype_breakdown(dataset_id, df)
            elif t == "missing_values":
                chart = _chart_missing(dataset_id, df)
            elif t == "cleaning_risk":
                chart = _chart_cleaning_risk(dataset_id, df)
            elif t == "cardinality_strip":
                chart = _chart_cardinality_strip(dataset_id, df)
            elif t == "numeric_distribution":
                chart = _chart_distribution(dataset_id, df, item["column"])
            elif t == "boxplot":
                chart = _chart_boxplot(dataset_id, df, item["column"])
            elif t == "violin":
                chart = _chart_violin(dataset_id, df, item["column"])
            elif t == "numeric_overview":
                chart = _chart_numeric_overview(dataset_id, df, item.get("columns") or [])
            elif t == "bid_ask_spread":
                chart = _chart_bid_ask(dataset_id, df, item["bid_column"], item["ask_column"])
            elif t == "categorical_count":
                chart = _chart_categorical(dataset_id, df, item["column"])
            elif t == "pareto_categorical":
                chart = _chart_pareto_categorical(dataset_id, df, item["column"])
            elif t == "correlation_heatmap":
                chart = _chart_correlation(dataset_id, df, profile.get("numeric_columns", []))
            elif t == "top_correlations":
                chart = _chart_top_correlations(dataset_id, df, profile.get("numeric_columns", []))
            elif t == "column_utility":
                chart = _chart_utility(dataset_id, advanced)
            elif t == "cleaning_impact":
                chart = _chart_cleaning_impact(dataset_id, cleaning)
            elif t == "anomaly_scores":
                chart = _chart_anomaly(dataset_id, advanced)
            elif t == "target_distribution":
                chart = _chart_target_distribution(dataset_id, df, item["column"])
            elif t == "target_vs_feature":
                chart = _chart_target_vs_feature(dataset_id, df, item["feature"], item["target"])
            elif t == "leaderboard_bars":
                chart = _chart_leaderboard_bars(dataset_id, model_leaderboard)
            elif t == "feature_importance_bars":
                chart = _chart_feature_importance_bars(dataset_id, explainability)
            elif t == "time_series_trend":
                chart = _chart_time_series_trend(dataset_id, df, time_series)
            elif t == "bivariate_highlights":
                chart = _chart_bivariate_highlights(dataset_id, deep_statistics_v2)
        except Exception as exc:  # noqa: BLE001 — never let a single chart kill the run
            chart = None
            # We deliberately don't log here at WARN level because chart fallout
            # is expected (e.g. all-NaN columns, single-class targets, etc.).

        if chart:
            chart["planner_reason"] = item.get("reason", "")
            charts.append(chart)

    return charts
