"""
Drift / Compare Analysis (WS-7)
================================
Compare two dataframes (a "reference" and a "current") column by column and
quantify how much each column's distribution has shifted.

Tests:
    Numeric columns:
        - Population Stability Index (PSI), 10 quantile bins
        - Kolmogorov-Smirnov two-sample test (ks_2samp)
        - Mean shift in standard deviations (z-shift)
    Categorical columns:
        - PSI on category proportions
        - Chi-square test of independence

Severity buckets (PSI):
    < 0.10  -> stable
    < 0.25  -> minor
    < 0.50  -> moderate
    >= 0.50 -> severe

Public entry points:
    compare_datasets(df_a, df_b, columns=None) -> dict
    split_by_date(df, date_column, ratio=0.5) -> tuple[DataFrame, DataFrame]

Output is fully JSON-safe.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, ndigits: int = 4) -> float | int | None:
    if value is None:
        return None
    try:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return None
            return round(v, ndigits)
        if pd.isna(value):
            return None
        return value
    except Exception:
        return None


def _classify_psi(psi: float | None) -> str:
    if psi is None:
        return "unknown"
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "minor"
    if psi < 0.50:
        return "moderate"
    return "severe"


def _shared_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    return [c for c in df_a.columns if c in df_b.columns]


# ---------------------------------------------------------------------------
# Numeric drift
# ---------------------------------------------------------------------------

def _psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float | None:
    """Population Stability Index using quantile bins from `ref`."""
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return None

    # Build bin edges from ref's quantiles, then expand bounds to capture cur extremes
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return None  # ref is constant or near-constant
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_props = ref_counts / max(ref_counts.sum(), 1)
    cur_props = cur_counts / max(cur_counts.sum(), 1)

    # Laplace smoothing to avoid log(0)
    ref_props = np.where(ref_props == 0, 1e-6, ref_props)
    cur_props = np.where(cur_props == 0, 1e-6, cur_props)

    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return psi if math.isfinite(psi) else None


def _numeric_column_drift(name: str, ref: pd.Series, cur: pd.Series) -> dict:
    # Force float64 — pd.to_numeric leaves boolean dtype alone, and newer
    # numpy refuses to subtract bool arrays inside np.quantile.
    ref_arr = pd.to_numeric(ref, errors="coerce").astype("float64", copy=False).to_numpy()
    cur_arr = pd.to_numeric(cur, errors="coerce").astype("float64", copy=False).to_numpy()
    ref_arr = ref_arr[np.isfinite(ref_arr)]
    cur_arr = cur_arr[np.isfinite(cur_arr)]

    if len(ref_arr) < 10 or len(cur_arr) < 10:
        return {"column": name, "type": "numeric", "available": False, "reason": "n<10 in one side"}

    psi = _psi_numeric(ref_arr, cur_arr)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ks_stat, ks_p = stats.ks_2samp(ref_arr, cur_arr)
        ks_stat = _safe_float(ks_stat)
        ks_p = _safe_float(ks_p)
    except Exception:
        ks_stat, ks_p = None, None

    ref_std = float(np.std(ref_arr, ddof=0))
    cur_std = float(np.std(cur_arr, ddof=0))
    ref_mean = float(np.mean(ref_arr))
    cur_mean = float(np.mean(cur_arr))
    pooled_std = ref_std if ref_std > 0 else cur_std if cur_std > 0 else 1.0
    z_shift = float((cur_mean - ref_mean) / pooled_std) if pooled_std > 0 else None

    severity = _classify_psi(psi)
    if ks_p is not None and ks_p < 0.01 and severity == "stable":
        severity = "minor"  # KS reveals a shift PSI missed

    # Compact histogram preview the frontend can plot directly
    edges = np.linspace(
        min(ref_arr.min(), cur_arr.min()),
        max(ref_arr.max(), cur_arr.max()),
        21,
    )
    ref_hist, _ = np.histogram(ref_arr, bins=edges)
    cur_hist, _ = np.histogram(cur_arr, bins=edges)
    histogram = {
        "edges": [_safe_float(v) for v in edges.tolist()],
        "ref": [int(v) for v in ref_hist.tolist()],
        "cur": [int(v) for v in cur_hist.tolist()],
    }

    return {
        "column": name,
        "type": "numeric",
        "available": True,
        "n_ref": int(len(ref_arr)),
        "n_cur": int(len(cur_arr)),
        "psi": _safe_float(psi),
        "psi_severity": severity,
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "ks_significant": bool(ks_p is not None and ks_p < 0.01),
        "ref_mean": _safe_float(ref_mean),
        "cur_mean": _safe_float(cur_mean),
        "ref_std": _safe_float(ref_std),
        "cur_std": _safe_float(cur_std),
        "z_shift": _safe_float(z_shift),
        "histogram": histogram,
    }


# ---------------------------------------------------------------------------
# Categorical drift
# ---------------------------------------------------------------------------

def _psi_categorical(ref: pd.Series, cur: pd.Series) -> tuple[float | None, list[str], dict]:
    ref_counts = ref.astype(str).value_counts(dropna=False)
    cur_counts = cur.astype(str).value_counts(dropna=False)

    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    if len(categories) < 2:
        return None, categories, {}

    ref_total = max(int(ref_counts.sum()), 1)
    cur_total = max(int(cur_counts.sum()), 1)

    ref_props = np.array([ref_counts.get(c, 0) / ref_total for c in categories], dtype=float)
    cur_props = np.array([cur_counts.get(c, 0) / cur_total for c in categories], dtype=float)

    ref_smoothed = np.where(ref_props == 0, 1e-6, ref_props)
    cur_smoothed = np.where(cur_props == 0, 1e-6, cur_props)
    psi = float(np.sum((cur_smoothed - ref_smoothed) * np.log(cur_smoothed / ref_smoothed)))

    distribution = {
        "categories": categories[:30],
        "ref_props": [round(float(p), 4) for p in ref_props[:30]],
        "cur_props": [round(float(p), 4) for p in cur_props[:30]],
    }

    return (psi if math.isfinite(psi) else None), categories, distribution


def _categorical_column_drift(name: str, ref: pd.Series, cur: pd.Series) -> dict:
    ref_clean = ref.dropna()
    cur_clean = cur.dropna()

    if len(ref_clean) < 10 or len(cur_clean) < 10:
        return {"column": name, "type": "categorical", "available": False, "reason": "n<10 in one side"}

    psi, categories, distribution = _psi_categorical(ref_clean, cur_clean)
    if psi is None:
        return {"column": name, "type": "categorical", "available": False, "reason": "single category"}

    chi2_stat, chi2_p, dof = None, None, None
    try:
        # Cap categories to keep contingency table manageable
        top_categories = categories[:30]
        ref_counts = ref_clean.astype(str).value_counts()
        cur_counts = cur_clean.astype(str).value_counts()
        table = np.array(
            [
                [int(ref_counts.get(c, 0)) for c in top_categories],
                [int(cur_counts.get(c, 0)) for c in top_categories],
            ]
        )
        if table.shape[1] >= 2 and (table.sum(axis=0) > 0).all():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chi2_stat, chi2_p, dof, _ = stats.chi2_contingency(table)
    except Exception:
        pass

    return {
        "column": name,
        "type": "categorical",
        "available": True,
        "n_ref": int(len(ref_clean)),
        "n_cur": int(len(cur_clean)),
        "psi": _safe_float(psi),
        "psi_severity": _classify_psi(psi),
        "chi2_statistic": _safe_float(chi2_stat),
        "chi2_p_value": _safe_float(chi2_p),
        "chi2_dof": int(dof) if dof is not None else None,
        "chi2_significant": bool(chi2_p is not None and chi2_p < 0.01),
        "n_categories_ref": int(ref_clean.nunique(dropna=True)),
        "n_categories_cur": int(cur_clean.nunique(dropna=True)),
        "new_categories": [c for c in cur_clean.unique() if c not in set(ref_clean.unique())][:10],
        "missing_categories": [c for c in ref_clean.unique() if c not in set(cur_clean.unique())][:10],
        "distribution": distribution,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def compare_datasets(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    columns: list[str] | None = None,
    max_columns: int = 30,
) -> dict:
    """
    Compare two dataframes column by column. Returns drift report.

    Args:
        df_ref: Reference dataframe (e.g. older / training data).
        df_cur: Current dataframe (e.g. newer / production data).
        columns: Restrict comparison to these columns.
        max_columns: Cap on columns analyzed to keep runtime bounded.

    Returns a JSON-safe dict.
    """
    shared = _shared_columns(df_ref, df_cur)
    if columns:
        shared = [c for c in shared if c in columns]
    shared = shared[:max_columns]

    if not shared:
        return {
            "available": False,
            "reason": "No shared columns between the two datasets.",
            "ref_shape": list(df_ref.shape),
            "cur_shape": list(df_cur.shape),
        }

    column_results: list[dict] = []
    for col in shared:
        ref_series = df_ref[col]
        cur_series = df_cur[col]

        if pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(cur_series):
            column_results.append(_numeric_column_drift(col, ref_series, cur_series))
        elif (
            pd.api.types.is_object_dtype(ref_series)
            or pd.api.types.is_string_dtype(ref_series)
            or isinstance(ref_series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(ref_series)
        ):
            column_results.append(_categorical_column_drift(col, ref_series, cur_series))
        else:
            column_results.append({
                "column": col,
                "type": "unsupported",
                "available": False,
                "reason": f"Unsupported dtype: {ref_series.dtype}",
            })

    # Aggregate severity
    severity_counts = {"stable": 0, "minor": 0, "moderate": 0, "severe": 0, "unknown": 0}
    for entry in column_results:
        sev = entry.get("psi_severity") if entry.get("available") else "unknown"
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Sort with most-drifted first (severe -> stable)
    severity_rank = {"severe": 0, "moderate": 1, "minor": 2, "stable": 3, "unknown": 4}

    def _rank(entry):
        sev = entry.get("psi_severity") if entry.get("available") else "unknown"
        psi = entry.get("psi")
        return (severity_rank.get(sev, 9), -(psi if isinstance(psi, (int, float)) else 0))

    column_results.sort(key=_rank)

    if severity_counts["severe"] > 0:
        verdict = "severe"
    elif severity_counts["moderate"] > 0:
        verdict = "moderate"
    elif severity_counts["minor"] > 0:
        verdict = "minor"
    else:
        verdict = "stable"

    return {
        "available": True,
        "ref_shape": list(df_ref.shape),
        "cur_shape": list(df_cur.shape),
        "shared_columns": shared,
        "summary": {
            "n_columns_compared": len(column_results),
            "severity_counts": severity_counts,
            "overall_verdict": verdict,
        },
        "interpretation": (
            "PSI < 0.10 stable · 0.10-0.25 minor · 0.25-0.50 moderate · ≥ 0.50 severe. "
            "Numeric columns also report KS test; categorical columns report chi-square."
        ),
        "columns": column_results,
    }


def split_by_date(
    df: pd.DataFrame,
    date_column: str,
    ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe chronologically into earlier / later halves.

    Returns (older_df, newer_df) sorted by date_column. Useful when the user
    has a single dataset with a time index and wants to compare drift over time.
    """
    if date_column not in df.columns:
        raise ValueError(f"Column not found: {date_column}")

    parsed = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
    if parsed.notna().mean() < 0.7:
        raise ValueError(f"{date_column} is not a parseable date column.")

    sorted_df = df.assign(__dt__=parsed).sort_values("__dt__").drop(columns="__dt__")
    cut = int(len(sorted_df) * ratio)
    if cut < 10 or len(sorted_df) - cut < 10:
        raise ValueError("Not enough rows on each side of the split.")

    return sorted_df.iloc[:cut].copy(), sorted_df.iloc[cut:].copy()
