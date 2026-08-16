"""Dataset drift and change analysis.

FrameVitals compares a reference and current dataframe across schema, types,
missingness, and value distributions. PSI remains available for compatibility,
but the public verdict now combines multiple interpretable signals rather than
letting one statistic decide the entire result.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


_SEVERITY_ORDER = {
    "unknown": -1,
    "stable": 0,
    "minor": 1,
    "moderate": 2,
    "severe": 3,
}


def _safe_float(value: Any, ndigits: int = 4) -> float | int | None:
    if value is None:
        return None
    try:
        if isinstance(value, np.integer):
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


def _classify_missingness(delta_percentage_points: float) -> str:
    delta = abs(delta_percentage_points)
    if delta < 2:
        return "stable"
    if delta < 5:
        return "minor"
    if delta < 15:
        return "moderate"
    return "severe"


def _classify_numeric_distance(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "minor"
    if value < 0.50:
        return "moderate"
    return "severe"


def _classify_js_distance(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.05:
        return "stable"
    if value < 0.10:
        return "minor"
    if value < 0.25:
        return "moderate"
    return "severe"


def _max_severity(*values: str) -> str:
    usable = [value for value in values if value in _SEVERITY_ORDER]
    if not usable:
        return "unknown"
    return max(usable, key=lambda value: _SEVERITY_ORDER[value])


def severity_at_least(actual: str, threshold: str) -> bool:
    """Return whether a drift severity meets or exceeds ``threshold``."""
    if threshold not in {"minor", "moderate", "severe"}:
        raise ValueError("threshold must be one of: minor, moderate, severe.")
    return _SEVERITY_ORDER.get(actual, -1) >= _SEVERITY_ORDER[threshold]


def _dtype_family(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "categorical"
    return "unsupported"


def _shared_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    other_columns = set(df_b.columns)
    return [column for column in df_a.columns if column in other_columns]


def _missingness_metrics(ref: pd.Series, cur: pd.Series) -> dict[str, Any]:
    ref_pct = float(ref.isna().mean() * 100) if len(ref) else 0.0
    cur_pct = float(cur.isna().mean() * 100) if len(cur) else 0.0
    delta = cur_pct - ref_pct
    return {
        "ref_missing_percent": _safe_float(ref_pct),
        "cur_missing_percent": _safe_float(cur_pct),
        "missingness_delta_percentage_points": _safe_float(delta),
        "missingness_severity": _classify_missingness(delta),
    }


def _psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float | None:
    """Population Stability Index using quantile bins from ``ref``."""
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_props = ref_counts / max(ref_counts.sum(), 1)
    cur_props = cur_counts / max(cur_counts.sum(), 1)
    ref_props = np.where(ref_props == 0, 1e-6, ref_props)
    cur_props = np.where(cur_props == 0, 1e-6, cur_props)

    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return psi if math.isfinite(psi) else None


def _normalized_wasserstein(ref: np.ndarray, cur: np.ndarray) -> float | None:
    if len(ref) < 2 or len(cur) < 2:
        return None
    distance = float(stats.wasserstein_distance(ref, cur))
    q1, q3 = np.quantile(ref, [0.25, 0.75])
    scale = float(q3 - q1)
    if scale <= 0:
        scale = float(np.std(ref, ddof=0))
    if scale <= 0:
        scale = max(abs(float(np.mean(ref))), 1.0)
    value = distance / scale
    return value if math.isfinite(value) else None


def _numeric_column_drift(name: str, ref: pd.Series, cur: pd.Series) -> dict[str, Any]:
    missingness = _missingness_metrics(ref, cur)
    ref_arr = pd.to_numeric(ref, errors="coerce").astype("float64").to_numpy()
    cur_arr = pd.to_numeric(cur, errors="coerce").astype("float64").to_numpy()
    ref_arr = ref_arr[np.isfinite(ref_arr)]
    cur_arr = cur_arr[np.isfinite(cur_arr)]

    if len(ref_arr) < 10 or len(cur_arr) < 10:
        severity = missingness["missingness_severity"]
        return {
            "column": name,
            "type": "numeric",
            "available": False,
            "reason": "n<10 in one side",
            "drift_severity": severity,
            **missingness,
        }

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

    wasserstein = _normalized_wasserstein(ref_arr, cur_arr)
    psi_severity = _classify_psi(psi)
    wasserstein_severity = _classify_numeric_distance(wasserstein)
    ks_severity = "minor" if ks_p is not None and ks_p < 0.01 else "stable"
    severity = _max_severity(
        psi_severity,
        wasserstein_severity,
        ks_severity,
        missingness["missingness_severity"],
    )

    combined_min = float(min(ref_arr.min(), cur_arr.min()))
    combined_max = float(max(ref_arr.max(), cur_arr.max()))
    if combined_min == combined_max:
        combined_min -= 0.5
        combined_max += 0.5
    edges = np.linspace(combined_min, combined_max, 21)
    ref_hist, _ = np.histogram(ref_arr, bins=edges)
    cur_hist, _ = np.histogram(cur_arr, bins=edges)

    return {
        "column": name,
        "type": "numeric",
        "available": True,
        "n_ref": int(len(ref_arr)),
        "n_cur": int(len(cur_arr)),
        "psi": _safe_float(psi),
        "psi_severity": psi_severity,
        "wasserstein_normalized": _safe_float(wasserstein),
        "wasserstein_severity": wasserstein_severity,
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "ks_significant": bool(ks_p is not None and ks_p < 0.01),
        "ref_mean": _safe_float(ref_mean),
        "cur_mean": _safe_float(cur_mean),
        "ref_std": _safe_float(ref_std),
        "cur_std": _safe_float(cur_std),
        "z_shift": _safe_float(z_shift),
        "drift_severity": severity,
        **missingness,
        "histogram": {
            "edges": [_safe_float(value) for value in edges.tolist()],
            "ref": [int(value) for value in ref_hist.tolist()],
            "cur": [int(value) for value in cur_hist.tolist()],
        },
    }


def _psi_categorical(
    ref: pd.Series,
    cur: pd.Series,
) -> tuple[float | None, list[str], dict[str, Any], pd.Series, pd.Series, np.ndarray, np.ndarray]:
    ref_counts = ref.astype(str).value_counts(dropna=False)
    cur_counts = cur.astype(str).value_counts(dropna=False)

    combined = ref_counts.add(cur_counts, fill_value=0).sort_values(ascending=False)
    categories = [str(value) for value in combined.index.tolist()]
    if len(categories) < 2:
        return None, categories, {}, ref_counts, cur_counts, np.array([]), np.array([])

    ref_total = max(int(ref_counts.sum()), 1)
    cur_total = max(int(cur_counts.sum()), 1)
    ref_props = np.array([ref_counts.get(c, 0) / ref_total for c in categories], dtype=float)
    cur_props = np.array([cur_counts.get(c, 0) / cur_total for c in categories], dtype=float)

    ref_smoothed = np.where(ref_props == 0, 1e-6, ref_props)
    cur_smoothed = np.where(cur_props == 0, 1e-6, cur_props)
    psi = float(
        np.sum((cur_smoothed - ref_smoothed) * np.log(cur_smoothed / ref_smoothed))
    )

    distribution = {
        "categories": categories[:30],
        "ref_props": [round(float(p), 4) for p in ref_props[:30]],
        "cur_props": [round(float(p), 4) for p in cur_props[:30]],
    }
    return (
        psi if math.isfinite(psi) else None,
        categories,
        distribution,
        ref_counts,
        cur_counts,
        ref_props,
        cur_props,
    )


def _categorical_column_drift(name: str, ref: pd.Series, cur: pd.Series) -> dict[str, Any]:
    missingness = _missingness_metrics(ref, cur)
    ref_clean = ref.dropna()
    cur_clean = cur.dropna()

    if len(ref_clean) < 10 or len(cur_clean) < 10:
        severity = missingness["missingness_severity"]
        return {
            "column": name,
            "type": "categorical",
            "available": False,
            "reason": "n<10 in one side",
            "drift_severity": severity,
            **missingness,
        }

    (
        psi,
        categories,
        distribution,
        ref_counts,
        cur_counts,
        ref_props,
        cur_props,
    ) = _psi_categorical(ref_clean, cur_clean)
    if psi is None:
        return {
            "column": name,
            "type": "categorical",
            "available": False,
            "reason": "single category",
            "drift_severity": missingness["missingness_severity"],
            **missingness,
        }

    js_distance = None
    if len(ref_props) and len(cur_props):
        try:
            js_distance = float(jensenshannon(ref_props, cur_props, base=2))
        except Exception:
            js_distance = None

    chi2_stat, chi2_p, dof = None, None, None
    try:
        top_categories = categories[:30]
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

    ref_unique = ref_clean.unique()
    cur_unique = cur_clean.unique()
    ref_unique_set = set(ref_unique)
    cur_unique_set = set(cur_unique)

    psi_severity = _classify_psi(psi)
    js_severity = _classify_js_distance(js_distance)
    chi_severity = "minor" if chi2_p is not None and chi2_p < 0.01 else "stable"
    severity = _max_severity(
        psi_severity,
        js_severity,
        chi_severity,
        missingness["missingness_severity"],
    )

    return {
        "column": name,
        "type": "categorical",
        "available": True,
        "n_ref": int(len(ref_clean)),
        "n_cur": int(len(cur_clean)),
        "psi": _safe_float(psi),
        "psi_severity": psi_severity,
        "jensen_shannon_distance": _safe_float(js_distance),
        "jensen_shannon_severity": js_severity,
        "chi2_statistic": _safe_float(chi2_stat),
        "chi2_p_value": _safe_float(chi2_p),
        "chi2_dof": int(dof) if dof is not None else None,
        "chi2_significant": bool(chi2_p is not None and chi2_p < 0.01),
        "n_categories_ref": int(len(ref_unique)),
        "n_categories_cur": int(len(cur_unique)),
        "new_categories": [value for value in cur_unique if value not in ref_unique_set][:10],
        "missing_categories": [value for value in ref_unique if value not in cur_unique_set][:10],
        "drift_severity": severity,
        **missingness,
        "distribution": distribution,
    }


def _datetime_column_drift(name: str, ref: pd.Series, cur: pd.Series) -> dict[str, Any]:
    missingness = _missingness_metrics(ref, cur)
    ref_dt = pd.to_datetime(ref, errors="coerce", utc=True).dropna()
    cur_dt = pd.to_datetime(cur, errors="coerce", utc=True).dropna()
    if len(ref_dt) < 10 or len(cur_dt) < 10:
        return {
            "column": name,
            "type": "datetime",
            "available": False,
            "reason": "n<10 in one side",
            "drift_severity": missingness["missingness_severity"],
            **missingness,
        }

    ref_days = ref_dt.astype("int64").to_numpy(dtype="float64") / 86_400_000_000_000
    cur_days = cur_dt.astype("int64").to_numpy(dtype="float64") / 86_400_000_000_000
    psi = _psi_numeric(ref_days, cur_days)
    wasserstein = _normalized_wasserstein(ref_days, cur_days)
    try:
        ks_stat, ks_p = stats.ks_2samp(ref_days, cur_days)
    except Exception:
        ks_stat, ks_p = None, None

    psi_severity = _classify_psi(psi)
    wasserstein_severity = _classify_numeric_distance(wasserstein)
    ks_severity = "minor" if ks_p is not None and ks_p < 0.01 else "stable"
    severity = _max_severity(
        psi_severity,
        wasserstein_severity,
        ks_severity,
        missingness["missingness_severity"],
    )

    return {
        "column": name,
        "type": "datetime",
        "available": True,
        "n_ref": int(len(ref_dt)),
        "n_cur": int(len(cur_dt)),
        "psi": _safe_float(psi),
        "psi_severity": psi_severity,
        "wasserstein_normalized": _safe_float(wasserstein),
        "wasserstein_severity": wasserstein_severity,
        "ks_statistic": _safe_float(ks_stat),
        "ks_p_value": _safe_float(ks_p),
        "ks_significant": bool(ks_p is not None and ks_p < 0.01),
        "ref_min": ref_dt.min().isoformat(),
        "ref_max": ref_dt.max().isoformat(),
        "cur_min": cur_dt.min().isoformat(),
        "cur_max": cur_dt.max().isoformat(),
        "drift_severity": severity,
        **missingness,
    }


def _column_drift(name: str, ref: pd.Series, cur: pd.Series) -> dict[str, Any]:
    ref_family = _dtype_family(ref)
    cur_family = _dtype_family(cur)
    if ref_family != cur_family:
        return {
            "column": name,
            "type": "type_mismatch",
            "available": False,
            "reason": f"Type changed from {ref_family} to {cur_family}.",
            "ref_type": ref_family,
            "cur_type": cur_family,
            "drift_severity": "severe",
            **_missingness_metrics(ref, cur),
        }
    if ref_family == "numeric":
        return _numeric_column_drift(name, ref, cur)
    if ref_family in {"categorical", "boolean"}:
        return _categorical_column_drift(name, ref, cur)
    if ref_family == "datetime":
        return _datetime_column_drift(name, ref, cur)
    return {
        "column": name,
        "type": "unsupported",
        "available": False,
        "reason": f"Unsupported dtype: {ref.dtype}",
        "drift_severity": "unknown",
        **_missingness_metrics(ref, cur),
    }


def compare_datasets(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    columns: list[str] | None = None,
    max_columns: int = 30,
) -> dict[str, Any]:
    """Compare two dataframes and return a structured drift/change report."""
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    ref_columns = set(df_ref.columns)
    cur_columns = set(df_cur.columns)
    all_shared = _shared_columns(df_ref, df_cur)
    requested_missing_ref: list[str] = []
    requested_missing_cur: list[str] = []

    shared = all_shared
    if columns:
        requested = list(dict.fromkeys(columns))
        requested_set = set(requested)
        requested_missing_ref = [name for name in requested if name not in ref_columns]
        requested_missing_cur = [name for name in requested if name not in cur_columns]
        shared = [name for name in all_shared if name in requested_set]

    total_selected = len(shared)
    truncated = total_selected > max_columns
    shared = shared[:max_columns]

    added_columns = sorted(cur_columns - ref_columns)
    removed_columns = sorted(ref_columns - cur_columns)
    dtype_changes = [
        {
            "column": name,
            "reference_type": _dtype_family(df_ref[name]),
            "current_type": _dtype_family(df_cur[name]),
        }
        for name in all_shared
        if _dtype_family(df_ref[name]) != _dtype_family(df_cur[name])
    ]

    row_delta_pct = None
    if len(df_ref):
        row_delta_pct = (len(df_cur) - len(df_ref)) / len(df_ref) * 100

    if not shared:
        schema_severity = "severe" if removed_columns or dtype_changes else "moderate" if added_columns else "stable"
        return {
            "available": False,
            "reason": "No shared selected columns between the two datasets.",
            "ref_shape": list(df_ref.shape),
            "cur_shape": list(df_cur.shape),
            "schema": {
                "added_columns": added_columns,
                "removed_columns": removed_columns,
                "dtype_changes": dtype_changes,
                "requested_missing_in_reference": requested_missing_ref,
                "requested_missing_in_current": requested_missing_cur,
                "severity": schema_severity,
            },
        }

    column_results = [_column_drift(name, df_ref[name], df_cur[name]) for name in shared]

    severity_counts = Counter(
        entry.get("drift_severity", "unknown") for entry in column_results
    )
    for label in _SEVERITY_ORDER:
        severity_counts.setdefault(label, 0)

    schema_severity = "stable"
    if removed_columns or dtype_changes:
        schema_severity = "severe"
    elif added_columns:
        schema_severity = "moderate"

    column_severity = _max_severity(
        *(entry.get("drift_severity", "unknown") for entry in column_results)
    )
    overall_verdict = _max_severity(column_severity, schema_severity)

    severity_rank = {
        "severe": 0,
        "moderate": 1,
        "minor": 2,
        "stable": 3,
        "unknown": 4,
    }

    def _rank(entry: dict[str, Any]) -> tuple[int, float]:
        severity = entry.get("drift_severity", "unknown")
        psi = entry.get("psi")
        return (
            severity_rank.get(severity, 9),
            -(psi if isinstance(psi, (int, float)) else 0),
        )

    column_results.sort(key=_rank)

    gate_reasons: list[str] = []
    if removed_columns:
        gate_reasons.append(f"{len(removed_columns)} reference columns are missing from current data.")
    if dtype_changes:
        gate_reasons.append(f"{len(dtype_changes)} shared columns changed type family.")
    if added_columns:
        gate_reasons.append(f"{len(added_columns)} new columns appeared in current data.")
    severe_columns = [
        entry["column"] for entry in column_results if entry.get("drift_severity") == "severe"
    ]
    moderate_columns = [
        entry["column"] for entry in column_results if entry.get("drift_severity") == "moderate"
    ]
    if severe_columns:
        gate_reasons.append(f"Severe drift detected in: {', '.join(severe_columns[:10])}.")
    elif moderate_columns:
        gate_reasons.append(f"Moderate drift detected in: {', '.join(moderate_columns[:10])}.")

    if overall_verdict == "severe":
        gate_status = "fail"
    elif overall_verdict in {"minor", "moderate"}:
        gate_status = "warn"
    else:
        gate_status = "pass"

    return {
        "available": True,
        "ref_shape": list(df_ref.shape),
        "cur_shape": list(df_cur.shape),
        "row_count_change_percent": _safe_float(row_delta_pct),
        "shared_columns": shared,
        "selection": {
            "total_shared_columns": len(all_shared),
            "total_selected_columns": total_selected,
            "max_columns": int(max_columns),
            "truncated": truncated,
            "requested_missing_in_reference": requested_missing_ref,
            "requested_missing_in_current": requested_missing_cur,
        },
        "schema": {
            "added_columns": added_columns,
            "removed_columns": removed_columns,
            "dtype_changes": dtype_changes,
            "severity": schema_severity,
        },
        "summary": {
            "n_columns_compared": len(column_results),
            "n_columns_available": sum(bool(entry.get("available")) for entry in column_results),
            "severity_counts": dict(severity_counts),
            "overall_verdict": overall_verdict,
        },
        "gate": {
            "status": gate_status,
            "severity": overall_verdict,
            "reasons": gate_reasons,
        },
        "interpretation": (
            "Drift severity combines PSI with normalized Wasserstein distance for numeric/date columns, "
            "Jensen-Shannon distance for categorical columns, statistical tests, missingness changes, "
            "and structural schema/type changes."
        ),
        "columns": column_results,
    }


def split_by_date(
    df: pd.DataFrame,
    date_column: str,
    ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe chronologically into earlier and later partitions."""
    if date_column not in df.columns:
        raise ValueError(f"Column not found: {date_column}")
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1.")

    parsed = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
    if parsed.notna().mean() < 0.7:
        raise ValueError(f"{date_column} is not a parseable date column.")

    sorted_df = df.assign(__dt__=parsed).sort_values("__dt__").drop(columns="__dt__")
    cut = int(len(sorted_df) * ratio)
    if cut < 10 or len(sorted_df) - cut < 10:
        raise ValueError("Not enough rows on each side of the split.")

    return sorted_df.iloc[:cut].copy(), sorted_df.iloc[cut:].copy()
