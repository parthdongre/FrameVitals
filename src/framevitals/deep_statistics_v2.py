"""
Deep Statistics v2 (WS-1)
=========================
Research-grade statistical battery for tabular data.

Adds on top of the original deep_statistics module:
- Shapiro-Wilk (small-n) and Anderson-Darling normality tests
- Best-fit distribution selection by AIC across 6 candidates
- Bootstrap 95% CIs for mean/median (BCa via scipy.stats.bootstrap)
- Robust z-score (MAD) outlier flags alongside IQR + classical z
- Mann-Whitney U / Kruskal-Wallis / Welch's t-test bivariate tests
- Cramér's V for categorical-categorical association strength
- Point-biserial correlation for binary-vs-numeric pairs
- Pearson/Spearman/Kendall with p-values for numeric pairs

All results are JSON-safe (no numpy scalars or NaN/Inf leaking out).

Public entry point:
    run_deep_statistics_v2(df, max_pairs=20) -> dict
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# JSON-safe helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | int | None:
    """Coerce numpy/pandas scalars to plain Python; drop NaN/Inf."""
    if value is None:
        return None
    try:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return None
            return round(v, 6)
        if pd.isna(value):
            return None
        return value
    except Exception:
        return None


def _safe_dict(d: dict) -> dict:
    return {k: _safe_float(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Per-column descriptive + outlier engine
# ---------------------------------------------------------------------------

def _classify_skew(skew: float | None) -> str:
    if skew is None:
        return "Unknown"
    a = abs(skew)
    if a < 0.5:
        return "Approximately Symmetric"
    if a < 1.0:
        return "Moderately Skewed"
    return "Highly Skewed"


def _classify_kurtosis(k: float | None) -> str:
    if k is None:
        return "Unknown"
    if k > 1:
        return "Heavy-tailed"
    if k < -1:
        return "Light-tailed"
    return "Approximately Normal-tailed"


def _outlier_flags(series: pd.Series) -> dict:
    """Three outlier views: IQR, classical z (|z|>3), robust MAD-z (|z|>3.5)."""
    s = series.dropna()
    n = len(s)
    if n < 3:
        return {"iqr": 0, "z3": 0, "mad_z": 0, "n": n}

    # IQR
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    iqr_count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr and iqr > 0 else 0

    # Classical z
    std = s.std()
    if std and std > 0:
        z = (s - s.mean()) / std
        z3_count = int((z.abs() > 3).sum())
    else:
        z3_count = 0

    # Robust z via MAD
    median = s.median()
    mad = (s - median).abs().median()
    if mad and mad > 0:
        rz = (s - median) / (1.4826 * mad)
        mad_count = int((rz.abs() > 3.5).sum())
    else:
        mad_count = 0

    return {"iqr": iqr_count, "z3": z3_count, "mad_z": mad_count, "n": n}


def _legacy_anderson_critical_5pct(result: Any) -> float | int | None:
    """Extract the legacy 5% critical value without assuming a fixed table index."""
    critical_values = getattr(result, "critical_values", None)
    significance_levels = getattr(result, "significance_level", None)
    if critical_values is None:
        return None

    if significance_levels is not None:
        try:
            levels = np.asarray(significance_levels, dtype=float)
            values = np.asarray(critical_values, dtype=float)
            if levels.size and values.size == levels.size:
                index = int(np.argmin(np.abs(levels - 5.0)))
                return _safe_float(values[index])
        except (TypeError, ValueError):
            pass

    try:
        return _safe_float(critical_values[2])
    except (IndexError, TypeError):
        return None


def _anderson_normality(series: pd.Series) -> dict[str, Any]:
    """Run Anderson-Darling using the modern p-value API when available.

    SciPy 1.17 introduced ``method`` and deprecated the legacy critical-value
    result attributes. FrameVitals supports older SciPy releases too, so this
    helper opts into the modern interpolated p-value and falls back only when
    the installed SciPy does not recognize the keyword.
    """
    try:
        result = stats.anderson(series, dist="norm", method="interpolate")
    except TypeError as exc:
        if "method" not in str(exc):
            raise
        result = stats.anderson(series, dist="norm")
        return {
            "statistic": _safe_float(result.statistic),
            "p_value": None,
            "critical_5pct": _legacy_anderson_critical_5pct(result),
            "method": "legacy_critical_values",
        }

    return {
        "statistic": _safe_float(result.statistic),
        "p_value": _safe_float(result.pvalue),
        "critical_5pct": None,
        "method": "interpolate",
    }


def _normality(series: pd.Series) -> dict:
    """Run multiple normality tests, prefer Shapiro for small n, D'Agostino for medium."""
    s = series.dropna()
    if len(s) < 8:
        return {"test": "skipped", "reason": "n<8", "p_value": None, "is_probably_normal": None}

    out: dict[str, Any] = {}

    # Shapiro-Wilk (most powerful for n<=5000)
    if len(s) <= 5000:
        try:
            stat, p = stats.shapiro(s)
            out["shapiro"] = {"statistic": _safe_float(stat), "p_value": _safe_float(p)}
        except Exception as exc:
            out["shapiro"] = {"error": str(exc)}

    # D'Agostino-Pearson (good for n>20)
    if len(s) >= 20:
        try:
            stat, p = stats.normaltest(s if len(s) <= 5000 else s.sample(5000, random_state=42))
            out["dagostino"] = {"statistic": _safe_float(stat), "p_value": _safe_float(p)}
        except Exception as exc:
            out["dagostino"] = {"error": str(exc)}

    # Anderson-Darling. Explicit p-value method avoids SciPy >=1.17 deprecation warnings.
    try:
        out["anderson"] = _anderson_normality(s)
    except Exception as exc:
        out["anderson"] = {"error": str(exc)}

    # Verdict (prefer shapiro p-value when available)
    p_use = None
    if "shapiro" in out and isinstance(out["shapiro"], dict):
        p_use = out["shapiro"].get("p_value")
    if p_use is None and "dagostino" in out and isinstance(out["dagostino"], dict):
        p_use = out["dagostino"].get("p_value")

    if p_use is None:
        verdict = None
        interpretation = "Could not run normality test."
    else:
        verdict = bool(p_use >= 0.05)
        interpretation = (
            "Distribution does not strongly reject normality."
            if verdict
            else "Distribution likely differs from normal."
        )

    out["is_probably_normal"] = verdict
    out["interpretation"] = interpretation
    return out


# ---------------------------------------------------------------------------
# Best-fit distribution by AIC
# ---------------------------------------------------------------------------

_DISTRIBUTION_CANDIDATES = [
    ("norm", stats.norm),
    ("lognorm", stats.lognorm),
    ("expon", stats.expon),
    ("gamma", stats.gamma),
    ("weibull_min", stats.weibull_min),
    ("beta", stats.beta),
]


def _fit_best_distribution(series: pd.Series) -> dict:
    """Try each candidate, pick the one with lowest AIC."""
    s = series.dropna()
    n = len(s)
    if n < 30:
        return {"available": False, "reason": "n<30 (need at least 30 to fit a distribution)"}

    # Sample if very large
    if n > 10000:
        s = s.sample(10000, random_state=42)
        n = len(s)

    best = None
    leaderboard = []

    for name, dist in _DISTRIBUTION_CANDIDATES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Some dists need positive support
                if name in {"lognorm", "expon", "gamma", "weibull_min"} and (s <= 0).any():
                    continue
                if name == "beta" and ((s.min() < 0) or (s.max() > 1)):
                    # Rescale to (0,1) for beta fitting
                    rng = s.max() - s.min()
                    if rng <= 0:
                        continue
                    s_scaled = (s - s.min()) / rng
                    s_scaled = s_scaled.clip(1e-6, 1 - 1e-6)
                    params = dist.fit(s_scaled)
                    log_lik = float(np.sum(dist.logpdf(s_scaled, *params)))
                else:
                    params = dist.fit(s)
                    log_lik = float(np.sum(dist.logpdf(s, *params)))

            if not math.isfinite(log_lik):
                continue

            k = len(params)
            aic = 2 * k - 2 * log_lik
            entry = {
                "name": name,
                "aic": _safe_float(aic),
                "params": [_safe_float(p) for p in params],
            }
            leaderboard.append(entry)
            if best is None or aic < best["aic"]:
                best = entry
        except Exception:
            continue

    leaderboard.sort(key=lambda e: e["aic"] if e["aic"] is not None else float("inf"))

    if best is None:
        return {"available": False, "reason": "No candidate distribution converged."}

    return {
        "available": True,
        "best_fit": best,
        "leaderboard": leaderboard[:6],
    }


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def _bootstrap_ci(series: pd.Series, statistic_fn, confidence: float = 0.95) -> dict:
    s = series.dropna().to_numpy()
    if len(s) < 20:
        return {"available": False, "reason": "n<20"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.bootstrap(
                (s,),
                statistic_fn,
                n_resamples=999,
                confidence_level=confidence,
                method="BCa",
                random_state=42,
            )
        return {
            "available": True,
            "low": _safe_float(res.confidence_interval.low),
            "high": _safe_float(res.confidence_interval.high),
            "method": "BCa",
            "n_resamples": 999,
        }
    except Exception as exc:
        # BCa can fail on degenerate data; fall back to percentile.
        try:
            res = stats.bootstrap(
                (s,),
                statistic_fn,
                n_resamples=999,
                confidence_level=confidence,
                method="percentile",
                random_state=42,
            )
            return {
                "available": True,
                "low": _safe_float(res.confidence_interval.low),
                "high": _safe_float(res.confidence_interval.high),
                "method": "percentile",
                "n_resamples": 999,
                "note": f"BCa failed ({exc}); used percentile.",
            }
        except Exception as exc2:
            return {"available": False, "reason": str(exc2)}


# ---------------------------------------------------------------------------
# Numeric per-column engine
# ---------------------------------------------------------------------------

def _numeric_column_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"status": "empty"}

    skew = _safe_float(s.skew())
    kurt = _safe_float(s.kurtosis())
    q1, q3 = _safe_float(s.quantile(0.25)), _safe_float(s.quantile(0.75))
    iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None

    return {
        "count": int(s.count()),
        "mean": _safe_float(s.mean()),
        "median": _safe_float(s.median()),
        "std": _safe_float(s.std()),
        "min": _safe_float(s.min()),
        "max": _safe_float(s.max()),
        "q1": q1,
        "q3": q3,
        "iqr": _safe_float(iqr),
        "skewness": skew,
        "skewness_label": _classify_skew(skew),
        "kurtosis": kurt,
        "kurtosis_label": _classify_kurtosis(kurt),
        "outliers": _outlier_flags(series),
        "normality": _normality(series),
        "distribution_fit": _fit_best_distribution(series),
        "bootstrap_mean_ci": _bootstrap_ci(series, np.mean),
        "bootstrap_median_ci": _bootstrap_ci(series, np.median),
    }


# ---------------------------------------------------------------------------
# Categorical per-column engine
# ---------------------------------------------------------------------------

def _categorical_column_stats(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False)
    total = max(len(series), 1)
    unique = int(series.nunique(dropna=True))

    if unique <= 1:
        cardinality = "Constant"
    elif unique <= 10:
        cardinality = "Low"
    elif unique <= 50:
        cardinality = "Medium"
    else:
        cardinality = "High"

    # Shannon entropy on category proportions
    p = counts.values / counts.values.sum()
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
    max_entropy = math.log2(unique) if unique > 1 else 0.0
    entropy_ratio = (entropy / max_entropy) if max_entropy > 0 else 0.0

    top = [
        {"value": str(v), "count": int(c), "percent": round(float(c / total * 100), 2)}
        for v, c in counts.head(10).items()
    ]

    return {
        "unique_count": unique,
        "cardinality_label": cardinality,
        "entropy": round(entropy, 4),
        "entropy_ratio": round(entropy_ratio, 4),
        "top_values": top,
    }


# ---------------------------------------------------------------------------
# Bivariate panel
# ---------------------------------------------------------------------------

def _classify_corr(value: float) -> str:
    a = abs(value)
    if a >= 0.8:
        return "Very Strong"
    if a >= 0.6:
        return "Strong"
    if a >= 0.4:
        return "Moderate"
    if a >= 0.2:
        return "Weak"
    return "Very Weak"


def _numeric_pair_stats(a: pd.Series, b: pd.Series) -> dict:
    both = a.notna() & b.notna()
    if both.sum() < 10:
        return {"available": False, "reason": "n<10 overlapping"}

    x = a[both].astype(float)
    y = b[both].astype(float)

    out: dict[str, Any] = {"available": True, "n": int(both.sum())}

    try:
        r, p = stats.pearsonr(x, y)
        out["pearson"] = {"r": _safe_float(r), "p": _safe_float(p), "strength": _classify_corr(r)}
    except Exception:
        out["pearson"] = {"r": None, "p": None}

    try:
        rho, p = stats.spearmanr(x, y)
        out["spearman"] = {"rho": _safe_float(rho), "p": _safe_float(p)}
    except Exception:
        out["spearman"] = {"rho": None, "p": None}

    try:
        tau, p = stats.kendalltau(x, y)
        out["kendall"] = {"tau": _safe_float(tau), "p": _safe_float(p)}
    except Exception:
        out["kendall"] = {"tau": None, "p": None}

    return out


def _cramers_v(a: pd.Series, b: pd.Series) -> dict:
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"available": False, "reason": "table too small"}

    try:
        chi2, p, dof, _ = stats.chi2_contingency(table)
        n = table.values.sum()
        v = math.sqrt(chi2 / (n * (min(table.shape) - 1))) if n > 0 else None

        if v is None:
            strength = "Unknown"
        elif v >= 0.5:
            strength = "Strong"
        elif v >= 0.3:
            strength = "Moderate"
        elif v >= 0.1:
            strength = "Weak"
        else:
            strength = "Very Weak"

        return {
            "available": True,
            "chi2": _safe_float(chi2),
            "p": _safe_float(p),
            "dof": int(dof),
            "cramers_v": _safe_float(v),
            "strength": strength,
            "relationship_likely": bool(p < 0.05) if p is not None else None,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _point_biserial(binary: pd.Series, numeric: pd.Series) -> dict:
    both = binary.notna() & numeric.notna()
    if both.sum() < 20:
        return {"available": False, "reason": "n<20"}

    b = binary[both]
    n = numeric[both].astype(float)

    # Encode binary to 0/1
    uniques = list(b.dropna().unique())
    if len(uniques) != 2:
        return {"available": False, "reason": "not binary"}
    mapping = {uniques[0]: 0, uniques[1]: 1}
    b_enc = b.map(mapping).astype(int)

    try:
        r, p = stats.pointbiserialr(b_enc, n)
        return {
            "available": True,
            "r": _safe_float(r),
            "p": _safe_float(p),
            "strength": _classify_corr(r),
            "encoding": {str(uniques[0]): 0, str(uniques[1]): 1},
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _group_difference_test(numeric: pd.Series, group: pd.Series) -> dict:
    """Mann-Whitney U for 2 groups; Kruskal-Wallis for >2."""
    both = numeric.notna() & group.notna()
    if both.sum() < 20:
        return {"available": False, "reason": "n<20"}

    n = numeric[both].astype(float)
    g = group[both]

    levels = list(g.unique())
    if len(levels) < 2 or len(levels) > 12:
        return {"available": False, "reason": "need 2-12 groups"}

    samples = [n[g == lv].values for lv in levels]
    samples = [s for s in samples if len(s) >= 3]
    if len(samples) < 2:
        return {"available": False, "reason": "groups too small"}

    try:
        if len(samples) == 2:
            stat, p = stats.mannwhitneyu(samples[0], samples[1], alternative="two-sided")
            return {
                "available": True,
                "test": "Mann-Whitney U",
                "statistic": _safe_float(stat),
                "p_value": _safe_float(p),
                "groups": [str(lv) for lv in levels[:2]],
                "significant": bool(p < 0.05),
            }
        else:
            stat, p = stats.kruskal(*samples)
            return {
                "available": True,
                "test": "Kruskal-Wallis H",
                "statistic": _safe_float(stat),
                "p_value": _safe_float(p),
                "groups": [str(lv) for lv in levels[: len(samples)]],
                "significant": bool(p < 0.05),
            }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_deep_statistics_v2(df: pd.DataFrame, max_pairs: int = 20) -> dict:
    """
    Run the full deep-statistics battery.

    Args:
        df: Input dataframe.
        max_pairs: Cap on bivariate pairs to compute (prevents O(n^2) blow-up).

    Returns:
        JSON-safe dict with numeric, categorical, and bivariate sections.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Per-column
        numeric_stats = {col: _numeric_column_stats(df[col]) for col in numeric_cols}
        categorical_stats = {col: _categorical_column_stats(df[col]) for col in categorical_cols}

        # Bivariate — numeric pairs
        numeric_pairs: list[dict] = []
        budget = max_pairs
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1 :]:
                if budget <= 0:
                    break
                stat = _numeric_pair_stats(df[a], df[b])
                if stat.get("available"):
                    numeric_pairs.append({"column_a": a, "column_b": b, **stat})
                    budget -= 1
            if budget <= 0:
                break
        numeric_pairs.sort(
            key=lambda p: abs(p.get("pearson", {}).get("r") or 0),
            reverse=True,
        )

        # Bivariate — categorical pairs
        categorical_pairs: list[dict] = []
        budget = max_pairs
        for i, a in enumerate(categorical_cols):
            for b in categorical_cols[i + 1 :]:
                if budget <= 0:
                    break
                if df[a].nunique(dropna=True) > 30 or df[b].nunique(dropna=True) > 30:
                    continue  # skip exploding contingency tables
                stat = _cramers_v(df[a], df[b])
                if stat.get("available"):
                    categorical_pairs.append({"column_a": a, "column_b": b, **stat})
                    budget -= 1
            if budget <= 0:
                break
        categorical_pairs.sort(
            key=lambda p: abs(p.get("cramers_v") or 0),
            reverse=True,
        )

        # Bivariate — binary categorical vs numeric (point-biserial)
        binary_numeric_pairs: list[dict] = []
        budget = max_pairs
        for cat in categorical_cols:
            if df[cat].nunique(dropna=True) != 2:
                continue
            for num in numeric_cols:
                if budget <= 0:
                    break
                stat = _point_biserial(df[cat], df[num])
                if stat.get("available"):
                    binary_numeric_pairs.append(
                        {"binary_column": cat, "numeric_column": num, **stat}
                    )
                    budget -= 1
            if budget <= 0:
                break

        # Group differences (low-card categorical vs numeric)
        group_difference_tests: list[dict] = []
        budget = max_pairs
        for cat in categorical_cols:
            uniq = df[cat].nunique(dropna=True)
            if uniq < 2 or uniq > 6:
                continue
            for num in numeric_cols:
                if budget <= 0:
                    break
                stat = _group_difference_test(df[num], df[cat])
                if stat.get("available"):
                    group_difference_tests.append(
                        {"group_column": cat, "numeric_column": num, **stat}
                    )
                    budget -= 1
            if budget <= 0:
                break

        return {
            "version": "v2",
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_statistics": numeric_stats,
            "categorical_statistics": categorical_stats,
            "bivariate": {
                "numeric_pairs": numeric_pairs,
                "categorical_pairs": categorical_pairs,
                "binary_numeric_pairs": binary_numeric_pairs,
                "group_difference_tests": group_difference_tests,
            },
            "summary": {
                "numeric_count": len(numeric_cols),
                "categorical_count": len(categorical_cols),
                "numeric_pairs_tested": len(numeric_pairs),
                "categorical_pairs_tested": len(categorical_pairs),
                "binary_numeric_pairs_tested": len(binary_numeric_pairs),
                "group_difference_tests_run": len(group_difference_tests),
            },
        }
