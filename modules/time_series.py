"""
Time-Series Auto-Detection (WS-5)
==================================
Detect a date-like column with high confidence, then run a research-grade
time-series mini-pipeline:

    1. Date detection: ≥70% parse-rate AND mostly-monotone, picks the strongest
       candidate when several columns qualify.
    2. Frequency inference: from inter-arrival mode + pandas.infer_freq.
    3. Numeric series selection: prefer the highest-variance non-ID numeric col
       (or use a target_column if supplied).
    4. Stationarity battery: ADF + KPSS (with consensus verdict).
    5. Decomposition: STL with FFT-detected period.
    6. Autocorrelation: ACF + PACF up to lag = min(40, n//5).
    7. Forecast preview: Holt-Winters (additive) one period ahead, with a naive
       baseline for honest comparison.

The whole module is **best-effort** — every step is wrapped, so a single
unsupported branch never breaks the pipeline. Result is JSON-safe.

Public entry point:
    detect_and_analyze_time_series(df, target_column=None) -> dict
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# JSON helpers
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


# ---------------------------------------------------------------------------
# Date detection
# ---------------------------------------------------------------------------

_DATE_KEYWORDS = (
    "date", "time", "timestamp", "created", "updated", "datetime",
    "ts", "event", "recv", "recorded", "occurred",
)


def _try_parse(series: pd.Series) -> pd.Series:
    """Best-effort to_datetime that tolerates mixed and integer-epoch formats."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    s = series.dropna()
    if s.empty:
        return pd.Series(pd.NaT, index=series.index)

    # Heuristic: if it looks like seconds-since-epoch (10 digits) or
    # milliseconds (13 digits), use the unit kwarg
    if pd.api.types.is_numeric_dtype(s):
        as_int = s.astype("int64", errors="ignore")
        if as_int.max() > 1e18:
            return pd.to_datetime(series, unit="ns", errors="coerce")
        if as_int.max() > 1e15:
            return pd.to_datetime(series, unit="us", errors="coerce")
        if as_int.max() > 1e12:
            return pd.to_datetime(series, unit="ms", errors="coerce")
        if as_int.max() > 1e9:
            return pd.to_datetime(series, unit="s", errors="coerce")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", format="mixed")


def detect_date_column(df: pd.DataFrame) -> dict | None:
    """Find the strongest date-like column. Returns None if none qualifies."""
    candidates: list[tuple[float, str, pd.Series]] = []

    for col in df.columns:
        lower = col.lower()
        looks_like_date_name = any(kw in lower for kw in _DATE_KEYWORDS)

        # Skip the obvious non-dates: low-cardinality numeric columns that
        # don't have a date-y name. They get falsely parsed as ns/us/ms epochs
        # otherwise (e.g. a binary `approved` column would parse to dates).
        original = df[col]
        if pd.api.types.is_numeric_dtype(original) and not looks_like_date_name:
            unique_count = int(original.nunique(dropna=True))
            if unique_count < 20:
                continue
            # Reject small integer columns regardless of cardinality — these are
            # clearly not epoch timestamps.
            try:
                max_abs = float(original.abs().max())
                if not math.isfinite(max_abs) or max_abs < 1e9:
                    continue
            except Exception:
                continue

        sample = original.dropna().head(50)
        if sample.empty:
            continue

        parsed = _try_parse(original)
        parse_rate = parsed.notna().mean()

        if parse_rate < (0.7 if looks_like_date_name else 0.95):
            continue

        # Reward monotone increasing or decreasing series
        sorted_view = parsed.dropna().sort_index()
        if len(sorted_view) < 5:
            continue

        diffs = sorted_view.diff().dropna()
        increasing = (diffs >= pd.Timedelta(0)).mean()
        decreasing = (diffs <= pd.Timedelta(0)).mean()
        monotone_score = max(increasing, decreasing)

        score = (
            parse_rate * 0.6
            + monotone_score * 0.3
            + (0.1 if looks_like_date_name else 0.0)
        )
        candidates.append((float(score), col, parsed))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    score, col, parsed = candidates[0]
    return {"column": col, "score": round(score, 4), "parsed": parsed, "parse_rate": float(parsed.notna().mean())}


# ---------------------------------------------------------------------------
# Frequency / period inference
# ---------------------------------------------------------------------------

def _infer_frequency(parsed: pd.Series) -> dict:
    parsed = parsed.dropna().sort_values()
    if len(parsed) < 3:
        return {"available": False, "reason": "n<3"}

    # First try pandas built-in
    inferred = None
    try:
        inferred = pd.infer_freq(parsed.head(200))
    except Exception:
        inferred = None

    diffs = parsed.diff().dropna()
    if diffs.empty:
        return {"available": False, "reason": "no diffs"}

    median_seconds = float(diffs.dt.total_seconds().median())
    mean_seconds = float(diffs.dt.total_seconds().mean())

    # Map seconds to a friendly label
    label = "irregular"
    if median_seconds <= 0:
        label = "irregular"
    elif median_seconds <= 1.5:
        label = "second"
    elif median_seconds <= 90:
        label = "minute"
    elif median_seconds <= 90 * 60:
        label = "hour"
    elif median_seconds <= 36 * 3600:
        label = "day"
    elif median_seconds <= 9 * 24 * 3600:
        label = "week"
    elif median_seconds <= 35 * 24 * 3600:
        label = "month"
    elif median_seconds <= 200 * 24 * 3600:
        label = "quarter"
    else:
        label = "year"

    return {
        "available": True,
        "pandas_inferred": inferred,
        "median_seconds": _safe_float(median_seconds),
        "mean_seconds": _safe_float(mean_seconds),
        "label": label,
        "n_samples": int(len(parsed)),
    }


def _guess_period_via_fft(values: np.ndarray, freq_label: str) -> int | None:
    """Pick the dominant non-DC frequency from the FFT spectrum."""
    if len(values) < 16:
        return None
    series = values - np.nanmean(values)
    series = np.where(np.isfinite(series), series, 0.0)
    spectrum = np.abs(np.fft.rfft(series))
    # ignore the DC component
    if len(spectrum) < 3:
        return None
    spectrum[0] = 0
    peak = int(np.argmax(spectrum))
    if peak == 0:
        return None
    period = max(2, int(round(len(values) / peak)))

    # Sanity: respect freq label
    bounds = {
        "second": (5, 300),
        "minute": (5, 240),
        "hour": (4, 168),
        "day": (5, 60),
        "week": (4, 26),
        "month": (4, 24),
        "quarter": (4, 12),
        "year": (3, 10),
    }
    lo, hi = bounds.get(freq_label, (4, 60))
    if not (lo <= period <= hi):
        return None
    return period


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def _stationarity(series: pd.Series) -> dict:
    out: dict[str, Any] = {"available": False}
    s = series.dropna()
    if len(s) < 30:
        return {"available": False, "reason": "n<30"}

    try:
        from statsmodels.tsa.stattools import adfuller, kpss

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
            try:
                kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
            except Exception:
                kpss_stat, kpss_p = None, None

        adf_stationary = adf_p < 0.05 if adf_p is not None else None
        kpss_stationary = kpss_p > 0.05 if kpss_p is not None else None

        if adf_stationary and kpss_stationary:
            verdict = "stationary"
        elif adf_stationary is False and kpss_stationary is False:
            verdict = "non-stationary"
        elif adf_stationary and kpss_stationary is False:
            verdict = "trend-stationary (consider differencing once)"
        elif adf_stationary is False and kpss_stationary:
            verdict = "difference-stationary"
        else:
            verdict = "inconclusive"

        return {
            "available": True,
            "adf": {"statistic": _safe_float(adf_stat), "p_value": _safe_float(adf_p)},
            "kpss": {"statistic": _safe_float(kpss_stat), "p_value": _safe_float(kpss_p)},
            "verdict": verdict,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# STL decomposition
# ---------------------------------------------------------------------------

def _stl_decomposition(series: pd.Series, period: int) -> dict:
    if period < 2 or len(series) < 2 * period + 4:
        return {"available": False, "reason": "series too short for the inferred period"}

    try:
        from statsmodels.tsa.seasonal import STL

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl = STL(series, period=period, robust=True).fit()

        # Strength of trend / seasonality (Hyndman, 2008)
        var_resid = float(np.var(stl.resid.dropna()))
        var_trend_plus_resid = float(np.var((stl.trend + stl.resid).dropna()))
        var_seasonal_plus_resid = float(np.var((stl.seasonal + stl.resid).dropna()))
        trend_strength = max(0.0, 1 - var_resid / var_trend_plus_resid) if var_trend_plus_resid > 0 else 0.0
        seasonal_strength = max(0.0, 1 - var_resid / var_seasonal_plus_resid) if var_seasonal_plus_resid > 0 else 0.0

        return {
            "available": True,
            "period": int(period),
            "trend_strength": _safe_float(trend_strength, 4),
            "seasonal_strength": _safe_float(seasonal_strength, 4),
            "trend_summary": {
                "first": _safe_float(stl.trend.iloc[0]),
                "last": _safe_float(stl.trend.iloc[-1]),
                "delta": _safe_float(stl.trend.iloc[-1] - stl.trend.iloc[0]),
            },
            "residual_summary": {
                "mean": _safe_float(stl.resid.mean()),
                "std": _safe_float(stl.resid.std()),
            },
            # Compact preview series so the frontend can plot them
            "trend_preview": [_safe_float(v) for v in stl.trend.tail(50).tolist()],
            "seasonal_preview": [_safe_float(v) for v in stl.seasonal.tail(50).tolist()],
            "residual_preview": [_safe_float(v) for v in stl.resid.tail(50).tolist()],
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# ACF / PACF
# ---------------------------------------------------------------------------

def _autocorrelation(series: pd.Series, max_lag: int) -> dict:
    s = series.dropna()
    if len(s) < 10 or max_lag < 2:
        return {"available": False, "reason": "n<10 or max_lag<2"}

    try:
        from statsmodels.tsa.stattools import acf, pacf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acf_values = acf(s, nlags=max_lag, fft=True)
            pacf_values = pacf(s, nlags=max_lag)

        return {
            "available": True,
            "max_lag": int(max_lag),
            "acf": [_safe_float(v) for v in acf_values.tolist()],
            "pacf": [_safe_float(v) for v in pacf_values.tolist()],
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Holt-Winters forecast
# ---------------------------------------------------------------------------

def _forecast_holt_winters(series: pd.Series, period: int | None) -> dict:
    s = series.dropna()
    if len(s) < 30:
        return {"available": False, "reason": "n<30"}

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception as exc:
        return {"available": False, "reason": f"statsmodels not available: {exc}"}

    horizon = max(period or 12, 12)
    horizon = min(horizon, max(6, len(s) // 4))

    train_end = max(20, len(s) - horizon)
    train = s.iloc[:train_end]
    test = s.iloc[train_end : train_end + horizon]

    seasonal_periods = period if period and period >= 2 and len(train) >= 2 * period else None
    seasonal = "add" if seasonal_periods else None
    trend = "add"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True)
        hw_forecast = model.forecast(len(test))
    except Exception as exc:
        return {"available": False, "reason": f"Holt-Winters failed: {exc}"}

    # Naive baseline: last value
    naive_forecast = pd.Series([float(train.iloc[-1])] * len(test), index=test.index)

    def _metrics(y_true: pd.Series, y_pred) -> dict:
        if y_pred is None or len(y_true) == 0:
            return {"mae": None, "rmse": None}
        # Coerce y_pred to a plain ndarray of the same length to avoid index-alignment
        # explosions when statsmodels returns a series indexed differently from y_true.
        try:
            y_true_arr = np.asarray(y_true, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            if y_pred_arr.shape != y_true_arr.shape:
                # truncate to common length
                k = min(y_true_arr.shape[0], y_pred_arr.shape[0])
                if k == 0:
                    return {"mae": None, "rmse": None}
                y_true_arr = y_true_arr[:k]
                y_pred_arr = y_pred_arr[:k]
            mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
            if not mask.any():
                return {"mae": None, "rmse": None}
            diff = y_pred_arr[mask] - y_true_arr[mask]
            mae = float(np.abs(diff).mean())
            rmse = float(np.sqrt((diff**2).mean()))
            return {"mae": _safe_float(mae), "rmse": _safe_float(rmse)}
        except Exception:
            return {"mae": None, "rmse": None}

    return {
        "available": True,
        "horizon": int(len(test)),
        "train_size": int(len(train)),
        "method": "Holt-Winters (additive)",
        "seasonal_periods": seasonal_periods,
        "holt_winters": {
            "metrics": _metrics(test, hw_forecast),
            "preview": [_safe_float(v) for v in hw_forecast.tail(20).tolist()],
        },
        "naive_baseline": {
            "metrics": _metrics(test, naive_forecast),
            "preview": [_safe_float(v) for v in naive_forecast.tail(20).tolist()],
        },
        "actual_preview": [_safe_float(v) for v in test.tail(20).tolist()],
    }


# ---------------------------------------------------------------------------
# Numeric series picker
# ---------------------------------------------------------------------------

_ID_KEYWORDS = ("id", "uuid", "hash", "sequence", "step", "index")


def _pick_numeric_series(df: pd.DataFrame, target: str | None) -> str | None:
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        return target

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return None

    candidates = []
    for col in numeric.columns:
        lower = col.lower()
        if any(kw in lower for kw in _ID_KEYWORDS):
            continue
        if numeric[col].nunique(dropna=True) <= 1:
            continue
        var = float(numeric[col].var())
        if not math.isfinite(var):
            continue
        candidates.append((var, col))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_and_analyze_time_series(df: pd.DataFrame, target_column: str | None = None) -> dict:
    """
    Auto-detect a time index and run the time-series mini-pipeline.

    Returns a JSON-safe dict. If no usable date column is detected,
    returns {"available": False, ...}.
    """
    detected = detect_date_column(df)
    if detected is None:
        return {"available": False, "reason": "No date-like column detected with high confidence."}

    parsed = detected["parsed"]
    column_name = detected["column"]

    numeric_col = _pick_numeric_series(df, target_column)
    if numeric_col is None:
        return {
            "available": False,
            "reason": "No usable numeric series found to pair with the detected date column.",
            "detected_date_column": column_name,
            "date_score": detected["score"],
        }

    # Build a regular series indexed by the detected datetime
    paired = pd.DataFrame({column_name: parsed, numeric_col: df[numeric_col]}).dropna()
    paired = paired.sort_values(column_name).set_index(column_name)
    series = paired[numeric_col].astype(float)

    if len(series) < 30:
        return {
            "available": False,
            "reason": f"Only {len(series)} observations after pairing — need ≥30.",
            "detected_date_column": column_name,
            "numeric_column": numeric_col,
        }

    freq = _infer_frequency(parsed)
    period = _guess_period_via_fft(series.values, freq.get("label", ""))
    stationarity = _stationarity(series)
    decomposition = _stl_decomposition(series, period) if period else {"available": False, "reason": "no period detected"}

    max_lag = min(40, max(2, len(series) // 5))
    acf_pacf = _autocorrelation(series, max_lag)

    forecast = _forecast_holt_winters(series, period)

    return {
        "available": True,
        "detected_date_column": column_name,
        "date_score": detected["score"],
        "numeric_column": numeric_col,
        "n_observations": int(len(series)),
        "date_range": {
            "start": str(series.index.min()),
            "end": str(series.index.max()),
        },
        "frequency": freq,
        "period_estimate": period,
        "stationarity": stationarity,
        "decomposition": decomposition,
        "autocorrelation": acf_pacf,
        "forecast": forecast,
    }
