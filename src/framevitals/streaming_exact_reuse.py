"""Reuse full-stream sufficient statistics in bounded downstream analyses.

The streaming profiler has already scanned every selected source row and owns
higher-quality sufficient statistics than any bounded diagnostic sample can
reconstruct. This module enforces FrameVitals' exact-once rule: downstream
analysis payloads may add sample-dependent diagnostics, but they must not replace
known full-stream moments with noisier sample estimates.
"""

from __future__ import annotations

from typing import Any


_EXACT_NUMERIC_FIELD_MAP = {
    "count": "count",
    "mean": "mean",
    "std": "std",
    "min": "min",
    "max": "max",
}


def reuse_streaming_exact_statistics(payload: dict[str, Any]) -> dict[str, Any]:
    """Overlay full-stream numeric sufficient statistics onto deep diagnostics.

    Quantiles, skew/kurtosis, outlier views, distribution fits, hypothesis tests,
    confidence intervals, and bivariate tests remain bounded-sample diagnostics.
    Only statistics that the streaming profile already computed over every
    profiled source row are reused.
    """
    profile = payload.get("profile")
    deep = payload.get("deep_statistics_v2")
    if not isinstance(profile, dict) or not isinstance(deep, dict):
        return payload

    deep_numeric = deep.get("numeric_statistics")
    full_numeric = profile.get("numeric_summary")
    if not isinstance(deep_numeric, dict) or not isinstance(full_numeric, dict):
        return payload

    streaming = profile.get("streaming_metadata", {})
    backend = streaming.get("numeric_backend")
    profile_metadata = profile.get("numeric_summary_metadata", {})
    method = profile_metadata.get("method") if isinstance(profile_metadata, dict) else None

    reused_columns = 0
    for column, sample_summary in deep_numeric.items():
        if not isinstance(sample_summary, dict):
            continue
        exact_summary = full_numeric.get(column)
        if not isinstance(exact_summary, dict):
            continue

        reused_fields: list[str] = []
        for target_field, source_field in _EXACT_NUMERIC_FIELD_MAP.items():
            if source_field not in exact_summary:
                continue
            sample_summary[target_field] = exact_summary[source_field]
            reused_fields.append(target_field)

        if reused_fields:
            reused_columns += 1
            sample_summary["summary_provenance"] = {
                "scope": "full_stream",
                "backend": backend,
                "method": method,
                "reused_exact_fields": reused_fields,
                "sample_derived_fields": [
                    "q1",
                    "median",
                    "q3",
                    "iqr",
                    "skewness",
                    "kurtosis",
                    "outliers",
                    "normality",
                    "distribution_fit",
                    "bootstrap_mean_ci",
                    "bootstrap_median_ci",
                ],
            }

    if reused_columns:
        execution = deep.setdefault("execution", {})
        if isinstance(execution, dict):
            execution["exact_once_reuse"] = {
                "enabled": True,
                "source": "streaming_profile",
                "backend": backend,
                "method": method,
                "columns_reused": reused_columns,
                "fields": list(_EXACT_NUMERIC_FIELD_MAP),
            }

    return payload
