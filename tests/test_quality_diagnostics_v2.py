import numpy as np
import pandas as pd

import framevitals
from framevitals.column_roles import infer_column_roles
from framevitals.profiler import build_profile
from framevitals.quality_diagnostics import run_quality_diagnostics


def _messy_frame() -> pd.DataFrame:
    rows = 40
    missing_pattern = [None if index < 10 else index for index in range(rows)]
    return pd.DataFrame({
        "customer_id": [f"CUST-{index:03d}" for index in range(rows)],
        "order_id": [f"ORD-{index // 2:03d}" for index in range(rows)],
        "status": ["active"] * 38 + ["inactive"] * 2,
        "base": list(range(rows)),
        "base_copy": list(range(rows)),
        "numeric_text": [str(index % 20) for index in range(rows)],
        "event_text": [f"2026-01-{(index % 20) + 1:02d}" for index in range(rows)],
        "city": ["Pune", " pune ", "PUNE", "Mumbai"] * 10,
        "notes": ["", "ok", "  ", "ready"] * 10,
        "ratio": [float(index) for index in range(rows - 1)] + [np.inf],
        "mixed": ["1", 2, "3", 4] * 10,
        "missing_a": missing_pattern,
        "missing_b": list(missing_pattern),
    })


def test_quality_diagnostics_detect_real_world_data_problems():
    df = _messy_frame()
    profile = build_profile(df)
    roles = infer_column_roles(df)

    result = run_quality_diagnostics(
        df,
        profile=profile,
        column_roles=roles,
    )

    assert result["available"] is True
    assert result["summary"]["issue_count"] > 0

    key_columns = {item["column"] for item in result["primary_key_candidates"]}
    assert "customer_id" in key_columns

    duplicate_id_columns = {
        item["column"] for item in result["identifier_duplicates"]
    }
    assert "order_id" in duplicate_id_columns

    quasi_columns = {item["column"] for item in result["quasi_constant_columns"]}
    assert "status" in quasi_columns

    duplicate_groups = result["duplicate_columns"]
    assert any(
        item["canonical_column"] == "base"
        and "base_copy" in item["duplicate_columns"]
        for item in duplicate_groups
    )

    coercions = {
        (item["column"], item["suggested_type"])
        for item in result["coercion_candidates"]
    }
    assert ("numeric_text", "numeric") in coercions
    assert ("event_text", "datetime") in coercions

    normalized_columns = {
        item["column"] for item in result["category_normalisation"]
    }
    assert "city" in normalized_columns

    blank_columns = {item["column"] for item in result["blank_strings"]}
    assert "notes" in blank_columns

    infinity_columns = {item["column"] for item in result["infinite_values"]}
    assert "ratio" in infinity_columns

    mixed_columns = {item["column"] for item in result["mixed_object_types"]}
    assert "mixed" in mixed_columns

    pairs = {
        tuple(item["columns"])
        for item in result["missingness_relationships"]
    }
    assert ("missing_a", "missing_b") in pairs


def test_quality_diagnostics_are_bounded_and_report_column_truncation():
    df = pd.DataFrame({
        f"col_{index}": list(range(100))
        for index in range(8)
    })

    result = run_quality_diagnostics(
        df,
        max_sample_rows=20,
        max_columns=3,
    )

    assert result["columns_checked"] == 3
    assert result["truncated_columns"] is True
    assert result["max_sample_rows"] == 20


def test_public_analysis_surfaces_quality_diagnostics_as_findings():
    result = framevitals.analyze(
        _messy_frame(),
        mode="quick",
        disabled_modules=["cleaning", "ai"],
    )

    diagnostics = result["quality_diagnostics"]
    assert diagnostics["available"] is True
    assert result["execution"]["module_status"]["quality_diagnostics"] == "ran"
    assert "quality_diagnostics" in result["timings_ms"]

    quality_findings = [
        finding
        for finding in result.findings
        if finding["code"].startswith("quality.")
    ]
    codes = {finding["code"] for finding in quality_findings}

    assert any(code.startswith("quality.identifier_duplicates.order_id") for code in codes)
    assert any(code.startswith("quality.duplicate_columns.base") for code in codes)
    assert any(code.startswith("quality.infinite_values.ratio") for code in codes)
    assert result.column("city")["quality_findings"]


def test_quality_diagnostics_can_be_disabled_without_removing_result_key():
    result = framevitals.analyze(
        _messy_frame(),
        mode="quick",
        disabled_modules=["quality_diagnostics", "cleaning", "ai"],
    )

    assert result["quality_diagnostics"]["skipped"] is True
    assert result["execution"]["module_status"]["quality_diagnostics"] == "disabled"
    assert not any(
        finding["code"].startswith("quality.")
        for finding in result.findings
    )
