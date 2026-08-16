import framevitals as fv


def test_core_public_surface_is_exported():
    expected = {
        "AnalysisConfig",
        "AnalysisPlan",
        "AnalysisResult",
        "AnalysisSnapshot",
        "SnapshotHistory",
        "CleaningPlan",
        "ColumnResult",
        "DiagnosticResult",
        "DataCheck",
        "CheckResult",
        "DriftResult",
        "GateResult",
        "ValidationResult",
        "inspect_source",
        "profile",
        "roles",
        "health",
        "ml_readiness",
        "quality",
        "statistics",
        "anomalies",
        "relationships",
        "system_info",
        "target_analysis",
        "analyze",
        "plan",
        "plan_cleaning",
        "clean",
        "compare",
        "infer_contract",
        "validate",
        "check",
        "run_checks",
        "discover_checks",
        "gate",
        "available_modules",
        "create_snapshot",
        "load_snapshot",
        "compare_snapshots",
        "__version__",
    }

    assert set(fv.__all__) == expected
    for name in expected:
        assert hasattr(fv, name), name


def test_result_types_remain_dict_compatible():
    for result_type in (
        fv.AnalysisResult,
        fv.AnalysisSnapshot,
        fv.ColumnResult,
        fv.DiagnosticResult,
        fv.CheckResult,
        fv.DriftResult,
        fv.GateResult,
        fv.ValidationResult,
    ):
        assert issubclass(result_type, dict)
