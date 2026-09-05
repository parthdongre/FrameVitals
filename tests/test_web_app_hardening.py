import pytest

pytest.importorskip("flask")

import app as web_app


def test_web_dataset_ids_reject_path_traversal():
    with pytest.raises(ValueError, match="dataset identifier"):
        web_app._validate_dataset_id("../../etc/passwd")
    assert web_app._validate_dataset_id("0123456789ab") == "0123456789ab"


def test_web_analysis_uses_canonical_mode_policy(monkeypatch):
    captured = {}

    def fake_run_full_analysis(**kwargs):
        captured.update(kwargs)
        return {"dataset_id": kwargs["dataset_id"]}

    monkeypatch.setattr(web_app, "run_full_analysis", fake_run_full_analysis)

    web_app._run_web_analysis(
        dataset_id="0123456789ab",
        original_filename="data.csv",
        analysis_mode="standard",
        target_column=None,
        dataframe=object(),
    )

    disabled = set(captured["disabled_modules"])
    assert "deep_statistics" in disabled
    assert "text_profile" in disabled
    assert "modeling" in disabled
    assert "explainability" in disabled


def test_web_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(web_app, "_WEB_CACHE_LIMIT", 2)
    with web_app.REPORT_LOCK:
        web_app.ANALYSIS_CACHE.clear()
        web_app.REPORT_JOBS.clear()

    web_app._cache_analysis("000000000001", {"value": 1})
    web_app._cache_analysis("000000000002", {"value": 2})
    web_app._cache_analysis("000000000003", {"value": 3})

    with web_app.REPORT_LOCK:
        assert list(web_app.ANALYSIS_CACHE) == ["000000000002", "000000000003"]
        web_app.ANALYSIS_CACHE.clear()
        web_app.REPORT_JOBS.clear()
