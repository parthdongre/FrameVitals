import numpy as np
import pandas as pd

import framevitals as fv
from framevitals.sources import DatasetMetadata


def _frame(rows: int = 800) -> pd.DataFrame:
    values = np.arange(rows, dtype=np.float64)
    return pd.DataFrame({
        "value": values,
        "other": values * 2.0 + 1.0,
        "third": np.sin(values / 10.0),
    })


class _LoadedSource:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def inspect(self):
        return DatasetMetadata(
            name="loaded-source",
            kind="remote",
            format="custom",
            rows=len(self.frame),
            columns=len(self.frame.columns),
            size_bytes=None,
            materialized=False,
            supports_projection=False,
            supports_streaming=False,
        )

    def load(self):
        return self.frame.copy()


def test_dataframe_statistics_use_execution_schema_v1():
    result = fv.statistics(_frame(), mode="quick", max_pairs=2)
    execution = result["execution"]

    assert execution["execution_schema_version"] == "1"
    assert execution["method"] == "bounded_deep_statistics"
    assert execution["scope"] == "bounded_deep_statistics"
    assert execution["full_materialization"] is False
    assert execution["source"]["format"] == "pandas"


def test_non_pandas_statistics_report_full_materialization():
    result = fv.statistics(_LoadedSource(_frame()), mode="quick", max_pairs=2)
    execution = result["execution"]

    assert execution["execution_schema_version"] == "1"
    assert execution["method"] == "bounded_deep_statistics"
    assert execution["full_materialization"] is True
    assert execution["source"]["format"] == "custom"


def test_dataframe_anomalies_use_execution_schema_v1():
    result = fv.anomalies(
        _frame(),
        mode="quick",
        max_columns=3,
        top_k=5,
    )
    execution = result["execution"]

    assert execution["execution_schema_version"] == "1"
    assert execution["method"] == "bounded_anomaly_detection"
    assert execution["scope"] == "bounded_anomaly_detection"
    assert execution["full_materialization"] is False
    assert execution["source"]["format"] == "pandas"


def test_relationships_use_execution_schema_v1_and_preserve_legacy_sample():
    result = fv.relationships(_frame(), max_sample_rows=64)
    execution = result["execution"]

    assert execution["execution_schema_version"] == "1"
    assert execution["method"] == "bounded_relationship_graph"
    assert execution["full_materialization"] is False
    assert execution["sampled"] is True
    assert execution["sample_rows"] == 64
    assert execution["source_rows"] == 800
    assert execution["source"]["format"] == "pandas"

    assert result["sample"]["sampled"] is True
    assert result["sample"]["sample_rows"] == 64
    assert result["full_materialization"] is False


def test_exact_validation_and_gate_use_execution_schema_v1():
    frame = _frame(100)
    contract = fv.infer_contract(frame)

    validation = fv.validate(frame, contract)
    assert validation["execution"]["execution_schema_version"] == "1"
    assert validation["execution"]["method"] == "exact_contract_validation"
    assert validation["execution"]["sampled"] is False

    gate = fv.gate(frame, contract=contract)
    assert gate["execution"]["execution_schema_version"] == "1"
    assert gate["execution"]["method"] == "quality_gate"
    assert gate["execution"]["validation"]["execution_schema_version"] == "1"
