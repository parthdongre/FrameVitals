import json

import numpy as np
import pandas as pd
import pytest

import framevitals as fv
from framevitals.snapshots import SnapshotHistory


def test_nested_prism_detects_conflicting_types():
    data = {
        "users": [
            {"id": 1, "score": 4.5, "tags": ["a"]},
            {"id": 2, "score": "unknown", "tags": ["b", "c"]},
        ]
    }

    inspected = fv.inspect_source(data)
    assert inspected["kind"] == "nested"

    result = fv.prism(data, depth="quick")

    assert result.analysis["source_kind"] == "nested"
    assert result.analysis["nested"]["path_type_conflict_count"] >= 1
    assert any(item["code"] == "nested.type_conflicts" for item in result.beacons)
    assert "NESTED" in result.summary_text()


def test_nested_tide_reports_key_and_shape_change():
    reference = {"user": {"id": 1, "name": "a"}}
    current = {"user": {"id": 1, "email": "a@example.com"}, "events": [1, 2, 3]}

    change = fv.tide(reference, current)

    assert change["source_kind"] == "nested"
    assert change["nested"]["current_nodes"] > change["nested"]["reference_nodes"]
    assert change.severity in {"minor", "moderate", "severe"}
    assert "NESTED" in change.summary_text()


def test_pulse_snapshots_tensor_state_and_history(tmp_path):
    baseline = np.eye(4, dtype=float)
    changed = baseline.copy()
    changed[0, 0] = 5.0

    first = fv.pulse(baseline)
    second = fv.pulse(changed)

    assert first["state"]["source_kind"] == "tensor"
    assert first["state"]["structured"]["shape"] == [4, 4]
    assert first["fingerprint"] != second["fingerprint"]

    history = SnapshotHistory(tmp_path / "history")
    history.add(first, label="baseline")
    history.add(second, label="changed")

    diff = history.compare_latest()
    assert diff["source_kind"]["reference"] == "tensor"
    assert diff["structured"]["changed"] is True

    timeline = history.timeline()
    assert [row["source_kind"] for row in timeline] == ["tensor", "tensor"]
    assert timeline[0]["structured"]["rank_ratio"] == pytest.approx(1.0)


def test_pulse_accepts_prism_result_for_nested_state():
    prism = fv.prism({"a": [1, 2, 3]})
    snapshot = fv.pulse(prism)

    assert snapshot["state"]["source_kind"] == "nested"
    assert snapshot["state"]["structured"]["nodes_observed"] >= 4


def test_onnx_prism_and_tide_when_optional_dependency_is_available(tmp_path):
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    TensorProto = onnx.TensorProto

    def build(path, output_features):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, output_features])
        weight = helper.make_tensor(
            "weight",
            TensorProto.FLOAT,
            [2, output_features],
            [0.1] * (2 * output_features),
        )
        bias = helper.make_tensor(
            "bias",
            TensorProto.FLOAT,
            [output_features],
            [0.0] * output_features,
        )
        node = helper.make_node(
            "Gemm",
            ["x", "weight", "bias"],
            ["y"],
            name="linear",
        )
        graph = helper.make_graph([node], "tiny", [x], [y], [weight, bias])
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18)],
        )
        onnx.save(model, path)

    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    build(first, 2)
    build(second, 3)

    inspected = fv.inspect_source(first)
    assert inspected["kind"] == "model"
    assert inspected["metadata"]["format"] == "onnx"

    result = fv.prism(first)
    model = result.analysis["model"]
    assert model["framework"] == "onnx"
    assert model["nodes"] == 1
    assert model["initializer_count"] == 2
    assert model["parameters"] == 6

    change = fv.tide(first, second)
    assert change["source_kind"] == "model"
    assert change["model"]["comparison_scope"] == "graph_and_metadata"
    assert len(change["model"]["shape_changes"]) >= 1



def test_relational_prism_infers_keys_and_referential_gaps():
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["a", "b", "c"],
    })
    orders = pd.DataFrame({
        "order_id": [10, 11, 12, 13],
        "customer_id": [1, 2, 2, 99],
        "amount": [5.0, 6.0, 7.0, 8.0],
    })

    project = {"customers": customers, "orders": orders}
    inspected = fv.inspect_source(project)
    assert inspected["kind"] == "relational"

    result = fv.prism(project, depth="quick")
    report = result.analysis["relational"]

    assert result.analysis["source_kind"] == "relational"
    assert report["table_count"] == 2
    assert report["relationship_count"] >= 1
    customer_links = [
        item for item in report["relationships"]
        if item["column"] == "customer_id"
    ]
    assert customer_links
    assert customer_links[0]["child_reference_coverage"] < 1.0
    assert any(
        item["code"] == "relational.referential_gaps"
        for item in result.beacons
    )


def test_relational_tide_and_pulse_track_project_change(tmp_path):
    reference = {
        "customers": pd.DataFrame({"customer_id": [1, 2]}),
        "orders": pd.DataFrame({
            "order_id": [10, 11],
            "customer_id": [1, 2],
        }),
    }
    current = {
        "customers": pd.DataFrame({"customer_id": [1, 2, 3]}),
        "orders": pd.DataFrame({
            "order_id": [10, 11, 12],
            "customer_id": [1, 2, 3],
            "status": ["ok", "ok", "ok"],
        }),
        "products": pd.DataFrame({"product_id": [100, 101]}),
    }

    change = fv.tide(reference, current)
    assert change["source_kind"] == "relational"
    assert change["relational"]["added_tables"] == ["products"]
    assert "orders" in change["relational"]["column_changes"]

    snapshot = fv.pulse(current)
    assert snapshot["state"]["source_kind"] == "relational"
    assert snapshot["state"]["structured"]["table_count"] == 3

    history = SnapshotHistory(tmp_path / "relational-history")
    history.add(fv.pulse(reference), label="before")
    history.add(snapshot, label="after")
    assert history.compare_latest()["structured"]["changed"] is True
