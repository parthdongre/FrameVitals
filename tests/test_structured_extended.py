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



def test_optimizer_diagnostics_find_untracked_trainable_parameters_and_pulse_state():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class TwoBlockNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 4)
            self.second = nn.Linear(4, 1)

        def forward(self, x):
            return self.second(torch.relu(self.first(x)))

    model = TwoBlockNet()
    optimizer = torch.optim.Adam(model.first.parameters(), lr=1e-3)

    result = fv.prism(model, optimizer=optimizer, depth="quick")
    optimizer_report = result.analysis["model"]["optimizer"]

    assert optimizer_report["available"] is True
    assert optimizer_report["group_count"] == 1
    assert optimizer_report["untracked_trainable_parameters"]
    assert any(
        item["code"] == "model.optimizer.untracked_parameters"
        for item in result.beacons
    )

    snapshot = fv.pulse(model, optimizer=optimizer, depth="quick")
    state = snapshot["state"]["structured"]["optimizer"]
    assert state["group_count"] == 1
    assert state["untracked_trainable_parameters"]



def test_tensor_axiom_detects_shape_change():
    reference = np.eye(4, dtype=float)
    current = np.eye(5, dtype=float)

    established = fv.axiom(reference)
    assert established.contract["source_kind"] == "tensor"

    checked = fv.axiom(reference, current=current)
    assert checked.status == "fail"
    assert any(
        item["code"] == "axiom.tensor.shape"
        for item in checked.validation.findings
    )


def test_nested_axiom_requires_reference_top_level_keys():
    reference = {"user": {"id": 1}, "events": []}
    current = {"user": {"id": 1}}

    checked = fv.axiom(reference, current=current)

    assert checked.status == "fail"
    assert any(
        item["code"] == "axiom.nested.keys"
        for item in checked.validation.findings
    )


def test_relational_axiom_detects_missing_table():
    reference = {
        "customers": pd.DataFrame({"customer_id": [1, 2]}),
        "orders": pd.DataFrame({
            "order_id": [10, 11],
            "customer_id": [1, 2],
        }),
    }
    current = {
        "customers": pd.DataFrame({"customer_id": [1, 2]}),
    }

    checked = fv.axiom(reference, current=current)

    assert checked.status == "fail"
    assert any(
        item["code"] == "axiom.relational.tables"
        for item in checked.validation.findings
    )


def test_model_axiom_detects_parameter_shape_change():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    reference = nn.Linear(4, 2)
    current = nn.Linear(4, 3)

    checked = fv.axiom(reference, current=current)

    assert checked.status == "fail"
    assert any(
        item["code"] == "axiom.model.shape"
        for item in checked.validation.findings
    )



def test_onnx_independent_output_branches_are_not_false_disconnected(tmp_path):
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    TensorProto = onnx.TensorProto

    x1 = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [None, 2])
    x2 = helper.make_tensor_value_info("x2", TensorProto.FLOAT, [None, 2])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [None, 2])
    y2 = helper.make_tensor_value_info("y2", TensorProto.FLOAT, [None, 2])

    first = helper.make_node("Identity", ["x1"], ["y1"], name="branch_one")
    second = helper.make_node("Identity", ["x2"], ["y2"], name="branch_two")
    graph = helper.make_graph([first, second], "multi", [x1, x2], [y1, y2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    path = tmp_path / "multi-output.onnx"
    onnx.save(model, path)

    result = fv.prism(path)
    report = result.analysis["model"]

    assert report["components"] == 2
    assert report["disconnected_nodes"] == 0
    assert not any(
        item["code"] == "model.onnx.disconnected"
        for item in result.beacons
    )



def test_graph_file_prism_tide_core_and_spectral_diagnostics(tmp_path):
    nx = pytest.importorskip("networkx")

    first_graph = nx.cycle_graph(12)
    second_graph = nx.path_graph(12)

    first = tmp_path / "first.graphml"
    second = tmp_path / "second.graphml"
    nx.write_graphml(first_graph, first)
    nx.write_graphml(second_graph, second)

    inspected = fv.inspect_source(first)
    assert inspected["kind"] == "graph"
    assert inspected["metadata"]["format"] == "graphml"

    result = fv.prism(first, depth="quick")
    graph = result.analysis["graph"]

    assert result.analysis["source_kind"] == "graph"
    assert graph["nodes"] == 12
    assert graph["core"]["available"] is True
    assert graph["core"]["max_core"] >= 2
    assert graph["spectral"]["available"] is True
    assert graph["spectral"]["spectral_radius"] > 0

    change = fv.tide(first, second)
    assert change["source_kind"] == "graph"
    assert change["graph"]["degree_js_distance"] > 0


def test_transformer_architecture_diagnostics_are_exposed():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 8)
            self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
            self.norm = nn.LayerNorm(8)

        def forward(self, token_ids):
            x = self.embed(token_ids)
            out, _ = self.attn(x, x, x, need_weights=False)
            return self.norm(out)

    model = TinyTransformer()
    result = fv.prism(model, depth="quick")
    report = result.analysis["model"]

    assert report["architecture"] == "transformer"
    details = report["architecture_diagnostics"]["transformer"]
    assert details["available"] is True
    assert details["attention_modules"]
    assert details["attention_modules"][0]["heads"] == 2
    assert details["embeddings"]
    assert details["normalization"]


def test_recurrent_architecture_diagnostics_include_gate_spectra():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class TinyRecurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(input_size=4, hidden_size=4, num_layers=1)
            self.out = nn.Linear(4, 1)

        def forward(self, x):
            values, _ = self.rnn(x)
            return self.out(values)

    model = TinyRecurrent()
    result = fv.prism(model, depth="quick")
    report = result.analysis["model"]

    assert report["architecture"] == "recurrent"
    details = report["architecture_diagnostics"]["recurrent"]
    assert details["available"] is True
    assert details["modules"]
    layer = details["modules"][0]["layers"][0]
    assert len(layer["gate_norms"]) == 4
    assert len(layer["spectral_radii"]) == 4
