import json

import numpy as np
import pytest

import framevitals as fv


def test_tensor_prism_detects_rank_collapse():
    matrix = np.ones((8, 8), dtype=float)

    result = fv.prism(matrix, depth="quick")

    assert result.analysis["source_kind"] == "tensor"
    assert result.analysis["tensor"]["matrix"]["rank"] == 1
    assert result.analysis["tensor"]["matrix"]["rank_ratio"] == pytest.approx(0.125)
    assert any(item["code"] == "tensor.rank_deficiency" for item in result.beacons)
    assert "TENSOR" in result.summary_text()


def test_graph_prism_runs_bounded_network_diagnostics():
    nx = pytest.importorskip("networkx")

    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    graph.add_node(99)
    graph.add_edge(3, 3)

    inspected = fv.inspect_source(graph)
    assert inspected["kind"] == "graph"
    assert "shortest_paths" in inspected["capabilities"]

    result = fv.prism(graph, depth="quick")
    payload = result.analysis["graph"]

    assert result.analysis["source_kind"] == "graph"
    assert payload["nodes"] == 6
    assert payload["edges"] == 5
    assert payload["shortest_paths"]["available"] is True
    assert "pagerank" in payload["centrality"]

    codes = {item["code"] for item in result.beacons}
    assert "graph.fragmentation" in codes
    assert "graph.isolates" in codes
    assert "graph.self_loops" in codes

    rendered = result.summary_text()
    assert "GRAPH" in rendered
    assert "Nodes" in rendered
    assert "Edges" in rendered


def test_pytorch_cnn_prism_inspects_model_without_owning_training():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    model = TinyCNN()
    with torch.no_grad():
        model.conv.weight[0].zero_()

    inspected = fv.inspect_source(model)
    assert inspected["kind"] == "model"
    assert inspected["metadata"]["framework"] == "pytorch"

    result = fv.prism(model, depth="quick")
    model_report = result.analysis["model"]

    assert result.analysis["source_kind"] == "model"
    assert model_report["architecture"] == "cnn"
    assert model_report["parameters"] > 0
    assert model_report["cnn"]["available"] is True
    assert model_report["cnn"]["dead_filters"] >= 1
    assert any(item["code"] == "model.cnn.dead_filters" for item in result.beacons)

    rendered = result.summary_text()
    assert "MODEL" in rendered
    assert "Architecture" in rendered


def test_structured_prism_rejects_tabular_only_focus_for_now():
    tensor = np.arange(8, dtype=float)

    with pytest.raises(ValueError, match="tabular Prism"):
        fv.prism(tensor, focus="target")


def test_pytorch_runtime_prism_observes_activations_and_gradients_without_grad_side_effects():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class TinyRuntimeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = TinyRuntimeNet()
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.bias.fill_(-1.0)

    x = torch.ones(6, 4)
    y = torch.zeros(6, 1)

    assert all(parameter.grad is None for parameter in model.parameters())

    result = fv.prism(
        model,
        depth="quick",
        sample_batch=x,
        targets=y,
        loss_fn=nn.MSELoss(),
        backward=True,
    )

    runtime = result.analysis["model"]["runtime"]
    assert runtime["available"] is True
    assert runtime["backward"] is True
    assert runtime["observed_modules"] >= 3
    assert runtime["parameter_gradients"]
    assert runtime["summary"]["dead_activations"] >= 1

    codes = {item["code"] for item in result.beacons}
    assert "model.runtime.dead_activations" in codes
    assert all(parameter.grad is None for parameter in model.parameters())

    for module in model.modules():
        assert not module._forward_hooks
        assert not module._backward_hooks


def test_runtime_options_are_rejected_for_tensor_prism():
    tensor = np.arange(8, dtype=float)

    with pytest.raises(ValueError, match="only valid for model"):
        fv.prism(tensor, sample_batch=np.arange(4))


def test_tensor_tide_reports_aligned_change_metrics():
    reference = np.eye(6, dtype=float)
    current = reference.copy()
    current[0, 0] = 4.0

    change = fv.tide(reference, current)

    assert change["source_kind"] == "tensor"
    assert change["tensor"]["aligned"]["relative_l2"] > 0
    assert change["tensor"]["aligned"]["cosine_similarity"] < 1.0
    assert "TENSOR" in change.summary_text()


def test_graph_tide_reports_topology_change():
    nx = pytest.importorskip("networkx")

    reference = nx.path_graph(6)
    current = nx.star_graph(7)

    change = fv.tide(reference, current)

    assert change["source_kind"] == "graph"
    assert change["graph"]["reference_nodes"] == 6
    assert change["graph"]["current_nodes"] == 8
    assert change["graph"]["degree_js_distance"] > 0
    assert change.severity in {"minor", "moderate", "severe"}
    assert "GRAPH" in change.summary_text()


def test_model_tide_ranks_parameter_movement():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    reference = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    current = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    current.load_state_dict(reference.state_dict())

    with torch.no_grad():
        current[0].weight.add_(3.0)

    change = fv.tide(reference, current)

    assert change["source_kind"] == "model"
    assert change["model"]["parameters_compared"] > 0
    assert change["model"]["most_changed"]
    assert change["model"]["most_changed"][0]["name"] == "0.weight"
    assert change["model"]["p95_relative_l2"] > 0
    assert "MODEL" in change.summary_text()


def test_structured_tide_rejects_mixed_source_kinds():
    nx = pytest.importorskip("networkx")

    with pytest.raises(TypeError, match="same source kind"):
        fv.tide(np.arange(4, dtype=float), nx.path_graph(4))



def _write_fake_safetensors(path, tensors):
    header = {}
    offset = 0
    payload = bytearray()

    for name, spec in tensors.items():
        shape = list(spec["shape"])
        dtype = spec.get("dtype", "F32")
        elements = int(np.prod(shape)) if shape else 1
        item_size = 4 if dtype == "F32" else 2
        size = elements * item_size
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        payload.extend(b"\x00" * size)
        offset += size

    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(raw)) % 8
    raw += b" " * padding
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)


def test_safetensors_prism_reads_metadata_without_tensor_runtime(tmp_path):
    checkpoint = tmp_path / "tiny.safetensors"
    _write_fake_safetensors(
        checkpoint,
        {
            "model.layers.0.self_attn.q_proj.weight": {"shape": [2, 2]},
            "model.layers.0.self_attn.k_proj.weight": {"shape": [2, 2]},
            "model.embed_tokens.weight": {"shape": [8, 2]},
        },
    )

    inspected = fv.inspect_source(checkpoint)
    assert inspected["kind"] == "model"
    assert inspected["metadata"]["format"] == "safetensors"

    result = fv.prism(checkpoint, depth="quick")
    model = result.analysis["model"]

    assert result.analysis["source_kind"] == "model"
    assert model["framework"] == "safetensors"
    assert model["architecture"] == "transformer"
    assert model["tensor_count"] == 3
    assert model["parameters"] == 24
    assert result.analysis["execution"]["tensor_payloads_loaded"] is False


def test_safetensors_tide_detects_checkpoint_structure_change(tmp_path):
    reference = tmp_path / "reference.safetensors"
    current = tmp_path / "current.safetensors"

    _write_fake_safetensors(
        reference,
        {
            "layer.weight": {"shape": [2, 2]},
            "layer.bias": {"shape": [2]},
        },
    )
    _write_fake_safetensors(
        current,
        {
            "layer.weight": {"shape": [3, 2]},
            "layer.bias": {"shape": [2]},
            "new.weight": {"shape": [2, 2]},
        },
    )

    change = fv.tide(reference, current)

    assert change["source_kind"] == "model"
    assert change["model"]["comparison_scope"] == "metadata_only"
    assert len(change["model"]["added_parameters"]) == 1
    assert len(change["model"]["shape_changes"]) == 1
    assert change.severity in {"minor", "moderate", "severe"}
