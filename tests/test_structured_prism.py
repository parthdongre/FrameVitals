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
