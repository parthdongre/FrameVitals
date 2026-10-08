"""Regression tests for bounded, non-training runtime model observation."""

import pytest

import framevitals as fv


def test_runtime_observation_preserves_batchnorm_dropout_and_torch_rng():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = nn.Sequential(
        nn.BatchNorm1d(4),
        nn.Dropout(p=0.75),
        nn.Linear(4, 2),
    )
    model.train()
    model[1].eval()  # Deliberately mixed training/eval state.

    batch = torch.randn(10, 4)
    before_modes = [module.training for module in model.modules()]
    before_mean = model[0].running_mean.clone()
    before_var = model[0].running_var.clone()
    before_batches = model[0].num_batches_tracked.clone()
    before_rng = torch.get_rng_state().clone()

    report = fv.prism(model, sample_batch=batch, depth="quick")
    runtime = report.analysis["model"]["runtime"]

    assert runtime["observation_mode"] == "temporary_eval"
    assert runtime["training_flags_restored"] is True
    assert runtime["torch_rng_restored"] is True
    assert runtime["model_buffer_purity_guaranteed"] is False
    assert [module.training for module in model.modules()] == before_modes
    assert torch.equal(model[0].running_mean, before_mean)
    assert torch.equal(model[0].running_var, before_var)
    assert torch.equal(model[0].num_batches_tracked, before_batches)
    assert torch.equal(torch.get_rng_state(), before_rng)
    for module in model.modules():
        assert not module._forward_hooks


def test_backward_observation_preserves_existing_gradient_buffers():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 2))
    model.train()
    batch = torch.randn(8, 4)
    targets = torch.randn(8, 2)

    old_grads = {}
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 7.0)
        old_grads[id(parameter)] = parameter.grad.clone()

    before_mean = model[0].running_mean.clone()
    before_var = model[0].running_var.clone()
    before_rng = torch.get_rng_state().clone()

    report = fv.prism(
        model,
        sample_batch=batch,
        targets=targets,
        loss_fn=nn.MSELoss(),
        backward=True,
        depth="quick",
    )

    runtime = report.analysis["model"]["runtime"]
    assert runtime["backward"] is True
    assert runtime["parameter_gradients"]
    assert model.training is True
    assert torch.equal(model[0].running_mean, before_mean)
    assert torch.equal(model[0].running_var, before_var)
    assert torch.equal(torch.get_rng_state(), before_rng)

    for parameter in model.parameters():
        assert torch.equal(parameter.grad, old_grads[id(parameter)])
    for module in model.modules():
        assert not module._forward_hooks


def test_runtime_observation_restores_model_state_when_forward_raises():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class FaultyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.BatchNorm1d(4)
            self.project = nn.Linear(4, 2)

        def forward(self, x):
            self.norm(x)
            torch.rand(12)  # Must not advance caller's global torch RNG.
            raise RuntimeError("intentional forward failure")

    model = FaultyNet().train()
    batch = torch.randn(8, 4)
    before_modes = [module.training for module in model.modules()]
    before_mean = model.norm.running_mean.clone()
    before_rng = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="intentional forward failure"):
        fv.prism(model, sample_batch=batch, depth="quick")

    assert [module.training for module in model.modules()] == before_modes
    assert torch.equal(model.norm.running_mean, before_mean)
    assert torch.equal(torch.get_rng_state(), before_rng)
    for module in model.modules():
        assert not module._forward_hooks


def test_graph_weight_validation_checks_beyond_first_256_edges():
    nx = pytest.importorskip("networkx")

    graph = nx.MultiGraph()
    for index in range(300):
        graph.add_edge(index % 20, (index + 1) % 20, weight=1.0)
    graph.add_edge(0, 1, weight=-4.0)

    report = fv.prism(graph, depth="quick")
    structure = report.analysis["graph"]

    assert structure["weight_attribute"] is None
    assert structure["shortest_paths"]["weighted"] is False


def test_structured_beacon_confidence_is_not_claimed_as_calibrated():
    from framevitals.core.beacons import beacon

    item = beacon(
        "model.runtime.example",
        "Possible suspicious gradient",
        confidence=0.95,
    )
    assert item["confidence"] == 0.95
    assert item["confidence_calibrated"] is False


def test_structured_health_is_labeled_heuristic():
    import numpy as np

    report = fv.prism(np.eye(4))
    health = report.analysis["health"]

    assert health["score_basis"] == "heuristic"
    assert health["calibrated"] is False
