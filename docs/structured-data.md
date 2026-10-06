# Structured data and model diagnostics

FrameVitals is evolving from a tabular-only analysis toolkit into a diagnostic
layer for structured data and ML systems. The public contract stays small:
`fv.prism(...)` recognizes the source and dispatches to a specialized bounded
engine.

## Source recognition

The first structured source kinds are:

| Kind | Current adapters | Status |
| --- | --- | --- |
| Tabular | pandas, files, Arrow, DuckDB | mature |
| Graph | NetworkX-compatible graphs | initial |
| Tensor | NumPy arrays, PyTorch tensors | initial |
| Model | PyTorch-style modules | initial |
| Nested | dict/list/tuple recognition | recognized; analyzer pending |

Recognition never imports PyTorch or NetworkX just to inspect an object. Optional
frameworks remain optional dependencies.

## Graph Prism

Install graph support with:

```bash
pip install "framevitals[graph]"
```

Then:

```python
import framevitals as fv

report = fv.prism(graph, depth="standard")
```

The graph engine is adaptive and resource bounded. Depending on graph size and
Prism depth it can use:

- connected / weakly connected / strongly connected components;
- isolate and self-loop inspection;
- robust degree-hub detection;
- PageRank;
- approximate betweenness centrality;
- Louvain community detection when available, with greedy modularity fallback;
- bridges and articulation points on the undirected structural projection;
- clustering coefficient and transitivity;
- sampled shortest paths;
- Dijkstra shortest paths automatically when a reliable non-negative numeric
  `weight` edge attribute is detected.

The engine intentionally avoids unconditional all-pairs shortest paths or other
work that can make large graphs unusable. Each depth has explicit edge/source
budgets.

## Tensor Prism

NumPy arrays work with the base install:

```python
import numpy as np
import framevitals as fv

matrix = np.random.randn(1024, 128)
report = fv.prism(matrix)
```

Tensor diagnostics include:

- shape, dtype, size, and memory;
- bounded distribution statistics;
- NaN / Inf detection;
- exact-zero and near-zero sparsity;
- L1/L2 magnitude summaries;
- bounded SVD-based rank and effective-rank diagnostics for matrices;
- condition-number diagnostics where the matrix is small enough for safe exact
  decomposition.

PyTorch tensors are also recognized when PyTorch already exists in the caller's
environment.

## PyTorch model Prism

FrameVitals does not install PyTorch as a dependency. If a caller already has a
PyTorch model, Prism can inspect it directly:

```python
report = fv.prism(model, depth="standard")
```

The first model engine includes:

- total/trainable/frozen parameter counts;
- module-type inventory and broad architecture recognition;
- dtype and memory distribution;
- bounded per-tensor weight statistics;
- NaN / Inf parameter checks;
- extreme norm detection;
- bounded matrix-rank diagnostics;
- existing-gradient inspection when a backward pass has already populated
  `.grad`;
- CNN-specific convolution filter analysis;
- dead / near-zero convolution filter detection.

This is inspection, not AutoML and not a replacement for PyTorch training.

### Runtime activation and gradient diagnostics

A caller can optionally give Prism a representative batch:

```python
report = fv.prism(
    model,
    sample_batch=x,
)
```

Prism temporarily installs hooks on a bounded sample of leaf modules, runs one
forward pass, captures activation statistics, and removes every hook in a
`finally` block.

For backward-flow diagnostics:

```python
report = fv.prism(
    model,
    sample_batch=x,
    targets=y,
    loss_fn=criterion,
    backward=True,
)
```

The runtime observer can surface:

- dead or collapsed ReLU/module activations;
- sigmoid/tanh saturation;
- NaN/Inf activations;
- robust activation-norm explosions or vanishing regions;
- module gradient-flow explosions/vanishing;
- non-finite gradients;
- trainable parameters receiving effectively zero gradients.

Backward observation uses `torch.autograd.grad` rather than `loss.backward()`.
FrameVitals therefore does not accumulate, zero, or replace the caller's
existing parameter `.grad` buffers.

Runtime work is bounded by Prism depth and `max_runtime_modules=`.

## Common result contract

Structured engines emit the existing `AnalysisResult` shape and common
FrameVitals findings/Beacons. This means protocol consumers can continue to use:

```python
result = fv.prism(source)

result.analysis
result.analysis.health
result.beacons
result.summary_text()
```

Each analysis includes `source_kind` so reporting layers can render
modality-specific summaries without leaking algorithm choreography into the
normal product UI.

## Safetensors checkpoint files

Local Safetensors checkpoints are recognized as model sources without requiring
PyTorch or the safetensors Python package:

```python
report = fv.prism("model.safetensors")
```

FrameVitals reads and validates the bounded Safetensors JSON header, then reports
tensor names, shapes, dtypes, parameter counts, checkpoint size, architecture
hints, precision mix, and largest tensors without loading tensor payloads.

Metadata-only checkpoint structure can also be compared:

```python
change = fv.tide(
    "checkpoint-1000.safetensors",
    "checkpoint-5000.safetensors",
)
```

This first file-level Tide detects added/removed tensors, shape changes, dtype
changes, and parameter-count movement. Numeric weight movement remains available
when comparing two in-memory PyTorch models; direct body-level Safetensors
comparison will be added as a deeper optional path.

## Structured Tide

Tide now dispatches by source kind as well:

```python
tensor_change = fv.tide(weights_v1, weights_v2)
graph_change = fv.tide(graph_january, graph_february)
model_change = fv.tide(checkpoint_model_a, checkpoint_model_b)
```

Tensor Tide reports aligned relative-L2 movement, cosine similarity,
Wasserstein distance, dtype/shape changes, and bounded rank changes.

Graph Tide reports node churn, node/edge count changes, density change, and
Jensen-Shannon distance between degree distributions.

Model Tide aligns parameters by name and reports additions/removals, shape and
dtype changes, relative-L2 movement, cosine similarity, Wasserstein movement,
and the most changed parameter tensors.

## Current limitations

Reference-aware Axiom orchestration is still tabular in this adapter release.
Passing a tabular Axiom/reference into graph/tensor/model Prism raises explicitly
rather than silently applying tabular semantics. Structured Tide requires both
inputs to be the same source kind.

The next planned structured phases are:

1. ONNX graph + initializer analysis;
2. deeper optional Safetensors tensor-body comparison;
3. persistent Pulse histories for training/model state;
4. nested JSON/Arrow-struct diagnostics;
5. relational multi-table sources.


## ONNX model files

Install the optional adapter with:

```bash
pip install "framevitals[onnx]"
```

Then:

```python
report = fv.prism("model.onnx")
change = fv.tide("before.onnx", "after.onnx")
```

ONNX Prism combines graph and model diagnostics. It inspects operator topology,
declared inputs/outputs, graph connectivity, initializer tensors, operator
counts, parameter counts, dtype distribution, unused initializers, dead-end
operators, duplicate node names, disconnected graph regions, and basic
architecture hints.

ONNX Tide compares graph size, operator mix, initializer additions/removals,
shape changes, dtype changes, and parameter-count movement.

## Nested JSON and struct-like data

Python dictionaries, lists, tuples, and JSON-like structures are recognized as
nested sources:

```python
report = fv.prism(payload)
change = fv.tide(payload_v1, payload_v2)
```

The nested engine uses bounded traversal to inspect nesting depth, array
lengths, null density, object shapes, top keys, cyclic Python references, and
field paths whose non-null values disagree on type.

Long arrays use wildcard paths so repeated records are compared structurally
instead of treating every array index as a different schema position.

## Relational / multi-table projects

Mappings of named tabular objects are recognized as relational projects:

```python
project = {
    "customers": customers,
    "orders": orders,
    "products": products,
}

report = fv.prism(project)
change = fv.tide(project_v1, project_v2)
snapshot = fv.pulse(project)
```

The relational engine inspects:

- table shape, dtypes, missingness, duplicate rows, and unique-key candidates;
- shared columns between tables;
- inferred one-to-one, one-to-many, many-to-one, and many-to-many relationships;
- child-key/reference coverage and potential orphan entities;
- sampled join-expansion risk;
- ID-like columns that fail to behave as unique keys;
- isolated tables in the inferred schema graph.

Relational Tide compares tables, columns, row counts, and inferred relationship
structure across project versions.

## Optimizer diagnostics

For in-memory PyTorch models, Prism and Pulse can also inspect an optimizer:

```python
report = fv.prism(
    model,
    optimizer=optimizer,
    sample_batch=x,
    targets=y,
    loss_fn=criterion,
    backward=True,
)

snapshot = fv.pulse(model, optimizer=optimizer)
```

Optimizer diagnostics cover parameter-group membership, learning-rate policy,
duplicate assignments, trainable parameters that are missing from the
optimizer, non-trainable parameters assigned to it, optimizer-state memory,
step spread, and sampled NaN/Inf state tensors.

## Structured Pulse

Pulse now snapshots all implemented structured source kinds, not only tables:

```python
fv.pulse(graph)
fv.pulse(tensor)
fv.pulse(model)
fv.pulse(model, optimizer=optimizer)
fv.pulse(nested_payload)
fv.pulse(relational_project)
fv.pulse("model.safetensors")
fv.pulse("model.onnx")
```

Snapshots store compact modality-specific state rather than raw data or model
weights. Existing SnapshotHistory can therefore track health, Beacons, model
runtime summaries, optimizer configuration, graph size/connectivity, tensor
rank/sparsity, nested structural health, and relational schema state over time.


## Structured Axiom

Axiom now works across implemented structured sources:

```python
graph_contract = fv.axiom(reference_graph)
graph_check = fv.axiom(reference_graph, current=current_graph)

tensor_check = fv.axiom(reference_tensor, current=current_tensor)
model_check = fv.axiom(reference_model, current=current_model)
nested_check = fv.axiom(reference_payload, current=current_payload)
relational_check = fv.axiom(reference_project, current=current_project)
```

Structured contracts remain modality-specific:

- graph contracts capture directionality, multigraph semantics, bounded node
  counts, isolate tolerance, and minimum largest-component coverage;
- tensor contracts capture shape, dtype, finite-value expectations, and bounded
  rank floors;
- model contracts capture source format/framework and parameter names,
  shapes, and dtypes;
- nested contracts capture required top-level keys, maximum depth, type-conflict
  tolerance, and cycle policy;
- relational contracts capture required tables/columns, preserved key
  candidates, and inferred relationship cardinalities.

Axiom validates structure rather than trying to impose tabular-column semantics
on non-tabular sources.
