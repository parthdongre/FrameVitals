# FrameVitals protocols

FrameVitals exposes a small protocol layer for everyday use. The protocols are
workflow orchestrators: they coordinate the stable lower-level APIs so users do
not have to assemble a multi-step analysis by hand.

The implementation remains open and inspectable. This page documents the
composition deliberately; routine CLI and web surfaces keep that machinery out
of the way.

## Prism

**Intent:** build the broadest coherent view of a dataset.

```python
import framevitals as fv

result = fv.prism(current)
```

When only `data` is provided, Prism first recognizes the source kind.
Tabular input delegates to the mature source-aware analysis dispatcher, while
supported structured inputs dispatch to bounded graph, tensor, or model
diagnostic engines.

```python
fv.prism(dataframe)
fv.prism(networkx_graph)
fv.prism(numpy_tensor)
fv.prism(pytorch_model)
```

All of these return the same protocol-level `PrismResult`, with modality-specific
details available under `result.analysis`.

When a trusted tabular reference is also supplied:

```python
result = fv.prism(
    current,
    reference=training,
    focus="churn",
)
```

Prism coordinates:

1. canonical FrameVitals analysis of the current dataset;
2. Axiom derivation from the reference when no explicit contract is supplied;
3. exact validation of the current dataset against that Axiom;
4. Tide comparison between the reference and current states;
5. one combined verdict over the requested trust checks.

The result keeps these components available for developers:

```python
result.analysis
result.contract
result.validation
result.tide
result.verdict
result.beacons
```

The protocol does not create a second implementation of these operations. It is
an orchestration layer over the same maintained engines used by the focused API.

## Axiom

**Intent:** establish what should remain true and optionally test a current
dataset against those expectations.

```python
axiom = fv.axiom(reference)
checked = fv.axiom(reference, current=current)
```

Axiom composes contract inference and exact contract validation. The inferred
expectation object remains JSON-serializable and can be stored or passed back to
Prism explicitly.

## Forge

**Intent:** prepare a conservative transformation plan, then apply it only when
requested.

```python
plan = fv.forge(data)
applied = fv.forge(data, apply=True)
cleaned = applied.cleaned
```

Forge composes the existing cleaning-plan inference and cleaning execution
paths. It does not mutate the caller's original input.

## Tide

**Intent:** understand how a dataset changed between two states.

```python
change = fv.tide(reference, current)
```

Tide is the protocol-facing entry point for FrameVitals' bounded, source-aware
dataset comparison path.

## Pulse

**Intent:** capture compact health state that can be retained over time.

```python
snapshot = fv.pulse(data, destination="pulse.json")
```

Pulse analyzes raw input when necessary and emits the same versioned compact
snapshot format used by FrameVitals monitoring history.

## Lower-level APIs

The protocol layer is the recommended starting point, but it does not remove
the focused APIs. Operations such as `analyze`, `compare`, `infer_contract`,
`validate`, `gate`, `profile`, `statistics`, and `anomalies` remain
available for advanced use, testing, integrations, and backward compatibility.

This separation is deliberate:

- **protocols** describe user intent;
- **focused APIs** expose individual analytical operations;
- **internal modules** contain the maintained implementations.

Product-facing CLI and web interfaces should use protocol terminology and
outcomes. Implementation details belong in documentation, debug output, and
source code rather than the default user experience.
