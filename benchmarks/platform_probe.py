"""Small platform overhead probe. Run each mode in a fresh process for fair comparisons."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc

parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=["import", "tensor", "tabular", "graph"], default="import")
args = parser.parse_args()
tracemalloc.start()
start = time.perf_counter()
import framevitals as fv

if args.kind == "tensor":
    import numpy as np

    result = fv.prism(np.arange(48.0).reshape(8, 6), depth="quick", artifacts=False)
elif args.kind == "tabular":
    import pandas as pd

    result = fv.prism(
        pd.DataFrame({"a": range(100), "b": [i % 7 for i in range(100)]}),
        depth="quick",
        artifacts=False,
    )
elif args.kind == "graph":
    import networkx as nx

    result = fv.prism(nx.path_graph(50), depth="quick", artifacts=False)
print(
    json.dumps(
        {
            "kind": args.kind,
            "elapsed_ms": (time.perf_counter() - start) * 1000,
            "python_peak_bytes": tracemalloc.get_traced_memory()[1],
            "note": "Python allocations only; excludes native buffers and is not a scientific algorithm benchmark.",
        }
    )
)
