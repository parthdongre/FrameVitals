"""Seeded fresh-process before/after scientific benchmarks (no optional GPU).

python benchmarks/astra_science.py --before /path/to/sol --after . --output docs/benchmarks/astra-results.json
Timings exclude imports and input construction; CSR conversion is inside measured method.
Peak RSS includes process/import/input memory. Tracemalloc is measured in a separate call.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def worker(repo, kind, seed):
    sys.path.insert(0, str(Path(repo).resolve() / "src"))
    import numpy as np
    import pandas as pd
    import resource
    import tracemalloc

    rng = np.random.default_rng(seed)
    if kind in {"graph", "graph_weighted"}:
        import networkx as nx
        from framevitals.analysis.graph import _sample_shortest_paths

        n = 10000 if kind == "graph" else 20000
        graph = (
            nx.path_graph(n)
            if kind == "graph"
            else nx.fast_gnp_random_graph(n, 0.0002, seed=seed, directed=True)
        )
        weight = None if kind == "graph" else "weight"
        if weight:
            for _, _, data in graph.edges(data=True):
                data["weight"] = float(rng.uniform(0.1, 2.0))
        run = lambda: _sample_shortest_paths(graph, nx, source_count=12, weight=weight)

        def accuracy(r):
            positions = (
                np.sort(np.random.default_rng(42).choice(n, 12, replace=False))
                if "source_sampling" in r
                else np.linspace(0, n - 1, 12, dtype=int)
            )
            if weight:
                distances = [
                    v
                    for source in positions
                    for node, v in nx.single_source_dijkstra_path_length(
                        graph, int(source), weight="weight"
                    ).items()
                    if node != source
                ]
                exact = float(np.mean(distances))
            else:
                exact = sum((i * (i + 1) + (n - 1 - i) * (n - i)) / 2 for i in positions) / (
                    12 * (n - 1)
                )
            return {
                "mean_absolute_error": abs(r["mean_distance"] - exact),
                "method": r.get("method", "networkx_bfs"),
                "quantile_sample_size": r.get("quantile_sample_size", 12 * (n - 1)),
            }
    elif kind == "tensor":
        from framevitals.analysis.tensor import tensor_metrics

        x = rng.normal(size=(1800, 640))
        run = lambda: tensor_metrics(x, sample_values=50000)

        def accuracy(r):
            m = r["matrix"]
            return {
                "stable_rank": m.get("stable_rank"),
                "population": m.get("population", "unspecified"),
                "analyzed_shape": m.get("analyzed_shape"),
            }
    elif kind == "sketch":
        from framevitals.streaming_sketches import NumpyLogQuantileSketch

        x = rng.lognormal(0, 4, 100000)
        run = lambda: NumpyLogQuantileSketch().update(x)

        def accuracy(r):
            errors = [
                abs(r.quantile(q) - np.quantile(x, q, method="lower"))
                / np.quantile(x, q, method="lower")
                for q in [0.01, 0.1, 0.5, 0.9, 0.99]
            ]
            return {"max_relative_error": max(errors), "bins": r.bin_count}
    elif kind == "moments":
        from framevitals.backends import numeric_state

        x = rng.standard_t(6, 100000)
        run = lambda: numeric_state(x, backend="numpy")

        def accuracy(r):
            from scipy import stats

            return {
                "variance_error": abs(r["variance"] - x.var(ddof=1)),
                "skew_error": abs(r["skewness"] - stats.skew(x, bias=False)),
                "kurtosis_error": abs(r["kurtosis"] - stats.kurtosis(x, bias=False)),
            }
    elif kind == "drift":
        from framevitals.drift_analysis import compare_datasets

        a = pd.DataFrame(rng.normal(size=(2000, 24)), columns=[f"x{i}" for i in range(24)])
        b = pd.DataFrame(rng.normal(size=(2000, 24)), columns=a.columns)
        b["x0"] += 1
        run = lambda: compare_datasets(a, b)

        def accuracy(r):
            rows = r["columns"]
            return {
                "shifted_severity": rows[0]["drift_severity"],
                "null_significant": sum(bool(c.get("ks_significant")) for c in rows[1:]),
                "multiplicity": rows[0].get("multiple_testing", "none"),
            }
    elif kind == "anomaly":
        from framevitals.fast_anomaly import fast_anomaly_scan

        x = rng.normal(size=(1000, 8))
        x[-50:] += 6
        frame = pd.DataFrame(x)
        run = lambda: fast_anomaly_scan(frame, top_k=1000, random_state=seed)

        def accuracy(r):
            top = r["top_rows"][:50]
            return {
                "precision_at_50": sum(int(row["row_index"]) >= 950 for row in top) / 50,
                "flagged": r["flagged_count"],
            }
    else:
        raise ValueError(kind)
    t = time.perf_counter()
    result = run()
    elapsed = time.perf_counter() - t
    measured = accuracy(result)
    tracemalloc.start()
    run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "seconds": elapsed,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "python_peak_bytes": peak,
        "seed": seed,
        "accuracy": measured,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before")
    parser.add_argument("--after", default=".")
    parser.add_argument("--output")
    parser.add_argument("--worker")
    parser.add_argument("--repo")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.repo, args.worker, args.seed), allow_nan=False))
        return
    report = {
        "environment": {"python": sys.version, "platform": sys.platform, "blas_threads": 1},
        "repeats": 3,
        "results": {},
    }
    for label, repo in [("before", args.before), ("after", args.after)]:
        report["results"][label] = {}
        for kind in ["graph", "graph_weighted", "tensor", "sketch", "moments", "drift", "anomaly"]:
            runs = []
            for seed in [11, 29, 47]:
                env = {
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "FRAMEVITALS_BACKEND": "numpy",
                }
                r = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        kind,
                        "--repo",
                        str(Path(repo).resolve()),
                        "--seed",
                        str(seed),
                    ],
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=180,
                )
                runs.append(json.loads(r.stdout))
            report["results"][label][kind] = {
                "median_seconds": statistics.median(r["seconds"] for r in runs),
                "max_rss_kib": max(r["peak_rss_kib"] for r in runs),
                "runs": runs,
            }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
