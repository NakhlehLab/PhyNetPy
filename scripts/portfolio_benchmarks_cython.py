"""
Cython-vs-pure-Python A/B and retained-memory diagnostics for PhyNetPy 0.6.0.

Two things the first harness could not measure honestly:

  1. **Cython speedup.** ``phynetpy._mpl`` keeps a pure-Python reference DP
     alongside the Cython triplet engine, gated on the module global
     ``_HAS_CYTHON_MPL`` which is read at call time. Flipping it gives a true
     same-process, same-data A/B of the compiled kernel.

  2. **Retained memory.** Peak tracemalloc during construction also counts the
     builder's own temporary lists, which unfairly inflates PhyNetPy. This
     measures memory still *live* after the builder returns and temporaries are
     collected -- i.e. the real cost of holding the graph.

Run:  .venv/Scripts/python.exe scripts/portfolio_benchmarks_cython.py
Out:  scripts/portfolio_benchmarks_cython.json
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np

OUT_PATH = Path(__file__).with_suffix(".json")
RESULTS: dict[str, Any] = {"meta": {}, "cython_ab": {}, "retained_memory": {}}


def banner(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def timed(fn: Callable[[], Any], loops: int, repeats: int = 5) -> float:
    """Median seconds per call."""
    fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(loops):
            fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) / loops


# ---------------------------------------------------------------------------
# 1. Cython MPL engine vs pure-Python reference DP
# ---------------------------------------------------------------------------
def bench_cython_ab() -> None:
    banner("1. MPL triplet DP: Cython kernel vs pure-Python reference")

    from phynetpy import _mpl
    from phynetpy.criteria import PseudoLikelihood
    from phynetpy.infer import score, simulate
    from phynetpy.models import MSC

    if not _mpl._HAS_CYTHON_MPL:
        print("  Cython MPL engine not available; nothing to compare.")
        RESULTS["cython_ab"]["error"] = "cython mpl unavailable"
        return

    print(f"  {'taxa':>5} {'gene trees':>11} "
          f"{'Cython (ms)':>13} {'Python (ms)':>13} {'speedup':>9} "
          f"{'log-PL match':>14}")
    print("  " + "-" * 70)

    for n_taxa, n_gts in ((6, 40), (8, 60), (10, 80), (12, 100)):
        try:
            gts = simulate(MSC(theta=0.02), taxa=n_taxa, n=n_gts,
                           data="gene_trees", seed=11)
            net = gts.true_network
            crit = PseudoLikelihood()

            def call() -> float:
                return score(net, gts, model=MSC(theta=0.02), criterion=crit)

            # Cython path (default)
            _mpl._HAS_CYTHON_MPL = True
            cy_val = call()
            cy_s = timed(call, loops=5, repeats=3)

            # Pure-Python reference path
            _mpl._HAS_CYTHON_MPL = False
            py_val = call()
            py_s = timed(call, loops=3, repeats=3)

            # restore
            _mpl._HAS_CYTHON_MPL = True

            agree = abs(cy_val - py_val)
            speedup = py_s / cy_s
            print(f"  {n_taxa:>5} {n_gts:>11} "
                  f"{cy_s * 1e3:>13.3f} {py_s * 1e3:>13.3f} "
                  f"{speedup:>8.2f}x {agree:>14.2e}")

            RESULTS["cython_ab"][f"{n_taxa}taxa_{n_gts}gts"] = {
                "taxa": n_taxa,
                "gene_trees": n_gts,
                "cython_ms": cy_s * 1e3,
                "python_ms": py_s * 1e3,
                "speedup": speedup,
                "log_pl_cython": cy_val,
                "log_pl_python": py_val,
                "abs_diff": agree,
            }
        except Exception as exc:
            _mpl._HAS_CYTHON_MPL = True
            print(f"  {n_taxa:>5} {n_gts:>11}   ERR {str(exc)[:44]}")
            RESULTS["cython_ab"][f"{n_taxa}taxa_{n_gts}gts"] = {
                "error": repr(exc)
            }


# ---------------------------------------------------------------------------
# 2. Retained memory, PhyNetPy Network vs NetworkX DiGraph
# ---------------------------------------------------------------------------
def build_phynetpy_tree(num_leaves: int):
    from phynetpy.Network import Edge, Network, Node

    net = Network()
    leaves = [Node(f"leaf_{i}") for i in range(num_leaves)]
    all_nodes = list(leaves)
    internal = 0
    level = list(leaves)
    pairs = []
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            parent = Node(f"internal_{internal}")
            internal += 1
            all_nodes.append(parent)
            pairs.append((parent, level[i], level[i + 1]))
            nxt.append(parent)
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    net.add_nodes(all_nodes)
    for parent, a, b in pairs:
        net.add_edges(Edge(parent, a))
        net.add_edges(Edge(parent, b))
    del leaves, all_nodes, level, pairs
    return net


def build_networkx_tree(num_leaves: int):
    import networkx as nx

    g = nx.DiGraph()
    leaves = [f"leaf_{i}" for i in range(num_leaves)]
    g.add_nodes_from(leaves)
    internal = 0
    level = list(leaves)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            parent = f"internal_{internal}"
            internal += 1
            g.add_node(parent)
            g.add_edge(parent, level[i])
            g.add_edge(parent, level[i + 1])
            nxt.append(parent)
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    del leaves, level
    return g


def retained_kib(builder: Callable[[], Any]) -> tuple[Any, float, float]:
    """Return (obj, retained KiB, peak KiB) for one construction."""
    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    obj = builder()
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return obj, current / 1024.0, peak / 1024.0


def bench_retained_memory() -> None:
    banner("2. Retained memory after construction (temporaries collected)")
    print(f"  {'leaves':>7} {'nodes':>7} {'edges':>7} "
          f"{'PhyNetPy':>12} {'NetworkX':>12} {'phy/nx':>8} "
          f"{'bytes/node':>11}")
    print("  " + "-" * 70)

    for num_leaves in (100, 500, 1000, 2000):
        net, phy_kib, phy_peak = retained_kib(
            lambda n=num_leaves: build_phynetpy_tree(n)
        )
        g, nx_kib, nx_peak = retained_kib(
            lambda n=num_leaves: build_networkx_tree(n)
        )
        n_nodes, n_edges = len(net.V()), len(net.E())
        ratio = phy_kib / nx_kib if nx_kib else float("nan")
        per_node = phy_kib * 1024 / n_nodes

        print(f"  {num_leaves:>7} {n_nodes:>7} {n_edges:>7} "
              f"{phy_kib:>9.1f} KiB {nx_kib:>8.1f} KiB "
              f"{ratio:>7.2f}x {per_node:>10.0f} B")

        RESULTS["retained_memory"][f"{num_leaves}_leaves"] = {
            "leaves": num_leaves,
            "nodes": n_nodes,
            "edges": n_edges,
            "phynetpy_retained_kib": phy_kib,
            "networkx_retained_kib": nx_kib,
            "phynetpy_peak_kib": phy_peak,
            "networkx_peak_kib": nx_peak,
            "ratio_phy_over_nx": ratio,
            "phynetpy_bytes_per_node": per_node,
        }
        del net, g
        gc.collect()


def main() -> None:
    import phynetpy

    RESULTS["meta"] = {
        "phynetpy_version": phynetpy.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("PhyNetPy Cython A/B + memory diagnostics")
    for k, v in RESULTS["meta"].items():
        print(f"  {k:20}{v}")

    bench_cython_ab()
    bench_retained_memory()

    OUT_PATH.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"\n\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
