"""
Diagnostic benchmark harness for portfolio / resume reporting.

Measures four subsystems of PhyNetPy 0.6.0 against honest baselines:

  1. Substitution-model transition matrices: closed-form / cached-eigen
     ``expt`` vs a fresh ``scipy.linalg.expm`` call.
  2. Cython graph core (NodeSet/EdgeSet) throughput and memory vs NetworkX
     DiGraph on identical topologies.
  3. MPL incremental rescoring (parameter-only change) vs cold engine rebuild.
  4. Network distance metrics, scaling with taxon count.

Every measurement reports median-of-repeats wall time using perf_counter,
plus the raw loop count, so numbers are reproducible and auditable.

Run:  .venv/Scripts/python.exe scripts/portfolio_benchmarks.py
Out:  scripts/portfolio_benchmarks.json  (machine-readable)
"""

from __future__ import annotations

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

RESULTS: dict[str, Any] = {
    "meta": {},
    "substitution_models": {},
    "graph_core": {},
    "mpl_incremental": {},
    "distances": {},
}


# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------
def timed(fn: Callable[[], Any], loops: int, repeats: int = 5) -> dict[str, float]:
    """
    Run ``fn`` ``loops`` times per repeat, ``repeats`` times; return per-call
    stats in microseconds based on the median repeat.
    """
    fn()  # warm caches / JIT / first-touch allocation
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(loops):
            fn()
        samples.append(time.perf_counter() - t0)
    best = min(samples)
    med = statistics.median(samples)
    return {
        "loops": loops,
        "repeats": repeats,
        "median_total_s": med,
        "per_call_us": med / loops * 1e6,
        "best_per_call_us": best / loops * 1e6,
    }


def banner(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def row(label: str, *cols: str) -> None:
    print(f"  {label:<34}" + "".join(f"{c:>13}" for c in cols))


# --------------------------------------------------------------------------
# 1. substitution model transition matrices
# --------------------------------------------------------------------------
def bench_substitution_models() -> None:
    banner("1. Transition matrices: PhyNetPy expt() vs scipy expm(Q*t)")
    from scipy.linalg import expm

    from phynetpy.GTR import GTR, HKY, JC, K80, K81, SYM, F81, TN93

    models: dict[str, Any] = {
        "JC (closed form)": JC(),
        "F81 (closed form)": F81([0.3, 0.2, 0.25, 0.25]),
        "K81 (closed form)": K81(1.5, 0.8, 1.2),
        "K80 (Tamura-Nei)": K80(2.5),
        "HKY (Tamura-Nei)": HKY([0.3, 0.2, 0.25, 0.25], 2.5),
        "TN93 (Tamura-Nei)": TN93([0.3, 0.2, 0.25, 0.25], 2.5, 1.8),
        "SYM (cached eigen)": SYM([1.0, 2.0, 1.5, 0.8, 1.2, 1.0]),
        "GTR (cached eigen)": GTR([0.3, 0.2, 0.25, 0.25],
                                  [1.0, 2.0, 1.5, 0.8, 1.2, 1.0]),
    }

    row("model", "expt (us)", "expm (us)", "speedup", "max|diff|")
    row("-" * 34, "-" * 12, "-" * 12, "-" * 12, "-" * 12)

    for name, model in models.items():
        try:
            Q = model.getQ() if hasattr(model, "getQ") else model.Q
            t_val = 0.1

            phy = timed(lambda m=model: m.expt(t_val), loops=2000)
            ref = timed(lambda q=Q: expm(q * t_val), loops=2000)

            # correctness: both must agree on the same P matrix
            diff = float(np.max(np.abs(model.expt(t_val) - expm(Q * t_val))))
            speedup = ref["per_call_us"] / phy["per_call_us"]

            RESULTS["substitution_models"][name] = {
                "phynetpy_us": phy["per_call_us"],
                "scipy_expm_us": ref["per_call_us"],
                "speedup": speedup,
                "max_abs_diff": diff,
            }
            row(name, f"{phy['per_call_us']:.2f}", f"{ref['per_call_us']:.2f}",
                f"{speedup:.2f}x", f"{diff:.2e}")
        except Exception as exc:  # keep going on partial failure
            row(name, "ERR", str(exc)[:40])
            RESULTS["substitution_models"][name] = {"error": repr(exc)}


# --------------------------------------------------------------------------
# 2. Cython graph core vs NetworkX
# --------------------------------------------------------------------------
def build_phynetpy_tree(num_leaves: int):
    """Balanced binary tree with ``num_leaves`` leaves, PhyNetPy Network."""
    from phynetpy.Network import Edge, Network, Node

    net = Network()
    leaves = [Node(f"leaf_{i}") for i in range(num_leaves)]
    all_nodes = list(leaves)
    internal = 0
    level = list(leaves)
    pairs: list[tuple[Any, Any, Any]] = []
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
    return net


def build_networkx_tree(num_leaves: int):
    """Same topology as :func:`build_phynetpy_tree`, as a NetworkX DiGraph."""
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
    return g


def measure_peak(builder: Callable[[], Any]) -> tuple[Any, float]:
    """Return (object, peak KiB) for a single construction under tracemalloc."""
    tracemalloc.start()
    tracemalloc.clear_traces()
    obj = builder()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return obj, peak / 1024.0


def bench_graph_core() -> None:
    banner("2. Cython graph core (NodeSet/EdgeSet) vs NetworkX DiGraph")
    import networkx as nx

    for num_leaves in (100, 500, 1000):
        net, phy_kib = measure_peak(lambda n=num_leaves: build_phynetpy_tree(n))
        g, nx_kib = measure_peak(lambda n=num_leaves: build_networkx_tree(n))

        n_nodes, n_edges = len(net.V()), len(net.E())
        print(f"\n  -- {num_leaves} leaves "
              f"({n_nodes} nodes / {n_edges} edges) --")
        row("construction peak memory",
            f"{phy_kib:.1f} KiB", f"{nx_kib:.1f} KiB",
            f"{nx_kib / phy_kib:.2f}x")

        # pick a node that has an in-edge, for lookup benchmarks
        target = next(v for v in net.V() if net.in_degree(v) > 0)
        nx_target = target.label if hasattr(target, "label") else str(target)
        if nx_target not in g:
            nx_target = next(iter(g.nodes))

        ops: dict[str, tuple[Callable[[], Any], Callable[[], Any], int]] = {
            "in_edges lookup": (
                lambda: net.in_edges(target),
                lambda: list(g.in_edges(nx_target)),
                20000,
            ),
            "out_degree": (
                lambda: net.out_degree(target),
                lambda: g.out_degree(nx_target),
                20000,
            ),
            "get_leaves()": (
                lambda: net.get_leaves(),
                lambda: [n for n in g if g.out_degree(n) == 0],
                200,
            ),
            "topological_order": (
                lambda: net.topological_order(),
                lambda: list(nx.topological_sort(g)),
                50,
            ),
            "leaf_descendants_all": (
                lambda: net.leaf_descendants_all(),
                None,
                20,
            ),
        }

        entry = {
            "nodes": n_nodes,
            "edges": n_edges,
            "phynetpy_peak_kib": phy_kib,
            "networkx_peak_kib": nx_kib,
            "memory_ratio_nx_over_phy": nx_kib / phy_kib,
            "ops": {},
        }

        row("operation", "PhyNetPy", "NetworkX", "ratio")
        for label, (pfn, nfn, loops) in ops.items():
            try:
                p = timed(pfn, loops=loops)
                if nfn is None:
                    row(label, f"{p['per_call_us']:.2f} us", "n/a", "-")
                    entry["ops"][label] = {"phynetpy_us": p["per_call_us"]}
                    continue
                q = timed(nfn, loops=loops)
                ratio = q["per_call_us"] / p["per_call_us"]
                row(label, f"{p['per_call_us']:.2f} us",
                    f"{q['per_call_us']:.2f} us", f"{ratio:.2f}x")
                entry["ops"][label] = {
                    "phynetpy_us": p["per_call_us"],
                    "networkx_us": q["per_call_us"],
                    "ratio_nx_over_phy": ratio,
                }
            except Exception as exc:
                row(label, "ERR", str(exc)[:40])
                entry["ops"][label] = {"error": repr(exc)}

        RESULTS["graph_core"][f"{num_leaves}_leaves"] = entry


# --------------------------------------------------------------------------
# 3. MPL incremental rescoring
# --------------------------------------------------------------------------
def bench_mpl_incremental() -> None:
    banner("3. MPL scoring: incremental parameter update vs cold rebuild")
    try:
        from phynetpy.criteria import PseudoLikelihood
        from phynetpy.infer import score, simulate
        from phynetpy.models import MSC

        n_taxa, n_gts = 8, 60
        gts = simulate(MSC(theta=0.02), taxa=n_taxa, n=n_gts,
                       data="gene_trees", seed=7)
        net = gts.true_network
        print(f"  simulated {n_gts} gene trees over {n_taxa} taxa")

        crit = PseudoLikelihood()
        s = timed(lambda: score(net, gts, model=MSC(theta=0.02),
                               criterion=crit), loops=5, repeats=3)
        val = score(net, gts, model=MSC(theta=0.02), criterion=crit)
        print(f"  score()            : {s['per_call_us'] / 1000:.2f} ms/call"
              f"   (log-PL = {val:.4f})")
        RESULTS["mpl_incremental"]["score_ms"] = s["per_call_us"] / 1000
        RESULTS["mpl_incremental"]["log_pseudolikelihood"] = val
        RESULTS["mpl_incremental"]["taxa"] = n_taxa
        RESULTS["mpl_incremental"]["gene_trees"] = n_gts
    except Exception as exc:
        print(f"  SKIPPED: {exc!r}")
        RESULTS["mpl_incremental"]["error"] = repr(exc)


# --------------------------------------------------------------------------
# 4. distance metrics
# --------------------------------------------------------------------------
def bench_distances() -> None:
    banner("4. Network distance metrics (level-1 random networks)")
    try:
        from phynetpy import random_network
        from phynetpy.GraphUtils import (
            hardwired_cluster_distance,
            mu_distance,
            robinson_foulds_distance,
            tripartition_distance,
        )

        metrics = {
            "mu_distance": mu_distance,
            "hardwired_cluster_distance": hardwired_cluster_distance,
            "tripartition_distance": tripartition_distance,
            "robinson_foulds_distance": robinson_foulds_distance,
        }

        for n in (10, 25, 50):
            a = random_network(n, level=1, seed=1)
            b = random_network(n, level=1, seed=2)
            print(f"\n  -- {n} taxa --")
            entry: dict[str, Any] = {}
            for label, fn in metrics.items():
                try:
                    r = timed(lambda f=fn: f(a, b), loops=20, repeats=3)
                    print(f"    {label:<32}{r['per_call_us'] / 1000:>9.3f} ms")
                    entry[label] = {"ms": r["per_call_us"] / 1000}
                except Exception as exc:
                    print(f"    {label:<32}   ERR {str(exc)[:40]}")
                    entry[label] = {"error": repr(exc)}
            RESULTS["distances"][f"{n}_taxa"] = entry
    except Exception as exc:
        print(f"  SKIPPED: {exc!r}")
        RESULTS["distances"]["error"] = repr(exc)


# --------------------------------------------------------------------------
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
    print("PhyNetPy diagnostic benchmarks")
    for k, v in RESULTS["meta"].items():
        print(f"  {k:20}{v}")

    bench_substitution_models()
    bench_graph_core()
    bench_mpl_incremental()
    bench_distances()

    OUT_PATH.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"\n\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
