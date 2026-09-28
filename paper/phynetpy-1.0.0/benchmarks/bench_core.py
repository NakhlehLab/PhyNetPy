"""
Core numerical and data-structure benchmarks for the PhyNetPy paper.

Four subsystems, each against a baseline that is available to any reader
rather than against an earlier version of PhyNetPy itself:

1. Substitution-model transition matrices: the closed-form and
   cached-eigendecomposition ``expt`` implementations against a fresh
   ``scipy.linalg.expm`` call.  Reports both speed and maximum absolute
   disagreement, so the comparison doubles as a correctness check.
2. The compiled graph core against NetworkX ``DiGraph`` on identical
   topologies -- construction memory and five operations.  NetworkX is the
   right baseline because it is what a Python phylogenetics developer would
   otherwise reach for.
3. Pseudo-likelihood scoring throughput on simulated data.
4. Network dissimilarity measures, scaling with taxon count.

Run: python paper/phynetpy-1.0.0/benchmarks/bench_core.py
"""

from __future__ import annotations

import tracemalloc
from typing import Any, Callable

import numpy as np

from common import banner, environment, row, timed, write

RESULTS: dict[str, Any] = {
    "meta": {},
    "substitution_models": {},
    "graph_core": {},
    "pseudolikelihood": {},
    "distances": {},
}


# ---------------------------------------------------------------------------
# 1. Substitution-model transition matrices
# ---------------------------------------------------------------------------
def bench_substitution_models() -> None:
    banner("1. Transition matrices: PhyNetPy expt() vs scipy expm(Q*t)")
    from scipy.linalg import expm

    from phynetpy.GTR import GTR, HKY, JC, K80, K81, SYM, F81, TN93

    pi = [0.3, 0.2, 0.25, 0.25]
    rates = [1.0, 2.0, 1.5, 0.8, 1.2, 1.0]
    models: dict[str, tuple[Any, str]] = {
        "JC": (JC(), "closed form"),
        "F81": (F81(pi), "closed form"),
        "K81": (K81(1.5, 0.8, 1.2), "closed form"),
        "K80": (K80(2.5), "Tamura-Nei"),
        "HKY": (HKY(pi, 2.5), "Tamura-Nei"),
        "TN93": (TN93(pi, 2.5, 1.8), "Tamura-Nei"),
        "SYM": (SYM(rates), "cached eigendecomposition"),
        "GTR": (GTR(pi, rates), "cached eigendecomposition"),
    }

    t_val = 0.1
    row("model", "expt (us)", "expm (us)", "speedup", "max|diff|")
    row("-" * 34, "-" * 12, "-" * 12, "-" * 12, "-" * 12)

    for name, (model, kind) in models.items():
        Q = model.getQ()
        phy = timed(lambda m=model: m.expt(t_val), loops=2000)
        ref = timed(lambda q=Q: expm(q * t_val), loops=2000)
        diff = float(np.max(np.abs(model.expt(t_val) - expm(Q * t_val))))

        RESULTS["substitution_models"][name] = {
            "kind": kind,
            "phynetpy_us": phy["per_call_us"],
            "scipy_expm_us": ref["per_call_us"],
            "speedup": ref["per_call_us"] / phy["per_call_us"],
            "max_abs_diff": diff,
            "protocol": {"phynetpy": phy, "scipy": ref},
        }
        row(
            f"{name} ({kind})",
            f"{phy['per_call_us']:.2f}",
            f"{ref['per_call_us']:.2f}",
            f"{ref['per_call_us'] / phy['per_call_us']:.1f}x",
            f"{diff:.1e}",
        )


# ---------------------------------------------------------------------------
# 2. Compiled graph core vs NetworkX
# ---------------------------------------------------------------------------
def build_phynetpy_tree(num_leaves: int):
    """Balanced binary tree with ``num_leaves`` leaves as a PhyNetPy Network."""
    from phynetpy.Network import Edge, Network, Node

    net = Network()
    leaves = [Node(f"leaf_{i}") for i in range(num_leaves)]
    all_nodes = list(leaves)
    pairs: list[tuple[Any, Any, Any]] = []
    internal = 0
    level = list(leaves)
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
    """The same topology as :func:`build_phynetpy_tree` as a NetworkX DiGraph."""
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
    """Return ``(object, peak KiB)`` for one construction under tracemalloc."""
    tracemalloc.start()
    tracemalloc.clear_traces()
    obj = builder()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return obj, peak / 1024.0


def bench_graph_core() -> None:
    banner("2. Compiled graph core vs NetworkX DiGraph (identical topologies)")
    import networkx as nx

    for num_leaves in (100, 500, 1000):
        net, phy_kib = measure_peak(lambda n=num_leaves: build_phynetpy_tree(n))
        g, nx_kib = measure_peak(lambda n=num_leaves: build_networkx_tree(n))

        n_nodes, n_edges = len(net.V()), len(net.E())
        print(
            f"\n  -- {num_leaves} leaves "
            f"({n_nodes} nodes / {n_edges} edges) --"
        )
        row(
            "construction peak memory",
            f"{phy_kib:.0f} KiB",
            f"{nx_kib:.0f} KiB",
            f"{nx_kib / phy_kib:.2f}x",
        )

        target = next(v for v in net.V() if net.in_degree(v) > 0)
        nx_target = target.label
        if nx_target not in g:
            nx_target = next(iter(g.nodes))

        ops: dict[str, tuple[Callable[[], Any], Callable[[], Any] | None, int]] = {
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
            "get_leaves": (
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

        entry: dict[str, Any] = {
            "nodes": n_nodes,
            "edges": n_edges,
            "phynetpy_peak_kib": phy_kib,
            "networkx_peak_kib": nx_kib,
            "memory_ratio_nx_over_phy": nx_kib / phy_kib,
            "ops": {},
        }

        row("operation", "PhyNetPy", "NetworkX", "ratio")
        for label, (pfn, nfn, loops) in ops.items():
            p = timed(pfn, loops=loops)
            if nfn is None:
                row(label, f"{p['per_call_us']:.2f} us", "n/a", "-")
                entry["ops"][label] = {"phynetpy_us": p["per_call_us"]}
                continue
            q = timed(nfn, loops=loops)
            ratio = q["per_call_us"] / p["per_call_us"]
            row(
                label,
                f"{p['per_call_us']:.2f} us",
                f"{q['per_call_us']:.2f} us",
                f"{ratio:.2f}x",
            )
            entry["ops"][label] = {
                "phynetpy_us": p["per_call_us"],
                "networkx_us": q["per_call_us"],
                "ratio_nx_over_phy": ratio,
            }

        RESULTS["graph_core"][f"{num_leaves}_leaves"] = entry


# ---------------------------------------------------------------------------
# 3. Pseudo-likelihood scoring throughput
# ---------------------------------------------------------------------------
def bench_pseudolikelihood() -> None:
    banner("3. Pseudo-likelihood scoring throughput")
    from phynetpy.criteria import PseudoLikelihood
    from phynetpy.infer import score, simulate
    from phynetpy.models import MSC

    # theta = 0.5 leaves roughly half the gene trees discordant with the
    # species topology. At the simulator's small default scale the gene trees
    # come out identical to the species tree, which is not the regime the
    # triplet dynamic program is built for.
    theta = 0.5
    crit = PseudoLikelihood()
    for n_taxa, n_gts in ((6, 50), (8, 100), (10, 200)):
        gts = simulate(
            MSC(theta=theta), taxa=n_taxa, n=n_gts, data="gene_trees", seed=7
        )
        net = gts.true_network
        stats = timed(
            lambda: score(net, gts, model=MSC(theta=theta), criterion=crit),
            loops=5,
            repeats=3,
        )
        value = score(net, gts, model=MSC(theta=theta), criterion=crit)
        key = f"{n_taxa}taxa_{n_gts}genes"
        RESULTS["pseudolikelihood"][key] = {
            "taxa": n_taxa,
            "gene_trees": n_gts,
            "theta": theta,
            "ms_per_score": stats["per_call_us"] / 1000,
            "log_pseudolikelihood": value,
            "seed": 7,
            "protocol": stats,
        }
        print(
            f"  {n_taxa:>3} taxa / {n_gts:>4} gene trees : "
            f"{stats['per_call_us'] / 1000:8.2f} ms   (log-PL = {value:.4f})"
        )


# ---------------------------------------------------------------------------
# 4. Network dissimilarity measures
# ---------------------------------------------------------------------------
def bench_distances() -> None:
    banner("4. Network dissimilarity measures (level-1 random networks)")
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
            stats = timed(lambda f=fn: f(a, b), loops=20, repeats=3)
            print(f"    {label:<32}{stats['per_call_us'] / 1000:>9.3f} ms")
            entry[label] = {
                "ms": stats["per_call_us"] / 1000,
                "protocol": stats,
            }
        RESULTS["distances"][f"{n}_taxa"] = {"taxa": n, "metrics": entry}


def main() -> None:
    RESULTS["meta"] = environment()
    RESULTS["meta"]["baselines"] = {
        "substitution_models": "scipy.linalg.expm on the same Q matrix",
        "graph_core": "networkx.DiGraph on an identical topology",
        "distances": "none; absolute scaling measurement",
        "pseudolikelihood": "none; absolute throughput measurement",
    }
    print("PhyNetPy core benchmarks")
    for k, v in RESULTS["meta"].items():
        if not isinstance(v, dict):
            print(f"  {k:24}{v}")

    bench_substitution_models()
    bench_graph_core()
    bench_pseudolikelihood()
    bench_distances()
    write(RESULTS, "core")


if __name__ == "__main__":
    main()
