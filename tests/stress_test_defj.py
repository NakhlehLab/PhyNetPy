"""
Stress test for MP Allop inference on the DEFJ benchmark dataset.

Runs inference on a diverse set of DEFJ scenarios varying in:
  - Scenario topology (D, E, F, J)
  - Gene tree count (g1, g3, g10)
  - ILS level (t4, t100)
  - Replicates

Reports: runtime, parsimony scores, mu-distance to true MUL tree,
         acceptance rate, and network structure.
"""

import copy
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from phynetpy.IO import read_newick
from phynetpy.Infer_MP_Allop import (
    Allop_MUL, MPAllopScorer, MPAllopComponent,
    allele_map_set, partition_gene_trees, InferMPAllop,
)
from phynetpy.ModelFactory import ModelFactory
from phynetpy.ModelGraph import Model
from phynetpy.ModelMove import SwitchParentage
from phynetpy.State import State
from phynetpy.Network import Network, MUL

try:
    from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance
    HAS_DISTANCES = True
except ImportError:
    HAS_DISTANCES = False

DEFJ_ROOT = os.path.join(os.path.dirname(__file__), "..", "DEFJ", "10Genes", "withOG")

GENE_MAPS = {
    "D": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
    "E": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
    "F": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
    "J": {
        "o": ["01oA"],
        "a": ["01aA"], "b": ["01bA"], "c": ["01cA"],
        "d": ["01dA"], "e": ["01eA"], "f": ["01fA"],
        "t": ["01tA", "01tB"], "u": ["01uA", "01uB"], "v": ["01vA", "01vB"],
        "w": ["01wA", "01wB"], "x": ["01xA", "01xB"],
        "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
}

TEST_CASES = [
    # (scenario, g, n, t, r, n_iters, description)
    ("D",  1, 1,   4, 1, 300, "D-simple: 1 gene, low ILS"),
    ("D",  3, 1,   4, 1, 300, "D-3genes: 3 genes, low ILS"),
    ("D", 10, 1,   4, 1, 300, "D-10genes: 10 genes, low ILS"),
    ("D",  1, 1, 100, 1, 300, "D-highILS: 1 gene, high ILS"),
    ("D", 10, 1, 100, 3, 300, "D-10g-highILS: 10 genes, high ILS, rep3"),
    ("E",  1, 1,   4, 1, 300, "E-simple: 1 gene, low ILS"),
    ("E", 10, 1,   4, 1, 300, "E-10genes: 10 genes"),
    ("F",  1, 1,   4, 1, 300, "F-simple: 1 gene"),
    ("F", 10, 1,   4, 1, 300, "F-10genes: 10 genes"),
    ("J",  10, 1,  20, 1, 800, "J-simple: 14spp, 10 genes"),
]


def load_gene_trees(scenario: str, g: int, n: int, t: int, r: int) -> list[Network]:
    """Read gene trees from a DEFJ newick file."""
    prefix = f"{scenario}2GT"
    filename = f"{prefix}g{g}n{n}t{t}r{r}-g_trees.newick"
    path = os.path.join(DEFJ_ROOT, scenario, f"g{g}", f"n{n}", f"t{t}", f"r{r}", filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Gene tree file not found: {path}")

    trees = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("("):
                trees.append(read_newick(line))
    return trees


def load_true_mul(scenario: str, g: int, n: int, t: int) -> Network:
    """Read the true MUL tree and add '01' prefix to match gene map naming."""
    path = os.path.join(DEFJ_ROOT, scenario, f"g{g}", f"n{n}", f"t{t}",
                        "multree.newick")
    with open(path) as f:
        nwk = f.read().strip()
    true_mul = read_newick(nwk)
    for leaf in true_mul.get_leaves():
        true_mul.update_node_name(leaf, f"01{leaf.label}")
    return true_mul


def compute_mul_distance_permuted(inferred_net: Network, true_mul: Network,
                                  gene_map: dict,
                                  rng: np.random.Generator) -> dict:
    """
    Compare inferred network's MUL tree to the true MUL tree, trying all
    A/B label permutations for polyploid species to find the best match.
    """
    if not HAS_DISTANCES:
        return {"mu_dist": "N/A", "hw_dist": "N/A"}

    try:
        mul_obj = Allop_MUL(gene_map, rng)
        mul_obj.to_mul(inferred_net)
        inferred_mul = mul_obj.mul

        polyploid_spp = [sp for sp, copies in gene_map.items()
                         if len(copies) > 1]
        n_poly = len(polyploid_spp)

        best_mu = float("inf")
        best_hw = float("inf")

        for mask in range(1 << n_poly):
            relabeled = copy.deepcopy(inferred_mul)
            for i, sp in enumerate(polyploid_spp):
                if mask & (1 << i):
                    a_name = f"01{sp}A"
                    b_name = f"01{sp}B"
                    a_leaf = b_leaf = None
                    for leaf in relabeled.get_leaves():
                        if leaf.label == a_name:
                            a_leaf = leaf
                        elif leaf.label == b_name:
                            b_leaf = leaf
                    if a_leaf and b_leaf:
                        relabeled.update_node_name(a_leaf, "TEMP_SWAP")
                        relabeled.update_node_name(b_leaf, a_name)
                        relabeled.update_node_name(a_leaf, b_name)

            inf_leaves = sorted([nd.label for nd in relabeled.get_leaves()])
            true_leaves = sorted([nd.label for nd in true_mul.get_leaves()])
            if inf_leaves != true_leaves:
                continue

            mu = mu_distance(relabeled, true_mul)
            hw = hardwired_cluster_distance(relabeled, true_mul)
            if mu < best_mu:
                best_mu = mu
                best_hw = hw

        if best_mu == float("inf"):
            return {"mu_dist": "no_valid_perm", "hw_dist": "no_valid_perm"}
        return {"mu_dist": best_mu, "hw_dist": best_hw}
    except Exception as e:
        return {"mu_dist": f"err:{e}", "hw_dist": f"err:{e}"}


def run_test(scenario: str, g: int, n: int, t: int, r: int,
             n_iters: int, seed: int = 42) -> dict:
    """Run a single inference test and return metrics."""
    rng = np.random.default_rng(seed)
    gene_map = GENE_MAPS[scenario]

    gts = load_gene_trees(scenario, g, n, t, r)
    true_mul = load_true_mul(scenario, g, n, t)

    for gt in gts:
        af = allele_map_set(gt, gene_map)
        gt.put_item("allele maps", af)
        gt.put_item("leaf descendants", gt.leaf_descendants_all())

    start_net = partition_gene_trees(gene_map, rng=rng)
    comp = MPAllopComponent(start_net, gene_map, gts, rng)
    model = ModelFactory(comp).build()
    state = State(model)

    init_score = state.likelihood()

    t0 = time.perf_counter()
    accepted = 0
    for i in range(n_iters):
        move = SwitchParentage(i)
        is_valid = state.generate_next(move)
        if is_valid:
            cur = state.likelihood()
            proposed = state.proposed().likelihood()
            if cur - proposed < 0:
                state.commit(move)
                accepted += 1
            else:
                state.revert(move)
    elapsed = time.perf_counter() - t0

    final_score = state.likelihood()
    net = state.current_model.network
    n_retics = sum(1 for nd in net.V() if nd.is_reticulation())
    n_leaves = len(list(net.get_leaves()))

    dists = compute_mul_distance_permuted(net, true_mul, gene_map, rng)

    return {
        "init_score": init_score,
        "final_score": final_score,
        "parsimony": -final_score,
        "accepted": accepted,
        "iters": n_iters,
        "accept_rate": f"{100*accepted/n_iters:.1f}%",
        "elapsed": elapsed,
        "n_genes": len(gts),
        "n_leaves": n_leaves,
        "n_retics": n_retics,
        "newick": net.newick(),
        **dists,
    }


def print_table(results: list[tuple[str, dict]]) -> None:
    """Pretty-print a results table."""
    hdr = (
        f"{'Test':<30} {'Taxa':>4} {'GTs':>3} {'Pars0':>6} {'Pars':>5} "
        f"{'Acc':>6} {'Rate':>6} {'mu_d':>6} {'hw_d':>6} "
        f"{'Retics':>6} {'Time':>7}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for name, r in results:
        mu_d = r["mu_dist"] if isinstance(r["mu_dist"], str) else f"{r['mu_dist']:>6}"
        hw_d = r["hw_dist"] if isinstance(r["hw_dist"], str) else f"{r['hw_dist']:>6}"
        print(
            f"{name:<30} {r['n_leaves']:>4} {r['n_genes']:>3} "
            f"{-r['init_score']:>6.0f} {r['parsimony']:>5.0f} "
            f"{r['accepted']:>3}/{r['iters']:<3} {r['accept_rate']:>6} "
            f"{mu_d:>6} {hw_d:>6} "
            f"{r['n_retics']:>6} {r['elapsed']:>7.2f}s"
        )
    print(sep)


def main():
    print("=" * 80)
    print("  DEFJ STRESS TEST -- MP Allop Inference")
    print("=" * 80)
    print()

    results = []
    for scenario, g, n, t, r, n_iters, desc in TEST_CASES:
        label = f"{scenario}-g{g}-t{t}-r{r}"
        print(f"Running {label} ({desc}) ...", end=" ", flush=True)
        try:
            metrics = run_test(scenario, g, n, t, r, n_iters)
            print(f"done  [pars: {metrics['parsimony']:.0f}, "
                  f"mu_d: {metrics['mu_dist']}, {metrics['elapsed']:.2f}s]")
            results.append((label, metrics))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((label, {
                "init_score": 0, "final_score": 0, "parsimony": -1,
                "accepted": 0, "iters": n_iters, "accept_rate": "N/A",
                "elapsed": 0, "n_genes": 0, "n_leaves": 0, "n_retics": 0,
                "mu_dist": "FAIL", "hw_dist": "FAIL", "newick": "",
            }))

    print()
    print_table(results)

    print("\nInferred network topologies:")
    for name, r in results:
        if r["newick"]:
            print(f"  {name}: {r['newick'][:120]}{'...' if len(r['newick']) > 120 else ''}")
    print()


if __name__ == "__main__":
    main()
