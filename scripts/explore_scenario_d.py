"""
Focused test for DEFJ Scenario D using the known true species network.

True network: (o, ((a, ((z,y),x)#H1), (b, #H1)))
  - 6 taxa: a, b, o (diploid), x, y, z (tetraploid)
  - 1 hybridization event: clade (x,(y,z)) parented by a-lineage and b-lineage
"""

import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from phynetpy.IO import read_newick
from phynetpy.Infer_MP_Allop import (
    Allop_MUL, MPAllopScorer, MPAllopComponent,
    allele_map_set, partition_gene_trees,
)
from phynetpy.ModelFactory import ModelFactory
from phynetpy.ModelMove import SwitchParentage
from phynetpy.State import State
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance

DEFJ_ROOT = os.path.join(os.path.dirname(__file__), "..", "DEFJ", "10Genes", "withOG")

GENE_MAP = {
    "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
    "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
}

TRUE_NETWORK_NWK = "(o,((a,((z,y),x)#H1),(b,#H1)));"

D_CASES = [
    # (g, n, t, r, n_iters, description)
    ( 1, 1,   4, 1, 100, "1 gene, low ILS"),
    ( 3, 1,   4, 1, 100, "3 genes, low ILS"),
    (10, 1,   4, 1, 200, "10 genes, low ILS"),
    ( 1, 1,  20, 1, 100, "1 gene, moderate ILS"),
    ( 3, 1,  20, 1, 150, "3 genes, moderate ILS"),
    (10, 1,  20, 1, 200, "10 genes, moderate ILS"),
    ( 1, 1, 100, 1, 100, "1 gene, high ILS"),
    (10, 1, 100, 1, 200, "10 genes, high ILS"),
    (10, 1, 100, 3, 200, "10 genes, high ILS, rep3"),
    (10, 1,   4, 5, 200, "10 genes, low ILS, rep5"),
]


def load_gene_trees(g: int, n: int, t: int, r: int) -> list:
    prefix = "D2GT"
    filename = f"{prefix}g{g}n{n}t{t}r{r}-g_trees.newick"
    path = os.path.join(DEFJ_ROOT, "D", f"g{g}", f"n{n}", f"t{t}", f"r{r}", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gene tree file not found: {path}")
    trees = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("("):
                trees.append(read_newick(line))
    return trees


def run_d_test(g: int, n: int, t: int, r: int,
               n_iters: int, true_net, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    gts = load_gene_trees(g, n, t, r)

    for gt in gts:
        af = allele_map_set(gt, GENE_MAP)
        gt.put_item("allele maps", af)
        gt.put_item("leaf descendants", gt.leaf_descendants_all())

    start_net = partition_gene_trees(GENE_MAP, rng=rng)
    comp = MPAllopComponent(start_net, GENE_MAP, gts, rng)
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

    net = state.current_model.network
    final_score = state.likelihood()

    mu = mu_distance(net, true_net)
    hw = hardwired_cluster_distance(net, true_net)

    n_retics = sum(1 for nd in net.V() if nd.is_reticulation())

    return {
        "init_pars": -init_score,
        "final_pars": -final_score,
        "accepted": accepted,
        "iters": n_iters,
        "rate": f"{100*accepted/n_iters:.1f}%",
        "mu_d": mu,
        "hw_d": hw,
        "retics": n_retics,
        "elapsed": elapsed,
        "n_genes": len(gts),
        "newick": net.newick(),
    }


def main():
    true_net = read_newick(TRUE_NETWORK_NWK)
    true_leaves = sorted([nd.label for nd in true_net.get_leaves()])
    print(f"True network: {TRUE_NETWORK_NWK.strip()}")
    print(f"True leaves:  {true_leaves}")
    print(f"True retics:  {sum(1 for nd in true_net.V() if nd.is_reticulation())}")
    print()

    hdr = (f"{'Case':<35} {'GTs':>3} {'Pars0':>5} {'Pars':>5} "
           f"{'Acc':>7} {'Rate':>6} {'mu_d':>5} {'hw_d':>5} "
           f"{'Ret':>3} {'Time':>7}")
    sep = "-" * len(hdr)

    results = []
    for g, n, t, r, n_iters, desc in D_CASES:
        label = f"D-g{g}-t{t}-r{r}"
        print(f"Running {label:20s} ({desc}) ...", end=" ", flush=True)
        try:
            res = run_d_test(g, n, t, r, n_iters, true_net)
            print(f"pars={res['final_pars']}, mu_d={res['mu_d']}, "
                  f"hw_d={res['hw_d']}, {res['elapsed']:.2f}s")
            results.append((label, desc, res))
        except Exception as e:
            print(f"FAILED: {e}")

    print()
    print("=" * 80)
    print("  SCENARIO D RESULTS -- True network comparison")
    print("=" * 80)
    print()
    print(sep)
    print(hdr)
    print(sep)
    for label, desc, res in results:
        print(f"{label + ' ' + desc:<35} {res['n_genes']:>3} "
              f"{res['init_pars']:>5.0f} {res['final_pars']:>5.0f} "
              f"{res['accepted']:>3}/{res['iters']:<3} {res['rate']:>6} "
              f"{res['mu_d']:>5} {res['hw_d']:>5} "
              f"{res['retics']:>3} {res['elapsed']:>7.2f}s")
    print(sep)

    print("\nInferred topologies:")
    for label, _, res in results:
        print(f"  {label}: {res['newick']}")

    # Summary stats
    pars_zero = [r for _, _, r in results if r['final_pars'] == 0]
    if pars_zero:
        avg_mu = np.mean([r['mu_d'] for r in pars_zero])
        avg_hw = np.mean([r['hw_d'] for r in pars_zero])
        print(f"\nAmong {len(pars_zero)} cases reaching parsimony 0:")
        print(f"  avg mu_d = {avg_mu:.1f}, avg hw_d = {avg_hw:.1f}")

    all_mu = [r['mu_d'] for _, _, r in results]
    print(f"\nOverall mu_d:  min={min(all_mu)}, max={max(all_mu)}, "
          f"mean={np.mean(all_mu):.1f}")


if __name__ == "__main__":
    main()
