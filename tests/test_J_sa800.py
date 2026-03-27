"""Quick test: Scenario J, SA T=5, 800 iterations."""

import copy, os, sys, time, traceback, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.setrecursionlimit(10000)

from phynetpy.IO import read_newick
from phynetpy.Infer_MP_Allop import (
    Allop_MUL, MPAllopScorer, MPAllopComponent,
    allele_map_set, partition_gene_trees,
)
from phynetpy.ModelFactory import ModelFactory
from phynetpy.ModelMove import SwitchParentage
from phynetpy.MetropolisHastings import SimulatedAnnealing, Infer_MP_Allop_Kernel
from phynetpy.State import State
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance

P = lambda *a, **kw: print(*a, **kw, flush=True)

DEFJ_ROOT = os.path.join(os.path.dirname(__file__), "..", "DEFJ", "10Genes", "withOG")

TRUE_J = "((((b:0.017,a:0.017):0.01,((c:0.011,(d:0.006,(v:0.006)#H1:0):0.005):0.012,(((x:0.013,(y:0.01,z:0.01):0.003):0.003,w:0.016):0.007)#H2:0):0.004):0.008,((((#H1:0,e:0.006):0.006,(t:0.003,u:0.003):0.009):0.013,#H2:0):0.005,f:0.032):0.003):0.002275,o:0.043225);"

GENE_MAP_J = {
    "o": ["01oA"],
    "a": ["01aA"], "b": ["01bA"], "c": ["01cA"],
    "d": ["01dA"], "e": ["01eA"], "f": ["01fA"],
    "t": ["01tA", "01tB"], "u": ["01uA", "01uB"], "v": ["01vA", "01vB"],
    "w": ["01wA", "01wB"], "x": ["01xA", "01xB"],
    "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
}


def load_gene_trees(g, n, t, r):
    filename = f"J2GTg{g}n{n}t{t}r{r}-g_trees.newick"
    path = os.path.join(DEFJ_ROOT, "J", f"g{g}", f"n{n}", f"t{t}", f"r{r}", filename)
    trees = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("("):
                trees.append(read_newick(line))
    return trees


def build_model(g, n, t, r, seed=42):
    rng = np.random.default_rng(seed)
    gts = load_gene_trees(g, n, t, r)
    for gt in gts:
        af = allele_map_set(gt, GENE_MAP_J)
        gt.put_item("allele maps", af)
        gt.put_item("leaf descendants", gt.leaf_descendants_all())
    start_net = partition_gene_trees(GENE_MAP_J, rng=rng)
    comp = MPAllopComponent(start_net, GENE_MAP_J, gts, rng)
    return ModelFactory(comp).build()


def main():
    try:
        true_net = read_newick(TRUE_J)

        P("=" * 70)
        P("  Scenario J  |  SA T=5  |  800 iterations  |  1 restart")
        P("=" * 70)

        P(f"\n  J-g10-t20-r1 (10g mod-ILS)")
        P(f"  Building model...", end="")
        model = build_model(10, 1, 20, 1)
        P(" done")

        init_score = State(copy.deepcopy(model)).likelihood()
        P(f"  Init parsimony: {-init_score:.0f}")

        P(f"  Running SA T=5 x1 @ 800 iters...")
        t0 = time.perf_counter()
        kernel = Infer_MP_Allop_Kernel()
        model_copy = copy.deepcopy(model)
        sa = SimulatedAnnealing(
            pkernel=kernel, model=model_copy,
            num_iter=800, t_start=5.0, t_end=0.01,
            n_restarts=1, seed=42,
        )
        final_state = sa.run()
        elapsed = time.perf_counter() - t0
        P(f"  SA x1 complete ({elapsed:.1f}s)")

        net1 = final_state.current_model.network
        acc1 = sum(s["accepted"] for s in sa.run_stats)
        up1 = sum(s["uphill"] for s in sa.run_stats)
        mu1 = mu_distance(net1, true_net)
        hw1 = hardwired_cluster_distance(net1, true_net)

        P(f"    Pars: {-sa.best_score:.0f}  Acc/Up: {acc1}/{up1}  mu_d: {mu1}  hw_d: {hw1}  Time: {elapsed:.1f}s")

        P(f"\n  Running SA T=5 x1 @ 800 iters (restart 2, fresh model)...")
        t0 = time.perf_counter()
        kernel2 = Infer_MP_Allop_Kernel()
        model_copy2 = copy.deepcopy(model)
        sa2 = SimulatedAnnealing(
            pkernel=kernel2, model=model_copy2,
            num_iter=800, t_start=5.0, t_end=0.01,
            n_restarts=1, seed=123,
        )
        final_state2 = sa2.run()
        elapsed2 = time.perf_counter() - t0
        P(f"  SA x1 run2 complete ({elapsed2:.1f}s)")

        net2 = final_state2.current_model.network
        acc2 = sum(s["accepted"] for s in sa2.run_stats)
        up2 = sum(s["uphill"] for s in sa2.run_stats)
        mu2 = mu_distance(net2, true_net)
        hw2 = hardwired_cluster_distance(net2, true_net)

        P(f"    Pars: {-sa2.best_score:.0f}  Acc/Up: {acc2}/{up2}  mu_d: {mu2}  hw_d: {hw2}  Time: {elapsed2:.1f}s")

        P(f"\n  Running SA T=5 x1 @ 800 iters (restart 3, fresh model)...")
        t0 = time.perf_counter()
        kernel3 = Infer_MP_Allop_Kernel()
        model_copy3 = copy.deepcopy(model)
        sa3 = SimulatedAnnealing(
            pkernel=kernel3, model=model_copy3,
            num_iter=800, t_start=5.0, t_end=0.01,
            n_restarts=1, seed=999,
        )
        final_state3 = sa3.run()
        elapsed3 = time.perf_counter() - t0
        P(f"  SA x1 run3 complete ({elapsed3:.1f}s)")

        net3 = final_state3.current_model.network
        acc3 = sum(s["accepted"] for s in sa3.run_stats)
        up3 = sum(s["uphill"] for s in sa3.run_stats)
        mu3 = mu_distance(net3, true_net)
        hw3 = hardwired_cluster_distance(net3, true_net)

        P(f"    Pars: {-sa3.best_score:.0f}  Acc/Up: {acc3}/{up3}  mu_d: {mu3}  hw_d: {hw3}  Time: {elapsed3:.1f}s")

        best_pars = min(-sa.best_score, -sa2.best_score, -sa3.best_score)
        best_mu = min(mu1, mu2, mu3)
        best_hw = min(hw1, hw2, hw3)
        total_time = elapsed + elapsed2 + elapsed3

        P(f"\n  Best across 3 manual restarts:")
        P(f"    Best parsimony  : {best_pars:.0f}")
        P(f"    Best mu_d       : {best_mu}")
        P(f"    Best hw_d       : {best_hw}")
        P(f"    Total wall time : {total_time:.1f}s")

        P(f"\n{'=' * 70}")

    except Exception as e:
        P(f"\nERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
