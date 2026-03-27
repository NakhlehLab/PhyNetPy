"""Get inferred network newicks for distance validation against PhyloNet cmpnets."""

import copy, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.setrecursionlimit(10000)

from phynetpy.IO import read_newick
from phynetpy.Infer_MP_Allop import (
    MPAllopComponent, allele_map_set, partition_gene_trees,
)
from phynetpy.ModelFactory import ModelFactory
from phynetpy.MetropolisHastings import SimulatedAnnealing, Infer_MP_Allop_Kernel
from phynetpy.State import State
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance

P = lambda *a, **kw: print(*a, **kw, flush=True)

DEFJ_ROOT = os.path.join(os.path.dirname(__file__), "..", "DEFJ", "10Genes", "withOG")

TRUE_NETWORKS = {
    "D": "(((b:0.009,((x:0.006,(y:0.003,z:0.003):0.003):0.003)#H1:0):0.003,(#H1:0,a:0.009):0.003):0.04366667,o:0.10233333);",
    "E": "(o:0.10283333,(((a:0.006,((y:0.003,z:0.003):0.003)#H1:0):0.003,(x:0.009)#H2:0):0.003,(#H2:0,(#H1:0,b:0.006):0.003):0.003):0.04316667);",
}

GENE_MAPS = {
    "D": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
    "E": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
}

CASES = [
    ("D", 10, 1, 4, 1, 300),
    ("E", 10, 1, 4, 1, 300),
]


def load_gene_trees(scenario, g, n, t, r):
    filename = f"{scenario}2GTg{g}n{n}t{t}r{r}-g_trees.newick"
    path = os.path.join(DEFJ_ROOT, scenario, f"g{g}", f"n{n}", f"t{t}", f"r{r}", filename)
    trees = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("("):
                trees.append(read_newick(line))
    return trees


def build_model(scenario, g, n, t, r, seed=42):
    rng = np.random.default_rng(seed)
    gene_map = GENE_MAPS[scenario]
    gts = load_gene_trees(scenario, g, n, t, r)
    for gt in gts:
        af = allele_map_set(gt, gene_map)
        gt.put_item("allele maps", af)
        gt.put_item("leaf descendants", gt.leaf_descendants_all())
    start_net = partition_gene_trees(gene_map, rng=rng)
    comp = MPAllopComponent(start_net, gene_map, gts, rng)
    return ModelFactory(comp).build()


def main():
    P("=" * 70)
    P("  Inferred network newicks for cmpnets validation")
    P("=" * 70)

    for scenario, g, n, t, r, n_iters in CASES:
        true_nwk = TRUE_NETWORKS[scenario]
        true_net = read_newick(true_nwk)

        P(f"\n--- {scenario}-g{g}-t{t}-r{r} ---")

        model = build_model(scenario, g, n, t, r)
        kernel = Infer_MP_Allop_Kernel()
        model_copy = copy.deepcopy(model)
        sa = SimulatedAnnealing(
            pkernel=kernel, model=model_copy,
            num_iter=n_iters, t_start=5.0, t_end=0.01,
            n_restarts=1, seed=42,
        )
        final_state = sa.run()

        inferred_net = final_state.current_model.network
        inferred_nwk = inferred_net.newick()

        our_mu = mu_distance(inferred_net, true_net)
        our_hw = hardwired_cluster_distance(inferred_net, true_net)

        P(f"  True network:     {true_nwk}")
        P(f"  Inferred network: {inferred_nwk}")
        P(f"  Our mu_distance:              {our_mu}")
        P(f"  Our hardwired_cluster_distance: {our_hw}")
        P(f"  Parsimony: {-sa.best_score:.0f}")

    P(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
