"""Get inferred network newicks for cases WITH nonzero distances."""

import copy, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.setrecursionlimit(10000)

from phynetpy.IO import read_newick
from phynetpy.Infer_MP_Allop import (
    MPAllopComponent, allele_map_set, partition_gene_trees,
)
from phynetpy.ModelFactory import ModelFactory
from phynetpy.ModelMove import SwitchParentage
from phynetpy.State import State
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance

P = lambda *a, **kw: print(*a, **kw, flush=True)

DEFJ_ROOT = os.path.join(os.path.dirname(__file__), "..", "DEFJ", "10Genes", "withOG")

TRUE_D = "(((b:0.009,((x:0.006,(y:0.003,z:0.003):0.003):0.003)#H1:0):0.003,(#H1:0,a:0.009):0.003):0.04366667,o:0.10233333);"
TRUE_E = "(o:0.10283333,(((a:0.006,((y:0.003,z:0.003):0.003)#H1:0):0.003,(x:0.009)#H2:0):0.003,(#H2:0,(#H1:0,b:0.006):0.003):0.003):0.04316667);"

GENE_MAP = {
    "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
    "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
}


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


def build_and_hillclimb(scenario, g, n, t, r, n_iters, seed=42):
    """Use HC (no SA) to get a suboptimal network with nonzero distance."""
    rng = np.random.default_rng(seed)
    gts = load_gene_trees(scenario, g, n, t, r)
    for gt in gts:
        af = allele_map_set(gt, GENE_MAP)
        gt.put_item("allele maps", af)
        gt.put_item("leaf descendants", gt.leaf_descendants_all())
    start_net = partition_gene_trees(GENE_MAP, rng=rng)
    comp = MPAllopComponent(start_net, GENE_MAP, gts, rng)
    model = ModelFactory(comp).build()

    state = State(copy.deepcopy(model))
    for i in range(n_iters):
        move = SwitchParentage(i)
        is_valid = state.generate_next(move)
        if is_valid:
            cur = state.likelihood()
            proposed = state.proposed().likelihood()
            if cur - proposed < 0:
                state.commit(move)
            else:
                state.revert(move)
    return state.current_model.network


def main():
    true_d = read_newick(TRUE_D)
    true_e = read_newick(TRUE_E)

    P("Getting HC-inferred networks (intentionally suboptimal for nonzero distances)...\n")

    # D high-ILS (should give nonzero distances)
    inf_d = build_and_hillclimb("D", 10, 1, 100, 1, 100)
    inf_d_nwk = inf_d.newick()
    mu_d = mu_distance(inf_d, true_d)
    hw_d = hardwired_cluster_distance(inf_d, true_d)

    P(f"=== D-g10-t100-r1 (HC 100 iters) ===")
    P(f"  True:     {TRUE_D}")
    P(f"  Inferred: {inf_d_nwk}")
    P(f"  Our mu_distance:  {mu_d}")
    P(f"  Our hw_distance:  {hw_d}")

    # E high-ILS
    inf_e = build_and_hillclimb("E", 10, 1, 100, 1, 100)
    inf_e_nwk = inf_e.newick()
    mu_e = mu_distance(inf_e, true_e)
    hw_e = hardwired_cluster_distance(inf_e, true_e)

    P(f"\n=== E-g10-t100-r1 (HC 100 iters) ===")
    P(f"  True:     {TRUE_E}")
    P(f"  Inferred: {inf_e_nwk}")
    P(f"  Our mu_distance:  {mu_e}")
    P(f"  Our hw_distance:  {hw_e}")

    # Also test two known networks manually
    P(f"\n=== Manual sanity check: true D vs true E ===")
    mu_de = mu_distance(true_d, true_e)
    hw_de = hardwired_cluster_distance(true_d, true_e)
    P(f"  mu_distance(D_true, E_true):  {mu_de}")
    P(f"  hw_distance(D_true, E_true):  {hw_de}")

    # Format nexus for cmpnets
    P(f"\n\n{'='*70}")
    P(f"  NEXUS blocks for PhyloNet cmpnets (copy-paste into .nex file)")
    P(f"{'='*70}")

    P(f"""
#NEXUS

BEGIN NETWORKS;
Network true_D = {TRUE_D}
Network inferred_D = {inf_d_nwk}
Network true_E = {TRUE_E}
Network inferred_E = {inf_e_nwk}
END;

BEGIN PHYLONET;
cmpnets true_D inferred_D;
cmpnets true_E inferred_E;
cmpnets true_D true_E;
END;
""")

if __name__ == "__main__":
    main()
