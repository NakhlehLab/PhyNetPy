"""Is the true reticulation even preferred by the SEQ *likelihood*?

Holds the gene trees at their TRUE simulated values and theta at the true
value, then compares the total log likelihood log P(S|G) + log P(G|Psi) of:
  * the true 1-reticulation network,
  * the same network with the reticulation deleted, keeping the MAJOR parent,
  * the same network with the reticulation deleted, keeping the MINOR parent.

If the true network's MSNC term is not clearly higher than both trees', the
reticulation is not identifiable from the coalescent signal at these settings
(a data/power issue), independent of any sampler mixing.  We sweep the number
of loci to see when/if the reticulation becomes identifiable.
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import JC69, simulate_multilocus
from phynetpy.Network import Network, Edge
from phynetpy._msnc_density import (
    build_network_msnc_index, build_gene_tree_msnc_index,
    msnc_log_density_prebuilt,
)

def msnc_total(species_net, gene_trees, species_of, theta):
    net_idx, sph = build_network_msnc_index(species_net)
    tot = 0.0
    for gt in gene_trees:
        gti, ev = build_gene_tree_msnc_index(gt, species_of)
        d = msnc_log_density_prebuilt(net_idx, sph, gti, ev, theta)
        tot += d
    return tot

def delete_retic_keep(net, keep_major=True):
    """Return a copy of net with the reticulation removed, keeping one parent."""
    net = copy.deepcopy(net)
    ret = [v for v in net.V() if v.is_reticulation()][0]
    in_edges = list(net.in_edges(ret))
    # gamma-major edge has the larger gamma
    ge = [(e, (e.get_gamma() or 0.0)) for e in in_edges]
    ge.sort(key=lambda x: x[1], reverse=True)
    drop = ge[1][0] if keep_major else ge[0][0]  # drop minor keeps major
    net.remove_edge(drop)
    ret.set_is_reticulation(False)
    # suppress degree-2 nodes
    for v in list(net.V()):
        if net.in_degree(v) == 1 and net.out_degree(v) == 1:
            ie = list(net.in_edges(v))[0]; oe = list(net.out_edges(v))[0]
            p, c = ie.src, oe.dest
            net.remove_edge(ie); net.remove_edge(oe); net.remove_nodes(v)
            net.add_edges(Edge(p, c))
    return net

true_net = build_true_network()
species_of = {a: sp for sp, al in MAPPING.items() for a in al}

for n_loci in [20, 50, 100, 200]:
    data = simulate_multilocus(true_net, MAPPING, n_loci=n_loci, seq_length=200,
                               theta=0.02, model=JC69(), seed=2024)
    gts = data.gene_trees
    d_true = msnc_total(true_net, gts, species_of, 0.02)
    d_maj = msnc_total(delete_retic_keep(true_net, True), gts, species_of, 0.02)
    d_min = msnc_total(delete_retic_keep(true_net, False), gts, species_of, 0.02)
    best_tree = max(d_maj, d_min)
    print(f"loci={n_loci:3d}: MSNC true={d_true:10.2f}  major={d_maj:10.2f}  "
          f"minor={d_min:10.2f}  (true-best_tree={d_true - best_tree:+.2f})")
