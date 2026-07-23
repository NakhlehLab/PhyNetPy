"""Direct bit-parity: Python _msnc_log_prob_network_int vs Cython network_dp_cy."""
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phynetpy.Network import Network
import phynetpy._msnc_density as M
from phynetpy._msnc_density import (
    _GeneTreeIndex, _NetworkIndex, MSCBranchKernel, _network_dp_cy,
)

# Reference: force the pure-Python frontier DP by temporarily disabling dispatch.
def py_score(net_idx, gti, engine):
    saved = M._CYTHON_NETWORK_DP
    M._CYTHON_NETWORK_DP = False
    try:
        return M._msnc_log_prob_network_int(net_idx, gti, engine)
    finally:
        M._CYTHON_NETWORK_DP = saved

def cy_score(net_idx, gti, engine):
    species_to_bits = {}
    for leaf_bit in gti.leaves:
        sp = gti.leaf_species_of.get(leaf_bit)
        if sp is not None:
            species_to_bits[sp] = species_to_bits.get(sp, 0) | (1 << leaf_bit)
    return _network_dp_cy(net_idx, gti, engine, species_to_bits)


def build_level1_3tax(gamma):
    from phynetpy.Network import Node, Edge
    labels = ["Root", "P1", "P2", "#H", "A", "B", "C"]
    nodes = {l: Node(l, is_reticulation=(l == "#H")) for l in labels}
    net = Network(); net.add_nodes(*nodes.values())
    net.add_edges([
        Edge(nodes["Root"], nodes["P1"], length=1.0),
        Edge(nodes["Root"], nodes["P2"], length=1.0),
        Edge(nodes["P1"], nodes["A"], length=1.5),
        Edge(nodes["P1"], nodes["#H"], length=0.5, gamma=gamma),
        Edge(nodes["P2"], nodes["#H"], length=0.5, gamma=1.0 - gamma),
        Edge(nodes["P2"], nodes["C"], length=1.5),
        Edge(nodes["#H"], nodes["B"], length=0.5),
    ])
    return net

sp_of = {"A": "A", "B": "B", "C": "C", "D": "D"}
gts_3 = [
    "((A:0.5,B:0.5):1.5,C:2);",
    "((A:0.5,C:0.5):1.5,B:2);",
    "((B:0.5,C:0.5):1.5,A:2);",
]

maxdiff = 0.0
n = 0
for gamma in [0.05, 0.2, 0.5, 0.73, 0.95]:
    for theta in [0.005, 0.02, 0.1]:
        net = build_level1_3tax(gamma)
        net_idx = _NetworkIndex(net)
        eng = MSCBranchKernel(theta=theta)
        for nwk in gts_3:
            gt = Network.from_newick(nwk)
            gti = _GeneTreeIndex(gt, sp_of)
            a = py_score(net_idx, gti, eng)
            b = cy_score(net_idx, gti, eng)
            d = abs(a - b)
            maxdiff = max(maxdiff, d)
            n += 1
            if d > 1e-9:
                print(f"MISMATCH gamma={gamma} theta={theta} {nwk}: py={a:.12f} cy={b:.12f} d={d:.2e}")

print(f"3-tax: {n} comparisons, max |py-cy| = {maxdiff:.3e}")
print("PARITY OK" if maxdiff < 1e-9 else "PARITY FAIL")


# ---- timed DP parity (MCMC_SEQ path) ---------------------------------
from phynetpy._msnc_density import (
    build_network_msnc_index, build_gene_tree_msnc_index,
    _msnc_log_density_timed, _network_dp_timed_cy,
)

def timed_py(net_idx, gti, events, sp_heights, theta):
    saved = M._CYTHON_TIMED_DP
    M._CYTHON_TIMED_DP = False
    try:
        return _msnc_log_density_timed(net_idx, gti, events, sp_heights, theta)
    finally:
        M._CYTHON_TIMED_DP = saved

maxdiff2 = 0.0
n2 = 0
for gamma in [0.05, 0.2, 0.5, 0.73, 0.95]:
    for theta in [0.005, 0.02, 0.1]:
        net = build_level1_3tax(gamma)
        net_idx, sph = build_network_msnc_index(net)
        for nwk in gts_3:
            gt = Network.from_newick(nwk)
            gti, events = build_gene_tree_msnc_index(gt, sp_of)
            a = timed_py(net_idx, gti, events, sph, theta)
            b = _network_dp_timed_cy(net_idx, gti, events, sph, theta)
            d = abs(a - b)
            maxdiff2 = max(maxdiff2, d)
            n2 += 1
            if d > 1e-9:
                print(f"TIMED MISMATCH gamma={gamma} theta={theta} {nwk}: py={a:.12f} cy={b:.12f} d={d:.2e}")

print(f"timed 3-tax: {n2} comparisons, max |py-cy| = {maxdiff2:.3e}")
print("TIMED PARITY OK" if maxdiff2 < 1e-9 else "TIMED PARITY FAIL")
