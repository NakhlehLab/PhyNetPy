"""Debug: verify network edge gammas and compare rho computation."""

from __future__ import annotations
import os
import math
from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL_reference import (
    MPL, _compute_rho, _subnet_triple_probs,
)
from phynetpy.GraphUtils import subnet_given_leaves
from phynetpy.IO import convert_newick

TESTFILES = os.path.join(os.path.dirname(__file__), "testfiles")

# ── Parse network ──
NET1 = "(((((t14:1.0,t75:1.0):0.8767426254184691,(t69:1.0,t114:1.0):2.0170864999696456):5.213391915869064,(t91:1.0)#H1:1.0::0.3011071643186969):0.8237808968694593,t133:1.0):2.054746072134383,(((t15:1.0,#H1:1.0::0.6988928356813031):1.481149764930965,(t72:1.0,t49:1.0):3.0976006419526203):1.2243080352496307,t68:1.0):0.12981149034576273);"

net = Network.from_newick(convert_newick(NET1, "PhyNetPy"))

print("=== Network edges with gammas ===")
for edge in net.E():
    gamma = edge.get_gamma()
    gamma_str = f" gamma={gamma}" if gamma else ""
    retic_str = " [RETIC]" if edge.dest.is_reticulation else ""
    print(f"  {edge.src.label} -> {edge.dest.label}: len={edge.get_length()}{gamma_str}{retic_str}")

print(f"\nReticulation node #H1 in-edges:")
h1 = net.has_node_named("#H1")
if h1:
    for e in net.in_edges(h1):
        print(f"  {e.src.label} -> {e.dest.label}: len={e.get_length()}, gamma={e.get_gamma()}")

# ── Load gene trees ──
gt_path = os.path.join(TESTFILES, "subgeneset_3_ret1.txt")
trees = []
with open(gt_path) as f:
    for line in f:
        line = line.strip()
        if line:
            trees.append(Network.from_newick(line))

TAXA = ["t14", "t15", "t49", "t68", "t69", "t72", "t75", "t91", "t114", "t133"]
MAPPING = {t: [t] for t in TAXA}
gts = GeneTrees(gene_tree_list=trees)
gts.species_gene_mapping = MAPPING

# ── Check rho for a few triplets ──
print("\n=== Rho for selected triplets ===")
triplets = [("t14", "t15", "t91"), ("t15", "t91", "t133"), ("t14", "t69", "t72")]
for t in triplets:
    rho = _compute_rho(t[0], t[1], t[2], gts, MAPPING)
    print(f"  rho({t}) = {rho}")
    print(f"    sum = {sum(rho)}")

# ── Compare individual triplet contributions ──
print("\n=== Triplet-by-triplet score breakdown ===")
from itertools import combinations
total = 0.0
all_triplets = list(combinations(sorted(TAXA), 3))
print(f"Total triplets: {len(all_triplets)}")

_LOG_FLOOR = math.log(1e-200)

for triplet in all_triplets:
    key = frozenset(triplet)
    rho = _compute_rho(triplet[0], triplet[1], triplet[2], gts, MAPPING)
    
    leaf_nodes = [net.has_node_named(t) for t in triplet]
    subnet = subnet_given_leaves(net, leaf_nodes)
    probs = _subnet_triple_probs(subnet)
    
    contribution = 0.0
    for rho_i, p_i in zip(rho, probs):
        if rho_i > 0.0:
            contribution += rho_i * (math.log(p_i) if p_i > 0.0 else _LOG_FLOOR)
    total += contribution

print(f"\nTotal score (manual sum): {total:.6f}")

# Now compute via MPL class
mpl = MPL(net, gts, MAPPING)
score = mpl.score()
print(f"Total score (MPL.score): {score:.6f}")
print(f"Expected:               -56625.667716")
