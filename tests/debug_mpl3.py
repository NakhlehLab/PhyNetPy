"""Debug: check if a branch length scaling factor explains the discrepancy."""

from __future__ import annotations
import os
import math
from itertools import combinations
from phynetpy.Network import Network, Edge
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL_reference import _compute_rho, _subnet_triple_probs
from phynetpy.GraphUtils import subnet_given_leaves, _displayed_trees_with_probs
from phynetpy.IO import convert_newick

TESTFILES = os.path.join(os.path.dirname(__file__), "testfiles")
TAXA = ["t14", "t15", "t49", "t68", "t69", "t72", "t75", "t91", "t114", "t133"]
MAPPING = {t: [t] for t in TAXA}
_LOG_FLOOR = math.log(1e-200)

NET1 = "(((((t14:1.0,t75:1.0):0.8767426254184691,(t69:1.0,t114:1.0):2.0170864999696456):5.213391915869064,(t91:1.0)#H1:1.0::0.3011071643186969):0.8237808968694593,t133:1.0):2.054746072134383,(((t15:1.0,#H1:1.0::0.6988928356813031):1.481149764930965,(t72:1.0,t49:1.0):3.0976006419526203):1.2243080352496307,t68:1.0):0.12981149034576273);"

# Load gene trees
gt_path = os.path.join(TESTFILES, "subgeneset_3_ret1.txt")
trees = []
with open(gt_path) as f:
    for line in f:
        line = line.strip()
        if line:
            trees.append(Network.from_newick(line))
gts = GeneTrees(gene_tree_list=trees)
gts.species_gene_mapping = MAPPING

# Precompute rho for all triplets
triplets = list(combinations(sorted(TAXA), 3))
rho_cache = {}
for t in triplets:
    rho_cache[frozenset(t)] = _compute_rho(t[0], t[1], t[2], gts, MAPPING)

def score_with_scale(net, scale_factor):
    """Score the network after scaling all non-leaf internal branch lengths."""
    total = 0.0
    for triplet in triplets:
        leaf_nodes = [net.has_node_named(t) for t in triplet]
        subnet = subnet_given_leaves(net, leaf_nodes)
        
        # Scale internal branches in subnet
        if scale_factor != 1.0:
            root = subnet.root()
            for edge in subnet.E():
                # Only scale the edge from root to the cherry (internal branch)
                # Actually, scale ALL edges except leaf edges
                if subnet.out_degree(edge.dest) > 0:  # not a leaf
                    edge.set_length(edge.get_length() * scale_factor)

        probs = _subnet_triple_probs(subnet)
        rho = rho_cache[frozenset(triplet)]
        
        for rho_i, p_i in zip(rho, probs):
            if rho_i > 0.0:
                total += rho_i * (math.log(p_i) if p_i > 0.0 else _LOG_FLOOR)
    return total


net = Network.from_newick(convert_newick(NET1, "PhyNetPy"))
expected = -56625.66771610746

# Try different scaling factors
print("Trying different internal branch length scale factors:")
for scale in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
    # Need fresh network each time since we modify in place
    net = Network.from_newick(convert_newick(NET1, "PhyNetPy"))
    s = score_with_scale(net, scale)
    print(f"  scale={scale:.1f}: score={s:.2f} (expected={expected:.2f}, diff={s-expected:.2f})")

# Also try: what if the leaf branches (length=1.0) contribute to tau somehow?
# Try scaling ALL branches (including leaves)
print("\nTrying scaling ALL edges (including leaf edges):")
for scale in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]:
    net = Network.from_newick(convert_newick(NET1, "PhyNetPy"))
    for edge in net.E():
        edge.set_length(edge.get_length() * scale)
    s = score_with_scale(net, 1.0)
    print(f"  scale={scale:.1f}: score={s:.2f} (expected={expected:.2f}, diff={s-expected:.2f})")

# Check: what if we should NOT include leaf branch lengths in the tau computation?
# In our implementation, when the subnet is cleaned, leaf edges DON'T affect tau.
# Let me verify by printing tau for one specific triplet.
print("\n\nDetailed look at triplet (t14, t75, t133):")
net = Network.from_newick(convert_newick(NET1, "PhyNetPy"))
triplet = ("t14", "t75", "t133")
leaf_nodes = [net.has_node_named(t) for t in triplet]
subnet = subnet_given_leaves(net, leaf_nodes)

print("Subnet structure:")
root = subnet.root()
def show(n, depth=0):
    for ch in subnet.get_children(n):
        e = subnet.get_edge(n, ch)
        g = f" gamma={e.get_gamma()}" if e.get_gamma() else ""
        children = subnet.get_children(ch)
        print(f"  {'  '*depth}{n.label} -> {ch.label}: len={e.get_length():.6f}{g} (leaf={'yes' if not children else 'no'})")
        show(ch, depth+1)
show(root)

rho = rho_cache[frozenset(triplet)]
probs = _subnet_triple_probs(subnet)
print(f"\nrho = {rho}")
print(f"probs = {probs}")
print(f"sum(probs) = {sum(probs)}")
contribution = sum(r * (math.log(p) if p > 0 else _LOG_FLOOR) for r, p in zip(rho, probs) if r > 0)
print(f"contribution = {contribution:.6f}")
