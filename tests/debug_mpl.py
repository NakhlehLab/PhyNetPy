"""Debug script: examine MPL internals for a single triplet on Network #1."""

from __future__ import annotations
import math
from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL_reference import (
    MPL, _induced_triple, _coalescent_triple_probs,
    _compute_rho, _subnet_triple_probs,
)
from phynetpy.GraphUtils import subnet_given_leaves, _displayed_trees_with_probs
from phynetpy.IO import convert_newick

NET1_PHYLONET = "(((((t14:1.0,t75:1.0):0.8767426254184691,(t69:1.0,t114:1.0):2.0170864999696456):5.213391915869064,(t91:1.0)#H1:1.0::0.3011071643186969):0.8237808968694593,t133:1.0):2.054746072134383,(((t15:1.0,#H1:1.0::0.6988928356813031):1.481149764930965,(t72:1.0,t49:1.0):3.0976006419526203):1.2243080352496307,t68:1.0):0.12981149034576273);"

phynetpy_nwk = convert_newick(NET1_PHYLONET, standard="PhyNetPy")
print("PhyNetPy Newick:")
print(phynetpy_nwk)
print()

net = Network.from_newick(phynetpy_nwk)
leaves = sorted(n.label for n in net.get_leaves())
print(f"Leaves: {leaves}")
print()

# Pick a triplet involving the reticulation node: t91, t15, t133
triplet = ("t15", "t91", "t133")
print(f"=== Examining triplet {triplet} ===")

leaf_nodes = [net.has_node_named(t) for t in triplet]
print(f"Leaf nodes found: {[n.label if n else None for n in leaf_nodes]}")

subnet = subnet_given_leaves(net, leaf_nodes)
sub_leaves = sorted(n.label for n in subnet.get_leaves())
print(f"Subnet leaves: {sub_leaves}")

# Print subnet structure
print("\nSubnet edges:")
root = subnet.root()
print(f"  Root: {root.label}")

def print_tree(net, node, indent=2):
    children = net.get_children(node)
    for ch in children:
        edge = net.get_edge(node, ch)
        gamma_str = f" (gamma={edge.get_gamma()})" if edge.get_gamma() else ""
        retic_str = " [RETIC]" if ch.is_reticulation else ""
        print(f"{'  '*indent}{node.label} -> {ch.label}: len={edge.get_length()}{gamma_str}{retic_str}")
        print_tree(net, ch, indent+1)

print_tree(subnet, root)

# Get displayed trees
displayed = _displayed_trees_with_probs(subnet)
print(f"\nDisplayed trees: {len(displayed)}")
for i, (tree, weight) in enumerate(displayed):
    tree_leaves = sorted(n.label for n in tree.get_leaves())
    tree_root = tree.root()
    print(f"\n  Tree {i+1} (weight={weight:.6f}):")
    print(f"    Leaves: {tree_leaves}")
    print_tree(tree, tree_root, indent=2)
    
    # Compute coalescent triple probs
    X, Y, Z = sorted(tree_leaves)
    tp = _coalescent_triple_probs(tree, X, Y, Z)
    print(f"    Coal probs (P({X}{Y}|{Z}), P({X}{Z}|{Y}), P({Y}{Z}|{X})): {tp}")

# Combined subnet triple probs
probs = _subnet_triple_probs(subnet)
X, Y, Z = sorted(sub_leaves)
print(f"\nCombined triple probs for ({X},{Y},{Z}):")
print(f"  P({X}{Y}|{Z}) = {probs[0]:.10f}")
print(f"  P({X}{Z}|{Y}) = {probs[1]:.10f}")
print(f"  P({Y}{Z}|{X}) = {probs[2]:.10f}")
print(f"  Sum = {sum(probs):.10f}")

# Also check a non-reticulation triplet
print("\n\n=== Examining triplet (t14, t69, t72) — no reticulation involved ===")
triplet2 = ("t14", "t69", "t72")
leaf_nodes2 = [net.has_node_named(t) for t in triplet2]
subnet2 = subnet_given_leaves(net, leaf_nodes2)
print(f"Subnet leaves: {sorted(n.label for n in subnet2.get_leaves())}")
print("\nSubnet edges:")
root2 = subnet2.root()
print(f"  Root: {root2.label}")
print_tree(subnet2, root2)

displayed2 = _displayed_trees_with_probs(subnet2)
print(f"\nDisplayed trees: {len(displayed2)}")
for i, (tree, weight) in enumerate(displayed2):
    tree_leaves = sorted(n.label for n in tree.get_leaves())
    tree_root = tree.root()
    print(f"\n  Tree {i+1} (weight={weight:.6f}):")
    print_tree(tree, tree_root, indent=2)
    
    X, Y, Z = sorted(tree_leaves)
    tp = _coalescent_triple_probs(tree, X, Y, Z)
    print(f"    Coal probs: {tp}")

probs2 = _subnet_triple_probs(subnet2)
X, Y, Z = sorted(triplet2)
print(f"\nCombined triple probs for ({X},{Y},{Z}):")
print(f"  P({X}{Y}|{Z}) = {probs2[0]:.10f}")
print(f"  P({X}{Z}|{Y}) = {probs2[1]:.10f}")
print(f"  P({Y}{Z}|{X}) = {probs2[2]:.10f}")
print(f"  Sum = {sum(probs2):.10f}")
