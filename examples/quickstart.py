"""
PhyNetPy v0.3.0 — Quick-Start Sample Script

Demonstrates core I/O, network construction, graph utilities,
simulation, and newick conversion.
"""

#Mark's preferred style of importing
from phynetpy.Network import Network, Node, Edge
from phynetpy.GraphUtils import * #check docs for all that is included here
from phynetpy.IO import read_newick, write_newick, read_newick_file, write_nexus, convert_newick, detect_newick_standard
from phynetpy.BirthDeath import CBDP
from phynetpy.GTR import JC


##THIS ALSO WORKS NOW THOUGH
# from phynetpy import (
#     # Core data structures
#     Network, Node, Edge,
#     # I/O functions
#     read_newick, write_newick, read_newick_file,
#     write_nexus, convert_newick, detect_newick_standard,
#     # Graph utilities (star-imported from GraphUtils)
#     count_reticulations, is_tree, dominant_tree, ascii,
#     level, pairwise_leaf_distance, is_isomorphic,
#     get_all_clusters, ascii_extended,
#     # Simulation
#     CBDP,
#     # Substitution models
#     JC,
# )

# ─── 1. Parse a network from an extended Newick string ─────────────────────

newick_str = "((C:0.3,(B:0.2)#H0[&gamma=0.7]:0.1):0.5,(A:0.4,#H0:0.15):0.35)Root;"
net = read_newick(newick_str)

print("=" * 60)
print("1. Parsed network from extended Newick")
print("=" * 60)
print(f"   Nodes  : {[n.label for n in net.V()]}")
print(f"   Edges  : {len(net.E())}")
print(f"   Root   : {net.root().label}")
print(f"   Leaves : {[l.label for l in net.get_leaves()]}")
print()

# ─── 2. Inspect network properties with GraphUtils ────────────────────────

print("=" * 60)
print("2. Network properties (GraphUtils)")
print("=" * 60)
print(f"   Is tree?            : {is_tree(net)}")
print(f"   Reticulation count  : {count_reticulations(net)}")
print(f"   Network level       : {level(net)}")
print()

# ─── 3. ASCII visualisation ───────────────────────────────────────────────

print("=" * 60)
print("3. ASCII visualisation")
print("=" * 60)
print(ascii(net))
print()

# ─── 4. Extract the dominant (major) tree ─────────────────────────────────

dom = dominant_tree(net)

print("=" * 60)
print("4. Dominant tree (highest-gamma edges only)")
print("=" * 60)
print(f"   Is tree?  : {is_tree(dom)}")
print(f"   Newick    : {write_newick(dom)}")
print(ascii(dom))
print()

# ─── 5. Pairwise leaf distances ──────────────────────────────────────────

print("=" * 60)
print("5. Pairwise leaf distances (branch-length aware)")
print("=" * 60)
dist = pairwise_leaf_distance(net, use_branch_lengths=True)
for (u, v), d in sorted(dist.items(), key=lambda kv: kv[1]):
    u_lbl = u.label if hasattr(u, "label") else str(u)
    v_lbl = v.label if hasattr(v, "label") else str(v)
    print(f"   d({u_lbl}, {v_lbl}) = {d:.4f}")
print()

# ─── 6. Rooted clusters ──────────────────────────────────────────────────

print("=" * 60)
print("6. Rooted clusters")
print("=" * 60)
clusters = get_all_clusters(net)
for cl in sorted(clusters, key=len):
    labels = sorted(n.label for n in cl)
    print(f"   {{{', '.join(labels)}}}")
print()

# ─── 7. Simulate a tree with the Birth-Death process ─────────────────────

print("=" * 60)
print("7. Simulate a 6-taxon tree (Constant-rate Birth-Death)")
print("=" * 60)
sim = CBDP(gamma=2.0, mu=0.5, n=6)
sim_net = sim.generate_network()
print(f"   Nodes   : {len(sim_net.V())}")
print(f"   Leaves  : {[l.label for l in sim_net.get_leaves()]}")
print(f"   Newick  : {write_newick(sim_net)}")
print()

# ─── 8. Build a network from scratch ─────────────────────────────────────

print("=" * 60)
print("8. Build a network programmatically")
print("=" * 60)
custom = Network()

root = Node("Root", t=0.0)
i1 = Node("I1", t=0.3)
i2 = Node("I2", t=0.3)
hybrid = Node("#H0", is_reticulation=True, t=0.5)
a = Node("A", t=1.0)
b = Node("B", t=1.0)
c = Node("C", t=1.0)

custom.add_nodes(root, i1, i2, hybrid, a, b, c)
custom.add_edges([
    Edge(root, i1, length=0.3),
    Edge(root, i2, length=0.3),
    Edge(i1, hybrid, length=0.2, gamma=0.6),
    Edge(i2, hybrid, length=0.2, gamma=0.4),
    Edge(hybrid, a, length=0.5),
    Edge(i1, b, length=0.7),
    Edge(i2, c, length=0.7),
])

print(f"   Reticulations : {count_reticulations(custom)}")
print(f"   Newick        : {write_newick(custom)}")
print(ascii(custom))
print()

# ─── 9. Newick format conversion ─────────────────────────────────────────

print("=" * 60)
print("9. Newick format conversion (PhyNetPy <-> PhyloNet <-> BEAST)")
print("=" * 60)
phynetpy_nwk = write_newick(custom)
detected = detect_newick_standard(phynetpy_nwk)
print(f"   Original ({detected}):")
print(f"     {phynetpy_nwk}")

phylonet_nwk = convert_newick(phynetpy_nwk, standard="Phylonet")
print(f"   PhyloNet format:")
print(f"     {phylonet_nwk}")

beast_nwk = convert_newick(phynetpy_nwk, standard="Beast")
print(f"   BEAST format:")
print(f"     {beast_nwk}")
print()

# ─── 10. Write to a Nexus file ───────────────────────────────────────────

print("=" * 60)
print("10. Write networks to Nexus file")
print("=" * 60)
write_nexus([net, custom], "example_output.nex", tree_prefix="net")
print("   Written 2 networks to example_output.nex")
print()

# ─── 11. Substitution model ──────────────────────────────────────────────

print("=" * 60)
print("11. JC substitution model -- transition matrix at t=0.5")
print("=" * 60)
jc = JC()
P = jc.expt(0.5)
bases = ["A", "C", "G", "T"]
header = "       " + "     ".join(f"  {b}" for b in bases)
print(header)
for i, row_label in enumerate(bases):
    row_vals = "  ".join(f"{P[i][j]:7.4f}" for j in range(4))
    print(f"   {row_label}  {row_vals}")
print()

print("Done!")
