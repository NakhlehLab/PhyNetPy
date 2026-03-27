"""
PhyNetPy v0.3.0 -- Tree of Blobs Analysis

Demonstrates blob decomposition of a phylogenetic network:
  - Parse a multi-reticulation network
  - Identify biconnected components (blobs)
  - Decompose into individual blob sub-networks
  - Compute network level, bridges, and articulation points
  - Visualise the full network and each blob
"""

from phynetpy.Network import Network, Node, Edge
from phynetpy.GraphUtils import (
    blobs, tree_of_blobs, level, count_reticulations,
    bridges_and_articulations, is_tree, ascii,
    count_displayed_trees, get_all_clusters,
)
from phynetpy.IO import read_newick, write_newick


# ---- 1. Build a level-2 network with two reticulation events ---------------
#
#  Topology (rooted, directed):
#
#         Root
#        /    \
#      I1      I2
#     / \     / \
#    A  #H0--+   I3
#        |      / \
#        B    #H1  C
#              |
#              D
#
#  #H0 receives edges from I1 (gamma=0.6) and I2 (gamma=0.4)
#  #H1 receives edges from I2 (gamma=0.7) and I3 (gamma=0.3)
#
#  This creates two overlapping reticulation cycles => level-2.

net = Network()

root = Node("Root", t=0.0)
i1   = Node("I1", t=0.5)
i2   = Node("I2", t=0.5)
i3   = Node("I3", t=1.0)
h0   = Node("#H0", is_reticulation=True, t=1.0)
h1   = Node("#H1", is_reticulation=True, t=1.5)
a    = Node("A", t=2.0)
b    = Node("B", t=2.0)
c    = Node("C", t=2.0)
d    = Node("D", t=2.0)

net.add_nodes(root, i1, i2, i3, h0, h1, a, b, c, d)
net.add_edges([
    Edge(root, i1, length=0.5),
    Edge(root, i2, length=0.5),
    Edge(i1, a, length=1.5),
    Edge(i1, h0, length=0.5, gamma=0.6),
    Edge(i2, h0, length=0.5, gamma=0.4),
    Edge(h0, b, length=1.0),
    Edge(i2, i3, length=0.5),
    Edge(i3, h1, length=0.5, gamma=0.3),
    Edge(i2, h1, length=1.0, gamma=0.7),
    Edge(h1, d, length=0.5),
    Edge(i3, c, length=1.0),
])

print("=" * 60)
print("1. Full network")
print("=" * 60)
print(f"   Newick  : {write_newick(net)}")
print(f"   Nodes   : {len(net.V())}")
print(f"   Edges   : {len(net.E())}")
print(f"   Leaves  : {sorted(l.label for l in net.get_leaves())}")
print(f"   Retics  : {count_reticulations(net)}")
print(f"   Level   : {level(net)}")
print(f"   Is tree : {is_tree(net)}")
print(f"   Displayed trees (upper bound): {count_displayed_trees(net)}")
print()
print(ascii(net))
print()


# ---- 2. Identify biconnected components (blobs) ----------------------------

print("=" * 60)
print("2. Biconnected components (blobs)")
print("=" * 60)

blob_node_sets = blobs(net)
print(f"   Found {len(blob_node_sets)} blobs:\n")

for i, comp in enumerate(blob_node_sets):
    labels = sorted(n.label for n in comp)
    retics = sum(1 for n in comp if n.is_reticulation())
    print(f"   Blob {i + 1}: {{{', '.join(labels)}}}")
    print(f"           size = {len(comp)} nodes, {retics} reticulation(s)")
print()


# ---- 3. Decompose into sub-networks (tree of blobs) -----------------------

print("=" * 60)
print("3. Tree of blobs decomposition")
print("=" * 60)

blob_nets = tree_of_blobs(net)
print(f"   Decomposed into {len(blob_nets)} blob sub-networks:\n")

for i, blob_net in enumerate(blob_nets):
    leaves = sorted(l.label for l in blob_net.get_leaves())
    retics = count_reticulations(blob_net)
    blob_level = level(blob_net)
    print(f"   --- Blob {i + 1} ---")
    print(f"   Nodes  : {sorted(n.label for n in blob_net.V())}")
    print(f"   Edges  : {len(blob_net.E())}")
    print(f"   Leaves : {leaves}")
    print(f"   Retics : {retics}")
    print(f"   Level  : {blob_level}")
    if len(blob_net.E()) > 0:
        print(ascii(blob_net))
    print()


# ---- 4. Bridges and articulation points ------------------------------------

print("=" * 60)
print("4. Bridges and articulation points")
print("=" * 60)

bridge_list, art_list = bridges_and_articulations(net)

print(f"   Bridges (cut edges): {len(bridge_list)}")
for u, v in bridge_list:
    print(f"     {u} -- {v}")

print(f"\n   Articulation points (cut vertices): {len(art_list)}")
for a_node in sorted(art_list):
    print(f"     {a_node}")
print()
print("   Bridges connect blobs to each other; removing a bridge")
print("   disconnects the network. Articulation points are nodes")
print("   shared between two or more blobs.")
print()


# ---- 5. Rooted clusters per blob -------------------------------------------

print("=" * 60)
print("5. Rooted clusters for the full network")
print("=" * 60)

clusters = get_all_clusters(net)
for cl in sorted(clusters, key=lambda c: (len(c), sorted(n.label for n in c))):
    labels = sorted(n.label for n in cl)
    print(f"   {{{', '.join(labels)}}}")
print()


# ---- 6. Summary ------------------------------------------------------------

print("=" * 60)
print("6. Summary")
print("=" * 60)
print(f"   A level-{level(net)} network with {count_reticulations(net)} "
      f"reticulations decomposes into {len(blob_nets)} blobs.")
print(f"   {len(bridge_list)} bridge(s) connect the blobs,")
print(f"   and {len(art_list)} articulation point(s) sit at blob boundaries.")
print()
print("Done!")
