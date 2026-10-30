"""
Test suite for displayed-tree enumeration in phynetpy.GraphUtils.

Covers ``get_displayed_trees`` and the ``get_all_subtrees`` /
``count_displayed_trees`` functions built on the same enumeration, with an
emphasis on the reductions a displayed tree needs beyond deleting hybrid
edges:

    - internal nodes left childless are suppressed, so a displayed tree has
      exactly the leaf set of the network it came from;
    - suppression cascades, since an orphaned node's parent can be orphaned
      in turn;
    - reticulation flags are cleared, so every result is a tree;
    - choices that display the same topology collapse to one tree.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from phynetpy.IO import read_newick
from phynetpy.Network import Network, Node, Edge
from phynetpy.GraphUtils import (
    count_displayed_trees,
    count_reticulations,
    get_all_subtrees,
    get_displayed_trees,
    is_tree,
    network_clusters,
    softwired_cluster_distance,
)


# ===================================================================
# Helpers and Network Builders
# ===================================================================

EdgeSpec = Tuple[str, str, Optional[float]]


def _edge(src: str, dest: str, gamma: Optional[float] = None) -> EdgeSpec:
    return (src, dest, gamma)


def _build_network(
    node_labels: Sequence[str],
    edge_specs: Sequence[EdgeSpec],
    retic_labels: Optional[set[str]] = None,
) -> Network:
    net = Network()
    retic_set = set(retic_labels or [])

    nodes = {
        label: Node(label, is_reticulation=label in retic_set)
        for label in node_labels
    }
    net.add_nodes(*nodes.values())

    for src, dest, gamma in edge_specs:
        net.add_edges(Edge(nodes[src], nodes[dest], gamma=gamma))

    return net


def _leaf_labels(net: Network) -> set[str]:
    return {n.label for n in net.get_leaves()}


def _clusters(net: Network) -> frozenset[frozenset[str]]:
    """Topology fingerprint: the non-trivial leaf-label clusters of a tree."""
    return frozenset(network_clusters(net))


def _topologies(trees: Sequence[Network]) -> set[frozenset[frozenset[str]]]:
    return {_clusters(tree) for tree in trees}


# ── The network from the bug report ──
#
#   (t2,((t4,t6),(((t3)#H1,(t1)#H2),((t5,#H1),#H2))));
#
# Both #H1 and #H2 hang off the same tree node, and each has its second
# parent elsewhere.  When both inherit from that second parent, their shared
# parent keeps no children at all.
REPORTED_NEWICK = "(t2,((t4,t6),(((t3)#H1,(t1)#H2),((t5,#H1),#H2))));"


def build_reported_network() -> Network:
    return read_newick(REPORTED_NEWICK)


# ── Level-1 network, one reticulation, two displayed trees ──
#        Root
#       /    \
#     I1      I2
#    / \     / \
#   A   P1  P2  B
#        \  /
#        #H0
#         |
#         C
def build_level1_network() -> Network:
    labels = ["Root", "I1", "I2", "P1", "P2", "#H0", "A", "B", "C"]
    edges = [
        _edge("Root", "I1"), _edge("Root", "I2"),
        _edge("I1", "A"), _edge("I1", "P1"),
        _edge("I2", "P2"), _edge("I2", "B"),
        _edge("P1", "#H0", 0.25), _edge("P2", "#H0", 0.75),
        _edge("#H0", "C"),
    ]
    return _build_network(labels, edges, retic_labels={"#H0"})


# ── Reticulation with two children ──
#
# Discarding one of #H0's in-edges strips its other parent of its only
# child, and #H0 itself survives the degree-2 contraction because it has two
# children -- so its reticulation flag has to be cleared explicitly.
def build_two_child_retic_network() -> Network:
    labels = ["Root", "I1", "I2", "P1", "P2", "#H0", "A", "B", "C", "D"]
    edges = [
        _edge("Root", "I1"), _edge("Root", "I2"),
        _edge("I1", "A"), _edge("I1", "P1"),
        _edge("I2", "P2"), _edge("I2", "B"),
        _edge("P1", "#H0", 0.4), _edge("P2", "#H0", 0.6),
        _edge("#H0", "C"), _edge("#H0", "D"),
    ]
    return _build_network(labels, edges, retic_labels={"#H0"})


# ── Cascading suppression ──
#
# ``Gone``'s only children are the two reticulations, ``Chain``'s only child
# is ``Gone``, and ``Stem``'s only child is ``Chain``.  When both
# reticulations inherit from the other side of the network, suppressing
# ``Gone`` orphans ``Chain``, which orphans ``Stem``, which leaves the root
# with a single child.
def build_cascade_network() -> Network:
    labels = [
        "Root", "Stem", "Chain", "Gone", "Other", "P1", "P2",
        "#H0", "#H1", "A", "B", "C", "D",
    ]
    edges = [
        _edge("Root", "Stem"), _edge("Root", "Other"),
        _edge("Stem", "Chain"), _edge("Chain", "Gone"),
        _edge("Gone", "#H0", 0.5), _edge("Gone", "#H1", 0.5),
        _edge("Other", "P1"), _edge("Other", "P2"),
        _edge("P1", "A"), _edge("P1", "#H0", 0.5),
        _edge("P2", "B"), _edge("P2", "#H1", 0.5),
        _edge("#H0", "C"), _edge("#H1", "D"),
    ]
    return _build_network(labels, edges, retic_labels={"#H0", "#H1"})


ALL_BUILDERS = [
    build_reported_network,
    build_level1_network,
    build_two_child_retic_network,
    build_cascade_network,
]


# ===================================================================
# The reported network
# ===================================================================

class TestReportedNetwork:

    def test_no_spurious_leaf(self):
        """
        The node whose only children are #H1 and #H2 must not survive as a
        leaf when both reticulations inherit from their other parent.
        """
        net = build_reported_network()
        expected = _leaf_labels(net)
        assert expected == {"t1", "t2", "t3", "t4", "t5", "t6"}

        for tree in get_displayed_trees(net):
            assert _leaf_labels(tree) == expected, (
                f"displayed tree has the wrong leaf set: {tree.newick()}"
            )

    def test_three_distinct_displayed_trees(self):
        """
        Four hybrid-edge choices, but the two that discard #H1's and #H2's
        shared parent both display the same tree.
        """
        net = build_reported_network()
        assert len(get_all_subtrees(net)) == 4
        assert len(get_displayed_trees(net)) == 3
        assert len(_topologies(get_displayed_trees(net))) == 3

    def test_expected_topologies(self):
        """The three displayed trees, written out by hand."""
        net = build_reported_network()
        expected = _topologies([
            read_newick("(t2,((t4,t6),((t3,t1),t5)));"),
            read_newick("(t2,((t4,t6),(t3,(t5,t1))));"),
            read_newick("(t2,((t4,t6),(t1,(t5,t3))));"),
        ])
        assert _topologies(get_displayed_trees(net)) == expected

    def test_count_exact_is_smaller_than_upper_bound(self):
        net = build_reported_network()
        assert count_displayed_trees(net) == 4
        assert count_displayed_trees(net, exact=True) == 3


# ===================================================================
# Invariants that hold of every displayed tree
# ===================================================================

class TestDisplayedTreeInvariants:

    def test_leaf_set_is_preserved(self):
        """Deleting hybrid edges never disconnects a leaf."""
        for builder in ALL_BUILDERS:
            net = builder()
            expected = _leaf_labels(net)
            for tree in get_displayed_trees(net):
                assert _leaf_labels(tree) == expected, (
                    f"{builder.__name__}: {sorted(_leaf_labels(tree))} "
                    f"!= {sorted(expected)}"
                )

    def test_results_are_trees(self):
        """No result may keep a reticulation flag or an in-degree above 1."""
        for builder in ALL_BUILDERS:
            net = builder()
            for tree in get_displayed_trees(net):
                assert is_tree(tree), f"{builder.__name__}: {tree.newick()}"
                assert count_reticulations(tree) == 0
                for node in tree.V():
                    assert tree.in_degree(node) <= 1

    def test_no_degree_two_nodes(self):
        """Suppression happens before contraction, so no chains survive it."""
        for builder in ALL_BUILDERS:
            net = builder()
            for tree in get_displayed_trees(net):
                for node in tree.V():
                    assert not (tree.in_degree(node) == 1
                                and tree.out_degree(node) == 1), (
                        f"{builder.__name__}: {node.label} was not contracted"
                    )

    def test_results_are_copies(self):
        """A displayed tree must not share nodes with the network."""
        for builder in ALL_BUILDERS:
            net = builder()
            original = {id(n) for n in net.V()}
            for tree in get_displayed_trees(net):
                for node in tree.V():
                    assert id(node) not in original

    def test_network_is_not_mutated(self):
        for builder in ALL_BUILDERS:
            net = builder()
            before = (len(net.V()), len(net.E()), count_reticulations(net))
            get_displayed_trees(net)
            after = (len(net.V()), len(net.E()), count_reticulations(net))
            assert before == after, builder.__name__

    def test_unique_is_a_subset_of_all_choices(self):
        for builder in ALL_BUILDERS:
            net = builder()
            choices = get_displayed_trees(net, unique=False)
            distinct = get_displayed_trees(net, unique=True)
            assert len(distinct) <= len(choices)
            assert _topologies(distinct) == _topologies(choices)

    def test_choice_count_matches_upper_bound(self):
        for builder in ALL_BUILDERS:
            net = builder()
            assert (len(get_displayed_trees(net, unique=False))
                    == count_displayed_trees(net)), builder.__name__

    def test_deterministic_across_calls(self):
        """
        Reticulations and their in-edges are enumerated by label, so the trees
        come back in the same order every time.
        """
        for builder in ALL_BUILDERS:
            net = builder()
            first = [_clusters(t) for t in get_displayed_trees(net)]
            second = [_clusters(t) for t in get_displayed_trees(net)]
            assert first == second, builder.__name__


# ===================================================================
# Individual topologies
# ===================================================================

class TestSpecificTopologies:

    def test_tree_input_displays_itself(self):
        tree = read_newick("((A,B),(C,D));")
        displayed = get_displayed_trees(tree)
        assert len(displayed) == 1
        assert _clusters(displayed[0]) == _clusters(tree)

    def test_level1_displays_both_choices(self):
        net = build_level1_network()
        displayed = get_displayed_trees(net)
        assert len(displayed) == 2
        assert _topologies(displayed) == _topologies([
            read_newick("((A,C),B);"),
            read_newick("(A,(C,B));"),
        ])

    def test_two_child_retic_keeps_both_children(self):
        """
        #H0 survives contraction because it has two children, so the flag
        has to be cleared rather than relied on disappearing.
        """
        net = build_two_child_retic_network()
        displayed = get_displayed_trees(net)
        assert len(displayed) == 2
        assert _topologies(displayed) == _topologies([
            read_newick("((A,(C,D)),B);"),
            read_newick("(A,((C,D),B));"),
        ])

    def test_cascade_displays_four_trees(self):
        net = build_cascade_network()
        assert _topologies(get_displayed_trees(net)) == _topologies([
            read_newick("((C,D),(A,B));"),
            read_newick("(C,(A,(B,D)));"),
            read_newick("(D,((A,C),B));"),
            read_newick("((A,C),(B,D));"),
        ])

    def test_cascading_suppression(self):
        """
        When both reticulations inherit from the other side of the network,
        the Stem -> Chain -> Gone chain has to be removed in full: a single
        suppression pass would leave ``Chain`` behind as a spurious leaf.
        """
        net = build_cascade_network()
        expected = _clusters(read_newick("((A,C),(B,D));"))

        stripped = [tree for tree in get_displayed_trees(net)
                    if _clusters(tree) == expected]
        assert len(stripped) == 1

        labels = {n.label for n in stripped[0].V()}
        assert labels.isdisjoint({"Stem", "Chain", "Gone"}), (
            f"suppression did not cascade: {sorted(labels)}"
        )


# ===================================================================
# Consumers of the enumeration
# ===================================================================

class TestDownstreamConsumers:

    def test_softwired_clusters_exclude_spurious_nodes(self):
        """
        Softwired clusters are unioned over displayed trees, so a suppressed
        internal node must not leak in as a taxon.
        """
        net = build_reported_network()
        taxa = _leaf_labels(net)
        for tree in get_displayed_trees(net):
            for cluster in network_clusters(tree):
                assert cluster <= taxa, f"cluster {sorted(cluster)} escapes {taxa}"

    def test_softwired_distance_to_a_displayed_tree(self):
        """
        Every cluster of a displayed tree is a softwired cluster of the
        network, so the symmetric difference only counts the network's extras.
        """
        net = build_reported_network()
        for tree in get_displayed_trees(net):
            assert softwired_cluster_distance(net, tree) >= 0
        assert softwired_cluster_distance(net, net) == 0
