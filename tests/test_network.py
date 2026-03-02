"""
Comprehensive test suite for the Network module (phynetpy.Network).

Covers the core data structures and algorithms:

    **Node**  construction, attributes, comparison, copying, reticulation flag.
    **Edge**  construction, src/dest, gamma, length, weight, tag, copy.
    **NodeSet**  add / remove / in/out-degree / edge bookkeeping.
    **EdgeSet**  add / remove / get / contains.
    **Network**  construction, add/remove nodes/edges, root/roots, get_leaves,
        get_parents, get_children, get_branches, clean, mrca, leaf_descendants,
        leaf_descendants_all, newick round-trip, is_acyclic, bfs_dfs, subnet,
        copy, to_networkx, is_isomorphic, get_subtree_at, dist_from_root,
        topological_order, distance_from_root, set_node_times_from_root,
        set_node_times_by_edge_count, edges_downstream_of_node,
        edges_upstream_of_node, subgenome_count.

Each test builder constructs a small network topology from scratch so
that tests are self-contained and deterministic.
"""

import pytest
from typing import Iterable, Optional, Sequence, Tuple

from phynetpy.Network import (
    Network,
    Node,
    Edge,
    UEdge,
    EdgeSet,
    NodeSet,
    NetworkError,
    EdgeError,
)
from phynetpy.GraphUtils import is_tree, level


# ===================================================================
# Helpers
# ===================================================================

EdgeSpec = Tuple[str, str, Optional[float]]


def _leaf_labels(prefix: str, count: int) -> list[str]:
    """Generate deterministic leaf labels with the provided prefix.

    Args:
        prefix: Short string prepended to each label (e.g. ``"simple"``).
        count: Number of labels to generate.

    Returns:
        A list of labels ``["{prefix}_leaf_1", ..., "{prefix}_leaf_{count}"]``.

    Raises:
        ValueError: If *count* is not positive.
    """
    if count <= 0:
        raise ValueError("Leaf count must be positive.")
    return [f"{prefix}_leaf_{i}" for i in range(1, count + 1)]


def _edge(src: str, dest: str, gamma: Optional[float] = None) -> EdgeSpec:
    """Small helper for describing an edge specification as a tuple."""
    return (src, dest, gamma)


def _build_network(
    node_labels: Sequence[str],
    edge_specs: Sequence[EdgeSpec],
    retic_labels: Optional[Iterable[str]] = None,
) -> Network:
    """Materialize a :class:`Network` from node labels and directed edge specs.

    Nodes whose labels appear in *retic_labels* are created with
    ``is_reticulation=True``.

    Args:
        node_labels: Ordered list of unique node label strings.
        edge_specs: Edge descriptions ``(src_label, dest_label, gamma|None)``.
        retic_labels: Optional iterable of labels to mark as reticulations.

    Returns:
        A fully constructed Network.

    Raises:
        KeyError: If an edge spec references an undefined node.
    """
    net = Network()
    retic_set = set(retic_labels or [])

    ordered_labels: list[str] = []
    seen: set[str] = set()
    for label in list(node_labels) + list(retic_set):
        if label not in seen:
            ordered_labels.append(label)
            seen.add(label)

    nodes: dict[str, Node] = {}
    for label in ordered_labels:
        nodes[label] = Node(label, is_reticulation=label in retic_set)

    net.add_nodes(*nodes.values())

    for spec in edge_specs:
        src, dest, gamma = spec
        if src not in nodes or dest not in nodes:
            raise KeyError(f"Edge references undefined nodes: {spec}")
        net.add_edges(Edge(nodes[src], nodes[dest], gamma=gamma))

    return net


def _node_by_label(net: Network, label: str) -> Node:
    """Look up a node by label, raising if absent."""
    node = net.has_node_named(label)
    assert node is not None, f"Node '{label}' not found"
    return node


def _leaf_label_set(net: Network) -> set[str]:
    """Return the set of leaf-node labels for a network."""
    return {n.label for n in net.get_leaves()}


# ===================================================================
# Reusable Network Topologies
# ===================================================================

def build_simple_tree() -> Network:
    """Strictly binary tree with 4 leaves, no reticulations::

            Root
           /    \\
         I1      I2
        /  \\   /  \\
       A    B  C    D
    """
    labels = ["Root", "I1", "I2", "A", "B", "C", "D"]
    edges = [
        _edge("Root", "I1"),
        _edge("Root", "I2"),
        _edge("I1", "A"),
        _edge("I1", "B"),
        _edge("I2", "C"),
        _edge("I2", "D"),
    ]
    net = _build_network(labels, edges)
    assert is_tree(net)
    return net


def build_level1_network() -> Network:
    """Level-1 network with one reticulation (#H0)::

            Root
           /    \\
         L1      R1
        /  \\   /  \\
       A   P1  P2   B
            \\  /
            #H0
              |
              C
    """
    labels = ["Root", "L1", "R1", "P1", "P2", "#H0", "A", "B", "C"]
    edges = [
        _edge("Root", "L1"),
        _edge("Root", "R1"),
        _edge("L1", "A"),
        _edge("L1", "P1"),
        _edge("R1", "P2"),
        _edge("R1", "B"),
        _edge("P1", "#H0", gamma=0.4),
        _edge("P2", "#H0", gamma=0.6),
        _edge("#H0", "C"),
    ]
    return _build_network(labels, edges, retic_labels={"#H0"})


def build_level2_network() -> Network:
    """Level-2 network with two reticulations in the same blob::

              Root
             /    \\
            I1     I2
           / \\   / \\
          A  P1  P2  B
              \\ /
              #H0
             /   \\
           P3     D
            |
           #H1
            |
            C

    (Plus an extra edge from I1 -> #H1 at gamma 0.7.)
    """
    labels = [
        "Root", "I1", "I2", "P1", "P2", "P3",
        "#H0", "#H1", "A", "B", "C", "D",
    ]
    edges = [
        _edge("Root", "I1"),
        _edge("Root", "I2"),
        _edge("I1", "A"),
        _edge("I1", "P1"),
        _edge("I2", "P2"),
        _edge("I2", "B"),
        _edge("P1", "#H0", gamma=0.5),
        _edge("P2", "#H0", gamma=0.5),
        _edge("#H0", "P3"),
        _edge("#H0", "D"),
        _edge("P3", "#H1", gamma=0.3),
        _edge("I1", "#H1", gamma=0.7),
        _edge("#H1", "C"),
    ]
    return _build_network(labels, edges, retic_labels={"#H0", "#H1"})


def build_small_network_with_lengths() -> Network:
    """Small tree where every edge has an explicit branch length::

            Root
           /    \\
         I1      I2
        /  \\   /  \\
       A    B  C    D

    Edge lengths: Root->I1 = 1.0, Root->I2 = 1.0,
                  I1->A = 0.5, I1->B = 0.5,
                  I2->C = 0.7, I2->D = 0.3
    """
    root = Node("Root")
    i1 = Node("I1")
    i2 = Node("I2")
    a = Node("A")
    b = Node("B")
    c = Node("C")
    d = Node("D")

    net = Network()
    net.add_nodes(root, i1, i2, a, b, c, d)
    net.add_edges(Edge(root, i1, length=1.0))
    net.add_edges(Edge(root, i2, length=1.0))
    net.add_edges(Edge(i1, a, length=0.5))
    net.add_edges(Edge(i1, b, length=0.5))
    net.add_edges(Edge(i2, c, length=0.7))
    net.add_edges(Edge(i2, d, length=0.3))
    return net


# ===================================================================
# Node Tests
# ===================================================================

class TestNode:
    """Tests for the Node class."""

    def test_basic_construction(self):
        """Node should store its label and default time correctly."""
        n = Node("A")
        assert n.label == "A"
        assert n.is_reticulation() is False

    def test_reticulation_flag(self):
        """is_reticulation should reflect the constructor flag."""
        n = Node("#H0", is_reticulation=True)
        assert n.is_reticulation() is True

    def test_set_is_reticulation(self):
        """set_is_reticulation should toggle the flag."""
        n = Node("X")
        assert n.is_reticulation() is False
        n.set_is_reticulation(True)
        assert n.is_reticulation() is True

    def test_set_and_get_name(self):
        """set_name should change the label."""
        n = Node("Old")
        n.set_name("New")
        assert n.label == "New"

    def test_time_accessors(self):
        """get_time / set_time should round-trip correctly."""
        n = Node("T")
        n.set_time(3.14)
        assert n.get_time() == pytest.approx(3.14)

    def test_attributes(self):
        """add_attribute / attribute_value should store arbitrary metadata."""
        n = Node("N")
        n.add_attribute("color", "red")
        assert n.attribute_value("color") == "red"

    def test_get_set_attributes(self):
        """get_attributes / set_attributes should expose the full dict."""
        n = Node("N")
        n.add_attribute("x", 1)
        attrs = n.get_attributes()
        assert "x" in attrs
        n.set_attributes({"y": 2})
        assert n.attribute_value("y") == 2

    def test_copy(self):
        """copy() should produce an independent node with the same label."""
        original = Node("A", is_reticulation=True)
        original.set_time(5.0)
        clone = original.copy()

        assert clone.label == "A"
        assert clone.is_reticulation() is True
        assert clone.get_time() == pytest.approx(5.0)
        assert clone is not original

    def test_equality_and_hash(self):
        """Two distinct Node objects with the same label should not compare
        equal by default (identity-based equality)."""
        n1 = Node("A")
        n2 = Node("A")
        # Node equality is identity-based
        assert n1 == n1
        # Hash should be stable
        assert hash(n1) == hash(n1)

    def test_comparison_operators(self):
        """Comparison uses (_time, label) tuple ordering."""
        a = Node("A")
        a.set_time(1.0)
        b = Node("B")
        b.set_time(2.0)
        assert a < b
        assert b > a
        assert a <= b
        assert b >= a

    def test_str(self):
        """__str__ should return a non-empty string representation."""
        n = Node("Test")
        assert len(str(n)) > 0

    def test_to_string(self):
        """to_string should return a human-readable summary."""
        n = Node("Test")
        assert "Test" in n.to_string()


# ===================================================================
# Edge Tests
# ===================================================================

class TestEdge:
    """Tests for the directed Edge class."""

    def test_basic_construction(self):
        """Edge should expose src and dest."""
        a, b = Node("A"), Node("B")
        e = Edge(a, b)
        assert e.src is a
        assert e.dest is b

    def test_length_accessors(self):
        """set_length / get_length should round-trip."""
        a, b = Node("A"), Node("B")
        e = Edge(a, b, length=2.5)
        assert e.get_length() == pytest.approx(2.5)
        e.set_length(4.0)
        assert e.get_length() == pytest.approx(4.0)

    def test_gamma_accessors(self):
        """set_gamma / get_gamma should round-trip."""
        a, b = Node("A"), Node("B")
        e = Edge(a, b, gamma=0.7)
        assert e.get_gamma() == pytest.approx(0.7)
        e.set_gamma(0.3)
        assert e.get_gamma() == pytest.approx(0.3)

    def test_weight_accessors(self):
        """set_weight / get_weight should round-trip."""
        a, b = Node("A"), Node("B")
        e = Edge(a, b)
        e.set_weight(9.9)
        assert e.get_weight() == pytest.approx(9.9)

    def test_tag_accessors(self):
        """set_tag / get_tag should round-trip."""
        a, b = Node("A"), Node("B")
        e = Edge(a, b)
        e.set_tag("my_tag")
        assert e.get_tag() == "my_tag"

    def test_to_names(self):
        """to_names should return (src.label, dest.label)."""
        a, b = Node("Alpha"), Node("Beta")
        e = Edge(a, b)
        assert e.to_names() == ("Alpha", "Beta")

    def test_copy(self):
        """copy() should produce a new edge between the given nodes."""
        a, b = Node("A"), Node("B")
        e = Edge(a, b, length=1.0, gamma=0.5)
        c, d = Node("C"), Node("D")
        clone = e.copy(c, d)

        assert clone.src is c
        assert clone.dest is d
        assert clone.get_length() == pytest.approx(1.0)
        assert clone.get_gamma() == pytest.approx(0.5)
        assert clone is not e


# ===================================================================
# Network Construction & Basic Queries
# ===================================================================

class TestNetworkConstruction:
    """Tests for building networks and basic V/E/root/leaves queries."""

    def test_empty_network(self):
        """An empty network should have no nodes or edges."""
        net = Network()
        assert len(net.V()) == 0
        assert len(net.E()) == 0

    def test_add_nodes(self):
        """add_nodes should register nodes in V."""
        net = Network()
        a, b = Node("A"), Node("B")
        net.add_nodes(a, b)
        assert len(net.V()) == 2
        assert a in net
        assert b in net

    def test_add_nodes_from_list(self):
        """add_nodes should also accept a plain list."""
        net = Network()
        nodes = [Node("A"), Node("B"), Node("C")]
        net.add_nodes(nodes)
        assert len(net.V()) == 3

    def test_add_edges(self):
        """add_edges should register edges in E and update adjacency."""
        net = Network()
        a, b = Node("A"), Node("B")
        net.add_nodes(a, b)
        e = Edge(a, b)
        net.add_edges(e)

        assert len(net.E()) == 1
        assert e in net

    def test_add_edges_from_list(self):
        """add_edges should also accept a list of edges."""
        net = Network()
        a, b, c = Node("A"), Node("B"), Node("C")
        net.add_nodes(a, b, c)
        e1 = Edge(a, b)
        e2 = Edge(a, c)
        net.add_edges([e1, e2])
        assert len(net.E()) == 2

    def test_simple_tree_root_and_leaves(self):
        """A simple 4-leaf tree should have one root and four leaves."""
        net = build_simple_tree()
        root = net.root()
        assert root.label == "Root"
        assert len(net.get_leaves()) == 4
        assert _leaf_label_set(net) == {"A", "B", "C", "D"}

    def test_roots_single(self):
        """roots() on a simple tree should return a single-element list."""
        net = build_simple_tree()
        assert len(net.roots()) == 1

    def test_v_and_e_counts(self):
        """V and E counts should match the network topology."""
        net = build_simple_tree()
        # 7 nodes: Root, I1, I2, A, B, C, D
        assert len(net.V()) == 7
        # 6 edges
        assert len(net.E()) == 6

    def test_contains_node(self):
        """'in' operator should work for nodes."""
        net = build_simple_tree()
        root = net.root()
        assert root in net
        orphan = Node("Orphan")
        assert orphan not in net

    def test_contains_edge(self):
        """'in' operator should work for edges."""
        net = build_simple_tree()
        edges = net.E()
        assert edges[0] in net

    def test_has_node_named(self):
        """has_node_named should find existing nodes and return None for missing."""
        net = build_simple_tree()
        assert net.has_node_named("Root") is not None
        assert net.has_node_named("I1") is not None
        assert net.has_node_named("NONEXISTENT") is None

    def test_add_uid_node(self):
        """add_uid_node should add a node with a unique auto-generated name."""
        net = Network()
        n = net.add_uid_node()
        assert n in net
        assert n.label.startswith("UID_")


# ===================================================================
# Degree & Adjacency
# ===================================================================

class TestDegreeAndAdjacency:
    """Tests for in/out-degree, in/out-edges, parents, and children."""

    def test_root_degrees(self):
        """The root of a tree should have in-degree 0 and out-degree 2."""
        net = build_simple_tree()
        root = net.root()
        assert net.in_degree(root) == 0
        assert net.out_degree(root) == 2

    def test_leaf_degrees(self):
        """Leaf nodes should have in-degree 1 and out-degree 0."""
        net = build_simple_tree()
        for leaf in net.get_leaves():
            assert net.out_degree(leaf) == 0
            assert net.in_degree(leaf) == 1

    def test_internal_degrees(self):
        """Internal tree nodes should have in-degree 1 and out-degree 2."""
        net = build_simple_tree()
        i1 = _node_by_label(net, "I1")
        assert net.in_degree(i1) == 1
        assert net.out_degree(i1) == 2

    def test_reticulation_in_degree(self):
        """A reticulation node should have in-degree >= 2."""
        net = build_level1_network()
        h0 = _node_by_label(net, "#H0")
        assert net.in_degree(h0) == 2

    def test_in_edges(self):
        """in_edges should return edges whose dest is the given node."""
        net = build_simple_tree()
        i1 = _node_by_label(net, "I1")
        in_edges = net.in_edges(i1)
        assert len(in_edges) == 1
        assert all(e.dest == i1 for e in in_edges)

    def test_out_edges(self):
        """out_edges should return edges whose src is the given node."""
        net = build_simple_tree()
        root = net.root()
        out_edges = net.out_edges(root)
        assert len(out_edges) == 2
        assert all(e.src == root for e in out_edges)

    def test_get_parents(self):
        """get_parents should return the parent node(s) of a node."""
        net = build_simple_tree()
        a = _node_by_label(net, "A")
        parents = net.get_parents(a)
        assert len(parents) == 1
        assert parents[0].label == "I1"

    def test_get_parents_reticulation(self):
        """A reticulation node should have two parents."""
        net = build_level1_network()
        h0 = _node_by_label(net, "#H0")
        parents = net.get_parents(h0)
        assert len(parents) == 2
        parent_labels = {p.label for p in parents}
        assert parent_labels == {"P1", "P2"}

    def test_get_children(self):
        """get_children should return child node(s)."""
        net = build_simple_tree()
        root = net.root()
        children = net.get_children(root)
        assert len(children) == 2
        child_labels = {c.label for c in children}
        assert child_labels == {"I1", "I2"}

    def test_get_branches(self):
        """get_branches should return dicts with parent and child branch lists."""
        net = build_simple_tree()
        i1 = _node_by_label(net, "I1")
        branches = net.get_branches(i1)
        assert "parent_branches" in branches
        assert "child_branches" in branches
        assert len(branches["parent_branches"]) == 1
        assert len(branches["child_branches"]) == 2


# ===================================================================
# Edge Retrieval
# ===================================================================

class TestEdgeRetrieval:
    """Tests for get_edge and EdgeSet.get lookups."""

    def test_get_edge_basic(self):
        """get_edge should retrieve the edge between two nodes."""
        net = build_simple_tree()
        root = net.root()
        i1 = _node_by_label(net, "I1")
        e = net.get_edge(root, i1)
        assert e.src is root
        assert e.dest is i1

    def test_get_edge_with_gamma(self):
        """get_edge with gamma should disambiguate bubble edges (reticulations)."""
        net = build_level1_network()
        p1 = _node_by_label(net, "P1")
        h0 = _node_by_label(net, "#H0")
        e = net.get_edge(p1, h0)
        assert e.get_gamma() == pytest.approx(0.4)


# ===================================================================
# Node & Edge Removal
# ===================================================================

class TestRemoval:
    """Tests for remove_nodes and remove_edge."""

    def test_remove_leaf_node(self):
        """Removing a leaf should remove it and its incident edge."""
        net = build_simple_tree()
        a = _node_by_label(net, "A")
        edge_count_before = len(net.E())
        net.remove_nodes(a)

        assert a not in net
        assert len(net.E()) == edge_count_before - 1

    def test_remove_edge(self):
        """Removing an edge should not remove the endpoint nodes."""
        net = build_simple_tree()
        root = net.root()
        i1 = _node_by_label(net, "I1")
        e = net.get_edge(root, i1)
        net.remove_edge(e)

        assert root in net
        assert i1 in net
        assert len(net.E()) == 5  # one edge removed

    def test_remove_edge_reclassifies_leaf(self):
        """After removing the only child edge, the source should become a leaf."""
        net = Network()
        a, b = Node("A"), Node("B")
        net.add_nodes(a, b)
        e = Edge(a, b)
        net.add_edges(e)

        # Before removal: a is root, b is leaf
        assert net.out_degree(a) == 1
        assert net.out_degree(b) == 0

        net.remove_edge(e)
        # After removal: a now has out-degree 0 (leaf-like)
        assert net.out_degree(a) == 0

    def test_remove_internal_node_prunes_edges(self):
        """Removing an internal node should prune all incident edges."""
        net = build_simple_tree()
        i1 = _node_by_label(net, "I1")
        # I1 has 1 in-edge + 2 out-edges = 3 edges to remove
        net.remove_nodes(i1)
        assert i1 not in net
        # A and B should still exist as disconnected nodes
        assert _node_by_label(net, "A") is not None


# ===================================================================
# update_node_name
# ===================================================================

class TestUpdateNodeName:
    """Tests for Graph.update_node_name."""

    def test_rename_node(self):
        """update_node_name should change the label and be findable by new name."""
        net = build_simple_tree()
        a = _node_by_label(net, "A")
        net.update_node_name(a, "Alpha")
        assert a.label == "Alpha"
        assert net.has_node_named("Alpha") is a
        assert net.has_node_named("A") is None


# ===================================================================
# Root & Leaf Classification
# ===================================================================

class TestRootLeafClassification:
    """Tests for dynamic root/leaf bookkeeping during mutations."""

    def test_new_node_is_root_and_leaf(self):
        """An isolated node added to a network is both a root (in-deg 0)
        and has out-degree 0, but get_leaves only returns connected leaves."""
        net = Network()
        n = Node("Orphan")
        net.add_nodes(n)
        assert net.in_degree(n) == 0
        assert net.out_degree(n) == 0

    def test_adding_edge_updates_classification(self):
        """After adding an edge, the parent loses leaf status and the child
        gains it; the child loses root status."""
        net = Network()
        a, b = Node("A"), Node("B")
        net.add_nodes(a, b)
        e = Edge(a, b)
        net.add_edges(e)

        # a is root (in-deg 0, out-deg 1)
        assert net.in_degree(a) == 0
        assert net.out_degree(a) == 1
        # b is leaf (out-deg 0)
        assert net.out_degree(b) == 0
        assert b in net.get_leaves()


# ===================================================================
# Network.clean
# ===================================================================

class TestClean:
    """Tests for the Network.clean() housekeeping method."""

    def test_remove_floaters(self):
        """clean([True, False, False]) should remove isolated nodes."""
        net = build_simple_tree()
        orphan = Node("Orphan")
        net.add_nodes(orphan)
        assert orphan in net

        net.clean([True, False, False])
        assert orphan not in net

    def test_remove_spurious_root(self):
        """clean([False, True, False]) should collapse a root with out-degree 1."""
        net = Network()
        r = Node("Root")
        bridge = Node("Bridge")
        a = Node("A")
        b = Node("B")
        net.add_nodes(r, bridge, a, b)
        net.add_edges(Edge(r, bridge))
        net.add_edges(Edge(bridge, a))
        net.add_edges(Edge(bridge, b))

        # Root has out-degree 1 → spurious
        net.clean([False, True, False])
        assert net.has_node_named("Root") is None
        assert net.root().label == "Bridge"

    def test_collapse_degree_one_chains(self):
        """clean([False, False, True]) should collapse chains of in/out deg-1 nodes."""
        net = Network()
        r = Node("Root")
        m1 = Node("M1")
        m2 = Node("M2")
        leaf = Node("Leaf")
        net.add_nodes(r, m1, m2, leaf)
        net.add_edges(Edge(r, m1))
        net.add_edges(Edge(m1, m2))
        net.add_edges(Edge(m2, leaf))

        # r -> m1 -> m2 -> leaf; m1, m2 are degree-1 chains
        # After adding a second child to root so root isn't spurious:
        leaf2 = Node("Leaf2")
        net.add_nodes(leaf2)
        net.add_edges(Edge(r, leaf2))

        net.clean([False, False, True])
        # m1, m2 should be collapsed; Root -> Leaf should be a single edge
        assert net.has_node_named("M1") is None
        assert net.has_node_named("M2") is None


# ===================================================================
# MRCA
# ===================================================================

class TestMRCA:
    """Tests for the most recent common ancestor (mrca) computation."""

    def test_mrca_of_siblings(self):
        """MRCA of two sibling leaves should be their shared parent."""
        net = build_simple_tree()
        a = _node_by_label(net, "A")
        b = _node_by_label(net, "B")
        mrca = net.mrca({a, b})
        assert mrca.label == "I1"

    def test_mrca_of_cousins(self):
        """MRCA of leaves from different subtrees should be the root."""
        net = build_simple_tree()
        a = _node_by_label(net, "A")
        c = _node_by_label(net, "C")
        mrca = net.mrca({a, c})
        assert mrca.label == "Root"

    def test_mrca_accepts_strings(self):
        """mrca should also accept a set of label strings."""
        net = build_simple_tree()
        mrca = net.mrca({"A", "B"})
        assert mrca.label == "I1"

    def test_mrca_of_single_node(self):
        """MRCA of a single node should be the node itself."""
        net = build_simple_tree()
        a = _node_by_label(net, "A")
        mrca = net.mrca({a})
        assert mrca is a

    def test_mrca_of_all_leaves(self):
        """MRCA of all leaves should be the root."""
        net = build_simple_tree()
        all_leaves = set(net.get_leaves())
        mrca = net.mrca(all_leaves)
        assert mrca.label == "Root"

    def test_mrca_with_reticulation(self):
        """MRCA on a network with reticulations should still work correctly."""
        net = build_level1_network()
        a = _node_by_label(net, "A")
        c = _node_by_label(net, "C")
        mrca = net.mrca({a, c})
        # A is under L1, C is under #H0 which is under both L1 and R1.
        # The LCA should be either L1 or Root depending on path weighting.
        assert mrca.label in {"L1", "Root"}


# ===================================================================
# Leaf Descendants
# ===================================================================

class TestLeafDescendants:
    """Tests for leaf_descendants and leaf_descendants_all."""

    def test_leaf_descendants_of_root(self):
        """Root's leaf descendants should be all leaves."""
        net = build_simple_tree()
        root = net.root()
        descs = net.leaf_descendants(root)
        assert {n.label for n in descs} == {"A", "B", "C", "D"}

    def test_leaf_descendants_of_internal(self):
        """An internal node's leaf descendants should be its subtree leaves."""
        net = build_simple_tree()
        i1 = _node_by_label(net, "I1")
        descs = net.leaf_descendants(i1)
        assert {n.label for n in descs} == {"A", "B"}

    def test_leaf_descendants_of_leaf(self):
        """A leaf's leaf descendants should be just itself."""
        net = build_simple_tree()
        a = _node_by_label(net, "A")
        descs = net.leaf_descendants(a)
        assert descs == {a}

    def test_leaf_descendants_all(self):
        """leaf_descendants_all should map every node to its leaf descendant set."""
        net = build_simple_tree()
        desc_map = net.leaf_descendants_all()
        root = net.root()
        assert {n.label for n in desc_map[root]} == {"A", "B", "C", "D"}

        i1 = _node_by_label(net, "I1")
        assert {n.label for n in desc_map[i1]} == {"A", "B"}

        a = _node_by_label(net, "A")
        assert desc_map[a] == {a}

    def test_leaf_descendants_not_in_graph_raises(self):
        """Calling leaf_descendants on a node not in the graph should raise."""
        net = build_simple_tree()
        outsider = Node("Outsider")
        with pytest.raises(NetworkError):
            net.leaf_descendants(outsider)


# ===================================================================
# Newick Round-trip
# ===================================================================

class TestNewick:
    """Tests for newick string generation and parsing."""

    def test_newick_contains_leaf_labels(self):
        """The newick string should contain all leaf labels."""
        net = build_simple_tree()
        nwk = net.newick()
        for label in ["A", "B", "C", "D"]:
            assert label in nwk

    def test_newick_ends_with_semicolon(self):
        """A valid newick string must end with ';'."""
        net = build_simple_tree()
        nwk = net.newick()
        assert nwk.endswith(";")

    def test_newick_round_trip_tree(self):
        """Parsing a tree's newick back should yield an isomorphic network."""
        net = build_simple_tree()
        nwk = net.newick()
        parsed = Network.from_newick(nwk)
        assert parsed.is_isomorphic(net)

    def test_newick_reticulation_labels(self):
        """Newick for a network with reticulations should include '#' labels."""
        net = build_level1_network()
        nwk = net.newick()
        assert "#H0" in nwk


# ===================================================================
# Acyclicity
# ===================================================================

class TestAcyclicity:
    """Tests for is_acyclic cycle detection."""

    def test_tree_is_acyclic(self):
        """A tree should be acyclic."""
        net = build_simple_tree()
        assert net.is_acyclic() is True

    def test_network_with_reticulation_is_acyclic(self):
        """A DAG network (with reticulations but no directed cycles) should
        be acyclic."""
        net = build_level1_network()
        assert net.is_acyclic() is True

    def test_network_with_directed_cycle_is_not_acyclic(self):
        """Adding a back-edge into an existing DAG should make is_acyclic
        return False.

        Note: is_acyclic() starts DFS from roots (in-degree-0 nodes), so a
        pure cycle with no root would go undetected.  We therefore build a
        graph that still has a root but contains a cycle among its descendants.
        """
        net = Network()
        r, a, b, c = Node("R"), Node("A"), Node("B"), Node("C")
        net.add_nodes(r, a, b, c)
        net.add_edges(Edge(r, a))
        net.add_edges(Edge(a, b))
        net.add_edges(Edge(b, c))
        net.add_edges(Edge(c, a))  # creates cycle reachable from root R
        assert net.is_acyclic() is False


# ===================================================================
# BFS / DFS
# ===================================================================

class TestBfsDfs:
    """Tests for the general bfs_dfs traversal method."""

    def test_bfs_distance_from_root(self):
        """BFS should compute correct hop-distances from the root."""
        net = build_simple_tree()
        dist, _ = net.bfs_dfs()
        root = net.root()
        assert dist[root] == 0

        i1 = _node_by_label(net, "I1")
        assert dist[i1] == 1

        a = _node_by_label(net, "A")
        assert dist[a] == 2

    def test_dfs_visits_all_nodes(self):
        """DFS should visit every node reachable from the root."""
        net = build_simple_tree()
        dist, _ = net.bfs_dfs(dfs=True)
        assert len(dist) == len(net.V())

    def test_bfs_with_accumulator(self):
        """The accumulator function should be called for each visited node."""
        net = build_simple_tree()
        visited_labels = []

        def acc(node, acc_list):
            acc_list.append(node.label)
            return acc_list

        _, result = net.bfs_dfs(accumulator=acc, accumulated=visited_labels)
        assert len(result) == len(net.V())


# ===================================================================
# Subnet & Copy
# ===================================================================

class TestSubnetAndCopy:
    """Tests for subnet() and copy()."""

    def test_subnet_produces_independent_copy(self):
        """subnet should produce a disconnected sub-network with fresh nodes."""
        net = build_simple_tree()
        i1 = _node_by_label(net, "I1")
        sub = net.subnet(i1)

        # Subnet should have nodes I1_copy*, A_copy*, B_copy*
        assert len(sub.V()) == 3
        original_ids = {id(n) for n in net.V()}
        for n in sub.V():
            assert id(n) not in original_ids

    def test_copy_preserves_topology(self):
        """copy() should yield a new network with identical structure."""
        net = build_simple_tree()
        clone, old_to_new = net.copy()

        assert len(clone.V()) == len(net.V())
        assert len(clone.E()) == len(net.E())
        assert clone.is_isomorphic(net)

        # Nodes should be distinct objects
        for old, new in old_to_new.items():
            assert old is not new
            assert old.label == new.label

    def test_copy_preserves_edge_gamma(self):
        """copy() should carry gamma values through to the new edges."""
        net = build_level1_network()
        clone, _ = net.copy()
        for e in clone.E():
            if e.dest.is_reticulation():
                assert e.get_gamma() is not None


# ===================================================================
# get_subtree_at
# ===================================================================

class TestGetSubtreeAt:
    """Tests for the get_subtree_at helper."""

    def test_subtree_at_root_is_all_nodes(self):
        """The subtree at the root should include every node."""
        net = build_simple_tree()
        sub = net.get_subtree_at(net.root())
        assert sub == set(net.V())

    def test_subtree_at_leaf_is_singleton(self):
        """The subtree at a leaf should be just that leaf."""
        net = build_simple_tree()
        a = _node_by_label(net, "A")
        sub = net.get_subtree_at(a)
        assert sub == {a}

    def test_subtree_at_internal(self):
        """The subtree at I1 should contain I1, A, B."""
        net = build_simple_tree()
        i1 = _node_by_label(net, "I1")
        sub = net.get_subtree_at(i1)
        labels = {n.label for n in sub}
        assert labels == {"I1", "A", "B"}


# ===================================================================
# Topological Order
# ===================================================================

class TestTopologicalOrder:
    """Tests for topological ordering of an acyclic network."""

    def test_topo_order_length(self):
        """Topological order should include every node exactly once."""
        net = build_simple_tree()
        order = net.topological_order()
        assert len(order) == len(net.V())

    def test_root_comes_first(self):
        """The root should appear before any of its descendants."""
        net = build_simple_tree()
        order = net.topological_order()
        root_idx = order.index(net.root())
        for leaf in net.get_leaves():
            assert order.index(leaf) > root_idx

    def test_parents_before_children(self):
        """Every parent should precede its child in topological order."""
        net = build_simple_tree()
        order = net.topological_order()
        idx = {n: i for i, n in enumerate(order)}
        for e in net.E():
            assert idx[e.src] < idx[e.dest]

    def test_cyclic_network_raises(self):
        """A cyclic graph should raise NetworkError from topological_order."""
        net = Network()
        a, b, c = Node("A"), Node("B"), Node("C")
        net.add_nodes(a, b, c)
        net.add_edges(Edge(a, b))
        net.add_edges(Edge(b, c))
        net.add_edges(Edge(c, a))
        with pytest.raises(NetworkError):
            net.topological_order()


# ===================================================================
# dist_from_root & distance_from_root
# ===================================================================

class TestDistFromRoot:
    """Tests for hop-count and branch-length distance from root."""

    def test_dist_from_root_hop_count(self):
        """dist_from_root should count edges from root."""
        net = build_simple_tree()
        root = net.root()
        assert net.dist_from_root(root) == 0

        a = _node_by_label(net, "A")
        assert net.dist_from_root(a) == 2

    def test_distance_from_root_edge_count_mode(self):
        """distance_from_root(use_time=False) should count edges."""
        net = build_small_network_with_lengths()
        a = _node_by_label(net, "A")
        dist = net.distance_from_root(a, use_time=False)
        assert dist == pytest.approx(2.0)

    def test_distance_from_root_branch_length_mode(self):
        """distance_from_root(use_time=True) should sum branch lengths.

        ``distance_from_root`` first tries ``node.get_time()`` which raises
        when no time is set, so we pre-populate times via
        ``set_node_times_from_root`` before calling with ``use_time=True``.
        """
        net = build_small_network_with_lengths()
        net.set_node_times_from_root()
        a = _node_by_label(net, "A")
        # Root->I1 = 1.0, I1->A = 0.5 => total = 1.5
        dist = net.distance_from_root(a, use_time=True)
        assert dist == pytest.approx(1.5)

    def test_distance_from_root_node_not_found(self):
        """distance_from_root on a node not in the network should raise."""
        net = build_simple_tree()
        outsider = Node("Outsider")
        with pytest.raises(NetworkError):
            net.distance_from_root(outsider)


# ===================================================================
# set_node_times_from_root / set_node_times_by_edge_count
# ===================================================================

class TestSetNodeTimes:
    """Tests for bulk time-setting utilities."""

    def test_set_times_from_root(self):
        """set_node_times_from_root should assign cumulative branch-length times."""
        net = build_small_network_with_lengths()
        net.set_node_times_from_root()

        root = net.root()
        assert root.get_time() == pytest.approx(0.0)

        a = _node_by_label(net, "A")
        assert a.get_time() == pytest.approx(1.5)  # 1.0 + 0.5

        c = _node_by_label(net, "C")
        assert c.get_time() == pytest.approx(1.7)  # 1.0 + 0.7

    def test_set_times_by_edge_count(self):
        """set_node_times_by_edge_count should assign hop-count times."""
        net = build_simple_tree()
        net.set_node_times_by_edge_count()

        root = net.root()
        assert root.get_time() == pytest.approx(0.0)

        a = _node_by_label(net, "A")
        assert a.get_time() == pytest.approx(2.0)  # 2 edges from root


# ===================================================================
# edges_downstream_of_node / edges_upstream_of_node
# ===================================================================

class TestEdgeTraversal:
    """Tests for downstream/upstream edge traversal."""

    def test_edges_downstream_of_root(self):
        """All edges should be downstream of the root."""
        net = build_simple_tree()
        root = net.root()
        downstream = net.edges_downstream_of_node(root)
        assert len(downstream) == len(net.E())

    def test_edges_downstream_of_leaf(self):
        """A leaf should have no downstream edges."""
        net = build_simple_tree()
        a = _node_by_label(net, "A")
        downstream = net.edges_downstream_of_node(a)
        assert len(downstream) == 0

    def test_edges_downstream_of_internal(self):
        """I1's downstream edges should be I1->A and I1->B."""
        net = build_simple_tree()
        i1 = _node_by_label(net, "I1")
        downstream = net.edges_downstream_of_node(i1)
        assert len(downstream) == 2

    def test_edges_upstream_of_leaf(self):
        """A leaf's upstream edges should trace back to the root."""
        net = build_simple_tree()
        a = _node_by_label(net, "A")
        upstream = net.edges_upstream_of_node(a)
        # A -> I1 (1 edge) + I1 -> Root (1 edge) = 2 upstream edges
        assert len(upstream) == 2

    def test_edges_upstream_of_root(self):
        """The root should have no upstream edges."""
        net = build_simple_tree()
        root = net.root()
        upstream = net.edges_upstream_of_node(root)
        assert len(upstream) == 0

    def test_node_not_in_graph_raises(self):
        """Passing a node not in the graph should raise NetworkError."""
        net = build_simple_tree()
        outsider = Node("Outsider")
        with pytest.raises(NetworkError):
            net.edges_downstream_of_node(outsider)
        with pytest.raises(NetworkError):
            net.edges_upstream_of_node(outsider)


# ===================================================================
# subgenome_count
# ===================================================================

class TestSubgenomeCount:
    """Tests for the subgenome count computation."""

    def test_root_subgenome_count_is_one(self):
        """The root always has subgenome count 1."""
        net = build_simple_tree()
        assert net.subgenome_count(net.root()) == 1

    def test_tree_node_subgenome_count_is_one(self):
        """In a tree (no reticulations), every node should have subgenome count 1."""
        net = build_simple_tree()
        for node in net.V():
            assert net.subgenome_count(node) == 1

    def test_reticulation_subgenome_count(self):
        """A reticulation node with 2 parents should have subgenome count 2."""
        net = build_level1_network()
        h0 = _node_by_label(net, "#H0")
        assert net.subgenome_count(h0) == 2

    def test_node_not_in_graph_raises(self):
        """subgenome_count on an external node should raise."""
        net = build_simple_tree()
        outsider = Node("Outsider")
        with pytest.raises(NetworkError):
            net.subgenome_count(outsider)


# ===================================================================
# to_networkx
# ===================================================================

class TestToNetworkX:
    """Tests for conversion to a NetworkX graph."""

    def test_node_count_matches(self):
        """The NetworkX graph should have the same number of nodes."""
        net = build_simple_tree()
        nx_graph = net.to_networkx()
        assert nx_graph.number_of_nodes() == len(net.V())

    def test_edge_count_matches(self):
        """The NetworkX graph should have the same number of edges."""
        net = build_simple_tree()
        nx_graph = net.to_networkx()
        assert nx_graph.number_of_edges() == len(net.E())

    def test_node_labels_preserved(self):
        """Node labels should appear in the NetworkX graph."""
        net = build_simple_tree()
        nx_graph = net.to_networkx()
        nx_labels = set(nx_graph.nodes())
        expected_labels = {n.label for n in net.V()}
        assert nx_labels == expected_labels


# ===================================================================
# is_isomorphic
# ===================================================================

class TestIsIsomorphic:
    """Tests for topology comparison via is_isomorphic."""

    def test_isomorphic_to_self(self):
        """A network should be isomorphic to itself."""
        net = build_simple_tree()
        assert net.is_isomorphic(net)

    def test_isomorphic_to_copy(self):
        """A copy should be isomorphic to the original."""
        net = build_simple_tree()
        clone, _ = net.copy()
        assert net.is_isomorphic(clone)

    def test_newick_round_trip_isomorphism(self):
        """Newick export → parse should yield an isomorphic network."""
        net = build_simple_tree()
        nwk = net.newick()
        parsed = Network.from_newick(nwk)
        assert net.is_isomorphic(parsed)

    def test_different_topologies_are_not_isomorphic(self):
        """Networks with different leaf counts should not be isomorphic."""
        tree = build_simple_tree()
        lvl1 = build_level1_network()
        assert not tree.is_isomorphic(lvl1)


# ===================================================================
# Reticulation-specific queries
# ===================================================================

class TestReticulationQueries:
    """Tests specific to networks with reticulation nodes."""

    def test_level1_has_one_reticulation(self):
        """The level-1 test network should have exactly one reticulation."""
        net = build_level1_network()
        retics = [n for n in net.V() if n.is_reticulation()]
        assert len(retics) == 1
        assert retics[0].label == "#H0"

    def test_level2_has_two_reticulations(self):
        """The level-2 test network should have exactly two reticulations."""
        net = build_level2_network()
        retics = [n for n in net.V() if n.is_reticulation()]
        assert len(retics) == 2
        retic_labels = {n.label for n in retics}
        assert retic_labels == {"#H0", "#H1"}

    def test_reticulation_gamma_sums_to_one(self):
        """Inheritance probabilities into a reticulation should sum to ~1."""
        net = build_level1_network()
        h0 = _node_by_label(net, "#H0")
        in_edges = net.in_edges(h0)
        gamma_sum = sum(e.get_gamma() for e in in_edges)
        assert gamma_sum == pytest.approx(1.0)

    def test_level_computation(self):
        """The level() utility should correctly classify network complexity."""
        tree = build_simple_tree()
        assert level(tree) == 0

        lvl1 = build_level1_network()
        assert level(lvl1) == 1

        lvl2 = build_level2_network()
        assert level(lvl2) == 2

    def test_is_tree_flag(self):
        """is_tree should be True only for tree networks."""
        assert is_tree(build_simple_tree()) is True
        assert is_tree(build_level1_network()) is False


# ===================================================================
# Newick for Networks (the original test_to_newick, preserved)
# ===================================================================

def build_simple_tree_10_leaves() -> Network:
    """Construct a strictly binary tree with 10 leaves and no reticulations.

    Used by the legacy ``test_to_newick`` test.
    """
    leaves = _leaf_labels("simple", 10)
    (leaf1, leaf2, leaf3, leaf4, leaf5,
     leaf6, leaf7, leaf8, leaf9, leaf10) = leaves

    internal_nodes = [
        "simple_root",
        "simple_left",
        "simple_right",
        "simple_left_inner_1",
        "simple_left_inner_2",
        "simple_right_inner_1",
        "simple_right_inner_2",
        "simple_right_inner_3",
        "simple_right_inner_4",
    ]

    edges = [
        _edge("simple_root", "simple_left"),
        _edge("simple_root", "simple_right"),
        _edge("simple_left", "simple_left_inner_1"),
        _edge("simple_left", "simple_left_inner_2"),
        _edge("simple_left_inner_1", leaf1),
        _edge("simple_left_inner_1", leaf2),
        _edge("simple_left_inner_2", leaf3),
        _edge("simple_left_inner_2", leaf4),
        _edge("simple_right", "simple_right_inner_1"),
        _edge("simple_right", "simple_right_inner_2"),
        _edge("simple_right_inner_1", leaf5),
        _edge("simple_right_inner_1", "simple_right_inner_3"),
        _edge("simple_right_inner_3", leaf6),
        _edge("simple_right_inner_3", leaf7),
        _edge("simple_right_inner_2", "simple_right_inner_4"),
        _edge("simple_right_inner_2", leaf10),
        _edge("simple_right_inner_4", leaf8),
        _edge("simple_right_inner_4", leaf9),
    ]

    node_labels = internal_nodes + leaves
    net = _build_network(node_labels, edges)
    assert len(net.get_leaves()) == 10
    assert is_tree(net)
    return net


def build_level2_10_leaves() -> Network:
    """Construct a binary level-2 network with 10 leaves and 2 reticulations.

    Used by the legacy ``test_to_newick`` test.
    """
    leaves = _leaf_labels("level2", 10)
    (leaf1, leaf2, leaf3, leaf4, leaf5,
     leaf6, leaf7, leaf8, leaf9, leaf10) = leaves

    internal_nodes = [
        "level2_root",
        "level2_left",
        "level2_right",
        "level2_left_a",
        "level2_left_b",
        "level2_right_a",
        "level2_right_b",
        "level2_e1",
        "level2_f1",
        "level2_g1",
        "level2_m1",
    ]
    retic_nodes = {"level2_h1", "level2_h2"}

    edges = [
        _edge("level2_root", "level2_left"),
        _edge("level2_root", "level2_right"),
        _edge("level2_left", "level2_left_a"),
        _edge("level2_left", "level2_left_b"),
        _edge("level2_left_a", leaf1),
        _edge("level2_left_a", leaf2),
        _edge("level2_left_b", leaf3),
        _edge("level2_left_b", leaf4),
        _edge("level2_right", "level2_right_a"),
        _edge("level2_right", "level2_right_b"),
        _edge("level2_right_a", leaf5),
        _edge("level2_right_a", "level2_e1"),
        _edge("level2_e1", leaf6),
        _edge("level2_e1", "level2_h1", gamma=0.4),
        _edge("level2_right_b", "level2_f1"),
        _edge("level2_right_b", "level2_g1"),
        _edge("level2_f1", leaf7),
        _edge("level2_f1", "level2_h2", gamma=0.6),
        _edge("level2_g1", leaf8),
        _edge("level2_g1", "level2_h1", gamma=0.6),
        _edge("level2_h1", "level2_m1"),
        _edge("level2_m1", leaf9),
        _edge("level2_m1", "level2_h2"),
        _edge("level2_h2", leaf10),
    ]

    node_labels = internal_nodes + list(retic_nodes) + leaves
    net = _build_network(node_labels, edges, retic_labels=retic_nodes)
    assert len([n for n in net.V() if n.is_reticulation()]) == 2
    assert len(net.get_leaves()) == 10
    assert level(net) == 2
    return net


class TestNewickRoundTrip:
    """Legacy test: newick → parse → isomorphism check for larger topologies."""

    def test_simple_tree_newick_roundtrip(self):
        """A 10-leaf tree should survive a newick round-trip."""
        sn = build_simple_tree_10_leaves()
        assert Network.from_newick(sn.newick()).is_isomorphic(sn)

    def test_level2_network_newick_roundtrip(self):
        """A level-2 network should survive a newick round-trip."""
        lvl2 = build_level2_10_leaves()
        assert Network.from_newick(lvl2.newick()).is_isomorphic(lvl2)


# ===================================================================
# put_item / get_item (blob storage)
# ===================================================================

class TestBlobStorage:
    """Tests for the Graph-level key/value blob storage."""

    def test_put_and_get_item(self):
        """put_item should store, get_item should retrieve."""
        net = build_simple_tree()
        net.put_item("meta", {"key": "value"})
        assert net.get_item("meta") == {"key": "value"}

    def test_put_item_does_not_overwrite(self):
        """put_item should not overwrite an existing key."""
        net = build_simple_tree()
        net.put_item("x", 1)
        net.put_item("x", 2)
        assert net.get_item("x") == 1  # first write wins


# ===================================================================
# EdgeSet standalone
# ===================================================================

class TestEdgeSet:
    """Tests for the EdgeSet data structure in isolation."""

    def test_add_and_contains(self):
        """Adding an edge should make it findable via 'in'."""
        es = EdgeSet()
        a, b = Node("A"), Node("B")
        e = Edge(a, b)
        es.add(e)
        assert e in es

    def test_remove(self):
        """Removing an edge should make it unfindable."""
        es = EdgeSet()
        a, b = Node("A"), Node("B")
        e = Edge(a, b)
        es.add(e)
        es.remove(e)
        assert e not in es

    def test_get_set(self):
        """get_set should return the internal set of edges."""
        es = EdgeSet()
        a, b = Node("A"), Node("B")
        e = Edge(a, b)
        es.add(e)
        assert e in es.get_set()

    def test_undirected_rejects_directed(self):
        """An undirected EdgeSet should reject Edge (directed) objects."""
        es = EdgeSet(directed=False)
        a, b = Node("A"), Node("B")
        e = Edge(a, b)
        with pytest.raises(TypeError):
            es.add(e)


# ===================================================================
# NodeSet standalone
# ===================================================================

class TestNodeSet:
    """Tests for the NodeSet data structure in isolation."""

    def test_add_and_contains(self):
        """Adding a node should make it findable via 'in'."""
        ns = NodeSet()
        n = Node("A")
        ns.add(n)
        assert n in ns

    def test_remove(self):
        """Removing a node should clear it from the set."""
        ns = NodeSet()
        n = Node("A")
        ns.add(n)
        ns.remove(n)
        assert n not in ns

    def test_get_by_name(self):
        """get(name) should retrieve a node by label."""
        ns = NodeSet()
        n = Node("Alpha")
        ns.add(n)
        assert ns.get("Alpha") is n

    def test_degree_tracking(self):
        """After processing an edge, in_deg/out_deg should update."""
        ns = NodeSet(directed=True)
        a, b = Node("A"), Node("B")
        ns.add(a, b)
        e = Edge(a, b)
        ns.process(e)
        assert ns.out_deg(a) == 1
        assert ns.in_deg(b) == 1

    def test_process_removal(self):
        """process(edge, removal=True) should decrement degrees."""
        ns = NodeSet(directed=True)
        a, b = Node("A"), Node("B")
        ns.add(a, b)
        e = Edge(a, b)
        ns.process(e)
        ns.process(e, removal=True)
        assert ns.out_deg(a) == 0
        assert ns.in_deg(b) == 0
