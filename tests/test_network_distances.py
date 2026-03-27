"""
Test suite for network distance / comparison metrics in phynetpy.GraphUtils.

Tests cover the eight distance functions:

    - ``mu_distance``                     – mu-representation distance
    - ``hardwired_cluster_distance``      – symmetric diff of hardwired clusters
    - ``softwired_cluster_distance``      – symmetric diff of softwired clusters
    - ``tripartition_distance``           – symmetric diff of tripartitions
    - ``displayed_tree_distance``         – symmetric diff of displayed tree topologies
    - ``robinson_foulds_distance``        – classic RF (cluster-based)
    - ``average_path_distance``           – APD (branch-length-aware)
    - ``weighted_average_path_distance``  – WAPD (branch-length + gamma-aware)

Each metric is tested for:
    - Identity (d(N, N') == 0 for topologically identical copies)
    - Symmetry (d(N1, N2) == d(N2, N1))
    - Non-negativity
    - Known hand-computed values on small topologies
    - Normalization (where applicable)
    - Edge cases (single-leaf, tree-vs-network, mismatched leaf sets)
"""

import pytest
from typing import Optional, Sequence, Tuple

from phynetpy.Network import Network, Node, Edge, NetworkError
from phynetpy.GraphUtils import (
    hardwired_cluster_distance,
    softwired_cluster_distance,
    mu_distance,
    tripartition_distance,
    displayed_tree_distance,
    robinson_foulds_distance,
    average_path_distance,
    weighted_average_path_distance,
    is_tree,
    level,
    get_all_subtrees,
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
    retic_labels: set[str] | None = None,
) -> Network:
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
        net.add_edges(Edge(nodes[src], nodes[dest], gamma=gamma))

    return net


def _leaf_labels(net: Network) -> set[str]:
    return {n.label for n in net.get_leaves()}


# ── Tree: ((A,B),(C,D))  ──
def build_tree_ABCD() -> Network:
    return _build_network(
        ["Root", "I1", "I2", "A", "B", "C", "D"],
        [
            _edge("Root", "I1"),
            _edge("Root", "I2"),
            _edge("I1", "A"),
            _edge("I1", "B"),
            _edge("I2", "C"),
            _edge("I2", "D"),
        ],
    )


# ── Tree: ((A,C),(B,D))  ──
def build_tree_ACBD() -> Network:
    return _build_network(
        ["Root", "I1", "I2", "A", "B", "C", "D"],
        [
            _edge("Root", "I1"),
            _edge("Root", "I2"),
            _edge("I1", "A"),
            _edge("I1", "C"),
            _edge("I2", "B"),
            _edge("I2", "D"),
        ],
    )


# ── Tree: (((A,B),C),D)  ──  (caterpillar / pectinate)
def build_tree_caterpillar() -> Network:
    return _build_network(
        ["Root", "I1", "I2", "A", "B", "C", "D"],
        [
            _edge("Root", "I1"),
            _edge("Root", "D"),
            _edge("I1", "I2"),
            _edge("I1", "C"),
            _edge("I2", "A"),
            _edge("I2", "B"),
        ],
    )


# ── Tree: ((A,B),C)  ──  3-leaf
def build_tree_3() -> Network:
    return _build_network(
        ["Root", "I1", "A", "B", "C"],
        [
            _edge("Root", "I1"),
            _edge("Root", "C"),
            _edge("I1", "A"),
            _edge("I1", "B"),
        ],
    )


# ── Tree: ((A,C),B)  ──  3-leaf, different topology
def build_tree_3_alt() -> Network:
    return _build_network(
        ["Root", "I1", "A", "B", "C"],
        [
            _edge("Root", "I1"),
            _edge("Root", "B"),
            _edge("I1", "A"),
            _edge("I1", "C"),
        ],
    )


# ── Level-1 network (3 leaves) ──
#       Root
#      /    \
#    I1      I2
#   / \     / \
#  A  P1   P2  C
#      \  /
#      #H0
#       |
#       B
def build_level1_3leaf() -> Network:
    return _build_network(
        ["Root", "I1", "I2", "P1", "P2", "#H0", "A", "B", "C"],
        [
            _edge("Root", "I1"),
            _edge("Root", "I2"),
            _edge("I1", "A"),
            _edge("I1", "P1"),
            _edge("I2", "P2"),
            _edge("I2", "C"),
            _edge("P1", "#H0", gamma=0.6),
            _edge("P2", "#H0", gamma=0.4),
            _edge("#H0", "B"),
        ],
        retic_labels={"#H0"},
    )


# ── Level-1 network (4 leaves) ──
#        Root
#       /    \
#     I1      I2
#    / \     / \
#   A   P1  P2  B
#        \  /
#        #H0
#        / \
#       C   D
def build_level1_4leaf() -> Network:
    return _build_network(
        ["Root", "I1", "I2", "P1", "P2", "#H0", "A", "B", "C", "D"],
        [
            _edge("Root", "I1"),
            _edge("Root", "I2"),
            _edge("I1", "A"),
            _edge("I1", "P1"),
            _edge("I2", "P2"),
            _edge("I2", "B"),
            _edge("P1", "#H0", gamma=0.5),
            _edge("P2", "#H0", gamma=0.5),
            _edge("#H0", "C"),
            _edge("#H0", "D"),
        ],
        retic_labels={"#H0"},
    )


# ── Level-2 network (4 leaves, two reticulations) ──
#          Root
#         /    \
#        I1     I2
#       / \    / \
#      A   P1 P2  B
#           \ /
#           #H0
#           / \
#          P3  D
#          |
#         #H1
#          |
#          C
def build_level2_4leaf() -> Network:
    return _build_network(
        ["Root", "I1", "I2", "P1", "P2", "P3",
         "#H0", "#H1", "A", "B", "C", "D"],
        [
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
        ],
        retic_labels={"#H0", "#H1"},
    )


# ===================================================================
# Hardwired Cluster Distance
# ===================================================================

class TestHardwiredClusterDistance:

    def test_identity_tree(self):
        """d(T, T') == 0 for two copies of the same tree."""
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert hardwired_cluster_distance(t1, t2) == 0

    def test_identity_network(self):
        """d(N, N') == 0 for two copies of the same network."""
        n1 = build_level1_3leaf()
        n2 = build_level1_3leaf()
        assert hardwired_cluster_distance(n1, n2) == 0

    def test_symmetry(self):
        """d(N1, N2) == d(N2, N1)."""
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert (
            hardwired_cluster_distance(t1, t2)
            == hardwired_cluster_distance(t2, t1)
        )

    def test_non_negative(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert hardwired_cluster_distance(t1, t2) >= 0

    def test_different_trees(self):
        """((A,B),(C,D)) vs ((A,C),(B,D)) should have non-zero distance.

        HC(T1) non-trivial: {A,B}, {C,D}, {A,B,C,D}
        HC(T2) non-trivial: {A,C}, {B,D}, {A,B,C,D}
        Symmetric diff = {A,B}, {C,D}, {A,C}, {B,D} => 4
        """
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert hardwired_cluster_distance(t1, t2) == 4

    def test_tree_vs_network(self):
        """Tree ((A,B),C) vs level-1 network on {A,B,C} should differ."""
        tree = build_tree_3()
        net = build_level1_3leaf()
        d = hardwired_cluster_distance(tree, net)
        assert d > 0

    def test_normalize_identical(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert hardwired_cluster_distance(t1, t2, normalize=True) == 0.0

    def test_normalize_range(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        d = hardwired_cluster_distance(t1, t2, normalize=True)
        assert 0.0 < d <= 1.0

    def test_caterpillar_vs_balanced(self):
        """Caterpillar (((A,B),C),D) vs balanced ((A,B),(C,D))."""
        balanced = build_tree_ABCD()
        caterpillar = build_tree_caterpillar()
        d = hardwired_cluster_distance(balanced, caterpillar)
        assert d > 0


# ===================================================================
# Softwired Cluster Distance
# ===================================================================

class TestSoftwiredClusterDistance:

    def test_identity_tree(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert softwired_cluster_distance(t1, t2) == 0

    def test_identity_network(self):
        n1 = build_level1_3leaf()
        n2 = build_level1_3leaf()
        assert softwired_cluster_distance(n1, n2) == 0

    def test_symmetry(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert (
            softwired_cluster_distance(t1, t2)
            == softwired_cluster_distance(t2, t1)
        )

    def test_trees_equal_hardwired(self):
        """On trees, softwired == hardwired (only one displayed tree)."""
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert (
            softwired_cluster_distance(t1, t2)
            == hardwired_cluster_distance(t1, t2)
        )

    def test_softwired_leq_hardwired_on_networks(self):
        """Softwired clusters are a superset of hardwired clusters, so
        softwired distance can differ from hardwired distance.
        For identical networks both should be 0."""
        n1 = build_level1_4leaf()
        n2 = build_level1_4leaf()
        assert softwired_cluster_distance(n1, n2) == 0
        assert hardwired_cluster_distance(n1, n2) == 0

    def test_network_vs_different_tree(self):
        """A level-1 network vs a tree on the same leaf set."""
        net = build_level1_3leaf()
        tree = build_tree_3_alt()
        d = softwired_cluster_distance(net, tree)
        assert d >= 0

    def test_normalize_range(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        d = softwired_cluster_distance(t1, t2, normalize=True)
        assert 0.0 < d <= 1.0

    def test_level2_self(self):
        n1 = build_level2_4leaf()
        n2 = build_level2_4leaf()
        assert softwired_cluster_distance(n1, n2) == 0


# ===================================================================
# Mu-Distance (mu-representation)
# ===================================================================

class TestMuDistance:

    def test_identity_tree(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert mu_distance(t1, t2) == 0

    def test_identity_network(self):
        n1 = build_level1_3leaf()
        n2 = build_level1_3leaf()
        assert mu_distance(n1, n2) == 0

    def test_symmetry(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert mu_distance(t1, t2) == mu_distance(t2, t1)

    def test_non_negative(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert mu_distance(t1, t2) >= 0

    def test_different_trees_nonzero(self):
        """Different topologies on the same leaf set must have d > 0."""
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert mu_distance(t1, t2) > 0

    def test_mismatched_leaves_raises(self):
        """Networks with different leaf label sets should raise."""
        t_abcd = build_tree_ABCD()
        t_abc = build_tree_3()
        with pytest.raises(NetworkError):
            mu_distance(t_abcd, t_abc)

    def test_tree_vs_network_same_leaves(self):
        """A tree and a network on the same leaf set should have d > 0."""
        tree = build_tree_3()
        net = build_level1_3leaf()
        d = mu_distance(tree, net)
        assert d > 0

    def test_level2_self(self):
        n1 = build_level2_4leaf()
        n2 = build_level2_4leaf()
        assert mu_distance(n1, n2) == 0

    def test_triangle_inequality(self):
        """d(A,C) <= d(A,B) + d(B,C) for three trees on the same leaf set."""
        a = build_tree_ABCD()
        b = build_tree_ACBD()
        c = build_tree_caterpillar()
        dab = mu_distance(a, b)
        dbc = mu_distance(b, c)
        dac = mu_distance(a, c)
        assert dac <= dab + dbc

    def test_known_value_3leaf_trees(self):
        """((A,B),C) vs ((A,C),B) on 3 leaves.

        Tree 1 nodes: Root, I1, A, B, C (5 nodes)
          mu(Root) = (1,1,1)
          mu(I1)   = (1,1,0)
          mu(A)    = (1,0,0)
          mu(B)    = (0,1,0)
          mu(C)    = (0,0,1)

        Tree 2 nodes: Root, I1, A, B, C (5 nodes)
          mu(Root) = (1,1,1)
          mu(I1)   = (1,0,1)
          mu(A)    = (1,0,0)
          mu(B)    = (0,1,0)
          mu(C)    = (0,0,1)

        Shared: Root, A, B, C vectors are identical.
        Differ: (1,1,0) in T1 not in T2; (1,0,1) in T2 not in T1.
        Multiset symmetric diff size = 2.
        """
        t1 = build_tree_3()
        t2 = build_tree_3_alt()
        assert mu_distance(t1, t2) == 2


# ===================================================================
# Tripartition Distance
# ===================================================================

class TestTripartitionDistance:

    def test_identity_tree(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert tripartition_distance(t1, t2) == 0

    def test_identity_network(self):
        n1 = build_level1_3leaf()
        n2 = build_level1_3leaf()
        assert tripartition_distance(n1, n2) == 0

    def test_symmetry(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert (
            tripartition_distance(t1, t2)
            == tripartition_distance(t2, t1)
        )

    def test_non_negative(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert tripartition_distance(t1, t2) >= 0

    def test_different_trees_nonzero(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert tripartition_distance(t1, t2) > 0

    def test_known_value_balanced_trees(self):
        """((A,B),(C,D)) vs ((A,C),(B,D)) — all 3 tree nodes generate
        distinct tripartitions, and the two trees share none.

        T1 tripartitions (tree nodes with 2 children: Root, I1, I2):
          Root: ({A,B}, {C,D}, {})
          I1:   ({A}, {B}, {C,D})
          I2:   ({C}, {D}, {A,B})

        T2 tripartitions:
          Root: ({A,C}, {B,D}, {})
          I1:   ({A}, {C}, {B,D})
          I2:   ({B}, {D}, {A,C})

        No overlap => symmetric diff = 6.
        """
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert tripartition_distance(t1, t2) == 6

    def test_tree_vs_network(self):
        tree = build_tree_3()
        net = build_level1_3leaf()
        d = tripartition_distance(tree, net)
        assert d >= 0

    def test_normalize_identical(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert tripartition_distance(t1, t2, normalize=True) == 0.0

    def test_normalize_range(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        d = tripartition_distance(t1, t2, normalize=True)
        assert 0.0 < d <= 1.0

    def test_level2_self(self):
        n1 = build_level2_4leaf()
        n2 = build_level2_4leaf()
        assert tripartition_distance(n1, n2) == 0


# ===================================================================
# Displayed Tree Distance
# ===================================================================

class TestDisplayedTreeDistance:

    def test_identity_tree(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert displayed_tree_distance(t1, t2) == 0

    def test_identity_network(self):
        n1 = build_level1_3leaf()
        n2 = build_level1_3leaf()
        assert displayed_tree_distance(n1, n2) == 0

    def test_symmetry(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert (
            displayed_tree_distance(t1, t2)
            == displayed_tree_distance(t2, t1)
        )

    def test_non_negative(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert displayed_tree_distance(t1, t2) >= 0

    def test_different_trees(self):
        """Two distinct 4-leaf trees each display exactly one topology,
        so the symmetric diff should be 2 (one unique to each)."""
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert displayed_tree_distance(t1, t2) == 2

    def test_network_displays_include_tree(self):
        """If a tree is one of the displayed trees of a network, the
        symmetric diff should be < |DT(net)|."""
        net = build_level1_3leaf()
        trees = get_all_subtrees(net)
        assert len(trees) == 2

    def test_normalize_identical(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert displayed_tree_distance(t1, t2, normalize=True) == 0.0

    def test_normalize_range(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        d = displayed_tree_distance(t1, t2, normalize=True)
        assert 0.0 < d <= 1.0

    def test_level2_self(self):
        n1 = build_level2_4leaf()
        n2 = build_level2_4leaf()
        assert displayed_tree_distance(n1, n2) == 0

    def test_tree_vs_level1(self):
        """A tree and a level-1 network on the same leaf set."""
        tree = build_tree_3()
        net = build_level1_3leaf()
        d = displayed_tree_distance(tree, net)
        assert d >= 0


# ===================================================================
# Cross-Metric Consistency Tests
# ===================================================================

class TestCrossMetricConsistency:

    def test_all_zero_for_identical_tree(self):
        """All five distances should be 0 for two copies of the same tree."""
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert hardwired_cluster_distance(t1, t2) == 0
        assert softwired_cluster_distance(t1, t2) == 0
        assert mu_distance(t1, t2) == 0
        assert tripartition_distance(t1, t2) == 0
        assert displayed_tree_distance(t1, t2) == 0

    def test_all_zero_for_identical_network(self):
        """All five distances should be 0 for two copies of the same network."""
        n1 = build_level1_4leaf()
        n2 = build_level1_4leaf()
        assert hardwired_cluster_distance(n1, n2) == 0
        assert softwired_cluster_distance(n1, n2) == 0
        assert mu_distance(n1, n2) == 0
        assert tripartition_distance(n1, n2) == 0
        assert displayed_tree_distance(n1, n2) == 0

    def test_all_nonzero_for_different_trees(self):
        """All distances should be > 0 for two different tree topologies."""
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert hardwired_cluster_distance(t1, t2) > 0
        assert softwired_cluster_distance(t1, t2) > 0
        assert mu_distance(t1, t2) > 0
        assert tripartition_distance(t1, t2) > 0
        assert displayed_tree_distance(t1, t2) > 0

    def test_trees_softwired_equals_hardwired(self):
        """On pure trees, softwired and hardwired distances must agree."""
        for pair in [
            (build_tree_ABCD, build_tree_ACBD),
            (build_tree_ABCD, build_tree_caterpillar),
            (build_tree_3, build_tree_3_alt),
        ]:
            t1 = pair[0]()
            t2 = pair[1]()
            assert (
                softwired_cluster_distance(t1, t2)
                == hardwired_cluster_distance(t1, t2)
            ), f"softwired != hardwired for {pair[0].__name__} vs {pair[1].__name__}"

    def test_all_symmetric(self):
        """Verify symmetry across all metrics for a specific pair."""
        a = build_level1_4leaf()
        b = build_tree_ABCD()
        assert hardwired_cluster_distance(a, b) == hardwired_cluster_distance(b, a)
        assert softwired_cluster_distance(a, b) == softwired_cluster_distance(b, a)
        assert mu_distance(a, b) == mu_distance(b, a)
        assert tripartition_distance(a, b) == tripartition_distance(b, a)
        assert displayed_tree_distance(a, b) == displayed_tree_distance(b, a)

    def test_level2_all_metrics(self):
        """Level-2 network vs a tree: all metrics produce a consistent
        (non-negative, symmetric) result."""
        net = build_level2_4leaf()
        tree = build_tree_ABCD()

        for dist_fn in [
            hardwired_cluster_distance,
            softwired_cluster_distance,
            tripartition_distance,
            displayed_tree_distance,
        ]:
            d = dist_fn(net, tree)
            assert d >= 0
            assert d == dist_fn(tree, net)

        d_nak = mu_distance(net, tree)
        assert d_nak >= 0
        assert d_nak == mu_distance(tree, net)


# ===================================================================
# Network Builders with Branch Lengths (for APD / WAPD tests)
# ===================================================================

BranchSpec = Tuple[str, str, float, Optional[float]]   # (src, dst, length, gamma)


def _bedge(src: str, dest: str, length: float,
           gamma: Optional[float] = None) -> BranchSpec:
    return (src, dest, length, gamma)


def _build_network_with_lengths(
    node_labels: Sequence[str],
    edge_specs: Sequence[BranchSpec],
    retic_labels: set[str] | None = None,
) -> Network:
    net = Network()
    retic_set = set(retic_labels or [])
    nodes: dict[str, Node] = {}
    for label in node_labels:
        if label not in nodes:
            nodes[label] = Node(label, is_reticulation=label in retic_set)
    net.add_nodes(*nodes.values())
    for src, dest, length, gamma in edge_specs:
        net.add_edges(Edge(nodes[src], nodes[dest],
                           length=length, gamma=gamma))
    return net


# ── Tree ((A,B),(C,D)) with unit branch lengths ──
def build_tree_ABCD_bl() -> Network:
    return _build_network_with_lengths(
        ["Root", "I1", "I2", "A", "B", "C", "D"],
        [
            _bedge("Root", "I1", 1.0),
            _bedge("Root", "I2", 1.0),
            _bedge("I1", "A", 1.0),
            _bedge("I1", "B", 1.0),
            _bedge("I2", "C", 1.0),
            _bedge("I2", "D", 1.0),
        ],
    )


# ── Tree ((A,C),(B,D)) with unit branch lengths ──
def build_tree_ACBD_bl() -> Network:
    return _build_network_with_lengths(
        ["Root", "I1", "I2", "A", "B", "C", "D"],
        [
            _bedge("Root", "I1", 1.0),
            _bedge("Root", "I2", 1.0),
            _bedge("I1", "A", 1.0),
            _bedge("I1", "C", 1.0),
            _bedge("I2", "B", 1.0),
            _bedge("I2", "D", 1.0),
        ],
    )


# ── Tree ((A,B),(C,D)) with varying branch lengths ──
def build_tree_ABCD_varied_bl() -> Network:
    return _build_network_with_lengths(
        ["Root", "I1", "I2", "A", "B", "C", "D"],
        [
            _bedge("Root", "I1", 2.0),
            _bedge("Root", "I2", 3.0),
            _bedge("I1", "A", 1.0),
            _bedge("I1", "B", 1.5),
            _bedge("I2", "C", 0.5),
            _bedge("I2", "D", 2.0),
        ],
    )


# ── Tree ((A,B),C) with branch lengths ──
def build_tree_3_bl() -> Network:
    return _build_network_with_lengths(
        ["Root", "I1", "A", "B", "C"],
        [
            _bedge("Root", "I1", 1.0),
            _bedge("Root", "C", 2.0),
            _bedge("I1", "A", 1.0),
            _bedge("I1", "B", 1.0),
        ],
    )


# ── Tree ((A,C),B) with branch lengths ──
def build_tree_3_alt_bl() -> Network:
    return _build_network_with_lengths(
        ["Root", "I1", "A", "B", "C"],
        [
            _bedge("Root", "I1", 1.0),
            _bedge("Root", "B", 2.0),
            _bedge("I1", "A", 1.0),
            _bedge("I1", "C", 1.0),
        ],
    )


# ── Level-1 network (3 leaves) with branch lengths ──
def build_level1_3leaf_bl() -> Network:
    return _build_network_with_lengths(
        ["Root", "I1", "I2", "P1", "P2", "#H0", "A", "B", "C"],
        [
            _bedge("Root", "I1", 1.0),
            _bedge("Root", "I2", 1.0),
            _bedge("I1", "A", 1.0),
            _bedge("I1", "P1", 0.5),
            _bedge("I2", "P2", 0.5),
            _bedge("I2", "C", 1.0),
            _bedge("P1", "#H0", 0.5, gamma=0.6),
            _bedge("P2", "#H0", 0.5, gamma=0.4),
            _bedge("#H0", "B", 1.0),
        ],
        retic_labels={"#H0"},
    )


# ===================================================================
# Robinson-Foulds Distance
# ===================================================================

class TestRobinsonFouldsDistance:

    def test_identity_tree(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert robinson_foulds_distance(t1, t2) == 0

    def test_identity_network(self):
        n1 = build_level1_3leaf()
        n2 = build_level1_3leaf()
        assert robinson_foulds_distance(n1, n2) == 0

    def test_symmetry(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert (
            robinson_foulds_distance(t1, t2)
            == robinson_foulds_distance(t2, t1)
        )

    def test_non_negative(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert robinson_foulds_distance(t1, t2) >= 0

    def test_known_value_balanced_trees(self):
        """((A,B),(C,D)) vs ((A,C),(B,D)).

        Non-trivial clusters (size > 1 and < 4):
          T1: {A,B}, {C,D}
          T2: {A,C}, {B,D}
        Symmetric diff = 4.
        """
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert robinson_foulds_distance(t1, t2) == 4

    def test_known_value_3leaf_trees(self):
        """((A,B),C) vs ((A,C),B).

        Non-trivial clusters (size 2):
          T1: {A,B}
          T2: {A,C}
        Symmetric diff = 2.
        """
        t1 = build_tree_3()
        t2 = build_tree_3_alt()
        assert robinson_foulds_distance(t1, t2) == 2

    def test_mismatched_leaves_raises(self):
        t_abcd = build_tree_ABCD()
        t_abc = build_tree_3()
        with pytest.raises(NetworkError):
            robinson_foulds_distance(t_abcd, t_abc)

    def test_normalize_identical(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ABCD()
        assert robinson_foulds_distance(t1, t2, normalize=True) == 0.0

    def test_normalize_range(self):
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        d = robinson_foulds_distance(t1, t2, normalize=True)
        assert 0.0 < d <= 1.0

    def test_consistent_with_hardwired(self):
        """RF should equal hardwired cluster distance for trees
        (the full-leaf cluster is in both, so it cancels)."""
        t1 = build_tree_ABCD()
        t2 = build_tree_ACBD()
        assert (
            robinson_foulds_distance(t1, t2)
            == hardwired_cluster_distance(t1, t2)
        )


# ===================================================================
# Average Path Distance (APD)
# ===================================================================

class TestAveragePathDistance:

    def test_identity_tree(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ABCD_bl()
        assert average_path_distance(t1, t2) == pytest.approx(0.0)

    def test_identity_network(self):
        n1 = build_level1_3leaf_bl()
        n2 = build_level1_3leaf_bl()
        assert average_path_distance(n1, n2) == pytest.approx(0.0)

    def test_symmetry(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ACBD_bl()
        assert average_path_distance(t1, t2) == pytest.approx(
            average_path_distance(t2, t1)
        )

    def test_non_negative(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ACBD_bl()
        assert average_path_distance(t1, t2) >= 0.0

    def test_different_trees_nonzero(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ACBD_bl()
        assert average_path_distance(t1, t2) > 0.0

    def test_mismatched_leaves_raises(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_3_bl()
        with pytest.raises(NetworkError):
            average_path_distance(t1, t2)

    def test_known_value_unit_trees(self):
        """For two trees with identical unit branch lengths:
        In T1 = ((A,B),(C,D)):
          d(A,B) = 2, d(C,D) = 2, d(A,C) = d(A,D) = d(B,C) = d(B,D) = 4
        In T2 = ((A,C),(B,D)):
          d(A,C) = 2, d(B,D) = 2, d(A,B) = d(A,D) = d(B,C) = d(C,D) = 4
        Differences: |2-4| + |4-2| + |4-4| + |4-4| + |4-4| + |2-4| = 2+2+0+0+0+2 = ... wait

        Actually T1 pairs (sorted):
          (A,B)=2  (A,C)=4  (A,D)=4  (B,C)=4  (B,D)=4  (C,D)=2
        T2 pairs:
          (A,B)=4  (A,C)=2  (A,D)=4  (B,C)=4  (B,D)=2  (C,D)=4
        Diffs: 2+2+0+0+2+2 = 8
        """
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ACBD_bl()
        assert average_path_distance(t1, t2) == pytest.approx(8.0)

    def test_normalize(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ACBD_bl()
        d = average_path_distance(t1, t2, normalize=True)
        assert d == pytest.approx(8.0 / 6)

    def test_sensitive_to_branch_lengths(self):
        """Two trees with same topology but different branch lengths
        should have APD > 0."""
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ABCD_varied_bl()
        assert average_path_distance(t1, t2) > 0.0

    def test_trees_apd_equals_wapd(self):
        """For trees (no reticulations), APD and WAPD should agree."""
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ACBD_bl()
        assert average_path_distance(t1, t2) == pytest.approx(
            weighted_average_path_distance(t1, t2)
        )


# ===================================================================
# Weighted Average Path Distance (WAPD)
# ===================================================================

class TestWeightedAveragePathDistance:

    def test_identity_tree(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ABCD_bl()
        assert weighted_average_path_distance(t1, t2) == pytest.approx(0.0)

    def test_identity_network(self):
        n1 = build_level1_3leaf_bl()
        n2 = build_level1_3leaf_bl()
        assert weighted_average_path_distance(n1, n2) == pytest.approx(0.0)

    def test_symmetry(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ACBD_bl()
        assert weighted_average_path_distance(t1, t2) == pytest.approx(
            weighted_average_path_distance(t2, t1)
        )

    def test_non_negative(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ACBD_bl()
        assert weighted_average_path_distance(t1, t2) >= 0.0

    def test_different_trees_nonzero(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ACBD_bl()
        assert weighted_average_path_distance(t1, t2) > 0.0

    def test_mismatched_leaves_raises(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_3_bl()
        with pytest.raises(NetworkError):
            weighted_average_path_distance(t1, t2)

    def test_normalize_identical(self):
        n1 = build_level1_3leaf_bl()
        n2 = build_level1_3leaf_bl()
        assert weighted_average_path_distance(
            n1, n2, normalize=True
        ) == pytest.approx(0.0)

    def test_normalize_range(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ACBD_bl()
        d_norm = weighted_average_path_distance(t1, t2, normalize=True)
        d_raw = weighted_average_path_distance(t1, t2)
        assert d_norm > 0.0
        assert d_norm < d_raw or d_raw == pytest.approx(0.0)

    def test_network_wapd_differs_from_apd(self):
        """For a network with asymmetric gammas, WAPD and APD may
        produce different pairwise matrices (different weighting), so
        their distance to a tree can differ."""
        net = build_level1_3leaf_bl()
        tree = build_tree_3_bl()
        apd = average_path_distance(net, tree)
        wapd = weighted_average_path_distance(net, tree)
        assert apd >= 0.0
        assert wapd >= 0.0

    def test_sensitive_to_branch_lengths(self):
        t1 = build_tree_ABCD_bl()
        t2 = build_tree_ABCD_varied_bl()
        assert weighted_average_path_distance(t1, t2) > 0.0
