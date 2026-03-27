"""
Test suite for network topology moves in phynetpy.NetworkMoves.

Tests cover:
    - ``add_hybrid``          – insert a reticulation edge (normal + bubble)
    - ``remove_hybrid``       – remove a reticulation edge
    - ``nni``                 – nearest-neighbor interchange
    - ``node_height_change``  – move a node in time (with and without extend)
    - ``_suppress_chain_node``– merge a degree-1 chain node
    - ``spr``                 – subtree prune-and-regraft
    - ``permute_leaves``      – lazy leaf permutation iterator

Each function is tested for:
    - Basic correctness on small hand-built topologies
    - Structural invariants (leaf set, node/edge counts, root preserved)
    - Edge-attribute preservation where applicable
    - Error / edge-case handling (too-small trees, bad inputs, bound violations)
    - Effectiveness (moves actually change the topology)
"""

from __future__ import annotations

import math
import random
import types
from itertools import islice
from typing import Optional, Sequence, Tuple

import pytest

from phynetpy.Network import Network, Node, Edge, NetworkError
from phynetpy.GraphUtils import level
from phynetpy.NetworkMoves import (
    add_hybrid,
    remove_hybrid,
    nni,
    node_height_change,
    _suppress_chain_node,
    spr,
    permute_leaves,
)


# ===================================================================
# Helpers
# ===================================================================

EdgeSpec = Tuple[str, str, Optional[float]]


def _build_network(
    node_labels: Sequence[str],
    edge_specs: Sequence[EdgeSpec],
    retic_labels: set[str] | None = None,
    times: dict[str, float] | None = None,
) -> Network:
    """Build a small directed Network from labels and edge specs."""
    net = Network()
    retic_set = set(retic_labels or [])

    nodes: dict[str, Node] = {}
    for label in node_labels:
        if label not in nodes:
            nodes[label] = Node(label, is_reticulation=label in retic_set)
    for label in retic_set:
        if label not in nodes:
            nodes[label] = Node(label, is_reticulation=True)

    net.add_nodes(*nodes.values())

    for src, dest, gamma in edge_specs:
        net.add_edges(Edge(nodes[src], nodes[dest], gamma=gamma))

    if times:
        for label, t in times.items():
            nodes[label].set_time(t)

    return net


def _leaf_labels(net: Network) -> set[str]:
    return {lf.label for lf in net.get_leaves()}


def _node_labels(net: Network) -> set[str]:
    return {n.label for n in net.V()}


def _all_edge_pairs(net: Network) -> set[tuple[str, str]]:
    return {(e.src.label, e.dest.label) for e in net.E()}


def _build_balanced_tree() -> Network:
    """
    Balanced 4-leaf tree:

         R
        / \\
       I1  I2
      / \\  / \\
     A   B C   D
    """
    return _build_network(
        ["R", "I1", "I2", "A", "B", "C", "D"],
        [
            ("R", "I1", None),
            ("R", "I2", None),
            ("I1", "A", None),
            ("I1", "B", None),
            ("I2", "C", None),
            ("I2", "D", None),
        ],
    )


def _build_caterpillar_tree() -> Network:
    """
    Caterpillar 4-leaf tree:

        R
       / \\
      I1   D
     / \\
    I2   C
   / \\
  A   B
    """
    return _build_network(
        ["R", "I1", "I2", "A", "B", "C", "D"],
        [
            ("R", "I1", None),
            ("R", "D", None),
            ("I1", "I2", None),
            ("I1", "C", None),
            ("I2", "A", None),
            ("I2", "B", None),
        ],
    )


def _build_timed_tree() -> Network:
    """
    Balanced tree with times set (root=0, leaves at 3.0):

         R (0.0)
        / \\
    (1.0) I1  I2 (1.0)
        / \\  / \\
       A   B C   D  (all at 3.0)
    """
    return _build_network(
        ["R", "I1", "I2", "A", "B", "C", "D"],
        [
            ("R", "I1", None),
            ("R", "I2", None),
            ("I1", "A", None),
            ("I1", "B", None),
            ("I2", "C", None),
            ("I2", "D", None),
        ],
        times={"R": 0.0, "I1": 1.0, "I2": 1.0, "A": 3.0, "B": 3.0, "C": 3.0, "D": 3.0},
    )


def _build_5leaf_tree() -> Network:
    """
    5-leaf tree for SPR / NNI with more room:

           R
          / \\
        I1   I2
       / \\   / \\
      A   I3 C   D
          / \\
         B   E
    """
    return _build_network(
        ["R", "I1", "I2", "I3", "A", "B", "C", "D", "E"],
        [
            ("R", "I1", None),
            ("R", "I2", None),
            ("I1", "A", None),
            ("I1", "I3", None),
            ("I3", "B", None),
            ("I3", "E", None),
            ("I2", "C", None),
            ("I2", "D", None),
        ],
    )


# ===================================================================
# add_hybrid
# ===================================================================


class TestAddHybrid:
    """Tests for add_hybrid (normal insertion and bubble)."""

    def test_basic_insertion_increases_level(self):
        net = _build_balanced_tree()
        assert level(net) == 0
        src_edge = net.get_edge(
            _get_node(net, "I1"), _get_node(net, "A")
        )
        dest_edge = net.get_edge(
            _get_node(net, "I2"), _get_node(net, "C")
        )
        add_hybrid(net, src_edge, dest_edge)
        assert level(net) >= 1

    def test_node_and_edge_counts(self):
        net = _build_balanced_tree()
        n_before = len(net.V())
        e_before = len(net.E())
        src_edge = net.get_edge(
            _get_node(net, "I1"), _get_node(net, "A")
        )
        dest_edge = net.get_edge(
            _get_node(net, "I2"), _get_node(net, "C")
        )
        add_hybrid(net, src_edge, dest_edge)
        assert len(net.V()) == n_before + 2
        assert len(net.E()) == e_before - 2 + 5

    def test_reticulation_flag_set(self):
        net = _build_balanced_tree()
        src_edge = net.get_edge(
            _get_node(net, "I1"), _get_node(net, "A")
        )
        dest_edge = net.get_edge(
            _get_node(net, "I2"), _get_node(net, "C")
        )
        add_hybrid(net, src_edge, dest_edge)
        retics = [n for n in net.V() if n.is_reticulation()]
        assert len(retics) == 1

    def test_leaf_set_preserved(self):
        net = _build_balanced_tree()
        leaves_before = _leaf_labels(net)
        src_edge = net.get_edge(
            _get_node(net, "I1"), _get_node(net, "A")
        )
        dest_edge = net.get_edge(
            _get_node(net, "I2"), _get_node(net, "C")
        )
        add_hybrid(net, src_edge, dest_edge)
        assert _leaf_labels(net) == leaves_before

    def test_time_parameters_applied(self):
        net = _build_timed_tree()
        src_edge = net.get_edge(
            _get_node(net, "I1"), _get_node(net, "A")
        )
        dest_edge = net.get_edge(
            _get_node(net, "I2"), _get_node(net, "C")
        )
        add_hybrid(net, src_edge, dest_edge, t_src=1.5, t_dest=2.0)
        uid_nodes = [n for n in net.V() if n.label.startswith("UID")]
        times = sorted(n.get_time() for n in uid_nodes)
        assert 1.5 in times
        assert 2.0 in times

    def test_bubble_creation(self):
        net = _build_balanced_tree()
        edge = net.get_edge(_get_node(net, "R"), _get_node(net, "I1"))
        with pytest.warns(UserWarning):
            add_hybrid(net, edge, edge)
        uid_nodes = [n for n in net.V() if n.label.startswith("UID")]
        assert len(uid_nodes) == 2

    def test_bubble_creates_two_parallel_edges(self):
        net = _build_balanced_tree()
        edge = net.get_edge(_get_node(net, "R"), _get_node(net, "I1"))
        with pytest.warns(UserWarning):
            add_hybrid(net, edge, edge)
        uid_nodes = sorted(
            [n for n in net.V() if n.label.startswith("UID")],
            key=lambda n: n.label,
        )
        n3, n4 = uid_nodes[0], uid_nodes[1]
        parallel = [e for e in net.E() if e.src == n3 and e.dest == n4]
        assert len(parallel) == 2
        gammas = sorted(e.get_gamma() for e in parallel)
        assert gammas == [0.5, 0.5]

    def test_root_preserved_after_add(self):
        net = _build_balanced_tree()
        root_label = net.root().label
        src_edge = net.get_edge(
            _get_node(net, "I1"), _get_node(net, "B")
        )
        dest_edge = net.get_edge(
            _get_node(net, "I2"), _get_node(net, "D")
        )
        add_hybrid(net, src_edge, dest_edge)
        assert net.root().label == root_label


# ===================================================================
# remove_hybrid
# ===================================================================


class TestRemoveHybrid:
    """Tests for remove_hybrid."""

    def _add_then_get_hybrid_edge(self, net: Network) -> Edge:
        src_edge = net.get_edge(
            _get_node(net, "I1"), _get_node(net, "A")
        )
        dest_edge = net.get_edge(
            _get_node(net, "I2"), _get_node(net, "C")
        )
        add_hybrid(net, src_edge, dest_edge)
        retic = [n for n in net.V() if n.is_reticulation()][0]
        non_retic_parents = [
            p for p in net.get_parents(retic) if not p.is_reticulation()
        ]
        return net.get_edge(non_retic_parents[0], retic)

    def test_roundtrip_restores_tree(self):
        net = _build_balanced_tree()
        n_v = len(net.V())
        n_e = len(net.E())
        leaves_before = _leaf_labels(net)
        hybrid_edge = self._add_then_get_hybrid_edge(net)
        remove_hybrid(net, hybrid_edge)
        assert level(net) == 0
        assert len(net.V()) == n_v
        assert len(net.E()) == n_e
        assert _leaf_labels(net) == leaves_before

    def test_error_on_non_hybrid_edge(self):
        net = _build_balanced_tree()
        tree_edge = net.get_edge(_get_node(net, "R"), _get_node(net, "I1"))
        with pytest.raises(NetworkError):
            remove_hybrid(net, tree_edge)

    def test_error_on_reticulation_source(self):
        """Construct a chain of two reticulations so the hybrid edge's
        source is itself a reticulation."""
        net = _build_network(
            ["R", "A", "B", "X", "Y", "H1", "H2"],
            [
                ("R", "A", None),
                ("R", "X", None),
                ("A", "H1", None),
                ("X", "H1", 0.5),
                ("H1", "H2", 0.3),
                ("A", "H2", 0.7),
                ("H2", "B", None),
                ("X", "Y", None),
            ],
            retic_labels={"H1", "H2"},
        )
        edge_h1_h2 = net.get_edge(_get_node(net, "H1"), _get_node(net, "H2"))
        with pytest.raises(NetworkError, match="source.*reticulation"):
            remove_hybrid(net, edge_h1_h2)

    def test_leaf_set_restored(self):
        net = _build_balanced_tree()
        leaves_before = _leaf_labels(net)
        hybrid_edge = self._add_then_get_hybrid_edge(net)
        remove_hybrid(net, hybrid_edge)
        assert _leaf_labels(net) == leaves_before


# ===================================================================
# nni
# ===================================================================


class TestNNI:
    """Tests for nearest-neighbor interchange."""

    def test_preserves_leaf_set(self):
        random.seed(42)
        net = _build_balanced_tree()
        leaves_before = _leaf_labels(net)
        nni(net)
        assert _leaf_labels(net) == leaves_before

    def test_preserves_node_count(self):
        random.seed(42)
        net = _build_balanced_tree()
        n = len(net.V())
        nni(net)
        assert len(net.V()) == n

    def test_preserves_edge_count(self):
        random.seed(42)
        net = _build_balanced_tree()
        e = len(net.E())
        nni(net)
        assert len(net.E()) == e

    def test_root_preserved(self):
        random.seed(42)
        net = _build_balanced_tree()
        r = net.root().label
        nni(net)
        assert net.root().label == r

    def test_actually_changes_topology(self):
        """Run NNI many times on a 5-leaf tree; at least once the edge
        set should differ from the original."""
        original = _all_edge_pairs(_build_5leaf_tree())
        changed = False
        for seed in range(100):
            net = _build_5leaf_tree()
            random.seed(seed)
            try:
                nni(net)
            except NetworkError:
                continue
            if _all_edge_pairs(net) != original:
                changed = True
                break
        assert changed, "NNI never changed the topology in 100 attempts"

    def test_edge_attributes_preserved(self):
        """Build a tree with custom branch lengths and verify they survive."""
        net = _build_network(
            ["R", "I1", "I2", "A", "B", "C", "D"],
            [
                ("R", "I1", None),
                ("R", "I2", None),
                ("I1", "A", None),
                ("I1", "B", None),
                ("I2", "C", None),
                ("I2", "D", None),
            ],
        )
        for e in net.E():
            e.set_length(5.0)
            e.set_weight(2.0)

        random.seed(42)
        nni(net)
        for e in net.E():
            assert e.get_length() in (1.0, 5.0)
            assert e.get_weight() in (0.0, 2.0)

    def test_no_internal_edges_raises(self):
        """A star tree (root → all leaves) has no internal edge."""
        net = _build_network(
            ["R", "A", "B", "C"],
            [("R", "A", None), ("R", "B", None), ("R", "C", None)],
        )
        with pytest.raises(NetworkError, match="No internal edges"):
            nni(net)

    def test_nni_skips_reticulation_edges(self):
        """If all internal edges touch a reticulation, NNI should raise."""
        net = _build_network(
            ["R", "H", "A", "B"],
            [
                ("R", "H", 0.5),
                ("R", "H", 0.5),
                ("H", "A", None),
                ("H", "B", None),
            ],
            retic_labels={"H"},
        )
        with pytest.raises(NetworkError):
            nni(net)

    def test_repeated_nni_keeps_valid_tree(self):
        """Apply NNI 20 times; tree should remain structurally valid."""
        random.seed(123)
        net = _build_5leaf_tree()
        leaves = _leaf_labels(net)
        for _ in range(20):
            try:
                nni(net)
            except NetworkError:
                pass
        assert _leaf_labels(net) == leaves
        assert net.root() is not None


# ===================================================================
# node_height_change
# ===================================================================


class TestNodeHeightChange:
    """Tests for node_height_change."""

    def test_basic_move(self):
        net = _build_timed_tree()
        i1 = _get_node(net, "I1")
        node_height_change(i1, net, 1.5)
        assert i1.get_time() == 1.5

    def test_move_to_boundary_of_range_raises(self):
        """Height equal to a parent or child time is out of bounds."""
        net = _build_timed_tree()
        i1 = _get_node(net, "I1")
        with pytest.raises(NetworkError, match="out of bounds"):
            node_height_change(i1, net, 0.0)
        with pytest.raises(NetworkError, match="out of bounds"):
            node_height_change(i1, net, 3.0)

    def test_move_below_parent_raises(self):
        net = _build_timed_tree()
        i1 = _get_node(net, "I1")
        with pytest.raises(NetworkError, match="out of bounds"):
            node_height_change(i1, net, -0.5)

    def test_move_above_child_raises(self):
        net = _build_timed_tree()
        i1 = _get_node(net, "I1")
        with pytest.raises(NetworkError, match="out of bounds"):
            node_height_change(i1, net, 4.0)

    def test_root_has_no_parents_raises(self):
        net = _build_timed_tree()
        root = net.root()
        with pytest.raises(NetworkError, match="parents and children"):
            node_height_change(root, net, 0.5)

    def test_leaf_has_no_children_raises(self):
        net = _build_timed_tree()
        a = _get_node(net, "A")
        with pytest.raises(NetworkError, match="parents and children"):
            node_height_change(a, net, 2.0)

    def test_extend_shifts_children(self):
        """Move I1 from 1.0 to 2.0 with extend; children A,B should
        shift from 3.0 to 4.0."""
        net = _build_network(
            ["R", "I1", "A", "B"],
            [
                ("R", "I1", None),
                ("I1", "A", None),
                ("I1", "B", None),
            ],
            times={"R": 0.0, "I1": 1.0, "A": 3.0, "B": 3.0},
        )
        i1 = _get_node(net, "I1")
        node_height_change(i1, net, 2.0, extend=True)
        assert i1.get_time() == 2.0
        assert _get_node(net, "A").get_time() == pytest.approx(4.0)
        assert _get_node(net, "B").get_time() == pytest.approx(4.0)

    def test_extend_rejects_grandchild_violation(self):
        """If shifting a child would collide with its own child, reject.

        Topology: R(0) -> I1(1) -> I2(2) -> I3(2.5) -> {A(3), B(3)}
        Moving I1 from 1.0 to 1.9 shifts I2 from 2.0 to 2.9, which
        exceeds I3's time of 2.5.
        """
        net = _build_network(
            ["R", "I1", "I2", "I3", "A", "B"],
            [
                ("R", "I1", None),
                ("I1", "I2", None),
                ("I2", "I3", None),
                ("I3", "A", None),
                ("I3", "B", None),
            ],
            times={"R": 0.0, "I1": 1.0, "I2": 2.0, "I3": 2.5, "A": 3.0, "B": 3.0},
        )
        i1 = _get_node(net, "I1")
        with pytest.raises(NetworkError, match="time ordering"):
            node_height_change(i1, net, 1.9, extend=True)

    def test_extend_no_grandchildren_is_fine(self):
        """Leaf children have no grandchildren, so extend always works
        within the parent/child bounds."""
        net = _build_network(
            ["R", "I1", "A", "B"],
            [
                ("R", "I1", None),
                ("I1", "A", None),
                ("I1", "B", None),
            ],
            times={"R": 0.0, "I1": 1.0, "A": 5.0, "B": 5.0},
        )
        i1 = _get_node(net, "I1")
        node_height_change(i1, net, 2.5, extend=True)
        assert _get_node(net, "A").get_time() == pytest.approx(6.5)


# ===================================================================
# _suppress_chain_node
# ===================================================================


class TestSuppressChainNode:
    """Tests for _suppress_chain_node."""

    def test_chain_node_removed(self):
        """Insert a chain node X between R and I1, then suppress it."""
        net = _build_network(
            ["R", "X", "I1", "A", "B"],
            [
                ("R", "X", None),
                ("X", "I1", None),
                ("I1", "A", None),
                ("I1", "B", None),
            ],
        )
        x = _get_node(net, "X")
        _suppress_chain_node(net, x)
        assert "X" not in _node_labels(net)
        assert ("R", "I1") in _all_edge_pairs(net)

    def test_lengths_summed(self):
        net = _build_network(
            ["R", "X", "L"],
            [("R", "X", None), ("X", "L", None)],
        )
        net.get_edge(_get_node(net, "R"), _get_node(net, "X")).set_length(2.0)
        net.get_edge(_get_node(net, "X"), _get_node(net, "L")).set_length(3.0)
        x = _get_node(net, "X")
        _suppress_chain_node(net, x)
        merged = net.get_edge(_get_node(net, "R"), _get_node(net, "L"))
        assert merged.get_length() == pytest.approx(5.0)

    def test_weights_summed(self):
        net = _build_network(
            ["R", "X", "L"],
            [("R", "X", None), ("X", "L", None)],
        )
        net.get_edge(_get_node(net, "R"), _get_node(net, "X")).set_weight(1.0)
        net.get_edge(_get_node(net, "X"), _get_node(net, "L")).set_weight(4.0)
        x = _get_node(net, "X")
        _suppress_chain_node(net, x)
        merged = net.get_edge(_get_node(net, "R"), _get_node(net, "L"))
        assert merged.get_weight() == pytest.approx(5.0)

    def test_noop_for_non_chain_node(self):
        """A node with out-degree 2 should not be suppressed."""
        net = _build_balanced_tree()
        i1 = _get_node(net, "I1")
        nodes_before = len(net.V())
        _suppress_chain_node(net, i1)
        assert len(net.V()) == nodes_before

    def test_gamma_propagated(self):
        net = _build_network(
            ["R", "X", "L"],
            [("R", "X", 0.0), ("X", "L", 0.7)],
        )
        x = _get_node(net, "X")
        _suppress_chain_node(net, x)
        merged = net.get_edge(_get_node(net, "R"), _get_node(net, "L"))
        assert merged.get_gamma() == pytest.approx(0.7)


# ===================================================================
# spr
# ===================================================================


class TestSPR:
    """Tests for subtree prune-and-regraft."""

    def test_preserves_leaf_set(self):
        random.seed(7)
        net = _build_5leaf_tree()
        leaves = _leaf_labels(net)
        spr(net)
        assert _leaf_labels(net) == leaves

    def test_stays_level_zero(self):
        random.seed(7)
        net = _build_5leaf_tree()
        spr(net)
        assert level(net) == 0

    def test_root_preserved(self):
        random.seed(7)
        net = _build_5leaf_tree()
        r = net.root().label
        spr(net)
        assert net.root().label == r

    def test_edge_count_stable(self):
        """For a bifurcating tree, SPR should keep 2n-2 edges (n = leaves)."""
        random.seed(7)
        net = _build_5leaf_tree()
        e_before = len(net.E())
        spr(net)
        assert len(net.E()) == e_before

    def test_total_branch_length_preserved(self):
        """SPR should redistribute but preserve total branch length (all
        edges default to length 1, total should stay the same)."""
        random.seed(7)
        net = _build_5leaf_tree()
        total_before = sum(e.get_length() for e in net.E())
        spr(net)
        total_after = sum(e.get_length() for e in net.E())
        assert total_after == pytest.approx(total_before)

    def test_actually_changes_topology(self):
        original = _all_edge_pairs(_build_5leaf_tree())
        changed = False
        for seed in range(100):
            net = _build_5leaf_tree()
            random.seed(seed)
            try:
                spr(net)
            except NetworkError:
                continue
            if _all_edge_pairs(net) != original:
                changed = True
                break
        assert changed, "SPR never changed the topology in 100 attempts"

    def test_rejects_non_tree(self):
        net = _build_balanced_tree()
        src = net.get_edge(_get_node(net, "I1"), _get_node(net, "A"))
        dst = net.get_edge(_get_node(net, "I2"), _get_node(net, "C"))
        add_hybrid(net, src, dst)
        assert level(net) >= 1
        with pytest.raises(NetworkError, match="level-0"):
            spr(net)

    def test_too_small_tree_raises(self):
        """A tree with only a root and one leaf has no valid prune edge."""
        net = _build_network(
            ["R", "A"],
            [("R", "A", None)],
        )
        with pytest.raises(NetworkError):
            spr(net)

    def test_repeated_spr_keeps_valid_tree(self):
        random.seed(999)
        net = _build_5leaf_tree()
        leaves = _leaf_labels(net)
        for _ in range(30):
            try:
                spr(net)
            except NetworkError:
                pass
        assert _leaf_labels(net) == leaves
        assert level(net) == 0
        assert net.root() is not None

    def test_three_leaf_tree(self):
        """Minimal tree that allows SPR: R -> (I1 -> (A, B), C)."""
        net = _build_network(
            ["R", "I1", "A", "B", "C"],
            [
                ("R", "I1", None),
                ("R", "C", None),
                ("I1", "A", None),
                ("I1", "B", None),
            ],
        )
        random.seed(0)
        spr(net)
        assert _leaf_labels(net) == {"A", "B", "C"}
        assert level(net) == 0


# ===================================================================
# permute_leaves
# ===================================================================


class TestPermuteLeaves:
    """Tests for permute_leaves."""

    def test_returns_generator(self):
        net = _build_balanced_tree()
        result = permute_leaves(net)
        assert isinstance(result, types.GeneratorType)

    def test_correct_count(self):
        """4 leaves → 4! = 24 permutations."""
        net = _build_balanced_tree()
        count = sum(1 for _ in permute_leaves(net))
        assert count == 24

    def test_each_network_has_same_leaves(self):
        net = _build_balanced_tree()
        expected_leaves = _leaf_labels(net)
        for pn in permute_leaves(net):
            assert _leaf_labels(pn) == expected_leaves

    def test_original_not_modified(self):
        net = _build_balanced_tree()
        edges_before = _all_edge_pairs(net)
        leaves_before = _leaf_labels(net)
        _ = list(islice(permute_leaves(net), 5))
        assert _all_edge_pairs(net) == edges_before
        assert _leaf_labels(net) == leaves_before

    def test_identity_permutation_included(self):
        """At least one yielded network should have the same edge set as
        a fresh copy of the original."""
        net = _build_balanced_tree()
        copy_net, _ = net.copy()
        copy_edges = _all_edge_pairs(copy_net)
        found = any(
            _all_edge_pairs(pn) == copy_edges
            for pn in permute_leaves(net)
        )
        assert found

    def test_empty_network_yields_nothing(self):
        net = Network()
        assert list(permute_leaves(net)) == []

    def test_single_leaf(self):
        """One leaf → 1! = 1 permutation."""
        net = _build_network(["R", "A"], [("R", "A", None)])
        perms = list(permute_leaves(net))
        assert len(perms) == 1

    def test_edge_attributes_travel_with_leaf(self):
        """Branch length on the pendant edge should follow the leaf."""
        net = _build_network(
            ["R", "I1", "A", "B"],
            [
                ("R", "I1", None),
                ("I1", "A", None),
                ("I1", "B", None),
            ],
        )
        net.get_edge(_get_node(net, "I1"), _get_node(net, "A")).set_length(10.0)
        net.get_edge(_get_node(net, "I1"), _get_node(net, "B")).set_length(20.0)

        for pn in permute_leaves(net):
            lengths = set()
            for lf in pn.get_leaves():
                par = pn.get_parents(lf)[0]
                lengths.add(pn.get_edge(par, lf).get_length())
            assert lengths == {10.0, 20.0}


# ===================================================================
# Integration / roundtrip tests
# ===================================================================


class TestIntegration:
    """Cross-function integration tests."""

    def test_add_then_remove_is_identity_on_leaves(self):
        net = _build_5leaf_tree()
        leaves = _leaf_labels(net)
        src = net.get_edge(_get_node(net, "I1"), _get_node(net, "A"))
        dst = net.get_edge(_get_node(net, "I2"), _get_node(net, "C"))
        add_hybrid(net, src, dst)
        retic = [n for n in net.V() if n.is_reticulation()][0]
        h_parent = [
            p for p in net.get_parents(retic) if not p.is_reticulation()
        ][0]
        h_edge = net.get_edge(h_parent, retic)
        remove_hybrid(net, h_edge)
        assert _leaf_labels(net) == leaves
        assert level(net) == 0

    def test_nni_then_spr_on_same_tree(self):
        random.seed(77)
        net = _build_5leaf_tree()
        leaves = _leaf_labels(net)
        nni(net)
        assert _leaf_labels(net) == leaves
        assert level(net) == 0
        spr(net)
        assert _leaf_labels(net) == leaves
        assert level(net) == 0

    def test_height_change_between_nni(self):
        random.seed(10)
        net = _build_timed_tree()
        i1 = _get_node(net, "I1")
        node_height_change(i1, net, 2.0)
        assert i1.get_time() == 2.0


# ===================================================================
# Utility used across tests
# ===================================================================


def _get_node(net: Network, label: str) -> Node:
    """Retrieve a node from a network by label."""
    for n in net.V():
        if n.label == label:
            return n
    raise ValueError(f"Node '{label}' not found in network")
