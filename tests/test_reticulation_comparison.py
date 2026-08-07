"""
Test suite for the reticulation-aware dissimilarity in
``phynetpy.ReticulationComparison`` (Nakhleh, *Reticulation-Aware
Dissimilarity for Phylogenetic Networks*, 2026).

Coverage follows the validation protocol of Section 10 of the note:

    - Definition 1 : reticulation tripartitions (A_h, B_h, C_h)
    - Equation 3   : tripartition dissimilarity delta (metric axioms)
    - Equation 4   : reticulation dissimilarity D_ret via padded matching
    - Equation 5   : the global metric D (Nakhleh 2010)
    - Definition 4 : reticulation precision / recall / F1
    - Section 6.1  : boundary behaviour (trees, tree-vs-network, self)
    - Section 8    : topology-aware refinement (leaf-set vs D*_ret)
    - E5           : boundary sanity checks
"""

from __future__ import annotations

import pytest
from typing import Optional, Sequence, Tuple

from phynetpy.Network import Network, Node, Edge, NetworkError
from phynetpy.ReticulationComparison import (
    JACCARD,
    SYMMETRIC,
    ReticulationTripartition,
    reticulation_tripartitions,
    set_distance,
    tripartition_dissimilarity,
    nakhleh_metric,
    reticulation_dissimilarity,
    combined_dissimilarity,
    compare_networks,
)


# ===================================================================
# Network builders
# ===================================================================

EdgeSpec = Tuple[str, str, Optional[float]]


def _build_network(
    node_labels: Sequence[str],
    edge_specs: Sequence[EdgeSpec],
    retic_labels: Optional[set] = None,
) -> Network:
    net = Network()
    retic_set = set(retic_labels or [])

    ordered_labels: list = []
    seen: set = set()
    for label in list(node_labels) + list(retic_set):
        if label not in seen:
            ordered_labels.append(label)
            seen.add(label)

    nodes = {}
    for label in ordered_labels:
        nodes[label] = Node(label, is_reticulation=label in retic_set)

    net.add_nodes(*nodes.values())
    for src, dest, gamma in edge_specs:
        net.add_edges(Edge(nodes[src], nodes[dest], gamma=gamma))
    return net


def _e(src: str, dest: str, gamma: Optional[float] = None) -> EdgeSpec:
    return (src, dest, gamma)


# ── Tree ((A,B),C) on {A,B,C} ──
def build_tree_ABC() -> Network:
    return _build_network(
        ["Root", "I1", "A", "B", "C"],
        [_e("Root", "I1"), _e("Root", "C"), _e("I1", "A"), _e("I1", "B")],
    )


# ── Tree ((A,C),B) on {A,B,C} ──
def build_tree_ACB() -> Network:
    return _build_network(
        ["Root", "I1", "A", "B", "C"],
        [_e("Root", "I1"), _e("Root", "B"), _e("I1", "A"), _e("I1", "C")],
    )


# ── One-reticulation network on {A,B,C}: hybrid C, parents keep A and B ──
#   Root -> U, V ;  U -> A, H ;  V -> B, H ;  H -> C
#   tripartition:  A_h = {A}, B_h = {C}, C_h = {B}
def build_retic_ACB_hybridC() -> Network:
    return _build_network(
        ["Root", "U", "V", "#H", "A", "B", "C"],
        [
            _e("Root", "U"), _e("Root", "V"),
            _e("U", "A"), _e("U", "#H", 0.5),
            _e("V", "B"), _e("V", "#H", 0.5),
            _e("#H", "C"),
        ],
        retic_labels={"#H"},
    )


# ── Different reticulation scenario on {A,B,C}: hybrid A, parents keep B,C ──
#   tripartition:  A_h = {B}, B_h = {A}, C_h = {C}
def build_retic_BCA_hybridA() -> Network:
    return _build_network(
        ["Root", "U", "V", "#H", "A", "B", "C"],
        [
            _e("Root", "U"), _e("Root", "V"),
            _e("U", "B"), _e("U", "#H", 0.5),
            _e("V", "C"), _e("V", "#H", 0.5),
            _e("#H", "A"),
        ],
        retic_labels={"#H"},
    )


# ── Two networks with identical reticulation leaf sets but different
#    substructure beneath the hybrid (for the topology-aware test, E4). ──
#    Taxa {A,B,C,D,E}; hybrid cluster {C,D,E} either as ((C,D),E) or ((C,E),D).
def _build_topo_case(shape: str) -> Network:
    #   Root -> U, V ; U -> A, H ; V -> B, H ; H -> M
    #   shape "CD": M -> N, E ; N -> C, D
    #   shape "CE": M -> N, D ; N -> C, E
    edges = [
        _e("Root", "U"), _e("Root", "V"),
        _e("U", "A"), _e("U", "#H", 0.5),
        _e("V", "B"), _e("V", "#H", 0.5),
        _e("#H", "M"),
    ]
    if shape == "CD":
        edges += [_e("M", "N"), _e("M", "E"), _e("N", "C"), _e("N", "D")]
    else:
        edges += [_e("M", "N"), _e("M", "D"), _e("N", "C"), _e("N", "E")]
    return _build_network(
        ["Root", "U", "V", "#H", "M", "N", "A", "B", "C", "D", "E"],
        edges,
        retic_labels={"#H"},
    )


# ===================================================================
# Set distances (Equations 1--2)
# ===================================================================

class TestSetDistance:

    def test_symmetric_difference(self):
        assert set_distance(frozenset("ab"), frozenset("bc"), SYMMETRIC) == 2.0

    def test_jaccard_range(self):
        d = set_distance(frozenset("ab"), frozenset("bc"), JACCARD)
        assert d == pytest.approx(2.0 / 3.0)

    def test_both_empty(self):
        assert set_distance(frozenset(), frozenset(), JACCARD) == 0.0
        assert set_distance(frozenset(), frozenset(), SYMMETRIC) == 0.0

    def test_identity(self):
        assert set_distance(frozenset("abc"), frozenset("abc"), JACCARD) == 0.0

    def test_unknown_raises(self):
        with pytest.raises(NetworkError):
            set_distance(frozenset("a"), frozenset("b"), "nonsense")


# ===================================================================
# Reticulation tripartitions (Definition 1)
# ===================================================================

class TestReticulationTripartitions:

    def test_tree_has_no_tripartitions(self):
        assert reticulation_tripartitions(build_tree_ABC()) == []

    def test_single_reticulation(self):
        trips = reticulation_tripartitions(build_retic_ACB_hybridC())
        assert len(trips) == 1
        theta = trips[0]
        assert theta.B == frozenset({"C"})
        assert {frozenset(theta.A), frozenset(theta.C)} == {
            frozenset({"A"}), frozenset({"B"})
        }

    def test_middle_disjoint_from_outer(self):
        theta = reticulation_tripartitions(build_retic_ACB_hybridC())[0]
        assert theta.A & theta.B == frozenset()
        assert theta.C & theta.B == frozenset()

    def test_properness(self):
        theta = reticulation_tripartitions(build_retic_ACB_hybridC())[0]
        assert theta.is_proper()

    def test_swapped(self):
        theta = ReticulationTripartition(
            frozenset({"A"}), frozenset({"C"}), frozenset({"B"}), "#H"
        )
        s = theta.swapped()
        assert s.A == theta.C and s.C == theta.A and s.B == theta.B


# ===================================================================
# Tripartition dissimilarity delta (Equation 3, Proposition 2)
# ===================================================================

class TestTripartitionDissimilarity:

    def _t1(self):
        return ReticulationTripartition(
            frozenset({"A"}), frozenset({"C"}), frozenset({"B"}), "h1"
        )

    def _t2(self):
        return ReticulationTripartition(
            frozenset({"B"}), frozenset({"A"}), frozenset({"C"}), "h2"
        )

    def test_identity(self):
        assert tripartition_dissimilarity(self._t1(), self._t1()) == 0.0

    def test_swap_invariance(self):
        t1 = self._t1()
        assert tripartition_dissimilarity(t1, t1.swapped()) == 0.0

    def test_symmetry(self):
        a, b = self._t1(), self._t2()
        assert (tripartition_dissimilarity(a, b)
                == tripartition_dissimilarity(b, a))

    def test_known_value(self):
        # mid d({C},{A}) = 1; outer swapped = d({A},{C}) + d({B},{B}) = 1;
        # delta = 1 + min(2, 1) = 2 under Jaccard.
        assert tripartition_dissimilarity(self._t1(), self._t2()) == 2.0

    def test_normalized_in_unit_interval(self):
        d = tripartition_dissimilarity(self._t1(), self._t2(), normalize=True)
        assert 0.0 <= d <= 1.0
        assert d == pytest.approx(2.0 / 3.0)

    def test_triangle_inequality(self):
        a = self._t1()
        b = self._t2()
        c = ReticulationTripartition(
            frozenset({"A"}), frozenset({"B"}), frozenset({"C"}), "h3"
        )
        dab = tripartition_dissimilarity(a, b)
        dbc = tripartition_dissimilarity(b, c)
        dac = tripartition_dissimilarity(a, c)
        assert dac <= dab + dbc + 1e-9


# ===================================================================
# Global metric D (Equation 5)
# ===================================================================

class TestNakhlehMetric:

    def test_identity_tree(self):
        assert nakhleh_metric(build_tree_ABC(), build_tree_ABC()) == 0.0

    def test_identity_network(self):
        n = build_retic_ACB_hybridC()
        assert nakhleh_metric(n, build_retic_ACB_hybridC()) == 0.0

    def test_symmetry(self):
        a, b = build_tree_ABC(), build_tree_ACB()
        assert nakhleh_metric(a, b) == nakhleh_metric(b, a)

    def test_different_trees_positive(self):
        assert nakhleh_metric(build_tree_ABC(), build_tree_ACB()) > 0.0

    def test_normalized_range(self):
        d = nakhleh_metric(build_tree_ABC(), build_tree_ACB(), normalize=True)
        assert 0.0 < d <= 1.0

    def test_triangle_inequality(self):
        a = build_tree_ABC()
        b = build_tree_ACB()
        c = build_retic_ACB_hybridC()
        assert nakhleh_metric(a, c) <= nakhleh_metric(a, b) + nakhleh_metric(b, c) + 1e-9


# ===================================================================
# Reticulation dissimilarity D_ret + precision/recall (Eq. 4, Def. 4)
# ===================================================================

class TestReticulationDissimilarity:

    def test_self_comparison_is_zero(self):
        n = build_retic_ACB_hybridC()
        result = compare_networks(n, build_retic_ACB_hybridC())
        assert result.D_ret == 0.0
        assert result.D_ret_hat == 0.0
        assert result.D == 0.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.recovered == 1

    def test_two_trees_zero_reticulation_signal(self):
        # Section 6.1: two trees agree perfectly on having no reticulations.
        result = compare_networks(build_tree_ABC(), build_tree_ACB())
        assert result.D_ret == 0.0
        assert result.D_ret_hat == 0.0
        # ...but the global term still separates the distinct trees.
        assert result.D > 0.0

    def test_network_vs_its_underlying_tree(self):
        # Section 6.1 / E5(b): network as inference -> D_ret_hat = 1, Prec = 0.
        tree = build_tree_ABC()
        net = build_retic_ACB_hybridC()
        result = compare_networks(tree, net)   # reference=tree, inferred=net
        assert result.D_ret_hat == pytest.approx(1.0)
        assert result.precision == 0.0
        assert result.recall == 1.0            # nothing to recover in the tree
        assert result.f1 == 0.0

    def test_tree_as_inference_flags_false_negatives(self):
        net = build_retic_ACB_hybridC()
        tree = build_tree_ABC()
        result = compare_networks(net, tree)   # reference=net, inferred=tree
        assert result.D_ret_hat == pytest.approx(1.0)
        assert result.precision == 1.0         # no inferred events => vacuous
        assert result.recall == 0.0            # the true event was missed

    def test_wrong_reticulation_scenario(self):
        ref = build_retic_ACB_hybridC()
        inf = build_retic_BCA_hybridA()
        result = compare_networks(ref, inf)
        # delta = 2 (hand-computed); single real-real pair, m = 1, delta_max = 3.
        assert result.D_ret == pytest.approx(2.0)
        assert result.D_ret_hat == pytest.approx(2.0 / 3.0)
        # Not recovered at tau = 0.
        assert result.recovered == 0
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_tolerance_allows_recovery(self):
        ref = build_retic_ACB_hybridC()
        inf = build_retic_BCA_hybridA()
        result = compare_networks(ref, inf, tolerance=2.0)
        assert result.recovered == 1
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_symmetry_of_dret(self):
        a = build_retic_ACB_hybridC()
        b = build_retic_BCA_hybridA()
        assert (reticulation_dissimilarity(a, b, normalize=False)
                == reticulation_dissimilarity(b, a, normalize=False))

    def test_result_is_iterable_tuple(self):
        result = compare_networks(build_tree_ABC(), build_tree_ACB())
        D, D_ret_hat, prec, rec = result
        assert (D, D_ret_hat, prec, rec) == result.as_tuple()

    def test_normalized_scores_in_unit_interval(self):
        result = compare_networks(
            build_retic_ACB_hybridC(), build_retic_BCA_hybridA()
        )
        assert 0.0 <= result.D_ret_hat <= 1.0
        assert 0.0 <= result.D_hat <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0


# ===================================================================
# Combined dissimilarity D_lambda (Equation 6, Proposition 5)
# ===================================================================

class TestCombinedDissimilarity:

    def test_lambda_zero_is_global(self):
        a, b = build_tree_ABC(), build_tree_ACB()
        assert combined_dissimilarity(a, b, lam=0.0) == pytest.approx(
            nakhleh_metric(a, b)
        )

    def test_lambda_one_is_reticulation(self):
        a = build_retic_ACB_hybridC()
        b = build_retic_BCA_hybridA()
        assert combined_dissimilarity(a, b, lam=1.0) == pytest.approx(
            reticulation_dissimilarity(a, b, normalize=False)
        )

    def test_separates_distinct_trees_for_lambda_below_one(self):
        # D_ret alone conflates the two trees (both have no reticulations),
        # but D_lambda with lambda < 1 keeps them apart.
        a, b = build_tree_ABC(), build_tree_ACB()
        assert combined_dissimilarity(a, b, lam=0.5) > 0.0

    def test_lambda_out_of_range_raises(self):
        with pytest.raises(NetworkError):
            combined_dissimilarity(build_tree_ABC(), build_tree_ACB(), lam=2.0)


# ===================================================================
# Topology-aware refinement (Section 8, E4)
# ===================================================================

class TestTopologyAware:

    def test_leaf_set_conflates_but_topology_resolves(self):
        # Same hybrid leaf set {C,D,E}, different substructure beneath it.
        net_cd = _build_topo_case("CD")
        net_ce = _build_topo_case("CE")

        # alpha = 1 (leaf-set) conflates: D_ret = 0.
        leaf_only = compare_networks(net_cd, net_ce, alpha=1.0)
        assert leaf_only.D_ret == pytest.approx(0.0)

        # alpha < 1 (topology-aware) resolves the difference: D_ret > 0.
        topo_aware = compare_networks(net_cd, net_ce, alpha=0.5)
        assert topo_aware.D_ret > 0.0

    def test_topology_aware_self_is_zero(self):
        net = _build_topo_case("CD")
        result = compare_networks(net, _build_topo_case("CD"), alpha=0.5)
        assert result.D_ret == pytest.approx(0.0)


# ===================================================================
# Input validation
# ===================================================================

class TestValidation:

    def test_bad_alpha(self):
        with pytest.raises(NetworkError):
            compare_networks(build_tree_ABC(), build_tree_ACB(), alpha=1.5)

    def test_bad_tolerance(self):
        with pytest.raises(NetworkError):
            compare_networks(build_tree_ABC(), build_tree_ACB(), tolerance=-1.0)

    def test_bad_distance(self):
        with pytest.raises(NetworkError):
            compare_networks(build_tree_ABC(), build_tree_ACB(), distance="foo")
