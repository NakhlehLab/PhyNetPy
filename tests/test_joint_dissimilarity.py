"""
Tests for ``phynetpy.JointDissimilarity`` -- the joint reticulation-and-tree
dissimilarity ``D_joint = D_ret + 3 D_tree``.

Hand-computed values use two one-reticulation networks that differ only in
which taxon the hybrid edge delivers::

    N1 = ((A,(B)#H1),(#H1,C));    hybrid block {B}, parent sides {A} and {C}
    N2 = (((A)#H1,B),(#H1,C));    hybrid block {A}, parent sides {B} and {C}

Under the ``leaf`` instantiation the tripartition cost is
``d({B},{A}) + min(d({A},{B}) + d({C},{C}), d({A},{C}) + d({C},{B}))
= 1 + 1 = 2``.  The remainders keep the taxa outside the hybrid block plus
one marker leaf per hybrid edge, so their label sets are ``{A, C, m(B)}`` and
``{B, C, m(A)}``: four of five labels differ, ``D_tree = 0.8``.
"""

import pytest

from phynetpy import joint_dissimilarity as top_level_joint_dissimilarity
from phynetpy.GraphUtils import JOINT_KINDS, joint_dissimilarity
from phynetpy.IO import read_newick
from phynetpy.Network import NetworkError
from phynetpy.SNPSimulator import random_network

N1 = "((A,(B)#H1),(#H1,C));"
N2 = "(((A)#H1,B),(#H1,C));"
TREE = "((A,B),C);"
OTHER_TREE = "((A,C),B);"

# The fixtures carry no branch lengths; the reader warns and defaults them.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TestHandComputedValues:
    def test_two_networks_under_leaf(self):
        r = joint_dissimilarity(read_newick(N1), read_newick(N2), "leaf")
        assert (r.r1, r.r2, r.m) == (1, 1, 1)
        assert r.D_ret == pytest.approx(2.0)
        assert r.D_tree == pytest.approx(0.8)
        assert r.D_joint == pytest.approx(4.4)
        assert r.D_ret_normalized == pytest.approx(2 / 3)
        assert r.D_joint_normalized == pytest.approx(4.4 / 6)

    def test_tree_against_network_pays_the_dummy_cost(self):
        # The tree has no tripartition, so the network's one is matched to a
        # dummy at cost 3; the remainders differ in B versus the marker m(B).
        r = joint_dissimilarity(read_newick(TREE), read_newick(N1), "leaf")
        assert (r.r1, r.r2) == (0, 1)
        assert r.D_ret == pytest.approx(3.0)
        assert r.D_ret_normalized == pytest.approx(1.0)
        assert r.D_tree == pytest.approx(0.5)

    def test_trees_are_told_apart_by_structure_not_leaves(self):
        t, u = read_newick(TREE), read_newick(OTHER_TREE)
        assert joint_dissimilarity(t, u, "leaf").D_joint == 0.0
        # Clusters {A,B} vs {A,C} differ: 2 of 6 distinct clusters.
        assert joint_dissimilarity(t, u, "cluster").D_tree == pytest.approx(1 / 3)
        # Nakhleh classes: 3 shared leaf classes, 4 unshared internal ones.
        assert joint_dissimilarity(t, u, "nakhleh").D_tree == pytest.approx(4 / 7)


class TestMetricProperties:
    @pytest.mark.parametrize("kind", JOINT_KINDS)
    def test_identity_is_zero_under_every_instantiation(self, kind):
        for newick in (N1, TREE):
            a, b = read_newick(newick), read_newick(newick)
            r = joint_dissimilarity(a, b, kind)
            assert r.D_ret == 0.0 and r.D_tree == 0.0 and r.D_joint == 0.0

    def test_internal_labels_and_branch_lengths_are_ignored(self):
        a = read_newick("((A:0.1,B:0.2)X:0.3,C:0.4)R;")
        b = read_newick("((A,B)Y,C)S;")
        assert joint_dissimilarity(a, b, "nakhleh").D_joint == 0.0

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("kind", JOINT_KINDS)
    def test_symmetric_and_normalised_in_unit_interval(self, seed, kind):
        a = random_network(n=8, level=2, seed=seed)
        b = random_network(n=8, level=1, seed=seed + 100)
        ab = joint_dissimilarity(a, b, kind)
        ba = joint_dissimilarity(b, a, kind)
        assert ab.D_ret == pytest.approx(ba.D_ret)
        assert ab.D_tree == pytest.approx(ba.D_tree)
        assert 0.0 <= ab.D_ret_normalized <= 1.0
        assert 0.0 <= ab.D_joint_normalized <= 1.0
        assert ab.D_joint == pytest.approx(ab.D_ret + 3 * ab.D_tree)


class TestSurface:
    def test_network_method_and_reexports_agree(self):
        a, b = read_newick(N1), read_newick(N2)
        direct = joint_dissimilarity(a, b)
        assert direct.kind == "nakhleh"
        assert a.joint_dissimilarity(b) == direct
        assert top_level_joint_dissimilarity(a, b) == direct

    def test_unknown_kind_raises_value_error(self):
        with pytest.raises(ValueError):
            joint_dissimilarity(read_newick(N1), read_newick(N2), "jaccard")

    def test_mismatched_taxa_raise(self):
        with pytest.raises(NetworkError):
            joint_dissimilarity(read_newick(TREE), read_newick("((A,B),D);"))

    def test_reticulation_with_two_children_is_refused(self):
        with pytest.raises(NetworkError, match="children"):
            joint_dissimilarity(read_newick("((A,(B,C)#H1),(#H1,D));"), read_newick("(((A,B),C),D);"))
