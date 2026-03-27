"""
Test suite for the MPL_reference module (maximum pseudo-likelihood scoring).

Validates correctness against hand-computed coalescent triple probabilities
from Yu & Nakhleh (2015) BMC Genomics 16(S10):S10.

Key formula for a resolved 3-taxon species tree ((X,Y):tau, Z):
    P(XY|Z) = 1 - (2/3) exp(-tau)
    P(XZ|Y) = P(YZ|X) = (1/3) exp(-tau)

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import math
from itertools import combinations

import pytest

from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL_reference import (
    MPL,
    _induced_triple,
    _coalescent_triple_probs,
    _compute_rho,
    _subnet_triple_probs,
)
from phynetpy.GraphUtils import subnet_given_leaves, _displayed_trees_with_probs


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _coal_match(tau: float) -> float:
    """P(sister pair) for branch length tau."""
    return 1.0 - (2.0 / 3.0) * math.exp(-tau)


def _coal_mismatch(tau: float) -> float:
    """P(non-sister pair) for branch length tau."""
    return (1.0 / 3.0) * math.exp(-tau)


def _identity_map(*taxa: str) -> dict[str, list[str]]:
    """One-to-one species-to-allele mapping."""
    return {t: [t] for t in taxa}


def _build_mpl(species_newick: str, gene_newicks: list[str],
               mapping: dict[str, list[str]]) -> MPL:
    """Convenience: build an MPL scorer from Newick strings."""
    species_net = Network.from_newick(species_newick)
    gene_tree_list = [Network.from_newick(nwk) for nwk in gene_newicks]
    gts = GeneTrees(gene_tree_list=gene_tree_list)
    gts.species_gene_mapping = mapping
    return MPL(species_net, gts, mapping)


# ═══════════════════════════════════════════════════════════════════════
# 1. _induced_triple — topology recognition on gene trees
# ═══════════════════════════════════════════════════════════════════════

class TestInducedTriple:

    def test_concordant_xy_z(self):
        gt = Network.from_newick("((A:1,B:1):1,C:2);")
        assert _induced_triple(gt, "A", "B", "C") == "xy|z"

    def test_discordant_xz_y(self):
        gt = Network.from_newick("((A:1,C:1):1,B:2);")
        assert _induced_triple(gt, "A", "B", "C") == "xz|y"

    def test_discordant_yz_x(self):
        gt = Network.from_newick("((B:1,C:1):1,A:2);")
        assert _induced_triple(gt, "A", "B", "C") == "yz|x"

    def test_star_topology(self):
        gt = Network.from_newick("(A:1,B:1,C:1);")
        assert _induced_triple(gt, "A", "B", "C") == "star"

    def test_4taxon_embedded_triple(self):
        gt = Network.from_newick("(((A:1,B:1):1,C:2):1,D:3);")
        assert _induced_triple(gt, "A", "B", "D") == "xy|z"
        assert _induced_triple(gt, "A", "C", "D") == "xy|z"
        assert _induced_triple(gt, "B", "C", "D") == "xy|z"


# ═══════════════════════════════════════════════════════════════════════
# 2. _coalescent_triple_probs — closed-form coalescent formula
# ═══════════════════════════════════════════════════════════════════════

class TestCoalescentTripleProbs:

    def test_resolved_tree_tau1(self):
        tree = Network.from_newick("((A:1,B:1):1,C:2);")
        p = _coalescent_triple_probs(tree, "A", "B", "C")
        assert p[0] == pytest.approx(_coal_match(1.0), abs=1e-10)
        assert p[1] == pytest.approx(_coal_mismatch(1.0), abs=1e-10)
        assert p[2] == pytest.approx(_coal_mismatch(1.0), abs=1e-10)

    def test_probs_sum_to_one(self):
        tree = Network.from_newick("((A:1,B:1):2.5,C:3.5);")
        p = _coalescent_triple_probs(tree, "A", "B", "C")
        assert sum(p) == pytest.approx(1.0, abs=1e-12)

    def test_long_branch_approaches_one(self):
        """With a very long internal branch, the sister pair probability -> 1."""
        tree = Network.from_newick("((A:1,B:1):50,C:51);")
        p = _coalescent_triple_probs(tree, "A", "B", "C")
        assert p[0] > 0.999

    def test_zero_branch_gives_uniform(self):
        """tau=0 => P = 1/3 for all topologies (no time for coalescence)."""
        tree = Network.from_newick("((A:1,B:1):0.0,C:1);")
        p = _coalescent_triple_probs(tree, "A", "B", "C")
        assert p[0] == pytest.approx(1.0 / 3.0, abs=1e-10)
        assert p[1] == pytest.approx(1.0 / 3.0, abs=1e-10)
        assert p[2] == pytest.approx(1.0 / 3.0, abs=1e-10)

    def test_different_sister_pair_xz(self):
        """Sister pair is (A,C) => P(XZ|Y) should be the match probability."""
        tree = Network.from_newick("((A:1,C:1):1,B:2);")
        p = _coalescent_triple_probs(tree, "A", "B", "C")
        assert p[1] == pytest.approx(_coal_match(1.0), abs=1e-10)  # P(AC|B)
        assert p[0] == pytest.approx(_coal_mismatch(1.0), abs=1e-10)

    def test_different_sister_pair_yz(self):
        """Sister pair is (B,C) => P(YZ|X) should be the match probability."""
        tree = Network.from_newick("((B:1,C:1):1,A:2);")
        p = _coalescent_triple_probs(tree, "A", "B", "C")
        assert p[2] == pytest.approx(_coal_match(1.0), abs=1e-10)  # P(BC|A)
        assert p[0] == pytest.approx(_coal_mismatch(1.0), abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# 3. _compute_rho — triple frequency accumulation
# ═══════════════════════════════════════════════════════════════════════

class TestComputeRho:

    def test_single_concordant_tree(self):
        gt = Network.from_newick("((A:1,B:1):1,C:2);")
        gts = GeneTrees(gene_tree_list=[gt])
        mapping = _identity_map("A", "B", "C")
        gts.species_gene_mapping = mapping
        rho = _compute_rho("A", "B", "C", gts, mapping)
        assert rho == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)

    def test_single_discordant_tree(self):
        gt = Network.from_newick("((A:1,C:1):1,B:2);")
        gts = GeneTrees(gene_tree_list=[gt])
        mapping = _identity_map("A", "B", "C")
        gts.species_gene_mapping = mapping
        rho = _compute_rho("A", "B", "C", gts, mapping)
        assert rho == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)

    def test_two_trees_mixed(self):
        gt1 = Network.from_newick("((A:1,B:1):1,C:2);")  # AB|C
        gt2 = Network.from_newick("((A:1,C:1):1,B:2);")  # AC|B
        gts = GeneTrees(gene_tree_list=[gt1, gt2])
        mapping = _identity_map("A", "B", "C")
        gts.species_gene_mapping = mapping
        rho = _compute_rho("A", "B", "C", gts, mapping)
        assert rho == pytest.approx((1.0, 1.0, 0.0), abs=1e-12)

    def test_star_gene_tree_splits_equally(self):
        gt = Network.from_newick("(A:1,B:1,C:1);")
        gts = GeneTrees(gene_tree_list=[gt])
        mapping = _identity_map("A", "B", "C")
        gts.species_gene_mapping = mapping
        rho = _compute_rho("A", "B", "C", gts, mapping)
        assert rho[0] == pytest.approx(1.0 / 3.0, abs=1e-12)
        assert rho[1] == pytest.approx(1.0 / 3.0, abs=1e-12)
        assert rho[2] == pytest.approx(1.0 / 3.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# 4. _subnet_triple_probs — subnetwork decomposition
# ═══════════════════════════════════════════════════════════════════════

class TestSubnetTripleProbs:

    def test_tree_no_reticulation(self):
        st = Network.from_newick("((A:1,B:1):1,C:2);")
        subnet = subnet_given_leaves(
            st,
            [st.has_node_named(x) for x in ("A", "B", "C")],
        )
        p = _subnet_triple_probs(subnet)
        assert p[0] == pytest.approx(_coal_match(1.0), abs=1e-10)
        assert p[1] == pytest.approx(_coal_mismatch(1.0), abs=1e-10)
        assert p[2] == pytest.approx(_coal_mismatch(1.0), abs=1e-10)

    def test_network_displayed_weights_sum_to_one(self):
        """Displayed tree probabilities from the subnet should sum to 1."""
        net = Network.from_newick(
            "((A:1.0,((B:0.5)#H1[&gamma=0.4]:0.5,C:1.0):0.5):0.5,"
            "(#H1[&gamma=0.6]:1.0,D:1.5):0.5);"
        )
        for triple in combinations(("A", "B", "C", "D"), 3):
            leaves = [net.has_node_named(x) for x in triple]
            subnet = subnet_given_leaves(net, leaves)
            displayed = _displayed_trees_with_probs(subnet)
            total_weight = sum(w for _, w in displayed)
            assert total_weight == pytest.approx(1.0, abs=1e-10), (
                f"Weights for triplet {triple} sum to {total_weight}"
            )

    def test_network_triple_probs_sum_to_one(self):
        """P(XY|Z) + P(XZ|Y) + P(YZ|X) must equal 1 for any subnet."""
        net = Network.from_newick(
            "((A:1.0,((B:0.5)#H1[&gamma=0.4]:0.5,C:1.0):0.5):0.5,"
            "(#H1[&gamma=0.6]:1.0,D:1.5):0.5);"
        )
        for triple in combinations(("A", "B", "C", "D"), 3):
            leaves = [net.has_node_named(x) for x in triple]
            subnet = subnet_given_leaves(net, leaves)
            p = _subnet_triple_probs(subnet)
            assert sum(p) == pytest.approx(1.0, abs=1e-10), (
                f"Triple probs for {triple} sum to {sum(p)}"
            )

    def test_network_gamma_affects_probs(self):
        """Changing gamma should change the triple probabilities."""
        net04 = Network.from_newick(
            "((A:1,((B:0.5)#H1[&gamma=0.4]:0.5,C:1):1):1,"
            "(#H1[&gamma=0.6]:1,D:2):1);"
        )
        net09 = Network.from_newick(
            "((A:1,((B:0.5)#H1[&gamma=0.9]:0.5,C:1):1):1,"
            "(#H1[&gamma=0.1]:1,D:2):1);"
        )
        for triple in [("A", "B", "D")]:
            leaves_04 = [net04.has_node_named(x) for x in triple]
            leaves_09 = [net09.has_node_named(x) for x in triple]
            sub04 = subnet_given_leaves(net04, leaves_04)
            sub09 = subnet_given_leaves(net09, leaves_09)
            p04 = _subnet_triple_probs(sub04)
            p09 = _subnet_triple_probs(sub09)
            assert p04 != pytest.approx(p09, abs=1e-6), (
                "Gamma change should alter probabilities"
            )


# ═══════════════════════════════════════════════════════════════════════
# 5. MPL.score — end-to-end scoring
# ═══════════════════════════════════════════════════════════════════════

class TestMPLScore:

    def test_3taxon_concordant(self):
        """Species tree ((A,B):1,C) with gene tree ((A,B),C).

        log PL = rho(AB|C) * log P(AB|C) = 1 * log(1 - 2/3 e^{-1}).
        """
        mpl = _build_mpl(
            "((A:1,B:1):1,C:2);",
            ["((A:0.5,B:0.5):1.5,C:2);"],
            _identity_map("A", "B", "C"),
        )
        score = mpl.score()
        expected = math.log(_coal_match(1.0))
        assert score == pytest.approx(expected, abs=1e-10)

    def test_3taxon_discordant(self):
        """Gene tree AC|B on species tree ((A,B):1,C).

        log PL = 1 * log(P(AC|B)) = log((1/3) e^{-1}).
        """
        mpl = _build_mpl(
            "((A:1,B:1):1,C:2);",
            ["((A:1,C:1):1,B:2);"],
            _identity_map("A", "B", "C"),
        )
        score = mpl.score()
        expected = math.log(_coal_mismatch(1.0))
        assert score == pytest.approx(expected, abs=1e-10)

    def test_concordant_beats_discordant(self):
        """A concordant gene tree should produce a higher score."""
        mapping = _identity_map("A", "B", "C")
        mpl_conc = _build_mpl(
            "((A:1,B:1):1,C:2);",
            ["((A:1,B:1):1,C:2);"],
            mapping,
        )
        mpl_disc = _build_mpl(
            "((A:1,B:1):1,C:2);",
            ["((A:1,C:1):1,B:2);"],
            mapping,
        )
        assert mpl_conc.score() > mpl_disc.score()

    def test_two_gene_trees(self):
        """1 concordant + 1 discordant gene tree."""
        mpl = _build_mpl(
            "((A:1,B:1):1,C:2);",
            [
                "((A:1,B:1):1,C:2);",   # AB|C => rho(AB|C)=1
                "((A:1,C:1):1,B:2);",   # AC|B => rho(AC|B)=1
            ],
            _identity_map("A", "B", "C"),
        )
        score = mpl.score()
        expected = math.log(_coal_match(1.0)) + math.log(_coal_mismatch(1.0))
        assert score == pytest.approx(expected, abs=1e-10)

    def test_score_is_negative(self):
        mpl = _build_mpl(
            "((A:1,B:1):1,C:2);",
            ["((A:1,B:1):1,C:2);"],
            _identity_map("A", "B", "C"),
        )
        assert mpl.score() < 0.0

    def test_score_is_finite(self):
        mpl = _build_mpl(
            "((A:1,B:1):1,C:2);",
            ["((A:1,C:1):1,B:2);"],
            _identity_map("A", "B", "C"),
        )
        assert math.isfinite(mpl.score())

    def test_4taxon_tree(self):
        """4-taxon species tree: all C(4,3)=4 triplets should contribute."""
        mapping = _identity_map("A", "B", "C", "D")
        mpl = _build_mpl(
            "(((A:1,B:1):1,C:2):1,D:3);",
            ["(((A:1,B:1):1,C:2):1,D:3);"],
            mapping,
        )
        score = mpl.score()
        assert math.isfinite(score)
        assert score < 0.0

    def test_many_concordant_gene_trees_improve_score(self):
        """Adding more concordant gene trees should increase the score
        (make it less negative, since each concordant tree adds
        log P(match) which is the largest of the three terms)."""
        mapping = _identity_map("A", "B", "C")
        gt_newick = "((A:1,B:1):1,C:2);"

        scores = []
        for n in [1, 5, 20]:
            mpl = _build_mpl(
                "((A:1,B:1):1,C:2);",
                [gt_newick] * n,
                mapping,
            )
            scores.append(mpl.score())

        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], (
                f"More concordant trees should give a more negative score "
                f"(score monotonically decreases): {scores}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 6. MPL on networks with reticulation
# ═══════════════════════════════════════════════════════════════════════

class TestMPLNetwork:

    @pytest.fixture
    def net4_gamma04(self):
        """4-taxon network: B is a hybrid child of two lineages (gamma=0.4/0.6)."""
        return Network.from_newick(
            "((A:1.0,((B:0.5)#H1[&gamma=0.4]:0.5,C:1.0):0.5):0.5,"
            "(#H1[&gamma=0.6]:1.0,D:1.5):0.5);"
        )

    def test_network_score_finite(self, net4_gamma04):
        gt = Network.from_newick("((A:1,B:1):1,(C:1,D:1):1);")
        mapping = _identity_map("A", "B", "C", "D")
        gts = GeneTrees(gene_tree_list=[gt])
        gts.species_gene_mapping = mapping
        mpl = MPL(net4_gamma04, gts, mapping)
        score = mpl.score()
        assert math.isfinite(score)
        assert score < 0.0

    def test_network_score_negative(self, net4_gamma04):
        gt = Network.from_newick("(((A:1,C:1):1,B:2):1,D:3);")
        mapping = _identity_map("A", "B", "C", "D")
        gts = GeneTrees(gene_tree_list=[gt])
        gts.species_gene_mapping = mapping
        mpl = MPL(net4_gamma04, gts, mapping)
        assert mpl.score() < 0.0

    def test_different_gammas_different_scores(self):
        """Two networks identical except for gamma should give different scores."""
        gt = Network.from_newick("((A:1,B:1):1,(C:1,D:1):1);")
        mapping = _identity_map("A", "B", "C", "D")

        net_a = Network.from_newick(
            "((A:1,((B:0.5)#H1[&gamma=0.3]:0.5,C:1):1):1,"
            "(#H1[&gamma=0.7]:1,D:2):1);"
        )
        net_b = Network.from_newick(
            "((A:1,((B:0.5)#H1[&gamma=0.8]:0.5,C:1):1):1,"
            "(#H1[&gamma=0.2]:1,D:2):1);"
        )

        gts_a = GeneTrees(gene_tree_list=[gt])
        gts_a.species_gene_mapping = mapping
        gts_b = GeneTrees(gene_tree_list=[gt])
        gts_b.species_gene_mapping = mapping

        score_a = MPL(net_a, gts_a, mapping).score()
        score_b = MPL(net_b, gts_b, mapping).score()

        assert score_a != pytest.approx(score_b, abs=1e-6), (
            "Different gammas should produce different scores"
        )


# ═══════════════════════════════════════════════════════════════════════
# 7. Multi-allele mapping
# ═══════════════════════════════════════════════════════════════════════

class TestMultiAllele:

    def test_two_alleles_per_species(self):
        """Species A has alleles A1,A2; B has B1,B2; C has C1."""
        st = Network.from_newick("((A:1,B:1):1,C:2);")
        gt = Network.from_newick("((A1:1,(A2:0.5,B1:0.5):0.5):1,(B2:1,C1:1):1);")
        mapping = {"A": ["A1", "A2"], "B": ["B1", "B2"], "C": ["C1"]}

        gts = GeneTrees(gene_tree_list=[gt])
        gts.species_gene_mapping = mapping
        mpl = MPL(st, gts, mapping)
        score = mpl.score()
        assert math.isfinite(score)
        assert score < 0.0


# ═══════════════════════════════════════════════════════════════════════
# 8. Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_extra_mapping_key_ignored(self):
        """Mapping entries for species not in the network don't cause errors.

        MPL only iterates triplets formed from the species-tree leaves,
        so extraneous mapping keys are silently ignored.
        """
        st = Network.from_newick("((A:1,B:1):1,C:2);")
        gt = Network.from_newick("((A:1,B:1):1,C:2);")
        mapping = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
        gts = GeneTrees(gene_tree_list=[gt])
        gts.species_gene_mapping = mapping
        mpl = MPL(st, gts, mapping)
        score = mpl.score()
        assert math.isfinite(score)

    def test_very_small_branch_length(self):
        """Near-zero branch length should give near-uniform probs."""
        mpl = _build_mpl(
            "((A:1,B:1):0.001,C:1.001);",
            ["((A:1,B:1):1,C:2);"],
            _identity_map("A", "B", "C"),
        )
        score = mpl.score()
        assert math.isfinite(score)
        assert score < 0.0

    def test_identical_score_for_relabeled_sister(self):
        """Swapping the two sisters shouldn't change the score."""
        mapping = _identity_map("A", "B", "C")
        mpl1 = _build_mpl(
            "((A:1,B:1):1,C:2);",
            ["((A:1,B:1):1,C:2);"],
            mapping,
        )
        mpl2 = _build_mpl(
            "((B:1,A:1):1,C:2);",
            ["((B:1,A:1):1,C:2);"],
            mapping,
        )
        assert mpl1.score() == pytest.approx(mpl2.score(), abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# 9. Analytical verification: 3-taxon tree with varying tau
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyticalVerification:

    @pytest.mark.parametrize("tau", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_concordant_score_matches_formula(self, tau):
        """For concordant gene tree on ((A,B):tau,C), score = log(1 - 2/3 e^{-tau})."""
        st_nwk = f"((A:1,B:1):{tau},C:{1+tau});"
        gt_nwk = "((A:1,B:1):1,C:2);"
        mpl = _build_mpl(st_nwk, [gt_nwk], _identity_map("A", "B", "C"))
        score = mpl.score()
        expected = math.log(_coal_match(tau))
        assert score == pytest.approx(expected, abs=1e-10), (
            f"tau={tau}: score={score}, expected={expected}"
        )

    @pytest.mark.parametrize("tau", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_discordant_score_matches_formula(self, tau):
        """For discordant gene tree AC|B on ((A,B):tau,C), score = log(1/3 e^{-tau})."""
        st_nwk = f"((A:1,B:1):{tau},C:{1+tau});"
        gt_nwk = "((A:1,C:1):1,B:2);"
        mpl = _build_mpl(st_nwk, [gt_nwk], _identity_map("A", "B", "C"))
        score = mpl.score()
        expected = math.log(_coal_mismatch(tau))
        assert score == pytest.approx(expected, abs=1e-10), (
            f"tau={tau}: score={score}, expected={expected}"
        )

    @pytest.mark.parametrize("tau", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_score_increases_with_tau_for_concordant(self, tau):
        """Longer internal branch => higher score for concordant gene trees."""
        mapping = _identity_map("A", "B", "C")
        gt_nwk = "((A:1,B:1):1,C:2);"
        scores = {}
        for t in [tau, tau + 1.0]:
            st_nwk = f"((A:1,B:1):{t},C:{1+t});"
            mpl = _build_mpl(st_nwk, [gt_nwk], mapping)
            scores[t] = mpl.score()
        assert scores[tau + 1.0] > scores[tau]
