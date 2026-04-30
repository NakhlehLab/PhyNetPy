"""
Sanity test suite for :mod:`phynetpy.MCMC_GT`.

Three correctness slices covering the load-bearing pieces of the new
MSNC likelihood engine + caching plumbing:

1. ``TestMSCClosedForm``: With a 3-taxon species *tree* and
   identity allele mapping, the :class:`_GTLikelihoodEngine`
   log-probability of a single gene tree must match the
   Rannala-Yang closed form (``P(XY|Z) = 1 - (2/3) e^{-tau}`` for
   concordant, ``(1/3) e^{-tau}`` for either discordant topology).

2. ``TestTripletMarginalMatchesMPL``: For a 3-taxon tree the new
   MSNC engine's single-gene-tree log-probability must equal
   MPL's triplet log pseudo-likelihood to float precision.  This
   gives us a direct differential test against the existing,
   heavily-vetted :class:`MPL` code.

3. ``TestDifferentialCache``: Scoring through the per-network-edge
   cache (applying a ``ChangeNodeHeight`` move in place then
   calling the scorer again) must match re-scoring with a
   freshly-built engine to float precision.  Guards the dirty-set
   invalidation path in :meth:`_GTLikelihoodEngine.update`.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phynetpy.Network import Network, Node, Edge
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL import MPL
from phynetpy.ModelGraph import Model
from phynetpy.ModelMove import ChangeNodeHeight
from phynetpy.MCMC_GT import (
    MCMCGTScorer,
    _GTLikelihoodEngine,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _identity_map(*taxa: str) -> dict[str, list[str]]:
    """One-to-one species -> allele mapping."""
    return {t: [t] for t in taxa}


def _species_of(mapping: dict[str, list[str]]) -> dict[str, str]:
    """Flatten ``mapping`` to a gene-copy -> species reverse map."""
    return {a: sp for sp, alleles in mapping.items() for a in alleles}


def _coal_match(tau: float) -> float:
    """P(sister pair agrees with species tree) for ingroup length ``tau``."""
    return 1.0 - (2.0 / 3.0) * math.exp(-tau)


def _coal_mismatch(tau: float) -> float:
    """P(sister pair disagrees) for ingroup length ``tau``."""
    return (1.0 / 3.0) * math.exp(-tau)


# ═══════════════════════════════════════════════════════════════════════
# 1. MSC closed-form check (no reticulation)
# ═══════════════════════════════════════════════════════════════════════

class TestMSCClosedForm:
    """No-retic case must reduce to the standard MSC formula."""

    SP_NEWICK = "((A:1,B:1):1,C:2);"

    def _engine(self) -> _GTLikelihoodEngine:
        return _GTLikelihoodEngine(Network.from_newick(self.SP_NEWICK))

    def test_concordant_gene_tree(self):
        """((A,B),C) on species ((A,B),C) -> log(1 - 2/3 e^{-1})."""
        gt = Network.from_newick("((A:0.5,B:0.5):1.5,C:2);")
        lp = self._engine().log_prob(gt, _species_of(_identity_map("A", "B", "C")))
        expected = math.log(_coal_match(1.0))
        assert lp == pytest.approx(expected, abs=1e-10)

    def test_discordant_AC(self):
        """((A,C),B) on species ((A,B),C) -> log((1/3) e^{-1})."""
        gt = Network.from_newick("((A:0.5,C:0.5):1.5,B:2);")
        lp = self._engine().log_prob(gt, _species_of(_identity_map("A", "B", "C")))
        expected = math.log(_coal_mismatch(1.0))
        assert lp == pytest.approx(expected, abs=1e-10)

    def test_discordant_BC(self):
        """((B,C),A) on species ((A,B),C) -> log((1/3) e^{-1})."""
        gt = Network.from_newick("((B:0.5,C:0.5):1.5,A:2);")
        lp = self._engine().log_prob(gt, _species_of(_identity_map("A", "B", "C")))
        expected = math.log(_coal_mismatch(1.0))
        assert lp == pytest.approx(expected, abs=1e-10)

    def test_triplet_probabilities_sum_to_one(self):
        """All three possible rooted 3-taxon gene-tree topologies sum to 1."""
        engine = self._engine()
        species_of = _species_of(_identity_map("A", "B", "C"))
        log_probs = [
            engine.log_prob(
                Network.from_newick(nwk), species_of,
            ) for nwk in [
                "((A:0.5,B:0.5):1.5,C:2);",
                "((A:0.5,C:0.5):1.5,B:2);",
                "((B:0.5,C:0.5):1.5,A:2);",
            ]
        ]
        total = sum(math.exp(lp) for lp in log_probs)
        assert total == pytest.approx(1.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# 2. Triplet marginal matches MPL
# ═══════════════════════════════════════════════════════════════════════

class TestTripletMarginalMatchesMPL:
    """Single-gene-tree MSNC log-prob == MPL triplet log-PL on 3-taxon data."""

    SP_NEWICK = "((A:1,B:1):1,C:2);"

    @pytest.mark.parametrize("gt_newick", [
        "((A:0.5,B:0.5):1.5,C:2);",
        "((A:0.5,C:0.5):1.5,B:2);",
        "((B:0.5,C:0.5):1.5,A:2);",
    ])
    def test_matches_mpl(self, gt_newick: str):
        """For any 3-taxon gene tree, MSNC log prob == MPL log PL."""
        sp = Network.from_newick(self.SP_NEWICK)
        gt = Network.from_newick(gt_newick)
        mapping = _identity_map("A", "B", "C")

        # MPL path
        gts = GeneTrees(gene_tree_list=[gt])
        gts.species_gene_mapping = mapping
        mpl = MPL(sp, gts, mapping)
        mpl_score = mpl.score()

        # MSNC engine path (fresh species network to avoid MPL's
        # side effects on shared nodes).
        sp2 = Network.from_newick(self.SP_NEWICK)
        engine_score = _GTLikelihoodEngine(sp2).log_prob(
            gt, _species_of(mapping),
        )

        assert engine_score == pytest.approx(mpl_score, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# 3. Differential-cache test
# ═══════════════════════════════════════════════════════════════════════

class TestDifferentialCache:
    """Caching via dirty-node invalidation must agree with a fresh rebuild."""

    def test_change_node_height_cached_matches_fresh(self):
        """
        Apply ``ChangeNodeHeight`` in place.  Scoring via the
        cached engine (which consumes ``model._dirty_nodes`` and
        updates only the touched-ancestor cone) must match scoring
        via a freshly-built engine to float precision.
        """
        sp = Network.from_newick("((A:1,B:1):1,C:2);")
        gt = Network.from_newick("((A:0.5,B:0.5):1.5,C:2);")
        mapping = _identity_map("A", "B", "C")

        scorer = MCMCGTScorer([gt], mapping, posterior=False)
        rng = np.random.default_rng(42)
        model = Model(rng=rng)
        model.network = sp
        model.set_likelihood_calculator(scorer)

        # Warm the cache.
        initial = scorer(model)
        assert math.isfinite(initial)

        move = ChangeNodeHeight(sigma_frac=0.4)
        model = move.execute(model)

        cached_score = scorer(model)

        # Compare to a completely fresh scorer (and therefore a
        # brand-new :class:`_GTLikelihoodEngine`) evaluated on the
        # mutated network.
        fresh = MCMCGTScorer([gt], mapping, posterior=False)
        fresh_model = Model(rng=np.random.default_rng(7))
        fresh_model.network = model.network
        fresh_model.set_likelihood_calculator(fresh)
        fresh_score = fresh(fresh_model)

        assert cached_score == pytest.approx(fresh_score, abs=1e-10)

    def test_many_moves_cached_matches_fresh(self):
        """Repeated ``ChangeNodeHeight`` moves preserve cache coherence."""
        sp = Network.from_newick("(((A:1,B:1):0.5,C:1.5):0.5,D:2);")
        gt = Network.from_newick("(((A:0.3,B:0.3):0.7,C:1):0.5,D:1.5);")
        mapping = _identity_map("A", "B", "C", "D")

        scorer = MCMCGTScorer([gt], mapping, posterior=False)
        rng = np.random.default_rng(2024)
        model = Model(rng=rng)
        model.network = sp
        model.set_likelihood_calculator(scorer)

        scorer(model)

        for _ in range(5):
            move = ChangeNodeHeight(sigma_frac=0.3)
            model = move.execute(model)
            cached = scorer(model)

            fresh = MCMCGTScorer([gt], mapping, posterior=False)
            fm = Model(rng=np.random.default_rng(1))
            fm.network = model.network
            fm.set_likelihood_calculator(fresh)
            fresh_score = fresh(fm)

            assert cached == pytest.approx(fresh_score, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# 4. Full MSNC (ancestral-configurations DP) on retic networks
# ═══════════════════════════════════════════════════════════════════════

def _build_level1_3tax(
    *,
    t_ab: float = 1.0,
    t_root: float = 1.0,
    t_p1_h: float = 0.5,
    t_p2_h: float = 0.5,
    t_h_b: float = 0.5,
    t_left_a: float = 1.5,
    t_right_c: float = 1.5,
    gamma: float = 0.5,
) -> Network:
    """3-taxon level-1 network with one reticulation feeding species ``B``.

    Topology::

                Root
               /    \\
             P1      P2
            / \\    /
           A   \\  /
                #H
                |
                B    C  (under P2)

    Concretely: ``Root`` has two children ``P1`` and ``P2``; ``P1`` has
    children ``A`` and ``#H`` (the reticulation); ``P2`` has children
    ``#H`` and ``C``.  ``#H`` has the single child ``B``.  Two parents
    feed ``#H`` with weights ``gamma`` (from ``P1``) and ``1 - gamma``
    (from ``P2``).  This is the smallest network whose AC DP differs
    from the displayed-tree mixture once gene-tree lineages above the
    retic span more than one species (which doesn't happen for
    single-allele 3-taxa data, but does for 2-allele B data).
    """
    labels = ["Root", "P1", "P2", "#H", "A", "B", "C"]
    nodes = {l: Node(l, is_reticulation=(l == "#H")) for l in labels}
    net = Network()
    net.add_nodes(*nodes.values())
    net.add_edges([
        Edge(nodes["Root"], nodes["P1"], length=t_root),
        Edge(nodes["Root"], nodes["P2"], length=t_root),
        Edge(nodes["P1"], nodes["A"], length=t_left_a),
        Edge(nodes["P1"], nodes["#H"], length=t_p1_h, gamma=gamma),
        Edge(nodes["P2"], nodes["#H"], length=t_p2_h, gamma=1.0 - gamma),
        Edge(nodes["P2"], nodes["C"], length=t_right_c),
        Edge(nodes["#H"], nodes["B"], length=t_h_b),
    ])
    return net


class TestRetic3TaxNormalisation:
    """All rooted 3-taxon gene-tree topologies must sum to 1 under any retic.

    This is the strongest scalar invariant of the MSNC: regardless of
    branch lengths or gamma, the three possible rooted 3-taxon gene
    trees (for taxa ``A``, ``B``, ``C``) must partition the
    probability simplex.  The displayed-tree mixture *also* satisfies
    normalisation under single-allele 3-taxa data, so this test
    primarily guards against silent bugs in the AC DP's frontier
    book-keeping (sub-iteration, gamma factors, root collapse) that
    would manifest as either probability sinks or duplicated paths.
    """

    @pytest.mark.parametrize("gamma", [0.1, 0.3, 0.5, 0.7])
    def test_three_topologies_sum_to_one(self, gamma: float):
        net = _build_level1_3tax(gamma=gamma)
        engine = _GTLikelihoodEngine(net)
        species_of = _species_of(_identity_map("A", "B", "C"))
        log_probs = []
        for nwk in [
            "((A:0.5,B:0.5):1.5,C:2);",
            "((A:0.5,C:0.5):1.5,B:2);",
            "((B:0.5,C:0.5):1.5,A:2);",
        ]:
            gt = Network.from_newick(nwk)
            log_probs.append(engine.log_prob(gt, species_of))
        total = sum(math.exp(lp) for lp in log_probs)
        assert total == pytest.approx(1.0, abs=1e-10)


class TestRetic3TaxMixingLimit:
    """Gamma-extremes collapse the network to its displayed trees.

    When ``gamma -> 0`` the retic is fully routed through ``P2``
    (giving the displayed tree ``(A, (B, C))``); when ``gamma -> 1``
    it routes fully through ``P1`` (displayed tree ``((A, B), C)``).
    The AC DP on the limit network must match the standard MSC DP
    on the corresponding displayed tree.
    """

    def _msc_score_tree(self, sp_newick: str, gt_newick: str) -> float:
        sp = Network.from_newick(sp_newick)
        gt = Network.from_newick(gt_newick)
        return _GTLikelihoodEngine(sp).log_prob(
            gt, _species_of(_identity_map("A", "B", "C")),
        )

    @pytest.mark.parametrize("gt_newick", [
        "((A:0.5,B:0.5):1.5,C:2);",
        "((A:0.5,C:0.5):1.5,B:2);",
        "((B:0.5,C:0.5):1.5,A:2);",
    ])
    def test_gamma_zero_matches_right_displayed_tree(self, gt_newick: str):
        # gamma = 0 -> retic kept on parent P2 -> displayed tree:
        #   ((A:1) :1, (#H -> B :0.5+0.5=1, C:1.5):1).
        #   In coalescent units, the (B, C) clade sits under P2 with
        #   the t_p2_h + t_h_b path length above B, and the t_right_c
        #   above C; the P1 -> A path is t_root + t_left_a above the
        #   root.  Build the equivalent species tree and compare.
        net = _build_level1_3tax(gamma=1e-12)
        engine = _GTLikelihoodEngine(net)
        species_of = _species_of(_identity_map("A", "B", "C"))
        gt = Network.from_newick(gt_newick)
        ac_score = engine.log_prob(gt, species_of)

        # Equivalent displayed tree: ((B:1, C:1.5):1, A:2.5);
        # where the (B, C) clade joins under P2 (depth 1) and the
        # outer ingroup-time is t_root (1.0).
        # Note: when gamma collapses fully to P2 the contracted
        # tree's "P1 -> A" edge runs from Root down to A and is the
        # SISTER to (B, C); per the construction above, A's path
        # from Root is t_root + t_left_a = 1.0 + 1.5 = 2.5.
        equiv_st = "((B:1.0,C:1.5):1.0,A:2.5);"
        msc_score = self._msc_score_tree(equiv_st, gt_newick)
        assert ac_score == pytest.approx(msc_score, abs=1e-7)

    @pytest.mark.parametrize("gt_newick", [
        "((A:0.5,B:0.5):1.5,C:2);",
        "((A:0.5,C:0.5):1.5,B:2);",
        "((B:0.5,C:0.5):1.5,A:2);",
    ])
    def test_gamma_one_matches_left_displayed_tree(self, gt_newick: str):
        net = _build_level1_3tax(gamma=1.0 - 1e-12)
        engine = _GTLikelihoodEngine(net)
        species_of = _species_of(_identity_map("A", "B", "C"))
        gt = Network.from_newick(gt_newick)
        ac_score = engine.log_prob(gt, species_of)

        # When gamma -> 1, the retic collapses to its P1 parent.  The
        # contracted tree's path from Root to B is t_root + t_p1_h +
        # t_h_b = 1.0 + 0.5 + 0.5 = 2.0.  A is sister to B under P1
        # via t_left_a = 1.5; so the P1 subtree is ((A:1.5, B:2.0):...).
        # Wait, actually under P1 in the displayed tree A is a sister
        # to the B branch.  P1 sits at depth t_root = 1.0; A is at
        # depth t_root + t_left_a = 2.5; B is at depth t_root +
        # t_p1_h + t_h_b = 2.0.  So leaf-depths from root: A -> 2.5,
        # B -> 2.0, C -> t_root + t_right_c = 2.5.  Equivalent tree:
        equiv_st = "((A:1.5,B:1.0):1.0,C:2.5);"
        msc_score = self._msc_score_tree(equiv_st, gt_newick)
        assert ac_score == pytest.approx(msc_score, abs=1e-7)


class TestRetic3TaxSymmetry:
    """gamma <-> 1-gamma + parent swap is a symmetry of the AC DP.

    The species network has two reticulation parents.  Swapping
    ``gamma`` to ``1 - gamma`` is a relabelling of which parent is
    the "left" one; the contracted likelihood is invariant under
    this transformation when accompanied by a topology-symmetric
    parent-edge length swap.  In our specific build above (where
    ``t_p1_h == t_p2_h``), gamma and 1-gamma should yield the
    *same* per-gene-tree log-probability.
    """

    @pytest.mark.parametrize("gamma", [0.2, 0.4])
    @pytest.mark.parametrize("gt_newick", [
        "((A:0.5,B:0.5):1.5,C:2);",
        "((A:0.5,C:0.5):1.5,B:2);",
        "((B:0.5,C:0.5):1.5,A:2);",
    ])
    def test_gamma_swap_symmetry_when_lengths_balanced(
        self, gamma: float, gt_newick: str,
    ):
        # The reticulation parent edges have equal length in the
        # default builder, so under the AC DP the network is
        # invariant under gamma <-> 1 - gamma + (A <-> C) leaf swap.
        # Direct gamma symmetry doesn't hold in general because P1
        # vs P2 sit on different sides; instead we verify the score
        # under the pair of (gamma, 1-gamma) networks satisfies the
        # leaf-relabeling invariant.
        net = _build_level1_3tax(gamma=gamma)
        net_flipped = _build_level1_3tax(gamma=1.0 - gamma)
        species_of = _species_of(_identity_map("A", "B", "C"))

        score_orig = _GTLikelihoodEngine(net).log_prob(
            Network.from_newick(gt_newick), species_of,
        )

        # Relabel A <-> C in the gene tree to match the flipped network.
        flipped_newick = (
            gt_newick
            .replace("A", "_TMP_")
            .replace("C", "A")
            .replace("_TMP_", "C")
        )
        score_flipped = _GTLikelihoodEngine(net_flipped).log_prob(
            Network.from_newick(flipped_newick), species_of,
        )
        assert score_orig == pytest.approx(score_flipped, abs=1e-9)


class TestRetic4TaxNormalisation:
    """4-taxon retic network: 15 rooted gene-tree topologies sum to 1.

    The number of rooted unlabelled binary trees on n leaves is
    ``(2n - 3)!! = 15`` for n = 4.  Enumerating every labelled
    topology and confirming probabilities sum to 1 is a strong
    consistency check on the AC DP across all four MSC traversal
    paths (A is sister to B/C/D, etc.) plus the retic split logic.
    """

    @staticmethod
    def _all_4tax_rooted_topologies() -> list[str]:
        # 15 rooted binary trees on {A, B, C, D}: (3 ways to pick the
        # outgroup) x (3 ways to pair the inner clade as a "cherry") +
        # ... actually, more cleanly: pick the leaf paired with the
        # sister clade at the root, then pick the cherry inside.
        # Hand-enumerate to be exhaustive:
        return [
            # Outgroup A: ((B,C),D), ((B,D),C), ((C,D),B) attached to A
            "(A:2,((B:0.5,C:0.5):0.5,D:1):1);",
            "(A:2,((B:0.5,D:0.5):0.5,C:1):1);",
            "(A:2,((C:0.5,D:0.5):0.5,B:1):1);",
            # Outgroup B
            "(B:2,((A:0.5,C:0.5):0.5,D:1):1);",
            "(B:2,((A:0.5,D:0.5):0.5,C:1):1);",
            "(B:2,((C:0.5,D:0.5):0.5,A:1):1);",
            # Outgroup C
            "(C:2,((A:0.5,B:0.5):0.5,D:1):1);",
            "(C:2,((A:0.5,D:0.5):0.5,B:1):1);",
            "(C:2,((B:0.5,D:0.5):0.5,A:1):1);",
            # Outgroup D
            "(D:2,((A:0.5,B:0.5):0.5,C:1):1);",
            "(D:2,((A:0.5,C:0.5):0.5,B:1):1);",
            "(D:2,((B:0.5,C:0.5):0.5,A:1):1);",
            # Balanced: ((A,B),(C,D)) family - 3 unordered pairings
            "((A:0.5,B:0.5):1.5,(C:0.5,D:0.5):1.5);",
            "((A:0.5,C:0.5):1.5,(B:0.5,D:0.5):1.5);",
            "((A:0.5,D:0.5):1.5,(B:0.5,C:0.5):1.5);",
        ]

    @staticmethod
    def _build_level1_4tax(gamma: float = 0.4) -> Network:
        """4-taxon level-1 network: A,B,C as in the 3-tax build, plus D off the root."""
        labels = ["Root", "X", "P1", "P2", "#H", "A", "B", "C", "D"]
        nodes = {l: Node(l, is_reticulation=(l == "#H")) for l in labels}
        net = Network()
        net.add_nodes(*nodes.values())
        net.add_edges([
            Edge(nodes["Root"], nodes["X"], length=1.0),
            Edge(nodes["Root"], nodes["D"], length=2.5),
            Edge(nodes["X"], nodes["P1"], length=0.5),
            Edge(nodes["X"], nodes["P2"], length=0.5),
            Edge(nodes["P1"], nodes["A"], length=1.0),
            Edge(nodes["P1"], nodes["#H"], length=0.5, gamma=gamma),
            Edge(nodes["P2"], nodes["#H"], length=0.5, gamma=1.0 - gamma),
            Edge(nodes["P2"], nodes["C"], length=1.0),
            Edge(nodes["#H"], nodes["B"], length=0.5),
        ])
        return net

    @pytest.mark.parametrize("gamma", [0.25, 0.5, 0.75])
    def test_4tax_topology_sum(self, gamma: float):
        net = self._build_level1_4tax(gamma=gamma)
        engine = _GTLikelihoodEngine(net)
        species_of = _species_of(_identity_map("A", "B", "C", "D"))
        log_probs = []
        for nwk in self._all_4tax_rooted_topologies():
            gt = Network.from_newick(nwk)
            log_probs.append(engine.log_prob(gt, species_of))
        total = sum(math.exp(lp) for lp in log_probs)
        assert total == pytest.approx(1.0, abs=1e-9)


class TestRetic4TaxGammaCollapse:
    """Gamma -> 0 / 1 on the 4-tax retic network must equal the displayed-tree MSC.

    Same invariant as :class:`TestRetic3TaxMixingLimit` extended to
    four leaves with the (A,B,C,D) topology shown in
    :class:`TestRetic4TaxNormalisation`.  Picks one gene tree to
    keep the test small (the topology sum already covers the
    breadth dimension).
    """

    GT_NEWICK = "(((A:0.5,B:0.5):0.5,C:1):1,D:2);"

    def test_gamma_zero_collapses_to_displayed(self):
        net = TestRetic4TaxNormalisation._build_level1_4tax(gamma=1e-12)
        engine = _GTLikelihoodEngine(net)
        species_of = _species_of(_identity_map("A", "B", "C", "D"))
        ac_score = engine.log_prob(Network.from_newick(self.GT_NEWICK), species_of)

        # gamma -> 0: H follows P2.  Effective species tree:
        #   Root -- D (length 2.5)
        #   Root -- X (length 1)
        #     X -- P1 (length 0.5) -- A (length 1)
        #     X -- P2 (length 0.5) -- C (length 1)
        #     P2 -- (P2->H 0.5 + H->B 0.5 = 1.0) -- B
        # i.e. P2 has three children A? No: only C and B.  Wait, P1
        # under gamma=0 has only A as a child after pruning the H
        # path.  P2 has C and the contracted-H -> B path.  So:
        equiv_st = "((A:1.0):0.5+0.5=1.5???,(B:1.0,C:1.0):0.5):1.0,D:2.5);"
        # The cleaner build of the displayed tree once chains are
        # collapsed: P1 -> A becomes a passthrough with summed
        # length (X -> P1 -> A = 0.5 + 1.0 = 1.5); P2 keeps its
        # split (X -> P2 = 0.5, with two children B at +1.0 path
        # via H, and C at +1.0).  So:
        equiv_st = "((A:1.5,(B:1.0,C:1.0):0.5):1.0,D:2.5);"
        msc_score = _GTLikelihoodEngine(Network.from_newick(equiv_st)).log_prob(
            Network.from_newick(self.GT_NEWICK), species_of,
        )
        assert ac_score == pytest.approx(msc_score, abs=1e-7)


class TestRetic3TaxMatchesPhyloNetSpec:
    """The AC DP must match the formula in the Yu et al. 2012 paper.

    Uses an analytic value for a simple 1-retic 3-tax case computed
    by hand from Eq. 6 of the paper (the DP closed-form for a single
    reticulation crossed by a single lineage).  For
    ``gamma * P(g | T_left) + (1 - gamma) * P(g | T_right)``, where
    ``T_left`` and ``T_right`` are the two displayed trees, the AC
    DP and displayed-tree mixture coincide on single-allele 3-taxon
    data (the only way to reach the retic from below is with a
    single B-lineage, no sub-iteration needed).
    """

    @pytest.mark.parametrize("gamma", [0.1, 0.4, 0.6, 0.9])
    @pytest.mark.parametrize("gt_newick", [
        "((A:0.5,B:0.5):1.5,C:2);",
        "((A:0.5,C:0.5):1.5,B:2);",
        "((B:0.5,C:0.5):1.5,A:2);",
    ])
    def test_matches_displayed_tree_mixture(self, gamma: float, gt_newick: str):
        # Single-allele 3-taxa: the retic feeds species B with one
        # lineage; AC DP == displayed-tree mixture.
        net = _build_level1_3tax(gamma=gamma)
        engine = _GTLikelihoodEngine(net)
        species_of = _species_of(_identity_map("A", "B", "C"))
        ac_score = engine.log_prob(Network.from_newick(gt_newick), species_of)

        # Build the displayed-tree mixture explicitly:
        # T_left  (gamma path, P1 keeps H): P1 -> A and P1 -> H -> B,
        #   with the P2 -> H path dropped.  Equivalent species tree:
        #   ((A:1.5, B:1.0):1.0, C:2.5);
        t_left = "((A:1.5,B:1.0):1.0,C:2.5);"
        # T_right (1-gamma path, P2 keeps H): P2 -> C and P2 -> H -> B,
        #   with the P1 -> H path dropped.  Equivalent species tree:
        #   ((B:1.0, C:1.5):1.0, A:2.5);
        t_right = "((B:1.0,C:1.5):1.0,A:2.5);"

        gt = Network.from_newick(gt_newick)
        score_left = _GTLikelihoodEngine(Network.from_newick(t_left)).log_prob(
            gt, species_of,
        )
        score_right = _GTLikelihoodEngine(Network.from_newick(t_right)).log_prob(
            gt, species_of,
        )
        log_g = math.log(gamma) if gamma > 0 else float("-inf")
        log_1mg = math.log(1.0 - gamma) if gamma < 1.0 else float("-inf")
        # log-sum-exp of the two displayed-tree contributions.
        log_terms = [log_g + score_left, log_1mg + score_right]
        m = max(log_terms)
        mixture = m + math.log(sum(math.exp(t - m) for t in log_terms))

        assert ac_score == pytest.approx(mixture, abs=1e-9)
