"""Tests for the unified inference search flags
(``opt_bl`` / ``fix_st`` / ``max_lvl`` / ``pseudo``).

Coverage:

1. ``TestResolveMoveTypes`` -- ``opt_bl`` drops the continuous-parameter
   moves, ``fix_st`` drops ``SPR``, both compose, and a custom base set is
   honoured without mutation.
2. ``TestLevelValidator`` -- ``make_level_validator`` is a no-op for
   ``max_lvl=None`` and otherwise rejects networks whose level exceeds the
   cap while accepting those within it.
3. ``TestMaxLevelSelfReject`` -- the per-move ``max_level`` bake-in: starting
   from a 2-reticulation level-1 network, ``RelocateReticulation`` /
   ``ChangeReticSource`` / ``ChangeReticDest`` proposals never leave a
   level-2 network when ``max_level=1`` is passed, even though the *same*
   moves (unguarded) *can* create a level-2 blob.  This is the targeted
   regression the reticulation-count cap alone would have missed.
4. ``TestPseudoRouting`` -- ``InferNetwork_ML.score(pseudo=True)`` routes
   through the triplet ``MPLScorer`` (matches ``MPL.score`` on a 3-taxon
   tree), and ``MCMC_GT.search(method='mh', pseudo=True)`` is rejected.
5. ``TestSearchFlagsSmoke`` -- each flag drives a tiny end-to-end search on
   all three methods to a finite score, and ``max_lvl=1`` is enforced on the
   returned network.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phynetpy.Network import Network, Node, Edge
from phynetpy.GeneTrees import GeneTrees
from phynetpy.ModelGraph import Model
from phynetpy.GraphUtils import level
from phynetpy.ModelMove import (
    SPR,
    ChangeNodeHeight,
    ChangeInheritanceProb,
    AddReticulation,
    RemoveReticulation,
    RelocateReticulation,
    ChangeReticSource,
    ChangeReticDest,
    FlipReticulation,
)
from phynetpy._search_flags import (
    DEFAULT_MOVE_TYPES,
    CONTINUOUS_MOVES,
    BACKBONE_MOVES,
    resolve_move_types,
    make_level_validator,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _identity_map(*taxa: str) -> dict[str, list[str]]:
    return {t: [t] for t in taxa}


def _build_network(node_labels, edge_specs, retic_labels=None) -> Network:
    """Build a network from (label list, (src, dest, gamma) specs)."""
    retic_set = set(retic_labels or [])
    nodes: dict[str, Node] = {}
    for label in list(node_labels) + list(retic_set):
        if label not in nodes:
            nodes[label] = Node(label, is_reticulation=label in retic_set)
    net = Network()
    net.add_nodes(*nodes.values())
    for src, dest, gamma in edge_specs:
        net.add_edges(Edge(nodes[src], nodes[dest], length=1.0, gamma=gamma))
    return net


def _two_retic_level1_network() -> Network:
    """Two reticulations in *disjoint* blobs -> overall level 1.

    Every reticulation parent branches (out-degree 2), so the network is
    structurally valid (passes ``network_invariants_routine``).  Relocating
    either reticulation into the other's blob raises the level to 2 without
    changing the reticulation count -- exactly the case a count-only cap
    misses and the level guard must catch.
    """
    labels = [
        "Root", "PA1", "PA2", "#H0", "X", "PB1", "PB2", "#H1",
        "A", "B", "C", "D", "E",
    ]
    edges = [
        ("Root", "PA1", None),
        ("Root", "PA2", None),
        ("PA1", "A", None),
        ("PA1", "#H0", 0.5),
        ("PA2", "#H0", 0.5),
        ("PA2", "X", None),
        ("#H0", "B", None),
        ("X", "PB1", None),
        ("X", "PB2", None),
        ("PB1", "C", None),
        ("PB1", "#H1", 0.6),
        ("PB2", "#H1", 0.4),
        ("PB2", "D", None),
        ("#H1", "E", None),
    ]
    net = _build_network(labels, edges, retic_labels={"#H0", "#H1"})
    assert level(net) == 1
    return net


def _level2_network() -> Network:
    """Valid network with two reticulations nested in a single blob -> level 2."""
    labels = ["Root", "N1", "N2", "N3", "#H1", "N4", "#H2", "A", "B", "C", "D"]
    edges = [
        ("Root", "N1", None),
        ("Root", "A", None),
        ("N1", "N2", None),
        ("N1", "N3", None),
        ("N2", "#H1", 0.5),
        ("N2", "B", None),
        ("N3", "#H1", 0.5),
        ("N3", "#H2", 0.5),
        ("#H1", "N4", None),
        ("N4", "#H2", 0.5),
        ("N4", "C", None),
        ("#H2", "D", None),
    ]
    net = _build_network(labels, edges, retic_labels={"#H1", "#H2"})
    assert level(net) == 2
    return net


# ═══════════════════════════════════════════════════════════════════════
# 1. resolve_move_types
# ═══════════════════════════════════════════════════════════════════════

class TestResolveMoveTypes:

    def test_default_is_full_set(self):
        assert resolve_move_types() == list(DEFAULT_MOVE_TYPES)

    def test_opt_bl_drops_continuous(self):
        moves = resolve_move_types(opt_bl=True)
        for m in CONTINUOUS_MOVES:
            assert m not in moves
        # SPR (backbone) is retained when only opt_bl is set.
        assert SPR in moves

    def test_fix_st_drops_backbone(self):
        moves = resolve_move_types(fix_st=True)
        assert SPR not in moves
        # Continuous moves retained when only fix_st is set.
        assert ChangeNodeHeight in moves
        assert ChangeInheritanceProb in moves

    def test_both_flags_compose(self):
        moves = resolve_move_types(opt_bl=True, fix_st=True)
        for m in (*CONTINUOUS_MOVES, *BACKBONE_MOVES):
            assert m not in moves
        # Reticulation moves survive.
        for m in (AddReticulation, RemoveReticulation, RelocateReticulation,
                  ChangeReticSource, ChangeReticDest, FlipReticulation):
            assert m in moves

    def test_custom_base_not_mutated(self):
        base = [SPR, AddReticulation, ChangeNodeHeight]
        out = resolve_move_types(opt_bl=True, fix_st=True, base=base)
        assert base == [SPR, AddReticulation, ChangeNodeHeight]  # unchanged
        assert out == [AddReticulation]


# ═══════════════════════════════════════════════════════════════════════
# 2. make_level_validator
# ═══════════════════════════════════════════════════════════════════════

class TestLevelValidator:

    def test_none_returns_base_unchanged(self):
        base = make_level_validator(None).__class__
        sentinel = lambda model: True  # noqa: E731
        assert make_level_validator(None, base=sentinel) is sentinel

    def test_rejects_above_cap_accepts_within(self):
        validator = make_level_validator(1)

        lvl1 = Model()
        lvl1.network = _two_retic_level1_network()
        assert validator(lvl1) is True

        # Same level-1 network is rejected by a stricter level-0 cap.
        assert make_level_validator(0)(lvl1) is False

        # A genuine level-2 network is rejected by the level-1 cap.
        lvl2 = Model()
        lvl2.network = _level2_network()
        assert validator(lvl2) is False


# ═══════════════════════════════════════════════════════════════════════
# 3. Per-move max_level self-reject (the targeted regression)
# ═══════════════════════════════════════════════════════════════════════

class TestMaxLevelSelfReject:

    RELOCATION_MOVES = (
        RelocateReticulation,
        ChangeReticSource,
        ChangeReticDest,
    )

    @pytest.mark.parametrize("move_cls", RELOCATION_MOVES)
    def test_guarded_move_never_exceeds_level(self, move_cls):
        """With ``max_level=1`` the resulting network is always level <= 1."""
        for seed in range(40):
            net = _two_retic_level1_network()
            model = Model(rng=np.random.default_rng(seed))
            model.network = net
            model.update_network()

            move = move_cls(max_level=1)
            move.execute(model)
            assert level(model.network) <= 1, (
                f"{move_cls.__name__} (seed {seed}) produced level "
                f"{level(model.network)} despite max_level=1"
            )

    def test_unguarded_move_can_exceed_level(self):
        """Sanity: the *same* moves without the guard CAN reach level 2.

        Confirms the guard is doing real work (a reticulation-count cap
        alone would not have caught these).
        """
        reached_level_2 = False
        for move_cls in self.RELOCATION_MOVES:
            for seed in range(60):
                net = _two_retic_level1_network()
                model = Model(rng=np.random.default_rng(seed))
                model.network = net
                model.update_network()
                move_cls().execute(model)  # no max_level guard
                if level(model.network) >= 2:
                    reached_level_2 = True
                    break
            if reached_level_2:
                break
        assert reached_level_2, (
            "Expected at least one unguarded relocation to create a "
            "level-2 network; topology/seeds may need adjusting."
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. pseudo routing
# ═══════════════════════════════════════════════════════════════════════

class TestPseudoRouting:

    def test_infernetworkml_pseudo_matches_mpl(self):
        """``InferNetwork_ML.score(pseudo=True)`` == ``MPL.score`` (triplets)."""
        from phynetpy.MPL import MPL
        from phynetpy._infernetworkml import InferNetwork_ML

        mapping = _identity_map("A", "B", "C")
        st_newick = "((A:1,B:1):1,C:2);"
        gt = Network.from_newick("((A:0.5,B:0.5):1.5,C:2);")

        gts_mpl = GeneTrees(gene_tree_list=[Network.from_newick(
            "((A:0.5,B:0.5):1.5,C:2);")])
        gts_mpl.species_gene_mapping = mapping
        mpl_score = MPL(Network.from_newick(st_newick), gts_mpl, mapping).score()

        gts_inf = GeneTrees(gene_tree_list=[gt])
        gts_inf.species_gene_mapping = mapping
        inf = InferNetwork_ML(
            Network.from_newick(st_newick), gts_inf, mapping,
            max_reticulations=0,
        )
        pseudo_score = inf.score(pseudo=True)

        assert math.isfinite(pseudo_score)
        assert pseudo_score == pytest.approx(mpl_score, abs=1e-9)

    def test_mcmc_gt_mh_pseudo_rejected(self):
        """pseudo is not a calibrated posterior -> rejected with method='mh'."""
        from phynetpy.MCMC_GT import MCMC_GT

        mapping = _identity_map("A", "B", "C")
        gts = GeneTrees(gene_tree_list=[Network.from_newick(
            "((A:0.5,B:0.5):1.5,C:2);")])
        gts.species_gene_mapping = mapping
        mcmc = MCMC_GT(Network.from_newick("((A:1,B:1):1,C:2);"), gts, mapping)

        with pytest.raises(ValueError):
            mcmc.search(method="mh", num_iter=5, pseudo=True)


# ═══════════════════════════════════════════════════════════════════════
# 5. End-to-end smoke tests for each flag
# ═══════════════════════════════════════════════════════════════════════

class TestSearchFlagsSmoke:

    MAP = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
    ST = "(((A:1,B:1):1,C:2):1,D:3);"
    GTS = [
        "(((A:0.5,B:0.5):1.5,C:2):1,D:3);",
        "(((A:0.5,C:0.5):1.5,B:2):1,D:3);",
        "((A:1,B:1):1,(C:1,D:1):1);",
    ]

    def _mpl(self):
        from phynetpy.MPL import MPL
        gts = GeneTrees(
            gene_tree_list=[Network.from_newick(g) for g in self.GTS])
        gts.species_gene_mapping = self.MAP
        return MPL(Network.from_newick(self.ST), gts, self.MAP)

    def _mcmc(self):
        from phynetpy.MCMC_GT import MCMC_GT
        gts = GeneTrees(
            gene_tree_list=[Network.from_newick(g) for g in self.GTS])
        gts.species_gene_mapping = self.MAP
        return MCMC_GT(Network.from_newick(self.ST), gts, self.MAP)

    def _infer(self, max_ret=1):
        from phynetpy._infernetworkml import InferNetwork_ML
        gts = GeneTrees(
            gene_tree_list=[Network.from_newick(g) for g in self.GTS])
        gts.species_gene_mapping = self.MAP
        return InferNetwork_ML(
            Network.from_newick(self.ST), gts, self.MAP,
            max_reticulations=max_ret,
        )

    def test_mpl_opt_bl_finite(self):
        score = self._mpl().search(
            method="hc", num_iter=12, max_reticulations=1, opt_bl=True,
        )
        assert math.isfinite(score)

    def test_mpl_fix_st_finite(self):
        score = self._mpl().search(
            method="hc", num_iter=12, max_reticulations=1, fix_st=True,
        )
        assert math.isfinite(score)

    def test_mpl_pseudo_false_warns(self):
        with pytest.warns(UserWarning):
            self._mpl().search(
                method="hc", num_iter=5, max_reticulations=0, pseudo=False,
            )

    def test_mpl_max_lvl_enforced(self):
        mpl = self._mpl()
        mpl.search(
            method="hc", num_iter=40, max_reticulations=2, max_lvl=1,
            seed=0,
        )
        assert level(mpl.net) <= 1

    def test_mcmc_gt_opt_bl_finite(self):
        result = self._mcmc().search(
            method="hc", num_iter=12, max_reticulations=1, opt_bl=True,
            seed=0,
        )
        assert math.isfinite(result.best_log_posterior)

    def test_mcmc_gt_pseudo_hc_finite(self):
        result = self._mcmc().search(
            method="hc", num_iter=12, max_reticulations=1, pseudo=True,
            seed=0,
        )
        assert math.isfinite(result.best_log_posterior)

    def test_mcmc_gt_max_lvl_enforced(self):
        result = self._mcmc().search(
            method="hc", num_iter=40, max_reticulations=2, max_lvl=1,
            seed=0,
        )
        assert level(result.best_network) <= 1

    def test_infernetworkml_opt_bl_finite(self):
        result = self._infer(max_ret=1).search(
            num_runs=1, num_iter=12, opt_bl=True, seed=0,
        )
        assert math.isfinite(result.best_log_likelihood)

    def test_infernetworkml_pseudo_finite(self):
        result = self._infer(max_ret=1).search(
            num_runs=1, num_iter=12, pseudo=True, seed=0,
        )
        assert math.isfinite(result.best_log_likelihood)

    def test_infernetworkml_max_lvl_enforced(self):
        result = self._infer(max_ret=2).search(
            num_runs=1, num_iter=40, max_lvl=1, seed=0,
        )
        assert level(result.best_network) <= 1
