"""
Tests for :mod:`phynetpy._sim_seq` -- coalescent simulation of multilocus
sequence data (the generative model behind MCMC_SEQ).

The simulator is validated against *independent* theory rather than against the
MCMC_SEQ likelihood it will be tested with:

1. ``TestGeneTreeSim``: the network coalescent must (a) be ultrametric with all
   alleles as leaves, (b) never place a cross-species coalescence below the
   species divergence, (c) reproduce the analytic mean TMRCA ``D + theta/2``
   for two alleles above a divergence, and (d) route lineages through a
   reticulation in the gamma : (1 - gamma) proportion.

2. ``TestSequenceSim``: sequences evolved down a branch must recover the
   Jukes-Cantor path distance, and deep tips must approach the model's
   stationary base composition.

3. ``TestEndToEnd``: a simulated data set drops into MCMC_SEQ and the sampler
   recovers the planted clade.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phynetpy.Network import Network
from phynetpy._seq_likelihood import _node_height, JC69, HKY85, GTR
from phynetpy.infer import (
    SimulatedData,
    simulate_gene_tree,
    simulate_sequences,
    simulate_multilocus,
    MCMC_SEQ,
    MCMCSeqPriors,
)


def _descendant_leaves(net: Network, node) -> frozenset:
    """Set of leaf labels at or below ``node``."""
    kids = net.get_children(node)
    if not kids:
        return frozenset({node.label})
    acc: set = set()
    for c in kids:
        acc |= _descendant_leaves(net, c)
    return frozenset(acc)


def _has_clade(net: Network, clade: set) -> bool:
    """True if some node's descendant-leaf set equals ``clade`` exactly."""
    return any(_descendant_leaves(net, v) == frozenset(clade) for v in net.V())


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _tmrca(gene_tree: Network) -> float:
    """Root height (time to most-recent common ancestor) of a gene tree."""
    return _node_height(gene_tree, gene_tree.root(), {})


def _leaf_heights(gene_tree: Network) -> list[float]:
    cache: dict = {}
    return [
        _node_height(gene_tree, leaf, cache)
        for leaf in gene_tree.get_leaves()
    ]


# ═══════════════════════════════════════════════════════════════════════
# Gene-tree simulation
# ═══════════════════════════════════════════════════════════════════════

class TestGeneTreeSim:
    def test_leaves_are_the_sampled_alleles(self):
        net = Network.from_newick("((A:0.05,B:0.05)I:0.05,C:0.10)R;")
        rng = np.random.default_rng(0)
        gt = simulate_gene_tree(net, {"A": ["A"], "B": ["B"], "C": ["C"]}, 0.02, rng)
        assert sorted(l.label for l in gt.get_leaves()) == ["A", "B", "C"]

    def test_ultrametric_and_above_divergence(self):
        # A,B diverge at 0.05; their coalescence can only happen at or above it.
        net = Network.from_newick("(A:0.05,B:0.05)R;")
        rng = np.random.default_rng(1)
        for _ in range(200):
            gt = simulate_gene_tree(net, {"A": ["A"], "B": ["B"]}, 0.02, rng)
            assert _tmrca(gt) >= 0.05 - 1e-12
            # ultrametric: every leaf at height 0
            assert max(_leaf_heights(gt)) == pytest.approx(0.0, abs=1e-12)

    def test_mean_tmrca_matches_coalescent_theory(self):
        # Two alleles above a divergence D coalesce at rate 2/theta, so
        # E[TMRCA] = D + theta/2.
        D, theta = 0.05, 0.02
        net = Network.from_newick(f"(A:{D},B:{D})R;")
        rng = np.random.default_rng(7)
        ts = [
            _tmrca(simulate_gene_tree(net, {"A": ["A"], "B": ["B"]}, theta, rng))
            for _ in range(8000)
        ]
        assert np.mean(ts) == pytest.approx(D + theta / 2.0, abs=2e-3)

    def test_multiple_alleles_per_species(self):
        net = Network.from_newick("(A:0.05,B:0.05)R;")
        rng = np.random.default_rng(3)
        mapping = {"A": ["A_0", "A_1", "A_2"], "B": ["B_0", "B_1"]}
        gt = simulate_gene_tree(net, mapping, 0.05, rng)
        assert sorted(l.label for l in gt.get_leaves()) == [
            "A_0", "A_1", "A_2", "B_0", "B_1",
        ]
        # 5 tips -> 4 internal coalescences -> 9 nodes.
        assert len(gt.get_leaves()) == 5

    def test_reticulation_routes_by_gamma(self):
        # B is a hybrid: 70% A-side, 30% C-side.  With tiny theta the routed
        # lineage coalesces with its parent population almost immediately, so
        # B's cherry-sibling reveals the routing.
        nw = (
            "((A:2.0,(B:1.0)#H1:1.0[&gamma=0.7])PA:1.0,"
            "(C:2.0,#H1:1.0[&gamma=0.3])PC:1.0)R;"
        )
        net = Network.from_newick(nw)
        rng = np.random.default_rng(2)
        a_side = 0
        n = 3000
        for _ in range(n):
            gt = simulate_gene_tree(
                net, {"A": ["A"], "B": ["B"], "C": ["C"]}, 0.002, rng
            )
            leaves = {l.label: l for l in gt.get_leaves()}
            sib = {c.label for c in gt.get_children(gt.get_parents(leaves["B"])[0])}
            if sib == {"A", "B"}:
                a_side += 1
        # 3 standard errors of a Binomial(n, 0.7) proportion ~ 0.025.
        assert a_side / n == pytest.approx(0.7, abs=0.04)


# ═══════════════════════════════════════════════════════════════════════
# Sequence simulation
# ═══════════════════════════════════════════════════════════════════════

class TestSequenceSim:
    def test_shape_and_alphabet(self):
        gt = Network.from_newick("(A:0.1,B:0.1)R;")
        rng = np.random.default_rng(0)
        aln = simulate_sequences(gt, JC69(), 150, rng)
        assert set(aln) == {"A", "B"}
        assert all(len(s) == 150 for s in aln.values())
        assert set("".join(aln.values())) <= set("ACGT")

    def test_recovers_jukes_cantor_path_distance(self):
        # Tips separated by path length 2d; the JC-corrected distance estimate
        # should recover it.
        d = 0.1
        gt = Network.from_newick(f"(A:{d},B:{d})R;")
        rng = np.random.default_rng(5)
        aln = simulate_sequences(gt, JC69(), 6000, rng)
        sa, sb = aln["A"], aln["B"]
        p = sum(x != y for x, y in zip(sa, sb)) / len(sa)
        jc = -0.75 * math.log(1.0 - (4.0 / 3.0) * p)
        assert jc == pytest.approx(2 * d, abs=0.03)

    def test_deep_tips_approach_stationary_frequencies(self):
        # A long branch saturates: tip base composition -> pi.
        pi = [0.4, 0.3, 0.2, 0.1]
        model = HKY85(kappa=2.0, pi=pi)
        gt = Network.from_newick("(A:5.0,B:5.0)R;")
        rng = np.random.default_rng(9)
        aln = simulate_sequences(gt, model, 20000, rng)
        seq = aln["A"]
        freqs = [seq.count(b) / len(seq) for b in "ACGT"]
        assert freqs == pytest.approx(pi, abs=0.03)


# ═══════════════════════════════════════════════════════════════════════
# End-to-end multilocus simulation + recovery
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_simulated_data_bundle_feeds_mcmc_seq(self):
        net = Network.from_newick("((A:0.05,B:0.05)I:0.05,C:0.10)R;")
        mapping = {"A": ["A"], "B": ["B"], "C": ["C"]}
        data = simulate_multilocus(net, mapping, n_loci=3, seq_length=120,
                                   theta=0.02, seed=11)
        assert isinstance(data, SimulatedData)
        assert len(data.loci) == 3 and len(data.gene_trees) == 3
        assert data.species_of == {"A": "A", "B": "B", "C": "C"}
        kwargs = data.to_mcmc_seq_kwargs()
        assert set(kwargs) == {"loci", "mapping", "model", "theta"}
        # The bundle must construct a sampler and score a finite start state.
        sampler = MCMC_SEQ(**kwargs)
        assert math.isfinite(sampler.score())

    @pytest.mark.slow
    def test_recovers_planted_clade(self):
        # True species tree groups (A,B); simulate many loci, run a short
        # tree-only chain (max_reticulations=0 isolates topology recovery from
        # the RJMCMC dimension moves) and confirm the MAP keeps A,B together.
        net = Network.from_newick("((A:0.04,B:0.04)I:0.06,C:0.10)R;")
        mapping = {"A": ["A"], "B": ["B"], "C": ["C"]}
        data = simulate_multilocus(net, mapping, n_loci=12, seq_length=400,
                                   theta=0.015, seed=23)
        sampler = MCMC_SEQ(
            **data.to_mcmc_seq_kwargs(),
            priors=MCMCSeqPriors(max_reticulations=0),
        )
        result = sampler.search(num_iter=4000, burn_in=1000, sample_freq=50,
                                seed=23)
        assert _has_clade(result.map_network, {"A", "B"})
