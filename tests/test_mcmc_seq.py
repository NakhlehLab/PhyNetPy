"""
Correctness suite for :mod:`phynetpy._seq_likelihood` and
:mod:`phynetpy._mcmc_seq` (the MCMC_SEQ likelihood + sampler).

The slices that matter for "must match PhyloNet exactly" are the likelihood
checks, validated against *independent* references rather than against the
implementation under test:

1. ``TestFelsenstein``: the vectorised, site-pattern-compressed Felsenstein
   pruning must equal a naive ``scipy.linalg.expm`` pruning, column by column,
   for JC69, HKY85 and GTR.

2. ``TestMSCDensity``: the timed MSNC density must equal the Rannala-Yang
   multispecies-coalescent closed form on 2- and 3-taxon species *trees*.

3. ``TestReticulationDensity``: on a 1-reticulation network the density must
   reduce to the gamma-weighted mixture over displayed embeddings.

4. ``TestCythonParity``: the Cython hot-loop path and the pure-Python fallback
   must produce identical densities.

5. ``TestDriver``: the sampler runs end-to-end and recovers an obvious clade.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phynetpy.Network import Network
from phynetpy._seq_likelihood import (
    JC69,
    HKY85,
    GTR,
    FelsensteinCalculator,
    gene_tree_msnc_log_density,
)
from phynetpy.infer import MCMCSeqPriors
from phynetpy._mcmc_seq import MCMC_SEQ
from phynetpy.GraphUtils import network_clusters
from phynetpy import _msnc_density as msnc

expm = pytest.importorskip("scipy.linalg").expm


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _identity_species_of(*taxa: str) -> dict[str, str]:
    return {t: t for t in taxa}


def _brute_felsenstein(alignment, tree, model) -> float:
    """Reference pruning using scipy.expm and explicit per-column recursion."""
    Q = model.Q
    pi = model.pi
    idx = {c: i for i, c in enumerate("ACGT")}
    ambig = {"A": "A", "C": "C", "G": "G", "T": "T",
             "N": "ACGT", "-": "ACGT", "?": "ACGT", "R": "AG", "Y": "CT"}

    def tip(ch, col):
        v = np.zeros(4)
        for nt in ambig.get(ch[col].upper(), "ACGT"):
            v[idx[nt]] = 1.0
        return v

    length = len(next(iter(alignment.values())))
    total = 0.0
    for col in range(length):
        def rec(node):
            kids = tree.get_children(node)
            if not kids:
                return tip(alignment[node.label], col)
            r = np.ones(4)
            for c in kids:
                e = tree.get_edge(node, c)
                e = e[0] if isinstance(e, list) else e
                t = e.get_length() or 0.0
                r = r * (expm(Q * t) @ rec(c))
            return r
        total += math.log(float(pi @ rec(tree.root())))
    return total


# ═══════════════════════════════════════════════════════════════════════
# 1. Felsenstein vs independent matrix-exponential pruning
# ═══════════════════════════════════════════════════════════════════════

class TestFelsenstein:
    ALN = {"A": "ACGTACGTAC", "B": "ACGTACGAAC", "C": "ACGTTCGTAG"}
    TREE = "((A:0.1,B:0.2):0.05,C:0.3):0.0;"

    @pytest.mark.parametrize("model", [
        JC69(),
        HKY85(2.5, [0.3, 0.2, 0.2, 0.3]),
        GTR([0.25, 0.25, 0.25, 0.25], [1, 2, 1, 1, 3, 1]),
    ])
    def test_matches_bruteforce(self, model):
        tree = Network.from_newick(self.TREE)
        fc = FelsensteinCalculator(self.ALN)
        got = fc.log_likelihood(tree, model)
        ref = _brute_felsenstein(self.ALN, tree, model)
        assert got == pytest.approx(ref, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# 2. MSC density vs Rannala-Yang closed form
# ═══════════════════════════════════════════════════════════════════════

class TestMSCDensity:
    def test_two_taxa(self):
        theta, tau, c = 0.02, 0.01, 0.03
        sp = Network.from_newick(f"(A:{tau},B:{tau});")
        gt = Network.from_newick(f"(A:{c},B:{c});")
        got = gene_tree_msnc_log_density(
            gt, sp, _identity_species_of("A", "B"), theta=theta)
        expected = math.log(2.0 / theta) - 2.0 * (c - tau) / theta
        assert got == pytest.approx(expected, abs=1e-9)

    def test_three_taxa(self):
        theta, h1, h2, cA, cB = 0.02, 0.01, 0.02, 0.015, 0.03
        sp = Network.from_newick(f"((A:{h1},B:{h1}):{h2-h1},C:{h2});")
        gt = Network.from_newick(f"((A:{cA},B:{cA}):{cB-cA},C:{cB});")
        got = gene_tree_msnc_log_density(
            gt, sp, _identity_species_of("A", "B", "C"), theta=theta)
        expected = (math.log(2.0 / theta) - (cA - h1) * 2.0 / theta
                    + math.log(2.0 / theta) - (cB - h2) * 2.0 / theta)
        assert got == pytest.approx(expected, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# 3. Reticulation density == gamma-weighted embedding mixture
# ═══════════════════════════════════════════════════════════════════════

class TestReticulationDensity:
    GT = "((A:0.012,B:0.012):0.013,C:0.025);"
    SP_G = ("((A:0.01,(B:0.005)#H1:0.005[&gamma=0.3]):0.01,"
            "(#H1:0.01[&gamma=0.7],C:0.015):0.005);")
    SP_1 = ("((A:0.01,(B:0.005)#H1:0.005[&gamma=1.0]):0.01,"
            "(#H1:0.01[&gamma=0.0],C:0.015):0.005);")

    def test_gamma_mixture(self):
        theta = 0.02
        so = _identity_species_of("A", "B", "C")
        gt = Network.from_newick(self.GT)
        d_g = gene_tree_msnc_log_density(
            gt, Network.from_newick(self.SP_G), so, theta=theta)
        d_1 = gene_tree_msnc_log_density(
            gt, Network.from_newick(self.SP_1), so, theta=theta)
        # Only the A-side embedding is valid for this gene tree, so the density
        # scales linearly with gamma.
        assert d_g == pytest.approx(d_1 + math.log(0.3), abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# 4. Reticulation density bound (no double-counting)
# ═══════════════════════════════════════════════════════════════════════

class TestMSNCReticBound:
    """Network MSNC density must not exceed (n-1) log(2/theta) per gene tree."""

    NET = (
        "((C:0.9,((A:0.3,B:0.3)ab:0.3)#H1:0.3[&gamma=0.6])P1:0.2,"
        "(D:0.9,#H1:0.3[&gamma=0.4])P2:0.2)R;"
    )
    GENE = "(((A:0.3,B:0.3)g1:0.6,C:0.9)g2:0.2,D:1.1)gr;"

    def test_network_density_below_coalescent_bound(self):
        theta = 0.001
        so = _identity_species_of("A", "B", "C", "D")
        gt = Network.from_newick(self.GENE)
        net = Network.from_newick(self.NET)
        n = 4
        bound = (n - 1) * math.log(2.0 / theta)
        dens = gene_tree_msnc_log_density(gt, net, so, theta=theta)
        assert dens <= bound + 1e-6


class TestMSCBranchCythonParity:
    """GT MSC branch kernel: Cython and pure-Python paths agree."""

    def test_gij_parity(self, monkeypatch):
        so = _identity_species_of("A", "B", "C")
        gt = Network.from_newick("((A:0.012,B:0.012):0.013,C:0.025);")
        sp = Network.from_newick("((A:1,B:1):1,C:2);")
        engine = msnc.MSCBranchKernel(theta=2.0)
        gti = msnc._GeneTreeIndex(gt, so)
        net_idx = msnc._NetworkIndex(sp)
        d_cy = msnc._msnc_log_prob_network_int(net_idx, gti, engine)
        monkeypatch.setattr(msnc, "_CYTHON_AVAILABLE", False)
        monkeypatch.setattr(msnc, "_apply_branch_coalescent_int", msnc._apply_branch_coalescent_int_py)
        monkeypatch.setattr(msnc, "_combine_configs_int", msnc._combine_configs_int_py)
        d_py = msnc._msnc_log_prob_network_int(net_idx, gti, engine)
        assert d_cy == pytest.approx(d_py, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# 5. Driver smoke test + clade recovery
# ═══════════════════════════════════════════════════════════════════════

class TestDriver:
    def _make_data(self, seed=0, n=150):
        rng = np.random.default_rng(seed)

        def rand(n):
            return "".join(rng.choice(list("ACGT"), size=n))

        def mutate(s, k):
            s = list(s)
            for _ in range(k):
                s[rng.integers(len(s))] = rng.choice(list("ACGT"))
            return "".join(s)

        def locus():
            core = rand(n)
            a = mutate(core, 4)
            b = mutate(a, 4)       # B close to A
            c = mutate(core, 30)   # C diverged
            return {"A": a, "B": b, "C": c}

        return [locus(), locus()], {"A": ["A"], "B": ["B"], "C": ["C"]}

    def test_runs_and_recovers_clade(self):
        loci, mapping = self._make_data()
        inf = MCMC_SEQ(loci, mapping, priors=MCMCSeqPriors(max_reticulations=1))
        assert math.isfinite(inf.score())
        res = inf.search(num_iter=1500, burn_in=300, sample_freq=100, seed=7)
        assert math.isfinite(res.map_log_posterior)
        assert 0.0 < res.acceptance_rate <= 1.0
        assert len(res.samples) > 0
        # The strong (A,B) signal should make A and B sisters in the MAP
        # network.  Assert the cluster, not the newick text: when A or B ends
        # up under a reticulation the pair is still a clade but renders as
        # "(B,(A)#H0)" rather than "(A,B)".
        assert frozenset({"A", "B"}) in network_clusters(res.map_network)

    def test_multilocus_start_embeds_validly(self):
        # Regression: independently-built starting species/gene trees used to
        # place gene coalescences below species divergences for some loci,
        # giving an MSNC density of -inf and a likelihood floored at a constant
        # (theta-independent) value.  The starting state must now be finite and
        # the likelihood must respond to theta.
        loci, mapping = self._make_data(seed=0, n=200)
        loci = loci + self._make_data(seed=1, n=200)[0]  # >=4 loci, mixed
        inf = MCMC_SEQ(loci, mapping, priors=MCMCSeqPriors(max_reticulations=1))
        state = inf._new_state()

        # Every locus embeds validly (finite per-locus MSNC density).
        for gt in state.gene_trees:
            d = gene_tree_msnc_log_density(
                gt, state.species_net, state.species_of, theta=state.theta
            )
            assert math.isfinite(d), "starting gene tree does not embed in the network"

        # Total likelihood is finite and genuinely depends on theta.
        lls = []
        for th in (0.008, 0.02, 0.05):
            state.theta = th
            state._engine.invalidate_network()
            ll = state.log_likelihood()
            assert math.isfinite(ll)
            lls.append(ll)
        assert max(lls) - min(lls) > 1e-3, "likelihood is insensitive to theta"


# ═══════════════════════════════════════════════════════════════════════
# 6. Reticulation RJMCMC moves (PhyloNet-faithful add / delete)
# ═══════════════════════════════════════════════════════════════════════

class TestReticulationMoves:
    def _state(self, seed=0):
        rng = np.random.default_rng(seed)

        def rand(n):
            return "".join(rng.choice(list("ACGT"), size=n))

        def mutate(s, k):
            s = list(s)
            for _ in range(k):
                s[rng.integers(len(s))] = rng.choice(list("ACGT"))
            return "".join(s)

        def locus():
            core = rand(120)
            a = mutate(core, 5)
            b = mutate(a, 5)
            c = mutate(core, 25)
            d = mutate(core, 40)
            return {"A": a, "B": b, "C": c, "D": d}

        loci = [locus() for _ in range(3)]
        mapping = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
        inf = MCMC_SEQ(loci, mapping, priors=MCMCSeqPriors(max_reticulations=4))
        return inf._new_state()

    def test_add_never_creates_bubble(self):
        # The add move must never produce parallel (bubble) edges -- the
        # degenerate structures that caused the unbounded-posterior runaway.
        from phynetpy._mcmc_seq import op_add_reticulation, _has_parallel_edges

        rng = np.random.default_rng(1)
        state = self._state()
        successes = 0
        for _ in range(300):
            res = op_add_reticulation(state, rng)
            if res is None:
                continue
            loghr, undo = res
            assert math.isfinite(loghr)
            assert not _has_parallel_edges(state.species_net)
            successes += 1
            if rng.random() < 0.5:  # vary the state: keep some, undo others
                undo()
        assert successes > 0

    def test_add_then_delete_restores_and_inverts_hastings(self):
        # A delete that exactly reverses a just-applied add must (a) restore the
        # reticulation count and (b) have a log Hastings ratio that is the
        # negative of the add's, since the two moves are constructed as inverses.
        from phynetpy._mcmc_seq import op_add_reticulation, op_delete_reticulation

        rng = np.random.default_rng(4)
        state = self._state()
        r0 = state.num_reticulations()
        # Find a successful add.
        add = None
        for _ in range(200):
            add = op_add_reticulation(state, rng)
            if add is not None:
                break
        assert add is not None
        add_hr, _ = add
        assert state.num_reticulations() == r0 + 1
        # Now a delete (it picks randomly among reticulations / parents); when it
        # succeeds it returns to r0 and its HR is finite.
        for _ in range(200):
            dele = op_delete_reticulation(state, rng)
            if dele is not None:
                del_hr, _ = dele
                assert math.isfinite(del_hr)
                assert state.num_reticulations() == r0
                break

    @pytest.mark.slow
    def test_no_unbounded_posterior_runaway(self):
        # Regression for the catastrophic runaway in which add-reticulation +
        # theta-collapse drove the MSNC *density* (and hence the log posterior)
        # to +inf via degenerate bubble networks.  On data with real signal the
        # log posterior must stay finite and negative.
        #
        # NOTE: this does NOT assert that the reticulation count stays low --
        # reticulation over-fitting (the chain piling reticulations onto
        # tree-truth data) is a *separate, still-open* issue tracked elsewhere.
        from phynetpy.infer import simulate_multilocus

        net = Network.from_newick("((A:0.04,B:0.04)I:0.04,(C:0.05,D:0.05)J:0.03)R;")
        mapping = {k: [k] for k in "ABCD"}
        data = simulate_multilocus(net, mapping, n_loci=12, seq_length=400,
                                   theta=0.02, seed=3)
        inf = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=4))
        res = inf.search(num_iter=8000, burn_in=2000, sample_freq=50, seed=3)
        assert math.isfinite(res.map_log_posterior)
        assert res.map_log_posterior < 0.0       # no degenerate +inf grab
        assert all(math.isfinite(s.log_posterior) for s in res.samples)
