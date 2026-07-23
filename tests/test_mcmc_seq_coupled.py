#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##############################################################################

"""Detailed-balance / reversibility tests for the *coupled* MCMC_SEQ moves.

The coupled add/delete (``op_add_reticulation_coupled`` /
``op_delete_reticulation_coupled``) change the species network *and* re-propose
every locus's gene tree in one reversible-jump Metropolis-Hastings step.  The
non-negotiable correctness property is exact reversibility of the log Hastings
ratio: for any transition ``x -> x'`` produced by the coupled add, the coupled
delete that returns ``x' -> x`` must report ``logH_del == -logH_add``.  These
tests verify that numerically, plus the structural invariants (NNI-neighbour
symmetry) the reverse-density bookkeeping relies on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors
from phynetpy import _mcmc_seq as ms


# ----------------------------------------------------------------------
# Fixture: a small 4-taxon, multi-locus SeqState with real sequence signal.
# ----------------------------------------------------------------------

def _state(seed: int = 0, *, max_level=None, max_reticulations: int = 4):
    rng = np.random.default_rng(seed)

    def rand(n):
        return "".join(rng.choice(list("ACGT"), size=n))

    def mutate(s, k):
        s = list(s)
        for _ in range(k):
            s[rng.integers(len(s))] = rng.choice(list("ACGT"))
        return "".join(s)

    def locus():
        core = rand(150)
        a = mutate(core, 6)
        b = mutate(a, 6)
        c = mutate(core, 22)
        d = mutate(c, 18)
        return {"A": a, "B": b, "C": c, "D": d}

    loci = [locus() for _ in range(4)]
    mapping = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
    inf = MCMC_SEQ(
        loci,
        mapping,
        priors=MCMCSeqPriors(
            max_reticulations=max_reticulations, max_level=max_level
        ),
    )
    return inf._new_state()


def _state_signature(state):
    """(network sig, per-locus gene-tree sigs, reticulation count)."""
    net_sig = ms._gt_signature(state.species_net, state.net_heights)
    gt_sigs = tuple(
        ms._gt_signature(gt, h)
        for gt, h in zip(state.gene_trees, state.gt_heights)
    )
    return net_sig, gt_sigs, state.num_reticulations()


# ----------------------------------------------------------------------
# 1. Structural: NNI-neighbour symmetry (reverse reachability).
# ----------------------------------------------------------------------

class TestCandidateSets:
    def test_candidate_sets_nontrivial(self):
        # The fixture gene trees must actually have NNI neighbours, otherwise
        # the coupling term is vacuous and the reversibility test proves nothing.
        state = _state(1)
        multi = 0
        for gt, h in zip(state.gene_trees, state.gt_heights):
            cset = ms._candidate_set(gt, h)
            if len(cset) > 1:
                multi += 1
        assert multi > 0, "no locus has a non-trivial NNI candidate set"

    def test_nni_neighbour_relation_symmetric(self):
        # If g' is an NNI neighbour of g, then g must lie in the candidate set
        # built around g' -- the property that guarantees the reverse gene-tree
        # proposal density is well-defined.
        state = _state(2)
        for gt, h in zip(state.gene_trees, state.gt_heights):
            g_sig = ms._gt_signature(gt, h)
            for cand, ch in ms._gene_tree_nni_neighbors(gt, h):
                back = ms._candidate_set(cand, ch)
                sigs = {ms._gt_signature(c, cch) for c, cch in back}
                assert g_sig in sigs, "NNI neighbour relation is not symmetric"


# ----------------------------------------------------------------------
# 2. Exact reversibility of the coupled Hastings ratio.
# ----------------------------------------------------------------------

class TestCoupledReversibility:
    def _successful_add(self, add_seed: int):
        """Build x, run one coupled add; return (state@x', logHR_add, x_sig)."""
        state = _state(0)
        x_sig = _state_signature(state)
        rng = np.random.default_rng(add_seed)
        add = None
        for _ in range(60):
            add = ms.op_add_reticulation_coupled(state, rng)
            if add is not None:
                break
        return state, add, x_sig

    def test_add_then_delete_inverts_hastings(self):
        # Master test: find an add x->x', then search delete seeds for one that
        # returns exactly to x; assert logH_del == -logH_add on that transition.
        add_seed = None
        for s in range(50):
            state, add, x_sig = self._successful_add(s)
            if add is not None and state.num_reticulations() == 1:
                add_seed = s
                add_hr = add[0]
                break
        assert add_seed is not None, "no coupled add succeeded across seeds"
        assert math.isfinite(add_hr)

        # x' is the current state; snapshot its signature for reset validation.
        xprime_sig = _state_signature(state)
        assert xprime_sig[2] == 1

        matched = 0
        checked_pairs = []
        for dseed in range(4000):
            # Rebuild x' deterministically (rebuild x, replay the same add).
            st, add, _ = self._successful_add(add_seed)
            assert add is not None and st.num_reticulations() == 1
            add_hr_rep = add[0]
            drng = np.random.default_rng(dseed)
            dele = ms.op_delete_reticulation_coupled(st, drng)
            if dele is None:
                continue
            del_hr, dundo = dele
            if _state_signature(st) == x_sig:
                matched += 1
                checked_pairs.append((add_hr_rep, del_hr))
                assert math.isclose(del_hr, -add_hr_rep, rel_tol=1e-9,
                                    abs_tol=1e-7), (
                    f"reversibility broken: logH_add={add_hr_rep:.10f} "
                    f"logH_del={del_hr:.10f} (sum={add_hr_rep + del_hr:.2e})"
                )
                if matched >= 3:
                    break
        assert matched > 0, (
            "no delete seed reproduced x exactly; cannot verify reversibility"
        )

    def test_delete_hastings_finite_and_restores(self):
        # A coupled delete that succeeds must report a finite HR and, via undo,
        # restore the pre-delete state exactly.
        state, add, _ = self._successful_add(7)
        if add is None or state.num_reticulations() != 1:
            pytest.skip("no successful add for this seed")
        pre_sig = _state_signature(state)
        drng = np.random.default_rng(3)
        for _ in range(200):
            dele = ms.op_delete_reticulation_coupled(state, drng)
            if dele is None:
                continue
            del_hr, dundo = dele
            assert math.isfinite(del_hr)
            dundo()
            assert _state_signature(state) == pre_sig
            return
        pytest.skip("no successful delete across attempts")


# ----------------------------------------------------------------------
# 2b. Dimension-preserving reticulation relocation.
# ----------------------------------------------------------------------

class TestRelocation:
    def _one_reticulation_state(self, add_seed: int):
        """A state carrying exactly one reticulation (add onto the fixture)."""
        state = _state(0)
        rng = np.random.default_rng(add_seed)
        for _ in range(80):
            add = ms.op_add_reticulation_coupled(state, rng)
            if add is not None and state.num_reticulations() == 1:
                return state
        return None

    def test_relocation_loghr_is_antisymmetric(self):
        # The core reversibility identity lives in the birth-Jacobian formula:
        # swapping the attached/detached roles (and the gene-tree densities)
        # must negate the ratio exactly.
        import random
        r = random.Random(0)
        for _ in range(200):
            a, b, c, d = (r.uniform(1e-3, 5.0) for _ in range(4))
            qf, qr = r.uniform(-8, 8), r.uniform(-8, 8)
            fwd = ms._relocation_loghr(a, b, c, d, qf, qr)
            rev = ms._relocation_loghr(c, d, a, b, qr, qf)
            assert math.isclose(fwd, -rev, rel_tol=1e-12, abs_tol=1e-12)

    def test_geometric_placements_gene_tree_independent(self):
        # Uniform placement selection is only reversible if the placement set
        # depends on the base network alone.  Two states with the SAME network
        # but different gene trees must enumerate the same placements.
        s1 = self._one_reticulation_state(3)
        if s1 is None:
            pytest.skip("no add produced a single reticulation")
        # Detach to a base network, then compare its placement set to a fresh
        # copy's (identical network, independent object / gene trees).
        p1 = ms._geometric_placements(s1.species_net, s1.net_heights)
        import copy as _copy
        net2 = _copy.deepcopy(s1.species_net)
        h2 = ms._heights(net2)
        p2 = ms._geometric_placements(net2, h2)
        assert len(p1) == len(p2) and len(p1) > 0
        assert {(*x["donor_labels"], *x["retic_labels"]) for x in p1} == \
               {(*x["donor_labels"], *x["retic_labels"]) for x in p2}

    def test_relocation_preserves_count_and_restores(self):
        # A successful relocation keeps the reticulation count fixed, reports a
        # finite HR, and undo restores the pre-move state exactly.
        state = self._one_reticulation_state(3)
        if state is None:
            pytest.skip("no add produced a single reticulation")
        pre_sig = _state_signature(state)
        rng = np.random.default_rng(11)
        for _ in range(300):
            rel = ms.op_relocate_reticulation_coupled(state, rng)
            if rel is None:
                continue
            hr, undo = rel
            assert math.isfinite(hr)
            assert state.num_reticulations() == 1  # dimension preserved
            undo()
            assert _state_signature(state) == pre_sig
            return
        pytest.skip("no successful relocation across attempts")


# ----------------------------------------------------------------------
# 3. Warm-start pipeline (MCMC_GT bootstrap -> coupled SEQ chain).
# ----------------------------------------------------------------------

class TestWarmStart:
    """The warm-start bootstrap must yield a valid seed and a runnable chain.

    Discovery/accuracy are properties of a full-length run (validated
    separately); here we only assert the plumbing: ``warm_start`` returns a
    network and installs it, and ``search(warm_start=True)`` runs and reports a
    finite MAP.  Kept small so it stays fast.
    """

    def _tiny(self, seed=0):
        rng = np.random.default_rng(seed)

        def rand(n):
            return "".join(rng.choice(list("ACGT"), size=n))

        def mutate(s, k):
            s = list(s)
            for _ in range(k):
                s[rng.integers(len(s))] = rng.choice(list("ACGT"))
            return "".join(s)

        def locus():
            core = rand(80)
            return {"A": mutate(core, 3), "B": mutate(core, 4),
                    "C": mutate(core, 14), "D": mutate(core, 16)}

        loci = [locus() for _ in range(3)]
        mapping = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
        return MCMC_SEQ(loci, mapping,
                        priors=MCMCSeqPriors(max_reticulations=2))

    def test_warm_start_returns_and_installs_network(self):
        from phynetpy.Network import Network
        s = self._tiny(1)
        net = s.warm_start(gt_iters=300, gt_thin=20, seed=1)
        assert isinstance(net, Network)
        assert s.species_net is net
        # A valid ultrametric species network scores finitely from this seed.
        assert math.isfinite(s.score())

    def test_search_with_warm_start_runs(self):
        s = self._tiny(2)
        res = s.search(num_iter=400, burn_in=100, sample_freq=20, seed=2,
                       warm_start=True, warm_start_kwargs={"gt_iters": 300})
        assert math.isfinite(res.map_log_posterior)
        assert res.num_iterations == 400
        assert all(math.isfinite(sm.log_posterior) for sm in res.samples)


# ----------------------------------------------------------------------
# 4. Model selection (AIC / BIC over reticulation count).
# ----------------------------------------------------------------------

class TestInformationCriteria:
    def test_free_parameter_count_penalty(self):
        # k = L + 3r: each extra reticulation must cost exactly 3 parameters.
        assert ms._count_free_parameters(6, 0) == 6
        assert ms._count_free_parameters(6, 1) == 9
        assert ms._count_free_parameters(6, 2) == 12

    def test_information_criteria_formulas(self):
        ic = ms._information_criteria(log_likelihood=-100.0, k=5, n=200)
        assert math.isclose(ic["AIC"], 2 * 5 - 2 * (-100.0))
        assert math.isclose(ic["BIC"], 5 * math.log(200) - 2 * (-100.0))
        # AICc > AIC for finite n, and reduces to AIC when n is unknown.
        assert ic["AICc"] > ic["AIC"]
        ic2 = ms._information_criteria(-100.0, 5, None)
        assert math.isclose(ic2["AICc"], ic2["AIC"])
        assert math.isnan(ic2["BIC"])

    def test_result_model_selection_populated(self):
        from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors
        rng = np.random.default_rng(5)

        def rand(n):
            return "".join(rng.choice(list("ACGT"), size=n))

        def mutate(s, k):
            s = list(s)
            for _ in range(k):
                s[rng.integers(len(s))] = rng.choice(list("ACGT"))
            return "".join(s)

        def locus():
            core = rand(100)
            return {"A": mutate(core, 4), "B": mutate(core, 5),
                    "C": mutate(core, 15), "D": mutate(core, 17)}

        loci = [locus() for _ in range(3)]
        mapping = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
        s = MCMC_SEQ(loci, mapping, priors=MCMCSeqPriors(max_reticulations=2))
        res = s.search(num_iter=600, burn_in=150, sample_freq=15, seed=5)
        assert res.total_sites == 300 and res.num_leaves == 4
        ic = res.information_criteria()
        assert ic is not None and math.isfinite(ic["AIC"])
        rows = res.model_selection_by_reticulation()
        assert rows and all("dAIC" in r for r in rows)
        # The best model has dAIC == 0.
        assert min(r["dAIC"] for r in rows) == 0.0


# ----------------------------------------------------------------------
# 5. Level cap (level-1 mode) -- state-space restriction + speed switch.
# ----------------------------------------------------------------------

from phynetpy.GraphUtils import level as _glevel


class TestLevelCap:
    """``MCMCSeqPriors.max_level`` restricts the sampler's support.

    The guard lives in the reticulation add / relocate operators and fires
    *before* the coupled gene-tree re-proposal + MSNC scoring, so it both
    (a) keeps every reachable network within the level cap and (b) skips the
    combinatorial displayed-tree blow-up that above-level networks incur.
    """

    def _make_one_reticulation(self, state, seed: int, tries: int = 120) -> bool:
        """Drive one coupled add so ``state`` carries a single reticulation."""
        rng = np.random.default_rng(seed)
        for _ in range(tries):
            if (
                ms.op_add_reticulation_coupled(state, rng) is not None
                and state.num_reticulations() == 1
            ):
                return True
        return False

    def test_add_operator_never_exceeds_cap(self):
        # Operator-level guarantee: starting from a level-1 network, every
        # coupled add that *succeeds* under max_level=1 keeps level <= 1.
        state = _state(0, max_level=1, max_reticulations=4)
        assert self._make_one_reticulation(state, seed=5)
        assert _glevel(state.species_net) == 1
        rng = np.random.default_rng(9)
        for _ in range(500):
            if ms.op_add_reticulation_coupled(state, rng) is not None:
                assert _glevel(state.species_net) <= 1

    def test_cap_is_not_vacuous(self):
        # Without the cap, the same add move DOES reach level >= 2 -- proving
        # the guard prevents genuinely reachable states, not a no-op.
        reached_two = False
        for seed in range(6):
            state = _state(0, max_level=None, max_reticulations=4)
            if not self._make_one_reticulation(state, seed=seed + 1):
                continue
            rng = np.random.default_rng(seed + 100)
            for _ in range(400):
                if ms.op_add_reticulation_coupled(state, rng) is not None:
                    if _glevel(state.species_net) >= 2:
                        reached_two = True
                        break
            if reached_two:
                break
        assert reached_two, "level-2 network was never reachable; test vacuous"

    def test_search_stays_within_level_cap(self):
        # End-to-end guarantee: a full short chain with max_level=1 (and a
        # generous reticulation cap) samples only level<=1 networks, and the
        # MAP network is level<=1.
        from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors

        rng = np.random.default_rng(7)

        def rand(n):
            return "".join(rng.choice(list("ACGT"), size=n))

        def mutate(s, k):
            s = list(s)
            for _ in range(k):
                s[rng.integers(len(s))] = rng.choice(list("ACGT"))
            return "".join(s)

        def locus():
            core = rand(120)
            return {"A": mutate(core, 4), "B": mutate(core, 6),
                    "C": mutate(core, 16), "D": mutate(core, 18)}

        loci = [locus() for _ in range(4)]
        mapping = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
        s = MCMC_SEQ(
            loci,
            mapping,
            priors=MCMCSeqPriors(max_reticulations=4, max_level=1),
        )
        res = s.search(num_iter=800, burn_in=200, sample_freq=20, seed=7)
        assert _glevel(res.map_network) <= 1
        from phynetpy.Network import Network
        for sm in res.samples:
            net = Network.from_newick(sm.network_newick)
            assert _glevel(net) <= 1
