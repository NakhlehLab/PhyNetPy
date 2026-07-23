r"""Which internal branch of _coupled_gene_tree_reproposal fails on add?"""
from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import phynetpy._mcmc_seq as m
from phynetpy._mcmc_seq import (
    MCMC_SEQ, MCMCSeqPriors, op_add_reticulation_coupled,
    _gt_signature, _enumerate_scored_candidates, _logsumexp,
)
from phynetpy._msnc_density import build_network_msnc_index
from phynetpy.infer import JC69, simulate_multilocus
from _prior_only_validation import _tiny_true_net, MAPPING

m.SeqState.log_likelihood = lambda self: 0.0

reason = {}


def bump(k):
    reason[k] = reason.get(k, 0) + 1


def instrumented_repro(state, target_net, reverse_net, rng):
    eng = state._engine
    tgt_idx, tgt_h = build_network_msnc_index(target_net)
    rev_idx, rev_h = build_network_msnc_index(reverse_net)
    if tgt_h is None or rev_h is None:
        bump("tgt_or_rev_h_None")
        return None
    for i in range(eng.n_loci):
        g_i = state.gene_trees[i]
        h_i = state.gt_heights[i]
        target_sig = _gt_signature(g_i, h_i)
        fwd_sigs, fwd_lw, fwd_rebuild = _enumerate_scored_candidates(
            state, i, tgt_idx, tgt_h, state.theta, g_i, h_i)
        fwd_lse = _logsumexp([float(x) for x in fwd_lw])
        if not math.isfinite(fwd_lse):
            bump("fwd_lse_not_finite")
            return None
        probs = np.exp(fwd_lw - fwd_lse)
        total = float(probs.sum())
        if not math.isfinite(total) or total <= 0.0:
            bump("fwd_total_bad")
            return None
        probs = probs / total
        j = int(rng.choice(len(fwd_sigs), p=probs))
        chosen_gt, chosen_h = fwd_rebuild(j)
        rev_sigs, rev_lw, _ = _enumerate_scored_candidates(
            state, i, rev_idx, rev_h, state.theta, chosen_gt, chosen_h)
        rev_lse = _logsumexp([float(x) for x in rev_lw])
        if not math.isfinite(rev_lse):
            bump("rev_lse_not_finite")
            return None
        match = -1
        for jj, sig in enumerate(rev_sigs):
            if sig == target_sig:
                match = jj
                break
        if match < 0:
            bump("no_match_in_reverse_set")
            return None
        if not math.isfinite(float(rev_lw[match])):
            bump("matched_rev_lw_not_finite")
            return None
    bump("SUCCESS")
    return "ok"  # short-circuit; we only care about the reason


m._coupled_gene_tree_reproposal = instrumented_repro

data = simulate_multilocus(_tiny_true_net(), MAPPING, n_loci=4, seq_length=200,
                           theta=0.02, model=JC69(), seed=1)
priors = MCMCSeqPriors(max_reticulations=1, max_level=1,
                       use_diameter_prior=False, gamma_alpha=1.0,
                       gamma_beta=1.0, poisson_mean=1.0)
sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=priors)
rng = np.random.default_rng(0)

for _ in range(400):
    op_add_reticulation_coupled(sampler._new_state(), rng)

print("reproposal (add: target=reticulated) failure reasons:")
for k, v in sorted(reason.items(), key=lambda kv: -kv[1]):
    print(f"  {k:26s}: {v}")
