r"""Find WHICH return-None path kills op_add_reticulation_coupled from a tree."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import phynetpy._mcmc_seq as m
from phynetpy._mcmc_seq import MCMC_SEQ, MCMCSeqPriors, op_add_reticulation_coupled
from phynetpy.infer import JC69, simulate_multilocus
from _prior_only_validation import _tiny_true_net, MAPPING

m.SeqState.log_likelihood = lambda self: 0.0

tally = {"placements_empty": 0, "placements_ok": 0,
         "repro_none": 0, "repro_ok": 0}

_orig_place = m._add_reticulation_placements
_orig_repro = m._coupled_gene_tree_reproposal


def place_wrap(state, net, heights):
    out = _orig_place(state, net, heights)
    if not out:
        tally["placements_empty"] += 1
    else:
        tally["placements_ok"] += 1
    return out


def repro_wrap(state, target_net, reverse_net, rng):
    out = _orig_repro(state, target_net, reverse_net, rng)
    if out is None:
        tally["repro_none"] += 1
    else:
        tally["repro_ok"] += 1
    return out


m._add_reticulation_placements = place_wrap
m._coupled_gene_tree_reproposal = repro_wrap

data = simulate_multilocus(_tiny_true_net(), MAPPING, n_loci=4, seq_length=200,
                           theta=0.02, model=JC69(), seed=1)
priors = MCMCSeqPriors(max_reticulations=1, max_level=1,
                       use_diameter_prior=False, gamma_alpha=1.0,
                       gamma_beta=1.0, poisson_mean=1.0)
sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=priors)
rng = np.random.default_rng(0)

n = 400
scored = 0
for _ in range(n):
    res = op_add_reticulation_coupled(sampler._new_state(), rng)
    if res is not None:
        scored += 1

print(f"add-from-tree calls: {n}, scored (returned loghr): {scored}")
print("instrumented tally:")
for k, v in tally.items():
    print(f"  {k:18s}: {v}")
