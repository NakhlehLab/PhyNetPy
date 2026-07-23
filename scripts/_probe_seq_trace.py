"""Trace reticulation count + logP over a SEQ chain started from the truth.

The target strongly prefers the reticulation (+194 logP at 60 loci), yet the
chain reports 0 reticulations at the MAP.  This logs, every `freq` iterations,
the current reticulation count and logP, and separately counts how often each
operator is *proposed*, *accepted*, and (for add/delete) how the acceptance
splits.  Goal: see whether the reticulation is lost early and never regained,
and which move is responsible.
"""
import os, sys, copy, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy import _mcmc_seq as M

true_net = build_true_network()
data = simulate_multilocus(true_net, MAPPING, n_loci=30, seq_length=400,
                           theta=0.02, model=JC69(), seed=7)

sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                   priors=MCMCSeqPriors(max_reticulations=2))
sampler.species_net = copy.deepcopy(true_net)

rng = np.random.default_rng(1)
kernel = M.MCMCSeqKernel(rng, max_reticulations=2)
state = sampler._new_state()
cur = state.log_posterior()

# per-op counters
names = {op.__name__: [0, 0] for op, _ in kernel._ops}  # [proposed, accepted]

def op_name_of(idx):
    return kernel._ops[idx][0].__name__

NUM = 20000
first_loss = None
regains = 0
prev_ret = state.num_reticulations()
for it in range(NUM):
    idx = int(kernel.rng.choice(len(kernel._ops), p=kernel._weights))
    op = kernel._ops[idx][0]
    nm = op.__name__
    try:
        proposal = op(state, kernel.rng)
    except Exception:
        proposal = None
    names[nm][0] += 1
    if proposal is not None:
        loghr, undo = proposal
        prop = state.log_posterior()
        if not math.isfinite(prop):
            undo()
        else:
            log_alpha = (prop - cur) + loghr
            if math.log(kernel.rng.random()) < log_alpha:
                cur = prop
                names[nm][1] += 1
            else:
                undo()
    ret = state.num_reticulations()
    if prev_ret >= 1 and ret == 0 and first_loss is None:
        first_loss = it
    if prev_ret == 0 and ret >= 1:
        regains += 1
    prev_ret = ret
    if it % 2000 == 0:
        print(f"it={it:6d} logP={cur:10.2f} retic={ret}")

print(f"\nfirst reticulation loss at iter: {first_loss}")
print(f"times reticulation regained (0->1): {regains}")
print(f"final retic={state.num_reticulations()} logP={cur:.2f}")
print("\nop proposed / accepted (accept-rate):")
for nm, (p, a) in names.items():
    print(f"  {nm:24s} {p:6d} / {a:6d}  ({a/max(1,p):.3f})")
