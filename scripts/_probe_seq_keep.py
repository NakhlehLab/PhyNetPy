"""Forced comparison: reticulated chain vs tree chain, gene trees free.

Runs two chains on the SAME data, both starting from the true network:
  (K) delete-reticulation DISABLED (reticulation is pinned; everything else
      -- gene trees, heights, gamma, theta -- samples freely),
  (F) the normal kernel (free to delete).
If (K) reaches a clearly higher logP than (F), the posterior genuinely prefers
the reticulation and the ONLY thing stopping the free chain is the add/delete
mixing barrier -- which is then the thing to fix.
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

def run(pinned, seed=1, num=20000):
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=2))
    sampler.species_net = copy.deepcopy(true_net)
    rng = np.random.default_rng(seed)
    kernel = M.MCMCSeqKernel(rng, max_reticulations=2)
    if pinned:
        # drop add + delete reticulation ops, renormalise weights
        kernel._ops = [(op, w) for op, w in kernel._ops
                       if op.__name__ not in
                       ("op_add_reticulation", "op_delete_reticulation")]
        kernel._weights = np.asarray([w for _, w in kernel._ops])
        kernel._weights /= kernel._weights.sum()
    state = sampler._new_state()
    cur = state.log_posterior()
    best = cur
    for it in range(num):
        proposal = kernel.propose(state)
        if proposal is not None:
            loghr, undo = proposal
            prop = state.log_posterior()
            if not math.isfinite(prop):
                undo()
            elif math.log(rng.random()) < (prop - cur) + loghr:
                cur = prop
                best = max(best, cur)
            else:
                undo()
    return best, state.num_reticulations()

bk, rk = run(pinned=True)
bf, rf = run(pinned=False)
print(f"[pinned retic] best logP={bk:.2f} final_retic={rk}")
print(f"[free        ] best logP={bf:.2f} final_retic={rf}")
print(f"pinned - free = {bk - bf:+.2f}  "
      f"({'reticulation preferred -> add barrier is the bug' if bk - bf > 5 else 'tree competitive'})")
