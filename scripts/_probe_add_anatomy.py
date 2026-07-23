"""Anatomy of add-reticulation rejections.

Runs a tree-state chain (gene trees pre-optimised), then repeatedly calls
op_add_reticulation and records, per attempt:
  * whether it produced a proposal at all (vs rejected by guards / None),
  * the log-likelihood delta (prop - cur),
  * the log Hastings ratio,
  * the resulting log acceptance prob.
Prints a histogram-ish summary to reveal whether adds die at the guards or at
the likelihood.
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
rng = np.random.default_rng(1)
kernel = M.MCMCSeqKernel(rng, max_reticulations=2)
kernel._ops = [(op, w) for op, w in kernel._ops
               if op.__name__ not in ("op_add_reticulation", "op_delete_reticulation")]
kernel._weights = np.asarray([w for _, w in kernel._ops]); kernel._weights /= kernel._weights.sum()
state = sampler._new_state()
cur = state.log_posterior()
# burn in the tree state
for _ in range(4000):
    p = kernel.propose(state)
    if p is not None:
        loghr, undo = p
        prop = state.log_posterior()
        if math.isfinite(prop) and math.log(rng.random()) < (prop - cur) + loghr:
            cur = prop
        else:
            undo()
print(f"tree state burned in: logP={cur:.2f} retic={state.num_reticulations()}")

none_count = 0
dl = []   # likelihood deltas
hr = []   # hastings ratios
la = []   # log acceptance
for _ in range(3000):
    prop = M.op_add_reticulation(state, rng)
    if prop is None:
        none_count += 1
        continue
    loghr, undo = prop
    p = state.log_posterior()
    undo()
    if not math.isfinite(p):
        dl.append(float("-inf")); continue
    d = p - cur
    dl.append(d); hr.append(loghr); la.append(d + loghr)

produced = len(dl)
finite = [d for d in dl if math.isfinite(d)]
print(f"attempts=3000  None(guards)={none_count}  produced={produced}  "
      f"inf_likelihood={produced - len(finite)}")
if finite:
    finite.sort()
    print(f"likelihood delta (prop-cur): min={min(finite):.1f} "
          f"median={finite[len(finite)//2]:.1f} max={max(finite):.1f}")
if hr:
    hr.sort()
    print(f"log Hastings ratio: min={min(hr):.2f} median={hr[len(hr)//2]:.2f} max={max(hr):.2f}")
if la:
    la.sort()
    best = max(la)
    n_accept = sum(1 for x in la if x >= 0)
    print(f"log acceptance (delta+HR): max={best:.2f} n(>=0)={n_accept}/{len(la)}")
    print(f"  -> best add would be accepted with prob exp(min(0,{best:.1f})) = {math.exp(min(0,best)):.2e}")
