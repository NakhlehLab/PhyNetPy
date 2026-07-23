"""Long single informed chain from a tree start: does retic fraction grow?

Tracks the reticulation count in windows across a long chain so we can see
whether discovery happens slowly (fraction of time with a reticulation grows)
or essentially never.
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy import _mcmc_seq as M

true_net = build_true_network()
data = simulate_multilocus(true_net, MAPPING, n_loci=30, seq_length=600,
                           theta=0.02, model=JC69(), seed=7)

sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                   priors=MCMCSeqPriors(max_reticulations=2))
rng = np.random.default_rng(21)
kernel = M.MCMCSeqKernel(rng, max_reticulations=2)
state = sampler._new_state()
cur = state.log_posterior()

NUM = 60000
WIN = 5000
win_retic = 0
win_i = 0
t0 = time.perf_counter()
adds = 0
for it in range(NUM):
    proposal = kernel.propose(state)
    if proposal is not None:
        loghr, undo = proposal
        prop = state.log_posterior()
        if not math.isfinite(prop):
            undo()
        elif math.log(rng.random()) < (prop - cur) + loghr:
            cur = prop
        else:
            undo()
    if state.num_reticulations() >= 1:
        win_retic += 1
    win_i += 1
    if win_i == WIN:
        print(f"iter {it+1:6d}: retic_fraction(last {WIN})={win_retic/WIN:.3f} logP={cur:.1f}")
        win_retic = 0
        win_i = 0
dt = time.perf_counter() - t0
print(f"done: {1000*dt/NUM:.2f} ms/it, final retic={state.num_reticulations()}")
