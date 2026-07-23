"""Discovery from a TREE start with the informed add move (short budget)."""
import os, sys, copy, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy import _mcmc_seq as M

true_net = build_true_network()
data = simulate_multilocus(true_net, MAPPING, n_loci=30, seq_length=400,
                           theta=0.02, model=JC69(), seed=7)

# Start from the default UPGMA tree (0 reticulations).
sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                   priors=MCMCSeqPriors(max_reticulations=2))
rng = np.random.default_rng(11)
kernel = M.MCMCSeqKernel(rng, max_reticulations=2)
state = sampler._new_state()
cur = state.log_posterior()

first_add = None
t0 = time.perf_counter()
NUM = 12000
n_add_prop = n_add_acc = 0
for it in range(NUM):
    idx = int(kernel.rng.choice(len(kernel._ops), p=kernel._weights))
    op = kernel._ops[idx][0]
    is_add = op.__name__ == "op_add_reticulation"
    try:
        proposal = op(state, kernel.rng)
    except Exception:
        proposal = None
    if is_add and proposal is not None:
        n_add_prop += 1
    if proposal is not None:
        loghr, undo = proposal
        prop = state.log_posterior()
        if not math.isfinite(prop):
            undo()
        elif math.log(kernel.rng.random()) < (prop - cur) + loghr:
            cur = prop
            if is_add:
                n_add_acc += 1
                if state.num_reticulations() >= 1 and first_add is None:
                    first_add = it
        else:
            undo()
    if it % 2000 == 0:
        print(f"it={it:6d} logP={cur:10.2f} retic={state.num_reticulations()}")
dt = time.perf_counter() - t0
print(f"\nfirst reticulation ADD accepted at iter: {first_add}")
print(f"add proposed(valid)={n_add_prop} accepted={n_add_acc}")
print(f"final retic={state.num_reticulations()} logP={cur:.2f}  ({1000*dt/NUM:.2f} ms/it)")
