"""Where does the informed add lose acceptance: selection, placement, or HR?

From a burned-in tree state, call the informed add many times and record:
  * best midpoint score delta over placements (what selection 'sees'),
  * the actual proposed network delta (prop - cur),
  * loghr,
  * log_alpha = (prop - cur) + loghr.
Compares the *idealised* acceptance (Sum exp(midpoint delta)/(2(R+1))) with the
*actual* one, to quantify the midpoint-vs-random-placement gap.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy import _mcmc_seq as M

true_net = build_true_network()
data = simulate_multilocus(true_net, MAPPING, n_loci=30, seq_length=600,
                           theta=0.02, model=JC69(), seed=7)
sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=MCMCSeqPriors(max_reticulations=2))
rng = np.random.default_rng(1)
kernel = M.MCMCSeqKernel(rng, max_reticulations=2)
kernel._ops = [(op, w) for op, w in kernel._ops
               if op.__name__ not in ("op_add_reticulation", "op_delete_reticulation")]
kernel._weights = np.asarray([w for _, w in kernel._ops]); kernel._weights /= kernel._weights.sum()
state = sampler._new_state()
cur = state.log_posterior()
for _ in range(5000):
    p = kernel.propose(state)
    if p is not None:
        loghr, undo = p
        prop = state.log_posterior()
        if math.isfinite(prop) and math.log(rng.random()) < (prop - cur) + loghr:
            cur = prop
        else:
            undo()
print(f"tree burned in: logP={cur:.2f}")

# Idealised acceptance from midpoint scores:
placements = M._add_reticulation_placements(state, state.species_net, state.net_heights)
s_reps = [p["s_rep"] for p in placements]
lse = M._logsumexp(s_reps)
s_cur = M._network_only_log_score(state, state.species_net)
ideal_log_alpha = M._logsumexp([s - s_cur for s in s_reps]) - math.log(2.0 * 1)
print(f"placements={len(placements)}  best midpoint delta={max(s_reps)-s_cur:.2f}  "
      f"idealised log_alpha(add)={ideal_log_alpha:.2f} -> accept~{math.exp(min(0,ideal_log_alpha)):.3f}")

# Actual informed-add attempts:
las = []
props = []
for _ in range(400):
    res = M.op_add_reticulation(state, rng)
    if res is None:
        continue
    loghr, undo = res
    prop = state.log_posterior()
    undo()
    if not math.isfinite(prop):
        continue
    la = (prop - cur) + loghr
    las.append(la); props.append(prop - cur)
if las:
    las.sort(); props.sort()
    print(f"actual attempts={len(las)}  proposed delta: min={props[0]:.1f} "
          f"med={props[len(props)//2]:.1f} max={props[-1]:.1f}")
    print(f"actual log_alpha: min={las[0]:.1f} med={las[len(las)//2]:.1f} max={las[-1]:.1f}  "
          f"n(>=0)={sum(1 for x in las if x>=0)}/{len(las)}")
    print(f"mean actual accept prob = {sum(math.exp(min(0,x)) for x in las)/len(las):.4f}")
