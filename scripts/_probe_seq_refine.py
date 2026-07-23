"""SEQ refinement: does the informed chain correct an over-fit start?

Two starts on the same data:
  (1) TRUE network  -> should HOLD retic=1 and estimate gamma ~ 0.65.
  (2) an over-fit 2-reticulation network (extra spurious hybrid) -> the
      now-working informed delete should drop the spurious reticulation back
      to 1 and recover gamma.
This validates the delete direction and gamma estimation independent of the
(separately hard) cold-start discovery problem.
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.mcmc_harness import build_true_network, MAPPING, TRUE_GAMMA_MAJOR
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy import _mcmc_seq as M

true_net = build_true_network()
data = simulate_multilocus(true_net, MAPPING, n_loci=30, seq_length=500,
                           theta=0.02, model=JC69(), seed=7)

def num_retic(net):
    return sum(1 for v in net.V() if v.is_reticulation())

def major_gamma(net):
    gs = [float(e.get_gamma()) for v in net.V() if v.is_reticulation()
          for e in net.in_edges(v) if e.get_gamma() is not None]
    return max(gs) if gs else None

# (1) start = true
kw = data.to_mcmc_seq_kwargs(); kw["species_net"] = copy.deepcopy(true_net)
s1 = MCMC_SEQ(**kw, priors=MCMCSeqPriors(max_reticulations=2))
r1 = s1.search(num_iter=20000, burn_in=6000, sample_freq=50, seed=2)
print(f"(1) start=TRUE : retic(MAP)={num_retic(r1.map_network)} gamma={major_gamma(r1.map_network)} "
      f"retic_post={r1.reticulation_posterior()}")

# (2) start = an over-fit 2-reticulation net: add a spurious reticulation to truth
over = copy.deepcopy(true_net)
s2 = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=MCMCSeqPriors(max_reticulations=2))
s2.species_net = over
# add one spurious reticulation with the informed move on a fresh state
tmp = s2._new_state()
rng = np.random.default_rng(0)
for _ in range(200):
    res = M.op_add_reticulation(tmp, rng)
    if res is not None and tmp.num_reticulations() == 2:
        break
s2.species_net = copy.deepcopy(tmp.species_net)
print(f"    (constructed start retic={num_retic(s2.species_net)})")
r2 = s2.search(num_iter=20000, burn_in=6000, sample_freq=50, seed=2)
print(f"(2) start=2retic: retic(MAP)={num_retic(r2.map_network)} gamma={major_gamma(r2.map_network)} "
      f"retic_post={r2.reticulation_posterior()}")
print(f"true gamma_major = {TRUE_GAMMA_MAJOR}")
