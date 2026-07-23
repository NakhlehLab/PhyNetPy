"""Does MC3 (Metropolis coupling) discover the reticulation from a tree start?

The single cold chain cannot cross the add-reticulation barrier (best single
add has log-accept ~ -6.8).  A heated chain at temperature T sees that barrier
divided by T, so it should add the reticulation, optimise gamma/heights, and
swap the improved state down to the cold chain.  This runs the built-in MC3 on
a tree start and reports whether the cold chain ends up with the reticulation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus

true_net = build_true_network()
data = simulate_multilocus(true_net, MAPPING, n_loci=30, seq_length=400,
                           theta=0.02, model=JC69(), seed=7)

def num_retic(net):
    return sum(1 for v in net.V() if v.is_reticulation())

for ladder in [[1.0, 1.6, 2.5, 4.0]]:
    s = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                 priors=MCMCSeqPriors(max_reticulations=2))
    res = s.search(num_iter=20000, burn_in=5000, sample_freq=50, seed=3,
                   temperatures=ladder, swap_interval=5)
    frac = sum(1 for x in res.samples if x.num_reticulations >= 1) / max(1, len(res.samples))
    print(f"ladder={ladder} MAP logP={res.map_log_posterior:.2f} "
          f"retic(MAP)={num_retic(res.map_network)} frac_retic={frac:.2f} "
          f"retic_post={res.reticulation_posterior()}")
