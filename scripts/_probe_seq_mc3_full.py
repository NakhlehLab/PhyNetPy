"""MC3 at the user's real 50k budget: does the cold chain find the reticulation?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus

true_net = build_true_network()
data = simulate_multilocus(true_net, MAPPING, n_loci=30, seq_length=400,
                           theta=0.02, model=JC69(), seed=7)

def num_retic(net):
    return sum(1 for v in net.V() if v.is_reticulation())

s = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
             priors=MCMCSeqPriors(max_reticulations=2))
t0 = time.perf_counter()
res = s.search(num_iter=50000, burn_in=15000, sample_freq=50, seed=3,
               temperatures=[1.0, 1.5, 2.25, 3.4, 5.0], swap_interval=3)
dt = time.perf_counter() - t0
frac = sum(1 for x in res.samples if x.num_reticulations >= 1) / max(1, len(res.samples))
print(f"MC3-50k: {dt:.1f}s  MAP logP={res.map_log_posterior:.2f} "
      f"retic(MAP)={num_retic(res.map_network)} frac_retic={frac:.2f} "
      f"retic_post={res.reticulation_posterior()}")
