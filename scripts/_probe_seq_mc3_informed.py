"""MC3 + informed add/delete: can heated chains cross the reticulation barrier?"""
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
res = s.search(num_iter=20000, burn_in=6000, sample_freq=50, seed=3,
               temperatures=[1.0, 2.0, 4.0, 8.0], swap_interval=2, progress=True)
dt = time.perf_counter() - t0
frac = sum(1 for x in res.samples if x.num_reticulations >= 1) / max(1, len(res.samples))
print(f"MC3-informed: {dt:.1f}s ({1000*dt/20000:.2f} ms/it) MAP logP={res.map_log_posterior:.2f} "
      f"retic(MAP)={num_retic(res.map_network)} frac_retic={frac:.2f} "
      f"retic_post={res.reticulation_posterior()}")
