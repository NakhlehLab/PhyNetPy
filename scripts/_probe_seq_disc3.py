"""Discovery from tree start with informed moves, at higher data levels.

If the informed move + adequate data recovers the reticulation from a plain
tree start, the move is doing its job (the residual barrier is statistical
power / co-adaptation, not a sampler bug).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus

true_net = build_true_network()

def num_retic(net):
    return sum(1 for v in net.V() if v.is_reticulation())

for n_loci, iters in [(50, 30000)]:
    data = simulate_multilocus(true_net, MAPPING, n_loci=n_loci, seq_length=400,
                               theta=0.02, model=JC69(), seed=7)
    s = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                 priors=MCMCSeqPriors(max_reticulations=2))
    t0 = time.perf_counter()
    res = s.search(num_iter=iters, burn_in=iters // 3, sample_freq=50, seed=5,
                   progress=True)
    dt = time.perf_counter() - t0
    frac = sum(1 for x in res.samples if x.num_reticulations >= 1) / max(1, len(res.samples))
    print(f"loci={n_loci} iters={iters}: {dt:.1f}s ({1000*dt/iters:.2f} ms/it) "
          f"MAP logP={res.map_log_posterior:.2f} retic(MAP)={num_retic(res.map_network)} "
          f"frac_retic={frac:.2f} retic_post={res.reticulation_posterior()}")
