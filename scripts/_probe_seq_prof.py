"""Profile MCMC_SEQ to locate the per-iteration bottleneck."""
import os, sys, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus

true_net = build_true_network()
data = simulate_multilocus(true_net, MAPPING, n_loci=8, seq_length=300,
                           theta=0.02, model=JC69(), seed=12345)
sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                   priors=MCMCSeqPriors(max_reticulations=2))

pr = cProfile.Profile()
pr.enable()
res = sampler.search(num_iter=2000, burn_in=200, sample_freq=20, seed=12345)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue())
print("map_log_posterior:", res.map_log_posterior)
