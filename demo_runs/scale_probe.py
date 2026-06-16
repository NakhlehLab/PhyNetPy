"""Throwaway scaling probe: measure MCMC_SEQ per-iteration cost at 20 taxa.

Not a recovery test -- only times the chain and checks it runs without
crashing, so we can project the cost of a 4-chain x 1e6-iter analysis.
"""
import sys
import time

import numpy as np

from phynetpy.BirthDeath import CBDP
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus

N_TAXA = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_LOCI = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SITES = int(sys.argv[3]) if len(sys.argv) > 3 else 500
PROBE_ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
MAX_RETIC = int(sys.argv[5]) if len(sys.argv) > 5 else 4

np.random.seed(0)
net = CBDP(1.0, 0.5, N_TAXA).generate_network()
leaves = sorted(n.label for n in net.get_leaves())
print(f"taxa={len(leaves)} loci={N_LOCI} sites={SITES} "
      f"max_retic={MAX_RETIC} probe_iters={PROBE_ITERS}", flush=True)

mapping = {l: [l] for l in leaves}
t_sim = time.time()
data = simulate_multilocus(
    net, mapping, n_loci=N_LOCI, seq_length=SITES,
    theta=0.02, model=JC69(), seed=7,
)
print(f"simulate_multilocus: {time.time() - t_sim:.2f}s", flush=True)

sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                   priors=MCMCSeqPriors(max_reticulations=MAX_RETIC))
print(f"start logP: {sampler.score():.2f}", flush=True)

t0 = time.time()
# burn_in == num_iter so nothing is stored; we only want raw throughput.
sampler.search(num_iter=PROBE_ITERS, burn_in=PROBE_ITERS,
               sample_freq=10, seed=1, progress=True)
dt = time.time() - t0
ms_per_iter = 1000.0 * dt / PROBE_ITERS
print(f"\nelapsed {dt:.2f}s for {PROBE_ITERS} iters -> {ms_per_iter:.3f} ms/iter")
hours_4x1m = 4 * 1_000_000 * (dt / PROBE_ITERS) / 3600.0
print(f"projected 4 chains x 1,000,000 iters (serial): {hours_4x1m:.2f} hours")
print(f"projected (4 chains in parallel):              {hours_4x1m / 4:.2f} hours")
