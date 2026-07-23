r"""Head-to-head: coupled vs decoupled dimension moves on REAL data.

From a plain species-tree start (NO warm start), on data with strong, verified
reticulation signal (#3), does the chain discover the reticulation?  The coupled
all-at-once re-proposal induces trans-dimensional hysteresis (chain stuck at
r=0); the decoupled network-only move lets the gene trees migrate gradually and
should climb to r=1 and recover gamma.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from phynetpy.infer import JC69, simulate_multilocus
from phynetpy._mcmc_seq import MCMC_SEQ, MCMCSeqPriors, MCMCSeqKernel, _major_gamma
from _run_weekend import build_true_network, MAPPING

N_LOCI = 15
SITES = 500
ITERS = 60_000
BURN = 10_000
SEED = 7


def run(coupled: bool, data):
    priors = MCMCSeqPriors(max_reticulations=1, max_level=1)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=priors)
    rng = np.random.default_rng(SEED)
    kernel = MCMCSeqKernel(rng, max_reticulations=1,
                           coupled_dimension_moves=coupled)

    traj = []

    def control(prog):
        if prog["iteration"] % 4000 == 0:
            traj.append((prog["iteration"], prog["num_reticulations"]))
        return "continue"

    t0 = time.time()
    res = sampler.search(num_iter=ITERS, burn_in=BURN, sample_freq=500,
                         seed=SEED, kernel=kernel, control=control,
                         check_every=1000, warm_start=False)
    dt = time.time() - t0
    # posterior reticulation-count occupancy
    rs = [s.num_reticulations for s in res.samples]
    frac1 = sum(1 for r in rs if r >= 1) / max(1, len(rs))
    gam = _major_gamma(res.map_network)
    label = "COUPLED  " if coupled else "DECOUPLED"
    print(f"\n[{label}] {dt:.0f}s  MAP r={sum(1 for v in res.map_network.V() if v.is_reticulation())}  "
          f"P(r>=1)={frac1:.3f}  MAP gamma={gam}  acc={res.acceptance_rate:.3f}")
    print(f"  r-trajectory: {' '.join(str(r) for _, r in traj)}")
    return frac1, gam


def main():
    true_net = build_true_network()
    data = simulate_multilocus(true_net, MAPPING, n_loci=N_LOCI,
                               seq_length=SITES, theta=0.02, model=JC69(),
                               seed=2024)
    print(f"Head-to-head: {N_LOCI} loci x {SITES} sites, {ITERS} iters, "
          f"plain-tree start, max_ret=1\n"
          f"(true gamma_major=0.70; ~30% loci carry the minor topology)")
    run(coupled=False, data=data)
    run(coupled=True, data=data)


if __name__ == "__main__":
    main()
