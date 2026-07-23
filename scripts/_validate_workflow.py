r"""Validate the intended MCMC_SEQ workflow after the grid + decoupled fixes.

Two arms on data with verified reticulation signal (#3):

  A. start-at-truth stability -- seed the chain at the true r=1 network; a
     correct sampler must HOLD r=1 and keep gamma near 0.70 (before the grid fix
     the coupled add's mate could spuriously collapse it to a tree it could
     never leave).

  B. warm-start discovery -- the real workflow: bootstrap a reticulation with
     MCMC_GT, then refine.  Should end at r=1 with gamma ~ 0.70.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from phynetpy.infer import JC69, simulate_multilocus
from phynetpy._mcmc_seq import MCMC_SEQ, MCMCSeqPriors, _major_gamma, _clone_net
from _run_weekend import build_true_network, MAPPING

N_LOCI, SITES, ITERS, BURN, SEED = 15, 500, 40_000, 8_000, 11


def summarize(tag, res, t):
    rs = [s.num_reticulations for s in res.samples]
    frac1 = sum(1 for r in rs if r >= 1) / max(1, len(rs))
    map_r = sum(1 for v in res.map_network.V() if v.is_reticulation())
    print(f"[{tag}] {t:.0f}s  MAP r={map_r}  P(r>=1)={frac1:.3f}  "
          f"MAP gamma={_major_gamma(res.map_network)}  "
          f"acc={res.acceptance_rate:.3f}")


def main():
    true_net = build_true_network()
    data = simulate_multilocus(true_net, MAPPING, n_loci=N_LOCI, seq_length=SITES,
                               theta=0.02, model=JC69(), seed=2024)
    print(f"{N_LOCI} loci x {SITES} sites, {ITERS} iters, decoupled kernel, "
          f"max_ret=1 (true gamma=0.70)\n")

    # Arm A: start at truth
    s = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                 priors=MCMCSeqPriors(max_reticulations=1, max_level=1))
    s.species_net = _clone_net(true_net)
    t0 = time.time()
    resA = s.search(num_iter=ITERS, burn_in=BURN, sample_freq=200, seed=SEED,
                    warm_start=False)
    summarize("A: start-at-truth", resA, time.time() - t0)

    # Arm B: warm-start discovery
    s2 = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                  priors=MCMCSeqPriors(max_reticulations=1, max_level=1))
    t0 = time.time()
    resB = s2.search(num_iter=ITERS, burn_in=BURN, sample_freq=200, seed=SEED,
                     warm_start=True, warm_start_kwargs={"gt_iters": 4000})
    summarize("B: warm-start    ", resB, time.time() - t0)


if __name__ == "__main__":
    main()
