"""De-risk the identifiability-guard changes before committing to the full
4 x 1.5M weekend run.

Two checks at the *actual* weekend data scale (10 taxa, 50 loci x 1000 sites,
shared data seed):

  1. WARM-START SEEDS: for each of the four chain seeds, run warm_start and
     report whether it seeded a *valid* reticulation (finite prior, cycle >= 4)
     or fell back to a plain tree.  This tells us if reticulation discovery is
     the bottleneck before wasting days of compute.

  2. SHORT CHAIN: run one short chain with frequent milestones and confirm
     (a) no degenerate "bubble" (cycle < 4) is ever sampled, and
     (b) gamma and the reticulation count are actually moving (non-trivial ESS
         / spread), i.e. the freeze is gone.

Exit status "CLEAN" is printed iff every warm start scores finitely and the
short chain samples no bubbles.
"""
from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy._mcmc_seq import (
    MCMCSeqResult, log_prior_seq, _reticulation_cycle_size, _major_gamma,
)
from phynetpy._chain_analysis import effective_sample_size
import _run_weekend as wk

# Leaner than the full weekend run so the de-risk finishes in a reasonable
# window while still exercising the *actual* 10-taxon / weekend data scale.
DATA_SEED = 20260710
CHAIN_SEEDS = [1001, 2002]
LOCI, SITES = 50, 1000
GT_ITERS = 5000
CHAIN_ITERS = 5000


def _net_diag(net):
    rets = [v for v in net.V() if v.is_reticulation()]
    sizes = [_reticulation_cycle_size(net, r) for r in rets]
    return len(rets), sizes


def main() -> None:
    true_net = wk.build_true_network()
    print(f"True net: {true_net.newick()}")
    print(f"Simulating {LOCI} loci x {SITES} sites (data_seed={DATA_SEED})...")
    data = simulate_multilocus(true_net, wk.MAPPING, n_loci=LOCI,
                               seq_length=SITES, theta=0.02, model=JC69(),
                               seed=DATA_SEED)

    print("\n=== 1. WARM-START SEED CHECK (weekend settings) ===")
    all_finite = True
    for cs in CHAIN_SEEDS:
        sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                           priors=MCMCSeqPriors(max_reticulations=2,
                                                max_level=1))
        sampler.warm_start(gt_iters=GT_ITERS, max_reticulations=2,
                           max_level=1, seed=cs)
        seed_net = sampler.species_net
        lp = log_prior_seq(seed_net, sampler.theta, sampler.priors)
        nret, sizes = _net_diag(seed_net)
        finite = math.isfinite(lp)
        all_finite &= finite
        kind = ("TREE (fell back)" if nret == 0
                else f"{nret}-retic, cycle sizes {sizes}")
        print(f"  seed {cs}: {kind:<34} prior="
              f"{'finite' if finite else '-INF'}")

    print(f"\n=== 2. SHORT CHAIN (seed 1001, {CHAIN_ITERS} iters) ===")
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=2, max_level=1))

    bubble_hits = {"n": 0}

    def control(prog):
        it = prog["iteration"]
        if it > 0 and it % 2000 == 0:
            samples = prog["samples"]
            gammas = [s.gamma_major for s in samples
                      if s.gamma_major is not None]
            rets = [s.num_reticulations for s in samples]
            gspread = (f"{min(gammas):.3f}-{max(gammas):.3f}"
                       if gammas else "n/a")
            rpost = {r: round(rets.count(r) / len(rets), 2)
                     for r in sorted(set(rets))} if rets else {}
            print(f"  it {it:>6}: acc={prog['acceptance_rate']:.3f}  "
                  f"n_samp={len(samples)}  retic_post={rpost}  "
                  f"gamma_range={gspread}")
        return "continue"

    res = sampler.search(num_iter=CHAIN_ITERS, burn_in=1000, sample_freq=50,
                         seed=1001, warm_start=True,
                         warm_start_kwargs={"gt_iters": GT_ITERS},
                         control=control, check_every=2000)

    gammas = [s.gamma_major for s in res.samples if s.gamma_major is not None]
    rets = [s.num_reticulations for s in res.samples]
    ess_g = effective_sample_size(gammas) if len(gammas) > 2 else float("nan")

    print("\n=== SHORT-CHAIN SUMMARY ===")
    print(f"  samples: {len(res.samples)}   acc={res.acceptance_rate:.3f}")
    print(f"  reticulation posterior: {res.reticulation_posterior()}")
    if gammas:
        print(f"  gamma_major: range {min(gammas):.3f}-{max(gammas):.3f}, "
              f"mean {np.mean(gammas):.3f}, ESS {ess_g:.1f} "
              f"(of {len(gammas)})")
    else:
        print("  gamma_major: none sampled (chain sat at r=0)")
    print(f"  MAP net: {res.map_network.newick()}")
    map_nret, map_sizes = _net_diag(res.map_network)
    map_bubble = any(s < 4 for s in map_sizes)
    print(f"  MAP reticulations={map_nret}, cycle sizes={map_sizes}, "
          f"bubble={map_bubble}")

    clean = all_finite and not map_bubble
    print("\n" + ("=" * 50))
    print("RESULT: " + ("CLEAN -- safe to launch the full weekend run"
                        if clean else
                        "NOT CLEAN -- investigate before the full run"))
    print(f"  (all warm starts finite: {all_finite}; MAP bubble: {map_bubble})")


if __name__ == "__main__":
    main()
