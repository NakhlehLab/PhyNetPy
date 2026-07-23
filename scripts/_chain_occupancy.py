r"""Decisive over-/under-fitting test for the branch-length prior.

Run a real MCMC_SEQ chain and report the post-burn-in reticulation-count
occupancy.  Two datasets from the SAME 10-taxon topology:

  * TREE data  (simulated from the major displayed tree, r=0 truth)
        -> occupancy should concentrate on r=0 (no spurious over-adding).
  * RETIC data (simulated from the true r=1 network)
        -> occupancy should put real mass on r=1 (reticulation discovered).

This is the honest test the single-shot probes can't give: detailed balance,
not one proposal.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from phynetpy.infer import JC69, simulate_multilocus
from phynetpy._mcmc_seq import MCMC_SEQ, MCMCSeqPriors, _clone_net
from _run_weekend import build_true_network, MAPPING
from _probe_add_treedata import _tree_from_true

ITERS, BURN = 40_000, 20_000


def run(species_net_truth, tag, seed_data, seed_chain):
    data = simulate_multilocus(species_net_truth, MAPPING, n_loci=15,
                               seq_length=500, theta=0.02, model=JC69(),
                               seed=seed_data)
    priors = MCMCSeqPriors(max_reticulations=1, max_level=1)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=priors)
    # cold start from the plain tree so occupancy reflects the sampler, not seed
    sampler.species_net = _tree_from_true(build_true_network())

    res = sampler.search(num_iter=ITERS, burn_in=BURN, sample_freq=20,
                         seed=seed_chain)
    counts = Counter(s.num_reticulations for s in res.samples)
    tot = max(1, sum(counts.values()))
    print(f"\n--- {tag} ---")
    for r in sorted(counts):
        print(f"  r={r}: {counts[r]/tot:6.1%}  ({counts[r]})")
    map_r = sum(1 for v in sampler.species_net.V() if v.is_reticulation())
    print(f"  MAP reticulations: {map_r}")


def main():
    true_net = build_true_network()
    tree = _tree_from_true(true_net)
    run(tree, "TREE data (r=0 truth) -> want mass on r=0", 7001, 101)
    run(true_net, "RETIC data (r=1 truth) -> want mass on r=1", 2024, 202)


if __name__ == "__main__":
    main()
