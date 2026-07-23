"""Isolation probe: does the gamma (inheritance-probability) move mix?

We pin the state at the *true* network with the *true* gene trees (so the
reticulation is real and correctly placed, gamma_true = major inheritance) and
then run ONLY ``op_change_gamma`` in an MH loop, at two data scales:

    A) weekend scale : 10 taxa, 50 loci x 1000 sites   (gamma froze in the run)
    B) old scale     :  6 taxa, 15 loci x  600 sites   (gamma was "fine")

For each we report the gamma-move acceptance rate and the gamma trace's spread /
ESS.  This separates two hypotheses:

  * If gamma mixes fine on the TRUE network at weekend scale, the freeze we saw
    is caused by the degenerate (bubble) reticulation the chain wandered into --
    a topology / identifiability problem, not the gamma move.
  * If gamma freezes even on the TRUE network at weekend scale, the move itself
    (window, or the sharp 50k-site likelihood) is the culprit.

Parallelism is irrelevant here (single process), which also demonstrates the
4-chains-at-once launch cannot be the cause.

    py scripts/_probe_gamma_freeze.py
"""
from __future__ import annotations

import os
import sys
import math
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from phynetpy.infer import JC69, simulate_multilocus
from phynetpy._mcmc_seq import (
    SeqState, MCMCSeqPriors, op_change_gamma, _major_gamma, _clone_net,
)
from phynetpy._chain_analysis import effective_sample_size

import _run_weekend as wk          # 10-taxon truth
import mcmc_harness as hz          # 6-taxon truth


def _gamma_only_run(true_net, mapping, *, loci, sites, iters, max_level,
                    seed, label):
    data = simulate_multilocus(true_net, mapping, n_loci=loci,
                               seq_length=sites, theta=0.02, model=JC69(),
                               seed=seed)
    state = SeqState(
        species_net=_clone_net(true_net),
        gene_trees=[_clone_net(gt) for gt in data.gene_trees],
        species_of=data.species_of,
        loci=data.loci,
        priors=MCMCSeqPriors(max_reticulations=2, max_level=max_level),
        model=data.model,
        theta=data.true_theta,
    )
    rng = np.random.default_rng(seed + 777)
    cur = state.log_posterior()
    g0 = _major_gamma(state.species_net)
    g_true = _major_gamma(true_net)

    proposed = accepted = none_props = 0
    gammas = [g0]
    t0 = time.perf_counter()
    for _ in range(iters):
        prop = op_change_gamma(state, rng)
        if prop is None:
            none_props += 1
            gammas.append(_major_gamma(state.species_net))
            continue
        proposed += 1
        loghr, undo = prop
        p = state.log_posterior()
        if not math.isfinite(p):
            undo()
        else:
            if math.log(rng.random()) < (p - cur) + loghr:
                cur = p
                accepted += 1
            else:
                undo()
        gammas.append(_major_gamma(state.species_net))
    dt = time.perf_counter() - t0

    arr = np.asarray(gammas, dtype=float)
    ess = effective_sample_size(list(arr))
    acc_rate = accepted / proposed if proposed else float("nan")

    print(f"\n=== {label} : {loci} loci x {sites} sites, "
          f"max_level={max_level} ===")
    print(f"  gamma_true (major)     : {g_true:.4f}")
    print(f"  start gamma_major      : {g0:.4f}")
    print(f"  gamma-move proposals   : {proposed}  (inapplicable: {none_props})")
    print(f"  gamma-move acceptance  : {acc_rate:.4f}  "
          f"({accepted}/{proposed})")
    print(f"  gamma trace  min/mean/max = "
          f"{arr.min():.4f} / {arr.mean():.4f} / {arr.max():.4f}")
    print(f"  gamma trace  std       : {arr.std():.5f}")
    print(f"  gamma trace  ESS       : {ess:.1f}  (of {len(arr)} states)")
    print(f"  distinct gamma values  : {len(np.unique(np.round(arr, 6)))}")
    print(f"  ({iters} steps in {dt:.1f}s, {1000*dt/iters:.2f} ms/step)")
    return acc_rate, ess, arr


def main() -> None:
    iters = int(os.environ.get("PROBE_ITERS", "8000"))

    # A) weekend scale, 10 taxa
    _gamma_only_run(
        wk.build_true_network(), wk.MAPPING,
        loci=50, sites=1000, iters=iters, max_level=1,
        seed=20260710, label="A weekend (wk 10taxa)",
    )
    # A') weekend topology but with the level cap removed, same data
    _gamma_only_run(
        wk.build_true_network(), wk.MAPPING,
        loci=50, sites=1000, iters=iters, max_level=None,
        seed=20260710, label="A' weekend (wk 10taxa, no level cap)",
    )
    # B) old scale, 6 taxa
    _gamma_only_run(
        hz.build_true_network(), hz.MAPPING,
        loci=15, sites=600, iters=iters, max_level=None,
        seed=12345, label="B old (hz 6taxa)",
    )


if __name__ == "__main__":
    main()
