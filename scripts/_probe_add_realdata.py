r"""Why is the (decoupled) add-reticulation rejected on the real starting state?

~30% of the starting UPGMA gene trees already carry the minor topology (#3), so
a reticulation that explains them *should* raise the MSNC likelihood.  Decompose
the add's log acceptance into:

    log_alpha = loghr            (informed-placement geometry / volume term)
              + d_loglik         (Felsenstein + MSNC gain from the new network)
              + d_logprior       (Poisson + topology-normaliser + guards)

to see which term blocks the move.
"""
from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from phynetpy.infer import JC69, simulate_multilocus
from phynetpy._mcmc_seq import (
    MCMC_SEQ, MCMCSeqPriors,
    op_add_reticulation_decoupled, op_add_reticulation_coupled,
)
from _run_weekend import build_true_network, MAPPING


def probe(op, name, sampler, n=300):
    rng = np.random.default_rng(0)
    rows = []
    declined = 0
    for _ in range(n):
        state = sampler._new_state()
        ll0 = state.log_likelihood()
        lp0 = state.log_prior()
        res = op(state, rng)
        if res is None:
            declined += 1
            continue
        loghr, undo = res
        ll1 = state.log_likelihood()
        lp1 = state.log_prior()
        dll = ll1 - ll0
        dlp = lp1 - lp0
        la = loghr + dll + dlp
        rows.append((loghr, dll, dlp, la))
        undo()
    print(f"\n--- {name}  (n={n}, declined={declined}) ---")
    if not rows:
        print("  no scored proposals")
        return
    a = np.array(rows)
    for j, nm in enumerate(["loghr(volume)", "d_loglik", "d_logprior", "log_alpha"]):
        col = a[:, j]
        col = col[np.isfinite(col)]
        if len(col):
            print(f"  {nm:14s}: min={col.min():+9.3f} med={np.median(col):+9.3f} "
                  f"max={col.max():+9.3f}")
    la = a[:, 3]
    la = la[np.isfinite(la)]
    n_acc = int((la > 0).sum())
    n_maybe = int((la > -5).sum())
    print(f"  would-accept (log_alpha>0): {n_acc}/{len(la)}   "
          f"log_alpha>-5: {n_maybe}/{len(la)}")


def main():
    true_net = build_true_network()
    data = simulate_multilocus(true_net, MAPPING, n_loci=15, seq_length=500,
                               theta=0.02, model=JC69(), seed=2024)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=1, max_level=1))
    print(f"start reticulations = {sampler._new_state().num_reticulations()}, "
          f"theta = {sampler.theta}")
    probe(op_add_reticulation_decoupled, "decoupled add (real data)", sampler)
    probe(op_add_reticulation_coupled, "coupled add (real data)", sampler)


if __name__ == "__main__":
    main()
