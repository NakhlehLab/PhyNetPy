r"""Isolate the coupled add/delete-reticulation moves (prior-only).

The prior-only chain froze in whichever dimension it started in (neither add
nor delete ever accepted).  This probe calls the two dimension operators
*directly* on a fixed state, many times, and classifies each call:

    * raised   -- the operator threw (the kernel would silently swallow this)
    * declined -- returned ``None`` (illegal / non-reversible corner)
    * scored   -- returned ``(loghr, undo)``; we then record the prior-only
                  log acceptance  log_alpha = loghr + (log_prior_new - log_prior_old)
                  and the implied accept probability min(1, e^{log_alpha}).

This pinpoints whether the freeze is (a) the move never firing, (b) declining,
or (c) firing but always rejected (a Hastings-ratio problem).
"""
from __future__ import annotations

import os
import sys
import math
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import phynetpy._mcmc_seq as seqmod
from phynetpy._mcmc_seq import (
    MCMC_SEQ, MCMCSeqPriors, _clone_net,
    op_add_reticulation_coupled, op_delete_reticulation_coupled,
)
from phynetpy.infer import JC69, simulate_multilocus

from _prior_only_validation import _tiny_true_net, MAPPING


def _probe(op_name, op, make_state, rng, n=400):
    raised = declined = scored = 0
    log_alphas = []
    errs = {}
    for _ in range(n):
        state = make_state()
        p_before = state.log_prior()
        try:
            res = op(state, rng)
        except Exception as e:  # noqa
            raised += 1
            key = f"{type(e).__name__}: {e}"
            errs[key] = errs.get(key, 0) + 1
            continue
        if res is None:
            declined += 1
            continue
        scored += 1
        loghr, undo = res
        p_after = state.log_prior()
        # prior-only: likelihood contributes 0 on both sides
        log_alpha = loghr + (p_after - p_before)
        log_alphas.append(log_alpha)
        undo()
    print(f"\n--- {op_name}  (n={n}) ---")
    print(f"  raised   : {raised}")
    if errs:
        for k, v in sorted(errs.items(), key=lambda kv: -kv[1])[:5]:
            print(f"      {v:4d}x  {k}")
    print(f"  declined : {declined}")
    print(f"  scored   : {scored}")
    if log_alphas:
        a = np.array(log_alphas)
        acc = np.minimum(1.0, np.exp(np.clip(a, -700, 0)))
        print(f"  log_alpha: min={a.min():+.3f} med={np.median(a):+.3f} "
              f"max={a.max():+.3f}")
        print(f"  mean accept prob = {acc.mean():.4f}  "
              f"(expected #accept/{scored} -> {acc.sum():.1f})")
    return {"raised": raised, "declined": declined, "scored": scored,
            "log_alphas": log_alphas}


def main() -> None:
    seqmod.SeqState.log_likelihood = lambda self: 0.0  # prior-only

    data = simulate_multilocus(_tiny_true_net(), MAPPING, n_loci=4,
                               seq_length=200, theta=0.02, model=JC69(), seed=1)
    rng = np.random.default_rng(0)

    priors = MCMCSeqPriors(max_reticulations=1, max_level=1,
                           use_diameter_prior=False,
                           gamma_alpha=1.0, gamma_beta=1.0, poisson_mean=1.0)

    # r=0 state factory (plain species tree)
    tree_sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=priors)
    print(f"tree start: reticulations = "
          f"{tree_sampler._new_state().num_reticulations()}")
    _probe("op_add_reticulation_coupled (from r=0)",
           op_add_reticulation_coupled,
           tree_sampler._new_state, rng)

    # r=1 state factory (true network)
    net_sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=priors)
    net_sampler.species_net = _tiny_true_net()
    print(f"\nnet start: reticulations = "
          f"{net_sampler._new_state().num_reticulations()}")
    _probe("op_delete_reticulation_coupled (from r=1)",
           op_delete_reticulation_coupled,
           net_sampler._new_state, rng)
    # also try add from r=1 should decline (cap) and delete from r=0 should decline
    _probe("op_add_reticulation_coupled (from r=1, expect all declined@cap)",
           op_add_reticulation_coupled,
           net_sampler._new_state, rng, n=50)
    _probe("op_delete_reticulation_coupled (from r=0, expect all declined)",
           op_delete_reticulation_coupled,
           tree_sampler._new_state, rng, n=50)


if __name__ == "__main__":
    main()
