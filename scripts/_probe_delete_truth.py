r"""Why does a chain started at the TRUE r=1 network collapse to r=0?

Decompose the decoupled delete's log acceptance at the true network under two
gene-tree states:
  * UPGMA gene trees (what the chain starts arm A with), and
  * the TRUE simulated gene trees (the oracle).
If delete is favored (log_alpha>0) under UPGMA but not under the true trees, the
collapse is a gene-tree-state problem (the starting trees don't yet 'use' the
reticulation), not a kernel bug.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from phynetpy.infer import JC69, simulate_multilocus
from phynetpy._mcmc_seq import (
    MCMC_SEQ, MCMCSeqPriors, SeqState,
    op_delete_reticulation_decoupled, _clone_net, _heights, _sync_lengths,
)
from _run_weekend import build_true_network, MAPPING


def probe(make_state, tag, n=200):
    rng = np.random.default_rng(0)
    rows, declined = [], 0
    for _ in range(n):
        st = make_state()
        ll0, lp0 = st.log_likelihood(), st.log_prior()
        res = op_delete_reticulation_decoupled(st, rng)
        if res is None:
            declined += 1
            continue
        loghr, undo = res
        la = loghr + (st.log_likelihood() - ll0) + (st.log_prior() - lp0)
        rows.append((loghr, la))
        undo()
    print(f"\n--- delete from truth: {tag} (n={n}, declined={declined}) ---")
    if rows:
        a = np.array(rows)
        la = a[:, 1][np.isfinite(a[:, 1])]
        print(f"  loghr    med={np.median(a[:,0]):+.3f}")
        print(f"  log_alpha min={la.min():+.3f} med={np.median(la):+.3f} "
              f"max={la.max():+.3f}")
        print(f"  delete would-accept (log_alpha>0): {int((la>0).sum())}/{len(la)}")
        print(f"  => {'DELETE FAVORED (collapses r=1->r=0)' if np.median(la)>0 else 'delete disfavored (holds r=1)'}")


def main():
    true_net = build_true_network()
    data = simulate_multilocus(true_net, MAPPING, n_loci=15, seq_length=500,
                               theta=0.02, model=JC69(), seed=2024)
    priors = MCMCSeqPriors(max_reticulations=1, max_level=1)
    kw = data.to_mcmc_seq_kwargs()

    # State with UPGMA gene trees (arm-A start)
    sampler = MCMC_SEQ(**kw, priors=priors)
    sampler.species_net = _clone_net(true_net)
    probe(sampler._new_state, "UPGMA gene trees")

    # State with the TRUE simulated gene trees (oracle)
    def make_true_state():
        return SeqState(_clone_net(true_net),
                        [_clone_net(gt) for gt in data.gene_trees],
                        data.species_of, kw["loci"], priors, kw["model"],
                        kw.get("theta", 0.02) or 0.02)
    probe(make_true_state, "TRUE gene trees")


if __name__ == "__main__":
    main()
