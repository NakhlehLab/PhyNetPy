r"""Over-correction check: does the branch-length prior make the sampler add
reticulations to data with NO reticulation?

Simulate sequences from a plain SPECIES TREE (r=0 truth), start at that tree with
UPGMA gene trees, and measure op_add_reticulation_decoupled's log acceptance.
If median log_alpha < 0, the balance is right (won't over-add on tree data);
if it is strongly > 0, the branch-length prior over-rewards extra short edges.

For contrast we also run the same add probe on data simulated WITH the true
reticulation, where add SHOULD be favored.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from phynetpy.infer import JC69, simulate_multilocus
from phynetpy._mcmc_seq import (
    MCMC_SEQ, MCMCSeqPriors, op_add_reticulation_decoupled, _clone_net,
)
from _run_weekend import build_true_network, MAPPING


def _tree_from_true(true_net):
    """Major displayed tree: drop the reticulation's minor parent edge."""
    from phynetpy.ModelMove import _suppress_deg2
    t = _clone_net(true_net)
    retics = [v for v in t.V() if v.is_reticulation()]
    for r in retics:
        in_e = list(t.in_edges(r))
        # keep the higher-gamma (major) edge, remove the lower (minor) one
        in_e.sort(key=lambda e: (e.get_gamma() or 0.0))
        minor = in_e[0]
        src = minor.src
        t.remove_edge(minor)
        _suppress_deg2(t, r)
        _suppress_deg2(t, src)
    return t


def probe(species_net, data, priors, tag, n=150):
    kw = data.to_mcmc_seq_kwargs()
    sampler = MCMC_SEQ(**kw, priors=priors)
    sampler.species_net = _clone_net(species_net)
    rng = np.random.default_rng(0)
    rows, declined = [], 0
    for _ in range(n):
        st = sampler._new_state()
        ll0, lp0 = st.log_likelihood(), st.log_prior()
        res = op_add_reticulation_decoupled(st, rng)
        if res is None:
            declined += 1
            continue
        loghr, undo = res
        la = loghr + (st.log_likelihood() - ll0) + (st.log_prior() - lp0)
        rows.append((loghr, la))
        undo()
    print(f"\n--- add: {tag} (n={n}, declined={declined}) ---")
    if rows:
        a = np.array(rows)
        la = a[:, 1][np.isfinite(a[:, 1])]
        if len(la):
            print(f"  loghr med={np.median(a[:,0]):+.3f}")
            print(f"  log_alpha min={la.min():+.3f} med={np.median(la):+.3f} "
                  f"max={la.max():+.3f}  (finite {len(la)}/{len(rows)})")
            print(f"  add would-accept (log_alpha>0): {int((la>0).sum())}/{len(la)}")
            v = np.median(la)
            print(f"  => {'ADD FAVORED' if v > 0 else 'add disfavored (stays tree)'}")
        else:
            print("  all add proposals -inf (rejected)")


def main():
    true_net = build_true_network()
    priors = MCMCSeqPriors(max_reticulations=1, max_level=1)

    # r=0 truth: simulate from the major displayed tree
    tree = _tree_from_true(true_net)
    data_tree = simulate_multilocus(tree, MAPPING, n_loci=15, seq_length=500,
                                    theta=0.02, model=JC69(), seed=7001)
    probe(tree, data_tree, priors, "TREE data, start at tree (should NOT add)")

    # r=1 truth: simulate from the reticulate network
    data_ret = simulate_multilocus(true_net, MAPPING, n_loci=15, seq_length=500,
                                   theta=0.02, model=JC69(), seed=2024)
    probe(tree, data_ret, priors, "RETIC data, start at tree (SHOULD add)")


if __name__ == "__main__":
    main()
