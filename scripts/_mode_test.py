r"""Is r=1 the posterior mode, or does the chain just fail to hold it?

Decisive test of "mixing race vs posterior preference".  Start at the TRUE r=1
network with UPGMA gene trees, then run ONLY the continuous / gene-tree moves
(dimension moves disabled) so the network is pinned at r=1 and the gene trees are
free to adapt.  Track:

  * total log-likelihood (does it climb as gene trees learn to use the retic?),
  * how many loci become 'reticulation-dependent' -- i.e. removing the retic
    would drop their embedding to -inf,
  * the would-be delete log-acceptance at the CURRENT (adapted) state.

Interpretation:
  * If delete flips from favored (+) to strongly rejected (-) as the gene trees
    adapt, then r=1 IS the posterior mode and the collapse was a proposal RACE
    (delete fires before gene trees adapt) -> initialising gene trees under the
    seed network is legitimate, not manipulative.
  * If delete stays favored even after long gene-tree adaptation, then at this
    data size the posterior does NOT prefer r=1 -> forcing it would be wrong.
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
    MCMC_SEQ, MCMCSeqPriors, _clone_net,
    op_gene_node_height, op_gene_tree_nni, op_net_node_height,
    op_change_gamma, op_change_theta, op_delete_reticulation_decoupled,
)
from _run_weekend import build_true_network, MAPPING

N_LOCI, SITES, ITERS, SEED = 15, 500, 30_000, 5


def delete_logalpha(state, rng):
    """Would-be log acceptance of deleting the reticulation NOW (not applied)."""
    ll0, lp0 = state.log_likelihood(), state.log_prior()
    res = op_delete_reticulation_decoupled(state, rng)
    if res is None:
        return None
    loghr, undo = res
    la = loghr + (state.log_likelihood() - ll0) + (state.log_prior() - lp0)
    undo()
    return la


def retic_dependent_loci(state, rng, sampler):
    """#loci whose likelihood would drop to -inf if the reticulation is removed.

    Approximated by deleting the reticulation (network-only) and counting loci
    whose MSNC becomes non-finite; restored immediately.
    """
    res = op_delete_reticulation_decoupled(state, rng)
    if res is None:
        return None
    _loghr, undo = res
    eng = state._engine
    net_idx, sph = eng._network_index(state.species_net)
    from phynetpy._msnc_density import (build_gene_tree_msnc_index,
                                        msnc_log_density_prebuilt)
    dep = 0
    for i in range(eng.n_loci):
        gti, ev = build_gene_tree_msnc_index(state.gene_trees[i], eng._species_of)
        m = msnc_log_density_prebuilt(net_idx, sph, gti, ev, state.theta)
        if not math.isfinite(m):
            dep += 1
    undo()
    return dep


def main():
    true_net = build_true_network()
    data = simulate_multilocus(true_net, MAPPING, n_loci=N_LOCI, seq_length=SITES,
                               theta=0.02, model=JC69(), seed=2024)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=1, max_level=1))
    sampler.species_net = _clone_net(true_net)
    state = sampler._new_state()
    rng = np.random.default_rng(SEED)

    # continuous / gene-tree moves only -- network pinned at r=1
    ops = [(op_gene_node_height, 0.35), (op_gene_tree_nni, 0.30),
           (op_net_node_height, 0.15), (op_change_gamma, 0.10),
           (op_change_theta, 0.10)]
    ws = np.array([w for _, w in ops]); ws /= ws.sum()

    print(f"Fixed-r=1 gene-tree adaptation ({N_LOCI} loci x {SITES} sites)\n")
    print(f"{'iter':>7} {'logLik':>12} {'retic-dep loci':>15} "
          f"{'delete logalpha':>16}")
    cur = state.log_posterior()

    def report(it):
        ll = state.log_likelihood()
        dep = retic_dependent_loci(state, rng, sampler)
        da = delete_logalpha(state, rng)
        verdict = ""
        if da is not None:
            verdict = "delete FAVORED" if da > 0 else "delete rejected(holds r=1)"
        print(f"{it:>7} {ll:>12.2f} {str(dep):>15} "
              f"{('%+.2f' % da) if da is not None else 'NA':>16}  {verdict}")

    report(0)
    for it in range(1, ITERS + 1):
        i = int(rng.choice(len(ops), p=ws))
        res = ops[i][0](state, rng)
        if res is not None:
            loghr, undo = res
            prop = state.log_posterior()
            if not math.isfinite(prop):
                undo()
            elif math.log(rng.random()) < (prop - cur) + loghr:
                cur = prop
            else:
                undo()
        if it % 5000 == 0:
            report(it)


if __name__ == "__main__":
    main()
