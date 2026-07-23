"""Likelihood delta of a backbone-preserving add as a function of birth gamma.

From a settled tree state, for a range of birth gammas, insert a reticulation
(v2 on a random reticulation-edge keeps its original parent as the 1-gamma
'backbone' parent; a new v1 on a donor edge is the gamma parent) and measure
the likelihood delta (prop - cur), maximised over many random edge pairs.

If small gamma -> delta ~ 0, then a birth-near-zero-gamma proposal makes the
add near-neutral and the move becomes acceptable once the Jacobian is tamed.
"""
import os, sys, copy, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy import _mcmc_seq as M
from phynetpy.Network import Edge

true_net = build_true_network()
data = simulate_multilocus(true_net, MAPPING, n_loci=30, seq_length=400,
                           theta=0.02, model=JC69(), seed=7)

sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                   priors=MCMCSeqPriors(max_reticulations=2))
rng = np.random.default_rng(1)
kernel = M.MCMCSeqKernel(rng, max_reticulations=2)
kernel._ops = [(op, w) for op, w in kernel._ops
               if op.__name__ not in ("op_add_reticulation", "op_delete_reticulation")]
kernel._weights = np.asarray([w for _, w in kernel._ops]); kernel._weights /= kernel._weights.sum()
state = sampler._new_state()
cur = state.log_posterior()
for _ in range(4000):
    p = kernel.propose(state)
    if p is not None:
        loghr, undo = p
        prop = state.log_posterior()
        if math.isfinite(prop) and math.log(rng.random()) < (prop - cur) + loghr:
            cur = prop
        else:
            undo()
print(f"tree state: logP={cur:.2f}")

def try_add_with_gamma(g, tries=400):
    """Best likelihood delta over random backbone-preserving adds with birth gamma g."""
    best = float("-inf")
    for _ in range(tries):
        old_net = state.species_net
        old_heights = state.net_heights
        snap = state._engine.clone_caches()
        new_net = copy.deepcopy(old_net)
        new_heights = M._heights(new_net)
        edges = [e for e in new_net.E()]
        E = len(edges)
        ia, ib = rng.choice(E, size=2, replace=False)
        e1, e2 = edges[int(ia)], edges[int(ib)]
        v3, v4 = e1.src, e1.dest
        v5, v6 = e2.src, e2.dest
        l1 = new_heights[v3] - new_heights[v4]
        l2 = new_heights[v5] - new_heights[v6]
        if l1 <= 0 or l2 <= 0:
            continue
        t1 = new_heights[v4] + l1 * rng.random()
        t2 = new_heights[v6] + l2 * rng.random()
        if t1 <= t2:
            continue
        try:
            v1 = M._split_edge(new_net, new_heights, v3, v4, t1)
            v2 = M._split_edge(new_net, new_heights, v5, v6, t2)
            in_e = list(new_net.in_edges(v2))
            if len(in_e) != 1:
                continue
            in_e[0].set_gamma(1.0 - g)
            v2.set_is_reticulation(True)
            new_net.add_edges(Edge(v1, v2, gamma=g))
        except Exception:
            continue
        if M._has_parallel_edges(new_net) or not new_net.is_acyclic():
            continue
        M._sync_lengths(new_net, new_heights)
        state.species_net = new_net
        state.net_heights = new_heights
        state._engine.invalidate_network()
        prop = state.log_posterior()
        # restore
        state.species_net = old_net
        state.net_heights = old_heights
        state._engine.restore_caches(snap)
        if math.isfinite(prop):
            best = max(best, prop - cur)
    return best

for g in [0.5, 0.2, 0.1, 0.05, 0.02, 0.005]:
    d = try_add_with_gamma(g)
    print(f"birth gamma={g:5.3f}: best likelihood delta = {d:+.2f}")
