"""Reproduce the cap=4 collapse, capture the MAP state, and decompose it.

Runs a single MCMC_SEQ chain at max_reticulations=4 on known-truth data,
tracks the full MAP state (network + gene trees + theta), and then splits the
MAP log-posterior into Felsenstein (sequence) vs MSNC (coalescent) vs prior.
Also reports geometry diagnostics: branch-length clamps and height inversions
(reticulation nodes above a parent), which would let the MSNC density credit
'negative-duration' coalescent intervals.
"""
import copy
import math

import numpy as np

from phynetpy.Network import Network
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy._seq_likelihood import (
    FelsensteinCalculator, gene_tree_msnc_log_density, _node_height,
)
from phynetpy._mcmc_seq import MCMCSeqKernel, _heights, log_prior_seq

TRUE_NETWORK = (
    "((((A:0.04,B:0.04)AB:0.03)#H1:0.02[&gamma=0.65],C:0.09)ABC:0.04,"
    "(#H1:0.04[&gamma=0.35],D:0.11)DR:0.02)R;"
)
N_ITER = 30_000


def decompose(net, gene_trees, theta, calcs, model, species_of, priors):
    fel = [c.log_likelihood(g, model) for c, g in zip(calcs, gene_trees)]
    ms = [gene_tree_msnc_log_density(g, net, species_of, theta=theta)
          for g in gene_trees]
    lp = log_prior_seq(net, theta, priors)
    return sum(fel), sum(ms), lp


def geometry_report(net):
    h = _heights(net)
    clamped = inversions = 0
    worst_inv = 0.0
    for e in net.E():
        gap = h[e.src] - h[e.dest]
        if gap < 0:
            inversions += 1
            worst_inv = min(worst_inv, gap)
        if abs(gap) < 1e-9:
            clamped += 1
    # reticulation node vs its two parents
    retic_above_parent = 0
    for v in net.V():
        if v.is_reticulation():
            for p in net.get_parents(v):
                if h[p] < h[v] - 1e-12:
                    retic_above_parent += 1
    return clamped, inversions, worst_inv, retic_above_parent


def main():
    true_net = Network.from_newick(TRUE_NETWORK)
    mapping = {sp: [sp] for sp in ("A", "B", "C", "D")}
    species_of = {sp: sp for sp in ("A", "B", "C", "D")}
    LOCI, SITES, TRUE_THETA = 25, 800, 0.02
    data = simulate_multilocus(
        true_net, mapping, n_loci=LOCI, seq_length=SITES,
        theta=TRUE_THETA, model=JC69(), seed=2024,
    )
    model = data.model
    calcs = [FelsensteinCalculator(aln) for aln in data.loci]

    priors = MCMCSeqPriors(max_reticulations=4)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=priors)

    rng = np.random.default_rng(2024)
    kernel = MCMCSeqKernel(rng, max_reticulations=4)
    state = sampler._new_state()
    cur = state.log_posterior()
    map_post = cur
    map_net = copy.deepcopy(state.species_net)
    map_gts = copy.deepcopy(state.gene_trees)
    map_theta = state.theta
    accepted = 0
    for it in range(N_ITER):
        proposal = kernel.propose(state)
        if proposal is None:
            continue
        loghr, undo = proposal
        prop = state.log_posterior()
        if not math.isfinite(prop):
            undo()
            continue
        if math.log(rng.random()) < (prop - cur) + loghr:
            cur = prop
            accepted += 1
            if cur > map_post:
                map_post = cur
                map_net = copy.deepcopy(state.species_net)
                map_gts = copy.deepcopy(state.gene_trees)
                map_theta = state.theta
        else:
            undo()
    print(f"ran {N_ITER} iters, acc={accepted / N_ITER:.3f}\n")

    # --- truth decomposition (reference) ---
    fel_t, ms_t, lp_t = decompose(
        true_net, data.gene_trees, TRUE_THETA, calcs, model, species_of, priors
    )
    print("TRUE config (true net, true gene trees, theta=0.02):")
    print(f"  Felsenstein={fel_t:.2f}  MSNC={ms_t:.2f}  prior={lp_t:.2f}  "
          f"total={fel_t + ms_t + lp_t:.2f}\n")

    # --- MAP decomposition ---
    fel_m, ms_m, lp_m = decompose(
        map_net, map_gts, map_theta, calcs, model, species_of, priors
    )
    n_ret = sum(1 for v in map_net.V() if v.is_reticulation())
    clamped, inversions, worst_inv, retic_above = geometry_report(map_net)
    print(f"MAP config: reti={n_ret}  theta={map_theta:.5f}  "
          f"map_post(stored)={map_post:.2f}")
    print(f"  Felsenstein={fel_m:.2f}  MSNC={ms_m:.2f}  prior={lp_m:.2f}  "
          f"total={fel_m + ms_m + lp_m:.2f}")
    print(f"  vs truth: dFelsenstein={fel_m - fel_t:+.2f}  "
          f"dMSNC={ms_m - ms_t:+.2f}  dprior={lp_m - lp_t:+.2f}")
    print(f"\nMAP network geometry:")
    print(f"  branches clamped to ~0 length : {clamped}")
    print(f"  height inversions (parent<child): {inversions} "
          f"(worst gap {worst_inv:.4f})")
    print(f"  reticulation nodes above a parent: {retic_above}")
    print(f"\nMAP network newick:\n  {map_net.newick()}")

    # per-locus MSNC: are a few loci contributing huge positive density?
    ms_each = sorted(
        (gene_tree_msnc_log_density(g, map_net, species_of, theta=map_theta)
         for g in map_gts), reverse=True
    )
    print(f"\nTop per-locus MSNC log-densities (MAP): "
          f"{[round(x, 1) for x in ms_each[:5]]}")
    print(f"(a single-locus coalescent log-DENSITY > ~5 is suspicious; with "
          f"theta={map_theta:.4f}, log(2/theta)={math.log(2 / map_theta):.2f} "
          f"per coalescence)")


if __name__ == "__main__":
    main()
