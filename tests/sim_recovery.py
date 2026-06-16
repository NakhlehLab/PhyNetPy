#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##############################################################################

"""
End-to-end MCMC_SEQ recovery / calibration harness on *known-truth* data.

Because the data are generated under the very model MCMC_SEQ assumes
(multispecies network coalescent + a nucleotide substitution model -- see
:mod:`phynetpy._sim_seq`), this is a fair test: we know the true species
network, the true population mutation rate ``theta``, and the true gene trees,
so we can ask whether the sampler *recovers* them and whether its credible
intervals are *calibrated*.

What it does
------------
1. Simulates ``--loci`` independent alignments on a chosen true species network
   (a 1-reticulation network by default, or a plain tree with ``--tree``).
2. Runs MCMC_SEQ from its own UPGMA starting point.
3. Reports recovery: the MAP network, whether each true clade was recovered,
   the posterior mean / 95% HPD of ``theta`` vs the truth, and the posterior on
   the reticulation count.
4. Writes Tracer-compatible ``.log`` and NEXUS ``.trees`` files (``--out``).
5. With ``--replicates K`` runs an independent calibration study: across K
   simulated data sets it reports how often the 95% HPD for ``theta`` covers
   the truth (well-calibrated ~= 0.95).

Examples
--------
    py tests/sim_recovery.py                      # default reticulate run
    py tests/sim_recovery.py --tree --loci 20 --iters 60000
    py tests/sim_recovery.py --replicates 20 --loci 8 --iters 20000
"""

from __future__ import annotations

import argparse
import os

from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.infer import (
    MCMC_SEQ,
    MCMCSeqPriors,
    MCMC_GT,
    MCMC_GTPriors,
    JC69,
    simulate_multilocus,
)


# True networks (ultrametric, substitution units).  The reticulate one is a
# clean "B is a hybrid of the A-lineage and the C-lineage" network with
# gamma = 0.65 on the A-side hybrid edge.
TRUE_TREE = "(((A:0.04,B:0.04)AB:0.04,C:0.08)ABC:0.05,D:0.13)R;"
TRUE_NETWORK = (
    "((((A:0.04,B:0.04)AB:0.03)#H1:0.02[&gamma=0.65],C:0.09)ABC:0.04,"
    "(#H1:0.04[&gamma=0.35],D:0.11)DR:0.02)R;"
)
TRUE_CLADES = [{"A", "B"}, {"A", "B", "C"}]


def _descendant_leaves(net: Network, node) -> frozenset:
    kids = net.get_children(node)
    if not kids:
        return frozenset({node.label})
    acc: set = set()
    for c in kids:
        acc |= _descendant_leaves(net, c)
    return frozenset(acc)


def _has_clade(net: Network, clade: set) -> bool:
    target = frozenset(clade)
    return any(_descendant_leaves(net, v) == target for v in net.V())


def _all_clades(net: Network) -> set:
    """Every non-trivial leaf-set induced by an internal node of ``net``."""
    clades = set()
    leaves = {n.label for n in net.get_leaves()}
    for v in net.V():
        ds = _descendant_leaves(net, v)
        if 1 < len(ds) < len(leaves):
            clades.add(ds)
    return clades


def _topology_recovered(map_net: Network, true_net: Network) -> bool:
    """True when ``map_net`` induces every clade the true network does."""
    return _all_clades(true_net).issubset(_all_clades(map_net))


def _map_reticulation_gamma(net: Network):
    """Largest inheritance probability on a reticulation in-edge, or None."""
    gammas = []
    for v in net.V():
        if v.is_reticulation():
            for e in net.in_edges(v):
                g = e.get_gamma()
                if g is not None:
                    gammas.append(float(g))
    if not gammas:
        return None
    # Report the gamma on the major hybrid edge (the larger of the pair).
    return max(gammas)


def _theta_posterior(result):
    """(mean, lo, hi, ess) for theta from a result's chain summary."""
    summ = result.summary()
    p = summ.parameters.get("theta")
    if p is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return p.mean, p.lower_hpd, p.upper_hpd, p.ess


def run_single(args) -> None:
    """Simulate one data set, run MCMC_SEQ, and report recovery."""
    true_newick = TRUE_TREE if args.tree else TRUE_NETWORK
    true_net = Network.from_newick(true_newick)
    mapping = {sp: [sp] for sp in ("A", "B", "C", "D")}

    print(f"True network : {true_newick}")
    print(f"True theta    : {args.theta}")
    print(f"Simulating {args.loci} loci x {args.sites} bp (seed={args.seed}) ...")
    data = simulate_multilocus(
        true_net, mapping, n_loci=args.loci, seq_length=args.sites,
        theta=args.theta, model=JC69(), seed=args.seed,
    )

    priors = MCMCSeqPriors(max_reticulations=0 if args.tree else 4)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=priors)
    print(f"Starting logP : {sampler.score():.3f}")
    print(
        f"Running MCMC_SEQ: {args.iters} iters "
        f"(burn-in {args.burnin}, thin {args.thin}) ..."
    )
    result = sampler.search(
        num_iter=args.iters, burn_in=args.burnin, sample_freq=args.thin,
        seed=args.seed, progress=True,
    )

    print()
    print(f"MAP network   : {result.map_network.newick()}")
    print(f"MAP logP      : {result.map_log_posterior:.3f}")
    print(f"MAP theta     : {result.map_theta:.6f}  (true {args.theta})")
    for clade in TRUE_CLADES:
        ok = "RECOVERED" if _has_clade(result.map_network, clade) else "missed"
        print(f"  clade {sorted(clade)}: {ok}")

    full = _topology_recovered(result.map_network, true_net)
    print(f"  full topology : {'RECOVERED' if full else 'missed'}")
    if not args.tree:
        g_map = _map_reticulation_gamma(result.map_network)
        g_true = _map_reticulation_gamma(true_net)
        if g_map is not None and g_true is not None:
            print(
                f"  gamma (major) : MAP={g_map:.3f}  true={g_true:.3f}  "
                f"|err|={abs(g_map - g_true):.3f}"
            )

    mean, lo, hi, ess = _theta_posterior(result)
    covers = lo <= args.theta <= hi
    print(
        f"theta posterior: mean={mean:.6f}  95% HPD=[{lo:.6f},{hi:.6f}]  "
        f"ESS={ess:.1f}  covers truth: {covers}"
    )
    print()
    print(result.summary())

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        result.write_log(args.out + ".log")
        result.write_networks(args.out + ".trees")
        print(f"\nWrote {args.out}.log and {args.out}.trees (open in Tracer).")


def run_single_gt(args) -> None:
    """Simulate, then drive the GENE-TREE (MCMC_GT) stack and report recovery.

    The same coalescent simulator produces both alignments and the true gene
    trees; the gene-tree sampler infers the species network directly from the
    simulated gene-tree topologies (no sequences).  This exercises the GT
    branch of the unified move machinery: the reversible-jump add/remove
    reticulation pair and the corrected parameter moves.
    """
    true_newick = TRUE_TREE if args.tree else TRUE_NETWORK
    true_net = Network.from_newick(true_newick)
    mapping = {sp: [sp] for sp in ("A", "B", "C", "D")}

    print(f"True network : {true_newick}")
    print(f"Simulating {args.loci} loci (seed={args.seed}) for GT inference ...")
    data = simulate_multilocus(
        true_net, mapping, n_loci=args.loci, seq_length=args.sites,
        theta=args.theta, model=JC69(), seed=args.seed,
    )

    genetrees = GeneTrees(
        gene_tree_list=list(data.gene_trees), species_gene_mapping=mapping
    )
    max_retics = 0 if args.tree else 4
    mcmc = MCMC_GT.from_consensus(genetrees, mapping, priors=MCMC_GTPriors())
    print(
        f"Running MCMC_GT (mh): {args.iters} iters "
        f"(burn-in {args.burnin}, thin {args.thin}) ..."
    )
    result = mcmc.search(
        method="mh", num_iter=args.iters, burn_in=args.burnin,
        thin=args.thin, max_reticulations=max_retics, seed=args.seed,
    )

    map_net = result.best_network
    print()
    print(f"MAP network   : {map_net.newick()}")
    print(f"MAP logP      : {result.best_log_posterior:.3f}")
    print(f"acceptance    : {result.num_accepted}/{result.num_iter} "
          f"= {result.num_accepted / max(1, result.num_iter):.3f}")
    for clade in TRUE_CLADES:
        ok = "RECOVERED" if _has_clade(map_net, clade) else "missed"
        print(f"  clade {sorted(clade)}: {ok}")
    full = _topology_recovered(map_net, true_net)
    print(f"  full topology : {'RECOVERED' if full else 'missed'}")

    n_retic = sum(1 for v in map_net.V() if v.is_reticulation())
    print(f"  reticulations : MAP has {n_retic} (true "
          f"{0 if args.tree else 1})")
    if not args.tree:
        g_map = _map_reticulation_gamma(map_net)
        g_true = _map_reticulation_gamma(true_net)
        if g_map is not None and g_true is not None:
            print(
                f"  gamma (major) : MAP={g_map:.3f}  true={g_true:.3f}  "
                f"|err|={abs(g_map - g_true):.3f}"
            )
        elif g_map is None:
            print("  gamma (major) : MAP network has no reticulation")


def run_calibration(args) -> None:
    """Repeat simulate->infer across replicates; report theta HPD coverage."""
    true_newick = TRUE_TREE if args.tree else TRUE_NETWORK
    true_net = Network.from_newick(true_newick)
    mapping = {sp: [sp] for sp in ("A", "B", "C", "D")}
    priors = MCMCSeqPriors(max_reticulations=0 if args.tree else 4)

    covered = 0
    clade_hits = {frozenset(c): 0 for c in TRUE_CLADES}
    print(f"Calibration: {args.replicates} replicates "
          f"({args.loci} loci x {args.sites} bp, theta={args.theta})")
    for rep in range(args.replicates):
        seed = args.seed + rep
        data = simulate_multilocus(
            true_net, mapping, n_loci=args.loci, seq_length=args.sites,
            theta=args.theta, model=JC69(), seed=seed,
        )
        sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=priors)
        result = sampler.search(
            num_iter=args.iters, burn_in=args.burnin, sample_freq=args.thin,
            seed=seed,
        )
        _, lo, hi, _ = _theta_posterior(result)
        hit = lo <= args.theta <= hi
        covered += int(hit)
        for c in TRUE_CLADES:
            if _has_clade(result.map_network, c):
                clade_hits[frozenset(c)] += 1
        print(f"  rep {rep:>2}: theta HPD=[{lo:.5f},{hi:.5f}] covers={hit}")

    print()
    print(f"theta 95% HPD coverage: {covered}/{args.replicates} "
          f"= {covered / args.replicates:.2f}  (target ~0.95)")
    for c, n in clade_hits.items():
        print(f"clade {sorted(c)} recovered in MAP: "
              f"{n}/{args.replicates} = {n / args.replicates:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", action="store_true",
                    help="use the true species TREE instead of the network")
    ap.add_argument("--gt", action="store_true",
                    help="drive the gene-tree (MCMC_GT) stack instead of "
                         "MCMC_SEQ (infers from simulated gene-tree topologies)")
    ap.add_argument("--loci", type=int, default=10)
    ap.add_argument("--sites", type=int, default=400)
    ap.add_argument("--theta", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--burnin", type=int, default=10000)
    ap.add_argument("--thin", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--replicates", type=int, default=0,
                    help="if >0, run a calibration study with this many reps")
    ap.add_argument("--out", type=str, default="",
                    help="path prefix for Tracer .log / .trees output")
    args = ap.parse_args()

    if args.replicates > 0:
        run_calibration(args)
    elif args.gt:
        run_single_gt(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
