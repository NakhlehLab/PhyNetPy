"""Rigorous probe of the theta->0 coalescent degeneracy in MCMC_SEQ.

Tests, on KNOWN-TRUTH data, whether the joint (gene-tree heights, theta)
log-density is unbounded as theta->0 when unconstrained coalescent gaps
collapse -- and whether the Felsenstein term and the theta prior can stop it.

If (A) the coalescent density at the FIXED true gene trees peaks at a finite
theta, but (B) shrinking the gene-tree root gap lets total log-density grow
without bound as theta->0, then the pathology is a *model* degeneracy
(integrable posterior spike), not a code decoupling bug.
"""
import math

import numpy as np

from phynetpy.Network import Network
from phynetpy.infer import JC69, simulate_multilocus
from phynetpy._seq_likelihood import (
    FelsensteinCalculator, gene_tree_msnc_log_density, _node_height,
)
from phynetpy._mcmc_seq import _heights, _sync_lengths, _log_gamma_pdf

TRUE_NETWORK = (
    "((((A:0.04,B:0.04)AB:0.03)#H1:0.02[&gamma=0.65],C:0.09)ABC:0.04,"
    "(#H1:0.04[&gamma=0.35],D:0.11)DR:0.02)R;"
)
THETA_SHAPE, THETA_PRIOR_MEAN = 2.0, 0.036


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
    gts = data.gene_trees

    # species-network root height: the embedding floor for the last coalescence
    sp_h: dict = {}
    sp_root_h = _node_height(true_net, true_net.root(), sp_h)

    def felsen_total(trees):
        return sum(c.log_likelihood(g, model) for c, g in zip(calcs, trees))

    def msnc_total(trees, theta):
        return sum(
            gene_tree_msnc_log_density(g, true_net, species_of, theta=theta)
            for g in trees
        )

    def theta_logprior(theta):
        return _log_gamma_pdf(theta, THETA_SHAPE, THETA_PRIOR_MEAN)

    # ---- (A) FIXED true gene trees: scan theta ----
    fel0 = felsen_total(gts)
    print(f"species-root height (embedding floor) = {sp_root_h:.4f}")
    print(f"Felsenstein (fixed true gene trees)    = {fel0:.2f}\n")
    print("(A) FIXED true gene trees -- coalescent density vs theta:")
    print(f"{'theta':>9} {'MSNC':>12} {'+thetaPrior':>12} {'total':>12}")
    best_a = (-math.inf, None)
    for theta in [0.0002, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08]:
        ms = msnc_total(gts, theta)
        tot = fel0 + ms + theta_logprior(theta)
        if tot > best_a[0]:
            best_a = (tot, theta)
        print(f"{theta:>9.4f} {ms:>12.2f} {theta_logprior(theta):>12.2f} "
              f"{tot:>12.2f}")
    print(f"  -> total maximised at theta={best_a[1]} "
          f"(true {TRUE_THETA}); degeneracy at fixed g? "
          f"{'NO' if best_a[1] and best_a[1] >= 0.005 else 'maybe'}\n")

    # ---- (B) collapse the TOP (unconstrained) coalescent gap, re-optimise theta ----
    # Lower each gene tree's root coalescence toward the embedding floor; the
    # gap between it and max(highest child, species root) is unconstrained.
    def collapsed_trees(keep_frac):
        out = []
        for g in gts:
            gc = __import__("copy").deepcopy(g)
            h = _heights(gc)
            root = gc.root()
            kids = gc.get_children(root)
            floor = max(max(h[c] for c in kids), sp_root_h)
            # keep_frac in (0,1]: fraction of the original top gap retained.
            h[root] = floor + (h[root] - floor) * keep_frac
            _sync_lengths(gc, h)
            out.append(gc)
        return out

    print("(B) collapse the top (root) coalescent gap, re-optimise theta:")
    print(f"{'keep_gap':>9} {'bestTheta':>10} {'Felsen':>11} {'MSNC':>11} "
          f"{'total':>11}")
    baseline = fel0 + msnc_total(gts, TRUE_THETA) + theta_logprior(TRUE_THETA)
    print(f"  baseline (true gene trees, theta={TRUE_THETA}): "
          f"total={baseline:.2f}")
    thetas = [0.0001, 0.0002, 0.0005, 0.001, 0.005, 0.02]
    for keep in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]:
        trees = collapsed_trees(keep)
        fel = felsen_total(trees)
        best = (-math.inf, None, None)
        for theta in thetas:
            ms = msnc_total(trees, theta)
            tot = fel + ms + theta_logprior(theta)
            if tot > best[0]:
                best = (tot, theta, ms)
        print(f"{keep:>9.3f} {best[1]:>10.4f} {fel:>11.2f} {best[2]:>11.2f} "
              f"{best[0]:>11.2f}")
    print("\nIf 'total' rises as keep_gap->0 (with bestTheta falling), the "
          "joint posterior has an unbounded theta->0 spike that Felsenstein "
          "(<=0 in log) and the Gamma(2) theta prior cannot cap.")


if __name__ == "__main__":
    main()
