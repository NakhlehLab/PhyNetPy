r"""Volume-agnostic correctness test of the coupled add/delete kernel.

Prior-only (likelihood == 0), two-state (max_reticulations=1).  The stationary
log-odds are  L(m) = log pi(r=1) - log pi(r=0) = log(m) + C, where C bundles the
(m-independent) topology-normaliser and the reachable-continuous-volume term.
At substitution scale C ~ -27, so we offset it with large Poisson means to make
r=1 visitable, then check:

    * slope of L(m) vs log(m)  == 1.0   (correct RJMCMC dimension balance)
    * residual  L(m) - log(m)  == const (C cancels)

A slope != 1 or a drifting residual is a Hastings-ratio / Jacobian bug.
"""
from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import phynetpy._mcmc_seq as seqmod
from phynetpy._mcmc_seq import MCMC_SEQ, MCMCSeqPriors
from phynetpy.infer import JC69, simulate_multilocus
from phynetpy.Network import Network, Node, Edge
from _prior_only_validation import _tiny_true_net, MAPPING


def _deep_species_tree() -> Network:
    """4-taxon caterpillar with LARGE divergences (weak coalescent coupling)."""
    h = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0,
         "AB": 3.0, "ABC": 6.0, "R": 9.0}
    nodes = {n: Node(n) for n in h}
    net = Network()
    net.add_nodes(*nodes.values())
    E = [("AB", "A"), ("AB", "B"), ("ABC", "AB"), ("ABC", "C"),
         ("R", "ABC"), ("R", "D")]
    net.add_edges([Edge(nodes[p], nodes[c], length=h[p] - h[c]) for p, c in E])
    return net


def _deep_gene_tree() -> Network:
    """A gene tree whose coalescences all sit ABOVE the species divergences.

    Keeping every coalescence above the deepest species node guarantees valid
    embedding (no height shrink) and, being deep, makes the MSNC density flat so
    the network and gene trees are weakly coupled -> the chain can cross r=0<->1
    freely, unconfounding the occupancy-based correctness test.
    """
    h = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0,
         "i1": 10.0, "i2": 12.0, "i3": 14.0}
    nodes = {n: Node(n) for n in h}
    gt = Network()
    gt.add_nodes(*nodes.values())
    E = [("i1", "A"), ("i1", "B"), ("i2", "i1"), ("i2", "C"),
         ("i3", "i2"), ("i3", "D")]
    gt.add_edges([Edge(nodes[p], nodes[c], length=h[p] - h[c]) for p, c in E])
    return gt


# Controllable linear reticulation bonus added to the (likelihood-free) target.
# Using a small O(1) bonus instead of a huge Poisson mean avoids catastrophic
# cancellation and lets us slide the r=0/r=1 balance across the transition zone.
_BETA = {"v": 0.0}


def occupancy(sampler, *, beta, seed, num_iter, burn_in):
    _BETA["v"] = beta
    counts = {0: 0, 1: 0}

    def control(prog):
        if prog["iteration"] >= burn_in:
            r = prog["num_reticulations"]
            counts[r] = counts.get(r, 0) + 1
        return "continue"

    sampler.search(num_iter=num_iter, burn_in=burn_in, sample_freq=10_000_000,
                   seed=seed, warm_start=False, control=control, check_every=1)
    return counts


def main() -> None:
    # Target := prior + beta * n_ret  (likelihood off).  A correct RJMCMC then
    # gives stationary log-odds  L(beta) = beta + C, i.e. slope 1 in beta.
    seqmod.SeqState.log_likelihood = lambda self: 0.0
    seqmod.SeqState.log_posterior = (
        lambda self: self.log_prior() + _BETA["v"] * self.num_reticulations()
    )
    # Deep-tree fixture: weak coalescent coupling so the chain crosses r=0<->1
    # freely (unconfounds the occupancy test from co-adaptation hysteresis).
    n_loci = 2
    data = simulate_multilocus(_deep_species_tree(), MAPPING, n_loci=n_loci,
                               seq_length=200, theta=5.0, model=JC69(), seed=1)

    NUM_ITER, BURN = 120_000, 20_000
    betas = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
    rows = []
    print("prior-only kernel correctness (max_reticulations=1, deep tree)")
    print("target = prior + beta*n_ret ; correct kernel -> L(beta)=beta+C\n")
    for b in betas:
        sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                           priors=MCMCSeqPriors(max_reticulations=1, max_level=1,
                                                use_diameter_prior=False,
                                                gamma_alpha=1.0, gamma_beta=1.0))
        sampler.species_net = _deep_species_tree()
        sampler.gene_trees = [_deep_gene_tree() for _ in range(n_loci)]
        sampler.theta = 5.0
        c = occupancy(sampler, beta=b, seed=123,
                      num_iter=NUM_ITER, burn_in=BURN)
        n0, n1 = c.get(0, 0), c.get(1, 0)
        if n0 > 0 and n1 > 0:
            L = math.log(n1 / n0)
            rows.append((b, L))
            print(f"  beta={b:5.1f}  n0={n0:6d} n1={n1:6d}  "
                  f"L={L:+.3f}  L-beta={L-b:+.3f}")
        else:
            print(f"  beta={b:5.1f}  n0={n0:6d} n1={n1:6d}  "
                  f"(one state unvisited)")

    if len(rows) >= 2:
        xs = np.array([r[0] for r in rows])
        ys = np.array([r[1] for r in rows])
        slope, intercept = np.polyfit(xs, ys, 1)
        resid = ys - xs
        print(f"\n  fitted slope       = {slope:.3f}   (correct -> 1.000)")
        print(f"  intercept (C est)  = {intercept:.3f}")
        print(f"  residual spread    = {resid.max()-resid.min():.3f}")
        ok = abs(slope - 1.0) < 0.15 and (resid.max() - resid.min()) < 0.5
        print(f"\n  VERDICT: {'PASS - kernel targets the prior' if ok else 'FAIL/SUSPECT'}")


if __name__ == "__main__":
    main()
