r"""Prior-only RJMCMC validation for the coupled add/delete-reticulation kernel.

Goal
----
Decide whether the dimension-changing moves (``op_add_reticulation_coupled`` /
``op_delete_reticulation_coupled``) are *correct* -- i.e. whether the sampler,
run with the likelihood switched off, targets the prior it is supposed to.  A
Hastings-ratio / Jacobian bug that biases against adding reticulations would
show up here regardless of how much data or how many iterations a real run uses.

Why not just "does the reticulation count match a truncated Poisson?"
--------------------------------------------------------------------
``log_prior_seq`` has **no explicit prior term on divergence times** (the times
are effectively flat/improper), so the prior mass at ``r=1`` is the Poisson x
topology-normaliser weight times an unknown *reachable continuous volume* the
add move integrates over.  That volume is not available in closed form, so the
raw count marginal is not a clean Poisson even for a correct kernel.

The volume-agnostic test
------------------------
For a two-state chain (``max_reticulations = 1``) the stationary log-odds are

    L(m) = log pi(r=1) - log pi(r=0)
         = [log Pois(1; m) - log Pois(0; m)] + log(topo_norm) + log(volume)
         = log(m) + C,

where ``C`` collects the (unknown, m-independent) topology-normaliser and
reachable-volume terms.  Sweeping the Poisson mean ``m`` and checking that

    L(m) - log(m) == C   (a constant, i.e. flat in m)

cancels ``C`` entirely.  A *correct* kernel gives a flat line; the slope of
``L(m)`` vs ``log(m)`` must be 1.  A biased add/delete pair breaks this.

Secondary diagnostics: add/delete acceptance rates and a two-start agreement
check (chains launched from r=0 and r=1 must reach the same occupancy).
"""
from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import phynetpy._mcmc_seq as seqmod
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy._mcmc_seq import _clone_net

# --- small, cheap fixture: the DATA CONTENT is irrelevant to a prior-only test,
#     we only need loci so the engine + gene-tree moves are exercised. ----------
from phynetpy.Network import Network, Node, Edge


def _tiny_true_net() -> Network:
    """A 4-taxon, 1-reticulation ultrametric network (only used to sim data)."""
    h = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0,
         "AB": 0.02, "H": 0.03, "P": 0.05, "Q": 0.06, "R": 0.08}
    nodes = {n: Node(n, is_reticulation=(n == "H")) for n in h}
    net = Network()
    net.add_nodes(*nodes.values())
    edges = [("R", "P", None), ("R", "Q", None), ("P", "AB", None),
             ("AB", "A", None), ("AB", "B", None), ("P", "H", 0.7),
             ("Q", "H", 0.3), ("H", "C", None), ("Q", "D", None)]
    E = []
    for p, c, g in edges:
        ln = h[p] - h[c]
        E.append(Edge(nodes[p], nodes[c], length=ln)
                 if g is None else
                 Edge(nodes[p], nodes[c], length=ln, gamma=g))
    net.add_edges(E)
    return net


MAPPING = {sp: [sp] for sp in ("A", "B", "C", "D")}


def _occupancy(sampler, *, poisson_mean, seed, num_iter, burn_in,
               start_net=None):
    """Run one prior-only chain; return (freq dict, add/del accept stats)."""
    sampler.priors.poisson_mean = poisson_mean
    if start_net is not None:
        sampler.species_net = _clone_net(start_net)

    counts = {0: 0, 1: 0, 2: 0}

    def control(prog):
        if prog["iteration"] >= burn_in:  # tally only at stationarity
            r = prog["num_reticulations"]
            counts[r] = counts.get(r, 0) + 1
        return "continue"

    sampler.search(num_iter=num_iter, burn_in=burn_in, sample_freq=10_000_000,
                   seed=seed, warm_start=False, control=control, check_every=1)
    total = sum(counts.values())
    freq = {k: v / total for k, v in counts.items() if total}
    return freq, counts


def main() -> None:
    # Turn the likelihood OFF: target := prior.  The informed proposals still
    # score candidates against the engine (that only shapes q, not pi), so this
    # exercises the *full* coupled kernel against a known target.
    seqmod.SeqState.log_likelihood = lambda self: 0.0

    print("Simulating tiny fixture data (content irrelevant to prior-only)...")
    data = simulate_multilocus(_tiny_true_net(), MAPPING, n_loci=4,
                               seq_length=200, theta=0.02, model=JC69(),
                               seed=1)

    means = [0.5, 1.0, 2.0, 4.0]
    NUM_ITER, BURN = 120_000, 20_000

    print("\n=== L(m) response test (max_reticulations=1) ===")
    print("  correct kernel: L(m) - log(m) is CONSTANT across m\n")
    rows = []
    for m in means:
        sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                           priors=MCMCSeqPriors(max_reticulations=1,
                                                max_level=1,
                                                use_diameter_prior=False,
                                                gamma_alpha=1.0, gamma_beta=1.0))
        freq, counts = _occupancy(sampler, poisson_mean=m, seed=100,
                                  num_iter=NUM_ITER, burn_in=BURN)
        p0 = freq.get(0, 0.0)
        p1 = freq.get(1, 0.0)
        if p0 > 0 and p1 > 0:
            L = math.log(p1 / p0)
            resid = L - math.log(m)
        else:
            L = resid = float("nan")
        rows.append((m, p0, p1, L, resid))
        print(f"  m={m:<4}  P(r0)={p0:.3f} P(r1)={p1:.3f}  "
              f"L(m)={L:+.3f}  L-log(m)={resid:+.3f}  "
              f"(counts {counts[0]}/{counts[1]})")

    resids = [r[4] for r in rows if r[4] == r[4]]
    if len(resids) >= 2:
        spread = max(resids) - min(resids)
        print(f"\n  residual spread (max-min of L-log(m)) = {spread:.3f}")
        print("  slope of L vs log(m):")
        xs = np.array([math.log(r[0]) for r in rows if r[4] == r[4]])
        ys = np.array([r[3] for r in rows if r[4] == r[4]])
        slope = np.polyfit(xs, ys, 1)[0]
        print(f"    fitted slope = {slope:.3f}  (correct kernel -> 1.000)")
        verdict = ("PASS" if spread < 0.25 and abs(slope - 1.0) < 0.15
                   else "FAIL / SUSPECT")
        print(f"\n  VERDICT: {verdict}")

    # Two-start agreement at m=1 (both must reach same occupancy).
    print("\n=== two-start agreement (m=1.0) ===")
    tiny = _tiny_true_net()
    for label, start in (("start r=0", None), ("start r=1", tiny)):
        sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                           priors=MCMCSeqPriors(max_reticulations=1,
                                                max_level=1,
                                                use_diameter_prior=False,
                                                gamma_alpha=1.0, gamma_beta=1.0))
        freq, counts = _occupancy(sampler, poisson_mean=1.0, seed=7,
                                  num_iter=NUM_ITER, burn_in=BURN,
                                  start_net=start)
        print(f"  {label}: P(r0)={freq.get(0,0):.3f} P(r1)={freq.get(1,0):.3f}")


if __name__ == "__main__":
    main()
