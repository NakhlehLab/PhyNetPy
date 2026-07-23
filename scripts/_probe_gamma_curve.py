"""Where does the posterior actually peak in gamma?

Removes every confound except gamma: seed a SeqState with the TRUE per-locus
gene trees and the TRUE network (topology + heights), then sweep the hybrid
inheritance probability over a grid and record the exact log-posterior at each
value.  The argmax tells us the target of the gamma move.

  * argmax ~ 0.65  -> the move/likelihood are correct; any run-time bias is
                      convergence / gene-tree uncertainty (needs more iters).
  * argmax  > 0.85 -> the *posterior itself* prefers high gamma on this data,
                      which is either finite-sample variance or a bug to chase.

We also sweep at several data sizes so we can see the peak sharpen toward 0.65
as information increases (the signature of an unbiased estimator).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from mcmc_harness import build_true_network, MAPPING, TRUE_GAMMA_MAJOR


def gamma_curve(sampler, grid):
    state = sampler._new_state()
    rets = [v for v in state.species_net.V() if v.is_reticulation()]
    r = rets[0]
    in_edges = list(state.species_net.in_edges(r))
    e0, e1 = in_edges[0], in_edges[1]
    out = []
    for g in grid:
        e0.set_gamma(float(g))
        e1.set_gamma(float(1.0 - g))
        state._engine.invalidate_network()
        out.append(state.log_posterior())
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    true_net = build_true_network()
    grid = np.linspace(0.05, 0.95, 19)

    print(f"true gamma_major = {TRUE_GAMMA_MAJOR}")
    print("gamma peak of the EXACT log-posterior (true gene trees + true net):\n")
    print(f"{'loci x sites':>14} | {'argmax gamma':>12} | "
          f"{'post-mean gamma':>15} | {'logP@peak':>12}")
    print("-" * 64)
    for loci, sites in [(15, 600), (30, 1000), (60, 2000)]:
        data = simulate_multilocus(true_net, MAPPING, n_loci=loci,
                                   seq_length=sites, theta=0.02, model=JC69(),
                                   seed=a.seed)
        sampler = MCMC_SEQ(loci=data.loci, mapping=data.mapping,
                           model=data.model, theta=data.true_theta,
                           species_net=true_net, gene_trees=data.gene_trees,
                           priors=MCMCSeqPriors(max_reticulations=1))
        lp = gamma_curve(sampler, grid)
        # Posterior mean over the grid (normalised) as a robustness check.
        w = np.exp(lp - lp.max())
        w /= w.sum()
        post_mean = float((grid * w).sum())
        argmax = float(grid[int(np.argmax(lp))])
        print(f"{loci:>4} x {sites:<7} | {argmax:>12.3f} | {post_mean:>15.3f} "
              f"| {lp.max():>12.2f}")

    # Fine grid at the largest size for a sharper peak read-out.
    data = simulate_multilocus(true_net, MAPPING, n_loci=60, seq_length=2000,
                               theta=0.02, model=JC69(), seed=a.seed)
    sampler = MCMC_SEQ(loci=data.loci, mapping=data.mapping, model=data.model,
                       theta=data.true_theta, species_net=true_net,
                       gene_trees=data.gene_trees,
                       priors=MCMCSeqPriors(max_reticulations=1))
    fine = np.linspace(0.40, 0.90, 26)
    lpf = gamma_curve(sampler, fine)
    print(f"\nfine peak (60 x 2000): argmax gamma = {fine[int(np.argmax(lpf))]:.3f}")


if __name__ == "__main__":
    main()
