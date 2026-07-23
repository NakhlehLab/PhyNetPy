"""Does the COUPLED add/delete let a single cold-start MCMC_SEQ chain discover
the reticulation within budget?  Simulates the canonical 6-taxon/1-reticulation
data, starts from a plain species tree, and reports:

  * ms/iteration (coupled moves are heavier -- we want the real number),
  * the reticulation-count trace over the run (when discovery happens),
  * the MAP reticulation count + major gamma vs the truth (0.65),
  * the first iteration at which a reticulation is accepted.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy import _mcmc_seq as ms

# Reuse the harness ground-truth network.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcmc_harness import build_true_network, MAPPING, TRUE_GAMMA_MAJOR, score_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", type=int, default=10)
    ap.add_argument("--sites", type=int, default=400)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--burnin", type=int, default=4000)
    ap.add_argument("--thin", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    true_net = build_true_network()
    data = simulate_multilocus(true_net, MAPPING, n_loci=args.loci,
                               seq_length=args.sites, theta=0.02, model=JC69(),
                               seed=args.seed)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=2))
    print(f"start (cold tree): {sampler.species_net.newick()}")

    t0 = time.perf_counter()
    res = sampler.search(num_iter=args.iters, burn_in=args.burnin,
                         sample_freq=args.thin, seed=args.seed, progress=True)
    dt = time.perf_counter() - t0

    ms_it = 1000.0 * dt / max(1, args.iters)
    retic_trace = [s.num_reticulations for s in res.samples]
    first_retic_it = next(
        (s.iteration for s in res.samples if s.num_reticulations >= 1), None
    )
    # posterior over reticulation count from the samples
    counts = {}
    for r in retic_trace:
        counts[r] = counts.get(r, 0) + 1
    n = max(1, len(retic_trace))
    post = {k: v / n for k, v in sorted(counts.items())}

    acc = score_accuracy(res.map_network, true_net)
    print("\n" + "=" * 64)
    print(f"iters={args.iters} loci={args.loci} sites={args.sites}  "
          f"{ms_it:.2f} ms/it  wall={dt:.1f}s  acc={res.acceptance_rate:.3f}")
    print(f"MAP logP={res.map_log_posterior:.3f}  MAP reticulations="
          f"{acc.num_reticulations} (true 1)")
    print(f"reticulation posterior (sampled): {post}")
    print(f"first sampled reticulation at iter: {first_retic_it}")
    if acc.gamma_major is not None:
        print(f"MAP major gamma: {acc.gamma_major:.3f} (true {TRUE_GAMMA_MAJOR})")
    else:
        print("MAP major gamma: (no reticulation in MAP)")
    print(f"backbone clades recovered: {acc.all_clades_recovered}")
    print(f"MAP net: {res.map_network.newick()}")


if __name__ == "__main__":
    main()
