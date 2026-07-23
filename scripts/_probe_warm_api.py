"""End-to-end check of the built-in MCMC_SEQ warm_start=True path on the
canonical 6-taxon / 1-reticulation harness data."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from mcmc_harness import build_true_network, MAPPING, TRUE_GAMMA_MAJOR, score_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", type=int, default=15)
    ap.add_argument("--sites", type=int, default=600)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--gt-iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    true_net = build_true_network()
    data = simulate_multilocus(true_net, MAPPING, n_loci=a.loci,
                               seq_length=a.sites, theta=0.02, model=JC69(),
                               seed=a.seed)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=2))
    t0 = time.perf_counter()
    res = sampler.search(num_iter=a.iters, burn_in=a.iters // 4, sample_freq=30,
                         seed=a.seed, progress=False, warm_start=True,
                         warm_start_kwargs={"gt_iters": a.gt_iters})
    dt = time.perf_counter() - t0
    acc = score_accuracy(res.map_network, true_net)
    retic = [s.num_reticulations for s in res.samples]
    held = sum(1 for r in retic if r >= 1) / max(1, len(retic))
    print(f"warm_start=True  {dt:.1f}s  {1000*dt/a.iters:.1f} ms/it (incl bootstrap)")
    print(f"MAP reticulations={acc.num_reticulations} (true 1)  held_frac={held:.2f}")
    print(f"MAP major gamma: {acc.gamma_major} (true {TRUE_GAMMA_MAJOR})")
    print(f"backbone clades recovered: {acc.all_clades_recovered}")
    print(f"MAP logP={res.map_log_posterior:.2f}")
    print(f"MAP net: {res.map_network.newick()}")


if __name__ == "__main__":
    main()
