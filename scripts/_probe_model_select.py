"""AIC/BIC model selection across reticulation counts, done the rigorous way.

A single co-estimation chain tends to sit at whatever reticulation count it was
seeded near, so grouping *its* samples by count rarely spans the models we want
to compare.  The standard remedy is to fit the model at each fixed complexity
(cap the reticulation count) and compare the fits by an information criterion
that penalises parameters.  This probe runs MCMC_SEQ at caps 0, 1, 2 on one
simulated data set and prints the AIC/BIC table so we can see whether the extra
reticulation earns its keep (truth = 1).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy._mcmc_seq import _count_free_parameters, _information_criteria
from mcmc_harness import build_true_network, MAPPING


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", type=int, default=15)
    ap.add_argument("--sites", type=int, default=600)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--gt-iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    true_net = build_true_network()
    data = simulate_multilocus(true_net, MAPPING, n_loci=a.loci,
                               seq_length=a.sites, theta=0.02, model=JC69(),
                               seed=a.seed)

    rows = []
    for cap in (0, 1, 2):
        sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                           priors=MCMCSeqPriors(max_reticulations=cap))
        t0 = time.perf_counter()
        # Warm start only helps when reticulations are allowed.
        res = sampler.search(num_iter=a.iters, burn_in=a.iters // 4,
                             sample_freq=25, seed=a.seed,
                             warm_start=(cap >= 1),
                             warm_start_kwargs={"gt_iters": a.gt_iters})
        dt = time.perf_counter() - t0
        r = sum(1 for v in res.map_network.V() if v.is_reticulation())
        k = _count_free_parameters(res.num_leaves, r)
        ic = _information_criteria(res.map_log_likelihood, k, res.total_sites)
        rows.append((cap, r, res.map_log_likelihood, k, ic["AIC"], ic["BIC"], dt))
        print(f"[cap {cap}] {dt:5.1f}s  MAP r={r}  logL={res.map_log_likelihood:.2f}"
              f"  k={k}  AIC={ic['AIC']:.1f}  BIC={ic['BIC']:.1f}")

    print("\n  cap  MAP_r     logL     k       AIC      dAIC       BIC      dBIC")
    min_aic = min(x[4] for x in rows)
    min_bic = min(x[5] for x in rows)
    for cap, r, ll, k, aic, bic, _dt in rows:
        print(f"  {cap:>3}  {r:>5}  {ll:>9.2f}  {k:>3}  {aic:>9.1f}  "
              f"{aic - min_aic:>8.1f}  {bic:>9.1f}  {bic - min_bic:>8.1f}")
    best_aic = min(rows, key=lambda x: x[4])
    best_bic = min(rows, key=lambda x: x[5])
    print(f"\n  AIC prefers cap={best_aic[0]} (MAP r={best_aic[1]});  "
          f"BIC prefers cap={best_bic[0]} (MAP r={best_bic[1]});  truth r=1")


if __name__ == "__main__":
    main()
