"""Isolate WHAT blocks reticulation discovery in MCMC_SEQ.

Three starts, all on the canonical 6-taxon/1-reticulation data:
  (A) true gene trees + TRUE reticulated net  -> baseline mode height.
  (B) true gene trees + species TREE net      -> can add discover from good GTs?
  (C) adapted gene trees + species TREE net    -> the real cold-start regime.

For each we report the network-only score gap (true retic vs tree) and, for
(B)/(C), the best coupled-add log-accept over many tries.  This tells us
whether good gene-tree *times* are the missing ingredient (if B works but C
doesn't, the coupling must re-propose times, not just topology).
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy import _mcmc_seq as ms
from mcmc_harness import build_true_network, MAPPING


def best_add(state, cur, tries=400, seed=1):
    rng = np.random.default_rng(seed)
    best = -1e9
    succ = 0
    for _ in range(tries):
        res = ms.op_add_reticulation_coupled(state, rng)
        if res is None:
            continue
        loghr, undo = res
        prop = state.log_posterior()
        if math.isfinite(prop):
            best = max(best, (prop - cur) + loghr)
            succ += 1
        undo()
    return best, succ


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", type=int, default=8)
    ap.add_argument("--sites", type=int, default=300)
    a = ap.parse_args()
    true_net = build_true_network()
    print(f"--- loci={a.loci} sites={a.sites} ---")
    data = simulate_multilocus(true_net, MAPPING, n_loci=a.loci, seq_length=a.sites,
                               theta=0.02, model=JC69(), seed=12345)
    kwargs = data.to_mcmc_seq_kwargs()
    true_gts = list(data.gene_trees)

    # (A) true gene trees + true net
    s = MCMC_SEQ(**{**kwargs, "gene_trees": [g for g in true_gts],
                    "species_net": true_net},
                 priors=MCMCSeqPriors(max_reticulations=2))
    stA = s._new_state()
    postA = stA.log_posterior()
    print(f"(A) true GTs + TRUE net : logP={postA:.3f}  "
          f"retic={stA.num_reticulations()}")

    # (B) true gene trees + species tree
    sB = MCMC_SEQ(**{**kwargs, "gene_trees": [g for g in true_gts]},
                  priors=MCMCSeqPriors(max_reticulations=2))
    stB = sB._new_state()
    postB = stB.log_posterior()
    gapB = ms._network_only_log_score(stB, true_net) - \
        ms._network_only_log_score(stB, stB.species_net)
    bestB, succB = best_add(stB, postB, seed=2)
    print(f"(B) true GTs + TREE net : logP={postB:.3f}  "
          f"net-only gap(true-tree)={gapB:.3f}  best coupled-add log-accept={bestB:.2f} "
          f"(succ {succB})")

    # (C) adapted gene trees + species tree (real cold start after burn-in)
    stC = sB._new_state()
    rng = np.random.default_rng(7)
    kernel = ms.MCMCSeqKernel(rng, max_reticulations=2)
    kernel._ops = [(op, w) for (op, w) in kernel._ops
                   if op not in (ms.op_add_reticulation_coupled,
                                 ms.op_delete_reticulation_coupled)]
    kernel._weights = np.asarray([w for _, w in kernel._ops], float)
    kernel._weights /= kernel._weights.sum()
    curC = stC.log_posterior()
    for _ in range(3000):
        prop = kernel.propose(stC)
        if prop is None:
            continue
        loghr, undo = prop
        p = stC.log_posterior()
        if math.isfinite(p) and math.log(rng.random()) < (p - curC) + loghr:
            curC = p
        else:
            undo()
    gapC = ms._network_only_log_score(stC, true_net) - \
        ms._network_only_log_score(stC, stC.species_net)
    bestC, succC = best_add(stC, curC, seed=3)
    print(f"(C) adapted GTs + TREE  : logP={curC:.3f}  "
          f"net-only gap(true-tree)={gapC:.3f}  best coupled-add log-accept={bestC:.2f} "
          f"(succ {succC})")


if __name__ == "__main__":
    main()
