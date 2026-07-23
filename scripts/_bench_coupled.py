"""Deterministic wall-clock benchmark of the coupled moves (no cProfile).

Runs a FIXED number of successful operations of each type from the SAME seed so
runs are directly comparable, and can A/B the clone strategy (fast
``Network.copy`` vs generic ``copy.deepcopy``) to isolate that contribution.
Reports ms per successful op for add / delete / relocate and the total.
"""
import os, sys, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy import _mcmc_seq as ms
from mcmc_harness import build_true_network, MAPPING


def build_state(loci, sites, seed):
    tn = build_true_network()
    data = simulate_multilocus(tn, MAPPING, n_loci=loci, seq_length=sites,
                               theta=0.02, model=JC69(), seed=seed)
    s = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), species_net=tn,
                 priors=MCMCSeqPriors(max_reticulations=2))
    return s._new_state()


def time_op(state, op, n_success, seed):
    """Time n_success successful applications of one op (each undone)."""
    rng = np.random.default_rng(seed)
    done = 0
    tries = 0
    t = 0.0
    while done < n_success and tries < n_success * 40:
        tries += 1
        t0 = time.perf_counter()
        res = op(state, rng)
        dt = time.perf_counter() - t0
        if res is None:
            continue
        _hr, undo = res
        undo()
        t += dt
        done += 1
    return t, done


def run_suite(state, n, seed):
    out = {}
    for name, op in [("add", ms.op_add_reticulation_coupled),
                     ("delete", ms.op_delete_reticulation_coupled),
                     ("relocate", ms.op_relocate_reticulation_coupled)]:
        t, done = time_op(state, op, n, seed + hash(name) % 1000)
        out[name] = (t, done)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", type=int, default=15)
    ap.add_argument("--sites", type=int, default=600)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    fast_clone = ms._clone_net

    def slow_clone(net):
        return copy.deepcopy(net)

    for tag, clone_fn in [("FAST (Network.copy)", fast_clone),
                          ("SLOW (copy.deepcopy)", slow_clone)]:
        ms._clone_net = clone_fn
        state = build_state(a.loci, a.sites, a.seed)
        # warmup
        run_suite(state, 3, a.seed)
        res = run_suite(state, a.n, a.seed)
        print(f"\n=== clone strategy: {tag} ===")
        tot_t = tot_n = 0
        for name in ("add", "delete", "relocate"):
            t, done = res[name]
            tot_t += t
            tot_n += done
            ms_op = 1000 * t / max(1, done)
            print(f"  {name:<9} {done:>3} ops  {ms_op:8.2f} ms/op")
        print(f"  {'TOTAL':<9} {tot_n:>3} ops  "
              f"{1000*tot_t/max(1,tot_n):8.2f} ms/op")

    ms._clone_net = fast_clone


if __name__ == "__main__":
    main()
