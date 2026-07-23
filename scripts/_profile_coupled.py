"""cProfile the coupled reticulation moves in isolation.

Seeds a SeqState that already carries a reticulation (so add/delete/relocate all
fire), then repeatedly invokes the coupled operators directly -- bypassing the
kernel's other moves -- so the profile is dominated by the code we want to slim
down.  Reports the top functions by cumulative and by total (self) time.
"""
import os, sys, cProfile, pstats, io, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy import _mcmc_seq as ms
from mcmc_harness import build_true_network, MAPPING


def build_state(loci, sites, seed):
    true_net = build_true_network()
    data = simulate_multilocus(true_net, MAPPING, n_loci=loci, seq_length=sites,
                               theta=0.02, model=JC69(), seed=seed)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       species_net=true_net,
                       priors=MCMCSeqPriors(max_reticulations=2))
    return sampler._new_state()


def exercise(state, n_calls, seed):
    """Fire coupled ops repeatedly, undoing each so the state stays valid."""
    rng = np.random.default_rng(seed)
    ops = [ms.op_relocate_reticulation_coupled,
           ms.op_delete_reticulation_coupled,
           ms.op_add_reticulation_coupled]
    done = 0
    tries = 0
    while done < n_calls and tries < n_calls * 20:
        tries += 1
        op = ops[tries % len(ops)]
        res = op(state, rng)
        if res is None:
            continue
        _hr, undo = res
        undo()  # keep state fixed; we only care about proposal cost
        done += 1
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", type=int, default=15)
    ap.add_argument("--sites", type=int, default=600)
    ap.add_argument("--calls", type=int, default=120)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sort", default="tottime", choices=["tottime", "cumtime"])
    a = ap.parse_args()

    state = build_state(a.loci, a.sites, a.seed)
    print(f"state ready: {a.loci} loci x {a.sites} sites, "
          f"reticulations={state.num_reticulations()}")

    # Warm up (JIT-y caches, imports) so the profile reflects steady state.
    exercise(state, 5, a.seed)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    done = exercise(state, a.calls, a.seed + 1)
    pr.disable()
    dt = time.perf_counter() - t0

    print(f"profiled {done} successful coupled ops in {dt:.2f}s "
          f"= {1000*dt/max(1,done):.1f} ms/op\n")

    for sort in ("tottime", "cumtime"):
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sort)
        ps.print_stats(22)
        print("=" * 72)
        print(f"TOP BY {sort}")
        print("=" * 72)
        # Trim pstats header noise; keep the table.
        for line in s.getvalue().splitlines():
            if line.strip():
                print(line)
        print()


if __name__ == "__main__":
    main()
