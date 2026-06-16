"""Smoke test for run_parallel_chains: 4 chains + a monitor that pauses,
resumes, then halts.  Verifies the programmatic control path end to end.
"""
import numpy as np

from phynetpy.BirthDeath import CBDP
from phynetpy.infer import (
    MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus,
    run_parallel_chains, MultiChainStatus,
)


def build_sampler():
    np.random.seed(0)
    net = CBDP(1.0, 0.5, 5).generate_network()
    leaves = sorted(n.label for n in net.get_leaves())
    mapping = {l: [l] for l in leaves}
    data = simulate_multilocus(
        net, mapping, n_loci=4, seq_length=300, theta=0.02,
        model=JC69(), seed=7,
    )
    return MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                    priors=MCMCSeqPriors(max_reticulations=2))


def main():
    sampler = build_sampler()

    calls = {"n": 0}

    def monitor(status: MultiChainStatus):
        calls["n"] += 1
        n = calls["n"]
        if n in (2, 3):
            print(f">>> MONITOR call {n}: PAUSE")
            return "pause"
        if n == 4:
            print(f">>> MONITOR call {n}: RESUME")
            return "continue"
        total = sum(st["n_samples"] for st in status.per_chain.values())
        if n >= 8:
            print(f">>> MONITOR call {n}: STOP "
                  f"(pooled_samples={total}, rhat={status.rhat})")
            return "stop"
        return "continue"

    result = run_parallel_chains(
        sampler,
        n_chains=4,
        num_iter=40_000,
        burn_in=2_000,
        sample_freq=50,
        seed=2024,
        check_every=1_000,
        monitor=monitor,
        progress=True,
        poll_interval=0.5,
    )

    print("\n================ RESULT ================")
    print(f"chains returned : {len(result.chains)}/{result.n_chains}")
    print(f"stopped_early   : {result.stopped_early}")
    print(f"errors          : {result.errors}")
    print(f"wall_time_sec   : {result.wall_time_sec:.1f}")
    for i, r in enumerate(result.chains):
        print(f"  chain {i}: iters={r.num_iterations} "
              f"samples={len(r.samples)} MAP={r.map_log_posterior:.2f} "
              f"acc={r.acceptance_rate:.3f}")
    print(f"final R-hat     : "
          + ", ".join(f"{k}={v:.3f}" for k, v in result.rhat.items()))
    best = result.best()
    if best is not None:
        print(f"best MAP logP   : {best.map_log_posterior:.2f}")


if __name__ == "__main__":
    main()
