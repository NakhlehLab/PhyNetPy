"""Compare adaptive vs uniform proposal kernel over 5000 iterations.

Usage (from repo root):
    python -m tests.test_adaptive_kernel
"""

import copy
import time

from phynetpy.MPL import MPL, MPLKernel
from phynetpy.IO import read_newick_file
from phynetpy.MetropolisHastings import HillClimbing
from phynetpy.ModelGraph import Model
from phynetpy.State import State

GT_FILE = "tests/testfiles/subgeneset_3_ret1.txt"
TAXA = ["t14", "t15", "t49", "t68", "t69", "t72", "t75", "t91", "t114", "t133"]
MAPPING = {t: [t] for t in TAXA}
NUM_ITER = 2000
MAX_RETICS = 2
LOG_INTERVAL = 500
SEED = 42


def build_starting_mpl(seed: int = SEED) -> MPL:
    """Load gene trees and build a consensus starting tree."""
    import random as _rng
    _rng.seed(seed)
    gts = read_newick_file(GT_FILE, return_type="genetrees",
                           species_gene_mapping=MAPPING)
    start_tree = gts.build_majority_rule_consensus_tree()
    return MPL(start_tree, gts, MAPPING)


def run_instrumented_search(label: str, mpl: MPL,
                            kernel: MPLKernel) -> float:
    """Run HC manually so we can log weights at intervals."""
    from phynetpy.MPL import MPLScorer

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    start_score = mpl.score()
    print(f"  Starting log-PL: {start_score:.4f}")

    scorer = MPLScorer(mpl._rho, mpl._triplets)
    model = Model()
    model.network = copy.deepcopy(mpl.net)
    model.set_likelihood_calculator(scorer)

    state = State(model)
    cached_cur = None
    best_score = start_score
    n_accepted = 0
    n_rejected = 0
    n_invalid = 0
    invalid_by_type: dict = {}
    accepted_by_type: dict = {}

    t0 = time.perf_counter()

    for i in range(1, NUM_ITER + 1):
        move = kernel.generate()
        move_name = type(move).__name__
        is_valid = state.generate_next(move)

        if is_valid:
            try:
                if cached_cur is None:
                    cached_cur = state.likelihood()
                cur = cached_cur
                proposed = state.proposed().likelihood()
            except Exception:
                try:
                    state.revert(move)
                except Exception:
                    pass
                n_invalid += 1
                invalid_by_type[move_name] = invalid_by_type.get(move_name, 0) + 1
                continue

            if proposed > cur:
                state.commit(move)
                cached_cur = proposed
                n_accepted += 1
                accepted_by_type[move_name] = accepted_by_type.get(move_name, 0) + 1
                kernel.report_outcome(True, delta=proposed - cur)
                if proposed > best_score:
                    best_score = proposed
            else:
                state.revert(move)
                n_rejected += 1
                kernel.report_outcome(False, delta=proposed - cur)
        else:
            n_invalid += 1
            invalid_by_type[move_name] = invalid_by_type.get(move_name, 0) + 1

        if i % LOG_INTERVAL == 0:
            elapsed = time.perf_counter() - t0
            total_decided = n_accepted + n_rejected
            acc_pct = (100.0 * n_accepted / total_decided
                       if total_decided else 0.0)
            print(f"  iter {i:5d} | best {best_score:12.1f} | "
                  f"acc {acc_pct:5.1f}% ({n_accepted}/{total_decided}) | "
                  f"invalid {n_invalid} | {elapsed:.1f}s")
            if hasattr(kernel, 'get_weights'):
                wts = kernel.get_weights()
                top3 = sorted(wts.items(), key=lambda x: -x[1])[:3]
                parts = [f"{n}={w:.3f}" for n, w in top3]
                print(f"           top weights: {', '.join(parts)}")

    elapsed = time.perf_counter() - t0
    print(f"  DONE | best {best_score:.4f} | time {elapsed:.1f}s")

    if hasattr(kernel, 'get_weights'):
        print(f"  Final weights:")
        for name, w in kernel.get_weights().items():
            print(f"    {name:30s} {w:.4f}")

    if invalid_by_type:
        print(f"  Invalids by type:")
        for name, ct in sorted(invalid_by_type.items(),
                                key=lambda x: -x[1]):
            print(f"    {name:30s} {ct}")

    if accepted_by_type:
        print(f"  Accepted by type:")
        for name, ct in sorted(accepted_by_type.items(),
                                key=lambda x: -x[1]):
            print(f"    {name:30s} {ct}")

    return best_score


def main():
    print("Loading data and building consensus starting tree ...")
    mpl_adaptive = build_starting_mpl()
    mpl_uniform  = build_starting_mpl()

    adaptive_kernel = MPLKernel(max_reticulations=MAX_RETICS,
                                adaptive=True, window_size=50,
                                min_weight=0.02)

    uniform_weights = [1.0] * 8
    uniform_kernel  = MPLKernel(max_reticulations=MAX_RETICS,
                                weights=uniform_weights)

    import random as _rng
    _rng.seed(SEED + 1)
    score_adaptive = run_instrumented_search("ADAPTIVE kernel",
                                             mpl_adaptive,
                                             adaptive_kernel)
    _rng.seed(SEED + 1)
    score_uniform  = run_instrumented_search("UNIFORM  kernel",
                                             mpl_uniform,
                                             uniform_kernel)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Adaptive best: {score_adaptive:.4f}")
    print(f"  Uniform  best: {score_uniform:.4f}")
    diff = score_adaptive - score_uniform
    print(f"  Difference:    {diff:+.4f}  "
          f"({'adaptive wins' if diff > 0 else 'uniform wins'})")


if __name__ == "__main__":
    main()
