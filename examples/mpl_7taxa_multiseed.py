"""
Multi-seed runner for the 7-taxa MPL tuning demo.

Runs the same SA schedule as ``mpl_7taxa_tune_demo`` over a small bank
of seeds and tabulates final log-PL plus the inferred Newick for each.
Use it to sanity-check consistency of the search across RNG variation
after a kernel/move change -- if most seeds land on the same topology
and a narrow log-PL band, the fix is stabilising the search; if they
spray across the landscape, the bottleneck is still exploration.

Run::

    python examples/mpl_7taxa_multiseed.py
"""

from __future__ import annotations

import time
from pathlib import Path

import phynetpy.IO as io
from phynetpy.criteria import PseudoLikelihood
from phynetpy.data import GeneTrees
from phynetpy.infer import infer
from phynetpy.models import MSC


_REPO = Path(__file__).resolve().parent.parent
GENE_TREES_FILE = _REPO / "tests" / "testfiles" / "mpl_20taxa_gt.txt"
CONSENSUS_CACHE = _REPO / "runs" / "mpl_7taxa_consensus_seed.nwk"
LOG_DIR = _REPO / "runs"

SPECIES_LABELS = ["t1", "t43", "t85", "t83", "t15", "t36", "t52"]
SPECIES_TO_ALLELES = {s: [s] for s in SPECIES_LABELS}

# Seeds to compare. Five is enough to see mode structure without
# blowing the wall-clock budget.
SEEDS = [1729, 7, 42, 101, 2024]
SEARCH_ITERATIONS = 20000

SA_SCHEDULE = "geometric_reheat"
SA_T_START = 125
SA_T_END = 0.05
SA_COOLING_ALPHA = 0.95
SA_STEPS_PER_TEMP = 75
SA_REHEAT_THRESHOLD = 600
SA_REHEAT_FACTOR = 1.8
SA_REHEAT_WINDOW = 250
SA_REHEAT_MIN_IMPROVE = 1.0
SA_REHEAT_ON_NO_UPHILL = True
SA_REHEAT_CAP_MULT = 1.0
SA_REHEAT_MAX_CONSECUTIVE = 4
SA_PROGRESS_EVERY = 2000  # quieter per-run output; we print a summary below


def load_gene_trees() -> GeneTrees:
    networks = io.read_newick_file(
        GENE_TREES_FILE,
        return_type="networks",
        restrict_to_taxa=SPECIES_LABELS,
        min_leaves_after_restrict=3,
    )
    return GeneTrees(list(networks), SPECIES_TO_ALLELES)


def load_or_build_seed(gene_trees: GeneTrees):
    if CONSENSUS_CACHE.exists():
        return io.read_newick(
            CONSENSUS_CACHE.read_text(encoding="utf-8").strip()
        )
    start_net = gene_trees.build_majority_rule_consensus_tree()
    CONSENSUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CONSENSUS_CACHE.write_text(start_net.newick() + "\n", encoding="utf-8")
    return start_net


def run_one_seed(seed: int, gene_trees: GeneTrees):
    start_net = load_or_build_seed(gene_trees)
    t0 = time.time()
    result = infer(
        gene_trees,
        model=MSC(),
        criterion=PseudoLikelihood(),
        start=start_net,
        method="sa",
        num_iter=SEARCH_ITERATIONS,
        max_reticulations=2,
        seed=seed,
        progress_every=SA_PROGRESS_EVERY,
        schedule=SA_SCHEDULE,
        t_start=SA_T_START,
        t_end=SA_T_END,
        cooling_alpha=SA_COOLING_ALPHA,
        steps_per_temp=SA_STEPS_PER_TEMP,
        reheat_threshold=SA_REHEAT_THRESHOLD,
        reheat_factor=SA_REHEAT_FACTOR,
        reheat_window=SA_REHEAT_WINDOW,
        reheat_min_improve=SA_REHEAT_MIN_IMPROVE,
        reheat_on_no_uphill=SA_REHEAT_ON_NO_UPHILL,
        reheat_cap_mult=SA_REHEAT_CAP_MULT,
        reheat_max_consecutive=SA_REHEAT_MAX_CONSECUTIVE,
    )
    elapsed = time.time() - t0
    return result.score, result.best.newick(), elapsed


def main() -> None:
    print("Loading gene trees (7-taxon subset)…", flush=True)
    gene_trees = load_gene_trees()
    print(f"  {len(gene_trees.trees)} trees\n", flush=True)

    results = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"=== Run {i}/{len(SEEDS)}: seed={seed} ===", flush=True)
        best_pl, newick, elapsed = run_one_seed(seed, gene_trees)
        print(
            f"  final log-PL: {best_pl:.4f}   ({elapsed:.1f}s)",
            flush=True,
        )
        print(f"  newick: {newick}\n", flush=True)
        results.append((seed, best_pl, newick, elapsed))

    print("=" * 70)
    print("Summary across seeds")
    print("=" * 70)
    print(f"{'seed':>8} {'log-PL':>16} {'elapsed(s)':>12}")
    for seed, best_pl, _nwk, elapsed in results:
        print(f"{seed:>8} {best_pl:>16.4f} {elapsed:>12.1f}")

    scores = [r[1] for r in results]
    best, worst = max(scores), min(scores)
    mean = sum(scores) / len(scores)
    print(
        f"\nbest={best:.4f}  worst={worst:.4f}  spread={best-worst:.4f}  "
        f"mean={mean:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
