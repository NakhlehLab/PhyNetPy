"""
Run MPL simulated annealing on the 20-taxa benchmark (1000 gene trees).

Data (under tests/testfiles/):
  mpl_20taxa_gt.txt — gene trees, one Newick string per line

Gene trees in the file may contain many leaves; ``IO.read_newick_file`` is
called with ``restrict_to_taxa=...`` so each line is parsed and then reduced to
the species labels in ``SPECIES_TO_ALLELES`` (via
``GraphUtils.induced_subnetwork_by_taxa`` inside I/O).

The starting topology is a **majority-rule consensus tree** from
``GeneTrees.build_majority_rule_consensus_tree()``. Counter-intuitively, the
polytomy-rich consensus is a *better* MPL search seed than a single binary
gene tree: polytomies let the kernel's early moves do informed resolution
across multiple compatible topologies (guided by triplet frequencies) while
T is still high, reaching basins that an already-committed binary tree
can't easily NNI/SPR its way to. The seed's raw MPL score is awful (-10^8
range) but that collapses within ~400 iterations to values better than any
binary seed reaches in the same budget.

If you want a binary seed instead (e.g. for starting-tree sensitivity
studies or ASTRAL-seeded pipelines), use
``GeneTrees.most_frequent_gene_tree(restrict_to_taxa=set(SPECIES_LABELS))``.

The first run spends noticeable time building triplet frequencies (rho) from
the gene trees; the search itself follows.

Run from anywhere:

  python examples/mpl_20taxa_search_demo.py
"""

from __future__ import annotations

from pathlib import Path

import phynetpy.IO as io
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL import MPL


#TODO: CHANGE PATHS FOR YOUR OWN FILE SYSTEM!!

# Repository root → stable paths to the bundled test files
_REPO = Path(__file__).resolve().parent.parent
GENE_TREES_FILE = _REPO / "tests" / "testfiles" / "mpl_20taxa_gt.txt"

# Cached consensus seed tree.  The consensus construction iterates through
# the Network's internal node/edge sets, which are hash-ordered (i.e.
# memory-address-ordered), so two in-process calls to
# ``build_majority_rule_consensus_tree`` produce structurally identical
# trees whose child ordering differs.  Writing the newick once and
# reloading it for subsequent runs pins the iteration order everywhere
# downstream, which is required for cross-process seed reproducibility.
CONSENSUS_CACHE = _REPO / "runs" / "mpl_20taxa_consensus_seed.nwk"

# Eighteen species leaves; one allele label per species (identity map)
SPECIES_LABELS = [
    "t1", "t4", "t15", "t36", "t38", "t43", "t49", "t52",
    "t74", "t83", "t84", "t85", "t94", "t109", "t111", "t123", "t124", "t130",
]
SPECIES_TO_ALLELES = {s: [s] for s in SPECIES_LABELS}

# Raise for full runs (e.g. 50_000); keep moderate for smoke tests.
SEARCH_ITERATIONS = 65000

# RNG seed — pin for reproducible SA trajectories across tuning runs.
SA_SEED = 1729

# SA progress: print every N iterations (0 = off). Shown lines use flush for logs.
SA_PROGRESS_EVERY = 1000

# Geometric cooling + multi-signal reheating (``schedule="geometric_reheat"``).
# ``t_end`` is the floor T (T_min) when ``t_min`` is omitted in ``SimulatedAnnealing``.
# Reheat triggers (in priority): rate-based plateau (primary), strict-stall
# backstop, and optional frozen-chain (zero uphill moves in a window).
SA_SCHEDULE = "geometric_reheat"
SA_T_START = 125           # typical |delta log-PL| is O(1-100); 250 was near-uniform-random
SA_T_END = 0.05
SA_COOLING_ALPHA = 0.95       # gentler geometric cool so we don't freeze early
SA_STEPS_PER_TEMP = 150
SA_REHEAT_THRESHOLD = 1200    # strict-stall backstop only (rarely primary)
SA_REHEAT_FACTOR = 1.8
SA_REHEAT_WINDOW = 500        # iterations per plateau-detection window
SA_REHEAT_MIN_IMPROVE = 1.0   # log-PL units; re-tune after one calibration run
SA_REHEAT_ON_NO_UPHILL = True # also reheat when a window accepts zero uphill moves
SA_REHEAT_CAP_MULT = 1.0      # keep classic cap at t_start (raise to 1.5 for hotter escapes)
SA_REHEAT_MAX_CONSECUTIVE = 4 # cascade guard: suspend reheats after N without improvement


def load_gene_trees(path: Path) -> GeneTrees:
    """Load gene trees restricted to ``SPECIES_TO_ALLELES`` (see ``restrict_to_taxa``)."""
    return io.read_newick_file(
        path,
        return_type="genetrees",
        species_gene_mapping=SPECIES_TO_ALLELES,
        restrict_to_taxa=SPECIES_LABELS,
        min_leaves_after_restrict=3,
    )


def main() -> None:
    print("Loading gene trees (pruned to species-mapping taxa)…", flush=True)
    gene_trees = load_gene_trees(GENE_TREES_FILE)
    print(f"  {len(gene_trees.trees)} trees after pruning", flush=True)

    if CONSENSUS_CACHE.exists():
        print(
            f"Loading cached consensus seed from {CONSENSUS_CACHE.relative_to(_REPO)}…",
            flush=True,
        )
        start_net = io.read_newick(CONSENSUS_CACHE.read_text(encoding="utf-8").strip())
    else:
        print(
            "Starting species tree (majority-rule consensus from pruned gene trees)…",
            flush=True,
        )
        start_net = gene_trees.build_majority_rule_consensus_tree()
        CONSENSUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CONSENSUS_CACHE.write_text(start_net.newick() + "\n", encoding="utf-8")
        print(
            f"  Cached seed newick -> {CONSENSUS_CACHE.relative_to(_REPO)} "
            "(delete the file to regenerate the consensus from scratch).",
            flush=True,
        )
    leaves = sorted(n.label for n in start_net.get_leaves())
    print(
        f"  Seed leaves ({len(leaves)}): same species set as mapping; "
        "no extra prune step needed.",
        flush=True,
    )
    print("  Seed tree (Newick):", flush=True)
    print(start_net.newick(), flush=True)
    print(flush=True)

    print("MPL score of seed (before search)…", flush=True)
    mpl = MPL(start_net, gene_trees, SPECIES_TO_ALLELES)
    start_pl = mpl.score()
    print(f"  log pseudo-likelihood: {start_pl:.6f}", flush=True)
    print(
        "  (Low vs later SA values is expected: the consensus has polytomies "
        "that tank raw MPL. The kernel resolves them in the first ~400 "
        "iterations, after which the search proceeds normally.)",
        flush=True,
    )
    print(flush=True)

    print(f"MPL simulated annealing ({SEARCH_ITERATIONS:,} moves)…", flush=True)
    best_log_pl = mpl.search(
        method="sa",
        num_iter=SEARCH_ITERATIONS,
        max_reticulations=2,
        seed=SA_SEED,
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

    print(f"\nBest log pseudo-likelihood: {best_log_pl:.6f}", flush=True)
    print("\nBest network (Newick):", flush=True)
    print(mpl.net.newick(), flush=True)


if __name__ == "__main__":
    main()
