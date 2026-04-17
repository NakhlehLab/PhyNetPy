"""
Run MPL simulated annealing on the 20-taxa benchmark (1000 gene trees).

Data (under tests/testfiles/):
  mpl_20taxa_gt.txt — gene trees, one Newick string per line

Gene trees in the file may contain many leaves; ``IO.read_newick_file`` is
called with ``restrict_to_taxa=...`` so each line is parsed and then reduced to
the species labels in ``SPECIES_TO_ALLELES`` (via
``GraphUtils.induced_subnetwork_by_taxa`` inside I/O). 

The starting topology is
then a **majority-rule consensus tree** from
``GeneTrees.build_majority_rule_consensus_tree()``. That consensus optimizes
greedy cluster compatibility (frequency ≥ 0.5), **not** MPL, so its log
pseudo-likelihood is often poor until search improves the topology.

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

# Eighteen species leaves; one allele label per species (identity map)
SPECIES_LABELS = [
    "t1", "t4", "t15", "t36", "t38", "t43", "t49", "t52",
    "t74", "t83", "t84", "t85", "t94", "t109", "t111", "t123", "t124", "t130",
]
SPECIES_TO_ALLELES = {s: [s] for s in SPECIES_LABELS}

# Raise for full runs (e.g. 50_000); keep moderate for smoke tests.
SEARCH_ITERATIONS = 2000

# SA progress: print every N iterations (0 = off). Shown lines use flush for logs.
SA_PROGRESS_EVERY = 50

# Geometric cooling + reheating (``schedule="geometric_reheat"``). ``t_end`` is
# the floor T (T_min) when ``t_min`` is omitted in ``SimulatedAnnealing``.
SA_SCHEDULE = "geometric_reheat"
SA_T_START = 250.0
SA_T_END = 0.05
SA_COOLING_ALPHA = 0.93
SA_STEPS_PER_TEMP = 100
SA_REHEAT_THRESHOLD = 1000
SA_REHEAT_FACTOR = 2.0


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

    print("Starting species tree (majority-rule consensus from pruned gene trees)…", flush=True)
    start_net = gene_trees.build_majority_rule_consensus_tree()
    leaves = sorted(n.label for n in start_net.get_leaves())
    print(
        f"  Consensus leaves ({len(leaves)}): same species set as mapping; "
        "no extra prune step needed.",
        flush=True,
    )
    print("  Consensus tree (Newick):", flush=True)
    print(start_net.newick(), flush=True)
    print(flush=True)

    print("MPL score of consensus (before search)…", flush=True)
    mpl = MPL(start_net, gene_trees, SPECIES_TO_ALLELES)
    start_pl = mpl.score()
    print(f"  log pseudo-likelihood: {start_pl:.6f}", flush=True)
    print(
        "  (Low vs later SA values is expected: consensus is not the MPL optimum.)",
        flush=True,
    )
    print(flush=True)

    print(f"MPL simulated annealing ({SEARCH_ITERATIONS:,} moves)…", flush=True)
    best_log_pl = mpl.search(
        method="sa",
        num_iter=SEARCH_ITERATIONS,
        max_reticulations=2,
        progress_every=SA_PROGRESS_EVERY,
        schedule=SA_SCHEDULE,
        t_start=SA_T_START,
        t_end=SA_T_END,
        cooling_alpha=SA_COOLING_ALPHA,
        steps_per_temp=SA_STEPS_PER_TEMP,
        reheat_threshold=SA_REHEAT_THRESHOLD,
        reheat_factor=SA_REHEAT_FACTOR,
    )

    print(f"\nBest log pseudo-likelihood: {best_log_pl:.6f}", flush=True)
    print("\nBest network (Newick):", flush=True)
    print(mpl.net.newick(), flush=True)


if __name__ == "__main__":
    main()
