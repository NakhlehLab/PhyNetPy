"""
Fast 7-taxa tuning harness for MPL simulated annealing.

Uses a 7-species subset of the 20-taxa benchmark's gene-tree file, so scoring
per iteration is ~5-10x faster than the full 18-taxa demo.  This makes it
cheap to iterate on kernel/SA tuning knobs: full 20k-50k runs finish in
a couple of minutes instead of 20+ minutes, and you can run many seeds
back-to-back to average out noise before committing a change.

Taxa were chosen to preserve structure across several root-level clades
of the full 18-taxa consensus:

  t1   — from ((t1,t111),(t130,t4))
  t43  — from ((t49,t43),(t85,t124),(t74,t94))
  t85  — from the same trio, different sister-pair
  t83  — from (t83,t38)
  t15  — singleton at root
  t36  — singleton at root
  t52  — singleton at root

The consensus seed is cached on first run (same strategy as the 18-taxa
demo) so child-ordering is pinned for reproducible SA trajectories.

Gene trees in the file may contain many leaves; ``IO.read_newick_file`` is
called with ``restrict_to_taxa=...`` so each line is parsed and then reduced
to the 7-taxon subset (via ``GraphUtils.induced_subnetwork_by_taxa`` inside
I/O).

Run from anywhere::

    python examples/mpl_7taxa_tune_demo.py
"""

from __future__ import annotations

from pathlib import Path

import phynetpy.IO as io
from phynetpy.criteria import PseudoLikelihood
from phynetpy.data import GeneTrees
from phynetpy.infer import infer, score
from phynetpy.models import MSC


_REPO = Path(__file__).resolve().parent.parent
GENE_TREES_FILE = _REPO / "tests" / "testfiles" / "mpl_20taxa_gt.txt"
CONSENSUS_CACHE = _REPO / "runs" / "mpl_7taxa_consensus_seed.nwk"

# Seven species leaves; identity map (one allele per species).
SPECIES_LABELS = ["t1", "t43", "t85", "t83", "t15", "t36", "t52"]
SPECIES_TO_ALLELES = {s: [s] for s in SPECIES_LABELS}

# Tuning harness defaults.  Short enough to iterate quickly; long enough
# for the SA trajectory to reach its cold-T regime and surface kernel
# diagnostics.
SEARCH_ITERATIONS = 20000

# Pin for reproducible SA trajectories across tuning runs; change when
# averaging across seeds.
SA_SEED = 1729
SA_PROGRESS_EVERY = 500

# Same schedule shape as the 18-taxa demo so tuning translates.  With
# 7 taxa the search converges faster, so ``steps_per_temp`` and the
# reheat windows are scaled down proportionally.
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


def load_gene_trees(path: Path) -> GeneTrees:
    """Load gene trees restricted to ``SPECIES_TO_ALLELES``."""
    networks = io.read_newick_file(
        path,
        return_type="networks",
        restrict_to_taxa=SPECIES_LABELS,
        min_leaves_after_restrict=3,
    )
    return GeneTrees(list(networks), SPECIES_TO_ALLELES)


def main() -> None:
    print("Loading gene trees (pruned to 7-taxon subset)…", flush=True)
    gene_trees = load_gene_trees(GENE_TREES_FILE)
    print(f"  {len(gene_trees.trees)} trees after pruning", flush=True)

    if CONSENSUS_CACHE.exists():
        print(
            f"Loading cached consensus seed from "
            f"{CONSENSUS_CACHE.relative_to(_REPO)}…",
            flush=True,
        )
        start_net = io.read_newick(
            CONSENSUS_CACHE.read_text(encoding="utf-8").strip()
        )
    else:
        print(
            "Building majority-rule consensus from pruned gene trees…",
            flush=True,
        )
        start_net = gene_trees.build_majority_rule_consensus_tree()
        CONSENSUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CONSENSUS_CACHE.write_text(
            start_net.newick() + "\n", encoding="utf-8"
        )
        print(
            f"  Cached seed newick -> "
            f"{CONSENSUS_CACHE.relative_to(_REPO)} "
            "(delete the file to regenerate).",
            flush=True,
        )

    leaves = sorted(n.label for n in start_net.get_leaves())
    print(f"  Seed leaves ({len(leaves)}): {leaves}", flush=True)
    print("  Seed tree (Newick):", flush=True)
    print(start_net.newick(), flush=True)
    print(flush=True)

    print("Pseudo-likelihood of seed (before search)…", flush=True)
    start_pl = score(
        start_net, gene_trees, model=MSC(), criterion=PseudoLikelihood(),
    )
    print(f"  log pseudo-likelihood: {start_pl:.6f}", flush=True)
    print(flush=True)

    print(f"Simulated annealing ({SEARCH_ITERATIONS:,} moves)…", flush=True)
    result = infer(
        gene_trees,
        model=MSC(),
        criterion=PseudoLikelihood(),
        start=start_net,
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

    print(f"\nBest log pseudo-likelihood: {result.score:.6f}", flush=True)
    print("\nBest network (Newick):", flush=True)
    print(result.best.newick(), flush=True)


if __name__ == "__main__":
    main()
