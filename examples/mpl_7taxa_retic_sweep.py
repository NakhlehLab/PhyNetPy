"""
Reticulation-count sweep for the 7-taxa MPL tuning demo.

Runs the same SA schedule as ``mpl_7taxa_tune_demo`` for
``max_reticulations`` in ``K_VALUES``, optionally repeating each ``k``
across a small bank of seeds, and reports the log-PL / AIC / BIC /
elbow summary.  The CSV and PNG outputs are written to ``runs/``.

This is a thin wrapper around the generic :func:`reticulation_sweep`
utility in ``phynetpy.ModelSelection``; the same pattern can be reused
by any search method that accepts a ``max_reticulations`` knob and
returns the best log-likelihood it found.

Run from anywhere::

    python examples/mpl_7taxa_retic_sweep.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import phynetpy.IO as io
from phynetpy.ModelSelection import reticulation_sweep
from phynetpy.criteria import PseudoLikelihood
from phynetpy.data import GeneTrees
from phynetpy.infer import infer
from phynetpy.models import MSC


_REPO = Path(__file__).resolve().parent.parent
GENE_TREES_FILE = _REPO / "tests" / "testfiles" / "mpl_20taxa_gt.txt"
CONSENSUS_CACHE = _REPO / "runs" / "mpl_7taxa_consensus_seed.nwk"
RUNS_DIR = _REPO / "runs"

SPECIES_LABELS = ["t1", "t43", "t85", "t83", "t15", "t36", "t52"]
SPECIES_TO_ALLELES = {s: [s] for s in SPECIES_LABELS}

# Sweep configuration -------------------------------------------------
K_VALUES = [0, 1, 2, 3]
SEEDS = [42, 7, 101]              # best-of-N per k

# Number of free parameters added per reticulation.  For MPL: one
# inheritance probability gamma + two new branch-length parameters on
# the inserted parent edges.  Use 1 for the "gamma only" convention.
PARAMS_PER_RETICULATION = 3

# Backbone parameter count for AIC/BIC.  An unrooted binary tree on
# ``n`` taxa has ``2n - 3`` internal branch lengths; we approximate
# that as the backbone cost.  Only differences across k matter for
# ranking, so the absolute value is not critical.
BASE_PARAMS = 2 * len(SPECIES_LABELS) - 3

# SA schedule (matches the 7-taxa tuning demo; see rationale there).
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
SA_PROGRESS_EVERY = 0  # quiet; the sweep driver prints summaries


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


def main() -> None:
    print("Loading gene trees (7-taxon subset)…", flush=True)
    gene_trees = load_gene_trees()
    n_trees = len(gene_trees.trees)
    print(f"  {n_trees} trees after pruning", flush=True)

    seed_template = load_or_build_seed(gene_trees)

    def run_k(k: int, seed: int) -> float:
        """Run one pseudo-likelihood SA search with max_reticulations=k."""
        return infer(
            gene_trees,
            model=MSC(),
            criterion=PseudoLikelihood(),
            start=copy.deepcopy(seed_template),
            method="sa",
            num_iter=SEARCH_ITERATIONS,
            max_reticulations=k,
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
        ).score

    print(
        f"\nSweeping k in {list(K_VALUES)} × seeds {list(SEEDS)} "
        f"({len(K_VALUES) * len(SEEDS)} total searches)\n",
        flush=True,
    )
    result = reticulation_sweep(
        run_k,
        k_values=K_VALUES,
        seeds=SEEDS,
        data_size=n_trees,
        params_per_reticulation=PARAMS_PER_RETICULATION,
        base_params=BASE_PARAMS,
        log_lik_label="log-pseudo-likelihood",
        progress=True,
    )

    print()
    result.print_summary()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RUNS_DIR / "mpl_7taxa_retic_sweep.csv"
    png_path = RUNS_DIR / "mpl_7taxa_retic_sweep.png"
    result.save_csv(csv_path)
    result.plot(
        png_path,
        title=(
            f"Reticulation sweep -- 7-taxa MPL "
            f"(n={n_trees} gene trees, "
            f"{len(SEEDS)} seeds/k)"
        ),
    )
    print(f"\nCSV  -> {csv_path.relative_to(_REPO)}")
    print(f"Plot -> {png_path.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
