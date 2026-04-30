"""
Demo: Bayesian MCMC_GT on a small gene-tree dataset.

Uses the same 20-taxa benchmark as ``mpl_20taxa_search_demo.py`` but
seeds a :class:`MCMC_GT` chain instead of :class:`MPL` simulated
annealing.  The driver is Metropolis-Hastings; the search returns a
list of posterior samples plus the MAP network.

Starting topology: majority-rule consensus tree from
``GeneTrees.build_majority_rule_consensus_tree()`` — same pattern as
the MPL demo.  Polytomies produce a deliberately low starting
posterior; the kernel resolves them within the first few hundred
iterations.

For quick smoke testing the chain length is modest (10,000 iter,
2,000 burn-in).  For production runs bump ``NUM_ITER`` and
``BURN_IN`` substantially and consider running several seeds in
parallel to assess convergence.

Run from anywhere:

  python examples/mcmc_gt_demo.py
"""

from __future__ import annotations

from pathlib import Path

import phynetpy.IO as io
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MCMC_GT import MCMC_GT, MCMC_GTPriors


# TODO: CHANGE PATHS FOR YOUR OWN FILE SYSTEM!!

_REPO = Path(__file__).resolve().parent.parent
GENE_TREES_FILE = _REPO / "tests" / "testfiles" / "mpl_20taxa_gt.txt"
CONSENSUS_CACHE = _REPO / "runs" / "mpl_20taxa_consensus_seed.nwk"

# Small subset for fast smoke-testing.  Expand for real inference.
SPECIES_LABELS = ["t1", "t4", "t15", "t36", "t38", "t43", "t49"]
SPECIES_TO_ALLELES = {s: [s] for s in SPECIES_LABELS}

# Chain settings.  Keep modest for demo purposes; scale up for real
# inference and always run multiple chains with different seeds.
NUM_ITER = 10_000
BURN_IN = 2_000
THIN = 25
MH_SEED = 1729
MAX_RETICULATIONS = 2


def load_gene_trees(path: Path) -> GeneTrees:
    """Load and prune gene trees to the demo taxon subset."""
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

    # Seed tree: majority-rule consensus from the gene trees.  This
    # matches ``mpl_20taxa_search_demo.py``'s seeding pattern so a
    # direct MPL <-> MCMC_GT comparison is fair.
    if CONSENSUS_CACHE.exists():
        print(
            f"Loading cached consensus seed from "
            f"{CONSENSUS_CACHE.relative_to(_REPO)}…",
            flush=True,
        )
        start_net = io.read_newick(CONSENSUS_CACHE.read_text(encoding="utf-8").strip())
    else:
        print("Building majority-rule consensus seed from gene trees…", flush=True)
        start_net = gene_trees.build_majority_rule_consensus_tree()

    print("  Seed (Newick):", flush=True)
    print(start_net.newick(), flush=True)
    print(flush=True)

    # Bayesian hyperparameters.  All defaults = Wen & Nakhleh (2016).
    priors = MCMC_GTPriors(
        branch_length_rate=1.0,
        gamma_alpha=1.0,
        gamma_beta=1.0,
        retic_count_mean=1.0,
    )

    mcmc = MCMC_GT(start_net, gene_trees, SPECIES_TO_ALLELES, priors=priors)

    print("MCMC_GT score of seed (before search)…", flush=True)
    start_post = mcmc.score(posterior=True)
    print(f"  log posterior: {start_post:.6f}", flush=True)
    print(flush=True)

    print(
        f"MCMC_GT Metropolis-Hastings ({NUM_ITER:,} moves, "
        f"burn-in {BURN_IN:,}, thin {THIN})…",
        flush=True,
    )
    result = mcmc.search(
        method="mh",
        num_iter=NUM_ITER,
        burn_in=BURN_IN,
        thin=THIN,
        max_reticulations=MAX_RETICULATIONS,
        seed=MH_SEED,
    )

    print(f"\nBest log posterior (MAP): {result.best_log_posterior:.6f}", flush=True)
    print(f"Acceptance rate:           {result.acceptance_rate:.3%}", flush=True)
    print(f"Samples collected:         {len(result.samples)}", flush=True)
    print(f"Wall-clock time:           {result.wall_time_sec:.1f} s", flush=True)

    print("\nMAP network (Newick):", flush=True)
    print(result.best_network.newick(), flush=True)

    if result.samples:
        print("\nPosterior sample summary:", flush=True)
        top_k = min(5, len(result.samples))
        posteriors = [s.log_posterior for s in result.samples]
        print(
            f"  mean   log-posterior: {sum(posteriors) / len(posteriors):.6f}",
            flush=True,
        )
        print(f"  min    log-posterior: {min(posteriors):.6f}", flush=True)
        print(f"  max    log-posterior: {max(posteriors):.6f}", flush=True)
        print(f"  shown last {top_k} sample log-posteriors:", flush=True)
        for s in result.samples[-top_k:]:
            print(f"    iter {s.iteration}: {s.log_posterior:.6f}", flush=True)


if __name__ == "__main__":
    main()
