"""
Demo: Bayesian inference from gene trees.

Uses the same 20-taxa benchmark as ``mpl_20taxa_search_demo.py`` but runs the
Bayesian criterion instead of maximum pseudo-likelihood.  The driver is
Metropolis-Hastings; the result carries the MAP network plus the retained
posterior sample.

The only thing that changes between this demo and the pseudo-likelihood one is
the ``criterion`` argument -- that is the point of the two-verb API.

Starting topology: majority-rule consensus tree from
``GeneTrees.build_majority_rule_consensus_tree()``, passed as ``start=``.
Polytomies produce a deliberately low starting posterior; the kernel resolves
them within the first few hundred iterations.

The chain length below is a smoke test, not an analysis.  The full MSNC
likelihood over 1,000 gene trees costs roughly half a second per iteration, so
the 1,000 iterations below take about nine minutes and a production chain
(10^5-10^6 iterations) is days.  Bump ``NUM_ITER`` and ``BURN_IN``
accordingly, and run several seeds to assess convergence.  If that is too
slow, ``criterion=PseudoLikelihood()`` is the cheaper objective, and thinning
the gene-tree set or the taxon subset cuts the per-iteration cost directly.

Run from anywhere:

  python examples/mcmc_gt_demo.py
"""

from __future__ import annotations

from pathlib import Path

import phynetpy.IO as io
from phynetpy.criteria import Bayesian, Likelihood
from phynetpy.data import GeneTrees
from phynetpy.infer import MCMC_GTPriors, infer, score
from phynetpy.models import MSC


# TODO: CHANGE PATHS FOR YOUR OWN FILE SYSTEM!!

_REPO = Path(__file__).resolve().parent.parent
GENE_TREES_FILE = _REPO / "tests" / "testfiles" / "mpl_20taxa_gt.txt"
CONSENSUS_CACHE = _REPO / "runs" / "mpl_20taxa_consensus_seed.nwk"

# Small subset for fast smoke-testing.  Expand for real inference.
SPECIES_LABELS = ["t1", "t4", "t15", "t36", "t38", "t43", "t49"]
SPECIES_TO_ALLELES = {s: [s] for s in SPECIES_LABELS}

# Chain settings.  Keep modest for demo purposes; scale up for real
# inference and always run multiple chains with different seeds.
NUM_ITER = 1_000
BURN_IN = 200
THIN = 25
MH_SEED = 1729
MAX_RETICULATIONS = 2


def load_gene_trees(path: Path) -> GeneTrees:
    """Load and prune gene trees to the demo taxon subset."""
    networks = io.read_newick_file(
        path,
        return_type="networks",
        restrict_to_taxa=SPECIES_LABELS,
        min_leaves_after_restrict=3,
    )
    return GeneTrees(list(networks), SPECIES_TO_ALLELES)


def main() -> None:
    print("Loading gene trees (pruned to species-mapping taxa)…", flush=True)
    gene_trees = load_gene_trees(GENE_TREES_FILE)
    print(f"  {len(gene_trees.trees)} trees after pruning", flush=True)

    # Seed tree: majority-rule consensus from the gene trees.  This matches
    # ``mpl_20taxa_search_demo.py``'s seeding pattern so a direct
    # pseudo-likelihood <-> Bayesian comparison is fair.
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

    # A Bayesian criterion is not scorable on its own -- the score of one
    # fixed network is its likelihood -- so scoring the seed asks for the
    # wrapped objective.
    print("Log likelihood of the seed (before search)…", flush=True)
    start_score = score(start_net, gene_trees, model=MSC(), criterion=Likelihood())
    print(f"  log likelihood: {start_score:.6f}", flush=True)
    print(flush=True)

    print(
        f"Bayesian sampling ({NUM_ITER:,} moves, "
        f"burn-in {BURN_IN:,}, thin {THIN})…",
        flush=True,
    )
    result = infer(
        gene_trees,
        model=MSC(),
        criterion=Bayesian(
            objective=Likelihood(),
            prior=priors,
            chain_length=NUM_ITER,
            burnin=BURN_IN,
            sample_freq=THIN,
            seed=MH_SEED,
        ),
        start=start_net,
        max_reticulations=MAX_RETICULATIONS,
    )

    print(f"\nBest log posterior (MAP): {result.score:.6f}", flush=True)
    print(f"Acceptance rate:           {result.acceptance_rate:.3%}", flush=True)
    print(f"Samples collected:         {len(result.posterior)}", flush=True)
    print(f"Wall-clock time:           {result.wall_time_sec:.1f} s", flush=True)

    print("\nMAP network (Newick):", flush=True)
    print(result.best.newick(), flush=True)

    # ``result.posterior`` holds the retained samples; anything the wrapper
    # does not define (``acceptance_rate``, ``wall_time_sec``,
    # ``reticulation_posterior``, ...) falls through to the engine's native
    # result object, also reachable as ``result.raw``.
    if result.posterior:
        print("\nPosterior sample summary:", flush=True)
        top_k = min(5, len(result.posterior))
        posteriors = [s.log_posterior for s in result.posterior]
        print(
            f"  mean   log-posterior: {sum(posteriors) / len(posteriors):.6f}",
            flush=True,
        )
        print(f"  min    log-posterior: {min(posteriors):.6f}", flush=True)
        print(f"  max    log-posterior: {max(posteriors):.6f}", flush=True)
        print(f"  shown last {top_k} sample log-posteriors:", flush=True)
        for s in result.posterior[-top_k:]:
            print(f"    iter {s.iteration}: {s.log_posterior:.6f}", flush=True)


if __name__ == "__main__":
    main()
