"""
Demo: the unified inference search flags (opt_bl / fix_st / max_lvl / pseudo).

Two inference "runs" are shown, both on the same small, self-contained
gene-tree dataset so you can compare them directly:

1. **CalGTProb run** -- :class:`InferNetwork_ML`.  This is the
   maximum-likelihood network search built on the *full* multispecies
   network coalescent (MSNC) gene-tree-probability engine -- i.e. the
   PhyloNet ``CalGTProb`` likelihood (Yu, Dong, Liu & Nakhleh 2014).  Its
   ``score()`` *is* a CalGTProb calculation; ``search()`` maximises it.

2. **InferNetwork MPL run** -- :class:`MPL`.  Maximum *pseudo*-likelihood
   network inference from gene-tree triplets (Yu & Nakhleh 2015), the
   analogue of PhyloNet's ``InferNetwork_MPL``.

For each method we sweep the four search flags so you can see exactly what
each one does:

* ``opt_bl``  -- drop the continuous-parameter moves during the topology
                 search, then Brent-optimise branch lengths + gammas once
                 on the best network at the end.
* ``fix_st``  -- fix the starting-tree backbone (no ``SPR``); only
                 reticulation add/remove/relocate + gamma moves propose.
* ``max_lvl`` -- cap the network level (here 1) -- no proposal may create a
                 blob with more than one reticulation.
* ``pseudo``  -- score with the triplet pseudo-likelihood instead of the
                 full MSNC likelihood (a no-op for ``MPL``, which already
                 is pseudo-likelihood).

The dataset and iteration counts are deliberately tiny so the whole demo
runs in a few seconds.  Bump ``NUM_ITER`` / ``MAX_RETIC`` and add gene
trees for anything resembling real inference.

Run from the repo root:

    python examples/search_flags_demo.py
"""

from __future__ import annotations

import contextlib
import io as _io
import os

from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.GraphUtils import level
from phynetpy.infer import MPL, InferNetwork_ML


# ──────────────────────────────────────────────────────────────────────
# Tiny self-contained dataset
# ──────────────────────────────────────────────────────────────────────
# Four species, one allele each.  The gene trees carry a dominant
# ((A,B),C) signal with a sprinkling of ((A,C),B) discordance -- enough
# that a single reticulation can improve the fit, so max_reticulations /
# max_lvl actually have something to act on.
SPECIES = ["A", "B", "C", "D"]
MAPPING = {s: [s] for s in SPECIES}

START_TREE = "(((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0);"

GENE_TREE_NEWICKS = [
    "(((A:0.5,B:0.5):1.0,C:1.6):1.0,D:2.8);",
    "(((A:0.6,B:0.6):0.9,C:1.7):1.1,D:3.1);",
    "(((A:0.5,C:0.5):1.1,B:1.7):1.0,D:2.9);",   # ((A,C),B) discordance
    "(((A:0.7,B:0.7):0.8,C:1.5):1.2,D:3.0);",
    "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);",   # ((A,B),(C,D))
    "(((B:0.6,C:0.6):1.0,A:1.7):1.0,D:3.0);",   # ((B,C),A) discordance
]

# Keep the demo fast.  These are *tiny*; scale up for real runs.
MAX_RETIC = 1
SEED = 20260624

# The four flag scenarios we sweep for each method (label -> kwargs).
FLAG_SCENARIOS: list[tuple[str, dict]] = [
    ("baseline (no flags)",      {}),
    ("opt_bl=True",              {"opt_bl": True}),
    ("fix_st=True",              {"fix_st": True}),
    ("max_lvl=1",                {"max_lvl": 1}),
    ("pseudo=True",              {"pseudo": True}),
    ("opt_bl+fix_st+max_lvl=1",  {"opt_bl": True, "fix_st": True, "max_lvl": 1}),
]


def load_gene_trees() -> GeneTrees:
    """Build the in-memory gene-tree collection for the demo."""
    gts = GeneTrees(
        gene_tree_list=[Network.from_newick(n) for n in GENE_TREE_NEWICKS]
    )
    gts.species_gene_mapping = MAPPING
    return gts


@contextlib.contextmanager
def quiet():
    """Suppress a method's chatty stdout (e.g. MPL's kernel-stats dump)."""
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _summarize(net: Network) -> str:
    """One-line topology summary: reticulation count + network level."""
    retics = sum(1 for v in net.V() if v.is_reticulation())
    return f"retics={retics}  level={level(net)}"


# ──────────────────────────────────────────────────────────────────────
# 1. CalGTProb run -- InferNetwork_ML (full-MSNC gene-tree probability)
# ──────────────────────────────────────────────────────────────────────

def demo_calgtprob(gene_trees: GeneTrees) -> None:
    print("=" * 70)
    print("1. CalGTProb run  --  InferNetwork_ML (full MSNC gene-tree prob.)")
    print("=" * 70)

    start_net = Network.from_newick(START_TREE)

    # `score()` is a literal CalGTProb calculation: the log-probability of
    # the gene trees under the MSNC on the starting network.  `pseudo=True`
    # swaps in the triplet pseudo-likelihood for the same network.
    seed_inf = InferNetwork_ML(
        Network.from_newick(START_TREE), gene_trees, MAPPING,
        max_reticulations=MAX_RETIC,
    )
    print(f"\nCalGTProb (full MSNC) log-prob of start tree : "
          f"{seed_inf.score():.4f}")
    print(f"Pseudo-likelihood    log-prob of start tree : "
          f"{seed_inf.score(pseudo=True):.4f}")

    print("\nSearch sweep over flags:")
    print(f"  {'scenario':<26}  {'best logL':>12}   topology")
    print(f"  {'-' * 26}  {'-' * 12}   {'-' * 22}")
    for label, flags in FLAG_SCENARIOS:
        inf = InferNetwork_ML(
            Network.from_newick(START_TREE), gene_trees, MAPPING,
            max_reticulations=MAX_RETIC,
        )
        result = inf.search(
            num_runs=1,
            num_iter=40,
            max_failures=25,
            seed=SEED,
            **flags,
        )
        print(
            f"  {label:<26}  {result.best_log_likelihood:>12.4f}   "
            f"{_summarize(result.best_network)}"
        )
        # max_lvl is a hard guarantee on the returned network:
        if flags.get("max_lvl") is not None:
            assert level(result.best_network) <= flags["max_lvl"]

    print()


# ──────────────────────────────────────────────────────────────────────
# 2. InferNetwork MPL run -- MPL (triplet pseudo-likelihood)
# ──────────────────────────────────────────────────────────────────────

def demo_mpl(gene_trees: GeneTrees) -> None:
    print("=" * 70)
    print("2. InferNetwork MPL run  --  MPL (triplet pseudo-likelihood)")
    print("=" * 70)

    seed_mpl = MPL(Network.from_newick(START_TREE), gene_trees, MAPPING)
    print(f"\nMPL log pseudo-likelihood of start tree : {seed_mpl.score():.4f}")

    # MPL *is* a pseudo-likelihood method, so `pseudo` is intrinsic (passing
    # pseudo=False just warns and is ignored).  The meaningful flags here are
    # opt_bl / fix_st / max_lvl.
    mpl_scenarios = [
        (lbl, fl) for (lbl, fl) in FLAG_SCENARIOS if "pseudo" not in fl
    ]

    print("\nSearch sweep over flags:")
    print(f"  {'scenario':<26}  {'best logPL':>12}   topology")
    print(f"  {'-' * 26}  {'-' * 12}   {'-' * 22}")
    for label, flags in mpl_scenarios:
        mpl = MPL(Network.from_newick(START_TREE), gene_trees, MAPPING)
        # MPL.search prints a verbose kernel-stats block; silence it so the
        # demo's own summary stays readable.
        with quiet():
            best = mpl.search(
                method="hc",
                num_iter=60,
                max_reticulations=MAX_RETIC,
                **flags,
            )
        print(
            f"  {label:<26}  {best:>12.4f}   {_summarize(mpl.net)}"
        )
        if flags.get("max_lvl") is not None:
            assert level(mpl.net) <= flags["max_lvl"]

    print()


def main() -> None:
    print("PhyNetPy search-flags demo (opt_bl / fix_st / max_lvl / pseudo)\n")
    gene_trees = load_gene_trees()
    print(f"Loaded {len(gene_trees.trees)} gene trees over species "
          f"{SPECIES}\n")

    demo_calgtprob(gene_trees)
    demo_mpl(gene_trees)

    print("Done.  Notes:")
    print("  * opt_bl  -> branch lengths/gammas Brent-optimised at the end;")
    print("              continuous-parameter moves dropped during search.")
    print("  * fix_st  -> SPR disabled; starting-tree backbone preserved.")
    print("  * max_lvl -> returned network is guaranteed level <= the cap.")
    print("  * pseudo  -> triplet pseudo-likelihood scoring (intrinsic to MPL;")
    print("              swaps the full-MSNC CalGTProb scorer in InferNetwork_ML).")


if __name__ == "__main__":
    main()
