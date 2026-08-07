"""
Demo: the search flags that cut across every criterion (opt_bl / fix_st /
max_lvl), and the criterion axis that replaced the old ``pseudo`` flag.

Two inference runs are shown, both on the same small, self-contained
gene-tree dataset so you can compare them directly.  The *only* difference
between them is the ``criterion`` argument:

1. **Full likelihood** -- ``criterion=Likelihood()``.  Maximum-likelihood
   network search on the *full* multispecies network coalescent (MSNC)
   gene-tree-probability engine -- i.e. the PhyloNet ``CalGTProb`` likelihood
   (Yu, Dong, Liu & Nakhleh 2014).  ``score`` computes it; ``infer``
   maximises it.

2. **Pseudo-likelihood** -- ``criterion=PseudoLikelihood()``.  Maximum
   *pseudo*-likelihood from gene-tree triplets (Yu & Nakhleh 2015), the
   analogue of PhyloNet's ``InferNetwork_MPL``.

The old ``pseudo=True`` flag is gone: swapping objectives is a change of
criterion, not a flag on a method, which is why the two runs below are the
same three lines with one word changed.

The remaining flags are genuinely cross-cutting search controls, so they are
still ``**search`` keywords:

* ``opt_bl``  -- drop the continuous-parameter moves during the topology
                 search, then Brent-optimise branch lengths + gammas once
                 on the best network at the end.
* ``fix_st``  -- fix the starting-tree backbone (no ``SPR``); only
                 reticulation add/remove/relocate + gamma moves propose.
* ``max_lvl`` -- cap the network level (here 1) -- no proposal may create a
                 blob with more than one reticulation.

``fix_st`` also has a stronger, first-class form: ``Start(net,
StartMode.AUGMENT)`` requires the *result* to contain the starting network,
which ``fix_st`` alone does not guarantee (it fixes the moves, not the
outcome).  The last scenario below shows it.

The dataset and iteration counts are deliberately tiny so the whole demo
runs in a few seconds.  Bump ``NUM_ITER`` / ``MAX_RETIC`` and add gene
trees for anything resembling real inference.

Run from the repo root:

    python examples/search_flags_demo.py
"""

from __future__ import annotations

import contextlib
import io as _io

from phynetpy.Network import Network
from phynetpy.GraphUtils import level
from phynetpy.criteria import Likelihood, PseudoLikelihood
from phynetpy.data import GeneTrees
from phynetpy.infer import Start, StartMode, infer, score
from phynetpy.models import MSC


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

# The flag scenarios we sweep for each criterion (label -> search kwargs).
FLAG_SCENARIOS: list[tuple[str, dict]] = [
    ("baseline (no flags)",      {}),
    ("opt_bl=True",              {"opt_bl": True}),
    ("fix_st=True",              {"fix_st": True}),
    ("max_lvl=1",                {"max_lvl": 1}),
    ("opt_bl+fix_st+max_lvl=1",  {"opt_bl": True, "fix_st": True, "max_lvl": 1}),
]


def load_gene_trees() -> GeneTrees:
    """Build the in-memory gene-tree collection for the demo."""
    return GeneTrees.from_newick(GENE_TREE_NEWICKS, MAPPING)


@contextlib.contextmanager
def quiet():
    """Suppress an engine's chatty stdout (e.g. the MPL kernel-stats dump)."""
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _summarize(net: Network) -> str:
    """One-line topology summary: reticulation count + network level."""
    retics = sum(1 for v in net.V() if v.is_reticulation())
    return f"retics={retics}  level={level(net)}"


def sweep(gene_trees: GeneTrees, criterion, *, title: str, column: str,
          **fixed) -> None:
    """Run every flag scenario under one criterion and tabulate the result."""
    print("=" * 70)
    print(title)
    print("=" * 70)

    start_net = Network.from_newick(START_TREE)
    seed_score = score(start_net, gene_trees, model=MSC(), criterion=criterion)
    print(f"\n{column} of start tree : {seed_score:.4f}")

    print("\nSearch sweep over flags:")
    print(f"  {'scenario':<26}  {column:>12}   topology")
    print(f"  {'-' * 26}  {'-' * 12}   {'-' * 22}")
    for label, flags in FLAG_SCENARIOS:
        with quiet():
            result = infer(
                gene_trees,
                model=MSC(),
                criterion=criterion,
                start=Network.from_newick(START_TREE),
                max_reticulations=MAX_RETIC,
                seed=SEED,
                **fixed,
                **flags,
            )
        print(
            f"  {label:<26}  {result.score:>12.4f}   "
            f"{_summarize(result.best)}"
        )
        # max_lvl is a hard guarantee on the returned network:
        if flags.get("max_lvl") is not None:
            assert level(result.best) <= flags["max_lvl"]

    # StartMode.AUGMENT: the stronger form of fix_st.  Not a flag on the
    # search but a property of the start, because it constrains the answer.
    with quiet():
        augmented = infer(
            gene_trees,
            model=MSC(),
            criterion=criterion,
            start=Start(Network.from_newick(START_TREE), StartMode.AUGMENT),
            max_reticulations=MAX_RETIC,
            seed=SEED,
            **fixed,
        )
    print(
        f"  {'Start(..., AUGMENT)':<26}  {augmented.score:>12.4f}   "
        f"{_summarize(augmented.best)}"
    )
    print()


def main() -> None:
    print("PhyNetPy search-flags demo (opt_bl / fix_st / max_lvl)\n")
    gene_trees = load_gene_trees()
    print(f"Loaded {len(gene_trees.trees)} gene trees over species "
          f"{SPECIES}\n")

    sweep(
        gene_trees,
        Likelihood(),
        title="1. Full MSNC likelihood  --  criterion=Likelihood()",
        column="best logL",
        num_runs=1,
        num_iter=40,
        max_failures=25,
    )
    sweep(
        gene_trees,
        PseudoLikelihood(),
        title="2. Triplet pseudo-likelihood  --  criterion=PseudoLikelihood()",
        column="best logPL",
        method="hc",
        num_iter=60,
    )

    print("Done.  Notes:")
    print("  * opt_bl  -> branch lengths/gammas Brent-optimised at the end;")
    print("              continuous-parameter moves dropped during search.")
    print("  * fix_st  -> SPR disabled; starting-tree backbone preserved.")
    print("  * max_lvl -> returned network is guaranteed level <= the cap.")
    print("  * AUGMENT -> returned network is guaranteed to *contain* the")
    print("              starting network, which fix_st alone does not")
    print("              promise; it restricts the moves, not the result.")
    print("  * The objective is the criterion argument, not a flag: the two")
    print("    sweeps above differ by one word.")


if __name__ == "__main__":
    main()
