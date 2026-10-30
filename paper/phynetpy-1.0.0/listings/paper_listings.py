"""
Every code listing in the manuscript, in executable form.

Each ``listing_*`` function is the complete, runnable version of one listing in
sections/examples.tex.  The listing in the paper is an excerpt of the function
body, so running this script is what verifies that the paper's code matches the
1.0.0 API rather than an API the author remembered.

Run: python paper/phynetpy-1.0.0/listings/paper_listings.py
Exits non-zero if any listing fails.
"""

from __future__ import annotations

import contextlib
import io
import traceback


# ---------------------------------------------------------------------------
# Listing 1 -- build a network, inspect its structure
# ---------------------------------------------------------------------------
def listing_1_build_and_inspect() -> None:
    from phynetpy import (
        Edge,
        Network,
        Node,
        blobs,
        count_reticulations,
        level,
        read_newick,
    )

    # Extended Newick: #H1 is a reticulation with two parents.
    net = read_newick("((A:0.1,(B:0.2)#H1:0.1):0.3,(#H1:0.2,C:0.3):0.2);")

    print(f"  leaves            {[n.label for n in net.get_leaves()]}")
    print(f"  reticulations     {count_reticulations(net)}")
    print(f"  level             {level(net)}")
    print(f"  acyclic           {net.is_acyclic()}")
    print(f"  blobs             {len(blobs(net))}")

    # The same type is built directly, node by node.
    root, internal = Node("Root"), Node("I1")
    leaf_a, leaf_b, leaf_c = Node("A"), Node("B"), Node("C")
    built = Network()
    built.add_nodes([root, internal, leaf_a, leaf_b, leaf_c])
    for edge in (
        Edge(root, internal),
        Edge(root, leaf_c),
        Edge(internal, leaf_a),
        Edge(internal, leaf_b),
    ):
        built.add_edges(edge)
    print(f"  built by hand     {built.newick()}")


# ---------------------------------------------------------------------------
# Listing 2 -- score a network, then infer one, under two criteria
# ---------------------------------------------------------------------------
def listing_2_score_then_infer() -> None:
    from phynetpy import BranchLengthUnit, convert_network_branch_lengths
    from phynetpy.criteria import Likelihood, PseudoLikelihood
    from phynetpy.GraphUtils import mu_distance
    from phynetpy.infer import infer, score, simulate
    from phynetpy.models import MSC

    theta = 0.5
    gts = simulate(MSC(theta=theta), taxa=6, n=200, data="gene_trees", seed=3)

    # The simulator writes branch lengths in substitutions per site; the
    # gene-tree criteria read coalescent units. The unit travels on the
    # network, so converting is explicit and a mismatch is an error rather
    # than a plausible wrong number.
    truth = convert_network_branch_lengths(
        gts.true_network, BranchLengthUnit.COALESCENT_2N, theta=theta
    )

    log_lik = score(truth, gts, model=MSC(theta=theta),
                    criterion=Likelihood())
    log_pl = score(truth, gts, model=MSC(theta=theta),
                   criterion=PseudoLikelihood())
    print(f"  log likelihood        {log_lik:.4f}")
    print(f"  log pseudo-likelihood {log_pl:.4f}")

    # Search for a network. Changing the criterion is the only change.
    result = infer(gts, criterion=PseudoLikelihood(), max_reticulations=0)
    print(f"  method                {result.method}")
    print(f"  score                 {result.score:.4f}")
    print(f"  mu to truth           {mu_distance(truth, result.best)}")


# ---------------------------------------------------------------------------
# Listing 3 -- simulate and recover
# ---------------------------------------------------------------------------
def listing_3_simulate_and_recover() -> None:
    from phynetpy.criteria import PseudoLikelihood
    from phynetpy.GraphUtils import mu_distance
    from phynetpy.infer import infer, simulate
    from phynetpy.models import MSC

    sim = simulate(MSC(theta=0.5), taxa=6, n=200, data="gene_trees", seed=11)
    recovered = infer(sim, criterion=PseudoLikelihood(), max_reticulations=0)

    # mu-distance compares topology, so the unit mismatch between the
    # simulated network and the inferred one does not affect it.
    print(f"  mu-distance to truth  "
          f"{mu_distance(sim.true_network, recovered.best)}")


# ---------------------------------------------------------------------------
# Listing 4 -- the validity matrix, and registering a new cell
# ---------------------------------------------------------------------------
def listing_4_extend() -> None:
    from phynetpy.criteria import Criterion
    from phynetpy.data import GeneTrees
    from phynetpy.infer import Engine, register, validity_matrix
    from phynetpy.models import MSC

    for data, row in validity_matrix().items():
        print(f"  {data:<18}{row}")

    class TreeLength(Criterion):
        """Total branch length: a stand-in for a real objective."""

        accepts_data = (GeneTrees,)
        scorable = True
        use_branch_lengths = False

    @register(GeneTrees, MSC, TreeLength)
    class TreeLengthEngine(Engine):
        method = "TreeLength"

        def infer(self, data):
            raise NotImplementedError("scoring only")

        def score(self, network, data, optimize=False):
            return sum(
                edge.get_length() or 0.0 for edge in network.E()
            )

    from phynetpy.infer import score

    # GeneTrees.trees is a set, so take an arbitrary member rather than index.
    data = GeneTrees.from_newick(["((A:0.1,B:0.2):0.3,C:0.4);"])
    net = next(iter(data.trees))
    total = score(net, data, criterion=TreeLength())
    print(f"  registered cell score {total:.2f}")


LISTINGS = (
    ("1: build and inspect a network", listing_1_build_and_inspect),
    ("2: score, then infer", listing_2_score_then_infer),
    ("3: simulate and recover", listing_3_simulate_and_recover),
    ("4: validity matrix and a new cell", listing_4_extend),
)


def main() -> int:
    failures = []
    for title, fn in LISTINGS:
        print(f"\n--- Listing {title} ---")
        sink = io.StringIO()
        try:
            # Engines print search diagnostics; keep the listing output clean.
            with contextlib.redirect_stdout(sink):
                fn()
        except Exception:
            failures.append(title)
            print(sink.getvalue(), end="")
            print("  FAILED")
            traceback.print_exc()
            continue
        print(sink.getvalue(), end="")

    if failures:
        print(f"\n{len(failures)} listing(s) failed: {failures}")
        return 1
    print(f"\nAll {len(LISTINGS)} listings ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
