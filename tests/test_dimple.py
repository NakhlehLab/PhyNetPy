"""
Tests for ``phynetpy.infer.infer_dimple`` and the DIMPLE helpers.

The searches are deliberately tiny (hill climbing, 20 proposals, one run):
these tests check the pipeline -- validation, partition, subset search,
merge, graft, scoring -- not the quality of the inferred network. The gene
trees back ``(((A,B),C),D)`` and ``(E,F)`` strongly, and the start tree is
already that topology, so a hill climb has nowhere better to go.
"""

import math

import pytest

from phynetpy.data import GeneTrees
from phynetpy.GraphUtils import (
    count_reticulations,
    group_reticulations,
    is_tree,
    reticulation_cycle_size,
    reticulation_signature,
    rooted_triples,
    same_reticulation,
)
from phynetpy.infer import InferenceResult, infer_dimple
from phynetpy.IO import read_newick
from phynetpy._dimple import (
    SUBSET_ITERS,
    SUBSET_RUNS,
    DimpleError,
    DimpleTrace,
    partition_scaffold,
    subset_chains,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

TAXA = list("ABCDEF")
TRUTH = "((((A:1,B:1):1,C:1):1,D:1):1,(E:1,F:1):1);"
GENE_TREES = [TRUTH] * 8 + [
    "((((A:1,C:1):1,B:1):1,D:1):1,(E:1,F:1):1);",
    "(((A:1,B:1):1,(C:1,D:1):1):1,(E:1,F:1):1);",
]
FAST = dict(method="hc", num_iter=20, n_runs=1)


def _leaves(net) -> list:
    return sorted(leaf.label for leaf in net.get_leaves())


def _clades(net) -> set:
    return {frozenset(leaf.label for leaf in leaves)
            for leaves in net.leaf_descendants_all().values()}


@pytest.fixture(scope="module")
def gts() -> GeneTrees:
    return GeneTrees.from_newick(GENE_TREES)


class TestInferDimple:
    def test_single_subset_blob_is_inferred_and_grafted(self, gts):
        result = infer_dimple(
            gts, read_newick("((A,B,C,D),(E,F));"), read_newick(TRUTH),
            max_subset_size=4, n_divisions=1, max_reticulations=0, **FAST,
        )
        assert isinstance(result, InferenceResult) and result.method == "DIMPLE"
        best = result.best
        assert _leaves(best) == TAXA
        assert is_tree(best) and best.is_acyclic()
        # The non-blob clade survives the graft and the blob keeps the
        # subset search's optimum.
        clades = _clades(best)
        assert frozenset("EF") in clades and frozenset("ABCD") in clades
        assert frozenset("AB") in clades
        assert isinstance(result.score, float) and math.isfinite(result.score)
        trace = result.raw
        assert isinstance(trace, DimpleTrace)
        assert len(trace.blobs) == 1 and len(trace.blobs[0].subsets) == 1

    def test_root_polytomy_is_merged_from_overlapping_subsets(self, gts):
        result = infer_dimple(
            gts, read_newick("(A,B,C,D,E,F);"), read_newick(TRUTH),
            max_subset_size=4, n_divisions=3, max_reticulations=0,
            score_final=False, **FAST,
        )
        assert _leaves(result.best) == TAXA
        assert result.best.is_acyclic()
        assert result.score is None
        assert len(result.raw.blobs) == 1
        assert len(result.raw.blobs[0].subsets) > 1

    def test_illegal_inputs_are_refused(self, gts):
        scaffold, start = read_newick("((A,B,C,D),(E,F));"), read_newick(TRUTH)
        with pytest.raises(DimpleError, match="reticulation"):
            infer_dimple(gts, read_newick("((A,(B)#H1,C,D),(#H1,E),F);"), start, **FAST)
        with pytest.raises(DimpleError, match="not binary"):
            infer_dimple(gts, scaffold, read_newick("((A,B,C,D),(E,F));"), **FAST)
        with pytest.raises(DimpleError, match="missing"):
            infer_dimple(gts, scaffold, read_newick("((((A,B),C),D),E);"), **FAST)
        with pytest.raises(DimpleError, match="max_subset_size"):
            infer_dimple(gts, scaffold, start, max_subset_size=3, **FAST)
        with pytest.raises(DimpleError, match="model_selection"):
            infer_dimple(gts, scaffold, start, model_selection="lasso", **FAST)


class TestPartitionAndChains:
    def test_partition_covers_the_blob_and_is_deterministic(self):
        scaffold = read_newick("((A,B,C,D,E,F,G,H),(I,J));")
        jobs = partition_scaffold(scaffold, max_subset_size=4, n_divisions=2, seed=1)
        assert len(jobs) == 1
        job = jobs[0]
        assert job.tob_leaves == frozenset("ABCDEFGH")
        assert job.external_siblings == frozenset("IJ")
        assert 1 <= len(job.divisions) <= 2
        for division in job.divisions:
            assert frozenset().union(*division) == job.tob_leaves
            assert all(3 <= len(subset) <= 4 for subset in division)
        again = partition_scaffold(scaffold, max_subset_size=4, n_divisions=2, seed=1)
        assert again[0].divisions == job.divisions

    def test_subset_chains_defaults_and_overrides(self):
        chains = subset_chains({})
        assert len(chains) == SUBSET_RUNS
        assert [chain["seed"] for chain in chains] == list(range(SUBSET_RUNS))
        assert all(chain["method"] == "sa" and chain["num_iter"] == SUBSET_ITERS
                   for chain in chains)
        (chain,) = subset_chains(dict(FAST))
        assert chain["method"] == "hc" and chain["num_iter"] == 20
        assert "phase_patience" in chain
        with pytest.raises(DimpleError):
            subset_chains({"chains": []})
        with pytest.raises(DimpleError):
            subset_chains({"n_runs": 0})


class TestGraphHelpers:
    def test_rooted_triples(self):
        assert rooted_triples(read_newick("(((A,B),C),D);")) == {
            ("A", "B", "C"), ("A", "B", "D"), ("A", "C", "D"), ("B", "C", "D"),
        }
        assert rooted_triples(read_newick("(A,B,C);")) == set()

    def test_reticulation_identity_is_by_parent_side_overlap(self):
        def signature(newick):
            net = read_newick(newick)
            hybrid = next(v for v in net.V() if v.is_reticulation())
            return reticulation_signature(net, hybrid)

        same = signature("((A,(B)#H1),(#H1,C));")
        swapped = signature("((C,(B)#H1),(#H1,A));")
        other = signature("((A,(C)#H1),(#H1,B));")
        child, parents = same
        assert child == (frozenset("B"),)
        assert set(parents) == {(frozenset("A"),), (frozenset("C"),)}
        assert same_reticulation(same, same)
        assert same_reticulation(same, swapped)
        assert not same_reticulation(same, other)
        groups = group_reticulations([{"signature": s} for s in (same, swapped, other)])
        assert [len(group) for group in groups] == [2, 1]

    def test_cycle_size_separates_detectable_reticulations(self):
        visible = read_newick("((A,(B)#H1),(#H1,C));")
        invisible = read_newick("(((B)#H1,A),#H1,C);")
        for net, size in ((visible, 4), (invisible, 3)):
            hybrid = next(v for v in net.V() if v.is_reticulation())
            assert reticulation_cycle_size(net, hybrid) == size
            assert count_reticulations(net) == 1
        assert count_reticulations(visible, min_cycle_size=4) == 1
        assert count_reticulations(invisible, min_cycle_size=4) == 0
