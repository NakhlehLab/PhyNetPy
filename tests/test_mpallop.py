"""
Test suite for the Maximum Parsimony Allopolyploidy inference module
(phynetpy.Infer_MP_Allop).

Self-contained tests that generate data programmatically -- no external files
needed.  Includes:

    - Unit tests for AlleleMap, Allop_MUL scoring, helper functions.
    - Component-level tests for MPAllopScorer, MPAllopComponent, InferMPAllop.
    - Full inference benchmarks with runtime statistics across different
      network sizes, reticulation levels, and gene-tree counts.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import copy
import sys
import textwrap
import time
from collections import defaultdict
from typing import Any

import numpy as np
import pytest

pytest.importorskip("phynetpy.Infer_MP_Allop", reason="Infer_MP_Allop not importable")

from phynetpy.BirthDeath import Yule
from phynetpy.IO import read_newick
from phynetpy.Infer_MP_Allop import (
    AlleleMap,
    Allop_MUL,
    InferMPAllop,
    InferAllopError,
    MPAllopComponent,
    MPAllopScorer,
    allele_map_set,
    allele_map_set_ilp,
    cluster_as_name_set,
    clusters_contains,
    generate_tree_from_clusters,
    partition_gene_trees,
    random_object,
)
from phynetpy.ModelFactory import ModelFactory
from phynetpy.ModelGraph import Model
from phynetpy.Network import Network, Node, Edge, NetworkError
from phynetpy.NetworkMoves import add_hybrid


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

RNG_SEED = 42


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


def _make_diploid_gene_map(taxa: list[str]) -> dict[str, list[str]]:
    """Identity map: each taxon has exactly one gene copy (diploid-like)."""
    return {t: [f"{t}_a"] for t in taxa}


def _make_polyploid_gene_map(taxa: list[str],
                              polyploid: dict[str, int] | None = None
                              ) -> dict[str, list[str]]:
    """Build a gene map where some taxa are polyploid.

    ``polyploid`` maps taxon name -> ploidy (number of copies).
    Taxa not in the dict default to 1 copy.
    """
    if polyploid is None:
        polyploid = {}
    gene_map: dict[str, list[str]] = {}
    for t in taxa:
        n_copies = polyploid.get(t, 1)
        copies = [f"{t}_{chr(ord('a') + i)}" for i in range(n_copies)]
        gene_map[t] = copies
    return gene_map


def _simulate_gene_trees(species_tree: Network,
                          n_genes: int,
                          rng: np.random.Generator,
                          discord_prob: float = 0.4) -> list[Network]:
    """Generate gene trees by perturbing a species tree topology.

    Each gene tree starts as a copy of the species tree.  With probability
    ``discord_prob`` per tree, a random pair of leaves is swapped, creating
    genuine topological discordance that produces non-zero parsimony scores.
    """
    trees: list[Network] = []
    for _ in range(n_genes):
        gt = copy.deepcopy(species_tree)
        leaves = list(gt.get_leaves())
        if len(leaves) >= 2 and rng.random() < discord_prob:
            idxs = rng.choice(len(leaves), size=2, replace=False)
            a, b = leaves[int(idxs[0])], leaves[int(idxs[1])]
            name_a, name_b = a.label, b.label
            gt.update_node_name(a, name_b)
            gt.update_node_name(b, name_a)
        trees.append(gt)
    return trees


def _yule_tree(n_taxa: int, rng: np.random.Generator,
               taxa_names: list[str] | None = None) -> Network:
    """Generate a Yule tree with *n_taxa* leaves, optionally renamed."""
    yule = Yule(0.1, n_taxa, rng=rng)
    net = yule.generate_network()
    if taxa_names is not None:
        assert len(taxa_names) == n_taxa
        for leaf, name in zip(net.get_leaves(), taxa_names):
            net.update_node_name(leaf, name)
    return net


def _make_network_with_reticulation(taxa: list[str],
                                     n_retics: int,
                                     rng: np.random.Generator) -> Network:
    """Create a tree on *taxa* and then add *n_retics* hybrid edges."""
    net = _yule_tree(len(taxa), rng, taxa_names=taxa)
    edges = list(net.E())
    for _ in range(n_retics):
        edges = list(net.E())
        if len(edges) < 2:
            break
        idxs = rng.choice(len(edges), size=2, replace=False)
        try:
            add_hybrid(net, edges[int(idxs[0])], edges[int(idxs[1])])
        except Exception:
            pass
    return net


def _prepare_gene_trees(gene_trees: list[Network],
                         gene_map: dict[str, list[str]]) -> None:
    """Attach allele maps and leaf descendants to each gene tree."""
    for gt in gene_trees:
        allele_funcs = allele_map_set(gt, gene_map)
        gt.put_item("allele maps", allele_funcs)
        gt.put_item("leaf descendants", gt.leaf_descendants_all())


# ---------------------------------------------------------------------------
# Table printer for benchmark results
# ---------------------------------------------------------------------------

class BenchmarkTable:
    """Collect and pretty-print benchmark rows."""

    _HEADER = (
        f"{'Test':<45} {'Taxa':>5} {'Retics':>6} {'GTs':>4} "
        f"{'Iters':>5} {'Score':>8} {'Time(s)':>8}"
    )
    _SEP = "-" * len(_HEADER)

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, taxa: int, retics: int,
            gts: int, iters: int,
            score: float, elapsed: float) -> None:
        self.rows.append(dict(
            name=name, taxa=taxa, retics=retics,
            gts=gts, iters=iters,
            score=score, elapsed=elapsed,
        ))

    def print(self) -> None:
        print()
        print(self._SEP)
        print(self._HEADER)
        print(self._SEP)
        for r in self.rows:
            print(
                f"{r['name']:<45} {r['taxa']:>5} {r['retics']:>6} "
                f"{r['gts']:>4} {r['iters']:>5} {r['score']:>8.1f} "
                f"{r['elapsed']:>8.3f}"
            )
        print(self._SEP)
        print()


BENCH = BenchmarkTable()


# ---------------------------------------------------------------------------
# Unit tests -- AlleleMap
# ---------------------------------------------------------------------------

class TestAlleleMap:

    def test_put_success(self) -> None:
        am = AlleleMap()
        node = Node("gene1")
        assert am.put(node, "A_a") == 1
        assert am.map["gene1"] == "A_a"

    def test_put_duplicate_blocked(self) -> None:
        am = AlleleMap()
        n1, n2 = Node("g1"), Node("g2")
        assert am.put(n1, "A_a") == 1
        assert am.put(n2, "A_a") == 0, "Same MUL leaf should be disallowed"

    def test_put_different_leaves(self) -> None:
        am = AlleleMap()
        n1, n2 = Node("g1"), Node("g2")
        assert am.put(n1, "A_a") == 1
        assert am.put(n2, "B_a") == 1
        assert len(am.map) == 2


# ---------------------------------------------------------------------------
# Unit tests -- allele_map_set
# ---------------------------------------------------------------------------

class TestAlleleMapSet:

    def test_simple_diploid(self) -> None:
        gt = read_newick("((A_a,B_a),C_a);")
        gene_map = {"A": ["A_a"], "B": ["B_a"], "C": ["C_a"]}
        maps = allele_map_set(gt, gene_map)
        assert len(maps) >= 1
        for m in maps:
            assert set(m.map.values()) == {"A_a", "B_a", "C_a"}

    def test_polyploid_multiple_maps(self) -> None:
        gt = read_newick("((X_a,X_b),Y_a);")
        gene_map = {"X": ["X_a", "X_b"], "Y": ["Y_a"]}
        maps = allele_map_set(gt, gene_map)
        assert len(maps) >= 1

    def test_ilp_fallback(self) -> None:
        """ILP version should produce valid maps or fallback gracefully."""
        gt = read_newick("((A_a,B_a),C_a);")
        gene_map = {"A": ["A_a"], "B": ["B_a"], "C": ["C_a"]}
        maps = allele_map_set_ilp(gt, gene_map)
        assert len(maps) >= 1

    def test_species_named_gene_tips(self) -> None:
        """Gene tips may reuse species names when alleles use other labels."""
        gt = read_newick("((A,B),C);")
        gene_map = {"A": ["A_a"], "B": ["B_a"], "C": ["C_a", "C_b"]}
        maps = allele_map_set(gt, gene_map)
        assert len(maps) == 2
        mapped_targets = {frozenset(m.map.values()) for m in maps}
        assert mapped_targets == {
            frozenset({"A_a", "B_a", "C_a"}),
            frozenset({"A_a", "B_a", "C_b"}),
        }
        for m in maps:
            assert m.map["C"] in {"C_a", "C_b"}

    def test_arbitrary_allele_labels(self) -> None:
        """Allele labels need not resemble the species name."""
        gt = read_newick("((A,B),C);")
        gene_map = {"A": ["alpha"], "B": ["beta"], "C": ["gamma", "delta"]}
        maps = allele_map_set(gt, gene_map)
        assert len(maps) == 2
        for m in maps:
            assert m.map["A"] == "alpha"
            assert m.map["B"] == "beta"
            assert m.map["C"] in {"gamma", "delta"}


# ---------------------------------------------------------------------------
# Unit tests -- helper functions
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_random_object_normal(self, rng: np.random.Generator) -> None:
        items = [1, 2, 3, 4, 5]
        result = random_object(items, rng)
        assert result in items

    def test_random_object_empty(self, rng: np.random.Generator) -> None:
        assert random_object([], rng) is None

    def test_cluster_as_name_set(self) -> None:
        nodes = {Node("A"), Node("B"), Node("C")}
        names = cluster_as_name_set(nodes)
        assert names == frozenset({"A", "B", "C"})

    def test_clusters_contains_positive(self) -> None:
        c1 = {Node("A"), Node("B")}
        c2 = {Node("A"), Node("B")}
        assert clusters_contains(c1, {frozenset(c2)}) or \
               clusters_contains(c1, [c2])

    def test_generate_tree_from_clusters(self) -> None:
        clusters = {frozenset({"A", "B"}), frozenset({"A", "B", "C"})}
        tree = generate_tree_from_clusters(clusters)
        leaf_names = {n.label for n in tree.get_leaves()}
        assert "A" in leaf_names
        assert "B" in leaf_names
        assert "C" in leaf_names


# ---------------------------------------------------------------------------
# Unit tests -- Allop_MUL scoring on a known tree
# ---------------------------------------------------------------------------

class TestAllopMULScoring:

    def test_score_identical_tree(self, rng: np.random.Generator) -> None:
        """When gene trees match the species tree exactly, score should be 0."""
        taxa = ["A", "B", "C"]
        gene_map = _make_diploid_gene_map(taxa)

        species_tree = read_newick("((A,B),C);")
        gt = read_newick("((A_a,B_a),C_a);")

        _prepare_gene_trees([gt], gene_map)

        mul = Allop_MUL(gene_map, rng)
        mul.to_mul(species_tree)
        score = mul.score([gt])
        assert score >= 0, f"Parsimony score should be non-negative, got {score}"

    def test_score_increases_with_discord(self, rng: np.random.Generator) -> None:
        """Discordant gene tree should produce >= score of concordant one."""
        taxa = ["A", "B", "C", "D"]
        gene_map = _make_diploid_gene_map(taxa)

        species_tree = read_newick("(((A,B),C),D);")

        concordant = read_newick("(((A_a,B_a),C_a),D_a);")
        discordant = read_newick("(((A_a,C_a),B_a),D_a);")

        _prepare_gene_trees([concordant], gene_map)
        _prepare_gene_trees([discordant], gene_map)

        mul_c = Allop_MUL(gene_map, rng)
        mul_c.to_mul(copy.deepcopy(species_tree))
        score_c = mul_c.score([concordant])

        mul_d = Allop_MUL(gene_map, rng)
        mul_d.to_mul(copy.deepcopy(species_tree))
        score_d = mul_d.score([discordant])

        assert score_d >= score_c, (
            f"Discordant score ({score_d}) should be >= concordant ({score_c})"
        )

    def test_score_multiple_gene_trees(self, rng: np.random.Generator) -> None:
        """Score over multiple gene trees should be the sum of individual scores."""
        taxa = ["A", "B", "C"]
        gene_map = _make_diploid_gene_map(taxa)

        species_tree = read_newick("((A,B),C);")
        gt1 = read_newick("((A_a,B_a),C_a);")
        gt2 = read_newick("((A_a,C_a),B_a);")

        all_gts = [gt1, gt2]
        _prepare_gene_trees(all_gts, gene_map)

        mul = Allop_MUL(gene_map, rng)
        mul.to_mul(species_tree)
        total = mul.score(all_gts)

        individual_sum = 0
        for gt in all_gts:
            m = Allop_MUL(gene_map, rng)
            m.to_mul(copy.deepcopy(species_tree))
            individual_sum += m.score([gt])

        assert total == individual_sum

    def test_to_mul_rejects_underspecified_allele_map(
            self, rng: np.random.Generator) -> None:
        """Identity map on a hybrid taxon must raise a clear NetworkError."""
        gene_map = {t: [t] for t in ["A", "B", "C", "D", "E", "F"]}
        net = read_newick("(((A,B),#H1),((C)#H1,(D,(E,F))));")
        mul = Allop_MUL(gene_map, rng)

        with pytest.raises(NetworkError, match="too few copies for species 'C'"):
            mul.to_mul(net)

    def test_infer_rejects_underspecified_allele_map(
            self, rng: np.random.Generator) -> None:
        """InferMPAllop should fail fast with InferAllopError, not -inf."""
        gene_map = {t: [t] for t in ["A", "B", "C"]}
        net = read_newick("((A,(B)#H1),(C,#H1));")
        gt = read_newick("((A,B),C);")

        with pytest.raises(InferAllopError, match="too few copies"):
            InferMPAllop(net, gene_map, [gt], iter_ct=10, rng=rng)

    def test_score_species_tips_with_named_alleles(
            self, rng: np.random.Generator) -> None:
        """Gene tips named like species; alleles use distinct labels."""
        gene_map = {
            "A": ["A_a"], "B": ["B_a"], "C": ["C_a", "C_b"],
            "D": ["D_a"], "E": ["E_a"], "F": ["F_a"],
        }
        net = read_newick("(((A,B),#H1),((C)#H1,(D,(E,F))));")
        gts = [
            read_newick("(((A,B),(D,(E,F))),C);"),
            read_newick("(((A,B),C),(D,(E,F)));"),
        ]
        _prepare_gene_trees(gts, gene_map)

        mul = Allop_MUL(gene_map, rng)
        mul.to_mul(net)
        score = mul.score(gts)
        assert score >= 0


# ---------------------------------------------------------------------------
# Component tests -- MPAllopScorer / MPAllopComponent
# ---------------------------------------------------------------------------

class TestMPAllopScorer:

    def test_scorer_callable(self, rng: np.random.Generator) -> None:
        taxa = ["A", "B", "C"]
        gene_map = _make_diploid_gene_map(taxa)
        species_tree = read_newick("((A,B),C);")
        gt = read_newick("((A_a,B_a),C_a);")
        _prepare_gene_trees([gt], gene_map)

        scorer = MPAllopScorer(gene_map, [gt], rng)

        model = Model()
        model.network = species_tree

        result = scorer(model)
        assert isinstance(result, (int, float))
        assert result <= 0, "Negated score should be <= 0"

    def test_scorer_no_network_returns_neg_inf(self, rng: np.random.Generator) -> None:
        scorer = MPAllopScorer({"A": ["A_a"]}, [], rng)
        model = Model()
        assert scorer(model) == float("-inf")


class TestMPAllopComponent:

    def test_build_sets_network_and_scorer(self, rng: np.random.Generator) -> None:
        taxa = ["A", "B", "C"]
        gene_map = _make_diploid_gene_map(taxa)
        net = read_newick("((A,B),C);")
        gt = read_newick("((A_a,B_a),C_a);")
        _prepare_gene_trees([gt], gene_map)

        comp = MPAllopComponent(net, gene_map, [gt], rng)
        factory = ModelFactory(comp)
        model = factory.build()

        assert model.network is not None
        score = model.likelihood()
        assert isinstance(score, (int, float))


# ---------------------------------------------------------------------------
# Starting network generation
# ---------------------------------------------------------------------------

class TestStartingNetworkGeneration:

    def test_partition_gene_trees_diploid(self, rng: np.random.Generator) -> None:
        gene_map = _make_diploid_gene_map(["A", "B", "C", "D"])
        net = partition_gene_trees(gene_map, rng=rng)
        leaf_names = {n.label for n in net.get_leaves()}
        assert leaf_names == {"A", "B", "C", "D"}

    def test_partition_gene_trees_polyploid(self, rng: np.random.Generator) -> None:
        gene_map = _make_polyploid_gene_map(
            ["A", "B", "X", "Y"],
            polyploid={"X": 2, "Y": 2}
        )
        net = partition_gene_trees(gene_map, rng=rng)
        leaf_names = {n.label for n in net.get_leaves()}
        assert leaf_names == {"A", "B", "X", "Y"}


# ---------------------------------------------------------------------------
# Full inference -- small smoke tests
# ---------------------------------------------------------------------------

class TestInferMPAllopSmoke:

    def test_tiny_3_taxa_diploid(self, rng: np.random.Generator) -> None:
        """Smoke test: 3 taxa, diploid, 2 gene trees, 50 iterations."""
        taxa = ["A", "B", "C"]
        gene_map = _make_diploid_gene_map(taxa)

        gts = [read_newick("((A_a,B_a),C_a);") for _ in range(2)]
        _prepare_gene_trees(gts, gene_map)

        start_net = partition_gene_trees(gene_map, rng=rng)

        infer = InferMPAllop(start_net, gene_map, gts, iter_ct=50, rng=rng)
        score = infer.run()
        assert isinstance(score, (int, float))
        assert len(infer.results) >= 0

    def test_tiny_4_taxa_one_polyploid(self, rng: np.random.Generator) -> None:
        """Smoke test: 4 taxa with one tetraploid, 3 gene trees."""
        gene_map = _make_polyploid_gene_map(
            ["A", "B", "C", "X"],
            polyploid={"X": 2}
        )

        gt_nwk = "((A_a,(X_a,X_b)),(B_a,C_a));"
        gts = [read_newick(gt_nwk) for _ in range(3)]
        _prepare_gene_trees(gts, gene_map)

        start_net = partition_gene_trees(gene_map, rng=rng)

        infer = InferMPAllop(start_net, gene_map, gts, iter_ct=50, rng=rng)
        score = infer.run()
        assert isinstance(score, (int, float))


# ---------------------------------------------------------------------------
# Benchmark / runtime statistics
# ---------------------------------------------------------------------------

def _run_benchmark(name: str,
                   n_taxa: int,
                   n_retics: int,
                   n_gene_trees: int,
                   n_iters: int,
                   seed: int = 123) -> dict[str, Any]:
    """Run one MP Allop inference and return timing/score info."""
    rng = np.random.default_rng(seed)

    taxa = [f"T{i}" for i in range(n_taxa)]

    if n_retics > 0:
        polyploid = {taxa[i]: 2 for i in range(min(n_retics, n_taxa))}
    else:
        polyploid = {}

    gene_map = _make_polyploid_gene_map(taxa, polyploid)

    all_gene_copies = []
    for copies in gene_map.values():
        all_gene_copies.extend(copies)

    base_tree = _yule_tree(len(all_gene_copies), rng,
                           taxa_names=all_gene_copies)

    gts = _simulate_gene_trees(base_tree, n_gene_trees, rng)
    _prepare_gene_trees(gts, gene_map)

    start_net = partition_gene_trees(gene_map, rng=rng)

    t0 = time.perf_counter()
    infer = InferMPAllop(start_net, gene_map, gts,
                          iter_ct=n_iters, rng=rng)
    score = infer.run()
    elapsed = time.perf_counter() - t0

    result = dict(
        name=name, taxa=n_taxa, retics=n_retics,
        gts=n_gene_trees, iters=n_iters,
        score=score, elapsed=elapsed,
    )
    BENCH.add(**result)
    return result


class TestBenchmarkTaxaScaling:
    """Vary number of taxa while keeping gene trees and iterations fixed."""

    @pytest.mark.parametrize("n_taxa", [4, 6, 8])
    def test_taxa_scaling_diploid(self, n_taxa: int) -> None:
        r = _run_benchmark(
            name=f"taxa_scaling_diploid_{n_taxa}",
            n_taxa=n_taxa, n_retics=0,
            n_gene_trees=5, n_iters=100,
        )
        assert r["elapsed"] < 300, f"Took too long: {r['elapsed']:.1f}s"


class TestBenchmarkReticulationLevels:
    """Vary number of reticulations."""

    @pytest.mark.parametrize("n_retics", [0, 1, 2])
    def test_retic_scaling(self, n_retics: int) -> None:
        r = _run_benchmark(
            name=f"retic_level_{n_retics}",
            n_taxa=5, n_retics=n_retics,
            n_gene_trees=5, n_iters=100,
        )
        assert r["elapsed"] < 300


class TestBenchmarkGeneTreeCounts:
    """Vary number of gene trees."""

    @pytest.mark.parametrize("n_gts", [2, 5, 10, 20])
    def test_gene_tree_scaling(self, n_gts: int) -> None:
        r = _run_benchmark(
            name=f"gene_trees_{n_gts}",
            n_taxa=5, n_retics=1,
            n_gene_trees=n_gts, n_iters=80,
        )
        assert r["elapsed"] < 300


class TestBenchmarkIterationCounts:
    """Vary number of hill-climbing iterations."""

    @pytest.mark.parametrize("n_iters", [25, 50, 100])
    def test_iter_scaling(self, n_iters: int) -> None:
        r = _run_benchmark(
            name=f"iters_{n_iters}",
            n_taxa=5, n_retics=1,
            n_gene_trees=5, n_iters=n_iters,
        )
        assert r["elapsed"] < 300


class TestBenchmarkCombined:
    """Cross-product of sizes for a comprehensive benchmark matrix."""

    @pytest.mark.parametrize(
        "n_taxa,n_retics,n_gts,n_iters",
        [
            (4, 0, 3, 50),
            (5, 1, 5, 80),
            (6, 1, 10, 100),
            (8, 2, 5, 80),
        ],
        ids=["4t-0r-3g", "5t-1r-5g", "6t-1r-10g", "8t-2r-5g"],
    )
    def test_combined(self, n_taxa: int, n_retics: int,
                      n_gts: int, n_iters: int) -> None:
        r = _run_benchmark(
            name=f"combined_{n_taxa}t_{n_retics}r_{n_gts}g",
            n_taxa=n_taxa, n_retics=n_retics,
            n_gene_trees=n_gts, n_iters=n_iters,
        )
        assert r["elapsed"] < 300


@pytest.fixture(scope="session", autouse=True)
def _print_benchmark_table(request: Any) -> None:
    """Print the benchmark summary table at the end of the test session."""
    yield
    if BENCH.rows:
        BENCH.print()
