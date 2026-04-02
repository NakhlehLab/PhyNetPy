"""
Runtime benchmark for MPL scoring on a 20-taxa species network.

Species network: 18 leaf taxa, 2 reticulations (#H192, #H185).
Gene trees:      1000 trees with 125 taxa each.
Mapping:         Identity (1-to-1) for the 18 species-network leaves.

This test does NOT verify likelihood correctness — it measures and
reports wall-clock time for the two main phases:

  1. Gene-tree triplet precomputation  (rho values, C(18,3)=816 triplets)
  2. Species-network scoring           (subnet decomposition per triplet)
"""

from __future__ import annotations

import os
import time

import pytest

from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL import MPL, compute_gene_tree_triplets
from phynetpy.IO import convert_newick

TESTFILES = os.path.join(os.path.dirname(__file__), "testfiles")

SPECIES_TAXA = [
    "t1", "t4", "t15", "t36", "t38", "t43", "t49", "t52",
    "t74", "t83", "t84", "t85", "t94", "t109", "t111", "t123", "t124", "t130",
]
MAPPING = {t: [t] for t in SPECIES_TAXA}


def _load_species_network() -> Network:
    """Parse the species network (PhyloNet extended Newick with ::gamma)."""
    path = os.path.join(TESTFILES, "mpl_20taxa.txt")
    with open(path) as f:
        raw = f.readline().strip()
    phynetpy_nwk = convert_newick(raw, standard="PhyNetPy")
    return Network.from_newick(phynetpy_nwk)


def _load_gene_trees() -> GeneTrees:
    """Parse 1000 gene trees from the test file."""
    path = os.path.join(TESTFILES, "mpl_20taxa_gt.txt")
    trees = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(Network.from_newick(line))
    gts = GeneTrees(gene_tree_list=trees)
    gts.species_gene_mapping = MAPPING
    return gts


@pytest.fixture(scope="module")
def species_net() -> Network:
    return _load_species_network()


@pytest.fixture(scope="module")
def gene_trees() -> GeneTrees:
    return _load_gene_trees()


class TestMPL20TaxaRuntime:
    """Runtime benchmarks for the 20-taxa MPL pipeline."""

    def test_gene_tree_loading(self, gene_trees: GeneTrees):
        """Verify gene trees loaded correctly."""
        assert len(gene_trees.trees) == 1000

    def test_species_network_structure(self, species_net: Network):
        """Verify species network has expected leaf count."""
        leaves = sorted(n.label for n in species_net.get_leaves())
        assert len(leaves) == 18

    def test_rho_precomputation_runtime(self, gene_trees: GeneTrees):
        """Time the gene-tree triplet (rho) precomputation phase.

        816 triplets x 1000 gene trees = 816k induced-triple evaluations.
        """
        t0 = time.perf_counter()
        result = compute_gene_tree_triplets(
            gene_trees=gene_trees,
            mapping=MAPPING,
            species_labels=SPECIES_TAXA,
        )
        elapsed = time.perf_counter() - t0

        assert len(result.triplets) == 816  # C(18, 3)
        for rho in result.rho_by_triplet.values():
            assert all(r >= 0 for r in rho)

        print(f"\n  Rho precomputation: {elapsed:.2f}s "
              f"({len(result.triplets)} triplets, "
              f"{len(gene_trees.trees)} gene trees)")

    def test_full_mpl_score_runtime(
        self, species_net: Network, gene_trees: GeneTrees
    ):
        """Time the full MPL pipeline: rho + network scoring."""
        t0 = time.perf_counter()
        mpl = MPL(species_net, gene_trees, MAPPING)
        t_init = time.perf_counter() - t0

        t1 = time.perf_counter()
        score = mpl.score()
        t_score = time.perf_counter() - t1

        total = t_init + t_score

        import math
        assert math.isfinite(score)
        assert score < 0.0

        print(f"\n  MPL init (rho):     {t_init:.2f}s")
        print(f"  MPL score (net):    {t_score:.2f}s")
        print(f"  Total:              {total:.2f}s")
        print(f"  Log pseudo-likelihood: {score:.4f}")

    def test_rescoring_is_fast(
        self, species_net: Network, gene_trees: GeneTrees
    ):
        """After init, re-scoring the same network should be fast
        since rho is cached and only network-side work is redone."""
        mpl = MPL(species_net, gene_trees, MAPPING)
        _ = mpl.score()  # warm up

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mpl.score()
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        print(f"\n  Rescore avg ({len(times)} runs): {avg:.3f}s")
        print(f"  Rescore times: {[f'{t:.3f}s' for t in times]}")


if __name__ == "__main__":
    print("Loading species network...")
    net = _load_species_network()
    leaves = sorted(n.label for n in net.get_leaves())
    print(f"  {len(leaves)} leaves: {leaves}")

    print("Loading 1000 gene trees...")
    t0 = time.perf_counter()
    gts = _load_gene_trees()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    print("\n--- Rho precomputation ---")
    t0 = time.perf_counter()
    rho_result = compute_gene_tree_triplets(
        gene_trees=gts, mapping=MAPPING, species_labels=SPECIES_TAXA,
    )
    rho_time = time.perf_counter() - t0
    print(f"  {len(rho_result.triplets)} triplets in {rho_time:.2f}s")

    print("\n--- Full MPL scoring ---")
    t0 = time.perf_counter()
    mpl = MPL(net, gts, MAPPING)
    init_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    score = mpl.score()
    score_time = time.perf_counter() - t0

    print(f"  Init (rho):  {init_time:.2f}s")
    print(f"  Score (net): {score_time:.2f}s")
    print(f"  Total:       {init_time + score_time:.2f}s")
    print(f"  Log PL:      {score:.6f}")

    print("\n--- Rescoring (rho cached) ---")
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        s = mpl.score()
        t = time.perf_counter() - t0
        times.append(t)
        print(f"  Run {i+1}: {t:.3f}s  (score={s:.4f})")
    print(f"  Avg: {sum(times)/len(times):.3f}s")
