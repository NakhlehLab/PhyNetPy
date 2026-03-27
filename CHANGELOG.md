# Changelog

All notable changes to PhyNetPy will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] (targeting 0.3.1)

### Added

- **Network distance metrics** in `GraphUtils`:
  - `mu_distance` -- mu-representation (path multiplicity) distance for
    reduced phylogenetic networks (Cardona et al. 2009). A true metric that
    reduces to Robinson-Foulds on trees.
  - `hardwired_cluster_distance` -- symmetric difference of hardwired cluster
    sets.
  - `softwired_cluster_distance` -- symmetric difference of softwired cluster
    sets (union over all displayed trees).
  - `robinson_foulds_distance` -- classic RF distance using non-trivial
    clusters; extends to networks via hardwired clusters.
  - `tripartition_distance` -- symmetric difference of tripartition sets
    induced by binary tree nodes.
  - `displayed_tree_distance` -- symmetric difference of unique displayed-tree
    topologies (identified by cluster representation).
  - `average_path_distance` (APD) -- branch-length-aware dissimilarity
    averaged uniformly over displayed trees (Yakici, Ogilvie & Nakhleh,
    RECOMB-CG 2022).
  - `weighted_average_path_distance` (WAPD) -- same as APD but weighted by
    displayed-tree probability (product of kept gamma values).
- All cluster/tripartition/displayed-tree distances support a `normalize`
  parameter that returns a value in [0, 1].
- `rooted_triplet_distance` method on `Network` (renamed from the former
  `nakhleh_distance` -- see *Changed* below).
- Comprehensive test suite for all distance metrics
  (`tests/test_network_distances.py`, 83 tests).

### Changed

- **Renamed `nakhleh_distance` to `mu_distance`** in `GraphUtils` to use the
  standard literature name and remove personal attribution per author request.
- **Renamed `Network.nakhleh_distance` to `Network.rooted_triplet_distance`**
  to better reflect what the method computes (rooted triplet symmetric
  difference) and to align with the renaming above.

### Fixed

- `get_all_subtrees` produced incorrect / non-deterministic results for
  networks with 2+ reticulations:
  - **Combination bug**: `_retic_edge_choice` generated duplicate (not all
    2^k) displayed-tree combinations for k >= 2 reticulations. Replaced with
    `itertools.product` over per-reticulation edge lists.
  - **Non-deterministic ordering**: `in_edges()` returns a set; converting to
    a list gave arbitrary edge ordering that varied across independently
    constructed copies of the same topology. In-edges are now sorted by source
    label for reproducibility.
- **`SwitchParentage` (PSPP) move** produced networks with spurious leaf
  nodes. When a reticulation node's only child edge was removed, `_cleanup_node`
  left the childless reticulation in the network as a dead-end (in-degree >= 1,
  out-degree 0). Added a cleanup case that removes dead-end internal nodes and
  cascades to their parents. Verified across 15,000 move attempts with zero
  failures.
- **`extra_lineages` scoring** could return negative values for certain MUL
  tree / gene tree embeddings produced during hill climbing. Clamped per-allele-
  map extra lineages to >= 0 in `Allop_MUL.XL`, matching the theoretical
  invariant. Scoring validated against the scenarioD_ideal / D10 benchmark
  (expected score = 3).

---

## [0.3.0] -- 2025

Initial public release of PhyNetPy.
