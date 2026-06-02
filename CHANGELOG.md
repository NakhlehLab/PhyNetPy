# Changelog

All notable changes to PhyNetPy will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.4.0] -- 2026-05-12

### Added

- **`phynetpy.infer`** -- new curated public inference API.  Single
  import surface (`MPL`, `MCMC_GT`, `INFER_MP_ALLOP`,
  `INFER_MP_ALLOP_BOOTSTRAP`, `ALLOP_SCORE`, `MCMC_BIMARKERS`,
  `SNP_LIKELIHOOD`) plus scorer, kernel, prior, and result types.
- **Cython MPL DP engine** (`phynetpy.cython.mpl_engine_cy`):
  C-level dynamic programming for MPL triplet scoring.
- **Cython gene-tree MSC kernel** (`phynetpy.cython.gt_msc_cy`):
  C-level scoring backbone for `MCMC_GT` and `MCMCGTScorer`.
- **`MCMCGTKernel`** -- adaptive proposal kernel for MCMC-GT with
  per-move acceptance statistics, phase reporting (`burn-in` /
  `tune` / `sample`), `freeze_adaptation()`, and
  `format_stats()` summaries.
- **`_ScoreManyPool`** -- parallel network-scoring worker pool used
  by the MCMC-GT search driver for multi-core speedups.
- **`RelocateReticulation`** topology move -- atomic remove + re-add
  of a reticulation destination edge.
- **Simulated-Annealing enhancements**: geometric and linear
  cooling / heating schedules, plateau detection with adaptive
  temperature kicks, and search save / compare hooks.
- **`Sync`** context manager re-exported at top level
  (`from phynetpy import Sync`).
- New example scripts: `mpl_demo.py`, `mpl_20taxa_search_demo.py`,
  `mpl_7taxa_tune_demo.py`, `mpl_7taxa_retic_sweep.py`,
  `mpl_7taxa_multiseed.py`, `mcmc_gt_demo.py`, `quickstart.py`,
  `tree_of_blobs.py`.

### Changed

- **Module reorganisation** -- the three largest inference modules
  have been split into private implementations
  (`_mcmc_gt`, `_mpl`, `_infer_mp_allop`) and 22-line re-export
  shims at the old import paths (`MCMC_GT`, `MPL`,
  `Infer_MP_Allop`).  Existing `from phynetpy.MCMC_GT import ...`
  style imports continue to work; new code should prefer
  `from phynetpy.infer import ...`.
- **`generate_docs.py`** now inherits method docstrings from
  same-module abstract base classes.  Concrete subclasses
  (`CPUExecutor`, `GPUExecutor`, every `Move` subclass) inherit
  documentation from their base method, eliminating dozens of empty
  method descriptions on the generated HTML pages.  Private
  implementation modules and back-compat shims are now skipped to
  prevent empty doc pages.
- Module headers and docstrings normalised so the doc-generator
  picks up `Author`, `Last Edit`, and `First Included in Version`
  lines on all 30 documented modules.
- `phynetpy.__version__` and `setup.py` `version=` are now
  single-sourced and both report `0.4.0` (these had drifted
  apart in 0.3.x).

### Fixed

- **`AddReticulation`** gamma preservation -- `insert_node_in_edge`
  now propagates the original reticulation gamma to the new in-edge
  instead of dropping it, eliminating the historical ~14%
  bad-gamma proposal rate.
- **`FlipReticulation`** sum-to-one invariant -- the flipped
  target's pre-existing in-edge is now assigned the complementary
  gamma, restoring the `gamma + (1 - gamma) = 1` invariant on the
  post-flip reticulation.  Lifts the historical 0% accept rate.
- **`ChangeReticDest`** sum-to-one invariant -- the redirected
  source edge keeps its saved gamma and the new tree-derived
  in-edge gets the complement.  Lifts the historical 0% accept
  rate for this move.

---

## [0.3.2] -- 2026-04-02

### Added

- **Maximum Pseudo-Likelihood (MPL) scoring** module (`MPL`):
  Implements Yu & Nakhleh (2015) log pseudo-likelihood for phylogenetic
  network inference from gene-tree triplet frequencies. Includes
  `GeneTreeTripletResult` container, `compute_gene_tree_triplets` for
  extracting rho values from gene trees, and `mpl_score` for scoring a
  network against observed triplet data.
- **`Sync` context manager** (`Sync`): provides atomic model/network
  reconciliation -- topology moves executed inside a `with Sync(model):`
  block are automatically rolled back on error and reconciled on success.
- **`Infer_MP_Allop`** module: refactored maximum-parsimony allopolyploid
  inference with `InferMPAllop`, `MPAllopComponent`, `MPAllopScorer`,
  `Allop_MUL`, and `AlleleMap` classes; bootstrapping support via
  `INFER_MP_ALLOP_BOOTSTRAP`.
- **`Infer_MP_Allop_Kernel`** in `MetropolisHastings`: dedicated proposal
  kernel for MP allopolyploid hill-climbing search.
- **`SimulatedAnnealing`** search algorithm in `MetropolisHastings`.
- **Automated deployment script** (`deploy.py`): version bumping, test
  running, package building, and PyPI upload in one command.
- New example scripts: `mpl_demo.py`, `quickstart.py`, `tree_of_blobs.py`.
- Expanded test suite: MPL scoring tests, network-moves stress tests
  (15,000+ move attempts), scenario-based validation (D, D_sa, J_sa800),
  and 20-taxon MPL benchmarks.

### Changed

- **`NetworkMoves`** significantly refactored: improved documentation,
  cleaner move implementations for `add_hybrid`, `remove_hybrid`,
  `switch_parentage`, and tail/head moves.
- **`ModelMove`** and **`ModelGraph`** updated to support the `Sync`
  reconciliation workflow and the new inference methods.
- Regenerated HTML documentation for all modules.

---

## [0.3.1] -- 2026

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
