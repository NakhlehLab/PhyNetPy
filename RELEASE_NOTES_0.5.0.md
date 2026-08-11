# PhyNetPy 0.5.0

*Release date: 2026-05-12*

PhyNetPy 0.5.0 is a substantial feature release focused on faster,
more reliable phylogenetic-network search. The headline items are a
unified public inference API (`phynetpy.infer`), a Cython dynamic-
programming engine for MPL scoring, new and overhauled MCMC building
blocks, and several critical correctness fixes for reticulation moves
that previously had near-zero acceptance.

The public API is fully backward compatible -- existing imports such
as `from phynetpy.MCMC_GT import MCMC_GT` continue to work via thin
re-export shims.

---

## Highlights

- **New `phynetpy.infer` module** -- single front door for every
  inference method. `from phynetpy.infer import MPL, MCMC_GT,
  INFER_MP_ALLOP, MCMC_BIMARKERS` now covers every supported workflow.
- **Cython DP engine for MPL** (`phynetpy.cython.mpl_engine_cy`)
  and **Cython gene-tree MSC scoring** (`phynetpy.cython.gt_msc_cy`).
- **Adaptive proposal kernels** with per-move statistics, phase
  reporting, and on-the-fly weight tuning.
- **Simulated Annealing** with geometric and plateau-aware cooling
  schedules, joining `MetropolisHastings` and `HillClimbing` as a
  first-class search driver.
- **`MCMCGTKernel` and parallel scoring (`_ScoreManyPool`)** for the
  MCMC-GT search, giving meaningful multi-core speedup on networks
  with many gene trees.
- **Critical reticulation-move fixes**: `AddReticulation`,
  `FlipReticulation`, and `ChangeReticDest` now correctly preserve
  the gamma sum-to-one invariant on every proposal.

---

## Added

### Inference API and search drivers

- **`phynetpy.infer`** -- curated public re-exports for `MPL`,
  `MCMC_GT`, `INFER_MP_ALLOP` / `INFER_MP_ALLOP_BOOTSTRAP` /
  `ALLOP_SCORE`, and `MCMC_BIMARKERS` / `SNP_LIKELIHOOD`, together
  with their scorer, kernel, prior, and result container classes for
  advanced users.
- **`MCMCGTKernel`** (`ProposalKernel` subclass) with adaptive
  weight tuning, per-move acceptance statistics, phase reporting
  (`burn-in`, `tune`, `sample`), `freeze_adaptation()`, and
  `format_stats()` summaries.
- **`_ScoreManyPool`** -- worker-pool helper that scores network
  proposals across CPU cores; transparently used by the MCMC-GT
  search driver.
- **`RelocateReticulation` move** -- relocates a reticulation
  destination edge in one atomic step, replacing the slower
  remove + add sequence.
- **Simulated-annealing improvements**:
  - geometric and linear cooling/heating schedules,
  - plateau detection with adaptive temperature kicks,
  - save / compare hooks so a search can be checkpointed mid-run and
    diff-ed against a reference run.
- **`Sync` context manager** is now re-exported at top level
  (`from phynetpy import Sync`) for atomic model / network
  reconciliation across topology moves.

### Cython acceleration

- **`phynetpy.cython.mpl_engine_cy`** -- C-level dynamic-programming
  engine for MPL triplet scoring. Drops a large constant factor off
  large-network searches.
- **`phynetpy.cython.gt_msc_cy`** -- Cython gene-tree MSC
  scoring kernel feeding `MCMC_GT` and `MCMCGTScorer`.
- `phynetpy.graph_core_cy` build is now wired through the standard
  `setup.py build_ext`, so `pip install phynetpy[fast]` (or any
  install with Cython available) gets the optimised path
  automatically.

### Examples

- `examples/mpl_demo.py` -- end-to-end MPL scoring walkthrough.
- `examples/mpl_20taxa_search_demo.py` -- benchmark search on a
  20-taxon synthetic dataset.
- `examples/mpl_7taxa_tune_demo.py`, `mpl_7taxa_retic_sweep.py`,
  `mpl_7taxa_multiseed.py` -- targeted MPL tuning and sweep demos.
- `examples/mcmc_gt_demo.py` -- MH/HC/SA search wired to MCMC_GT.
- `examples/quickstart.py`, `examples/tree_of_blobs.py`.

---

## Changed

### Module reorganisation (back-compat preserved)

The three largest inference modules have been split into a public
re-export shim and a private implementation module:

| Old path (still works) | New canonical path |
|---|---|
| `phynetpy.MCMC_GT` | `phynetpy._mcmc_gt` (via `phynetpy.infer`) |
| `phynetpy.MPL` | `phynetpy._mpl` (via `phynetpy.infer`) |
| `phynetpy.Infer_MP_Allop` | `phynetpy._infer_mp_allop` (via `phynetpy.infer`) |

The 22-line shim at each old path re-exports every public name
unchanged, so any `from phynetpy.MCMC_GT import MCMC_GT` style import
that worked in 0.3.x keeps working. **New code should import from
`phynetpy.infer`**, which is the documented public surface from this
release forward.

### Documentation tooling

- `generate_docs.py` now **inherits method docstrings from same-module
  abstract base classes**. Concrete subclasses like `CPUExecutor`,
  `GPUExecutor`, and every `Move` subclass now render with the
  documentation from their base method, removing dozens of empty
  "method body" descriptions on the generated pages.
- `MODULE_META` extended with entries for the new `infer`,
  `ModelSelection`, and `Sync` modules.
- Private implementation modules (`_mcmc_gt`, `_mpl`,
  `_infer_mp_allop`) and the back-compat shims (`MCMC_GT`, `MPL`,
  `Infer_MP_Allop`) are explicitly skipped to avoid empty doc pages.

### Smaller refinements

- `phynetpy.__version__` now matches `setup.py` and is bumped to
  `0.4.0` (these had drifted apart in 0.3.x; the install metadata
  and runtime value are now single-sourced as expected).
- Module-level docstrings normalised to satisfy `ast.get_docstring`
  (BiMarkers, GraphUtils, MetropolisHastings, NetworkMoves, State)
  and to include `Author` / `Last Edit` / `First Included in Version`
  header lines that the doc generator scans for. All 30 documented
  modules now produce a complete header on their generated HTML page.

---

## Fixed

### Reticulation gamma invariants

Three topology moves silently violated the requirement that the two
in-edges of a reticulation node carry inheritance probabilities
summing to 1.0. The downstream MPL DP saw a one-sided gamma split,
the proposal scored at the log floor, and the move's acceptance
rate collapsed.

- **`AddReticulation`** -- `insert_node_in_edge` now propagates the
  original reticulation gamma to the new `c -> b` edge instead of
  dropping it. Eliminates the historical ~14% bad-gamma rate.
- **`FlipReticulation`** -- the post-flip target's pre-existing
  in-edge is now assigned the complementary gamma, restoring the
  sum-to-one invariant. Lifts the historical 0% accept rate.
- **`ChangeReticDest`** -- the redirected source edge keeps its
  saved gamma and the new tree-derived in-edge gets the complement,
  again restoring the sum-to-one invariant. Lifts the historical 0%
  accept rate.

All three fixes are verified by the existing 15,000+ stress test in
`tests/test_network_moves.py` and by the targeted diagnostics in
`runs/diag_retic_fix_proof.py`.

---

## Internal / Developer

- Cython 3.x supported (tested with Cython 3.2.4 on Python 3.14.2).
- Cython compiler directives standardised to `language_level=3`,
  `boundscheck=False`, `wraparound=False`, `cdivision=True`.
- Full test suite (`pytest`): 437 passed, 1 skipped, 0 failed in
  ~15 s on a single core; benchmark / scenario scripts under
  `tests/test_scenario_*.py`, `test_J_sa800.py`,
  `test_mpl_search_poc.py`, `test_adaptive_kernel.py`, and
  `test_all_scenarios.py` continue to be runnable as manual driver
  scripts.

---

## Upgrade notes

For users on 0.3.x:

1. **No code changes required** -- legacy import paths
   (`phynetpy.MCMC_GT`, `phynetpy.MPL`, `phynetpy.Infer_MP_Allop`)
   continue to work.
2. **Recommended**: migrate inference imports to
   `from phynetpy.infer import ...` for forward compatibility.
3. **Rebuild Cython** when upgrading from an editable install --
   `pip install -e .[fast]` or `python setup.py build_ext --inplace`
   to pick up the new `mpl_engine_cy` and `gt_msc_cy` extensions.

---

## Acknowledgements

PhyNetPy is developed in the
[Nakhleh Lab](https://github.com/NakhlehLab/PhyNetPy) at Rice
University. Issues and pull requests welcome.
