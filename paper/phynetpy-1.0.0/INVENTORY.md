# PhyNetPy 1.0.0 paper — claim/evidence inventory

Internal working document. Not part of the submitted manuscript.

Every factual claim the manuscript makes must appear here with a pointer to a
file, a test, or a benchmark JSON key. Status codes:

- **V** — verified against this repository on the date below.
- **X** — requires an experiment that has not yet been run.
- **A** — requires information only the author can supply.
- **F** — future (1.0.0 scope), must be marked as scheduled, not shipped.

Verification date: 2026-09-25. Repository state: `main` at `b3a1a411`,
`src/_version.py` = `0.6.0`, tag `v0.6.0`.

---

## 0. Version and release identity

| # | Claim | Status | Evidence |
|---|---|---|---|
| 0.1 | Version under description is 1.0.0 | **F** | Repo is `0.6.0`. `src/_version.py`; `.github/workflows/ci.yml:73,78` assert `0.6.0`; only tag is `v0.6.0`. Manuscript writes 1.0.0 with `[TODO: verify at 1.0.0]` on anything version-specific. |
| 0.2 | 1.0.0 adds MP from gene trees (`InferNetwork_MP`) | **F** | Currently a legal-but-unregistered cell; `src/_engines.py:34-40`. Author-confirmed as 1.0.0 scope. |
| 0.3 | 1.0.0 adds biallelic pseudo-likelihood (`MLE_BiMarkers -pseudo`) | **F** | Currently legal-but-unregistered; `src/_engines.py:34-40`. Author-confirmed as 1.0.0 scope. |
| 0.4 | Released on PyPI as `phynetpy`, MIT licensed | **V** | `pyproject.toml:3,9`; `LICENSE`. |
| 0.5 | Archival DOI | **A** | No Zenodo record found. `[TODO: DOI]` in the software-citation section. |
| 0.6 | Prior release dates / download counts | **A** | `reports/NSF_Annual_Report_Year1.md:55-63` lists release dates. No download statistics exist (report §2 says so explicitly). Do not state adoption numbers. |

## 1. Scale of the codebase

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1.1 | 52 Python modules, 42,610 lines under `src/` | **V** | Measured 2026-09-25, excluding `__pycache__`. Supersedes the "49 modules / 40,763 lines" in the NSF report. |
| 1.2 | Four Cython extension modules, 1,879 lines of `.pyx` | **V** | `src/cython/{graph_core_cy,gt_msc_cy,mpl_engine_cy,seq_engine_cy}.pyx`. |
| 1.3 | 121 names in the top-level public API | **V** | `len(phynetpy.__all__)` = 121. `src/__init__.py:216-310`. |
| 1.4 | 26 test files, 11,198 lines | **V** | `tests/test_*.py`, measured 2026-09-25. |

## 2. Public API and architecture

| # | Claim | Status | Evidence |
|---|---|---|---|
| 2.1 | Inference is two verbs, `infer` and `score` | **V** | `src/infer.py:404` (`infer`), `:465` (`score`). |
| 2.2 | A third verb, `simulate`, inverts the same axes | **V** | `src/_simulate.py`; re-exported `src/infer.py:80`. |
| 2.3 | Dispatch is on three axes: data x model x criterion | **V** | `src/data/`, `src/models/_processes.py`, `src/criteria/_objectives.py`; `resolve()` at `src/_registry.py:205`. |
| 2.4 | Data axis: `GeneTrees`, `Alignment`, `BiallelicMarkers` | **V** | `src/data/_genetrees.py:111`, `src/data/_sequence.py:65,184`. Base `Data` at `src/data/_base.py:52`. |
| 2.5 | Model axis: `MSC`, `Allopolyploid` | **V** | `src/models/_processes.py:97,167`. |
| 2.6 | Criterion axis: `MDC`, `Likelihood`, `PseudoLikelihood`, `Bayesian` | **V** | `src/criteria/_objectives.py:103,136,170,224`. |
| 2.7 | `Bayesian` wraps an objective rather than sitting parallel to the likelihoods | **V** | `src/criteria/_objectives.py:224` (`objective` parameter), `:286-294` (delegates `accepts_data`). |
| 2.8 | The registry doubles as an executable validity matrix | **V** | `validity_matrix()` at `src/_registry.py:293-333`; built from `_ENGINES`, not a hand-written table. |
| 2.9 | Three failure modes are distinguished, never conflated | **V** | `src/_registry.py:249-255` (`TypeError`), `:257-264` (`ValueError`), `:266-273` (`NotImplementedError`). |
| 2.10 | `register()` is the extension point for a new method | **V** | `src/_registry.py:137-171`; duplicate keys rejected at `:162-167`. Exercised by all seven of the library's own cells. |
| 2.11 | `infer` returns one type regardless of criterion | **V** | `InferenceResult`, `src/infer.py:327-396`. `__getattr__` falls through to `.raw` at `:364`. |
| 2.12 | `Start(net, StartMode.AUGMENT)` is enforced, not approximated | **V** | `src/infer.py:262-308`. Engines that cannot honour it refuse explicitly: `src/_engines.py:507-508,729-730`. |
| 2.13 | Branch-length units are typed and conversions explicit | **V** | `BranchLengthUnit` (`UNSPECIFIED`, `SUBSTITUTIONS_PER_SITE`, `COALESCENT_2N`) at `src/_units.py:18-29`; `convert_network_branch_lengths` at `:150-211`, refuses `UNSPECIFIED` on either side (`:166-169`). |
| 2.14 | Compiled kernels sit behind a pure-Python API | **V** | `src/Network.py:46-54` imports `NodeSet`/`EdgeSet` from `graph_core_cy` and raises a build-instruction `ImportError` on failure. No public API exposes the Cython types. |

### 2.15 Registered cells — the complete set

From `registered_cells()`, executed 2026-09-25. **V** for all seven.

| Data | Model | Criterion | Method name | `_engines.py` |
|---|---|---|---|---|
| `GeneTrees` | `MSC` | `Likelihood` | `InferNetwork_ML` | 218-227 |
| `GeneTrees` | `MSC` | `PseudoLikelihood` | `InferNetwork_MPL` | 282-291 |
| `GeneTrees` | `MSC` | `Bayesian` | `MCMC_GT` | 342-356 |
| `Alignment` | `MSC` | `Bayesian` | `MCMC_SEQ` | 441-450 |
| `BiallelicMarkers` | `MSC` | `Likelihood` | `MLE_BiMarkers` | 601-610 |
| `BiallelicMarkers` | `MSC` | `Bayesian` | `MCMC_BiMarkers` | 630-637 |
| `GeneTrees` | `Allopolyploid` | `MDC` | `MP_Allop` | 699-709 |

Legal but unregistered under `MSC` (render as `-`, raise `NotImplementedError`): **V**

- `(GeneTrees, MSC, MDC)` — PhyloNet's `InferNetwork_MP`. See 0.2.
- `(BiallelicMarkers, MSC, PseudoLikelihood)` — `MLE_BiMarkers -pseudo`. See 0.3.
- `(Alignment, MSC, Likelihood)` — accepted by the criterion, no engine.

Not meaningful (render as `x`, raise `TypeError` via `accepts_data`): **V**

- `Alignment` x `MDC`, `Alignment` x `PseudoLikelihood`, `BiallelicMarkers` x `MDC`.

## 3. Network representation

| # | Claim | Status | Evidence |
|---|---|---|---|
| 3.1 | Rooted DAG of `Network`/`Node`/`Edge`/`Branch` | **V** | `src/Network.py:942,267,638,117`. |
| 3.2 | Trees are networks with no reticulations; no separate type | **V** | `GraphUtils.is_tree`; `GeneTrees` holds `Network` objects (`src/data/_genetrees.py:111`). |
| 3.3 | Reticulation nodes are not capped at two parents | **V** | `get_parents` docstring, `src/Network.py:1777-1779`; `add_edges` (`:1500-1539`) performs no parent-count check. Caveat: `reticulation_tripartitions` requires exactly two (`src/ReticulationComparison.py:261-267`), so state the representation is unconstrained while some analyses assume binary. |
| 3.4 | Traversal, structural query, mutation, validation, conversion | **V** | Traversal `bfs_dfs:2027`, `topological_order:2179`, `get_subtree_at:2154`. Query `in_degree:1193`, `in_edges:1226`, `leaf_descendants_all:1993`, `mrca:1852`, `subnet:2080`. Mutation `add_nodes:1052`, `add_edges:1500`, `remove_edge:1574`, `clean:1828`. Validation `is_acyclic:2016`. Conversion `newick:2005`, `copy:2093`, `to_networkx:2107`. |
| 3.5 | Ploidy / subgenome representation | **V** | `subgenome_count:1894`, `subgenome_ct_edges:1938`, `edges_to_subgenome_count:1965`; `MUL.to_mul:2260`. |
| 3.6 | SPR and other topology moves are reusable classes | **V** | `src/ModelMove.py`; ten exported at `src/__init__.py:178-190`: `SPR`, `AddReticulation`, `RemoveReticulation`, `FlipReticulation`, `RelocateReticulation`, `SwitchParentage`, `ChangeNodeHeight`, `ChangeInheritanceProb`, `ChangeReticSource`, `ChangeReticDest`. |
| 3.7 | 22 distinct proposal operators in total | **A** | `reports/NSF_Annual_Report_Year1.md:191-202` states 22 (10 in `ModelMove.py` + 12 in `_mcmc_seq.py`). The 10 are verified above; the 12 in `_mcmc_seq.py` need an explicit enumeration before the number is printed. `[TODO: enumerate the MCMC_SEQ operators]` |
| 3.8 | No SPR-equivalent for the allopolyploid search | **V** | `SwitchParentage` is its only move; `reports/NSF_Annual_Report_Year1.md:624-626`. |

## 4. Network analysis

| # | Claim | Status | Evidence |
|---|---|---|---|
| 4.1 | Nine dissimilarity measures | **V** | `GraphUtils.__all__:79-87`: `mu_distance`, `hardwired_cluster_distance`, `softwired_cluster_distance`, `robinson_foulds_distance`, `tripartition_distance`, `displayed_tree_distance`, `average_path_distance`, `weighted_average_path_distance`, `pairwise_leaf_distance`. Plus `Network.rooted_triplet_distance:2135`. |
| 4.2 | Reticulation-specific comparison is a separate module | **V** | `src/ReticulationComparison.py`; `reticulation_tripartitions:235`, `reticulation_dissimilarity:844`, `reticulation_precision_recall:880`, `combined_dissimilarity:919`, `compare_networks:720`. |
| 4.3 | Decomposition: blobs, tree of blobs, bridges, level, displayed trees | **V** | `GraphUtils.__all__:70-78`: `blobs`, `tree_of_blobs`, `bridges_and_articulations`, `get_all_subtrees`, `get_all_clusters`, `network_clusters`, `dominant_tree`, `count_displayed_trees`, `level`, `induced_subnetwork_by_taxa`. |
| 4.4 | Rendering is ASCII only | **V** | `GraphUtils.ascii` is the only renderer (`__all__:102-103`). No plotting of networks anywhere in `src/`. State as a limitation. |
| 4.5 | Model selection over reticulation count | **V** | `reticulation_sweep`, `SweepResult`, `SweepRow`; `src/ModelSelection.py`, exported `src/__init__.py:191`. |

## 5. Input/output

| # | Claim | Status | Evidence |
|---|---|---|---|
| 5.1 | Extended Newick, NEXUS, FASTA, VCF — read and write | **V** | `src/__init__.py:105-121`: `read_newick`/`write_newick`(`_file`), `read_nexus`/`write_nexus`/`read_nexus_msa`, `read_fasta`/`write_fasta`/`read_fasta_records`/`write_fasta_from_network`, `read_vcf`/`write_vcf`/`read_vcf_metadata`. |
| 5.2 | Newick dialect detection and conversion | **V** | `detect_newick_standard`, `convert_newick`; `src/__init__.py:120-121`. |
| 5.3 | PHYLIP is not supported | **V** | Absent from `src/IO.py` exports. `reports/NSF_Annual_Report_Year1.md:550` confirms it remains to be added. |
| 5.4 | Tracer log and NEXUS tree output for interoperability | **V** | `write_tracer_log`, `read_tracer_log`, `write_trees_nexus`; `src/_chain_analysis.py`, re-exported `src/infer.py:165-167`. |
| 5.5 | Four alphabets (DNA, RNA, protein, codon) | **V** | `src/Alphabet.py`; 
`tests/test_alphabet.py`. |

## 6. Dependencies and packaging

| # | Claim | Status | Evidence |
|---|---|---|---|
| 6.1 | Requires Python >= 3.9 | **V** | `pyproject.toml:14`. |
| 6.2 | Runtime dependencies: NumPy, SciPy, scikit-learn, matplotlib, NetworkX, Biopython, python-nexus, newick, PuLP | **V** | `pyproject.toml:37-47`. |
| 6.3 | A C compiler is a build prerequisite | **V** | Cython kernels compiled unconditionally; `src/Network.py:46-54` makes the import hard. `pyproject.toml:52-53` notes `[fast]` is now a no-op. |
| 6.4 | Version is single-sourced | **V** | `pyproject.toml:73-74` reads `phynetpy._version.__version__`. |

## 7. Testing and CI

| # | Claim | Status | Evidence |
|---|---|---|---|
| 7.1 | 1,130 tests collected | **V** | `pytest --collect-only -q` → "1130 tests collected", 2026-09-25. |
| 7.2 | 1,125 pass, 5 deselected as slow, 0 failures, 36.3 s | **V** | `pytest -m "not slow" -q` → "1125 passed, 5 deselected, 86 warnings in 36.31s", 2026-09-25. Note: **no skips** in this environment because the PhyloNet jar and BEAGLE are installed; on a machine without them the cross-check module skips. |
| 7.3 | CI matrix covers Python 3.9-3.14 | **V** | `.github/workflows/ci.yml:25`. |
| 7.4 | CI also runs Windows smoke tests and a packaging job | **V** | `.github/workflows/ci.yml:38-49` (Windows), `:51-78` (`check-manifest`, `build`, `twine check`, clean-install of wheel **and** sdist). |
| 7.5 | Slow suite is opt-in via `workflow_dispatch` | **V** | `.github/workflows/ci.yml:80-93`. |
| 7.6 | 432 substitution-model tests | **V** | `pytest tests/test_gtr.py --collect-only` → 432, 2026-09-25. Supersedes the "339" in `CHANGELOG.md:223`. |
| 7.7 | 153 tests for distances, reticulation comparison, and blobs | **V** | `pytest tests/test_network_distances.py tests/test_reticulation_comparison.py tests/test_blobs_and_subnet.py --collect-only` → 153. |
| 7.8 | Memory and mutation-cost bounds are regression-tested | **V** | `tests/test_memory.py`. Specific figures (0.96 MB, 0.12 us, 2 us, 3.46 KB) are quoted in the NSF report; re-measure before printing. **X** |
| 7.9 | CI exists | **V** | Note that `reports/NSF_Annual_Report_Year1.md:552-561` says there is *no* CI; that section is out of date relative to `.github/workflows/ci.yml`. Do not cite the report for this. |

## 8. Correctness validation against PhyloNet

| # | Claim | Status | Evidence |
|---|---|---|---|
| 8.1 | PhyNetPy agrees with PhyloNet on the MSNC gene-tree density and the Felsenstein likelihood | **V** | `pytest tests/test_crosscheck_phylonet.py -q` → **23 passed** in 1.48 s, 2026-09-25, with `PHYLONET_JAR` set to `C:\Users\Marky\Documents\PhyloNetJar\phylonet.jar`. |
| 8.2 | 14 case specifications, deliberately adversarial | **V** | `tests/crosscheck/run_crosscheck.py:71-219`. Names: `tree2taxa`, `tree3taxa_concordant`, `tree3taxa_deepILS`, `network1retic`, `gtr_skewed`, `multiallele`, `net2retic`, `bigtheta_heavyILS`, `tinytheta`, `invalid_embedding`, `ambiguity_gaps`, `saturation`, `constant_sites`, `tree5taxa`. |
| 8.3 | Agreement tolerance is 1e-5, and `-inf` must match exactly | **V** | `run_crosscheck.py:318-334` (`tol=1e-5`, `:329-331` for `-inf`); `tests/test_crosscheck_phylonet.py:117-119`. |
| 8.4 | PhyloNet's own classes are invoked, not a reimplementation of them | **V** | `tests/crosscheck/CrossCheck.java:95-97` (`GeneTreeBrSpeciesNetDistribution.calculateGTDistribution`), `:101-108` (BEAGLE-backed `UltrametricTree.logDensity()`). |
| 8.5 | The suite skips cleanly when the jar or BEAGLE is absent | **V** | `tests/test_crosscheck_phylonet.py:45-53`; also skips without `javac` (`:64-66`). |
| 8.6 | PhyloNet version is 3.8.2 | **V** | `C:\Users\Marky\Documents\PhyloNetJar\phylonet.jar` is 41,019,851 bytes, byte-identical in size to `PhyloNetv3_8_2.jar`. |

## 9. Other correctness evidence

| # | Claim | Status | Evidence |
|---|---|---|---|
| 9.1 | All eight substitution models agree with `scipy.linalg.expm` to machine precision | **V** | `scripts/portfolio_benchmarks.json` → `substitution_models[*].max_abs_diff`, range 1.4e-17 to 4.4e-16. Also asserted by `tests/test_gtr.py`. |
| 9.2 | The GTR audit fixed real errors that changed no inference result | **V** | `CHANGELOG.md:217-274`. The reason no result changed: nothing on the inference path used `GTR.py` (`CHANGELOG.md:226-229`). |
| 9.3 | Incremental MPL rescoring matches a cold rebuild bit-for-bit | **V** | `tests/test_mpl_incremental.py`; `CHANGELOG.md:569-573`. |
| 9.4 | Reversible-jump add/delete Hastings ratios cancel to zero | **V** | `reports/NSF_Annual_Report_Year1.md:203-206`. Locate the specific test before citing. `[TODO: cite the test]` |
| 9.5 | Three proposal moves had 0% historical acceptance and were fixed | **V** | `CHANGELOG.md:692-705` (`AddReticulation`, `FlipReticulation`, `ChangeReticDest`). |
| 9.6 | Simulate-and-recover works end to end | **V** | `tests/test_sim_seq.py::TestEndToEnd::test_recovers_planted_clade`; `Examples/sim_recovery.py`. |

## 10. Performance evidence

| # | Claim | Status | Evidence |
|---|---|---|---|
| 10.1 | `expt()` beats a fresh `scipy.linalg.expm` call by 8.5-19.4x | **V** | `scripts/portfolio_benchmarks.json` → `substitution_models`. Re-run into the paper's own benchmark dir. Note this differs from the 2.9-5.4x in `CHANGELOG.md:282-288`; the changelog figures are older. Use the re-run. |
| 10.2 | Cython graph core beats NetworkX on incidence lookup | **V** | `portfolio_benchmarks.json` → `graph_core.100_leaves.ops`: `in_edges` 10.7x, `get_leaves` 3.7x, `out_degree` 1.28x. |
| 10.3 | Cython graph core loses to NetworkX on `topological_order` and on construction memory | **V** | Same JSON: `topological_order` ratio 0.72x; `memory_ratio_nx_over_phy` 0.578, i.e. NetworkX peak is 58% of PhyNetPy's. **Must be reported.** Contradicts the memory framing in `reports/NSF_Annual_Report_Year1.md:105-131`. |
| 10.4 | "PhyloNet ~40 min vs PhyNetPy ~30 sec" | **X** | Not supported as a general claim, and dropped from the paper. `runs/defj/phylonet_results.csv` (682 rows): PhyloNet median 13.5 s, mean 338.8 s, max 7200 s. The 7200 s figure is a **timeout cap**, i.e. censored. Replaced by the measured head-to-head in `benchmarks/phylonet.json`, where the largest observed advantage is about 6x. |
| 10.5 | "30 min to 10 sec workflow" | **X** | No artefact on disk supports this. Dropped from the paper. |
| 10.6 | SNP likelihood ~0.012 s for 3 taxa / 1,000 sites | **V** | Measured: **1.1-1.2 ms** warm, 1.6 ms cold (`benchmarks/snp.json`, `3taxa_1000sites`). The original lead was conservative by about 10x. |
| 10.11 | \phynetpy\ is faster than PhyloNet on matched pseudo-likelihood inference | **V** | `benchmarks/phylonet.json`: faster on 8 of 10 conditions, up to ~6x, margin growing with size. Slower on the two smaller single-reticulation conditions. Against PhyloNet's self-reported time, which excludes JVM start-up. |
| 10.12 | \phynetpy\ is **less accurate** than PhyloNet on the same benchmark | **V** | Same file: 8/30 exact recoveries against PhyloNet's 16/30. Cause unresolved; see §14.1. Must not be softened without new measurements. |
| 10.7 | MP-Allop-2 is more accurate than PhyloNet on DEFJ | **V** | `paper_figures/defj_summary.txt`: mean mu-distance 4.33 (MP-SA3, 1160 runs) vs 12.48 (PhyloNet, 669 runs); hardwired-cluster 2.70 vs 4.60. |
| 10.8 | MP-Allop-2 is faster than PhyloNet on DEFJ *on average* | **X** | False as stated. Same file: 187.4 s vs 205.5 s — a near-tie. Only a per-condition or scaling claim is defensible. |
| 10.9 | 20-taxon MPL search is tractable | **V** | `Examples/mpl_20taxa_search_demo.py`; `tests/test_mpl_20taxa.py`. Runtime not stated in the file. **X** for a number. |
| 10.10 | MCMC_GT on the 20-taxon benchmark takes about nine minutes for 1,000 iterations | **V** | Stated in `Examples/mcmc_gt_demo.py:17-19` and `README.md:90-92`. Re-measure before printing. **X** |

## 11. Documentation and examples

| # | Claim | Status | Evidence |
|---|---|---|---|
| 11.1 | 26 generated API reference pages | **V** | `docs/api/*.html`, counted 2026-09-25. Produced by `generate_docs.py`. |
| 11.2 | Project site with six pages | **V** | `docs/{index,documentation,demos,releases,news,advertisements}.html`. |
| 11.3 | Guides for installation, I/O, validation | **V** | `Guides/{INSTALLATION_GUIDE,IO_guide,VALIDATION_GUIDE}.md`. |
| 11.4 | Twelve runnable examples | **V** | `Examples/` holds 12 `.py` scripts. **Note a real bug:** the directory is `Examples/` but `README.md:85-92` links to `examples/`. Case-insensitive on Windows, broken on Linux. Flag to the author. |
| 11.5 | Changelog with migration tables for every breaking change | **V** | `CHANGELOG.md`, e.g. `:165-180`. |
| 11.6 | A dedicated project website with tutorials | **A** | Not stood up; `reports/NSF_Annual_Report_Year1.md:409-411`. `README.md:153` points at the lab site. |

## 12. Known limitations — all verified, all to be stated plainly

| # | Limitation | Evidence |
|---|---|---|
| 12.1 | `seq_engine_cy` is compiled but wired into nothing | No occurrence of `seq_engine_cy` anywhere in `src/*.py` (grep, 2026-09-25). |
| 12.2 | `MLE_BiMarkers` can score a network but cannot search | `src/_engines.py:619-622` — `infer` raises unconditionally. |
| 12.3 | Simulation under `Allopolyploid` is unimplemented | `src/_simulate.py:220-225`. |
| 12.4 | No across-site rate heterogeneity (+Gamma) | No categorised-rate machinery in `src/_seq_likelihood.py`; `reports/NSF_Annual_Report_Year1.md:100-103`. |
| 12.5 | `MCMC_SEQ` and `MCMC_BiMarkers` reject heterogeneous `branch_thetas` for topology search | `src/_engines.py:492-496,658-662`; `src/_mcmc_seq.py:3434-3435`. |
| 12.6 | `MCMC_SEQ` and `MP_Allop` reject `StartMode.AUGMENT` | `src/_engines.py:507-508,729-730`. |
| 12.7 | `Bayesian(objective=PseudoLikelihood())` is refused by design | `src/_engines.py:384-385,497-498,663-664`. A triplet pseudo-likelihood is not a normalised probability, so it gives no calibrated posterior. |
| 12.8 | `score(..., optimize=True)` unavailable for parsimony and for Bayesian | `src/_engines.py:544-545,580-581`. |
| 12.9 | Visualization is ASCII text only | See 4.4. |
| 12.10 | No network classification (tree-child, galled, normal); only `level` | `GraphUtils.__all__` has `level` and no classifier. |
| 12.11 | No set-level summarization (consensus networks, hybrid frequencies) | Absent from `src/`. `reports/NSF_Annual_Report_Year1.md:311-315`. |
| 12.12 | No PHYLIP I/O | See 5.3. |
| 12.13 | A C compiler is required to build from sdist | See 6.3. |
| 12.14 | `MDC(weighting=...)` raises; only unweighted MDC exists | `src/criteria/_objectives.py:103-109`. |
| 12.15 | `PseudoLikelihood(subsets=...)` supports only `"trinets"` | `src/criteria/_objectives.py:170-180`. |

## 13. Citation requirements

Each implemented method must cite its own primary source, not PhyloNet
generically. Sources named in `README.md:127-130` and `src/_engines.py`:

| Method | Primary source | Status |
|---|---|---|
| `InferNetwork_ML` | Yu et al. 2014 | **A** — need full reference |
| `InferNetwork_MPL` | Yu & Nakhleh 2015 | **A** |
| `MCMC_GT` | Wen & Nakhleh 2018 | **A** |
| `MCMC_SEQ` | Wen & Nakhleh 2018 | **A** |
| `MLE_BiMarkers` / `MCMC_BiMarkers` | Bryant et al. 2012; Zhu et al. 2018 | **A** |
| `MP_Allop` | Hejase et al.; Yan et al. 2022 | **V** — `paper/refs.bib` has `YanAllop` (doi 10.1093/sysbio/syab081). Hejase attribution needs checking; `README.md:124` says Hejase, the NSF report §Thrust 2 says Hejase, `paper/refs.bib` credits Yan. **Resolve before submission.** |
| MDC criterion | Than & Nakhleh 2009 | **V** — `paper/refs.bib` `ThanNakhlehMDC`. |
| PhyloNet | Than et al. 2008; Wen et al. 2018 | **V** — `paper/refs.bib` `PhyloNet`, `PhyloNetWheeler`. |
| mu-distance | Cardona et al. 2009 | **A** — named in `CHANGELOG.md:751-753`. |
| APD / WAPD | Yakici, Ogilvie & Nakhleh, RECOMB-CG 2022 | **A** — `CHANGELOG.md:764-766`. |
| MSC / coalescent | **A** | Standard source needed. |

Anything still **A** at drafting time gets `\cite{TODO_...}` plus a stub entry
in `references.bib`.

## 14. Issues found while writing the paper

These surfaced from running the API rather than reading it. Each is reproducible
and each affects what the manuscript may claim. **Items 14.1 and 14.2 should be
resolved before submission**, because a reviewer reproducing the benchmarks will
hit both.

### 14.1 The pseudo-likelihood search returns networks far worse than the truth

Reproduce:

```python
gts = simulate(MSC(theta=0.5), taxa=8, n=200, data="gene_trees", seed=20260925)
score(gts.true_network, gts, model=MSC(theta=0.5), criterion=PseudoLikelihood())
# -8259.315
infer(gts, criterion=PseudoLikelihood(), max_reticulations=0, seed=1).score
# -915339.363
```

The search reports a log pseudo-likelihood roughly two orders of magnitude worse
than a network it could have reached, and the value does not change when
`num_iter` is raised to 2,000, 10,000 or 50,000 — it is identical at 700 default
proposals and at 50,000. The recovered topology is close ($\mu = 1$), so this
looks like a scoring or branch-length-handling problem rather than a search that
is merely under-run. Candidate causes not yet separated: the inferred network's
branch lengths are left un-optimised; the score reported by the engine is not the
score the accept path used; or the unit mismatch in 14.2.

**Consequence for the paper:** the head-to-head accuracy result
(Section~\ref{sec:headtohead}) is reported as an observation with the cause
stated as unresolved. It should not be presented as a settled property of the
method. Across 30 trials (10 conditions x 3 replicates) PhyloNet recovered the
generating topology 16 times against PhyNetPy's 8.

### 14.2 `PseudoLikelihood` does not enforce branch-length units

`Likelihood` refuses a network whose unit is `SUBSTITUTIONS_PER_SITE`
(`require_branch_length_unit` at `src/_units.py:144`), which is the intended
guard. `PseudoLikelihood` declares `use_branch_lengths = False` and so accepts
the same network without complaint, scoring substitution-scale lengths as
coalescent units. On the network simulated above the two readings differ by a
factor of `2 / theta`, and the resulting scores differ by roughly 800 log units.
The guard described as a safeguard in the architecture section therefore covers
one criterion and not the other.

### 14.3 Triplet probabilities can exceed 1, making the score positive

`_mpl.py` documents the score as "always <= 0 since it is a sum of `rho * log(p)`
terms" (`:212-213`, `:2846`). It is not, when the network carries many
reticulations relative to taxon count:

```python
gts = simulate(MSC(theta=0.02), taxa=6, n=100, data="gene_trees", seed=3)
r = infer(gts, criterion=PseudoLikelihood())   # default reticulation budget
r.score            # +723.90, with 7 reticulations on 6 taxa
```

`score_species_network_triplets` then reports individual triplet probabilities of
1.527 and 1.167 — 6 of 20 triplets exceed 1. `calculate_triple_probability` is
returning values outside $[0,1]$, so `log(p) > 0`. The third probability is
clamped by `max(1 - p_xy - p_xz, 0)` but the first two are not.

**Consequence for the paper:** no reported figure is affected, because every
benchmark run here used a reticulation budget matched to the generating network.
Verified by auditing all benchmark JSON: every log-likelihood and
log-pseudo-likelihood value recorded is $\le 0$. The manuscript does not describe
the pseudo-likelihood as bounded above by zero.

### 14.4 The default simulation scale produces no ILS

`simulate(MSC(theta=0.02), taxa=6, n=200)` yields 200 gene trees with a single
distinct topology, identical to the species tree. Any inference benchmark built
on it measures nothing. Concordance against theta, six taxa, 200 gene trees,
seed 3:

| theta | concordant | distinct topologies |
|---|---|---|
| 0.02 | 200/200 | 1 |
| 0.2 | 158/200 | 5 |
| 0.5 | 102/200 | 25 |
| 1.0 | 45/200 | 62 |
| 2.0 | 16/200 | 105 |

The paper's benchmarks use `theta = 0.5` and record per-condition concordance so
the difficulty is visible. Worth considering whether the documented examples,
which use `theta=0.02`, should change too.

### 14.5 `README.md` code does not run

- `read_newick(...)[0]` appears twice (`README.md:81`, and in `src/__init__.py:37`
  as `read_newick("((A,B),C);")`). `read_newick` returns a `Network`
  (`src/IO.py:1082`), which is not subscriptable; `read_newick_file` is the one
  that returns a list. Both docstring examples raise `TypeError`.
- `README.md` links to `examples/quickstart.py` and five siblings, but the
  directory is `Examples/`. Case-insensitive on Windows, broken on Linux.
- `GeneTrees.trees` is a `set`, so it cannot be indexed; worth a docstring note
  since the natural guess is a list.

## 15. Open questions for the author

1. Items 14.1 and 14.2 above, which block a clean accuracy claim.
2. `MP_Allop` attribution — Hejase or Yan? (§13, last row.)
3. The 12 `MCMC_SEQ` proposal operators need enumerating to print "22". (§3.7)
4. Archival DOI. (§0.5)
5. Confirm the 1.0.0 feature freeze is exactly the two cells in §0.2 and §0.3.
6. PhyloNet 3.8.2 rows marked `[TODO]` in `tables/comparison.tex`
   (visualisation, rate heterogeneity, its own testing and CI, license).
