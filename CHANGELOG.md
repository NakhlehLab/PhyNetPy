# Changelog

All notable changes to PhyNetPy will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Removed -- **BREAKING**: the component/visitor model-building layer

The probabilistic-graphical-model scaffolding around `Model` has been removed.
It advertised an extension surface nobody could use: `ModelFactory` resolved
component dependencies through a priority queue to order three components whose
order was hardcoded at both call sites, and `Visitor` and `Strategy` had exactly
one subclass each, both private to the biallelic-marker likelihood.

- **`phynetpy.ModelFactory` and `phynetpy.ModelTraversal` are gone**, along with
  `ModelFactory`, `ModelComponent`, `NetworkComponent`, `MSAComponent`,
  `Traversal`, `TraversalOrder`, `Visitor`, and `Strategy`.
- **The `ModelNode` hierarchy is no longer public.** `ModelNode`, `LeafNode`,
  `InternalNode`, `ReticulationNode`, `RootNode`, and `RootAggregatorNode` moved
  to the private `phynetpy._snp_model`, whose only consumer is `BiMarkers`. That
  graph is rebuilt on every likelihood evaluation and was never the same object
  as the `Model` the search loop mutates, so nothing outside the biallelic
  algorithm could meaningfully build or walk one.
- **`phynetpy.infer.MPAllopComponent` is gone.** Its `build` performed two
  assignments; `_infer_mp_allop.build_mp_allop_model` does them directly.
- `ModelGraph` is now just `Model` and `ModelError`.

The extension surface that methods actually use is unchanged: attach a scorer
with `Model.set_likelihood_calculator`, propose topologies with `Move`
subclasses or a custom `ProposalKernel`, and drive the search with
`HillClimbing` or `SimulatedAnnealing`.

Also removed, having had no callers: `Model.summary`, `Model.execute_move`,
`Model.get_root`, `State.write_line_to_summary`, `BiMarkers.build_model`, and
the unused `ModelNode` accessors (`unjoin`, `remove_child`, `remove_parent`,
`get_model_parents`, `in_degree`, `out_degree`, `set_branch`, `get_samples`).

### Changed -- **BREAKING**: inference is now two verbs on three axes

Every inference entry point has been replaced by `infer` and `score`. What used
to be encoded in a command name is now a typed argument on one of three
independent axes, so choosing a method is choosing arguments rather than
learning a new call:

```python
from phynetpy.infer import infer, score
from phynetpy.data import GeneTrees
from phynetpy.models import MSC, Allopolyploid
from phynetpy.criteria import MDC, Likelihood, PseudoLikelihood, Bayesian

gts = GeneTrees.from_file("gene_trees.nex", mapping)
result = infer(gts, model=MSC(), criterion=PseudoLikelihood())
```

Three new subpackages hold the axes. They are new subpackages rather than
top-level names because `phynetpy.ModelGraph.Model` and
`phynetpy.ModelSelection.Criterion` already exist and are unrelated; the bare
names `Model` and `Criterion` are deliberately *not* re-exported at top level.

- **`phynetpy.data`** -- `GeneTrees`, `Alignment`, `BiallelicMarkers`. The input
  knows its own type, so you never declare it. `GeneTrees.has_branch_lengths` is
  new, and is computed from the source text at parse time because
  `phynetpy.IO` back-fills missing lengths with `1` and a parsed network can no
  longer tell you whether its lengths were observed or invented. `from_file` and
  `from_newick` constructors wrap the existing readers.
- **`phynetpy.models`** -- `MSC(theta=...)` and `Allopolyploid(subgenome_map=...)`.
  Previously the generative process was implicit in which scorer you picked.
- **`phynetpy.criteria`** -- `MDC`, `Likelihood`, `PseudoLikelihood`, and
  `Bayesian`, which *wraps* an objective (`Bayesian(objective=Likelihood())`)
  rather than sitting parallel to the likelihoods, because MCMC is not a fourth
  thing to optimise. This is what the `pseudo=True` kwarg becomes.

**Dispatch is a registry that doubles as an executable validity matrix.**
Whether a run is legal depends on the whole `(data, model, criterion)` triple,
so the three failure modes are now distinguished instead of conflated:
`TypeError` when the criterion is not defined on that data type, `ValueError`
when it needs branch lengths the data lacks, and `NotImplementedError` when the
combination is meaningful but nobody has built it. `validity_matrix()` prints
the table; `register` lets you add a cell.

Two cells are legal and deliberately unregistered, so they raise
`NotImplementedError` rather than pretending: MDC under the MSC (PhyloNet's
`InferNetwork_MP` -- the only parsimony code here is defined on multiple-labelled
trees for the allopolyploid model) and the biallelic pseudo-likelihood
(`MLE_BiMarkers -pseudo`). `Bayesian(objective=PseudoLikelihood())` is also
refused: a triplet pseudo-likelihood is not a normalised probability of the
data, so using it as an MH target does not give a calibrated posterior.

**`infer` returns one type regardless of criterion.** `InferenceResult` carries
`.best`, `.score`, `.posterior` (populated only for Bayesian runs), and `.trace`,
replacing five incompatible return types (`MCMCGTResult`, `InferNetworkMLResult`,
`MCMCSeqResult`, `MultiChainResult`, plus bare `float`, `dict[Network, float]`,
and `int`). Nothing is lost: the engine's native object stays reachable as
`.raw`, and attribute access falls through to it, so `result.summary()`,
`result.write_log(path)`, `result.information_criteria()` and
`result.reticulation_posterior()` keep working.

**A starting phylogeny is now a `Start`, not just a network.** `Start(net)` (or a
bare network) is a free starting point, as before. `Start(net,
StartMode.AUGMENT)` is new and stronger: the *result* must contain that network.
It is enforced by a cluster-containment validator in the accept path, not merely
approximated by dropping moves -- adding reticulations preserves a network's
clusters, so this is a real constraint on the search space. `MCMC_SEQ` and the
allopolyploid search reject it explicitly, since their kernels cannot honour it.

**`score(..., optimize=True)` now works for every criterion**, not just
`InferNetwork_ML`. It routes through the already scorer-agnostic
`optimize_network_parameters`, and refuses the request for parsimony rather than
silently ignoring it.

#### Migrating

| Was | Now |
| --- | --- |
| `MPL(net, gts, map).search(...)` | `infer(gts, criterion=PseudoLikelihood(), start=net)` |
| `MPL(net, gts, map).score()` | `score(net, gts, criterion=PseudoLikelihood())` |
| `InferNetwork_ML(...).search(...)` | `infer(gts, criterion=Likelihood(), start=net)` |
| `InferNetwork_ML(...).score(optimize=True)` | `score(net, gts, criterion=Likelihood(), optimize=True)` |
| `MCMC_GT(...).search(method="mh")` | `infer(gts, criterion=Bayesian())` |
| `MCMC_GT(...).search(pseudo=True)` | `infer(gts, criterion=PseudoLikelihood())` |
| `MCMC_GT(...).score(posterior=True)` | `score(net, gts, criterion=Likelihood())` (see below) |
| `MCMC_SEQ(loci, map).search(...)` | `infer(Alignment(loci, map), criterion=Bayesian())` |
| `INFER_MP_ALLOP(path, ...)` | `infer(gts, model=Allopolyploid(), criterion=MDC())` |
| `ALLOP_SCORE(path, ...)` | `score(net, gts, model=Allopolyploid(), criterion=MDC())` |
| `MCMC_BIMARKERS(path, ...)` | `infer(BiallelicMarkers.from_file(path), criterion=Bayesian())` |
| `SNP_LIKELIHOOD(path, net, ...)` | `score(net, BiallelicMarkers.from_file(path), criterion=Likelihood())` |

Three notes on the table. Search settings that are *about the objective* moved
onto the criterion (`Bayesian(chain_length=..., burnin=..., prior=...)`), while
generic search controls (`max_reticulations`, `num_iter`, `method`, `seed`,
`preset`) stay as keyword arguments on the verb. The path-taking wrappers are
gone rather than refactored -- the internals were always object-based, so the
data-axis constructors replace them. And `Bayesian` is not scorable on its own,
because a Bayesian score of one fixed network collapses to its objective;
`score` says so and points you at `.objective`.

The implementation classes still exist in their private modules (`phynetpy._mpl`,
`phynetpy._mcmc_gt`, `phynetpy._mcmc_seq`, `phynetpy._infernetworkml`) for engine
authors and for tests that exercise kernel internals, but they are no longer the
public API and are not re-exported.

### Added -- `simulate`, the generative inverse

`simulate(model, network, n, data=...)` runs the same axes backwards: it takes a
model and a network and returns a data-axis object, so a recovery or null-model
check composes with no glue code.

```python
sim = simulate(MSC(theta=0.02), true_net, n=200, data="gene_trees")
recovered = infer(sim, criterion=PseudoLikelihood())
```

It dispatches over the existing simulators (`_sim_seq.simulate_gene_tree` /
`simulate_multilocus`, `SNPSimulator.simulate`) rather than duplicating them,
and returns `GeneTrees`, `Alignment`, or `BiallelicMarkers`. Omitting the
network draws a species tree under a pure-birth process instead (`taxa=`,
`birth_rate=`, via `BirthDeath.Yule`). The generating network is attached to
the result as `.true_network`.

Note that the simulators read branch lengths in expected substitutions per site
while the gene-tree criteria read them in coalescent units, so a round trip
recovers the topology but not the lengths.

### Fixed -- substitution models (`phynetpy.GTR`)

An audit of every formula in `GTR.py` against the primary sources. The rate
matrix was not time reversible for non-uniform base frequencies, `e^(Q*t)` was
computed with a symmetric eigensolver on a non-symmetric matrix, and three of
the eight models could not be constructed at all. Every model is now verified
against `scipy.linalg.expm` to machine precision (~1e-14) by 339 new tests in
`tests/test_gtr.py`.

Nothing on the inference path used these classes -- `State.bootstrap` accepted a
`submodel` argument and ignored it, and the sequence-likelihood engine has its
own (correct) implementation in `_seq_likelihood.py` -- so no inference result
changes. The two implementations now agree exactly, which cross-validates both.

- **`buildQ` mapped exchangeabilities to the wrong cells.** The upper triangle
  used `trans[i + j]` and the lower `trans[i + j - 1]`, so `trans[0]` never
  reached the upper triangle, `trans[5]` never reached the lower, and
  `trans[2]`/`trans[3]` were each used twice. Mirrored cells therefore received
  *different* parameters and Q was not symmetric even when it had to be. Indices
  now use upper-triangular row-major order (`AC, AG, AT, CG, CT, GT`), matching
  what every subclass's equivalency pattern already assumed and what PhyloNet's
  `-gtr` flag uses.
- **`buildQ` omitted the stationary frequency factor.** Off-diagonals were
  `r_ij` in the upper triangle and `r_ij * pi_j / pi_i` in the lower, rather than
  `Q[i][j] = r_ij * pi_j` throughout. Consequences for any model with free base
  frequencies (F81, HKY, TN93, GTR): Q was not reversible, and `pi` was **not**
  its stationary distribution, so the frequencies a user supplied were not the
  ones the model equilibrated to. JC, K80, K81 and SYM were unaffected, since
  uniform frequencies make the two forms coincide.
- **`expt` called `numpy.linalg.eigh` on a non-symmetric matrix.** `eigh` reads
  only the lower triangle, so for non-uniform frequencies it silently
  decomposed the wrong matrix: rows of the resulting `P(t)` summed to values
  like 1.97 instead of 1. Q is reversible but not symmetric, so it is now
  decomposed via the similarity transform `B = P^(1/2) Q P^(-1/2)`, which *is*
  symmetric -- the same approach `_seq_likelihood.py` and BEAST use. The
  decomposition is also cached now, as the docstring already claimed.
- **`K80.expt` returned a constant matrix.** It assigned
  `0.25 * (1 - 2e^(-4t)) + e^(-8*beta*t)` to *every* entry, with no distinction
  between the diagonal, transitions, and transversions, and ignored `alpha`
  entirely. `P(0)` was all `0.75` rather than the identity. Replaced with the
  Kimura 2-parameter closed form.
- **`K80`, `F81` and `SYM` could not be constructed.** All three built
  parameters as `np.ones((n, 1))`, whose elements are length-1 arrays rather
  than scalars, so `buildQ` raised `ValueError: setting an array element with a
  sequence`. Inputs are now flattened, so column vectors are accepted too.
- **Base frequencies were compared to 1 with `==`.** `[0.4, 0.3, 0.2, 0.1]` sums
  to `0.9999999999999999` in IEEE-754 and was rejected as malformed. Now uses
  `math.isclose`.
- **`K80.set_hyperparams` had no effect on Q.** It updated the rate parameters
  but rebuilt Q from the stale rate list. It now regenerates the rates.
- **`F81.set_hyperparams` was unusable.** Its `_disable_for_subclass` decorator
  compared a *type* to a *string* (`type(self) is not "F81"`), which is never
  false, so every call raised -- including on `F81` itself. The decorator has
  since been removed outright (see below).
- **F81's commented-out closed form was the Jukes-Cantor formula** -- it
  hardcoded `0.25` and `e^(-4t/3)`, discarding the free base frequencies that
  are the entire point of F81. Correctly disabled, now correctly implemented.

### Added -- closed-form transition matrices

Six of the eight models have an exact closed form for `e^(Q*t)` and no longer
touch a matrix exponential. Each is validated against `expm` across branch
lengths from `1e-4` to `200`, plus Chapman-Kolmogorov (`P(s)P(t) == P(s+t)`),
convergence to `pi`, and the model-nesting identities.

| Model | Closed form | Speedup vs `expm` |
|---|---|---|
| JC | `P_ii = 1/4 + 3/4 e^(-4t/3)` | 5.4x |
| F81 | `P = e^(-ut) I + (1 - e^(-ut)) 1 pi^T`, `u = 1/(1 - sum pi^2)` | 4.9x |
| K81 | Klein four-group characters (K3ST) | 4.5x |
| K80, HKY, TN93 | Tamura-Nei (1993) equations | 3.3x |
| SYM, GTR | none -- cached eigendecomposition | 2.9x |

A single Tamura-Nei kernel covers K80, HKY and TN93, since K80 is HKY with
uniform frequencies and HKY is TN93 with one shared transition rate; F81 is also
a special case but keeps its simpler dedicated form. SYM and GTR have all six
exchangeabilities free and have no closed form, but still beat a fresh `expm`
call because the eigendecomposition is cached.

### Changed -- models take their own parameters (**breaking**)

Every constrained model now takes exactly the free parameters its literature
defines, instead of a full 6-element exchangeability list that the caller had to
lay out in a model-specific pattern. Only `SYM` and `GTR` leave all six rates
free, so only they still take a list.

| Before | After |
|---|---|
| `K80(alpha, beta)` | `K80(kappa)` |
| `HKY(pi, [1, k, 1, 1, k, 1])` | `HKY(pi, kappa)` |
| `TN93(pi, [1, kr, 1, 1, ky, 1])` | `TN93(pi, kappa_r, kappa_y)` |
| `K81([b, a, g, g, a, b])` | `K81(alpha, beta, gamma)` |
| `JC()`, `F81(pi)`, `SYM(rates)`, `GTR(pi, rates)` | unchanged |

Migration, in each case a mechanical rewrite:

- **`K80`** -- pass `kappa = beta / alpha` *using the old argument names*. The
  old `alpha` was the *transversion* rate and `beta` the *transition* rate, the
  reverse of Kimura's notation, so `K80(0.2, 0.8)` becomes `K80(4.0)`. The old
  two-positional call now raises `TypeError` rather than silently inverting the
  ratio.
- **`HKY` / `TN93`** -- pass the rate ratios that were sitting at indices 1 and 4
  of the list. `HKY(pi, [1, 3.5, 1, 1, 3.5, 1])` becomes `HKY(pi, 3.5)`;
  `TN93(pi, [1, 5, 1, 1, 2, 1])` becomes `TN93(pi, 5.0, 2.0)`.
- **`K81`** -- note the reordering: the list was `[beta, alpha, gamma, gamma,
  alpha, beta]`, so `K81([1, 5, 2, 2, 5, 1])` becomes `K81(5.0, 1.0, 2.0)`.
  `alpha` is the transition rate, `beta` the A-C/G-T transversion class, and
  `gamma` the A-T/C-G class, following Kimura (1981).

Every old call either raises or is a compile-time-obvious rewrite; none of them
silently changes meaning.

Rationale:

- **The old signatures implied free parameters the models do not have.** HKY has
  one rate parameter, not six, and asking for six meant the redundant entries had
  to be validated for mutual consistency -- and could be got wrong. The new
  signatures make an invalid equivalency pattern *unrepresentable*, which
  deleted three `_is_valid` overrides whose only job was to reject one.
- **Only ratios are identifiable.** Q is normalized to unit mean rate, so the
  overall scale of any rate argument is discarded. Taking two rates for K80
  implied two degrees of freedom where there is one, which is what forced the old
  `alpha + beta == 1` constraint -- an arbitrary scale convention users had to
  satisfy by hand. Every rate argument is now explicitly scale free:
  `K81(1, 5, 2)` and `K81(2, 10, 4)` are the same model.
- **It matches the literature and the ecosystem.** BEAST, MrBayes, RAxML, PAML
  and PhyloNet all parameterize K80/HKY by kappa and TN93 by two kappas.
- **It matches this codebase.** `_seq_likelihood.HKY85` already took
  `(kappa, pi)`, so the two substitution-model hierarchies now agree on how the
  transition/transversion ratio is expressed.
- **The old K80 names meant the opposite of the literature.** `alpha` denoting
  transversions actively misleads anyone comparing against Kimura's paper.

Kimura's K80 notation remains available, now correctly oriented, as the read-only
properties `K80.alpha` (`kappa / (kappa + 2)`) and `K80.beta`
(`1 / (kappa + 2)`). These satisfy `alpha / beta == kappa` and
`alpha + 2 * beta == 1`, and are exactly the corresponding entries of Q
(`alpha == Q[A][G]`, `beta == Q[A][C]`). `K81.alpha`/`beta`/`gamma` read back the
values passed in; use `getQ()` for their normalized counterparts.

### Changed -- `set_hyperparams` rejects unknown parameter names

Each model class now declares the names it accepts in a `HYPERPARAMS` tuple, and
`set_hyperparams` raises on anything else instead of ignoring it:

| Model | `HYPERPARAMS` |
|---|---|
| `JC` | *(none)* |
| `K80` | `kappa` |
| `F81` | `base frequencies` |
| `HKY` | `kappa`, `base frequencies` |
| `TN93` | `kappa_r`, `kappa_y`, `base frequencies` |
| `K81` | `alpha`, `beta`, `gamma` |
| `SYM` | `transitions` |
| `GTR` | `states`, `base frequencies`, `transitions` |

This fixes a silent desync. `JC` and the other closed-form models inherited the
general setter, so `JC().set_hyperparams({"transitions": [1, 5, 1, 1, 5, 1]})`
would rebuild Q while `expt` kept returning Jukes-Cantor probabilities --
`getQ()` and `expt()` then described different models with no error. Unsettable
names are now reported, and a test asserts `expt(t) == expm(getQ() * t)` after
every accepted update.

### Removed -- redundant machinery in `phynetpy.GTR`

- **`_disable_for_subclass`** -- a decorator guarding `F81.set_hyperparams`
  against subclasses. It compared a *type* to a *string*
  (`type(self) is not "F81"`), which is never false, so it in fact broke
  `F81.set_hyperparams` on `F81` itself. No class in the module subclasses F81
  (the hierarchy is flat -- every model derives from `GTR` directly), so rather
  than repair a guard for a case that does not exist, it is deleted.
- **The `_is_valid` overrides on `HKY`, `K81` and `TN93`** -- all three existed
  only to reject exchangeability lists that violated the model's equivalency
  pattern, which the new constructors make impossible to express. Base-class
  frequency and length validation still applies.
- **Seven near-identical `set_hyperparams` implementations** -- collapsed onto a
  single base implementation that validates names and delegates to a small
  per-model hook.
- **Redundant re-validation** in `F81.__init__` and `SYM.set_hyperparams`, which
  repeated checks the base class had already performed.

### Changed -- package structure and public API

This is a housekeeping pass that shrinks the public surface and removes dead
code. Behaviour of every inference method is unchanged; the test suite is
unaffected.

- **`phynetpy` top level is now explicit.** The five wildcard re-exports
  (`GraphUtils`, `BiMarkers`, `ModelGraph`, `ModelFactory`, `infer`) are gone,
  replaced by named imports and an `__all__`. The namespace went from an
  unbounded star-import union to 148 declared names. `BiMarkers` alone had been
  leaking 68 internals such as `MockCuda`, `GPU_SPECS`, `state_dim`, and the
  sparse split/merge tensor builders.
- **Fixed `phynetpy.GTR` name collision.** `from .GTR import GTR` followed by
  `from .infer import *` meant `phynetpy.GTR` silently resolved to
  `_seq_likelihood.GTR` *and* shadowed the `phynetpy.GTR` module, so
  `phynetpy.GTR.JC` raised `AttributeError`. `phynetpy.GTR` is now the module
  again. The model-graph class is `phynetpy.GTR.GTR`; the sequence-likelihood
  class is `phynetpy.infer.GTR`. `JC`/`K80`/`HKY`/`F81`/`K81`/`SYM`/`TN93` are
  re-exported at the top level, but the ambiguous bare `GTR` name is not.
- **One front door for inference.** `phynetpy.infer` is the only supported
  inference entry point. The `phynetpy.MPL`, `phynetpy.MCMC_GT`, and
  `phynetpy.Infer_MP_Allop` compatibility shims are removed --
  **migrate `from phynetpy.MPL import MPL` to `from phynetpy.infer import MPL`.**
  Implementation-private helpers (e.g. `_HAS_CYTHON_MPL`, `_TripleDPEngine`,
  `_GTLikelihoodEngine`, `allele_map_set`) come from the underscore modules.
- **Modules declare their own public surface.** `GraphUtils` and `BiMarkers`
  now define `__all__`, so `import *` from them is well-defined.
- **No wildcard imports between modules either.** `GraphUtils`, `GeneTrees`,
  `BirthDeath`, `ModelFactory`, `Matrix`, and `BiMarkers` now name what they
  import. Two accidental dependencies fell out of this: `BirthDeath` had been
  getting `math` through `from .Network import *`, and `Network.from_newick`
  raised `NameError` instead of `NewickParserError` on a malformed comment
  block, because the exception was never imported.
- **Single source of truth for the version.** `setup.py` said `0.5.0`,
  `__init__.py` said `0.4.0`, and the installed metadata said `0.3.1`. The
  version now lives only in `src/_version.py`; `pyproject.toml` reads it
  statically and `phynetpy.__version__` re-exports it.
- **Packaging metadata moved to `pyproject.toml`.** `setup.cfg` is deleted and
  `setup.py` is now just the optional-Cython build hook. `requires-python` is
  corrected to `>=3.9` (the package uses PEP 585 generics at runtime, so the
  advertised 3.8 support could never have worked), and classifiers now cover
  3.9 through 3.13.
- **Generated API reference moved** from `src/docs/` (inside the shipped
  package) to `docs/api/`, next to the project site.
- **`generate_docs.py`** now excludes all underscore-prefixed modules by rule
  rather than a hand-maintained list of three, folds the symbols
  `phynetpy.infer` re-exports onto a single `api/infer.html` page (previously
  empty, because `infer.py` is pure re-exports), and prunes pages for deleted
  modules.

### Removed -- dead code

- **`LikelihoodStrategies`** (834 lines): a speculative data-type/strategy
  facade with zero callers anywhere in the repo, and a third competing way to
  start an inference run alongside `phynetpy.infer` and the top-level names.
- **`Executor`** (1164 lines): a CPU/GPU backend abstraction whose only
  remaining tie to the codebase was a type annotation on `SNP_LIKELIHOOD`'s
  `executor` parameter, which was threaded through but never read. The
  parameter is removed too, since passing it had no effect.
- **`PhyloNet`** and `Network.compare_network`: shelled out to a
  `PhyloNetv3_8_2.jar` that is not in the repo, and `compare_network`
  unconditionally returned `0.0` after printing the subprocess output. Use
  `compare_networks` (from `ReticulationComparison`) or the `GraphUtils`
  distances instead.
- **`NetworkMoves`** (636 lines): deprecated at 0.5.0 and documented as unfit
  for Metropolis-Hastings (in-place mutation, unseeded `random`, no Hastings
  ratio). Only `add_hybrid` had a caller; it now lives in `GraphUtils`
  alongside the other structural edits. `remove_hybrid`, `nni`,
  `node_height_change`, `spr`, and `permute_leaves` are gone -- use the
  `ModelMove` classes.
- **`MetropolisHastings.MetropolisHastings`** (150 lines): never instantiated.
  The real samplers are `MCMC_GT`, `MCMC_SEQ`, and `MCMC_BIMARKERS`;
  `HillClimbing` and `SimulatedAnnealing` remain.
- **`graph_core`**: an orphaned Cython-shim facade duplicating logic already
  inlined in `Network.py`, which is what actually loads the accelerator.
- **Dead MSC dynamic program in `_mcmc_gt`** (221 lines): `_msc_log_prob_tree`,
  `_apply_branch_coalescent`, and `_combine_configs` were superseded by the
  bitmask versions in `_msnc_density`. They called `_enum_coarsenings` and
  `_linear_extensions`, which are not even imported -- they would have raised
  `NameError` if reached.
- **`ModelGraph.vec_bin_array`**, **`State.acyclic_routine`** (superseded by
  `network_invariants_routine`), **`GeneTrees.external_naming`** (carried a
  "remove if not needed" TODO), and `Traversal.LevelParallelTraversal`.
- **109,397 lines of generated Cython C** (`src/cython/*.c`, 5.4 MB) that
  `setup.py` never compiled -- it cythonizes from `.pyx`. Now gitignored.
  `nodeset_cy.c` had no corresponding `.pyx` at all, and `network_cy.pyx` was
  neither built nor imported.
- **Stale snapshots and one-off probes**: the `1.1/` pre-restructure snapshot
  (17 files), `MPLTest/` (18 files, unreferenced CAMUS sandbox), `demo_runs/`
  (6 probes superseded by `examples/sim_recovery.py`), and 62 ad-hoc
  `scripts/_*.py` investigation probes.
- Tracked `.DS_Store` files (now gitignored) and the orphaned
  `src/docs/Sync.html`, which documented a module that no longer exists.
- **The Numba CUDA shim in `BiMarkers`** (`MockCuda`, `NUMBA_CUDA_AVAILABLE`,
  `CUPY_AVAILABLE`): nothing referenced `cuda`, `float64`, `int32`, or `int64`
  once the hand-written CUDA kernels went away. GPU offload is CuPy-only and is
  gated on `CUPY_RUNTIME_OK`.
- **Exception classes nothing raises**: `MetropolisHastings.HillClimbException`,
  `MetropolisHastings.MetropolisHastingsException`, `Validation.FileFormatError`,
  and `Validation.DataIntegrityError`. Validation failures raise
  `ValidationError`.
- **`Network.print_graph`, `Network.pretty_print_edges`**, and
  `ModelMove.connect_nodes`: no callers. Use `GraphUtils.ascii` /
  `ascii_extended` to display a network.
- **A no-op in `EdgeSet.rehash_node`** that copied the hash map onto itself
  before the real rebuild, plus a `stale_keys` list comprehension whose
  predicate could never match. The pure-Python and Cython `EdgeSet`
  implementations now do the same thing.
- Unused accessors and locals across the search stack: `MPLKernel.phase`,
  `MPLKernel.phase_switches`, `MultiChainStatus.min_rhat_ok`, and roughly a
  hundred unused imports and dead assignments in `src/`, `tests/`, `scripts/`,
  and `examples/`.
- **`GraphUtils._retic_edge_choice`**: a recursive hybrid-edge chooser with no
  caller -- `get_all_subtrees` enumerates the same choices with
  `itertools.product`.
- **`_infer_mp_allop._nodes_to_improve`**: only ever called itself; `_attach`
  computes ploidy deficits from cluster minima instead.

### Consolidated

- **`Strategy`, `Visitor`, and `Traversal`** merged into **`ModelTraversal`**.
  These were three files of abstract interfaces used only together, by
  `BiMarkers` and `ModelGraph`; traversal order and per-node dispatch now live
  side by side with the guidance on choosing between them.
- **`Phylo.Branch`** moved into `Network`, next to `Node` and `Edge`. `Phylo`
  existed only to hold that one class.
- **Graph helpers that had been reimplemented per module now live in
  `GraphUtils`**, which is where a reader looks for them. Behaviour is
  unchanged; there is simply one copy of each:
  - `network_clusters` is now a documented `GraphUtils` export (it was a public
    name inside the private `_search_flags`, imported from there by four
    modules) and is defined in terms of the hardwired-cluster pass it always
    duplicated. `_leaf_labels_below` is the label-valued form of
    `Network.leaf_descendants_all`, replacing four hand-rolled memoised
    post-order walks in `_mcmc_seq`, `_infernetworkml`, and `_mpl`.
  - Ultrametric node heights (`_node_height`, `_node_heights`), the
    `parent -> child` edge lookup (`_edge_between`), and the fast structural
    snapshot (`_clone_net`) were each defined twice or three times across
    `_msnc_density`, `_mcmc_seq`, `_sim_seq`, and `ModelMove`, so
    `_seq_likelihood` no longer has to re-export `_node_height` to reach its
    consumers.
  - Ten inline `sum(1 for n in net.V() if n.is_reticulation())` expressions
    (plus `InferNetwork_ML._count_reticulations`) now call
    `count_reticulations`.
  - The degree-invariant scan shared by `State.network_invariants_routine` and
    the MCMC_GT proposal gate is `_valid_network_degrees`. Unlike
    `validate_binary` it tolerates polytomies, insists on a single root, and
    answers with one boolean, which is what both callers wanted.
  - Descendant walks in `SNPSimulator` and `MUL.to_mul` go through
    `Network.get_subtree_at`, and the MPL comparison report through
    `Network.leaf_descendants`.

### Added

- **Search presets** -- a single `preset=` argument shared by every
  gene-tree-topology method (`MPL`, `MCMC_GT`, `InferNetwork_ML`) that
  expands to a coherent bundle of behaviour flags so users no longer have
  to combine several booleans to get the result they want:
  `"default"` (recommended -- accurate r>=1 inference at ~baseline speed),
  `"fast"` (raw climb), `"accurate"` (per-topology optimisation of gammas
  + incident branches), and `"phylonet"` (reproduce PhyloNet's
  optimise-everything-per-topology behaviour for cross-checking).  Any
  individual flag passed explicitly overrides the preset.  Lives in
  `phynetpy._search_flags` (`SearchSettings`, `SEARCH_PRESETS`,
  `resolve_search_preset`) so the three methods stay in lock-step.
- **Per-topology continuous-parameter optimisation for `MPL`**
  (`optimize_params` / `optimize_scope` / `optimize_band`): reticulation
  topologies are judged near their parameter optimum during the climb
  (matching PhyloNet's per-round behaviour) instead of at the
  reticulation's birth gamma of 0.5.  The `"gamma"` scope fixes the
  systematic r>=1 accuracy gap at essentially no runtime cost.
- **Incremental MPL scorer** (`MPLScorer` lever 3): when only branch
  lengths / gammas change, the cached `_TripleDPEngine` + extracted Cython
  topology are reused and only the parameter arrays refreshed, instead of
  rebuilding the engine every call.  Validated bit-for-bit against full
  rebuilds (`tests/test_mpl_incremental.py`).
- **Unified inference search flags** for the gene-tree-topology methods
  (`MPL`, `MCMC_GT`, `InferNetwork_ML`): `opt_bl` (optimize branch
  lengths once at the end of the topology search via Brent coordinate
  ascent; drops the continuous-parameter moves during the search),
  `fix_st` (fix the starting tree backbone -- only reticulation
  add/remove/relocate and gamma moves are proposed, no `SPR`),
  `max_lvl` (cap the network level via `GraphUtils.level`; the
  authoritative guard runs in the accept path so it catches *every*
  level-raising move -- including reticulation relocation / endpoint
  moves that keep the reticulation count fixed -- while the
  level-raising moves themselves self-reject early as an efficiency
  layer), and `pseudo` (score with the triplet pseudo-likelihood
  `MPLScorer` instead of the full MSNC likelihood).
- **`phynetpy._optimize`** -- the Brent coordinate-ascent
  branch-length/gamma optimiser (formerly embedded in
  `_infernetworkml`) extracted into a scorer-agnostic module so all
  three methods share the `opt_bl` optimisation;
  `optimize_network_parameters` is re-exported from `_infernetworkml`
  for backward compatibility.
- **`phynetpy._search_flags`** -- shared `resolve_move_types` /
  `make_level_validator` helpers backing the new flags.

### Changed

- **Default search behaviour is now the `"default"` preset**, which turns
  on the near-free gamma optimisation.  This changes results for r>=1
  inference versus the previous raw-climb default (now reachable via
  `preset="fast"`): inferred networks are more accurate (better
  log-pseudo-likelihood and lower topological distance to truth) at
  approximately the same wall-clock time.  The flags controlled by a
  preset (`optimize_params`, `optimize_scope`, `opt_bl`, `fix_st`,
  `final_optimize`) now default to `None` and are filled from the preset;
  passing any of them explicitly overrides the preset as before.
- **`InferNetwork_ML` defaults to full-scope per-topology optimisation.**
  Because it maximises the full MSNC likelihood (where every branch length
  is identifiable and matters), an unset `optimize_scope` resolves to
  `"all"` for `InferNetwork_ML` rather than the cheaper `"gamma"` MPL
  default -- prioritising correctness over runtime.  Pass an explicit
  `optimize_scope` (e.g. `"gamma"`) to dial it back.

### Notes

- **Pseudo-likelihood scope.** The `pseudo` flag currently applies to
  **gene-tree inference only** (it swaps in the existing triplet
  `MPLScorer`).  Pseudo-likelihood is a general composite-likelihood
  technique and could be extended to BiMarkers/SNP (`BiMarkers`) and
  sequence (`MCMC_SEQ`) inference by decomposing over taxon subsets and
  computing the exact per-subset likelihood, but no such scorers exist
  yet, so `pseudo` is intentionally not wired into those methods for
  now.  Tracked as future work.
- The search flags are **not** applied to `INFER_MP_ALLOP`: its only
  move is `SwitchParentage`, so `fix_st` would leave it with nothing
  useful to do.
- **Level-1 scoring shortcut (investigated, not implemented).** When
  `max_lvl == 1` the scoring DP could in principle be cheaper
  (`count_displayed_trees` is exact for level-1, and per-blob displayed
  tree enumeration is small), but the MSNC ancestral-configuration DP
  and the triplet DP are already blob-aware and the shortcut would
  require correctness-sensitive changes inside the scorers themselves.
  It is deferred as future work rather than implemented now, since the
  `max_lvl` flag's correctness must not depend on it.

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
