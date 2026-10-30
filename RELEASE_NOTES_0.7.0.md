# PhyNetPy 0.7.0

*Release date: 2026-10-30*

PhyNetPy 0.7.0 adds DIMPLE, a divide-and-conquer pipeline for inferring large
networks from a tree of blobs, and a joint reticulation-and-tree dissimilarity
for comparing networks. Underneath, network edits are now transactional and
every fixed-seed search reproduces across processes. It is an alpha release
with one small breaking change; see the migration notes.

## Highlights

- `infer_dimple`: partition a tree of blobs, infer each subset under MSC
  triplet pseudo-likelihood, merge the subset networks, and graft the pieces
  back onto the scaffold.
- `joint_dissimilarity`: a dissimilarity on rooted networks that scores
  matched reticulations and the reticulation-free remainders separately, under
  leaf, cluster, or Nakhleh structure comparison.
- `hybrid_ancestry_error`: the balanced hybrid-ancestry error (HAE) of an
  estimated network against the truth.
- Displayed-tree enumeration returns distinct topologies with the major tree
  first, over exactly the network's own leaves.
- `Network` transactions: journaled, nestable, in-place undo for any sequence
  of edits. Every topology move now uses them instead of copying the network.
- Process-independent node and edge hashes, so a seeded search gives the same
  answer in every process.
- Annotated Newick output (`annotate=True`) that the readers parse back,
  including inheritance probabilities and non-standard reticulation labels.

## DIMPLE

```python
from phynetpy.data import GeneTrees
from phynetpy.infer import infer_dimple
from phynetpy.IO import read_newick_file

gene_trees = GeneTrees.from_file("gene_trees.txt", format="newick")
tree_of_blobs = read_newick_file("tob.tre")[0]     # polytomies are the blobs
start_tree = read_newick_file("start.tre")[0]      # binary, same taxa

result = infer_dimple(
    gene_trees, tree_of_blobs, start_tree,
    max_subset_size=12, n_divisions=15, max_reticulations=1,
    model_selection="gain", outgroup="OUT",
)
print(result.best.newick(), result.score)
trace = result.raw        # DimpleTrace: every blob, subset, chain, and trial
```

Each subset is searched by simulated annealing, three runs by default; any
`infer` keyword (`num_iter`, `n_runs`, `method`, ...) overrides that, and
`chains=[...]` runs a portfolio per subset. `infer_dimple` is not a cell of
the `(data, model, criterion)` registry: the partition and the stitch are
specific to this pipeline.

## Comparing networks

```python
from phynetpy import read_newick, joint_dissimilarity, hybrid_ancestry_error, mu_distance

estimate = read_newick("((A,(B)#H1),(#H1,C));")
truth = read_newick("(((A)#H1,B),(#H1,C));")

r = joint_dissimilarity(estimate, truth, kind="nakhleh")
r.D_ret, r.D_tree, r.D_joint_normalized       # matching cost, remainder, in [0, 1]
hybrid_ancestry_error(estimate, truth)        # asymmetric: estimate first
mu_distance(estimate, truth)
```

## Breaking changes

### `phynetpy.Logger` is removed

The debug logger had a single consumer, `SwitchParentage`, which no longer
logs. Remove the import; `SwitchParentage(debug_id=...)` is still accepted
and ignored.

### `State` holds one live model

`State.current_model` and `State.proposed_model` are now properties that
return the same live model, and `State.likelihood()` returns the cached
accepted score rather than re-scoring. Custom drivers that kept a reference
to `current_model` as a frozen snapshot must take one with `Network.copy()`
instead. The built-in drivers and engines already do.

## Migration notes

- Replace `from phynetpy import Logger` with your own logging.
- Code that relied on the iteration order of `Network.V()` or `Network.E()`
  will see a different (now stable) order; nothing that sorts is affected.
- `random_network(n, level)` may return fewer than `level` reticulations on a
  small topology, with a warning, instead of raising. Use `n >= 5` for
  `level >= 1`.

## Packaging and compatibility

- Supported Python versions: 3.9 through 3.14.
- Cython extensions are required; wheels avoid a local compiler, while source
  installs need a working C compiler.
- Package metadata is defined in `pyproject.toml`; `src/_version.py` is the
  version source of truth.

## Verification

The release process runs the default suite, slow recovery/stress tests,
PhyloNet numerical cross-checks when their external dependencies are present,
package build validation, and clean-install smoke tests. New test modules
cover DIMPLE, the joint dissimilarity, and network transactions.

For the exhaustive list of changes, see [`CHANGELOG.md`](CHANGELOG.md).
