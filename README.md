# PhyNetPy

PhyNetPy is a Python library for phylogenetic network inference and analysis. It
provides improved implementations of methods from PhyloNet, plus a framework for
building new Bayesian and simulation-based methods.

Current version: **v0.6.0** (see [CHANGELOG.md](CHANGELOG.md) for history).

## Install

```bash
pip install phynetpy
```

PhyNetPy's graph core and likelihood kernels are compiled with Cython. Standard
installation uses a published wheel when one is available; installing from the
source distribution requires a C compiler:

```bash
pip install .
```

See [`Guides/INSTALLATION_GUIDE.md`](Guides/INSTALLATION_GUIDE.md) for virtual
environment and IDE setup.

## Quick start

There are two things you can do with a phylogenetic network method -- find a
network, or evaluate one -- so there are two verbs, `infer` and `score`. What
they do is set by three independent arguments: the **data** you have, the
**model** of the biology, and the **criterion** you are optimising.

```python
from phynetpy.infer import infer, score
from phynetpy.data import GeneTrees
from phynetpy.models import MSC
from phynetpy.criteria import MDC, Likelihood, PseudoLikelihood, Bayesian

gts = GeneTrees.from_file("gene_trees.nex", {"A": ["A1", "A2"], "B": ["B1"]})

result = infer(gts, model=MSC(), criterion=PseudoLikelihood())
print(result.best, result.score)

log_lik = score(result.best, gts, model=MSC(), criterion=Likelihood())
```

Switching methods means changing one argument, not learning a new command.
`criterion=Bayesian()` samples a posterior instead of maximising, and
`result.posterior` is then populated; everything else about the call is the
same. Strings work as shortcuts where an axis needs no parameters, so
`infer(gts, criterion="MPL")` is the same call.

Whether a combination is legal depends on all three axes, so dispatch goes
through a registry that doubles as a validity matrix. An impossible request
fails as a `TypeError` and an unimplemented one as a `NotImplementedError`; the
two are never confused. `phynetpy.infer.validity_matrix()` returns that table,
built from the registry itself, and `registered_cells()` lists every implemented
combination:

```python
from phynetpy.infer import validity_matrix

for data, row in validity_matrix().items():
    print(data, row)
```

```text
GeneTrees       {'MDC': '-', 'Likelihood': 'InferNetwork_ML', 'PseudoLikelihood': 'InferNetwork_MPL', 'Bayesian': 'MCMC_GT'}
Alignment       {'MDC': 'x', 'Likelihood': '-', 'PseudoLikelihood': 'x', 'Bayesian': 'MCMC_SEQ'}
BiallelicMarkers {'MDC': 'x', 'Likelihood': 'MLE_BiMarkers', 'PseudoLikelihood': '-', 'Bayesian': 'MCMC_BiMarkers'}
```

A method name means implemented, `-` means legal but not yet implemented, and
`x` means the combination is not meaningful. The table above is for `MSC`; pass
a model class to tabulate another, such as `validity_matrix(Allopolyploid)`.

Data structures, I/O, and analysis helpers live at the top level:

```python
from phynetpy import read_newick, Network, Node, Edge, compare_networks

net = read_newick("(((A,B),C),D);")[0]
```

For the data structures and I/O, start with
[`examples/quickstart.py`](examples/quickstart.py). For inference, start with
[`examples/mpl_demo.py`](examples/mpl_demo.py) (scoring) or
[`examples/search_flags_demo.py`](examples/search_flags_demo.py), which runs the
same data under two criteria side by side. Both finish in seconds.
[`examples/mcmc_gt_demo.py`](examples/mcmc_gt_demo.py) shows Bayesian search but
takes about nine minutes, since the full gene-tree likelihood is the expensive
objective.

## Layout

| Path | What it holds |
| --- | --- |
| `src/` | The `phynetpy` package. |
| `src/infer.py` | The public inference API: `infer`, `score`, `simulate`, result types, diagnostics. |
| `src/data/`, `src/models/`, `src/criteria/` | The three axes those verbs dispatch on. |
| `src/_registry.py`, `src/_engines.py` | The validity matrix and one adapter per implemented cell. |
| `src/_*.py` | Method implementations. Private; reach them through `phynetpy.infer`. |
| `src/cython/` | Required compiled graph core and likelihood/scoring kernels. |
| `examples/` | Runnable end-to-end workflows. |
| `tests/` | Test suite (`pytest`; `-m "not slow"` skips long MCMC recovery runs). |
| `docs/` | Project site, plus the generated API reference in `docs/api/`. |
| `Guides/` | Installation, I/O, and validation guides. |
| `scripts/` | Benchmark and research tooling (not part of the library). |

## Inference methods

Under the multispecies network coalescent, `MSC()`, each data-and-criterion
pair is one method. The PhyloNet command each corresponds to is named so the
literature stays findable; `-` marks a combination that is meaningful but not
implemented, and `x` one that is not defined at all.

| `MSC()` | `MDC()` | `Likelihood()` | `PseudoLikelihood()` | `Bayesian()` |
| --- | --- | --- | --- | --- |
| `GeneTrees` | `-` | `InferNetwork_ML` | `InferNetwork_MPL` | `MCMC_GT` |
| `Alignment` | `x` | `-` | `x` | `MCMC_SEQ` |
| `BiallelicMarkers` | `x` | `MLE_BiMarkers` | `-` | `MCMC_BiMarkers` |

Allopolyploidy is a different model rather than a different command, so
maximum-parsimony allopolyploid inference (Hejase et al., PhyloNet's `MPAllopp`)
is `infer(gts, model=Allopolyploid(), criterion=MDC())`.

The references behind those cells: `InferNetwork_ML` (Yu et al., 2014),
`InferNetwork_MPL` (Yu & Nakhleh, 2015), `MCMC_GT` and `MCMC_SEQ` (Wen &
Nakhleh, 2018), `MLE_BiMarkers` and `MCMC_BiMarkers` (Bryant et al., 2012; Zhu
et al., 2018).

A third verb, `simulate`, runs the same axes backwards: it takes a model and a
network and returns a data-axis object, so a recovery check composes directly.

```python
from phynetpy.infer import simulate

sim = simulate(MSC(theta=0.02), taxa=6, n=200)      # draws its own species tree
recovered = infer(sim, criterion=PseudoLikelihood())
```

## Development

```bash
python -m pytest -m "not slow"   # test suite
python generate_docs.py          # regenerate docs/api/
python deploy.py --dry-run --no-bump # test + build 0.6.0, no upload
```

The version lives in `src/_version.py` and is the single source of truth;
`pyproject.toml` reads it and `phynetpy.__version__` re-exports it.

A tutorial and further instructions will be made available on our website,
<https://phylogenomics.rice.edu>.
