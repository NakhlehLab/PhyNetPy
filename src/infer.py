#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##############################################################################

"""
Author : Mark Kessler
First Included in Version : 0.6.0

Public inference API for PhyNetPy: two verbs.

There are two things a user wants to do with a phylogenetic network method
-- find a network, or evaluate one -- so there are two verbs, ``infer`` and
``score``.  Everything that used to be encoded in a command name is now a
typed object on one of three orthogonal axes:

* **data** -- what you observed (:mod:`phynetpy.data`).  The input knows its
  own type, so you never declare it.
* **model** -- the generative biology (:mod:`phynetpy.models`).
* **criterion** -- the statistical objective (:mod:`phynetpy.criteria`),
  with Bayesian treated as a mode layered on a likelihood rather than as a
  fourth objective.

::

    from phynetpy.infer import infer, score
    from phynetpy.data import GeneTrees
    from phynetpy.models import MSC, Allopolyploid
    from phynetpy.criteria import MDC, Likelihood, PseudoLikelihood, Bayesian

    gts = GeneTrees.from_file("gene_trees.nex", mapping)

    result = infer(gts, model=MSC(), criterion=Likelihood())
    print(result.best, result.score)

    s = score(network, gts, model=MSC(), criterion=MDC())

Strings are accepted as shortcuts wherever an axis object needs no
parameters, so ``infer(gts, model="MSC", criterion="MPL")`` works too.

Whether a given run is legal depends on the whole triple, so dispatch goes
through a registry that doubles as an executable validity matrix
(:mod:`phynetpy._registry`).  An impossible request fails as a
``TypeError``, an unimplemented one as a ``NotImplementedError``, and the
two are never confused.  :func:`validity_matrix` returns the current table.

Beyond the verbs, this module re-exports what you need *around* a run: the
prior and substitution-model objects the axes take as parameters, the native
result types reachable as :attr:`InferenceResult.raw`, MCMC chain
diagnostics, and the scorers and proposal kernels a new
:func:`register`-ed :class:`Engine` can be built from.

.. note::
   The per-method entry points this replaced (``MPL``, ``MCMC_GT``,
   ``MCMC_SEQ``, ``InferNetwork_ML``, ``INFER_MP_ALLOP``, ``ALLOP_SCORE``,
   ``MCMC_BIMARKERS``, ``SNP_LIKELIHOOD``) are gone, not deprecated.  Each
   was one cell of the matrix above; see :func:`validity_matrix` for where
   it went.  The implementation classes still exist in their private
   modules for engine authors, but they are no longer the public API.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Union

from . import _engines as _engines_module  # noqa: F401  (populates the registry)
from ._registry import Engine, registered_cells, register, resolve, validity_matrix
from ._simulate import simulate
from .criteria import (
    Bayesian,
    Criterion,
    Likelihood,
    MDC,
    PseudoLikelihood,
    resolve_criterion,
)
from .data import Alignment, BiallelicMarkers, Data, GeneTrees
from .models import (
    Allopolyploid,
    BranchLengthUnit,
    MSC,
    Model,
    convert_network_branch_lengths,
    resolve_model,
)
from .Network import Network

# ── Configuration objects the axes take as parameters ─────────────────
# Priors for ``Bayesian(prior=...)`` and substitution models for
# ``Alignment(substitution_model=...)``.
from ._mcmc_gt import MCMC_GTPriors
from ._mcmc_seq import MCMCSeqPriors
from ._seq_likelihood import SubstitutionModel, JC69, HKY85, GTR

# ── Native result types, reachable as ``InferenceResult.raw`` ─────────
from ._infernetworkml import InferNetworkMLResult, optimize_network_parameters
from ._mcmc_gt import MCMCGTResult, MCMCSample
from ._mcmc_seq import (
    MCMCSeqResult,
    MCMCSeqSample,
    MultiChainResult,
    MultiChainStatus,
    run_parallel_chains,
)

# ── Extension points ─────────────────────────────────────────────────
# Scorers, proposal kernels, and the scoring primitives, so a new engine can
# be registered against the existing numerical machinery rather than
# reimplementing it.  These are what ``@register`` is for.
from ._mpl import (
    MPLScorer,
    MPLKernel,
    compute_gene_tree_triplets,
    score_species_network_triplets,
    format_mpl_reference_comparison,
    save_mpl_network_newick,
    GeneTreeTripletResult,
    SpeciesNetworkTripletResult,
)
from ._mcmc_gt import MCMCGTScorer, MCMCGTKernel, log_prior_network
from ._mcmc_seq import MCMCSeqKernel
from ._seq_likelihood import FelsensteinCalculator, gene_tree_msnc_log_density
from ._infer_mp_allop import (
    InferMPAllop,
    MPAllopScorer,
    Allop_MUL,
    AlleleMap,
    allop_parsimony_score,
)

# ── Low-level coalescent simulation ──────────────────────────────────
# The primitives behind the ``simulate`` verb, for callers who want a single
# gene tree or a bare ``{label: sequence}`` dict rather than a data-axis
# object.
from ._sim_seq import (
    SimulatedData,
    simulate_gene_tree,
    simulate_sequences,
    simulate_multilocus,
)

# ── Post-analysis: chain diagnostics + Tracer / NEXUS interop ─────────
from ._chain_analysis import (
    effective_sample_size,
    autocorrelation_time,
    standard_error_of_mean,
    hpd_interval,
    geweke,
    summarize,
    summarize_traces,
    ParameterSummary,
    ChainSummary,
    write_tracer_log,
    read_tracer_log,
    write_trees_nexus,
)

__all__ = [
    # The three verbs.
    "infer",
    "score",
    "simulate",
    # Results and the starting-phylogeny option.
    "InferenceResult",
    "Start",
    "StartMode",
    # The three axes, re-exported so one import is enough for a whole run.
    "Data",
    "GeneTrees",
    "Alignment",
    "BiallelicMarkers",
    "Model",
    "MSC",
    "Allopolyploid",
    "BranchLengthUnit",
    "convert_network_branch_lengths",
    "Criterion",
    "MDC",
    "Likelihood",
    "PseudoLikelihood",
    "Bayesian",
    # Axis configuration.
    "MCMC_GTPriors",
    "MCMCSeqPriors",
    "SubstitutionModel",
    "JC69",
    "HKY85",
    "GTR",
    # Native result types (InferenceResult.raw).
    "MCMCGTResult",
    "MCMCSample",
    "InferNetworkMLResult",
    "MCMCSeqResult",
    "MCMCSeqSample",
    "MultiChainResult",
    "MultiChainStatus",
    "run_parallel_chains",
    # Extension points: register your own engine.
    "Engine",
    "register",
    "resolve",
    "registered_cells",
    "validity_matrix",
    "optimize_network_parameters",
    "MPLScorer",
    "MPLKernel",
    "compute_gene_tree_triplets",
    "score_species_network_triplets",
    "format_mpl_reference_comparison",
    "save_mpl_network_newick",
    "GeneTreeTripletResult",
    "SpeciesNetworkTripletResult",
    "MCMCGTScorer",
    "MCMCGTKernel",
    "log_prior_network",
    "MCMCSeqKernel",
    "FelsensteinCalculator",
    "gene_tree_msnc_log_density",
    "InferMPAllop",
    "MPAllopScorer",
    "Allop_MUL",
    "AlleleMap",
    "allop_parsimony_score",
    # Low-level simulation primitives.
    "SimulatedData",
    "simulate_gene_tree",
    "simulate_sequences",
    "simulate_multilocus",
    # Post-analysis / Tracer interop.
    "effective_sample_size",
    "autocorrelation_time",
    "standard_error_of_mean",
    "hpd_interval",
    "geweke",
    "summarize",
    "summarize_traces",
    "ParameterSummary",
    "ChainSummary",
    "write_tracer_log",
    "read_tracer_log",
    "write_trees_nexus",
]


# ======================================================================
# Cross-cutting option: the starting phylogeny
# ======================================================================


class StartMode(Enum):
    """How a search may use a user-supplied starting phylogeny.

    Attributes:
        FREE: A free starting point.  The search may reshape it without
            restriction; only the initial position is influenced.
        AUGMENT: A backbone the result must contain.  The search may only
            add to it, never remove or rearrange what is already there.
            Well-defined because adding reticulations to a network yields a
            network containing the original as a subgraph, so this is a real
            constraint on the search space rather than an initialisation
            hint.
    """

    FREE = "free"
    AUGMENT = "augment"


@dataclass
class Start:
    """A starting phylogeny bundled with how the search may use it.

    The network and its usage mode travel together, because a bare network
    is ambiguous: "start here" and "the answer must contain this" are
    different requests that happen to take the same argument.

    Attributes:
        network: The starting network (which may be a tree).
        mode: Whether the search may reshape it (:attr:`StartMode.FREE`,
            the default) or only add to it (:attr:`StartMode.AUGMENT`).
    """

    network: Network
    mode: StartMode = StartMode.FREE

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = StartMode(self.mode.lower())
        if not isinstance(self.mode, StartMode):
            raise TypeError(
                f"Start.mode must be a StartMode; got {type(self.mode).__name__}."
            )

    @property
    def augment_only(self) -> bool:
        """Whether the final network must contain this one as a subgraph."""
        return self.mode is StartMode.AUGMENT


def _coerce_start(start: Union[Start, Network, None]) -> Optional[Start]:
    """Accept a bare network as shorthand for ``Start(net)``."""
    if start is None or isinstance(start, Start):
        return start
    if isinstance(start, Network):
        return Start(start)
    raise TypeError(
        f"start must be a Start or a Network; got {type(start).__name__}."
    )


# ======================================================================
# Result
# ======================================================================


@dataclass
class InferenceResult:
    """Uniform return type for :func:`infer`.

    The return type does not change with the criterion: a point estimate
    (ML, MDC) and a posterior sample (Bayesian) both arrive here, with
    ``posterior`` populated only in the latter case.  That is what lets a
    caller swap criteria without rewriting the code that consumes the
    result.

    Attributes:
        best: The best network found.  For a Bayesian run this is the
            maximum a posteriori network.
        score: The objective value of :attr:`best`.  A log likelihood, log
            pseudo-likelihood, or log posterior for the likelihood-based
            criteria; an extra-lineage count for :class:`~phynetpy.criteria.MDC`.
        method: The method that ran, named after the PhyloNet command it
            corresponds to (e.g. ``"InferNetwork_MPL"``).
        lower_is_better: Whether :attr:`score` is minimised.  ``True`` only
            for parsimony.
        posterior: Retained posterior samples, for Bayesian runs; ``None``
            otherwise.
        trace: The search path or MCMC chain, when the engine records one.
        raw: The engine's native result object, so nothing is lost by
            passing through this wrapper.  Attribute access falls through to
            it, which is why ``result.summary()`` and
            ``result.write_log(path)`` keep working.
    """

    best: Network
    score: float
    method: str = ""
    lower_is_better: bool = False
    posterior: Optional[List[Any]] = None
    trace: Optional[Any] = None
    raw: Any = field(default=None, repr=False)

    def __getattr__(self, item: str) -> Any:
        """Delegate unknown attributes to the engine's native result.

        Keeps the per-method richness reachable -- ``write_log``,
        ``summary``, ``information_criteria``, ``reticulation_posterior``,
        ``trace_table`` and friends -- without this wrapper having to
        enumerate every method of every engine.
        """
        # __getattr__ runs only on misses, but guard against recursion during
        # unpickling, when ``raw`` may not be set yet.
        if item.startswith("__") or item == "raw":
            raise AttributeError(item)

        raw = self.__dict__.get("raw")
        if raw is None:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {item!r}."
            )
        if hasattr(raw, item):
            return getattr(raw, item)
        raise AttributeError(
            f"{type(self).__name__} has no attribute {item!r} (and neither "
            f"does its .raw {type(raw).__name__})."
        )

    def __repr__(self) -> str:
        direction = "min" if self.lower_is_better else "max"
        n_post = len(self.posterior) if self.posterior is not None else 0
        return (
            f"InferenceResult(method={self.method!r}, score={self.score:.6g} "
            f"({direction}), posterior={n_post} samples)"
        )


# ======================================================================
# The two verbs
# ======================================================================


def infer(
    data: Data,
    model: Union[Model, str, None] = None,
    criterion: Union[Criterion, str, None] = None,
    start: Union[Start, Network, None] = None,
    **search: Any,
) -> InferenceResult:
    """Infer a phylogenetic network from data.

    Args:
        data: The observations: a :class:`~phynetpy.data.GeneTrees`,
            :class:`~phynetpy.data.Alignment`, or
            :class:`~phynetpy.data.BiallelicMarkers`.  Carries the
            species-to-allele mapping.
        model: The generative process.  Defaults to
            :class:`~phynetpy.models.MSC` (diploid + ILS).  A string
            shortcut such as ``"MSC"`` is accepted.
        criterion: The statistical objective.  Defaults to
            :class:`~phynetpy.criteria.Likelihood`.  A string shortcut such
            as ``"MPL"`` or ``"MDC"`` is accepted.
        start: Optional starting phylogeny.  A bare
            :class:`~phynetpy.Network.Network` is treated as
            ``Start(net)`` -- a free starting point.  Wrap it as
            ``Start(net, mode=StartMode.AUGMENT)`` to require that the
            result contain it.  ``None`` lets the engine build its own seed
            (usually a majority-rule consensus of the gene trees).
        **search: Generic search controls, forwarded to the engine:
            ``max_reticulations``, ``max_lvl``, ``num_iter``, ``preset``,
            ``seed``, ``n_workers``, ``progress``, and driver-specific
            extras.  Objective-specific settings belong on the criterion
            instead (``Bayesian(chain_length=...)``,
            ``PseudoLikelihood(subsets=...)``).

    Returns:
        InferenceResult: The best network, its score, and -- for Bayesian
        runs -- the retained posterior sample.

    Raises:
        TypeError: If the criterion is not defined on this data type.
        ValueError: If the criterion needs branch lengths the data lacks.
        NotImplementedError: If the combination is valid but unimplemented.

    Examples:
        Maximum pseudo-likelihood from gene trees::

            result = infer(gts, criterion=PseudoLikelihood())

        Bayesian sampling on top of a pseudo-likelihood::

            result = infer(gts, criterion=Bayesian(
                objective=PseudoLikelihood(), chain_length=200_000,
            ))
    """
    model_obj = resolve_model(model)
    criterion_obj = resolve_criterion(criterion)
    engine_cls = resolve(data, model_obj, criterion_obj)
    engine = engine_cls(
        model_obj, criterion_obj, start=_coerce_start(start), **search,
    )
    return engine.infer(data)


def score(
    network: Network,
    data: Data,
    model: Union[Model, str, None] = None,
    criterion: Union[Criterion, str, None] = None,
    optimize: bool = False,
    **search: Any,
) -> float:
    """Score a single network against data under some criterion.

    Args:
        network: The network to score.
        data: The observations.
        model: The generative process.  Defaults to
            :class:`~phynetpy.models.MSC`.
        criterion: The statistical objective.  Defaults to
            :class:`~phynetpy.criteria.Likelihood`.  Must be scorable:
            :class:`~phynetpy.criteria.Bayesian` is not, because a Bayesian
            score of one fixed network collapses to its wrapped objective.
        optimize: ``False`` (default) scores the network exactly as given,
            using its own branch lengths and inheritance probabilities.
            ``True`` optimises those continuous parameters and reports the
            best score attainable for this topology.
        **search: Controls for the ``optimize=True`` parameter search
            (``max_rounds``, ``improve_threshold``, ``scope``, ...).

    Returns:
        float: The objective value.  A log likelihood, log
        pseudo-likelihood, or log posterior density for the
        likelihood-based criteria (higher is better); an extra-lineage
        count for :class:`~phynetpy.criteria.MDC` (lower is better).

    Raises:
        TypeError: If the criterion is not defined on this data type, or is
            not scorable on its own.
        ValueError: If the criterion needs branch lengths the data lacks.
        NotImplementedError: If the combination is valid but unimplemented.

    Examples:
        The deep-coalescence count of a network, PhyloNet's separate
        parsimony-scoring command::

            score(net, gts, model=MSC(), criterion=MDC())

        The best likelihood this topology can attain::

            score(net, gts, criterion=Likelihood(), optimize=True)
    """
    model_obj = resolve_model(model)
    criterion_obj = resolve_criterion(criterion)

    if not criterion_obj.scorable:
        objective = getattr(criterion_obj, "objective", None)
        hint = (
            f"pass its .objective ({objective!r})"
            if objective is not None
            else "pass a likelihood (e.g. Likelihood())"
        )
        raise TypeError(
            f"{type(criterion_obj).__name__} scores nothing on its own: a "
            f"Bayesian score of one fixed network collapses to its "
            f"objective. {hint}."
        )

    engine_cls = resolve(data, model_obj, criterion_obj)
    engine = engine_cls(model_obj, criterion_obj, **search)
    return engine.score(network, data, optimize=optimize)
