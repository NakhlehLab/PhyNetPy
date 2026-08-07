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
Dispatch for the two-verb inference API: an executable validity matrix.

Single dispatch on the data type is not enough, because whether a run is
legal depends on the whole triple (data x model x criterion).  MDC is
defined on gene trees but not on alignments; pseudo-likelihood exists for
gene trees and biallelic markers but not for alignments; allopolyploidy has
a parsimony method but no likelihood one.  A registry keyed on all three
therefore *is* the validity matrix, in code, and the table in the design
memo can be regenerated from it (see :func:`validity_matrix`).

Resolution separates two failures that users constantly conflate:

* **The math does not exist.** ``TypeError`` -- e.g. MDC over an alignment.
  Parsimony is defined over gene-tree topologies; asking for it on
  sequences is not a missing feature, it is a category error.
* **Nobody has built it yet.** ``NotImplementedError`` -- e.g. the
  likelihood of an alignment, which is well-defined (integrate over gene
  trees) but unimplemented.

A third check sits between them: a criterion's branch-length policy has to
be satisfiable by the data, which is a ``ValueError`` because it is a
mismatch between two things the user supplied rather than a fact about the
method.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .criteria import Criterion
from .data import Data
from .models import Model

__all__ = [
    "Engine",
    "register",
    "resolve",
    "registered_cells",
    "validity_matrix",
]


#: (data class, model class, criterion class) -> Engine subclass.
_ENGINES: Dict[Tuple[type, type, type], Type["Engine"]] = {}


class Engine(ABC):
    """Base class for a concrete (data, model, criterion) implementation.

    An engine is the adapter between the declarative axis objects and the
    numerical code.  Each registered engine implements *both* verbs, which
    is why scoring is not a separate command: PhyloNet's ``CalGTProb`` and
    its deep-coalescence count are simply ``score(net, data, MSC(),
    Likelihood())`` and ``score(net, data, MSC(), MDC())``.

    Attributes:
        model: The generative process for this run.
        criterion: The statistical objective for this run.
        start: Optional :class:`~phynetpy.infer.Start` describing the
            starting phylogeny and how the search may use it.
        search: Generic search controls, forwarded from the verb's
            ``**search`` kwargs.
    """

    #: Name of the PhyloNet command (or method) this engine corresponds to,
    #: used in error messages and :attr:`InferenceResult.method`.
    method: str = "Engine"

    def __init__(
        self,
        model: Model,
        criterion: Criterion,
        *,
        start: Optional[Any] = None,
        **search: Any,
    ) -> None:
        """Bind the axis objects and the search controls for one run.

        Args:
            model: The generative process.
            criterion: The statistical objective.
            start: Optional starting phylogeny and usage mode.
            **search: Generic search controls (``max_reticulations``,
                ``preset``, ``seed``, ...).
        """
        self.model = model
        self.criterion = criterion
        self.start = start
        self.search = search

    @abstractmethod
    def infer(self, data: Data) -> Any:
        """Infer a network from *data*.

        Args:
            data: The observations.

        Returns:
            InferenceResult: The best network, its score, and any posterior
            sample or search trace.
        """

    @abstractmethod
    def score(self, network, data: Data, *, optimize: bool = False) -> float:
        """Score a single fixed network.

        Args:
            network: The network to score.
            data: The observations.
            optimize: When ``True``, optimise the network's continuous
                parameters (branch lengths and inheritance probabilities)
                and report the best attainable score for that topology.
                When ``False``, score the network exactly as given.

        Returns:
            float: The objective value.
        """


def register(
    data_cls: type, model_cls: type, criterion_cls: type,
) -> Callable[[Type[Engine]], Type[Engine]]:
    """Register an engine for one cell of the validity matrix.

    Every implemented method is exactly one registration site, which makes
    the mapping from PhyloNet commands to PhyNetPy code one-to-one and
    self-documenting.

    Args:
        data_cls: The :class:`~phynetpy.data.Data` subclass.
        model_cls: The :class:`~phynetpy.models.Model` subclass.
        criterion_cls: The :class:`~phynetpy.criteria.Criterion` subclass.

    Returns:
        A decorator that registers the engine class and returns it
        unchanged.

    Raises:
        ValueError: If the cell is already registered.
    """
    key = (data_cls, model_cls, criterion_cls)

    def deco(engine_cls: Type[Engine]) -> Type[Engine]:
        if key in _ENGINES:
            raise ValueError(
                f"({data_cls.__name__}, {model_cls.__name__}, "
                f"{criterion_cls.__name__}) is already registered to "
                f"{_ENGINES[key].__name__}."
            )
        _ENGINES[key] = engine_cls
        return engine_cls

    return deco


def _lookup(
    data_cls: type, model_cls: type, criterion_cls: type,
) -> Optional[Type[Engine]]:
    """Find the most specific engine registered for a triple.

    Exact matches win.  Failing that, the most specific registration whose
    three classes are all ancestors of the requested ones is used, so
    subclassing an axis type inherits its engines.
    """
    key = (data_cls, model_cls, criterion_cls)
    if key in _ENGINES:
        return _ENGINES[key]

    best: Optional[Type[Engine]] = None
    best_distance = None
    for (d, m, c), engine in _ENGINES.items():
        if not (issubclass(data_cls, d) and issubclass(model_cls, m)
                and issubclass(criterion_cls, c)):
            continue
        # Rank candidates by total MRO distance: the nearest ancestors are
        # the most specific applicable registration.
        distance = (
            data_cls.__mro__.index(d)
            + model_cls.__mro__.index(m)
            + criterion_cls.__mro__.index(c)
        )
        if best_distance is None or distance < best_distance:
            best, best_distance = engine, distance
    return best


def resolve(
    data: Data, model: Model, criterion: Criterion,
) -> Type[Engine]:
    """Resolve a triple to its engine, or explain why it cannot be run.

    Runs the three checks in order, so the most fundamental objection is
    always the one reported.

    Args:
        data: The observations.
        model: The generative process.
        criterion: The statistical objective.

    Returns:
        The :class:`Engine` subclass implementing this cell.

    Raises:
        TypeError: If *criterion* is not defined on this data type, or if an
            argument is not an axis object at all.
        ValueError: If the criterion was asked to use branch lengths the
            data does not carry.
        NotImplementedError: If the combination is legal in principle but
            has no engine yet.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data must be a phynetpy.data.Data (GeneTrees, Alignment, "
            f"BiallelicMarkers); got {type(data).__name__}."
        )
    if not isinstance(model, Model):
        raise TypeError(
            f"model must be a phynetpy.models.Model (MSC, Allopolyploid); "
            f"got {type(model).__name__}."
        )
    if not isinstance(criterion, Criterion):
        raise TypeError(
            f"criterion must be a phynetpy.criteria.Criterion; got "
            f"{type(criterion).__name__}."
        )

    # 1. Hard semantic constraint -- the math is undefined, not missing.  An
    # empty accepts_data means "any data type"; otherwise subclasses of an
    # accepted type are accepted too, so a user-defined
    # WeightedGeneTrees(GeneTrees) works without re-declaring anything.
    accepted_types = tuple(criterion.accepts_data)
    if accepted_types and not isinstance(data, accepted_types):
        accepted = [c.__name__ for c in accepted_types]
        raise TypeError(
            f"{type(criterion).__name__} is not defined on "
            f"{type(data).__name__}; it requires one of {accepted}."
        )

    # 2. The branch-length policy must be satisfiable by the data.
    if criterion.use_branch_lengths and not data.has_branch_lengths:
        raise ValueError(
            f"{type(criterion).__name__} was asked to use branch lengths, "
            f"but the supplied {type(data).__name__} carries none. Pass "
            "use_branch_lengths=False (or None to use them only when "
            "present)."
        )

    # 3. Implemented? -- a missing feature, not a type error.
    engine = _lookup(type(data), type(model), type(criterion))
    if engine is None:
        raise NotImplementedError(
            f"no engine for ({type(data).__name__}, {type(model).__name__}, "
            f"{type(criterion).__name__}) yet. This combination is valid in "
            "principle; it just has not been built."
        )

    model.validate(data)
    return engine


def registered_cells() -> List[Tuple[str, str, str, str]]:
    """List every implemented cell of the validity matrix.

    Returns:
        list[tuple[str, str, str, str]]: ``(data, model, criterion,
        method)`` tuples, sorted, where ``method`` is the PhyloNet command
        the engine corresponds to.
    """
    return sorted(
        (d.__name__, m.__name__, c.__name__, e.method)
        for (d, m, c), e in _ENGINES.items()
    )


def validity_matrix(model_cls: Optional[type] = None) -> Dict[str, Dict[str, str]]:
    """Render the validity matrix for one model as nested dicts.

    Regenerates the design memo's table from the code, so the
    documentation cannot drift from what is actually registered.  Each cell
    is one of:

    * the engine's method name -- implemented;
    * ``"-"`` -- valid in principle, no engine yet
      (``NotImplementedError``);
    * ``"x"`` -- illegal in principle (``TypeError``).

    Args:
        model_cls: The :class:`~phynetpy.models.Model` subclass to tabulate.
            ``None`` uses :class:`~phynetpy.models.MSC`.

    Returns:
        dict[str, dict[str, str]]: ``matrix[data_name][criterion_name]``.
    """
    from .criteria import MDC, Bayesian, Likelihood, PseudoLikelihood
    from .data import Alignment, BiallelicMarkers, GeneTrees
    from .models import MSC

    model_cls = model_cls if model_cls is not None else MSC
    data_classes = (GeneTrees, Alignment, BiallelicMarkers)
    criterion_classes = (MDC, Likelihood, PseudoLikelihood, Bayesian)

    matrix: Dict[str, Dict[str, str]] = {}
    for data_cls in data_classes:
        row: Dict[str, str] = {}
        for criterion_cls in criterion_classes:
            # Bayesian delegates accepts_data to its wrapped objective, so
            # instantiate to read the effective constraint.
            accepted = criterion_cls().accepts_data
            if accepted and not issubclass(data_cls, tuple(accepted)):
                row[criterion_cls.__name__] = "x"
                continue
            engine = _lookup(data_cls, model_cls, criterion_cls)
            row[criterion_cls.__name__] = engine.method if engine else "-"
        matrix[data_cls.__name__] = row
    return matrix
