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
Axis 3 of the inference API: the criterion.

The criterion is the statistical objective -- what number is being
optimised, independent of the data type and the biology::

    from phynetpy.criteria import MDC, Likelihood, PseudoLikelihood, Bayesian

    MDC()                                     # parsimony, gene trees only
    Likelihood()                              # full likelihood
    PseudoLikelihood()                        # product over rooted triples
    Bayesian(objective=PseudoLikelihood())    # sample a pseudo-posterior

:class:`Bayesian` wraps an objective instead of sitting beside the
likelihoods, because MCMC is a *mode* on top of a likelihood rather than a
fourth thing to optimise.  A flat menu is still what the user sees; the
factoring underneath is just honest about the relationship.

Each criterion declares the data types it is defined on
(``accepts_data``), which lets an illegal run fail as a ``TypeError``
before any computation starts -- distinct from a legal-but-unimplemented
run, which fails as ``NotImplementedError``.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from typing import Union

from ._objectives import (
    MDC,
    Bayesian,
    Criterion,
    CriterionError,
    Likelihood,
    PseudoLikelihood,
)

__all__ = [
    "Criterion",
    "CriterionError",
    "MDC",
    "Likelihood",
    "PseudoLikelihood",
    "Bayesian",
    "resolve_criterion",
]


# String shortcuts, accepted anywhere a Criterion is expected as long as the
# objective needs no parameters.  Aliases cover both the PhyNetPy names and
# the PhyloNet command suffixes users arrive with.
_CRITERION_ALIASES = {
    "mdc": MDC,
    "mp": MDC,
    "parsimony": MDC,
    "likelihood": Likelihood,
    "ml": Likelihood,
    "full": Likelihood,
    "pseudolikelihood": PseudoLikelihood,
    "pseudo_likelihood": PseudoLikelihood,
    "pseudo": PseudoLikelihood,
    "mpl": PseudoLikelihood,
    "bayesian": Bayesian,
    "mcmc": Bayesian,
}


def resolve_criterion(spec: Union[Criterion, str, type, None]) -> Criterion:
    """Coerce a criterion shortcut into a :class:`Criterion` instance.

    Lets the casual user write ``criterion="MPL"`` and reserve the object
    form for when parameters are actually needed.

    Args:
        spec: A :class:`Criterion` instance (returned unchanged), a
            :class:`Criterion` subclass (instantiated with defaults), a
            case-insensitive shortcut string, or ``None`` for the default
            (:class:`Likelihood`).

    Returns:
        Criterion: The resolved criterion.

    Raises:
        CriterionError: If *spec* is not a recognised criterion.
    """
    if spec is None:
        return Likelihood()
    if isinstance(spec, Criterion):
        return spec
    if isinstance(spec, type) and issubclass(spec, Criterion):
        return spec()
    if isinstance(spec, str):
        key = spec.strip().lower().replace("-", "_").replace(" ", "_")
        if key in _CRITERION_ALIASES:
            return _CRITERION_ALIASES[key]()
        known = sorted(set(_CRITERION_ALIASES))
        raise CriterionError(
            f"unknown criterion {spec!r}; expected one of {known}, or a "
            "Criterion instance."
        )
    raise CriterionError(
        f"criterion must be a Criterion, a subclass, or a string; got "
        f"{type(spec).__name__}."
    )
