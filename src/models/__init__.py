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
Axis 2 of the inference API: the model.

The model is the *biology* -- the generative process that produced the
data, and therefore what an inferred network actually means::

    from phynetpy.models import MSC, Allopolyploid

    MSC()                                  # diploid + ILS (the default)
    Allopolyploid(subgenome_map={...})     # allopolyploidy

Keeping biology on its own axis is what makes the design extensible: a new
process (gene duplication and loss, say) slots in as a new
:class:`Model` subclass plus its registrations, and neither
:func:`phynetpy.infer.infer` nor :func:`phynetpy.infer.score` changes
signature.  It is also what makes ``simulate`` nearly free -- simulation
reuses this axis and runs it in the opposite direction.

.. note::
   ``phynetpy.models.Model`` is the *biological process*.  It is unrelated
   to :class:`phynetpy.ModelGraph.Model`, which is the probabilistic
   graphical model (network + parameters + scorer) that the numerical
   engines mutate during a search.  The bare name ``Model`` is deliberately
   not re-exported at the top level, so ``phynetpy.Model`` remains the
   graph class it has always been.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from typing import Union

from ._processes import MSC, Allopolyploid, Model, ModelSpecError
from .._units import (
    BranchLengthUnit,
    convert_network_branch_lengths,
)

__all__ = [
    "Model",
    "ModelSpecError",
    "MSC",
    "Allopolyploid",
    "BranchLengthUnit",
    "convert_network_branch_lengths",
    "resolve_model",
]


# String shortcuts, accepted anywhere a Model is expected as long as the
# process needs no parameters.
_MODEL_ALIASES = {
    "msc": MSC,
    "msnc": MSC,
    "coalescent": MSC,
    "diploid": MSC,
    "allopolyploid": Allopolyploid,
    "allopoly": Allopolyploid,
    "allop": Allopolyploid,
    "polyploid": Allopolyploid,
}


def resolve_model(spec: Union[Model, str, type, None]) -> Model:
    """Coerce a model shortcut into a :class:`Model` instance.

    Lets the casual user write ``model="MSC"`` and reserve the object form
    for when process parameters are actually needed.

    Args:
        spec: A :class:`Model` instance (returned unchanged), a
            :class:`Model` subclass (instantiated with defaults), a
            case-insensitive shortcut string, or ``None`` for the default
            (:class:`MSC`).

    Returns:
        Model: The resolved model.

    Raises:
        ModelSpecError: If *spec* is not a recognised model.
    """
    if spec is None:
        return MSC()
    if isinstance(spec, Model):
        return spec
    if isinstance(spec, type) and issubclass(spec, Model):
        return spec()
    if isinstance(spec, str):
        key = spec.strip().lower().replace("-", "_").replace(" ", "_")
        if key in _MODEL_ALIASES:
            return _MODEL_ALIASES[key]()
        known = sorted(set(_MODEL_ALIASES))
        raise ModelSpecError(
            f"unknown model {spec!r}; expected one of {known}, or a Model "
            "instance."
        )
    raise ModelSpecError(
        f"model must be a Model, a subclass, or a string; got "
        f"{type(spec).__name__}."
    )
