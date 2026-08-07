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
The model axis: the base class and the generative processes PhyNetPy models.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from abc import ABC
from typing import Dict, List, Optional


class ModelSpecError(Exception):
    """Raised when a model is misconfigured for the data it is given."""


class Model(ABC):
    """Base class for a generative biological process.

    A model says what produced the data and therefore how to read the
    inferred network: under :class:`~phynetpy.models.MSC` a reticulation is
    a hybridization or introgression event between diploid lineages, while
    under :class:`~phynetpy.models.Allopolyploid` it is a polyploid
    formation event and the network is interpreted through its
    multiple-labelled (MUL) tree.

    Models are declarative parameter carriers, not engines.  They hold the
    process parameters (population mutation rate, mutation rates, subgenome
    assignment) and are handed to the engine the registry selects; the
    numerical work lives in the engines.

    .. note::
       Unrelated to :class:`phynetpy.ModelGraph.Model`, which is the
       probabilistic graphical model a search mutates.  See the
       :mod:`phynetpy.models` module docstring.
    """

    def validate(self, data) -> None:
        """Check this model against the data it will be run on.

        Called by the registry after dispatch succeeds, so a model may
        assume the (data, model, criterion) triple is legal and only needs
        to police its own parameters.  The default accepts everything.

        Args:
            data: The :class:`~phynetpy.data.Data` the run will consume.

        Raises:
            ModelSpecError: If the model cannot be applied to *data*.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MSC(Model):
    """The multispecies network coalescent: diploid lineages with ILS.

    The default model, and the one nearly every method in the field is
    built on.  Gene copies coalesce within the branches of the species
    network at a rate set by the population mutation rate; at a
    reticulation, each lineage passes into one parent with probability
    given by that edge's inheritance probability (gamma).  Incomplete
    lineage sorting therefore arises from the coalescent, and a
    reticulation means hybridization or introgression.

    The process parameters live here because they *are* the biology.  The
    mutation rates ``u`` and ``v`` describe the biallelic transition model
    and are read only when scoring
    :class:`~phynetpy.data.BiallelicMarkers`; ``theta`` is the population
    mutation rate ``4*N*mu``.

    Attributes:
        theta: Population mutation rate.  ``None`` lets the engine pick its
            own default (the prior mean, when sampling it).
        u: Red-to-green mutation rate for biallelic markers.
        v: Green-to-red mutation rate for biallelic markers.
        coal: Coalescent rate constant used by the marker likelihood.
        pop_sizes: Optional per-branch population sizes; ``None`` uses a
            single rate across the network.
    """

    def __init__(
        self,
        theta: Optional[float] = None,
        *,
        u: float = 1.0,
        v: float = 1.0,
        coal: float = 1.0,
        pop_sizes: Optional[Dict] = None,
    ) -> None:
        """Configure the multispecies network coalescent.

        Args:
            theta: Population mutation rate ``4*N*mu``.  ``None`` defers to
                the engine (which samples it, for the Bayesian criterion).
            u: Red-to-green mutation rate (biallelic markers only).
            v: Green-to-red mutation rate (biallelic markers only).
            coal: Coalescent rate constant (biallelic markers only).
            pop_sizes: Per-branch population sizes, keyed by node.

        Raises:
            ModelSpecError: If any rate is non-positive.
        """
        if theta is not None and theta <= 0.0:
            raise ModelSpecError(f"theta must be positive; got {theta}.")
        for label, value in (("u", u), ("v", v), ("coal", coal)):
            if value <= 0.0:
                raise ModelSpecError(f"{label} must be positive; got {value}.")

        self.theta = theta
        self.u = u
        self.v = v
        self.coal = coal
        self.pop_sizes = pop_sizes

    def __repr__(self) -> str:
        theta = "auto" if self.theta is None else f"{self.theta:g}"
        return f"MSC(theta={theta})"


class Allopolyploid(Model):
    """Allopolyploidy: networks read through their MUL tree.

    Models polyploid taxa whose subgenomes descend from distinct diploid
    progenitors (Hejase et al.; MP-Allop / Polyphest).  A reticulation is a
    polyploid formation event, and the network is scored by expanding it
    into a multiple-labelled (MUL) tree in which each subgenome of a
    polyploid appears as its own tip.

    Because subgenome assignment -- which gene copies belong to which
    subgenome of which taxon -- is a statement about the biology of the
    taxa, it lives on the model rather than on the data.  When omitted, it
    is derived from the data's own mapping.

    Attributes:
        subgenome_map: Subgenome / species -> list of gene-copy labels.
            ``None`` derives it from the data.
    """

    def __init__(
        self, subgenome_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Configure the allopolyploid process.

        Args:
            subgenome_map: Subgenome -> list of gene-copy labels.  ``None``
                derives the assignment from the data's mapping (or, failing
                that, from :meth:`phynetpy.GeneTrees.GeneTrees.mp_allop_map`).
        """
        self.subgenome_map = subgenome_map

    def resolve_subgenome_map(self, data) -> Dict[str, List[str]]:
        """Return the subgenome assignment to use for *data*.

        Args:
            data: The :class:`~phynetpy.data.Data` being analysed.

        Returns:
            dict[str, list[str]]: Subgenome -> gene-copy labels.

        Raises:
            ModelSpecError: If no assignment is available.
        """
        if self.subgenome_map is not None:
            return self.subgenome_map

        resolved = data.resolved_mapping()
        if not resolved:
            raise ModelSpecError(
                "Allopolyploid needs a subgenome assignment: pass "
                "Allopolyploid(subgenome_map=...) or attach a mapping to the "
                "data."
            )
        return resolved

    def validate(self, data) -> None:
        """Check that a subgenome assignment can be resolved.

        Args:
            data: The data the run will consume.

        Raises:
            ModelSpecError: If no assignment is available.
        """
        self.resolve_subgenome_map(data)

    def __repr__(self) -> str:
        if self.subgenome_map is None:
            return "Allopolyploid(subgenome_map=auto)"
        return f"Allopolyploid({len(self.subgenome_map)} subgenomes)"
