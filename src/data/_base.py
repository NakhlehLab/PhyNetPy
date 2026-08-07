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
Base class for the data axis.

Axis 1 of the three-axis inference API: *what data you have*.  The input
knows its own type, so the user never declares it as a flag -- the type of
the object passed to :func:`phynetpy.infer.infer` is half of the dispatch
key (see :mod:`phynetpy._registry`).

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class DataError(Exception):
    """Raised when a data container is malformed or cannot be loaded."""


class Data(ABC):
    """Base class for every input type on the data axis.

    Subclasses declare *what kind of observation* the user has:
    :class:`~phynetpy.data.GeneTrees` (gene-tree estimates),
    :class:`~phynetpy.data.Alignment` (per-locus sequences), or
    :class:`~phynetpy.data.BiallelicMarkers` (SNPs).  A criterion declares
    which of these it is defined on via ``Criterion.accepts_data``, and the
    registry keys engines on the concrete class.

    Every data type carries the species-to-allele mapping alongside the
    observations, because the mapping describes *the sample*, not the
    biology or the objective: which gene copies were sequenced from which
    species is a property of the data set.  This is why neither verb takes
    a separate ``mapping`` argument.
    """

    def __init__(self, mapping: Optional[Dict[str, List[str]]] = None) -> None:
        """Initialise the shared mapping slot.

        Args:
            mapping: Species -> list of allele / gene-copy labels.  ``None``
                defers to :meth:`resolved_mapping`, which falls back to an
                identity mapping (each label is its own species).
        """
        self._mapping: Optional[Dict[str, List[str]]] = mapping

    @property
    @abstractmethod
    def taxa(self) -> set:
        """The set of allele / sequence labels present in the data."""

    @property
    def mapping(self) -> Optional[Dict[str, List[str]]]:
        """Explicit species -> allele mapping, or ``None`` if unset."""
        return self._mapping

    @mapping.setter
    def mapping(self, value: Optional[Dict[str, List[str]]]) -> None:
        self._mapping = value

    def resolved_mapping(self) -> Dict[str, List[str]]:
        """Return an explicit mapping, falling back to identity.

        Returns:
            dict[str, list[str]]: Species -> allele labels.  When no mapping
            was supplied, every label is treated as its own species, which
            is the right default for single-copy (one allele per species)
            data.
        """
        if self._mapping is not None:
            return self._mapping
        return {name: [name] for name in sorted(self.taxa)}

    @property
    def has_branch_lengths(self) -> bool:
        """Whether the data carries usable branch lengths.

        Only meaningful for :class:`~phynetpy.data.GeneTrees`; sequence and
        marker data have no branch lengths of their own, so the base
        implementation reports ``False``.  Consulted by the registry when a
        criterion's ``use_branch_lengths`` policy is ``True``.
        """
        return False
