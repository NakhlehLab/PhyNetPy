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
Axis 1 of the inference API: the data.

The input knows its own type.  Rather than declaring "I have gene trees"
as a string flag, you hand :func:`phynetpy.infer.infer` a typed object and
the type is half of the dispatch key::

    from phynetpy.data import GeneTrees, Alignment, BiallelicMarkers

    gts = GeneTrees.from_file("gene_trees.nex", mapping={"A": ["a1", "a2"]})
    aln = Alignment.from_file(["locus1.nex", "locus2.nex"], mapping)
    snp = BiallelicMarkers.from_file("markers.nex", samples={"A": 2})

Each data object also carries the species-to-allele mapping, because which
gene copies came from which species is a property of the sample rather than
of the biology or the objective.  That is why neither verb takes a separate
``mapping`` argument.

Three types, matching the three kinds of observation the multispecies
network coalescent literature is built on:

* :class:`GeneTrees` -- gene-tree estimates; the only type all four
  criteria accept.
* :class:`Alignment` -- multilocus nucleotide sequences.
* :class:`BiallelicMarkers` -- unlinked SNPs.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from ._base import Data, DataError
from ._genetrees import GeneTrees
from ._sequence import Alignment, BiallelicMarkers

__all__ = [
    "Data",
    "DataError",
    "GeneTrees",
    "Alignment",
    "BiallelicMarkers",
]
