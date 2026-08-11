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
Sequence and marker data on the data axis.

Two data-axis types share the underlying :class:`phynetpy.MSA.MSA`
container but mean very different things to a criterion, so they are
distinct classes rather than one auto-detected one:

* :class:`Alignment` -- multilocus nucleotide sequences.  A Bayesian
  criterion integrates over the gene trees they imply.
* :class:`BiallelicMarkers` -- unlinked biallelic (SNP) sites, scored site
  by site under the Bryant et al. transition model.

Making the distinction explicit rather than sniffing the alphabet means a
mislabelled data file fails with a clear message instead of silently
routing to the wrong likelihood.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union

from ..MSA import MSA
from ._base import Data, DataError


class Alignment(Data):
    """Multilocus nucleotide sequence alignments.

    One :class:`Alignment` holds *all* loci for the analysis; each locus is
    a separate alignment over the same allele labels.  Accepted by the
    Bayesian criterion, which co-estimates the species network and the gene
    trees (Wen & Nakhleh 2018).

    PhyNetPy will not estimate gene trees from an alignment, so the MDC and
    pseudo-likelihood criteria reject this type outright: those are defined
    over gene-tree topologies.  Summarise the alignment externally and pass
    :class:`~phynetpy.data.GeneTrees` instead.

    The nucleotide substitution model lives here rather than on the model
    axis, because it describes how *these sequences* were generated, while
    the model axis describes how the gene copies are related.

    Attributes:
        loci: Per-locus alignments as ``{label: sequence}`` dicts.
        substitution_model: Nucleotide substitution model
            (:class:`~phynetpy.infer.JC69` by default).
    """

    def __init__(
        self,
        loci: Union[MSA, Sequence[Any]],
        mapping: Optional[Dict[str, List[str]]] = None,
        *,
        substitution_model: Optional[Any] = None,
    ) -> None:
        """Wrap per-locus alignments as a data-axis object.

        Args:
            loci: A single :class:`~phynetpy.MSA.MSA`, or a sequence of
                per-locus alignments (each an ``MSA`` or a
                ``{label: sequence}`` dict).
            mapping: Species -> list of allele labels, where the allele
                labels are the alignment row names.
            substitution_model: A
                :class:`~phynetpy.infer.SubstitutionModel` (``JC69``,
                ``HKY85``, ``GTR``).  ``None`` selects ``JC69``.

        Raises:
            DataError: If no loci are supplied.
        """
        super().__init__(mapping=mapping)

        if isinstance(loci, (MSA, dict)):
            raw: List[Any] = [loci]
        else:
            raw = list(loci)

        if not raw:
            raise DataError("Alignment requires at least one locus.")

        from .._mcmc_seq import _normalise_loci

        self.loci: List[Dict[str, str]] = _normalise_loci(raw)
        self.substitution_model = substitution_model

    @property
    def taxa(self) -> set:
        """The set of sequence labels across all loci."""
        names: set = set()
        for locus in self.loci:
            names.update(locus.keys())
        return names

    @property
    def n_loci(self) -> int:
        """Number of loci."""
        return len(self.loci)

    @property
    def n_sites(self) -> int:
        """Total aligned site count summed across loci."""
        total = 0
        for locus in self.loci:
            if locus:
                total += len(next(iter(locus.values())))
        return total

    @classmethod
    def from_file(
        cls,
        path: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        mapping: Optional[Dict[str, List[str]]] = None,
        *,
        format: Optional[str] = None,
        substitution_model: Optional[Any] = None,
    ) -> "Alignment":
        """Load one locus per file.

        Args:
            path: A single path, or a sequence of paths (one per locus).
            mapping: Species -> list of allele labels.
            format: ``"nexus"``, ``"fasta"``, or ``"vcf"``.  ``None`` infers
                from each file's extension.
            substitution_model: See :meth:`__init__`.

        Returns:
            Alignment: The loaded alignments, one locus per file.
        """
        paths = (
            [path]
            if isinstance(path, (str, os.PathLike))
            else list(path)
        )
        if not paths:
            raise DataError("no alignment files supplied.")

        return cls(
            [_read_alignment(p, format) for p in paths],
            mapping,
            substitution_model=substitution_model,
        )

    def __repr__(self) -> str:
        return (
            f"Alignment({self.n_loci} loci, {len(self.taxa)} labels, "
            f"{self.n_sites} sites)"
        )


class BiallelicMarkers(Data):
    """Unlinked biallelic markers (SNPs).

    Sites are encoded as per-taxon counts of the "red" allele, so a row of
    a diploid data set reads ``0``/``1``/``2``.  Accepted by the likelihood
    and Bayesian criteria (Bryant et al. 2012; Zhu et al. 2018).

    ``samples`` -- how many gene copies were sequenced per taxon -- belongs
    here rather than on the model axis: it describes the sampling effort,
    not the biology.  The mutation rates ``u``/``v`` and the coalescent
    rate live on :class:`~phynetpy.models.MSC` instead, since those *are*
    the generative process.

    Attributes:
        alignment: The underlying :class:`~phynetpy.MSA.MSA`.
        samples: Taxon label -> number of sampled gene copies.
    """

    def __init__(
        self,
        alignment: MSA,
        mapping: Optional[Dict[str, List[str]]] = None,
        *,
        samples: Optional[Dict[str, int]] = None,
    ) -> None:
        """Wrap a biallelic marker matrix as a data-axis object.

        Args:
            alignment: The marker matrix, as an :class:`~phynetpy.MSA.MSA`
                whose sequences hold per-site red-allele counts.
            mapping: Species -> list of taxon labels.  Usually the identity
                for marker data, where each row is its own taxon.
            samples: Taxon label -> sampled gene-copy count.  ``None``
                assumes one sampled copy per taxon.

        Raises:
            DataError: If *alignment* is not an ``MSA``.
        """
        super().__init__(mapping=mapping)

        if not isinstance(alignment, MSA):
            raise DataError(
                "BiallelicMarkers needs a phynetpy.MSA.MSA; got "
                f"{type(alignment).__name__}. Use "
                "BiallelicMarkers.from_file() to load from NEXUS."
            )

        self.alignment = alignment
        names = [record.get_name() for record in alignment.get_records()]
        self.samples: Dict[str, int] = (
            dict(samples) if samples is not None else {name: 1 for name in names}
        )

    @property
    def taxa(self) -> set:
        """The set of taxon labels in the marker matrix."""
        return {record.get_name() for record in self.alignment.get_records()}

    @property
    def n_sites(self) -> int:
        """Number of marker sites."""
        return int(self.alignment.dim()[1])

    @classmethod
    def from_file(
        cls,
        path: Union[str, os.PathLike],
        mapping: Optional[Dict[str, List[str]]] = None,
        *,
        samples: Optional[Dict[str, int]] = None,
        format: Optional[str] = None,
    ) -> "BiallelicMarkers":
        """Load markers from a NEXUS or VCF file.

        Args:
            path: Path to the file.
            mapping: Species -> list of taxon labels.
            samples: Taxon label -> sampled gene-copy count.
            format: ``"nexus"`` or ``"vcf"``.  ``None`` infers from the
                extension.

        Returns:
            BiallelicMarkers: The loaded marker matrix.
        """
        return cls(_read_alignment(path, format), mapping, samples=samples)

    def __repr__(self) -> str:
        return (
            f"BiallelicMarkers({len(self.taxa)} taxa, {self.n_sites} sites)"
        )


def _read_alignment(
    path: Union[str, os.PathLike], format: Optional[str] = None,
) -> MSA:
    """Read one alignment file into an :class:`~phynetpy.MSA.MSA`.

    Args:
        path: Path to the file.
        format: ``"nexus"``, ``"fasta"``, or ``"vcf"``.  ``None`` infers
            from the file extension, defaulting to NEXUS.

    Returns:
        MSA: The parsed alignment.

    Raises:
        DataError: If the format is unknown or the file cannot be parsed.
    """
    from .. import IO as io

    path_str = str(path)
    if format is None:
        suffix = os.path.splitext(path_str)[1].lower()
        if suffix in (".fa", ".fasta", ".fas", ".fna"):
            format = "fasta"
        elif suffix == ".vcf":
            format = "vcf"
        else:
            format = "nexus"

    fmt = format.lower()
    try:
        if fmt == "fasta":
            return io.read_fasta(path_str)
        if fmt == "vcf":
            return io.read_vcf(path_str)
        if fmt == "nexus":
            return io.read_nexus_msa(path_str)
    except Exception as exc:
        raise DataError(f"could not read alignment {path_str!r}: {exc}") from exc

    raise DataError(
        f"unknown alignment format {format!r}; expected 'nexus', 'fasta', or "
        "'vcf'."
    )
