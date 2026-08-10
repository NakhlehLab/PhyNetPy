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
Small, explicit data structures for population-genetic data and graphs.

The module covers two representations that fit PhyNetPy's current scope:

* :class:`PopulationData` stores biallelic alternate-allele and callable-allele
  counts by population and site.
* :class:`AdmixtureGraph` gives an existing :class:`~.Network.Network` the
  topology and inheritance semantics of a pulse-admixture history.

This boundary is deliberate. Continuous-migration graphs and ancestral
recombination graphs have different time and edge semantics, so they are not
forced into ``Network`` here. In particular, recombination can give different
sites different local genealogies (Liu, Ogilvie, and Nakhleh, 2021), whereas
an admixture graph is one directed acyclic population history.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .Network import Network, NetworkError, Node


__all__ = [
    "PopulationGeneticsError",
    "PopulationData",
    "AdmixtureGraph",
]


class PopulationGeneticsError(ValueError):
    """Raised when population-genetic data or graph semantics are invalid."""


class PopulationData:
    """
    Biallelic allele counts arranged as populations by sites.

    The numerator and denominator are stored separately. This avoids treating
    missing genotypes as reference alleles and supports different sample sizes
    among populations and sites. A callable count of zero denotes a missing
    population/site observation.

    This is an analysis container, not a ``phynetpy.data.Data`` inference-axis
    type; :func:`phynetpy.infer.infer` and :func:`phynetpy.infer.score` do not
    consume it.
    """

    def __init__(
        self,
        alternate_allele_counts: Any,
        called_allele_counts: Any,
        populations: Sequence[str],
        *,
        sites: Sequence[str] | None = None,
    ) -> None:
        """
        Create a population-by-site allele-count table.

        Args:
            alternate_allele_counts: Two-dimensional nonnegative integer counts.
            called_allele_counts: Matching counts of callable gene copies.
            populations: Unique population labels, one per matrix row.
            sites: Optional unique site labels, one per matrix column.

        Raises:
            PopulationGeneticsError: If dimensions, labels, or counts are
                inconsistent.
        """
        alternate = _count_matrix(
            alternate_allele_counts, "alternate_allele_counts"
        )
        called = _count_matrix(called_allele_counts, "called_allele_counts")

        if alternate.shape != called.shape:
            raise PopulationGeneticsError(
                "alternate and called allele-count matrices must have the "
                f"same shape; got {alternate.shape} and {called.shape}."
            )
        if alternate.shape[0] == 0 or alternate.shape[1] == 0:
            raise PopulationGeneticsError(
                "population data requires at least one population and one site."
            )
        if np.any(alternate > called):
            raise PopulationGeneticsError(
                "alternate-allele counts cannot exceed called-allele counts."
            )

        labels = _labels(populations, alternate.shape[0], "population")
        site_labels = (
            None if sites is None else _labels(sites, alternate.shape[1], "site")
        )

        alternate.setflags(write=False)
        called.setflags(write=False)
        self._alternate_allele_counts = alternate
        self._called_allele_counts = called
        self._populations = labels
        self._sites = site_labels
        self._population_indices = {
            population: index for index, population in enumerate(labels)
        }

    @property
    def populations(self) -> tuple[str, ...]:
        """Population labels in matrix-row order."""
        return self._populations

    @property
    def sites(self) -> tuple[str, ...] | None:
        """Site labels in matrix-column order, if supplied."""
        return self._sites

    @property
    def n_populations(self) -> int:
        """Number of populations."""
        return len(self._populations)

    @property
    def n_sites(self) -> int:
        """Number of sites."""
        return self._alternate_allele_counts.shape[1]

    @property
    def alternate_allele_counts(self) -> np.ndarray:
        """Read-only population-by-site alternate-allele counts."""
        return self._alternate_allele_counts

    @property
    def called_allele_counts(self) -> np.ndarray:
        """Read-only population-by-site callable-allele counts."""
        return self._called_allele_counts

    def allele_frequencies(self, population: str | None = None) -> np.ndarray:
        """
        Calculate alternate-allele frequencies without imputing missing data.

        Args:
            population: Optional population label. If omitted, return the full
                population-by-site matrix.

        Returns:
            A floating-point array. Entries with no called alleles are ``NaN``.

        Raises:
            PopulationGeneticsError: If *population* is unknown.
        """
        frequencies = np.full(
            self._alternate_allele_counts.shape, np.nan, dtype=float
        )
        np.divide(
            self._alternate_allele_counts,
            self._called_allele_counts,
            out=frequencies,
            where=self._called_allele_counts > 0,
        )
        if population is None:
            return frequencies
        try:
            return frequencies[self._population_indices[population]]
        except KeyError as exc:
            raise PopulationGeneticsError(
                f"unknown population {population!r}."
            ) from exc

    @classmethod
    def from_genotypes(
        cls,
        genotypes: Any,
        samples: Sequence[str],
        population_members: Mapping[str, Sequence[str]],
        *,
        ploidy: int | Sequence[int] = 2,
        sites: Sequence[str] | None = None,
    ) -> PopulationData:
        """
        Aggregate a sample-by-site matrix into population allele counts.

        Genotypes are alternate-allele copy counts. ``None``, ``NaN``, ``"."``,
        ``"?"``, ``"-"``, ``"./."``, and ``".|."`` are treated as missing.
        Every sample must belong to exactly one population.

        Args:
            genotypes: Two-dimensional sample-by-site genotype-count matrix.
            samples: Unique sample labels in matrix-row order.
            population_members: Population label -> member sample labels.
            ploidy: Positive ploidy shared by all samples, or one value per
                sample.
            sites: Optional site labels.

        Returns:
            Aggregated :class:`PopulationData`.

        Raises:
            PopulationGeneticsError: If genotypes, ploidy, or membership are
                malformed.
        """
        if isinstance(samples, str):
            raise PopulationGeneticsError(
                "samples must be a sequence of labels, not one string."
            )
        sample_labels = tuple(samples)
        if not sample_labels:
            raise PopulationGeneticsError("at least one sample is required.")
        if any(not isinstance(label, str) or not label for label in sample_labels):
            raise PopulationGeneticsError(
                "sample labels must be non-empty strings."
            )
        if len(set(sample_labels)) != len(sample_labels):
            raise PopulationGeneticsError("sample labels must be unique.")

        matrix = _genotype_matrix(genotypes)
        if matrix.shape[0] != len(sample_labels):
            raise PopulationGeneticsError(
                "the genotype matrix must have one row per sample; got "
                f"{matrix.shape[0]} rows for {len(sample_labels)} samples."
            )
        if matrix.shape[1] == 0:
            raise PopulationGeneticsError("at least one genotype site is required.")

        ploidies = _ploidies(ploidy, len(sample_labels))
        for row, sample_ploidy in zip(matrix, ploidies):
            observed = row[~np.isnan(row)]
            if np.any(observed > sample_ploidy):
                raise PopulationGeneticsError(
                    "a genotype alternate-allele count cannot exceed its "
                    f"sample ploidy ({sample_ploidy})."
                )

        member_indices = _population_member_indices(
            sample_labels, population_members
        )
        population_labels = tuple(member_indices)
        alternate = np.zeros(
            (len(population_labels), matrix.shape[1]), dtype=np.int64
        )
        called = np.zeros_like(alternate)

        for population_index, population in enumerate(population_labels):
            for sample_index in member_indices[population]:
                observed = ~np.isnan(matrix[sample_index])
                alternate[population_index, observed] += matrix[
                    sample_index, observed
                ].astype(np.int64)
                called[population_index, observed] += ploidies[sample_index]

        return cls(alternate, called, population_labels, sites=sites)

    @classmethod
    def from_biallelic_markers(
        cls,
        markers: Any,
        *,
        called_alleles: int | Mapping[str, int] | None = None,
    ) -> PopulationData:
        """
        Aggregate PhyNetPy biallelic-marker rows by their population mapping.

        The marker object's species-to-allele mapping supplies population
        membership. Callable gene-copy counts must be supplied explicitly:
        ``BiallelicMarkers`` defaults its ``samples`` values to one even when a
        VCF row contains diploid genotype dosages, so that default is not a
        reliable denominator.

        Args:
            markers: A :class:`phynetpy.data.BiallelicMarkers` object.
            called_alleles: A positive count shared by all marker rows, or a
                row-label -> count mapping. Pass ``markers.samples`` explicitly
                when those values are known to have been supplied correctly.

        Returns:
            Aggregated :class:`PopulationData`.

        Raises:
            PopulationGeneticsError: If *markers* is the wrong type or
                callable-copy counts are absent or incomplete.
        """
        from .data import BiallelicMarkers

        if not isinstance(markers, BiallelicMarkers):
            raise PopulationGeneticsError(
                "from_biallelic_markers requires a BiallelicMarkers object."
            )

        records = markers.alignment.get_records()
        samples = [record.get_name() for record in records]
        if called_alleles is None:
            raise PopulationGeneticsError(
                "called_alleles is required because BiallelicMarkers does not "
                "record whether its sample counts were explicit or defaulted."
            )
        if isinstance(called_alleles, (int, np.integer)):
            ploidies: int | list[int] = int(called_alleles)
        elif isinstance(called_alleles, Mapping):
            try:
                ploidies = [called_alleles[sample] for sample in samples]
            except KeyError as exc:
                raise PopulationGeneticsError(
                    "missing callable allele count for marker row "
                    f"{exc.args[0]!r}."
                ) from exc
        else:
            raise PopulationGeneticsError(
                "called_alleles must be a positive integer or a mapping from "
                "marker-row labels to positive integers."
            )

        return cls.from_genotypes(
            [record.get_seq() for record in records],
            samples,
            markers.resolved_mapping(),
            ploidy=ploidies,
        )

    def __repr__(self) -> str:
        return (
            f"PopulationData({self.n_populations} populations, "
            f"{self.n_sites} sites)"
        )


class AdmixtureGraph:
    """
    Validated pulse-admixture topology and proportions over a ``Network``.

    ``Network``, ``Node``, and ``Edge`` remain the graph implementation. This
    wrapper adds only population-genetic invariants:

    * one rooted directed acyclic component;
    * nodes with multiple parents are explicitly marked as reticulations;
    * their incoming inheritance proportions are strictly between zero and
      one and sum to one.

    The wrapper intentionally does not interpret ``Edge.length``. PhyNetPy
    networks currently carry sequence or coalescent units, while classical
    allele-frequency admixture graphs use genetic-drift lengths. Conflating
    those quantities would be scientifically incorrect.

    Divergence nodes may be unary or multifurcating, and admixture may involve
    more than two sources. Thus the wrapper does not impose the binary
    phylogenetic-network assumptions used by some PhyNetPy algorithms.
    """

    def __init__(self, network: Network, *, tolerance: float = 1e-8) -> None:
        """
        Attach admixture-graph semantics to an existing network.

        Args:
            network: The directed graph to validate and wrap.
            tolerance: Absolute tolerance for inheritance proportions summing
                to one and for recognizing an unused zero gamma.

        Raises:
            PopulationGeneticsError: If the graph is not a valid admixture
                history.
        """
        if not isinstance(network, Network):
            raise PopulationGeneticsError(
                f"AdmixtureGraph requires a Network; got {type(network).__name__}."
            )
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise PopulationGeneticsError(
                "tolerance must be finite and nonnegative."
            )
        self._network = network
        self._tolerance = tolerance
        self.validate()

    @property
    def network(self) -> Network:
        """The wrapped graph. Call :meth:`validate` after mutating it."""
        return self._network

    @property
    def root(self) -> Node:
        """The unique ancestral root."""
        return next(
            node for node in self._network.V()
            if self._network.in_degree(node) == 0
        )

    @property
    def terminal_populations(self) -> tuple[Node, ...]:
        """Terminal population nodes in deterministic label order."""
        return tuple(sorted(
            (
                node for node in self._network.V()
                if self._network.out_degree(node) == 0
            ),
            key=lambda node: node.label,
        ))

    @property
    def admixture_nodes(self) -> tuple[Node, ...]:
        """Nodes with ancestry from two or more parent populations."""
        return tuple(sorted(
            (
                node for node in self._network.V()
                if self._network.in_degree(node) > 1
            ),
            key=lambda node: node.label,
        ))

    def admixture_proportions(
        self, node: Node | str
    ) -> dict[str, float]:
        """
        Map each parent population to its contribution at an admixture node.

        Args:
            node: An admixture node or its label.

        Returns:
            Parent label -> inheritance proportion, sorted by parent label.

        Raises:
            PopulationGeneticsError: If the node is absent or is not an
                admixture node.
        """
        resolved = self._resolve_node(node)
        if self._network.in_degree(resolved) <= 1:
            raise PopulationGeneticsError(
                f"node {resolved.label!r} is not an admixture node."
            )
        return {
            edge.src.label: edge.get_gamma()
            for edge in sorted(
                self._network.in_edges(resolved),
                key=lambda edge: edge.src.label,
            )
        }

    def validate(self) -> None:
        """
        Validate topology, reticulation flags, and inheritance proportions.

        Raises:
            PopulationGeneticsError: At the first violated invariant.
        """
        nodes = self._network.V()
        if not nodes:
            raise PopulationGeneticsError(
                "an admixture graph requires at least one population node."
            )

        try:
            self._network.topological_order()
        except NetworkError as exc:
            raise PopulationGeneticsError(
                "an admixture graph must be acyclic."
            ) from exc

        roots = [
            node for node in nodes if self._network.in_degree(node) == 0
        ]
        if len(roots) != 1:
            raise PopulationGeneticsError(
                "an admixture graph must have exactly one root; "
                f"found {len(roots)}."
            )

        endpoint_pairs: set[tuple[Node, Node]] = set()
        for edge in self._network.E():
            endpoints = (edge.src, edge.dest)
            if endpoints in endpoint_pairs:
                raise PopulationGeneticsError(
                    "parallel ancestry edges are ambiguous in an admixture "
                    f"graph ({edge.src.label!r} -> {edge.dest.label!r})."
                )
            endpoint_pairs.add(endpoints)

        for node in nodes:
            incoming = self._network.in_edges(node)
            is_admixture = len(incoming) > 1
            if node.is_reticulation() != is_admixture:
                expected = "marked" if is_admixture else "unmarked"
                raise PopulationGeneticsError(
                    f"node {node.label!r} must be {expected} as a "
                    "reticulation to agree with its number of parents."
                )

            if not is_admixture:
                for edge in incoming:
                    if not math.isclose(
                        edge.get_gamma(),
                        0.0,
                        rel_tol=0.0,
                        abs_tol=self._tolerance,
                    ):
                        raise PopulationGeneticsError(
                            "inheritance proportions are only defined on edges "
                            f"entering admixture nodes; found gamma "
                            f"{edge.get_gamma()} on {edge.src.label!r} -> "
                            f"{edge.dest.label!r}."
                        )
                continue

            if self._network.out_degree(node) != 1:
                raise PopulationGeneticsError(
                    f"admixture node {node.label!r} must have exactly one child."
                )

            gammas: list[float] = []
            for edge in incoming:
                try:
                    gamma = float(edge.get_gamma())
                except (TypeError, ValueError) as exc:
                    raise PopulationGeneticsError(
                        "admixture proportions must be numeric."
                    ) from exc
                if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
                    raise PopulationGeneticsError(
                        "each admixture proportion must be finite and strictly "
                        f"between zero and one; got {gamma} on "
                        f"{edge.src.label!r} -> {node.label!r}."
                    )
                gammas.append(gamma)

            if not math.isclose(
                sum(gammas), 1.0, rel_tol=0.0, abs_tol=self._tolerance
            ):
                raise PopulationGeneticsError(
                    f"incoming proportions at admixture node {node.label!r} "
                    f"must sum to one; got {sum(gammas)}."
                )

    def _resolve_node(self, node: Node | str) -> Node:
        """Resolve a node object or exact label against the wrapped network."""
        if isinstance(node, str):
            resolved = self._network.has_node_named(node)
            if resolved is None:
                raise PopulationGeneticsError(
                    f"unknown population node {node!r}."
                )
            return resolved
        if isinstance(node, Node):
            for existing in self._network.V():
                if existing is node:
                    return existing
        raise PopulationGeneticsError("node does not belong to this graph.")

    def __repr__(self) -> str:
        return (
            f"AdmixtureGraph({len(self._network.V())} nodes, "
            f"{len(self._network.E())} edges, "
            f"{len(self.admixture_nodes)} admixture events)"
        )


def _count_matrix(values: Any, name: str) -> np.ndarray:
    """Return a copied two-dimensional nonnegative integer matrix."""
    masked = np.ma.asarray(values)
    if np.any(np.ma.getmaskarray(masked)):
        raise PopulationGeneticsError(f"{name} cannot contain missing values.")
    try:
        numeric = np.asarray(masked.data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PopulationGeneticsError(
            f"{name} must contain only numeric counts."
        ) from exc
    if numeric.ndim != 2:
        raise PopulationGeneticsError(
            f"{name} must be two-dimensional; got {numeric.ndim} dimensions."
        )
    if np.any(~np.isfinite(numeric)):
        raise PopulationGeneticsError(f"{name} cannot contain missing values.")
    if np.any(numeric < 0) or np.any(numeric != np.floor(numeric)):
        raise PopulationGeneticsError(
            f"{name} must contain nonnegative integer counts."
        )
    return numeric.astype(np.int64)


def _labels(
    values: Sequence[str], expected: int, kind: str
) -> tuple[str, ...]:
    """Validate a label sequence and return an immutable copy."""
    if isinstance(values, str):
        raise PopulationGeneticsError(
            f"{kind} labels must be a sequence, not one string."
        )
    labels = tuple(values)
    if len(labels) != expected:
        raise PopulationGeneticsError(
            f"expected {expected} {kind} labels; got {len(labels)}."
        )
    if any(not isinstance(label, str) or not label for label in labels):
        raise PopulationGeneticsError(
            f"{kind} labels must be non-empty strings."
        )
    if len(set(labels)) != len(labels):
        raise PopulationGeneticsError(f"{kind} labels must be unique.")
    return labels


def _genotype_matrix(values: Any) -> np.ndarray:
    """Coerce a sample-by-site genotype matrix, preserving missing calls."""
    masked = np.ma.asarray(values, dtype=object)
    raw = np.asarray(masked.data, dtype=object)
    mask = np.ma.getmaskarray(masked)
    if raw.ndim != 2:
        raise PopulationGeneticsError(
            "genotypes must be a two-dimensional sample-by-site matrix."
        )

    matrix = np.full(raw.shape, np.nan, dtype=float)
    for index, value in np.ndenumerate(raw):
        if mask[index] or _is_missing(value):
            continue
        try:
            count = float(value)
        except (TypeError, ValueError) as exc:
            raise PopulationGeneticsError(
                f"genotype at row {index[0]}, site {index[1]} is not numeric."
            ) from exc
        if (
            not math.isfinite(count)
            or count < 0.0
            or count != math.floor(count)
        ):
            raise PopulationGeneticsError(
                "observed genotypes must be nonnegative integer "
                "alternate-allele counts."
            )
        matrix[index] = count
    return matrix


def _is_missing(value: Any) -> bool:
    """Recognize common in-memory and VCF-style missing genotype markers."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", ".", "?", "-", "./.", ".|."}
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _ploidies(
    value: int | Sequence[int], n_samples: int
) -> np.ndarray:
    """Validate scalar or per-sample ploidy."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        raw = [int(value)] * n_samples
    else:
        try:
            raw = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise PopulationGeneticsError(
                "ploidy must be an integer or one integer per sample."
            ) from exc
        if len(raw) != n_samples:
            raise PopulationGeneticsError(
                f"expected {n_samples} ploidy values; got {len(raw)}."
            )

    ploidies: list[int] = []
    for item in raw:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, np.integer))
            or int(item) <= 0
        ):
            raise PopulationGeneticsError(
                "ploidy values must be positive integers."
            )
        ploidies.append(int(item))
    return np.asarray(ploidies, dtype=np.int64)


def _population_member_indices(
    samples: tuple[str, ...],
    population_members: Mapping[str, Sequence[str]],
) -> dict[str, tuple[int, ...]]:
    """Resolve a complete, non-overlapping population membership mapping."""
    if not population_members:
        raise PopulationGeneticsError(
            "at least one population membership group is required."
        )

    sample_indices = {sample: index for index, sample in enumerate(samples)}
    assigned: set[str] = set()
    result: dict[str, tuple[int, ...]] = {}

    for population, members_value in population_members.items():
        if not isinstance(population, str) or not population:
            raise PopulationGeneticsError(
                "population labels must be non-empty strings."
            )
        if isinstance(members_value, str):
            raise PopulationGeneticsError(
                f"members of population {population!r} must be a sequence of "
                "sample labels, not one string."
            )
        members = tuple(members_value)
        if not members:
            raise PopulationGeneticsError(
                f"population {population!r} has no samples."
            )

        indices: list[int] = []
        for sample in members:
            if sample not in sample_indices:
                raise PopulationGeneticsError(
                    f"population {population!r} references unknown sample "
                    f"{sample!r}."
                )
            if sample in assigned:
                raise PopulationGeneticsError(
                    f"sample {sample!r} belongs to more than one population."
                )
            assigned.add(sample)
            indices.append(sample_indices[sample])
        result[population] = tuple(indices)

    unassigned = set(samples) - assigned
    if unassigned:
        raise PopulationGeneticsError(
            "every sample must belong to exactly one population; unassigned: "
            f"{sorted(unassigned)}."
        )
    return result
