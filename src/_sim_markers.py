#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Exact unlinked biallelic-marker simulation under the MSNC."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from .Network import Network, Node
from ._sim_seq import simulate_gene_tree
from ._units import (
    BranchLengthUnit,
    BranchThetaKey,
    require_branch_length_unit,
    validate_branch_thetas,
)


def _mutate_biallelic_state(
    state: int,
    length: float,
    u: float,
    v: float,
    rng: np.random.Generator,
) -> int:
    """Evolve one green/red lineage under the exact two-state CTMC."""

    length = float(length)
    if not np.isfinite(length) or length < 0.0:
        raise ValueError(
            f"gene-tree branch lengths must be finite and non-negative; "
            f"got {length}."
        )
    total = u + v
    decay = np.exp(-total * length)
    if state == 1:
        p_red = v / total + (u / total) * decay
    else:
        p_red = (v / total) * (1.0 - decay)
    return int(rng.random() < p_red)


def simulate_biallelic_on_gene_tree(
    gene_tree: Network,
    u: float,
    v: float,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Simulate one biallelic site down an already sampled genealogy."""

    require_branch_length_unit(
        gene_tree,
        BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        context="biallelic mutation simulation",
    )
    if not np.isfinite(u) or not np.isfinite(v) or u <= 0.0 or v <= 0.0:
        raise ValueError(
            f"u and v must be finite and positive; got u={u}, v={v}."
        )
    root = gene_tree.root()
    states: dict[Node, int] = {
        root: int(rng.random() < (v / (u + v)))
    }
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in sorted(
            gene_tree.get_children(parent), key=lambda node: node.label
        ):
            edge = gene_tree.get_edge(parent, child)
            states[child] = _mutate_biallelic_state(
                states[parent], edge.get_length(), u, v, rng
            )
            stack.append(child)
    return {
        leaf.label: states[leaf]
        for leaf in sorted(
            gene_tree.get_leaves(), key=lambda node: node.label
        )
    }


def simulate_biallelic_markers(
    network: Network,
    n_sites: int,
    mapping: Mapping[str, list[str]],
    *,
    theta: float,
    u: float,
    v: float,
    branch_thetas: Optional[Mapping[BranchThetaKey, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, list[int]]:
    """Simulate independent red-allele counts under the Bryant/MSNC model.

    Every marker receives an independently sampled MSNC genealogy. A
    stationary two-state mutation process then evolves down that genealogy,
    and sampled gene-copy states are aggregated into species-level red counts.
    """

    require_branch_length_unit(
        network,
        BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        context="biallelic-marker simulation",
    )
    if n_sites <= 0:
        raise ValueError(f"n_sites must be positive; got {n_sites}.")
    if not np.isfinite(theta) or theta <= 0.0:
        raise ValueError(f"theta must be finite and positive; got {theta}.")
    if not np.isfinite(u) or not np.isfinite(v) or u <= 0.0 or v <= 0.0:
        raise ValueError(
            f"u and v must be finite and positive; got u={u}, v={v}."
        )
    validate_branch_thetas(branch_thetas)

    species = {leaf.label for leaf in network.get_leaves()}
    if set(mapping) != species:
        raise ValueError(
            "marker mapping keys must exactly match network leaves; "
            f"expected {sorted(species)}, got {sorted(mapping)}."
        )
    sampled_alleles: set[str] = set()
    normalized: dict[str, list[str]] = {}
    for taxon, labels in mapping.items():
        copied = [str(label) for label in labels]
        if not copied:
            raise ValueError(f"taxon {taxon!r} must have at least one sample.")
        for label in copied:
            if label in sampled_alleles:
                raise ValueError(f"duplicate sampled allele label {label!r}.")
            sampled_alleles.add(label)
        normalized[taxon] = copied

    generator = rng if rng is not None else np.random.default_rng()
    data = {taxon: [] for taxon in normalized}
    for _ in range(n_sites):
        genealogy = simulate_gene_tree(
            network,
            normalized,
            theta,
            generator,
            branch_thetas=branch_thetas,
        )
        states = simulate_biallelic_on_gene_tree(genealogy, u, v, generator)
        for taxon, labels in normalized.items():
            data[taxon].append(sum(states[label] for label in labels))
    return data


__all__ = [
    "simulate_biallelic_markers",
    "simulate_biallelic_on_gene_tree",
]
