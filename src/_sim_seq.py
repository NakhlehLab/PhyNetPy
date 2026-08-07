#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

"""
Author : Mark Kessler
First Included in Version : 0.5.0

Coalescent simulation of multilocus sequence data under the multispecies
*network* coalescent (MSNC) -- the *generative* counterpart of the likelihood
in :mod:`phynetpy._seq_likelihood` and the sampler in
:mod:`phynetpy._mcmc_seq`.

Given a known (true) species network with node heights, a population mutation
rate ``theta``, and a sampling scheme (how many gene copies / alleles are drawn
per species), this module:

1. **Simulates a gene tree per locus** by running the coalescent *backward in
   time* through the branches of the species network.  Within a branch ``u``
   ancestral lineages coalesce at total rate ``u(u-1)/theta`` (each of the
   ``C(u,2)`` pairs at rate ``2/theta``); at a reticulation node every lineage
   independently follows one of the two parent edges with probability ``gamma``
   / ``1 - gamma``; at the root the remaining lineages coalesce on an infinite
   branch down to a single MRCA.  This is *exactly* the process whose density
   :func:`phynetpy._seq_likelihood.gene_tree_msnc_log_density` evaluates, so
   simulated data are a fair, self-consistent test bed for the sampler.

2. **Evolves an alignment down each gene tree** under any
   :class:`~phynetpy._seq_likelihood.SubstitutionModel` (JC69 / HKY85 / GTR):
   the root sequence is drawn from the model's stationary frequencies and each
   site is propagated across every branch with that branch's transition matrix
   ``P(t) = exp(Qt)`` (vectorised across all sites at once).

These are the primitives behind the public :func:`phynetpy.infer.simulate`
verb, which wraps them and returns data-axis objects that feed straight back
into :func:`phynetpy.infer.infer`.  Reach for this module directly only when
you want a single gene tree or a bare ``{label: sequence}`` dict.

The result is a :class:`SimulatedData` bundle that drops straight into
:class:`phynetpy._mcmc_seq.MCMC_SEQ` (``MCMC_SEQ(**data.to_mcmc_seq_kwargs())``),
so you can check *recovery* (does the sampler find the true topology /
divergence times?) and *calibration* (do the 95% HPD intervals cover the truth
at the nominal rate?) against a known ground truth.

Units match the rest of the MCMC_SEQ stack: heights / branch lengths are in
expected substitutions per site and ``theta = 4 N mu`` (per-pair coalescent
rate ``2/theta``).

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .Network import Network, Node, Edge
from .GraphUtils import _edge_between, _node_heights
from ._seq_likelihood import (
    SubstitutionModel,
    JC69,
    DNA_STATES,
)


__all__ = [
    "SimulatedData",
    "simulate_gene_tree",
    "simulate_sequences",
    "simulate_multilocus",
]


# ======================================================================
# Result bundle
# ======================================================================

@dataclass
class SimulatedData:
    """A simulated multilocus data set with its ground truth.

    The fields under "inputs to inference" are exactly what
    :class:`phynetpy._mcmc_seq.MCMC_SEQ` consumes; the "ground truth" fields are
    what a recovery / calibration check compares the posterior against.

    Attributes:
        loci: Per-locus alignments (allele label -> aligned sequence string).
        mapping: Species label -> list of allele (gene-copy) labels sampled.
        species_of: Allele label -> species label (inverse of ``mapping``).
        gene_trees: The *true* simulated gene tree for each locus.
        true_network: The species network the data were generated on.
        true_theta: The population mutation rate used.
        model: The substitution model used.
        seq_length: Number of sites simulated per locus.
    """

    # -- inputs to inference -----------------------------------------
    loci: list[dict[str, str]]
    mapping: dict[str, list[str]]
    species_of: dict[str, str]
    # -- ground truth ------------------------------------------------
    gene_trees: list[Network]
    true_network: Network
    true_theta: float
    model: SubstitutionModel
    seq_length: int = 0

    def to_mcmc_seq_kwargs(self) -> dict[str, Any]:
        """Keyword arguments to construct an :class:`MCMC_SEQ` on this data.

        Returns:
            ``{"loci", "mapping", "model", "theta"}`` -- spreadable directly,
            e.g. ``MCMC_SEQ(**data.to_mcmc_seq_kwargs())``.  The true network
            and gene trees are deliberately *not* passed (let the sampler start
            from its own UPGMA guess so recovery is a fair test).
        """
        return {
            "loci": self.loci,
            "mapping": self.mapping,
            "model": self.model,
            "theta": self.true_theta,
        }


# ======================================================================
# Reticulation inheritance probabilities
# ======================================================================

def _reticulation_parent_gammas(
    net: Network, retic: Node
) -> list[tuple[Edge, float]]:
    """The two parent edges of a reticulation with normalised inheritance probs.

    Robust to networks parsed from extended Newick where only one of the two
    hybrid edges carried a ``[&gamma=...]`` annotation (the other defaulting to
    ``0``): if the two stored gammas do not already sum to ~1 we recover the
    complement, falling back to an even ``0.5 / 0.5`` split when neither edge
    was annotated.

    Args:
        net: The species network.
        retic: A reticulation node (in-degree 2).

    Returns:
        ``[(edge0, gamma0), (edge1, gamma1)]`` with ``gamma0 + gamma1 == 1``.
    """
    in_edges = list(net.in_edges(retic))
    if len(in_edges) != 2:
        raise ValueError(
            f"Reticulation {retic.label!r} must have exactly two parent edges."
        )
    e0, e1 = in_edges
    g0 = e0.get_gamma() or 0.0
    g1 = e1.get_gamma() or 0.0
    total = g0 + g1
    if abs(total - 1.0) < 1e-9:
        return [(e0, g0), (e1, g1)]
    if total <= 0.0:
        return [(e0, 0.5), (e1, 0.5)]
    # One edge annotated (the other left at the 0.0 default): treat the set
    # value as gamma and give its complement to the sibling edge.
    if g1 == 0.0:
        return [(e0, g0), (e1, 1.0 - g0)]
    if g0 == 0.0:
        return [(e0, 1.0 - g1), (e1, g1)]
    # Both set but not normalised: renormalise.
    return [(e0, g0 / total), (e1, g1 / total)]


# ======================================================================
# Gene-tree simulation (multispecies network coalescent)
# ======================================================================

class _Lineage:
    """One ancestral gene lineage during the coalescent sweep.

    Attributes:
        newick: Newick string of the subtree subtended by this lineage.
        height: Height (substitution units) of the lineage's youngest open end.
    """

    __slots__ = ("newick", "height")

    def __init__(self, newick: str, height: float) -> None:
        self.newick = newick
        self.height = height


def _coalesce_within(
    lineages: list[_Lineage],
    t_start: float,
    t_end: Optional[float],
    theta: float,
    rng: np.random.Generator,
    counter: list[int],
) -> list[_Lineage]:
    """Run the coalescent on ``lineages`` over ``[t_start, t_end)``.

    Each pair of lineages coalesces at rate ``2/theta``; with ``u`` lineages the
    total rate is ``u(u-1)/theta`` and the waiting time is
    ``Exponential(rate)``.  Coalescences that would occur at or beyond
    ``t_end`` do not happen on this (finite) branch; ``t_end = None`` denotes
    the infinite root branch, on which lineages coalesce until one remains.

    Args:
        lineages: Open lineages entering the bottom of the branch.
        t_start: Branch bottom height (younger endpoint).
        t_end: Branch top height, or ``None`` for the infinite root branch.
        theta: Population mutation rate on this branch.
        rng: Random generator.
        counter: One-element list holding the next internal-node id (mutated).

    Returns:
        The lineages exiting the top of the branch (length 1 when ``t_end`` is
        ``None``).
    """
    pool = list(lineages)
    t = t_start
    while len(pool) > 1:
        u = len(pool)
        rate = u * (u - 1) / theta
        wait = rng.exponential(1.0 / rate)
        if t_end is not None and t + wait >= t_end:
            break
        t += wait
        i, j = rng.choice(u, size=2, replace=False)
        a, b = pool[int(i)], pool[int(j)]
        label = f"i{counter[0]}"
        counter[0] += 1
        merged = _Lineage(
            f"({a.newick}:{t - a.height:.10f},{b.newick}:{t - b.height:.10f}){label}",
            t,
        )
        # Remove the two children (higher index first) and add the parent.
        for k in sorted((int(i), int(j)), reverse=True):
            pool.pop(k)
        pool.append(merged)
    return pool


def simulate_gene_tree(
    species_net: Network,
    alleles_per_species: dict[str, list[str]],
    theta: float,
    rng: np.random.Generator,
    *,
    pop_sizes: Optional[dict[Node, float]] = None,
) -> Network:
    """Simulate one gene tree under the multispecies network coalescent.

    Performs a backward-in-time, height-ordered sweep of ``species_net``.
    Sampled alleles enter at the leaves (height 0); lineages coalesce within
    each branch (:func:`_coalesce_within`); at a reticulation node each
    surviving lineage independently ascends one of the two parent edges with
    probability ``gamma`` / ``1 - gamma``; at the root the remaining lineages
    coalesce to a single MRCA.

    Args:
        species_net: Timed species network (ultrametric, substitution units)
            with reticulation inheritance probabilities on hybrid edges.
        alleles_per_species: Map species label -> the allele (gene-copy) labels
            to sample; these become the gene-tree leaf labels.  Species absent
            from the map contribute a single lineage named after the species.
        theta: Constant population mutation rate (used for every branch unless
            ``pop_sizes`` overrides it).
        rng: Random generator.
        pop_sizes: Optional per-branch ``theta`` keyed by the *child* node of
            the branch (the branch *above* that node), mirroring the density's
            ``pop_sizes`` argument.

    Returns:
        A rooted, ultrametric :class:`Network` (a tree) with branch lengths in
        substitution units and leaves labelled by the sampled alleles.
    """
    heights = _node_heights(species_net)
    counter: list[int] = [0]

    def theta_above(child: Node) -> float:
        if pop_sizes is not None and child in pop_sizes:
            return float(pop_sizes[child])
        return theta

    # Lineages waiting at the bottom (child end) of each species edge.
    edge_bottom: dict[Edge, list[_Lineage]] = {}

    # Seed the leaves.
    for leaf in species_net.get_leaves():
        labels = alleles_per_species.get(leaf.label, [leaf.label])
        lineages = [_Lineage(lab, 0.0) for lab in labels]
        in_edges = list(species_net.in_edges(leaf))
        if in_edges:  # leaf below the root
            edge_bottom.setdefault(in_edges[0], []).extend(lineages)
        else:  # degenerate single-leaf "network"
            edge_bottom[("__root__", leaf)] = lineages  # type: ignore[index]

    # Process every non-leaf node from youngest to oldest.
    internal = [v for v in species_net.V() if species_net.get_children(v)]
    internal.sort(key=lambda v: heights[v])

    root_newick: Optional[str] = None
    for v in internal:
        # Gather lineages arriving at v: coalesce each child branch up to h[v].
        arriving: list[_Lineage] = []
        for child in species_net.get_children(v):
            e = _edge_between(species_net, v, child)
            bottom = edge_bottom.get(e, [])
            top = _coalesce_within(
                bottom, heights[child], heights[v], theta_above(child), rng, counter
            )
            arriving.extend(top)

        parents = species_net.get_parents(v)
        if not parents:
            # Root: coalesce remaining lineages on the infinite branch.
            final = _coalesce_within(
                arriving, heights[v], None, theta_above(v), rng, counter
            )
            root_newick = final[0].newick if final else ""
        elif v.is_reticulation():
            # Distribute lineages across the two parent edges by gamma.
            (eA, gA), (eB, _gB) = _reticulation_parent_gammas(species_net, v)
            for lineage in arriving:
                target = eA if rng.random() < gA else eB
                edge_bottom.setdefault(target, []).append(lineage)
        else:
            # Tree node: all arriving lineages enter the single parent branch.
            pe = _edge_between(species_net, parents[0], v)
            edge_bottom.setdefault(pe, []).extend(arriving)

    if not root_newick:
        raise RuntimeError("Coalescent simulation produced no root lineage.")
    if "(" not in root_newick:
        # A single sampled lineage: build a trivial one-node tree.
        net = Network()
        net.add_nodes(Node(name=root_newick))
        return net
    return Network.from_newick(root_newick + ";")


# ======================================================================
# Sequence simulation (evolve an alignment down a gene tree)
# ======================================================================

def _evolve_states(
    parent_states: np.ndarray, P: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample child states for every site given parent states and ``P(t)``.

    Vectorised inverse-CDF sampling: for each site we draw the child state from
    the row of ``P`` indexed by that site's parent state.

    Args:
        parent_states: ``int`` array of length ``n_sites`` (values in 0..3).
        P: ``4x4`` row-stochastic transition matrix.
        rng: Random generator.

    Returns:
        ``int`` array of child states, length ``n_sites``.
    """
    cdf = np.cumsum(P, axis=1)          # (4, 4); cdf[s, 3] == 1
    rows = cdf[parent_states]           # (n_sites, 4)
    u = rng.random(parent_states.shape[0])
    return (u[:, None] < rows).argmax(axis=1).astype(np.intp)


def simulate_sequences(
    gene_tree: Network,
    model: SubstitutionModel,
    n_sites: int,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Evolve an alignment of ``n_sites`` down ``gene_tree`` under ``model``.

    The root sequence is drawn i.i.d. from the model's stationary
    frequencies; each branch then mutates every site with that branch's
    transition matrix ``P(t)``.

    Args:
        gene_tree: Rooted gene tree with branch lengths in substitution units.
        model: Nucleotide substitution model.
        n_sites: Number of sites (alignment columns) to simulate.
        rng: Random generator.

    Returns:
        Map from leaf label -> simulated nucleotide string (length ``n_sites``,
        alphabet ``A,C,G,T``).
    """
    if n_sites <= 0:
        raise ValueError("n_sites must be positive.")
    root = gene_tree.root()
    states: dict[Node, np.ndarray] = {
        root: rng.choice(4, size=n_sites, p=model.pi)
    }

    # Pre-order: propagate each parent's states to its children.
    stack = [root]
    while stack:
        node = stack.pop()
        parent_states = states[node]
        for child in gene_tree.get_children(node):
            edge = _edge_between(gene_tree, node, child)
            t = edge.get_length()
            t = 0.0 if t is None else float(t)
            P = model.p_matrix(t)
            states[child] = _evolve_states(parent_states, P, rng)
            stack.append(child)

    out: dict[str, str] = {}
    for leaf in gene_tree.get_leaves():
        seq = states[leaf]
        out[leaf.label] = "".join(DNA_STATES[int(s)] for s in seq)
    return out


# ======================================================================
# End-to-end multilocus simulation
# ======================================================================

def simulate_multilocus(
    species_net: Network,
    mapping: dict[str, list[str]],
    n_loci: int,
    seq_length: int,
    *,
    theta: float = 0.02,
    model: Optional[SubstitutionModel] = None,
    pop_sizes: Optional[dict[Node, float]] = None,
    seed: Any = None,
) -> SimulatedData:
    """Simulate a full multilocus data set on a known species network.

    For each of ``n_loci`` independent loci a gene tree is drawn under the MSNC
    (:func:`simulate_gene_tree`) and an alignment is evolved down it
    (:func:`simulate_sequences`).

    Args:
        species_net: The *true* timed species network (with gammas on hybrid
            edges).
        mapping: Species label -> list of allele labels to sample at every
            locus (use one allele per species for the single-individual case,
            or several to exercise multi-allele coalescence).
        n_loci: Number of independent loci to simulate.
        seq_length: Sites per locus.
        theta: Population mutation rate (constant across branches unless
            ``pop_sizes`` is given).
        model: Substitution model (default :class:`JC69`).
        pop_sizes: Optional per-branch ``theta`` keyed by child node.
        seed: Seed for all randomness.

    Returns:
        A :class:`SimulatedData` bundle ready for
        ``MCMC_SEQ(**data.to_mcmc_seq_kwargs())``.
    """
    if n_loci <= 0:
        raise ValueError("n_loci must be positive.")
    rng = np.random.default_rng(seed)
    model = model if model is not None else JC69()
    species_of = {a: sp for sp, alleles in mapping.items() for a in alleles}

    loci: list[dict[str, str]] = []
    gene_trees: list[Network] = []
    for _ in range(n_loci):
        gt = simulate_gene_tree(
            species_net, mapping, theta, rng, pop_sizes=pop_sizes
        )
        aln = simulate_sequences(gt, model, seq_length, rng)
        gene_trees.append(gt)
        loci.append(aln)

    return SimulatedData(
        loci=loci,
        mapping=mapping,
        species_of=species_of,
        gene_trees=gene_trees,
        true_network=species_net,
        true_theta=theta,
        model=model,
        seq_length=seq_length,
    )
