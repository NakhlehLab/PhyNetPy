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
Last Stable Edit : 10/30/26
First Included in Version : 0.7.0

Docs   - [x]
Tests  - [x]
Design - [x]

DIMPLE: divide a tree of blobs, infer each piece, stitch.

The scaffold's polytomies (out-degree greater than 2) are the blobs. This is
not :func:`phynetpy.GraphUtils.blobs`, which is a biconnected-component
decomposition. Each blob is cut into subsets of at most ``max_subset_size``
leaves, each subset is inferred with the existing MSC triplet
pseudo-likelihood engine, and the pieces are grafted back onto the scaffold.

Neighbor-joining of the overlapping subset trees follows NJMerge
(Molloy and Warnow, 2018, *Algorithms for Molecular Biology*). Each join
minimizes the Saitou and Nei (1987, *Molecular Biology and Evolution*)
criterion Q(i, j) = (n - 2) d(i, j) - r_i - r_j. The subset search itself
is Yu and Nakhleh's triplet pseudo-likelihood, already implemented by
:func:`phynetpy.infer.infer`.

The entry point is :func:`infer_dimple`. It is not a cell of the
``(data, model, criterion)`` registry: the partition and the stitch are
specific to this pipeline.
"""

from __future__ import annotations

import math
import random
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Callable, Optional, Sequence

import networkx as nx
import numpy as np

from .criteria import PseudoLikelihood
from .data import GeneTrees
from .GraphUtils import (
    add_hybrid,
    count_reticulations,
    get_all_subtrees,
    group_reticulations,
    induced_subnetwork_by_taxa,
    is_tree,
    reticulation_signature,
    rooted_triples,
)
from .models import MSC
from .ModelSelection import reticulation_sweep
from .Network import Edge, Network, NetworkError, Node
from ._mcmc_gt import _populate_default_branch_lengths
from ._units import BranchLengthUnit

# A near-duplicate of a reticulation already accepted has to clear this
# relative gain. Any strict gain is enough for a placement that is not near
# one. Relative gain is (new - current) / |current| on the log
# pseudo-likelihood (higher is better).
REDUNDANCY_THRESHOLD = 0.02

# A blob taxon's distance to the virtual outgroup is this factor times its
# mean gene-tree distance to the scaffold siblings. DIMPLE's merger uses 2.0
# (``compute_outgroup_distances``): on estimated gene trees the raw mean let
# NJ slip the outgroup between blob leaves. 1.0 is exact on an additive tree
# metric (``test_dimple_nj`` pins that), but on DIMPLE's own subnetworks the
# merger reproduced its network for 79% of blobs at 2.0 against 69% at 1.0
# (``reports/dimple_merger_data_final_after_fix*.md``).
OUTGROUP_SCALE = 2.0

# DIMPLE fills missing divisions from the full partition enumeration only
# when a blob has at most this many items; beyond it, the seeded packings
# are all there is.
ENUM_MAX_ITEMS = 12

# Per-subset model selection. Each reticulation is charged this many
# parameters (one inheritance probability and two branch lengths) in the
# AIC/BIC rules, both when a subset's reticulation count is chosen and
# when the merger decides whether a candidate reticulation earns its place.
SELECTION_PARAMS_PER_RETICULATION = 3
# The ``gain`` rule: a reticulation is kept only when it raises the log
# pseudo-likelihood by more than this. Calibrated on 86 DIMPLE subsets of
# the 50-taxon replicates (reports/dimple_retic_calibration.md): true trees
# gain a median of 0 and a 75th percentile of 1.6, true networks a median of
# 347 and a lower quartile of 134; AIC's 3 let one true tree in five
# through, while 50 to 100 kept 40 to 41 of 44 trees and found 34 to 35 of
# 42 networks. This is DIMPLE's rule only; nothing else in PhyNetPy uses it.
SELECTION_GAIN_THRESHOLD = 50.0
SELECTION_RULES = ("gain", "aic", "bic", "elbow")

# Subset searches are simulated annealing, three independent runs, and
# the network with the higher log pseudo-likelihood is kept. The
# schedule is the 7-taxon geometric-reheat harness. 5000 iterations
# cools from 125 to about 4; it does not reach the 0.05 floor, and the
# run does not stop early. Callers override any of these by passing the
# same names to infer_dimple (num_iter, n_runs, method, t_start, ...).
SUBSET_ITERS = 5000
SUBSET_RUNS = 3
_SUBSET_ANNEALING = {
    "schedule": "geometric_reheat",
    "t_start": 125.0,
    "t_end": 0.05,
    "cooling_alpha": 0.95,
    "steps_per_temp": 75,
    "reheat_threshold": 600,
    "reheat_factor": 1.8,
    "reheat_window": 250,
    "reheat_min_improve": 1.0,
    "reheat_on_no_uphill": True,
    "reheat_cap_mult": 1.0,
    "reheat_max_consecutive": 4,
}

# Only used when a caller forces method="hc" on a reticulate subset.
# The kernel default of 150 non-improvements before the reticulation
# phase was chosen for a long annealing run.
SUBSET_PHASE_PATIENCE = 40

# Network.clean selects routines by index: 0 drops degree-0 floaters,
# 1 drops a single-child root, 2 contracts in-degree-1/out-degree-1
# chains. The stitch still uses the root as an attachment point, so
# only the contraction runs.
_CONTRACT_DEGREE2 = [False, False, True]

_OUT = "OUT"


class DimpleError(ValueError):
    """The DIMPLE inputs or a stitch step cannot be carried out."""


# ── Records ───────────────────────────────────────────────────────────


@dataclass
class PrunedSubtree:
    """A non-blob clade cut off a blob, and where it has to be put back.

    Attributes:
        leaves: Taxa in the cut clade.
        anchor_leaves: Taxa that were siblings of the cut at the moment it
            was made, on the residual tree (earlier cuts already removed).
        parent_leaves: Every taxon under the cut's parent at that moment,
            including the cut. Identifies the parent on the scaffold.
        subtree: The cut clade as its own network.
        source_type: ``blob_item`` when the anchor still sits in the blob,
            ``prune`` when the anchor is another pruned clade.
        index: Stable tie-break for attachment order.
    """

    leaves: frozenset[str]
    anchor_leaves: frozenset[str]
    parent_leaves: frozenset[str]
    subtree: Network
    source_type: str = "blob_item"
    index: int = 0


@dataclass
class BlobJob:
    """One mega-blob's partition.

    Attributes:
        name: Member labels joined by ``+``.
        tob_leaves: Scaffold leaves of this blob, including nested blobs
            and clades that were pruned.
        external_siblings: Scaffold leaves outside the blob, under the
            blob's parent. Empty when the blob is the root.
        divisions: ``n_divisions`` partitions. Each partition is a list of
            taxon sets, every one of size at most ``max_subset_size``.
        pruned: Clades cut off before inference, in the order they were cut.
        is_root: The blob node is the scaffold root.
    """

    name: str
    tob_leaves: frozenset[str]
    external_siblings: frozenset[str]
    divisions: list[list[frozenset[str]]]
    pruned: list[PrunedSubtree] = field(default_factory=list)
    is_root: bool = False


@dataclass
class SubsetRecord:
    """One subset search.

    Attributes:
        blob_name: Blob the subset belongs to (empty until the pipeline
            files the record).
        taxa: Leaves that were inferred.
        network: The chosen network.
        score: Log pseudo-likelihood of ``network`` on the subset gene
            trees, re-scored after the search.
        bound: The reticulation cap the search was given (the true count
            when a ground-truth network was supplied).
        k_selected: Reticulation cap the chosen network came from. Equals
            ``bound`` without model selection.
        selection: ``gain``, ``aic``, ``bic``, ``elbow``, or None.
        sweep: One dict per cap tried (``k``, ``best_log_lik``, ``n_params``,
            ``aic``, ``bic``, ``delta_log_lik``, ``n_retics``), or None.
        chains: One dict per search chain at the chosen cap (``method``,
            ``num_iter``, ``seed``, ``score``, ``seconds``, ``chosen``), or
            None when the record was built without them.
    """

    blob_name: str
    taxa: frozenset[str]
    network: Network
    score: float
    bound: int = 1
    k_selected: Optional[int] = None
    selection: Optional[str] = None
    sweep: Optional[list[dict]] = None
    chains: Optional[list[dict]] = None


@dataclass
class ReticTrial:
    """One attempted reticulation insertion.

    Attributes:
        child_leaves: Leaves below the reticulation.
        edge_leaves: Leaves below the edge the minor parent was placed on.
            Empty when no edge was a candidate.
        accepted: Whether the trial replaced the current network.
        score_before: Log pseudo-likelihood before the trial.
        score_after: Log pseudo-likelihood of the trial (or ``score_before``
            when there was nothing to try).
        reason: Why a trial was not scored. ``child_not_clade``,
            ``no_candidate_edge``, or ``no_destination``. Empty when the
            candidate was scored.
    """

    child_leaves: frozenset[str]
    edge_leaves: frozenset[str]
    accepted: bool
    score_before: float
    score_after: float
    reason: str = ""


@dataclass
class BlobResult:
    """One blob after merging and reattachment.

    Attributes:
        job: The partition this blob was inferred from.
        subsets: Subset inferences, in division order.
        merged: Blob network with pruned clades reattached.
        nj_relaxations: How many neighbor-joining steps ignored a constraint.
        trials: Reticulation insertions, in the order they were tried.
    """

    job: BlobJob
    subsets: list[SubsetRecord]
    merged: Network
    nj_relaxations: int = 0
    trials: list[ReticTrial] = field(default_factory=list)


@dataclass
class DimpleTrace:
    """What :func:`infer_dimple` did, attached as ``InferenceResult.raw``.

    Attributes:
        blobs: One entry per mega-blob, scaffold order.
        final: The full-taxon network.
        score: Log pseudo-likelihood of ``final`` on the gene trees, or
            None when the caller skipped it.
    """

    blobs: list[BlobResult]
    final: Network
    score: Optional[float]


@dataclass
class GraftPiece:
    """A combined blob ready to replace its scaffold clade.

    Attributes:
        tob_leaves: Scaffold leaves this piece replaces.
        network: Combined blob network.
        anchor_leaves: Where to hang this piece if an outer graft has to
            reattach it. The blob's external siblings.
    """

    tob_leaves: frozenset[str]
    network: Network
    anchor_leaves: frozenset[str]


# ── Small graph helpers ───────────────────────────────────────────────


def _labels(net: Network, node: Optional[Node] = None) -> frozenset[str]:
    """Leaf labels of ``net``, or of the clade under ``node``."""
    if node is None:
        return frozenset(leaf.label for leaf in net.get_leaves())
    return frozenset(leaf.label for leaf in net.leaf_descendants(node))


def _label_map(net: Network) -> dict[Node, frozenset[str]]:
    """Leaf labels under every node, from one pass over ``net``.

    The clade lookups, compatibility checks and candidate-edge scans each
    need the leaf set of every node; one ``leaf_descendants_all`` is far
    cheaper than a ``leaf_descendants`` per node.
    """
    return {
        node: frozenset(leaf.label for leaf in leaves)
        for node, leaves in net.leaf_descendants_all().items()
    }


def _copy(net: Network) -> Network:
    """A structural copy of ``net``."""
    return net.copy()[0]


def _blobs(net: Network) -> list[Node]:
    """Polytomies: nodes with more than two children. Not biconnected components."""
    return [node for node in net.V() if net.out_degree(node) > 2]


def _is_binary_tree(net: Network) -> bool:
    """A rooted tree in which every internal node has exactly two children."""
    if not net.is_acyclic():
        return False
    for node in net.V():
        if net.in_degree(node) > 1 or node.is_reticulation():
            return False
        out = net.out_degree(node)
        if out == 0:
            continue
        if out != 2:
            return False
    return True


def _node_with_leaves(net: Network, leaves: frozenset[str]) -> Optional[Node]:
    """The node whose descendant leaf set is exactly ``leaves``, if there is one."""
    target = frozenset(leaves)
    for node, below in _label_map(net).items():
        if below == target:
            return node
    return None


def _below_retics(net: Network, node: Node) -> Node:
    """Walk down through stacked reticulations to the first ordinary node.

    A pruned clade attached at a reticulation has to land inside the hybrid
    clade. Each reticulation has one child; stop when the node is not one.
    """
    current = node
    while net.in_degree(current) > 1:
        children = net.get_children(current)
        if len(children) != 1:
            break
        current = children[0]
    return current


def _detach(net: Network, parent: Node, child: Node) -> Network:
    """Cut ``parent -> child`` and return the clade under ``child``.

    ``net`` loses that clade. The returned network is an induced copy, so
    it does not share nodes with ``net``. Degree-2 nodes left on ``net``
    are contracted; the root is left in place.

    Args:
        net (Network): Network mutated by the cut.
        parent (Node): Source of the cut edge.
        child (Node): Root of the clade being removed.

    Returns:
        Network: The removed clade on its own leaves.

    Raises:
        NetworkError: If a leaf under ``child`` cannot be induced.
    """
    # Capture the clade before the cut so the returned network does not
    # share nodes with ``net``.
    clade = induced_subnetwork_by_taxa(net, sorted(_labels(net, child)))
    doomed = list(net.get_subtree_at(child))
    net.remove_edge(net.get_edge(parent, child))
    for node in doomed:
        if node in net.V():
            net.remove_nodes(node)
    # The cut leaves a degree-2 chain between the parent and its surviving child.
    net.clean(_CONTRACT_DEGREE2)
    return clade


def strip_outgroup(net: Network, outgroup: Optional[str]) -> Network:
    """Copy ``net`` without the outgroup leaf.

    The label is only removed from the scaffold. It is not added to gene
    trees. An absent label, or ``None``, returns a plain copy.

    Args:
        net (Network): Rooted scaffold.
        outgroup (str | None): Leaf label to drop.

    Returns:
        Network: Scaffold on the remaining leaves.
    """
    if outgroup is None or net.has_node_named(outgroup) is None:
        return _copy(net)
    keep = [leaf.label for leaf in net.get_leaves() if leaf.label != outgroup]
    if len(keep) < 3:
        raise DimpleError(
            f"Removing outgroup '{outgroup}' leaves fewer than 3 taxa."
        )
    # The MRCA of the remaining leaves is the new root, so a root edge that
    # existed only to hold the outgroup disappears with it.
    stripped = induced_subnetwork_by_taxa(net, keep)
    stripped.clean()
    return stripped


def _ensure_internal_labels(net: Network) -> None:
    """Give every unlabeled node a name so blob identity survives copies.

    ``induced_subnetwork_by_taxa`` keeps labels and drops nodes that have
    none. ``Network._fresh_label`` is the same collision check the graft
    path uses.

    Args:
        net (Network): Scaffold or blob piece, mutated in place.
    """
    for node in net.V():
        if not node.label:
            net.update_node_name(node, net._fresh_label("B"))


# ── Input checks ──────────────────────────────────────────────────────


def validate_inputs(
    gene_trees: GeneTrees,
    tree_of_blobs: Network,
    start_tree: Network,
    *,
    max_subset_size: int = 12,
    n_divisions: int = 15,
    outgroup: Optional[str] = None,
) -> None:
    """Reject a scaffold, start tree, or gene-tree set DIMPLE cannot run on.

    Incomplete gene trees are allowed: a locus may simply lack a taxon.
    A reticulation on the scaffold is not. A blob here is a polytomy on a
    tree.

    Args:
        gene_trees (GeneTrees): One rooted tree per locus.
        tree_of_blobs (Network): Rooted scaffold whose polytomies are blobs.
        start_tree (Network): Binary tree on the scaffold's taxa.
        max_subset_size (int): Largest subset. Must be at least 4.
        n_divisions (int): Alternative partitions per blob. Must be at least 1.
        outgroup (str | None): Scaffold leaf to ignore. May be absent.

    Raises:
        DimpleError: If a size is illegal, the scaffold is reticulate, the
            start tree is not a binary tree on the scaffold taxa, or a gene
            tree is itself a network.
    """
    if max_subset_size < 4:
        raise DimpleError("max_subset_size must be at least 4.")
    if n_divisions < 1:
        raise DimpleError("n_divisions must be at least 1.")
    if not isinstance(gene_trees, GeneTrees):
        raise DimpleError("gene_trees must be a GeneTrees object.")

    for node in tree_of_blobs.V():
        if node.is_reticulation() or tree_of_blobs.in_degree(node) > 1:
            raise DimpleError(
                "The tree of blobs contains a reticulation. A blob is a "
                "polytomy (out-degree > 2) on a tree, not a biconnected "
                "component."
            )
    for tree in gene_trees.trees:
        if not is_tree(tree) or any(tree.in_degree(node) > 1 for node in tree.V()):
            raise DimpleError("A gene tree is a network; gene trees must be trees.")

    scaffold_leaves = _labels(tree_of_blobs)
    if outgroup is not None:
        scaffold_leaves = frozenset(leaf for leaf in scaffold_leaves if leaf != outgroup)
    start_leaves = _labels(start_tree)
    missing = scaffold_leaves - start_leaves
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise DimpleError(f"The start tree is missing scaffold taxa: {sample}.")
    if not _is_binary_tree(start_tree):
        raise DimpleError(
            "The start tree is not binary. A polytomous tree of blobs is "
            "not a legal pseudo-likelihood start."
        )


# ── Partition ─────────────────────────────────────────────────────────


def _forbidden(net: Network, blobs: Sequence[Node]) -> set[tuple[str, str]]:
    """Original out-edges of every polytomy, as label pairs. Never cut these."""
    banned = set()
    for blob in blobs:
        for child in net.get_children(blob):
            banned.add((blob.label, child.label))
    return banned


def _mega_groups(net: Network, blobs: Sequence[Node]) -> list[list[Node]]:
    """Union blobs that are joined by a direct parent-child edge.

    A polytomy nested inside another through an ordinary node stays its own
    mega-blob. Only an edge from one polytomy to another merges them.
    """
    parent = {blob.label: blob.label for blob in blobs}
    by_label = {blob.label: blob for blob in blobs}

    def find(label: str) -> str:
        """Root of ``label``'s set, with path halving."""
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(a: str, b: str) -> None:
        """Merge the sets of ``a`` and ``b``."""
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    blob_set = set(blobs)
    for blob in blobs:
        for child in net.get_children(blob):
            if child in blob_set:
                union(blob.label, child.label)

    groups: dict[str, list[Node]] = {}
    for blob in blobs:
        groups.setdefault(find(blob.label), []).append(blob)
    return list(groups.values())


def _top_member(net: Network, members: Sequence[Node]) -> Node:
    """The member no other member is an ancestor of."""
    member_set = set(members)
    for blob in members:
        if not any(
            other is not blob and blob in net.get_subtree_at(other)
            for other in member_set
        ):
            return blob
    return members[0]


def _candidate_cuts(
    net: Network,
    forbidden: set[tuple[str, str]],
    blob_set: set[Node],
    max_subset_size: int,
) -> list[tuple[int, Edge]]:
    """Legal cuts: not a polytomy edge, no blob below, both sides at least 3, size at most S."""
    total = len(net.get_leaves())
    found = []
    for edge in net.E():
        if (edge.src.label, edge.dest.label) in forbidden:
            continue
        if edge.src in blob_set:
            continue
        # Cutting above a blob would drop that blob out of its allocation.
        if any(blob in net.get_subtree_at(edge.dest) for blob in blob_set):
            continue
        count = len(net.leaf_descendants(edge.dest))
        if count < 3 or total - count < 3 or count > max_subset_size:
            continue
        found.append((count, edge))
    return found


def _oversized(net: Network, blobs: Sequence[Node], max_subset_size: int) -> list[Node]:
    """Blobs whose children's leaf counts still sum to more than S."""
    too_big = []
    for blob in blobs:
        if blob not in net.V():
            continue
        size = sum(len(net.leaf_descendants(child)) for child in net.get_children(blob))
        if size > max_subset_size:
            too_big.append(blob)
    return too_big


def _prune_piece(
    net: Network,
    blobs: Sequence[Node],
    forbidden: set[tuple[str, str]],
    max_subset_size: int,
) -> list[PrunedSubtree]:
    """Cut the largest legal clade until every blob in ``net`` has size at most S.

    Mutates ``net``. Each recorded anchor is the sibling set on the tree as
    it stood when that cut was made, so an earlier cut is already gone.
    Stops when no blob is oversized or no legal cut remains, even if a blob
    is still larger than S.

    Args:
        net (Network): Working tree, mutated.
        blobs (Sequence[Node]): Polytomies whose child-leaf totals must fit in S.
        forbidden (set[tuple[str, str]]): Polytomy out-edges, as label pairs,
            that must not be cut.
        max_subset_size (int): S.

    Returns:
        list[PrunedSubtree]: Cuts in the order they were made.
    """
    pruned: list[PrunedSubtree] = []
    blob_set = set(blobs)
    while True:
        pending = _oversized(net, blobs, max_subset_size)
        if not pending:
            break
        cuts = _candidate_cuts(net, forbidden, blob_set, max_subset_size)
        if not cuts:
            break
        # Largest clade first. Equal sizes break toward the lexicographically
        # smaller leaf set so the same scaffold always cuts the same edge.
        cuts.sort(key=lambda item: (-item[0], tuple(sorted(_labels(net, item[1].dest)))))
        chosen = None
        for _, edge in cuts:
            if any(edge.dest in net.get_subtree_at(blob) for blob in pending):
                chosen = edge
                break
        if chosen is None:
            break
        parent_leaves = _labels(net, chosen.src)
        cut_leaves = _labels(net, chosen.dest)
        siblings = parent_leaves - cut_leaves
        clade = _detach(net, chosen.src, chosen.dest)
        pruned.append(PrunedSubtree(
            leaves=cut_leaves,
            anchor_leaves=siblings,
            parent_leaves=parent_leaves,
            subtree=clade,
            index=len(pruned),
        ))
    return pruned


def _split_clade(net: Network, leaves: frozenset[str], max_subset_size: int) -> list[frozenset[str]]:
    """Split a leaf set larger than S by the same size rules, with no blob edges to protect."""
    if len(leaves) <= max_subset_size:
        return [frozenset(leaves)]
    piece = induced_subnetwork_by_taxa(net, sorted(leaves))
    parts: list[frozenset[str]] = []
    # Each iteration removes at least 3 leaves, so this cannot run forever.
    while len(piece.get_leaves()) > max_subset_size:
        cuts = _candidate_cuts(piece, set(), set(), max_subset_size)
        if not cuts:
            raise DimpleError(
                f"A clade of {len(piece.get_leaves())} leaves has no legal "
                f"cut of size at most {max_subset_size}."
            )
        cuts.sort(key=lambda item: (-item[0], tuple(sorted(_labels(piece, item[1].dest)))))
        _, edge = cuts[0]
        cut_leaves = _labels(piece, edge.dest)
        _detach(piece, edge.src, edge.dest)
        parts.append(cut_leaves)
    rest = _labels(piece)
    if rest:
        parts.append(rest)
    return parts


def _enumerate_partitions(weights: list[int], max_size: int, min_size: int, cap: int = 20000) -> list[list[list[int]]]:
    """Unique bin partitions in restricted-growth order, each bin in ``[min_size, max_size]``.

    DIMPLE enumerates without a cap; ``cap`` is a safety valve for a blob
    whose items are many and tiny, where the partition count explodes.
    """
    count = len(weights)
    results: list[list[list[int]]] = []
    bins: list[list] = []

    def backtrack(index: int) -> bool:
        """Place item ``index`` in each open bin, then a new one; True once ``cap`` is hit."""
        if len(results) >= cap:
            return True
        if index == count:
            if bins and all(bin_[0] >= min_size for bin_ in bins):
                results.append([list(bin_[1]) for bin_ in bins])
            return False
        weight = weights[index]
        for bin_ in bins:
            if bin_[0] + weight <= max_size:
                bin_[0] += weight
                bin_[1].append(index)
                if backtrack(index + 1):
                    bin_[0] -= weight
                    bin_[1].pop()
                    return True
                bin_[0] -= weight
                bin_[1].pop()
        bins.append([weight, [index]])
        stop = backtrack(index + 1)
        bins.pop()
        return stop

    if count:
        backtrack(0)
    return results


def _random_pack(
    weights: list[int],
    max_size: int,
    min_size: int,
    rng: random.Random,
    items: Optional[Sequence[int]] = None,
) -> list[list[int]]:
    """One seeded first-fit packing, with DIMPLE's repair of an undersized last bin.

    ``items`` (default: every index) are shuffled with ``rng`` and placed
    first-fit. When more than one bin results and the last is under
    ``min_size``, it is merged into the first earlier bin with room. Failing
    that, items move into it from bins that stay at least ``min_size``,
    chosen with ``rng``, until it reaches ``min_size`` or nothing can move.
    This mirrors ``bin_pack_items`` in DIMPLE's divider: ``min_size`` is
    soft, ``max_size`` is hard.
    """
    order = list(range(len(weights)) if items is None else items)
    rng.shuffle(order)
    bins: list[list[int]] = []
    sizes: list[int] = []
    for index in order:
        placed = False
        for bin_index, size in enumerate(sizes):
            if size + weights[index] <= max_size:
                bins[bin_index].append(index)
                sizes[bin_index] += weights[index]
                placed = True
                break
        if not placed:
            bins.append([index])
            sizes.append(weights[index])
    if len(bins) > 1 and sizes[-1] < min_size:
        last_size = sizes[-1]
        for bin_index, size in enumerate(sizes[:-1]):
            if size + last_size <= max_size:
                bins[bin_index].extend(bins[-1])
                sizes[bin_index] += last_size
                bins.pop()
                sizes.pop()
                return bins
        while sizes[-1] < min_size:
            movable = [
                (bin_index, index)
                for bin_index in range(len(bins) - 1)
                if sizes[bin_index] > min_size
                for index in bins[bin_index]
                if sizes[bin_index] - weights[index] >= min_size
                and sizes[-1] + weights[index] <= max_size
            ]
            if not movable:
                break
            bin_index, index = rng.choice(movable)
            bins[bin_index].remove(index)
            sizes[bin_index] -= weights[index]
            bins[-1].append(index)
            sizes[-1] += weights[index]
    return bins


def _smallest_first_pack(weights: list[int], max_size: int, min_size: int, rng: random.Random) -> list[list[int]]:
    """DIMPLE's ``smallest_first`` strategy, used for a blob's first division.

    The lightest items fill one bin while they fit; the rest are first-fit
    packed by :func:`_random_pack`. When the first bin would be under
    ``min_size``, every item is packed by :func:`_random_pack` instead.
    ``rng`` only orders the remaining items.
    """
    order = sorted(range(len(weights)), key=lambda index: weights[index])
    first: list[int] = []
    first_size = 0
    rest: list[int] = []
    for index in order:
        if first_size + weights[index] <= max_size:
            first.append(index)
            first_size += weights[index]
        else:
            rest.append(index)
    if first_size < min_size:
        return _random_pack(weights, max_size, min_size, rng)
    bins = [first] if first else []
    if rest:
        bins.extend(_random_pack(weights, max_size, min_size, rng, items=rest))
    return bins


def _bins_to_division(bins: list[list[int]], items: list[tuple[str, frozenset[str]]]) -> list[frozenset[str]]:
    """Turn item-index bins into taxon sets."""
    division = []
    for bin_ in bins:
        leaves: set[str] = set()
        for index in bin_:
            leaves |= set(items[index][1])
        if leaves:
            division.append(frozenset(leaves))
    division.sort(key=lambda subset: tuple(sorted(subset)))
    return division


def _allocations(
    items: list[tuple[str, frozenset[str]]],
    max_subset_size: int,
    n_divisions: int,
    seed: int,
) -> list[list[frozenset[str]]]:
    """Up to ``n_divisions`` distinct partitions of ``items``. No duplicates.

    Follows DIMPLE's divider (``generate_k_divisions``). When the items
    already fit in one subset, that one subset is the only division.
    Otherwise division 0 is the deterministic smallest-first packing and
    divisions ``1..K-1`` are seeded first-fit random packings (seed
    ``seed + i``), deduplicated. Only when fewer than K distinct packings
    result, and the blob has at most :data:`ENUM_MAX_ITEMS` items, is the
    remainder filled from the enumeration of all partitions with subsets
    in ``[3, S]``, shuffled with ``seed``. If no partition can keep every
    subset at 3 or more, the minimum is relaxed to 1, as DIMPLE does.
    """
    if not items:
        return []
    # Child order in the network is hash order, which is not stable across
    # parses. Sort so the same leaves always get the same item index.
    items = sorted(items, key=lambda item: (tuple(sorted(item[1])), item[0]))
    weights = [len(leaves) for _, leaves in items]
    if sum(weights) <= max_subset_size:
        everything: set[str] = set()
        for _, leaves in items:
            everything |= set(leaves)
        return [[frozenset(everything)]]

    min_size = 3
    chosen: list[list[frozenset[str]]] = []
    seen: set[frozenset[frozenset[str]]] = set()

    def accept(bins: list[list[int]], floor: int) -> None:
        """Keep a packing whose subsets all fall in ``[floor, max_subset_size]`` and is new."""
        division = _bins_to_division(bins, items)
        if any(len(subset) < floor or len(subset) > max_subset_size for subset in division):
            return
        key = frozenset(division)
        if not key or key in seen:
            return
        seen.add(key)
        chosen.append(division)

    # Phase 1: DIMPLE's K seeded packings. Run 0 is smallest-first, so the
    # primary division of a scaffold is deterministic.
    packings: list[list[list[int]]] = []
    for attempt in range(n_divisions):
        rng = random.Random(seed + attempt)
        if attempt == 0:
            packings.append(_smallest_first_pack(weights, max_subset_size, min_size, rng))
        else:
            packings.append(_random_pack(weights, max_subset_size, min_size, rng))
    for bins in packings:
        accept(bins, min_size)

    # Phase 2: fill from the enumeration, falling back to a minimum of 1.
    if len(chosen) < n_divisions and len(items) <= ENUM_MAX_ITEMS:
        floor = min_size
        parts = _enumerate_partitions(weights, max_subset_size, min_size)
        if not parts:
            floor = 1
            parts = _enumerate_partitions(weights, max_subset_size, 1)
        random.Random(seed).shuffle(parts)
        for bins in parts:
            if len(chosen) >= n_divisions:
                break
            accept(bins, floor)

    # Every packing had an undersized bin and enumeration did not run or
    # found nothing: keep the packings anyway, as DIMPLE's soft minimum does.
    if not chosen:
        for bins in packings:
            accept(bins, 1)
    if not chosen:
        raise DimpleError(
            "No partition puts every subset in [1, max_subset_size] "
            f"({max_subset_size})."
        )
    return chosen


def _items_of(
    net: Network,
    members: set[Node],
    other_blob_labels: set[str],
    max_subset_size: int,
) -> list[tuple[str, frozenset[str]]]:
    """Allocation items: each non-member child clade of a member blob.

    Leaves that belong to a nested blob outside this mega-blob are subtracted.
    A clade still larger than S is split.
    """
    member_labels = {node.label for node in members}
    present = [node for node in members if node in net.V()]
    if not present:
        return []
    root = _top_member(net, present)
    items: list[tuple[str, frozenset[str]]] = []

    def walk(node: Node) -> None:
        """Descend through this job's polytomies; every other clade is an item."""
        if node.label in member_labels:
            for child in net.get_children(node):
                walk(child)
            return
        if node.label in other_blob_labels:
            return
        # Another mega-blob can sit inside this child. Its leaves are
        # allocated on that blob's own job, so they are not an item here.
        # Membership is by label over every vertex: ``members`` is only
        # this job's polytomies.
        leaves = set(_labels(net, node))
        for other in net.V():
            if other.label not in other_blob_labels:
                continue
            if other is node:
                leaves.clear()
                break
            if other in net.get_subtree_at(node):
                leaves -= set(_labels(net, other))
        if not leaves:
            return
        for index, part in enumerate(_split_clade(net, frozenset(leaves), max_subset_size)):
            name = node.label if index == 0 else f"{node.label}:{index}"
            items.append((name, part))

    walk(root)
    return items


def _classify_prunes(pruned: list[PrunedSubtree], residual: frozenset[str]) -> None:
    """Mark a prune ``prune`` when its anchor has already left the residual blob.

    ``source_type`` defaults to ``blob_item``. An empty anchor stays that
    way: there is no earlier prune to wait for.

    Args:
        pruned (list[PrunedSubtree]): Cuts just made, mutated in place.
        residual (frozenset[str]): Leaves still in the blob after the cuts.
    """
    for prune in pruned:
        if prune.anchor_leaves and not (prune.anchor_leaves & residual):
            prune.source_type = "prune"


def partition_scaffold(
    net: Network,
    *,
    max_subset_size: int = 12,
    n_divisions: int = 15,
    seed: int = 0,
    outgroup: Optional[str] = None,
) -> list[BlobJob]:
    """Cut a tree of blobs into subset taxon sets.

    Forbidden edges are the original out-edges of each polytomy. The prune
    removes the largest legal clade (both sides at least 3 leaves, the cut
    itself at most S) until every blob's children sum to at most S. Blobs
    joined by a direct edge are one mega-blob and are packed together.
    ``seed`` seeds the random packings and the enumeration shuffle; the
    first division of every blob is deterministic.

    Args:
        net (Network): Rooted tree of blobs.
        max_subset_size (int): S, at least 4.
        n_divisions (int): K, at least 1.
        seed (int): Seed for the partition shuffle.
        outgroup (str | None): Scaffold leaf to drop before cutting.

    Returns:
        list[BlobJob]: One job per mega-blob, sorted by name.

    Raises:
        DimpleError: If S or K is illegal, the scaffold is reticulate, or a
            clade cannot be split under the size rules.
    """
    if max_subset_size < 4:
        raise DimpleError("max_subset_size must be at least 4.")
    if n_divisions < 1:
        raise DimpleError("n_divisions must be at least 1.")
    for node in net.V():
        if node.is_reticulation() or net.in_degree(node) > 1:
            raise DimpleError(
                "The tree of blobs contains a reticulation. A blob is a "
                "polytomy on a tree."
            )

    work = strip_outgroup(net, outgroup)
    _ensure_internal_labels(work)
    blobs = _blobs(work)
    if not blobs:
        leaves = _labels(work)
        if len(leaves) > max_subset_size:
            raise DimpleError(
                "The scaffold has no polytomy and more leaves than "
                "max_subset_size."
            )
        return [BlobJob(
            name="tree",
            tob_leaves=leaves,
            external_siblings=frozenset(),
            divisions=[[leaves]] if len(leaves) >= 3 else [],
            is_root=True,
        )]

    forbidden = _forbidden(work, blobs)
    groups = _mega_groups(work, blobs)
    tops = [_top_member(work, group) for group in groups]
    jobs: list[BlobJob] = []

    for group, top in zip(groups, tops):
        member_labels = {node.label for node in group}
        other_labels = {node.label for node in blobs if node.label not in member_labels}
        full_leaves = _labels(work, top)
        parents = work.get_parents(top)
        if parents:
            siblings = _labels(work, parents[0]) - full_leaves
            is_root = False
        else:
            siblings = frozenset()
            is_root = True

        # Nested mega-blobs are allocated on their own. Take them out of this
        # piece before pruning so a cut cannot swallow them.
        piece = induced_subnetwork_by_taxa(work, sorted(full_leaves))
        _ensure_internal_labels(piece)
        nested = [
            node for node in piece.V()
            if node.label in other_labels and piece.out_degree(node) > 2
        ]
        for nested_blob in list(nested):
            if nested_blob not in piece.V():
                continue
            nested_parents = piece.get_parents(nested_blob)
            if not nested_parents:
                continue
            _detach(piece, nested_parents[0], nested_blob)

        piece_blobs = [node for node in piece.V() if node.label in member_labels and piece.out_degree(node) > 2]
        pruned = _prune_piece(piece, piece_blobs, forbidden, max_subset_size)
        residual = _labels(piece)
        _classify_prunes(pruned, residual)
        items = _items_of(piece, set(piece_blobs), other_labels, max_subset_size)
        if not items and residual:
            items = [("residual", residual)]
        divisions = _allocations(items, max_subset_size, n_divisions, seed)
        jobs.append(BlobJob(
            name="+".join(sorted(member_labels)),
            tob_leaves=full_leaves,
            external_siblings=siblings,
            divisions=divisions,
            pruned=pruned,
            is_root=is_root,
        ))
    # Blob discovery follows vertex iteration order, which is not stable
    # across processes. DIMPLE indexes blobs by sorted mega-blob name.
    jobs.sort(key=lambda job: job.name)
    return jobs


# ── Subset restriction and inference ──────────────────────────────────


def restrict_gene_trees(gene_trees: GeneTrees, taxa: Sequence[str]) -> GeneTrees:
    """Gene trees induced on ``taxa``.

    A tree that shares fewer than three labels with ``taxa`` is omitted: it
    has no rooted triple on the subset. A tree whose leaves are exactly
    ``taxa`` is kept as the same object, not copied: every consumer only
    reads gene trees. The mapping is the parent mapping restricted to
    alleles that remain.

    Args:
        gene_trees (GeneTrees): Full gene trees. Individual trees may lack
            taxa.
        taxa (Sequence[str]): Subset labels.

    Returns:
        GeneTrees: The restricted collection.

    Raises:
        DimpleError: If every gene tree misses the subset.
    """
    wanted = set(taxa)
    kept: list[Network] = []
    for tree in gene_trees.trees:
        present = [leaf.label for leaf in tree.get_leaves() if leaf.label in wanted]
        if len(present) < 3:
            continue
        if set(present) == {leaf.label for leaf in tree.get_leaves()}:
            kept.append(tree)
        else:
            kept.append(induced_subnetwork_by_taxa(tree, present))
    if not kept:
        raise DimpleError(
            "No gene tree has three taxa in common with the subset "
            f"{sorted(wanted)}."
        )
    remaining = set()
    for tree in kept:
        remaining.update(leaf.label for leaf in tree.get_leaves())
    parent = gene_trees.resolved_mapping()
    mapping = {
        species: [allele for allele in alleles if allele in remaining]
        for species, alleles in parent.items()
    }
    mapping = {species: alleles for species, alleles in mapping.items() if alleles}
    if not mapping:
        mapping = {label: [label] for label in sorted(remaining)}
    return GeneTrees(trees=kept, mapping=mapping)


def restrict_start(start_tree: Network, taxa: Sequence[str]) -> Network:
    """Binary start tree induced on ``taxa``.

    Args:
        start_tree (Network): Binary tree that contains every label in ``taxa``.
        taxa (Sequence[str]): Subset labels.

    Returns:
        Network: The induced tree, with coalescent branch-length units set
        and missing lengths filled in.

    Raises:
        DimpleError: If a label is missing, or the induced tree is not binary.
    """
    missing = [label for label in taxa if start_tree.has_node_named(label) is None]
    if missing:
        raise DimpleError(
            "The start tree is missing subset taxa: "
            + ", ".join(sorted(missing)[:5]) + "."
        )
    induced = induced_subnetwork_by_taxa(start_tree, list(taxa))
    if not _is_binary_tree(induced):
        raise DimpleError("The start tree restricted to the subset is not binary.")
    induced.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
    _populate_default_branch_lengths(induced)
    return induced


def true_reticulation_bound(true_network: Network, taxa: Sequence[str]) -> int:
    """Reticulations of the true network restricted to ``taxa`` that triplets can see.

    This is the protocol of the DIMPLE paper: each subset search is given
    the reticulation count of the ground-truth subnetwork on its taxa as
    its bound. Restriction can collapse a reticulation's cycle to three
    nodes (its two parents become parent and grandparent); no rooted
    triple distinguishes that from a tree, so such reticulations are not
    counted (:func:`phynetpy.GraphUtils.count_reticulations` with
    ``min_cycle_size=4``).

    Args:
        true_network (Network): Ground-truth network on the full taxon set.
        taxa (Sequence[str]): Subset labels.

    Returns:
        int: Detectable reticulations in the induced true subnetwork.

    Raises:
        DimpleError: If a label is not in the true network.
    """
    missing = [label for label in taxa if true_network.has_node_named(label) is None]
    if missing:
        raise DimpleError(
            "The true network lacks subset taxa: " + ", ".join(sorted(missing)[:5]) + "."
        )
    induced = induced_subnetwork_by_taxa(true_network, sorted(taxa))
    return count_reticulations(induced, min_cycle_size=4)


def _selection_data_size(subset_trees: GeneTrees) -> int:
    """Sample size ``n`` for the BIC rule: the number of gene trees.

    This is the module convention of :mod:`phynetpy.ModelSelection`. The
    log pseudo-likelihood is a sum over triplet counts, so ``n`` is a
    convention rather than a derivation; it lives here so one line can
    change it.
    """
    return len(subset_trees.trees)


def selection_min_gain(selection: Optional[str], data_size: int) -> float:
    """Log pseudo-likelihood a reticulation must add to earn its parameters.

    ``gain``: :data:`SELECTION_GAIN_THRESHOLD`. ``aic``:
    :data:`SELECTION_PARAMS_PER_RETICULATION`. ``bic``: half the
    parameters times ``ln(data_size)``. ``elbow`` and None: 0, which is
    DIMPLE's rule (any strict improvement).

    Args:
        selection (str | None): The model-selection rule.
        data_size (int): ``n`` for BIC.

    Returns:
        float: The minimum gain.
    """
    if selection == "gain":
        return float(SELECTION_GAIN_THRESHOLD)
    if selection == "aic":
        return float(SELECTION_PARAMS_PER_RETICULATION)
    if selection == "bic":
        return 0.5 * SELECTION_PARAMS_PER_RETICULATION * math.log(max(data_size, 1))
    return 0.0


def subset_chains(search: dict) -> list[dict]:
    """Expand subset search settings into one complete ``infer`` kwargs dict per chain.

    ``search`` is what :func:`infer_dimple` forwards. Two forms are read:

    - flat: ``method`` (default ``"sa"``), ``num_iter`` (default
      :data:`SUBSET_ITERS`), ``n_runs`` (default :data:`SUBSET_RUNS`),
      ``seed`` (default 0) and any other ``infer`` keyword. The chains
      differ only in seed: ``seed``, ``seed + 1``, ...
    - ``chains``: a list of dicts, each with its own ``method``,
      ``num_iter`` and ``n_runs`` (and any ``infer`` keyword), run as a
      portfolio and the best score kept. The flat keys of ``search`` are
      the defaults every entry starts from. Seeds continue across entries,
      so no two chains of one subset share a seed.

    Annealing chains get :data:`_SUBSET_ANNEALING` for the keys they do
    not set; hill-climbing chains get :data:`SUBSET_PHASE_PATIENCE`. Each
    dict also carries ``chain`` (its index); callers pop it before
    calling ``infer``.

    Args:
        search (dict): Search settings as described.

    Returns:
        list[dict]: One kwargs dict per chain, in seed order.

    Raises:
        DimpleError: If ``chains`` is empty or an entry asks for no runs.
    """
    shared = dict(search)
    entries = shared.pop("chains", None)
    if entries is None:
        entries = [{}]
    elif not entries:
        raise DimpleError("chains must list at least one search configuration.")
    base_seed = int(shared.pop("seed", 0))
    shared_runs = shared.pop("n_runs", SUBSET_RUNS)
    out: list[dict] = []
    for entry in entries:
        settings = dict(shared)
        settings.update(entry)
        n_runs = int(settings.pop("n_runs", shared_runs))
        if n_runs < 1:
            raise DimpleError("each chain configuration needs n_runs >= 1.")
        settings.setdefault("method", "sa")
        settings.setdefault("num_iter", SUBSET_ITERS)
        if settings["method"] == "sa":
            for key, value in _SUBSET_ANNEALING.items():
                settings.setdefault(key, value)
        else:
            # In-loop gamma fitting is hill-climbing only. The kernel default
            # of 150 non-improvements before the reticulation phase was
            # chosen for a long annealing run.
            settings.setdefault("phase_patience", SUBSET_PHASE_PATIENCE)
        for _ in range(n_runs):
            chain = dict(settings)
            chain["seed"] = base_seed + len(out)
            chain["chain"] = len(out)
            out.append(chain)
    return out


def run_subset_chain(
    subset_trees: GeneTrees,
    subset_start: Network,
    max_reticulations: int,
    chain: dict,
):
    """Run one search chain on an already-restricted subset.

    Args:
        subset_trees (GeneTrees): Gene trees restricted to the subset.
        subset_start (Network): Start network on the subset's taxa.
        max_reticulations (int): Reticulation cap for this chain.
        chain (dict): One entry of :func:`subset_chains`.

    Returns:
        InferenceResult: The chain's best network and search score.
    """
    from .infer import Start, infer

    settings = dict(chain)
    settings.pop("chain", None)
    if settings.get("method") == "sa":
        settings.pop("phase_patience", None)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="optimize_params is only supported for method='hc'",
        )
        return infer(
            subset_trees,
            model=MSC(),
            criterion=PseudoLikelihood(),
            start=Start(subset_start),
            max_reticulations=max_reticulations,
            **settings,
        )


def finish_subset(
    subset_trees: GeneTrees,
    taxa: frozenset[str],
    network: Network,
    max_reticulations: int,
    *,
    k_selected: Optional[int] = None,
    selection: Optional[str] = None,
    sweep: Optional[list[dict]] = None,
    chains: Optional[list[dict]] = None,
) -> SubsetRecord:
    """Re-score a chosen subset network and file it as a :class:`SubsetRecord`.

    The score is a fresh :func:`phynetpy.infer.score` of ``network`` on
    ``subset_trees``, so the trace does not depend on a search statistic.
    ``blob_name`` is left empty for the caller to fill.

    Args:
        subset_trees (GeneTrees): Gene trees restricted to the subset.
        taxa (frozenset[str]): The subset's taxa.
        network (Network): The network chosen for the subset.
        max_reticulations (int): The cap the search ran under.
        k_selected (int | None): Reticulation count model selection chose;
            defaults to the cap.
        selection (str | None): The selection rule used, if any.
        sweep (list[dict] | None): Per-cap rows of the selection sweep.
        chains (list[dict] | None): Per-chain summaries from
            :func:`_chain_summary`.

    Returns:
        SubsetRecord: The subset's network, score, and search trace.
    """
    from .infer import score

    value = float(score(
        network, subset_trees, model=MSC(), criterion=PseudoLikelihood(),
    ))
    return SubsetRecord(
        "", taxa, network, value, bound=max_reticulations,
        k_selected=max_reticulations if k_selected is None else k_selected,
        selection=selection, sweep=sweep, chains=chains,
    )


def _chain_summary(chain: dict, result, seconds: float, chosen: bool) -> dict:
    """One trace row for a subset chain: its settings, score, time, and whether it won."""
    return {
        "chain": chain["chain"], "method": chain["method"],
        "num_iter": int(chain["num_iter"]), "seed": int(chain["seed"]),
        "score": float(result.score), "seconds": float(seconds), "chosen": chosen,
    }


def _infer_subset(
    gene_trees: GeneTrees,
    start_tree: Network,
    taxa: frozenset[str],
    max_reticulations: int,
    search: dict,
    *,
    model_selection: Optional[str] = None,
) -> SubsetRecord:
    """Infer one subset by a portfolio of search chains and keep the best.

    The chains come from :func:`subset_chains`: by default
    :data:`SUBSET_RUNS` annealing runs of :data:`SUBSET_ITERS` iterations
    that differ only in seed, or an explicit ``chains`` portfolio (for
    instance annealing runs plus hill-climbing runs). Without model
    selection every chain searches at the cap ``max_reticulations`` and
    the network with the higher log pseudo-likelihood is kept. With
    ``model_selection`` the same chains are run for every cap from 0 to
    ``max_reticulations`` (one more for the elbow, which needs two steps
    to see a bend) through :func:`phynetpy.ModelSelection.reticulation_sweep`,
    and the cap is chosen by the rule. Each cap above 0 starts its chains
    from the best network of the cap below (a free start, the search may
    still reshape it), so a step measures what one more reticulation adds
    rather than the luck of an independent search. ``gain`` keeps raising
    the cap while a step adds more than :data:`SELECTION_GAIN_THRESHOLD`.
    The elbow's extra cap is never selected: a choice above
    ``max_reticulations`` is clamped. The score returned is a fresh
    :func:`phynetpy.infer.score` of the chosen network, so the trace does
    not depend on a search statistic.

    Args:
        gene_trees (GeneTrees): Full-taxon gene trees. Each tree may omit taxa.
        start_tree (Network): Binary start tree on the full taxon set.
        taxa (frozenset[str]): Labels in this subset.
        max_reticulations (int): Reticulation cap (the true count when a
            ground-truth network drives the run).
        search (dict): Search settings, see :func:`subset_chains`.
        model_selection (str | None): ``gain``, ``aic``, ``bic``,
            ``elbow``, or None.

    Returns:
        SubsetRecord: The chosen network, its score, the sweep and the
        chains, with an empty ``blob_name`` for the caller to fill.

    Raises:
        DimpleError: If the subset gene trees or the restricted start tree
            are illegal, or the rule is unknown.
    """
    if model_selection is not None and model_selection not in SELECTION_RULES:
        raise DimpleError(
            f"model_selection must be one of {SELECTION_RULES} or None, "
            f"not {model_selection!r}."
        )
    subset_trees = restrict_gene_trees(gene_trees, sorted(taxa))
    subset_start = restrict_start(start_tree, sorted(taxa))
    chains = subset_chains(search)

    found: dict[tuple[int, int], object] = {}
    timings: dict[tuple[int, int], float] = {}

    def best_key(cap: int) -> tuple[int, int]:
        """``(cap, chain)`` of the highest-scoring finished chain at ``cap``."""
        keys = [(cap, chain["chain"]) for chain in chains if (cap, chain["chain"]) in found]
        return max(keys, key=lambda key: found[key].score)

    def run_one(cap: int, chain: dict, warm: bool):
        """Run ``chain`` at ``cap``, warm-started from the cap below when asked, and file it."""
        start_net = subset_start
        if warm and cap > 0 and any((cap - 1, c["chain"]) in found for c in chains):
            # Sweep step: continue from the best network one cap below. The
            # engine copies the start, so the stored result is untouched.
            start_net = found[best_key(cap - 1)].best
            start_net.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
        started = time.perf_counter()
        result = run_subset_chain(subset_trees, start_net, cap, chain)
        found[(cap, chain["chain"])] = result
        timings[(cap, chain["chain"])] = time.perf_counter() - started
        return result

    if model_selection is None:
        for chain in chains:
            run_one(max_reticulations, chain, warm=False)
        chosen_k = max_reticulations
        sweep = None
    else:
        caps = list(range(0, max_reticulations + 1))
        if model_selection == "elbow":
            caps.append(max_reticulations + 1)
        by_index = {chain["chain"]: chain for chain in chains}

        def search_fn(cap: int, index: int) -> float:
            """Score of chain ``index`` at ``cap``, in the form ``reticulation_sweep`` calls."""
            return float(run_one(cap, by_index[index], warm=True).score)

        swept = reticulation_sweep(
            search_fn, caps, seeds=[chain["chain"] for chain in chains],
            data_size=_selection_data_size(subset_trees),
            params_per_reticulation=SELECTION_PARAMS_PER_RETICULATION,
            log_lik_label="log-pseudo-likelihood", progress=False,
        )
        if model_selection == "gain":
            chosen_k = swept.rows[0].k
            for row in swept.rows[1:]:
                if row.delta_log_lik is not None and row.delta_log_lik > SELECTION_GAIN_THRESHOLD:
                    chosen_k = row.k
                else:
                    break
        else:
            chosen_k = int(swept.best_by(model_selection))
        chosen_k = min(chosen_k, max_reticulations)
        sweep = [
            {
                "k": row.k,
                "best_log_lik": float(row.best_log_lik),
                "n_params": int(row.n_params),
                "aic": float(row.aic),
                "bic": float(row.bic),
                "delta_log_lik": None if row.delta_log_lik is None else float(row.delta_log_lik),
                "n_retics": count_reticulations(found[best_key(row.k)].best),
            }
            for row in swept.rows
        ]
    winner = best_key(chosen_k)
    summaries = [
        _chain_summary(chain, found[(chosen_k, chain["chain"])],
                       timings[(chosen_k, chain["chain"])], (chosen_k, chain["chain"]) == winner)
        for chain in chains if (chosen_k, chain["chain"]) in found
    ]
    return finish_subset(
        subset_trees, taxa, found[winner].best, max_reticulations,
        k_selected=chosen_k, selection=model_selection, sweep=sweep, chains=summaries,
    )


# ── Displayed trees ───────────────────────────────────────────────────


def _miss_fraction(
    tree: Network,
    keep: set[str],
    observed: dict[tuple[str, str, str], int],
) -> float:
    """Fraction of observed rooted triples on ``keep`` that ``tree`` does not display.

    Lower is better. Fewer than three shared taxa is a total miss. This is
    not an MSC pseudo-likelihood.
    """
    present = {leaf.label for leaf in tree.get_leaves()}
    use = present & keep
    if len(use) < 3:
        return 1.0
    viewed = tree if use == present else induced_subnetwork_by_taxa(tree, sorted(use))
    displayed = rooted_triples(viewed)
    total = 0
    matched = 0
    for (a, b, c), count in observed.items():
        if a not in use or b not in use or c not in use:
            continue
        total += count
        x, y = (a, b) if a < b else (b, a)
        if (x, y, c) in displayed:
            matched += count
    if total == 0:
        return 1.0
    return 1.0 - (matched / total)



def _displayed_tree_choices(net: Network) -> list[dict[Node, Edge]]:
    """The in-edge each tree of :func:`get_all_subtrees` keeps per reticulation, in its order."""
    retics = sorted((node for node in net.V() if node.is_reticulation()), key=lambda n: n.label)
    options = [
        sorted(net.in_edges(node), key=lambda edge: (-(edge.get_gamma() or 0.0), edge.src.label))
        for node in retics
    ]
    return [dict(zip(retics, combo)) for combo in product(*options)]


def select_displayed_trees_with_choices(
    networks: Sequence[Network],
    observed: Optional[dict[tuple[str, str, str], int]] = None,
) -> tuple[list[Network], list[Optional[dict[Node, Edge]]]]:
    """:func:`select_displayed_trees`, also returning which hybrid edge each tree kept.

    The second list has one entry per network: the kept in-edge per
    reticulation of the chosen tree, or None for a tree (nothing to choose).
    :func:`reticulation_records` reads a network's leaf sets on that tree,
    so the sides it records are clades of the tree the backbone was built
    from.

    Args:
        networks (Sequence[Network]): Inferred subnetworks, some of them trees.
        observed (dict | None): Counts of rooted triples ``(a, b, c)`` with
            ``a < b``. Required only when some network has two or more
            displayed trees.

    Returns:
        tuple[list[Network], list[dict[Node, Edge] | None]]: The chosen tree
        per input, and the hybrid in-edge it kept per reticulation.

    Raises:
        DimpleError: If a network has several displayed trees and
            ``observed`` is None.
    """
    leaf_sets = [{leaf.label for leaf in net.get_leaves()} for net in networks]
    chosen: list[Network] = []
    choices: list[Optional[dict[Node, Edge]]] = []
    for index, net in enumerate(networks):
        if is_tree(net):
            chosen.append(net)
            choices.append(None)
            continue
        candidates = get_all_subtrees(net)
        options = _displayed_tree_choices(net)
        if len(options) != len(candidates):
            # Not the enumeration this module mirrors; fall back to the major tree.
            options = [None] * len(candidates)
        if len(candidates) <= 1:
            chosen.append(candidates[0] if candidates else net)
            choices.append(options[0] if options else None)
            continue
        if observed is None:
            raise DimpleError(
                "Displayed-tree selection needs rooted-triple counts."
            )
        others: set[str] = set()
        for other_index, leaves in enumerate(leaf_sets):
            if other_index != index:
                others |= leaves
        shared = leaf_sets[index] & others
        best_index = 0
        best_score = _miss_fraction(candidates[0], shared, observed)
        for position, candidate in enumerate(candidates[1:], start=1):
            score = _miss_fraction(candidate, shared, observed)
            # Strict comparison: an equal score keeps the earlier tree.
            if score < best_score:
                best_index = position
                best_score = score
        chosen.append(candidates[best_index])
        choices.append(options[best_index])
    return chosen, choices


def select_displayed_trees(
    networks: Sequence[Network],
    observed: Optional[dict[tuple[str, str, str], int]] = None,
) -> list[Network]:
    """Replace each network with the displayed tree that misses the fewest triples.

    A tree is returned as itself. A network with one displayed tree is
    returned without consulting ``observed``. Ties keep the earlier tree
    from :func:`phynetpy.GraphUtils.get_all_subtrees`.

    Args:
        networks (Sequence[Network]): Inferred subnetworks, some of them trees.
        observed (dict | None): Counts of rooted triples ``(a, b, c)`` with
            ``a < b``. Required only when some network has two or more
            displayed trees.

    Returns:
        list[Network]: One tree per input, same order.
    """
    return select_displayed_trees_with_choices(networks, observed)[0]


def observed_triples(gene_trees: GeneTrees, taxa: Sequence[str]) -> dict[tuple[str, str, str], int]:
    """Count rooted triples of the gene trees on ``taxa``.

    Args:
        gene_trees (GeneTrees): Source trees.
        taxa (Sequence[str]): Labels that may enter a triple.

    Returns:
        dict: ``(a, b, c)`` with ``a < b`` mapped to how many gene trees
        display it.
    """
    wanted = set(taxa)
    counts: Counter[tuple[str, str, str]] = Counter()
    for tree in gene_trees.trees:
        present = {leaf.label for leaf in tree.get_leaves()} & wanted
        if len(present) < 3:
            continue
        if present != {leaf.label for leaf in tree.get_leaves()}:
            tree = induced_subnetwork_by_taxa(tree, sorted(present))
        for triple in rooted_triples(tree):
            if triple[0] in wanted and triple[1] in wanted and triple[2] in wanted:
                counts[triple] += 1
    return dict(counts)


# ── Compatibility and neighbor-joining ────────────────────────────────


def _clade_sets(net: Network) -> list[frozenset[str]]:
    """Leaf set of every node of ``net``, computed once per tree."""
    return list(_label_map(net).values())


def _splits(clades: Sequence[frozenset[str]], keep: frozenset[str]) -> list[frozenset[str]]:
    """Nontrivial clades among ``clades`` after restriction to ``keep``."""
    found = []
    for clade in clades:
        below = clade & keep
        if 2 <= len(below) <= len(keep) - 2:
            found.append(below)
    return found


def trees_incompatible(
    left: Network,
    right: Network,
    left_clades: Optional[Sequence[frozenset[str]]] = None,
    right_clades: Optional[Sequence[frozenset[str]]] = None,
) -> bool:
    """Whether two trees conflict on their shared taxa.

    Fewer than four shared taxa cannot conflict. Otherwise some clade of
    one tree crosses a clade of the other (all four intersections nonempty).

    Args:
        left (Network): A tree.
        right (Network): A tree.
        left_clades, right_clades: Each tree's clade leaf sets from
            :func:`_clade_sets`, so a caller comparing many pairs computes
            them once per tree. Computed here when omitted.

    Returns:
        bool: True when the unrooted shapes disagree on the shared labels.
    """
    if left_clades is None:
        left_clades = _clade_sets(left)
    if right_clades is None:
        right_clades = _clade_sets(right)
    shared = _labels(left) & _labels(right)
    if len(shared) < 4:
        return False
    for left_split in _splits(left_clades, shared):
        left_other = shared - left_split
        for right_split in _splits(right_clades, shared):
            right_other = shared - right_split
            if (
                left_split & right_split
                and left_split & right_other
                and left_other & right_split
                and left_other & right_other
            ):
                return True
    return False


def greedy_compatible(trees: Sequence[Network]) -> list[Network]:
    """First-fit subset of pairwise compatible trees.

    Args:
        trees (Sequence[Network]): Candidate trees, in the order they should
            be considered.

    Returns:
        list[Network]: The trees that were kept.
    """
    clades = [_clade_sets(tree) for tree in trees]
    kept: list[int] = []
    for index, tree in enumerate(trees):
        if all(
            not trees_incompatible(tree, trees[earlier], clades[index], clades[earlier])
            for earlier in kept
        ):
            kept.append(index)
    return [trees[index] for index in kept]


def max_clique_compatible(trees: Sequence[Network]) -> list[Network]:
    """Largest set of pairwise compatible trees, in input order.

    DIMPLE's default compatibility filter (``select_compatible_subset`` with
    strategy ``maxclique``). Two trees are joined in the compatibility graph
    when :func:`trees_incompatible` is false; the largest clique is kept.
    Among equally large cliques the first one networkx reports wins.
    :func:`greedy_compatible` is the first-fit alternative.

    Args:
        trees (Sequence[Network]): Candidate trees.

    Returns:
        list[Network]: The trees of the largest clique, in input order.
    """
    if len(trees) <= 1:
        return list(trees)
    clades = [_clade_sets(tree) for tree in trees]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(trees)))
    for left in range(len(trees)):
        for right in range(left + 1, len(trees)):
            if not trees_incompatible(trees[left], trees[right], clades[left], clades[right]):
                graph.add_edge(left, right)
    best = max(nx.find_cliques(graph), key=len, default=[])
    return [trees[index] for index in sorted(best)]


class _Constraint:
    """Undirected, degree-2-free tree of current cluster labels.

    A cherry is two labels that share a neighbour. Neighbor-joining may join
    two labels that both occur in this tree only when they are a cherry.
    """

    def __init__(self, net: Network) -> None:
        """Build the constraint from a subset tree.

        Args:
            net (Network): A tree whose leaves are the cluster labels.
        """
        self.leaves: set[str] = set()
        self.adj: dict[str, set[str]] = {}
        self.broken: int = 0
        self._build(net)

    def _build(self, net: Network) -> None:
        """Copy ``net`` as an undirected graph and suppress every non-leaf of degree <= 2."""
        names: dict[Node, str] = {}
        for node in net.V():
            if net.out_degree(node) == 0:
                names[node] = node.label
                self.leaves.add(node.label)
            else:
                names[node] = f"u{id(node)}"
        adj: dict[str, set[str]] = {name: set() for name in names.values()}
        for edge in net.E():
            adj[names[edge.src]].add(names[edge.dest])
            adj[names[edge.dest]].add(names[edge.src])
        # Suppress non-leaves of degree <= 2 so cherries do not depend on
        # where the tree was rooted. Restart after each deletion: the
        # neighbours' degrees change, and ``_suppress`` is the same edit.
        self.adj = adj
        changed = True
        while changed:
            changed = False
            for node in list(self.adj):
                if node in self.leaves or len(self.adj[node]) > 2:
                    continue
                self._suppress(node)
                changed = True
                break

    def cherries(self) -> set[frozenset[str]]:
        """Pairs of labels that currently share a neighbour.

        Returns:
            set[frozenset[str]]: Every unordered pair of leaves that is a cherry.
        """
        pairs: set[frozenset[str]] = set()
        if len(self.leaves) == 2:
            pairs.add(frozenset(self.leaves))
        for node, neighbours in self.adj.items():
            leaf_neighbours = [item for item in neighbours if item in self.leaves]
            if node in self.leaves:
                for leaf in leaf_neighbours:
                    pairs.add(frozenset((node, leaf)))
            else:
                for i in range(len(leaf_neighbours)):
                    for j in range(i + 1, len(leaf_neighbours)):
                        pairs.add(frozenset((leaf_neighbours[i], leaf_neighbours[j])))
        return pairs

    def _suppress(self, node: str) -> None:
        """Splice out a non-leaf of degree 2, or delete one of degree <= 1."""
        neighbours = self.adj.get(node)
        if neighbours is None or node in self.leaves:
            return
        if len(neighbours) == 2:
            left, right = tuple(neighbours)
            self.adj[left].discard(node)
            self.adj[right].discard(node)
            self.adj[left].add(right)
            self.adj[right].add(left)
            del self.adj[node]
        elif len(neighbours) <= 1:
            for neighbour in list(neighbours):
                self.adj[neighbour].discard(node)
            del self.adj[node]

    def rename(self, old: str, new: str) -> None:
        """Replace a leaf label. Used when only one side of a join is in this tree.

        Args:
            old (str): The leaf to relabel; ignored when absent.
            new (str): Its new label.
        """
        if old not in self.leaves:
            return
        neighbours = self.adj.pop(old)
        self.adj[new] = set(neighbours)
        for neighbour in neighbours:
            self.adj[neighbour].discard(old)
            self.adj[neighbour].add(new)
        self.leaves.discard(old)
        self.leaves.add(new)

    def remove_leaf(self, label: str) -> None:
        """Delete a leaf and suppress the internal node it leaves behind.

        Args:
            label (str): Leaf to drop. A label that is not a leaf is ignored.
        """
        if label not in self.leaves:
            return
        neighbours = self.adj.pop(label)
        self.leaves.discard(label)
        for neighbour in neighbours:
            self.adj[neighbour].discard(label)
        # Sorted so suppression does not follow set order.
        for neighbour in sorted(neighbours):
            self._suppress(neighbour)

    def contract(self, left: str, right: str, new: str) -> bool:
        """Replace a cherry with ``new``. Return False when the pair is not a cherry.

        Args:
            left (str): One leaf of the proposed cherry.
            right (str): The other leaf.
            new (str): Label that replaces the cherry.

        Returns:
            bool: True when the pair was a cherry and was replaced.
        """
        shared = self.adj.get(left, set()) & self.adj.get(right, set())
        if shared:
            parent = next(iter(shared))
            self.adj[parent].discard(left)
            self.adj[parent].discard(right)
            del self.adj[left]
            del self.adj[right]
            self.leaves.discard(left)
            self.leaves.discard(right)
            self.adj[new] = {parent}
            self.adj[parent].add(new)
            self.leaves.add(new)
            self._suppress(parent)
            return True
        # The two labels became directly adjacent (a two-leaf tree).
        if right in self.adj.get(left, set()):
            del self.adj[left]
            del self.adj[right]
            self.leaves.discard(left)
            self.leaves.discard(right)
            self.adj[new] = set()
            self.leaves.add(new)
            return True
        return False

    def join(self, left: str, right: str, new: str) -> None:
        """Apply a neighbor-joining cluster merge to this constraint.

        A pair that is in this tree but is not a cherry is a relaxed join.
        Both leaves are deleted and ``new`` is not added: that cluster is
        unconstrained here, and the other leaves keep their splits.

        Args:
            left (str): One cluster label.
            right (str): The other cluster label.
            new (str): Label of the joined cluster.
        """
        has_left = left in self.leaves
        has_right = right in self.leaves
        if has_left and has_right:
            if not self.contract(left, right, new):
                self.remove_leaf(left)
                self.remove_leaf(right)
                self.broken += 1
        elif has_left:
            self.rename(left, new)
        elif has_right:
            self.rename(right, new)


def _leaf_network(label: str) -> Network:
    """A one-node network whose only node is the leaf ``label``."""
    net = Network()
    net.add_nodes(Node(label))
    return net


def _join_clusters(left: Network, right: Network) -> Network:
    """A new root with ``left`` and ``right`` as its two clades."""
    joined = Network()
    root = Node(joined._fresh_label("J"))
    joined.add_nodes(root)
    joined.graft_subtree(left, under=root)
    joined.graft_subtree(right, under=root)
    return joined


def _distance(dist: dict[tuple[str, str], float], left: str, right: str) -> float:
    """Pairwise distance. The matrix stores no diagonal, so a label against itself is 0."""
    if left == right:
        return 0.0
    return dist[(left, right)]


def _reverse_upward(net: Network, start: Node, ignore: Optional[Node] = None) -> None:
    """Reverse edges from ``start`` up to the root, skipping ``ignore``.

    After a new root is inserted, the path from the old root down to
    ``start`` has to point the other way, or the new root is not the only
    in-degree-0 node.

    Args:
        net (Network): Tree mutated in place.
        start (Node): Lowest node of the path that is reversed.
        ignore (Node | None): Parent to skip. The new root, when the edge
            into ``start`` was just replaced.
    """
    # Record the path before any edge is flipped. Reversing while walking
    # would follow the edge that was just turned around.
    chain: list[tuple[Node, Node]] = []
    current = start
    seen: set[Node] = set()
    while current not in seen:
        seen.add(current)
        parents = [parent for parent in net.get_parents(current) if parent is not ignore]
        if not parents:
            break
        parent = parents[0]
        chain.append((parent, current))
        current = parent
    for parent, child in chain:
        edge = net.get_edge(parent, child)
        length = edge.get_length()
        net.remove_edge(edge)
        net.add_edges(Edge(child, parent, length=length))


def _reroot_on_edge(net: Network, parent: Node, child: Node) -> None:
    """Subdivide ``parent`` to ``child`` with a new root and reverse the path above it.

    Args:
        net (Network): Tree to reroot in place.
        parent (Node): Source of the edge that becomes the root edge.
        child (Node): Destination of that edge.
    """
    edge = net.get_edge(parent, child)
    length = edge.get_length()
    net.remove_edge(edge)
    root = Node(net._fresh_label("R"))
    net.add_nodes(root)
    # The new root sits halfway along the old edge, so the path length
    # between the old parent and child is unchanged.
    half = None if length is None else length / 2.0
    net.add_edges([
        Edge(root, child, length=half),
        Edge(root, parent, length=half),
    ])
    _reverse_upward(net, parent, ignore=root)
    net.clean()


def _reroot_above(net: Network, label: str) -> None:
    """Make the parent of leaf ``label`` the root, by reversing the path upward.

    Used to delete the virtual outgroup after neighbor-joining: the outgroup
    leaf hangs off the new root, and removing it leaves that root in place.
    A missing label, or a leaf that is already the root, is left unchanged.

    Args:
        net (Network): Tree mutated in place.
        label (str): Leaf whose parent becomes the root.
    """
    leaf = net.has_node_named(label)
    if leaf is None:
        return
    parents = net.get_parents(leaf)
    if not parents:
        return
    _reverse_upward(net, parents[0])


def _smaller_side(net: Network, child: Node) -> tuple[str, ...]:
    """Sorted labels of the smaller side of the edge into ``child``."""
    below = tuple(sorted(_labels(net, child)))
    universe = tuple(sorted(leaf.label for leaf in net.get_leaves()))
    complement = tuple(label for label in universe if label not in set(below))
    if (len(below), below) <= (len(complement), complement):
        return below
    return complement


def _root_by_triples(
    tree: Network,
    counts: dict[tuple[str, str, str], int],
) -> Network:
    """Copy of ``tree`` rooted on the edge that displays the most observed triples.

    The two edges under the current root are the same rooting; one of them is
    kept as the current-root candidate and the other is skipped. Ties break
    by the sorted labels of the smaller side of the edge. An empty count, or
    a count that no rooting displays, returns ``tree`` unchanged.

    Args:
        tree (Network): Unrooted-in-practice neighbor-joining tree.
        counts (dict): Observed rooted triples ``(a, b, c)`` with ``a < b``.

    Returns:
        Network: The chosen rooting.
    """
    if not counts or not any(counts.values()):
        return tree
    n_leaves = len(tree.get_leaves())
    if n_leaves > 60:
        warnings.warn(
            f"Rooting a {n_leaves}-leaf blob by rooted triples scores every edge.",
            stacklevel=2,
        )
    root = tree.root()
    under_root = sorted(tree.out_edges(root), key=lambda edge: edge.dest.label)
    skip = set(under_root[1:])
    best_score = 0
    best_side: tuple[str, ...] = ()
    best_copy: Optional[Network] = None
    for edge in tree.E():
        if edge in skip:
            continue
        side = _smaller_side(tree, edge.dest)
        copy, mapping = tree.copy()
        _reroot_on_edge(copy, mapping[edge.src], mapping[edge.dest])
        displayed = rooted_triples(copy)
        score = sum(counts.get(triple, 0) for triple in displayed)
        if best_copy is None or score > best_score or (score == best_score and side < best_side):
            best_score = score
            best_side = side
            best_copy = copy
    if best_copy is None or best_score == 0:
        return tree
    return best_copy


def overlap_nj(
    trees: Sequence[Network],
    labels: Sequence[str],
    matrix: np.ndarray,
    *,
    outgroup_distances: Optional[dict[str, float]] = None,
) -> tuple[Network, int]:
    """Neighbor-joining constrained by sibling pairs of ``trees``.

    ``matrix`` is indexed in the same order as ``labels``. A virtual outgroup
    is added from ``outgroup_distances`` (mean distance to scaffold siblings,
    already scaled) and then deleted, so the tree roots toward the scaffold
    parent. The outgroup is not a leaf of any constraint tree. When every
    pair violates a constraint, that step is relaxed and counted.

    The pair chosen at each step minimizes Q(i, j) = (n - 2) d(i, j) - r_i - r_j
    (Saitou and Nei, 1987, *Molecular Biology and Evolution*). Which pairs
    are legal is the NJMerge constraint (Molloy and Warnow, 2018).

    Args:
        trees (Sequence[Network]): Compatible constraint trees.
        labels (Sequence[str]): Taxon order of ``matrix``.
        matrix (np.ndarray): Square pairwise distances.
        outgroup_distances (dict | None): Distance from each label to the
            virtual outgroup. Omit to skip the outgroup.

    Returns:
        tuple[Network, int]: The merged tree (no virtual outgroup) and how
        many steps were relaxed.

    Raises:
        DimpleError: If a distance the merge needs is undefined, or the
            outgroup distances do not cover ``labels``.
    """
    names = list(labels)
    if len(names) == 0:
        raise DimpleError("Neighbor-joining was given no taxa.")
    if len(names) == 1:
        return _leaf_network(names[0]), 0
    if matrix.shape != (len(names), len(names)):
        raise DimpleError("Distance matrix shape does not match the taxon list.")
    if not np.isfinite(matrix).all():
        raise DimpleError(
            "A taxon pair this blob needs never co-occurs in a gene tree."
        )

    dist: dict[tuple[str, str], float] = {}
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if i < j:
                dist[(left, right)] = dist[(right, left)] = float(matrix[i, j])

    if outgroup_distances is not None:
        missing = [name for name in names if name not in outgroup_distances]
        if missing:
            raise DimpleError(
                "Outgroup distances are missing for " + ", ".join(missing[:5]) + "."
            )
        for name in names:
            value = float(outgroup_distances[name])
            dist[(name, _OUT)] = dist[(_OUT, name)] = value
        names.append(_OUT)

    constraints = [_Constraint(tree) for tree in trees]
    clusters = {name: _leaf_network(name) for name in names}
    active = list(names)
    relaxations = 0
    counter = 0

    while len(active) > 1:
        count = len(active)
        row_sum = {
            label: sum(_distance(dist, label, other) for other in active if other != label)
            for label in active
        }
        pairs = []
        for i, left in enumerate(active):
            for right in active[i + 1:]:
                # Saitou and Nei (1987): smallest Q is the next join.
                # Label order only breaks ties.
                q_value = (count - 2) * _distance(dist, left, right) - row_sum[left] - row_sum[right]
                pairs.append((q_value, left, right))
        pairs.sort(key=lambda item: (item[0], item[1], item[2]))

        def violates(left: str, right: str) -> bool:
            """True when some subset tree holds both labels but not as a cherry."""
            for constraint in constraints:
                if left in constraint.leaves and right in constraint.leaves:
                    if frozenset((left, right)) not in constraint.cherries():
                        return True
            return False

        chosen = None
        for _, left, right in pairs:
            if not violates(left, right):
                chosen = (left, right)
                break
        if chosen is None:
            # Every legal pair is blocked. Take the best Q pair and record it.
            relaxations += 1
            chosen = (pairs[0][1], pairs[0][2])
        left, right = chosen
        new_label = f"NJ{counter}"
        counter += 1
        for other in active:
            if other == left or other == right:
                continue
            updated = 0.5 * (
                _distance(dist, left, other)
                + _distance(dist, right, other)
                - _distance(dist, left, right)
            )
            dist[(new_label, other)] = dist[(other, new_label)] = updated
        for constraint in constraints:
            constraint.join(left, right, new_label)
        clusters[new_label] = _join_clusters(clusters[left], clusters[right])
        active = [label for label in active if label != left and label != right]
        active.append(new_label)

    merged = clusters[active[0]]
    if merged.has_node_named(_OUT) is not None:
        _reroot_above(merged, _OUT)
        merged.remove_nodes(merged.has_node_named(_OUT))
        merged.clean()
    return merged, relaxations


def _leaf_path_lengths(tree: Network, wanted: set[str]) -> dict[tuple[str, str], float]:
    """Path length between every pair of ``wanted`` leaves of one tree.

    One post-order pass: each node carries the distance to every wanted
    leaf below it, and two leaves meet once, at their MRCA. An edge with no
    length counts as one unit, as the newick readers default it.

    Args:
        tree (Network): A gene tree.
        wanted (set[str]): Leaf labels to measure.

    Returns:
        dict[tuple[str, str], float]: ``(a, b)`` with ``a < b`` to the path length.
    """
    root = next(node for node in tree.V() if tree.in_degree(node) == 0)
    order: list[Node] = []
    stack: list[tuple[Node, bool]] = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        stack.append((node, True))
        for child in tree.get_children(node):
            stack.append((child, False))
    below: dict[Node, dict[str, float]] = {}
    lengths: dict[tuple[str, str], float] = {}
    for node in order:
        children = tree.get_children(node)
        if not children:
            below[node] = {node.label: 0.0} if node.label in wanted else {}
            continue
        merged: dict[str, float] = {}
        for child in children:
            length = tree.get_edge(node, child).get_length()
            length = 1.0 if length is None else float(length)
            lifted = {label: dist + length for label, dist in below.pop(child).items()}
            for left, left_dist in merged.items():
                for right, right_dist in lifted.items():
                    key = (left, right) if left < right else (right, left)
                    lengths[key] = left_dist + right_dist
            merged.update(lifted)
        below[node] = merged
    return lengths


def gene_tree_distance_matrix(
    gene_trees: GeneTrees,
    taxa: Sequence[str],
    *,
    strict: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Mean path length over gene trees in which both taxa occur.

    This is not :func:`phynetpy.GraphUtils.pairwise_leaf_distance` (one tree)
    and not :func:`phynetpy.GraphUtils.average_path_distance` (a distance
    between two networks). Each tree contributes through
    :func:`_leaf_path_lengths`; the mean is taken only over the trees that
    contain the pair. A pair that never co-occurs is NaN. The trees are
    read on their full leaf sets; only ``taxa`` are measured.

    Args:
        gene_trees (GeneTrees): Source trees. Branch lengths are path weights.
        taxa (Sequence[str]): Labels to index the matrix, in sorted order
            in the return value.
        strict (bool): When True, a NaN pair raises. When False, NaN is left
            in the matrix for a caller that will not use that pair.

    Returns:
        tuple[list[str], np.ndarray]: Sorted labels and the square matrix.

    Raises:
        DimpleError: If ``strict`` and some pair never co-occurs.
    """
    labels = sorted(set(taxa))
    index = {label: position for position, label in enumerate(labels)}
    wanted = set(labels)
    size = len(labels)
    total = np.zeros((size, size))
    counts = np.zeros((size, size))
    for tree in gene_trees.trees:
        present = {leaf.label for leaf in tree.get_leaves()} & wanted
        if len(present) < 2:
            continue
        for (left, right), distance in _leaf_path_lengths(tree, wanted).items():
            i, j = index[left], index[right]
            total[i, j] += distance
            total[j, i] += distance
            counts[i, j] += 1
            counts[j, i] += 1
    never = (counts == 0) & ~np.eye(size, dtype=bool)
    if strict and np.any(never):
        raise DimpleError(
            f"{int(np.sum(never)) // 2} taxon pair(s) never co-occur in a "
            "gene tree; their distance is undefined."
        )
    matrix = total / np.maximum(counts, 1)
    matrix[never] = np.nan
    np.fill_diagonal(matrix, 0.0)
    return labels, matrix


def _outgroup_distances(
    labels: Sequence[str],
    matrix_labels: Sequence[str],
    matrix: np.ndarray,
    siblings: frozenset[str],
    *,
    scale: Optional[float] = None,
) -> dict[str, float]:
    """``scale`` times each blob taxon's mean distance to the scaffold siblings.

    If the sibling distances are a tree metric, adding a constant only
    lengthens the outgroup's pendant edge. Multiplying them does not stay
    additive, and neighbor-joining can then attach the outgroup to the
    wrong edge.

    Args:
        labels (Sequence[str]): Blob taxa to give a distance.
        matrix_labels (Sequence[str]): Row order of ``matrix``.
        matrix (np.ndarray): Pairwise distances.
        siblings (frozenset[str]): Scaffold leaves outside the blob.
        scale (float | None): Multiplier on the mean. Default
            :data:`OUTGROUP_SCALE`, DIMPLE's 2.0.

    Returns:
        dict[str, float]: Outgroup distance for each blob taxon that has
        a finite distance to every sibling.
    """
    index = {label: position for position, label in enumerate(matrix_labels)}
    sibs = [label for label in siblings if label in index]
    distances = {}
    for label in labels:
        if label not in index or not sibs:
            continue
        row = index[label]
        values = [matrix[row, index[sibling]] for sibling in sibs]
        if values and np.isfinite(values).all():
            factor = OUTGROUP_SCALE if scale is None else scale
            distances[label] = factor * float(np.mean(values))
    return distances


# ── Reticulation placement ────────────────────────────────────────────



def _major_edges(net: Network) -> dict[Node, Edge]:
    """The in-edge the major tree keeps at every node of in-degree two or more.

    Highest gamma first, then source label: the order
    :func:`phynetpy.GraphUtils.get_all_subtrees` enumerates, so its first
    tree keeps exactly these edges.
    """
    kept: dict[Node, Edge] = {}
    for node in net.V():
        if net.in_degree(node) < 2:
            continue
        incoming = sorted(
            net.in_edges(node), key=lambda edge: (-(edge.get_gamma() or 0.0), edge.src.label),
        )
        kept[node] = incoming[0]
    return kept


def _view(net: Network, drop: Sequence[Edge]) -> tuple[Network, dict[Node, Node]]:
    """A copy of ``net`` without the edges in ``drop``.

    An internal node left childless is removed, and so on upward, because
    :meth:`Network.leaf_descendants` would report it as a leaf. Degree-2
    chains are kept, so a surviving node of ``net`` is still found through
    the returned map; a removed node is simply absent from the view.
    """
    view, node_map = net.copy()
    leaves = {leaf.label for leaf in net.get_leaves()}
    for edge in drop:
        view.remove_edge([node_map[edge.src], node_map[edge.dest]])
    pruning = True
    while pruning:
        pruning = False
        for node in list(view.V()):
            if view.out_degree(node) == 0 and node.label not in leaves:
                view.remove_nodes(node)
                pruning = True
    return view, node_map


def _attachment_leaves(
    net: Network,
    kept: dict[Node, Edge],
    drop: Sequence[Edge],
    start: Node,
    taxa: frozenset[str],
) -> tuple[frozenset[str], int]:
    """Clade under the point a removed hybrid edge attaches to, and its height.

    ``start`` is the hybrid edge's source. Once ``drop`` (that edge and the
    other reticulations' non-reference edges) is gone, the edge attaches
    where ``start`` stands in the remaining tree: under ``start`` itself if
    it still has a child, otherwise under its nearest surviving ancestor
    (a reticulation's ancestor is through its ``kept`` edge). The clade is
    empty when nothing above ``start`` survives.

    The height counts the one-child nodes between the attachment point and
    the clade's top node. Those are the positions of other hybrid edges
    that leave or enter the same clade; on a backbone where they have been
    inserted, the edges above the clade form a chain, and the height says
    which link this edge belongs on.
    """
    view, node_map = _view(net, drop)
    present = set(view.V())
    node = start
    while node_map[node] not in present:
        if net.in_degree(node) >= 2 and node in kept:
            node = kept[node].src
        else:
            parents = net.get_parents(node)
            if not parents:
                return frozenset(), 0
            node = parents[0]
    anchor = node_map[node]
    height = 0
    current = anchor
    while view.out_degree(current) == 1:
        current = view.get_children(current)[0]
        if view.out_degree(current) == 1:
            height += 1
    return _labels(view, anchor) & taxa, height


def _chain_depth(net: Network, node: Node) -> int:
    """Reticulations stacked directly under ``node``, each the other's only child."""
    depth = 0
    current = node
    while True:
        children = net.get_children(current)
        if len(children) != 1 or net.in_degree(children[0]) < 2:
            return depth
        depth += 1
        current = children[0]


def reticulation_records(net: Network, kept: Optional[dict[Node, Edge]] = None) -> list[dict]:
    """One record per in-degree-2 node, with parent sides ordered by gamma.

    A node of any other in-degree is not a record. ``kept_sibling_leaves``
    is the higher-gamma side and ``minor_sibling_leaves`` is the lower-gamma
    side, which is the side added back when the backbone kept the major
    parent.

    Leaf sets are read on a displayed tree, not on the network. ``kept``
    names the in-edge that tree keeps at each other reticulation (default:
    the major tree, :func:`_major_edges`). ``retic_leaves`` is the clade
    under the reticulation in that tree; each side is the clade under the
    point the parent's hybrid edge attaches to once it is removed
    (:func:`_attachment_leaves`); the signature is read on the same tree.
    On a level-1 network these are the network's own leaf sets. With nested
    reticulations the network's leaf sets leak through other hybrid edges:
    the child is then not a clade of any tree the backbone displays, and a
    side can contain the child itself or be empty when the parent is a
    reticulation whose only child this is. ``retic_depth`` counts
    reticulations stacked directly below this one; the merger places the
    upper one of such a chain first. ``kept_height`` and ``minor_height``
    say how many other hybrid-edge positions lie between each side's
    attachment point and its clade (see :func:`_attachment_leaves`).

    Args:
        net (Network): A network, possibly a tree (then the list is empty).
        kept (dict[Node, Edge] | None): The displayed tree's kept in-edge
            per reticulation, as :func:`select_displayed_trees_with_choices`
            returns. None means the major tree.

    Returns:
        list[dict]: Records carrying ``signature``, leaf sets, gammas and
        ``retic_depth``, in label order of the reticulation node.
    """
    reference = _major_edges(net)
    if kept:
        reference.update({node: edge for node, edge in kept.items() if node in reference})
    records = []
    taxa = _labels(net)
    retics = sorted((node for node in net.V() if net.in_degree(node) == 2), key=lambda n: n.label)
    for node in retics:
        incoming = list(net.in_edges(node))
        # Higher gamma is the major parent. Equal gammas break by label.
        incoming.sort(key=lambda edge: (-(edge.get_gamma() or 0.0), edge.src.label))
        major_edge, minor_edge = incoming
        # The other reticulations as the displayed tree shows them; this
        # one keeps both parents so its own sides can be read. A chain of
        # reticulations directly below it hangs from it here, whatever the
        # displayed tree kept: otherwise this node has no child clade at all.
        local = dict(reference)
        current = node
        while True:
            children = net.get_children(current)
            if len(children) != 1 or net.in_degree(children[0]) < 2:
                break
            local[children[0]] = net.get_edge(current, children[0])
            current = children[0]
        others = [
            edge
            for other, kept_edge in local.items() if other is not node
            for edge in net.in_edges(other) if edge is not kept_edge
        ]
        base, base_map = _view(net, others)
        if base_map[node] not in set(base.V()):
            # Nothing of the displayed tree hangs below this reticulation.
            continue
        kept_side, kept_height = _attachment_leaves(
            net, local, others + [major_edge], major_edge.src, taxa,
        )
        minor_side, minor_height = _attachment_leaves(
            net, local, others + [minor_edge], minor_edge.src, taxa,
        )
        records.append({
            "signature": reticulation_signature(base, base_map[node]),
            "retic_leaves": _labels(base, base_map[node]),
            "kept_sibling_leaves": kept_side,
            "minor_sibling_leaves": minor_side,
            "kept_height": kept_height,
            "minor_height": minor_height,
            "major_gamma": float(major_edge.get_gamma() or 0.0),
            "minor_gamma": float(minor_edge.get_gamma() or 0.0),
            "network_taxa": taxa,
            "retic_depth": _chain_depth(net, node),
        })
    return records


def collect_reticulation_candidates(
    networks: Sequence[Network],
    references: Optional[Sequence[Optional[dict[Node, Edge]]]] = None,
) -> list[dict]:
    """One candidate per hybridization event across ``networks``.

    Records from every network are grouped with
    :func:`phynetpy.GraphUtils.group_reticulations`: two records are the
    same event when their parent-side sibling clades overlap, in either
    parent order. Each group contributes the record with the largest child
    clade (ties keep the earliest), which is how DIMPLE's
    ``collect_unique_retics`` picks a representative. Literal repeats of a
    representative are then dropped. The same event seen by several
    overlapping subsets is therefore tried once, not once per subset.

    Args:
        networks (Sequence[Network]): Inferred subnetworks.
        references (Sequence[dict | None] | None): Per network, the in-edge
            its displayed tree kept at each reticulation (see
            :func:`reticulation_records`); None means the major tree.

    Returns:
        list[dict]: Records from :func:`reticulation_records`, one per event.
    """
    records = [
        record
        for index, net in enumerate(networks)
        for record in reticulation_records(net, references[index] if references else None)
    ]
    representatives = [
        max(group, key=lambda record: len(record["retic_leaves"]))
        for group in group_reticulations(records)
    ]
    found = []
    seen = set()
    for record in representatives:
        key = (
            frozenset(record["network_taxa"]),
            frozenset(record["retic_leaves"]),
            frozenset(record["minor_sibling_leaves"]),
            frozenset(record["kept_sibling_leaves"]),
            round(record["major_gamma"], 6),
            round(record["minor_gamma"], 6),
        )
        if key in seen:
            continue
        seen.add(key)
        found.append(record)
    return found


def _mrca_or_none(net: Network, leaves: frozenset[str]) -> Optional[Node]:
    """MRCA of ``leaves``, or None when a label is absent."""
    if not leaves or any(net.has_node_named(label) is None for label in leaves):
        return None
    try:
        return net.mrca(set(leaves))
    except NetworkError:
        return None


def _clade_node(net: Network, leaves: frozenset[str]) -> Optional[Node]:
    """The node whose leaf set is exactly ``leaves``, or their MRCA.

    An exact clade is preferred. The MRCA is the fallback when a graft has
    left extra leaves inside that region, so the original set is no longer
    one node's descendant set.

    Args:
        net (Network): Tree or network to search.
        leaves (frozenset[str]): Target taxon set.

    Returns:
        Node | None: The clade node, the MRCA, or None when a label is
        absent or the MRCA cannot be computed.
    """
    node = _node_with_leaves(net, leaves)
    if node is not None:
        return node
    return _mrca_or_none(net, leaves)


def _child_clade(net: Network, record: dict) -> Optional[Node]:
    """MRCA of the reticulation's leaves, or None if it is not their clade.

    Leaves from other subsets may sit inside the clade: the source network
    never saw them. A leaf of the source network that sits inside the clade
    but was not under the reticulation means this is the wrong node.

    Args:
        net (Network): Current backbone.
        record (dict): A record from :func:`reticulation_records`.

    Returns:
        Node | None: The clade node, or None when the leaves are not a clade
        on the source network's taxa.
    """
    node = _mrca_or_none(net, record["retic_leaves"])
    if node is None:
        return None
    below = _labels(net, node) & frozenset(record["network_taxa"])
    return node if below == record["retic_leaves"] else None


def _side_leaves(
    net: Network,
    child: Node,
    parent: Node,
    taxa: set[str] | frozenset[str],
) -> set[str]:
    """Leaves beside ``child`` under ``parent``, restricted to ``taxa``."""
    current: set[str] = set()
    for sibling in net.get_children(parent):
        if sibling is not child:
            current |= set(_labels(net, sibling))
    return current & taxa


def choose_inserted_side(net: Network, record: dict) -> frozenset[str]:
    """Sibling leaves of the parental side the backbone does not already hold.

    Returns one of the record's own sets, so identity distinguishes the
    major side from the minor side. A reticulation child has two parents.
    Each parent's sibling leaves are compared with the two recorded sides,
    and the parent label breaks ties, so the choice does not follow
    ``get_parents()`` order.

    Args:
        net (Network): Current backbone.
        record (dict): A record from :func:`reticulation_records`.

    Returns:
        frozenset[str]: The side that should be inserted.
    """
    minor = record["minor_sibling_leaves"]
    kept = record["kept_sibling_leaves"]
    child = _child_clade(net, record)
    if child is None:
        return minor
    parents = net.get_parents(child)
    if not parents or not kept:
        return minor
    taxa = set(record["network_taxa"])
    decided: Optional[frozenset[str]] = None
    best = -1.0
    for parent in sorted(parents, key=lambda node: node.label):
        current = _side_leaves(net, child, parent, taxa)
        if not current:
            continue
        # ``current`` is the sibling set this parent already holds. Insert
        # the recorded side it does not match. A proper subset is decisive;
        # otherwise the higher Jaccard overlap is the side already held.
        # The choice is the record's own frozenset, so ``is`` still tells
        # the major side from the minor side.
        if current <= minor and not current <= kept:
            quality, choice = 1.0, kept
        elif current <= kept and not current <= minor:
            quality, choice = 1.0, minor
        else:
            kept_overlap = len(current & kept) / max(len(current | kept), 1)
            minor_overlap = len(current & minor) / max(len(current | minor), 1)
            if kept_overlap >= minor_overlap:
                quality, choice = kept_overlap, minor
            else:
                quality, choice = minor_overlap, kept
        if quality > best:
            best = quality
            decided = choice
    return decided if decided is not None else minor


def _destination_edge(
    net: Network,
    child: Node,
    held: frozenset[str],
    taxa: set[str],
) -> Optional[Edge]:
    """Edge above ``child`` on the parental side the backbone already holds.

    One parent means one edge. Two parents means the incoming edge whose
    other children match ``held``. The source label breaks a tie.

    Args:
        net (Network): Current backbone.
        child (Node): MRCA of the reticulation's descendant leaves.
        held (frozenset[str]): Sibling leaves of the side not being inserted.
        taxa (set[str]): Taxa of the source network.

    Returns:
        Edge | None: The edge to split as the hybrid's destination.
    """
    incoming = list(net.in_edges(child))
    if not incoming:
        return None
    if len(incoming) == 1:
        return incoming[0]

    def match(edge: Edge) -> tuple[int, str]:
        """Rank an in-edge by how many held leaves sit on its side, then by label."""
        covered = _side_leaves(net, child, edge.src, taxa)
        return (len(covered & held), edge.src.label)

    return max(incoming, key=match)


def candidate_edges(net: Network, record: dict) -> list[Edge]:
    """Edges where the missing parental side can be attached.

    Exact leaf-set matches come first. With none, partial matches are
    ordered by how many of the target leaves they cover. The edge into the
    reticulation child itself is excluded, so the source and destination of
    the edit are never the same edge (that edit would be a bubble).

    Args:
        net (Network): Current backbone.
        record (dict): Reticulation to place.

    Returns:
        list[Edge]: Candidate source edges, best match first. Empty when
        nothing on the tree carries the target leaves.
    """
    child = _child_clade(net, record)
    if child is None:
        return []
    target = choose_inserted_side(net, record)
    excluded = net.get_subtree_at(child)
    taxa = set(record["network_taxa"])
    labels = _label_map(net)
    exact: list[Edge] = []
    partial: list[tuple[int, str, Edge]] = []
    for edge in net.E():
        if edge.dest in excluded:
            continue
        covered = labels[edge.dest] & taxa
        if not covered or not covered <= target:
            continue
        # The destination of the hybrid edit is the edge above the child.
        # Offering that same edge as the source would build a bubble.
        if edge.dest is child:
            continue
        if covered == target:
            exact.append(edge)
        else:
            partial.append((len(covered), "".join(sorted(covered)), edge))
    if exact:
        # Exact matches with one clade below them form a chain of edges above
        # it (the positions of hybrid edges already inserted). The source
        # network says which link this edge belongs on; that one is tried
        # first, so an equal score keeps it, and the rest follow bottom-up.
        # Edge sets iterate in memory order, so the order is fixed by depth
        # and label rather than left to the process.
        depth: dict[Node, int] = {}
        frontier = [(root, 0) for root in net.roots()]
        while frontier:
            node, level = frontier.pop()
            if node in depth:
                continue
            depth[node] = level
            frontier.extend((child, level + 1) for child in net.get_children(node))
        exact.sort(key=lambda edge: (-depth.get(edge.dest, 0), edge.src.label, edge.dest.label))
        height = record.get("kept_height" if target is record["kept_sibling_leaves"] else "minor_height", 0)
        if 0 < height < len(exact):
            exact.insert(0, exact.pop(height))
        return exact
    partial.sort(key=lambda item: (-item[0], item[1]))
    return [edge for _, _, edge in partial]


def _edge_distance(net: Network, start: Node, goal: Node, limit: int = 2) -> int:
    """Undirected distance between two nodes, or ``limit + 1`` when farther."""
    if start is goal:
        return 0
    seen = {start}
    frontier = [start]
    for distance in range(1, limit + 1):
        nxt = []
        for node in frontier:
            neighbours = list(net.get_children(node)) + list(net.get_parents(node))
            for neighbour in neighbours:
                if neighbour is goal:
                    return distance
                if neighbour not in seen:
                    seen.add(neighbour)
                    nxt.append(neighbour)
        frontier = nxt
    return limit + 1


def _sort_records(records: list[dict], net: Network) -> list[dict]:
    """Place an event before one whose inserted side contains its child leaves.

    ``net`` is the backbone as the placement sees it (see
    :func:`_placement_view`). An upper reticulation of a chain also goes
    before the lower one.
    """
    # Place i before j when i's child leaves meet j's inserted side. The
    # outer edge is only a clade after the inner hybrid is in. Kahn order;
    # a cycle falls through and keeps the input order.
    sides = [choose_inserted_side(net, record) for record in records]
    count = len(records)
    dependencies = [set() for _ in range(count)]
    for i in range(count):
        child = records[i]["retic_leaves"]
        for j in range(count):
            if i == j:
                continue
            if child & sides[j]:
                dependencies[j].add(i)
            # Stacked reticulations share a child clade; the upper one goes
            # first, so the lower one is inserted into the edge below it.
            if child & records[j]["retic_leaves"] and (
                records[i].get("retic_depth", 0) > records[j].get("retic_depth", 0)
            ):
                dependencies[j].add(i)
    indegree = [len(deps) for deps in dependencies]
    queue = [index for index in range(count) if indegree[index] == 0]
    order: list[int] = []
    while queue:
        index = queue.pop(0)
        order.append(index)
        for other in range(count):
            if index in dependencies[other]:
                dependencies[other].discard(index)
                indegree[other] -= 1
                if indegree[other] == 0:
                    queue.append(other)
    for index in range(count):
        if index not in order:
            order.append(index)
    return [records[index] for index in order]


def _insert_hybrid(
    net: Network,
    source: Edge,
    destination: Edge,
    major_gamma: float,
    minor_gamma: float,
) -> tuple[str, str]:
    """Split ``source`` and ``destination`` and set the two inheritance probabilities.

    ``destination`` is the edge above the reticulation child (the side the
    backbone already holds). ``source`` is the side being added. Gamma on an
    edge that already entered a reticulation stays on the lower half of the
    split. ``add_hybrid`` does not take inheritance probabilities, so they
    are written onto the new edges after the split.

    Args:
        net (Network): Backbone mutated in place.
        source (Edge): Edge that donates the new parental side.
        destination (Edge): Edge above the reticulation child.
        major_gamma (float): Inheritance probability of the side the backbone
            already holds.
        minor_gamma (float): Inheritance probability of the side being added.

    Returns:
        tuple[str, str]: Labels of the new hybrid edge's source node and of
        the reticulation node, which name the inserted edge on any copy.
    """
    dest_gamma = destination.get_gamma() if destination.dest.is_reticulation() else None
    src_gamma = source.get_gamma() if source.dest.is_reticulation() else None
    dest_child_retic = destination.dest.is_reticulation()
    src_child_retic = source.dest.is_reticulation()
    before = {node.label for node in net.V()}
    add_hybrid(net, source, destination)
    created = [node for node in net.V() if node.label not in before]
    retic = next(node for node in created if node.is_reticulation())
    minor = next(node for node in created if node is not retic)
    for edge in net.in_edges(retic):
        if edge.src is minor:
            edge.set_gamma(minor_gamma)
        else:
            edge.set_gamma(major_gamma)
    # 0.0 is a real inheritance probability. Only a missing gamma is skipped.
    if dest_child_retic and dest_gamma is not None:
        for edge in net.out_edges(retic):
            edge.set_gamma(dest_gamma)
    if src_child_retic and src_gamma is not None:
        for edge in net.out_edges(minor):
            if edge.dest is not retic:
                edge.set_gamma(src_gamma)
    _populate_default_branch_lengths(net)
    return minor.label, retic.label





def _placement_view(net: Network, inserted: set[tuple[str, str]]) -> Network:
    """The backbone as its displayed tree: ``net`` without the hybrid edges placed so far.

    Clades are read on this view, so a reticulation already inserted does
    not leak its child's leaves into the clade of an edge above its source
    and spoil a later candidate's clade test or exact edge match. ``net``
    itself is returned while nothing has been inserted; a pre-existing
    reticulation of the input backbone is part of the view.
    """
    if not inserted:
        return net
    drop = [edge for edge in net.E() if (edge.src.label, edge.dest.label) in inserted]
    return _view(net, drop)[0]


def _edge_by_labels(net: Network, edge: Edge) -> Optional[Edge]:
    """The edge of ``net`` between the nodes named like ``edge``'s endpoints, or None."""
    src = net.has_node_named(edge.src.label)
    dest = net.has_node_named(edge.dest.label)
    if src is None or dest is None:
        return None
    try:
        return net.get_edge(src, dest)
    except (AssertionError, NetworkError, KeyError):
        return None


def place_reticulations(
    tree: Network,
    networks: Sequence[Network],
    score_fn: Callable[[Network], float],
    min_gain: float = 0.0,
    references: Optional[Sequence[Optional[dict[Node, Edge]]]] = None,
) -> tuple[Network, list[ReticTrial]]:
    """Greedily add reticulations from ``networks`` while the score rises.

    Each trial is a copy. The copy is kept only when it is the one that was
    scored and its log pseudo-likelihood is higher by more than
    ``min_gain`` (DIMPLE's rule is any strict gain, ``min_gain=0``; an
    AIC or BIC rule charges the reticulation's parameters, see
    :func:`selection_min_gain`). A non-finite score of the starting tree is
    an error, not a reason to add nothing. A candidate within two edges of
    an accepted reticulation with an overlapping child clade must clear
    :data:`REDUNDANCY_THRESHOLD`.

    Clades are read on the backbone's displayed tree
    (:func:`_placement_view`): the hybrid edges inserted here are not
    followed, so nested events can be placed in either order.

    Args:
        tree (Network): Backbone, usually the neighbor-joined tree.
        networks (Sequence[Network]): Inferred subnetworks the reticulations
            are taken from.
        score_fn (Callable): ``network -> log pseudo-likelihood``. Higher is
            better.
        min_gain (float): Log pseudo-likelihood a candidate must add on top
            of a strict improvement. Default 0.
        references (Sequence[dict | None] | None): Per network, the in-edge
            its displayed tree kept at each reticulation, from
            :func:`select_displayed_trees_with_choices`. None reads every
            network on its major tree.

    Returns:
        tuple[Network, list[ReticTrial]]: The tree with accepted reticulations
        and every trial.

    Raises:
        RuntimeError: If ``score_fn`` on the starting tree is not finite.
    """
    current = tree
    base = float(score_fn(current))
    if not math.isfinite(base):
        raise RuntimeError(
            f"Pseudo-likelihood of the base tree is not finite ({base})."
        )
    records = collect_reticulation_candidates(networks, references)
    # (source label, reticulation label) of every hybrid edge placed here.
    inserted: set[tuple[str, str]] = set()
    view = _placement_view(current, inserted)
    pending = _sort_records(records, view)
    trials: list[ReticTrial] = []
    # Child-leaf sets of accepted reticulations. A Node from the network
    # that was just copied is not in the next graph, so the set is resolved
    # again on ``current`` whenever a later candidate is scored.
    accepted: list[frozenset[str]] = []
    score = base

    while pending:
        record = pending.pop(0)
        child = record["retic_leaves"]
        child_node = _child_clade(view, record)
        if child_node is None:
            trials.append(ReticTrial(
                child, frozenset(), False, score, score, reason="child_not_clade",
            ))
            continue
        edges = candidate_edges(view, record)
        # The view hides the hybrid edges placed so far, so an edge may hang
        # below the child through one of them without being in its view
        # subtree. A source edge below the child would close a directed
        # cycle; reachability is the full network's.
        anchor = current.has_node_named(child_node.label)
        if anchor is not None:
            below = {node.label for node in current.get_subtree_at(anchor)}
            edges = [edge for edge in edges if edge.dest.label not in below]
        if not edges:
            trials.append(ReticTrial(
                child, frozenset(), False, score, score, reason="no_candidate_edge",
            ))
            continue
        side = choose_inserted_side(view, record)
        # The side being inserted is the major side exactly when the backbone
        # kept the minor parent. The gammas follow that side, not the names.
        if side is record["kept_sibling_leaves"]:
            major_gamma, minor_gamma = record["minor_gamma"], record["major_gamma"]
            held = record["minor_sibling_leaves"]
        else:
            major_gamma, minor_gamma = record["major_gamma"], record["minor_gamma"]
            held = record["kept_sibling_leaves"]
        destination = _destination_edge(
            view, child_node, held, set(record["network_taxa"]),
        )
        if destination is None:
            trials.append(ReticTrial(
                child, frozenset(), False, score, score, reason="no_destination",
            ))
            continue

        best_score = -math.inf
        best_trial: Optional[Network] = None
        best_edge_leaves: frozenset[str] = frozenset()
        best_near = False
        best_pair: Optional[tuple[str, str]] = None
        for edge in edges:
            if edge is destination:
                continue
            trial, _ = current.copy()
            source = _edge_by_labels(trial, edge)
            dest = _edge_by_labels(trial, destination)
            if source is None or dest is None:
                continue
            if source is dest or source.src is dest.src and source.dest is dest.dest:
                # Same edge: add_hybrid would build a bubble. Not a candidate.
                continue
            pair = _insert_hybrid(trial, source, dest, major_gamma, minor_gamma)
            if not trial.is_acyclic():
                continue
            trial_score = float(score_fn(trial))
            if trial_score > best_score:
                best_score = trial_score
                best_trial = trial
                best_pair = pair
                best_edge_leaves = _labels(view, edge.dest)
                anchor = current.has_node_named(edge.dest.label)
                best_near = anchor is not None and _near_accepted(current, anchor, child, accepted)

        threshold = REDUNDANCY_THRESHOLD if best_near else 0.0
        gain = best_score - score
        relative = gain / abs(score) if score != 0 else gain
        keep = (
            best_trial is not None
            and math.isfinite(best_score)
            and best_score > score + min_gain
            and relative >= threshold
        )
        trials.append(ReticTrial(
            child, best_edge_leaves, keep, score, best_score if math.isfinite(best_score) else score,
        ))
        if keep and best_trial is not None:
            current = best_trial
            score = best_score
            accepted.append(child)
            if best_pair is not None:
                inserted.add(best_pair)
            view = _placement_view(current, inserted)
            if pending:
                pending = _sort_records(pending, view)
    return current, trials


def _near_accepted(
    net: Network,
    anchor: Node,
    child: frozenset[str],
    accepted: Sequence[frozenset[str]],
) -> bool:
    """Whether ``anchor`` is within two edges of a reticulation ``child`` overlaps.

    Accepted events are leaf sets. Each one is resolved on ``net``, which is
    the graph ``anchor`` belongs to.
    """
    for existing in accepted:
        if not (child & existing):
            continue
        retic = _mrca_or_none(net, existing)
        if retic is not None and _edge_distance(net, anchor, retic) <= 2:
            return True
    labels = None
    for node in net.V():
        if net.in_degree(node) < 2:
            continue
        if labels is None:
            labels = _label_map(net)
        if child & labels[node] and _edge_distance(net, anchor, node) <= 2:
            return True
    return False


# ── Prune reattachment ────────────────────────────────────────────────


def _attachment_rank(scaffold: Optional[Network], prune: PrunedSubtree) -> tuple:
    """Higher means closer to the anchor, and therefore attached later (inner)."""
    if scaffold is None or not prune.anchor_leaves or not prune.parent_leaves:
        return (0, 0, prune.index)
    parent = _clade_node(scaffold, prune.parent_leaves)
    anchor = _mrca_or_none(scaffold, prune.anchor_leaves)
    if parent is None or anchor is None:
        return (0, 0, prune.index)
    try:
        common = scaffold.mrca({parent, anchor})
    except NetworkError:
        return (0, 0, prune.index)
    return (
        scaffold.dist_from_root(common),
        scaffold.dist_from_root(parent),
        prune.index,
    )


def _place_stuck_prune(
    net: Network,
    waiting: list[PrunedSubtree],
    attach: Callable[[PrunedSubtree, Network, Node], Network],
) -> tuple[Network, list[PrunedSubtree]]:
    """DIMPLE's host search for a reattachment pass that placed nothing.

    Every waiting clade is matched against every host: the blob, then each
    other waiting clade. The pair sharing the most anchor leaves wins (ties
    keep the first found), and the clade is attached inside that host at the
    MRCA of the shared leaves. A clade attached inside another waiting clade
    rides along when that clade is placed. One placement per call. Mirrors
    the generalized host search in ``add_pruned_subtree.add_pruned``.

    Args:
        net (Network): Blob with the clades placed so far.
        waiting (list[PrunedSubtree]): Clades no pass could place.
        attach (Callable): ``attach(prune, host, anchor)`` from
            :func:`reattach_prunes`.

    Returns:
        tuple[Network, list[PrunedSubtree]]: The blob and the clades still
        waiting, with a host clade replaced by its enlarged copy.

    Raises:
        DimpleError: If no host shares a leaf with any waiting anchor.
    """
    blob_labels = set(_labels(net))
    best_count = 0
    best: Optional[tuple[PrunedSubtree, Optional[PrunedSubtree]]] = None
    for prune in waiting:
        hosts: list[tuple[Optional[PrunedSubtree], set[str]]] = [(None, blob_labels)]
        hosts.extend(
            (other, set(_labels(other.subtree)))
            for other in waiting if other is not prune
        )
        for host, labels in hosts:
            count = len(set(prune.anchor_leaves) & labels)
            if count > best_count:
                best_count = count
                best = (prune, host)
    if best is None:
        raise DimpleError(
            f"{len(waiting)} pruned subtree(s) could not be attached: no host "
            "shares a leaf with their anchors."
        )
    prune, host = best
    if host is None:
        available = frozenset(set(prune.anchor_leaves) & blob_labels)
        anchor = _mrca_or_none(net, available)
        if anchor is None:
            raise DimpleError("A stuck pruned subtree has no MRCA on the blob.")
        net = attach(prune, net, anchor)
        return net, [other for other in waiting if other is not prune]
    host_tree = _copy(host.subtree)
    available = frozenset(set(prune.anchor_leaves) & set(_labels(host_tree)))
    anchor = _mrca_or_none(host_tree, available)
    if anchor is None:
        raise DimpleError("A stuck pruned subtree has no MRCA on its host clade.")
    host_tree = attach(prune, host_tree, anchor)
    carried = replace(host, subtree=host_tree)
    return net, [
        carried if other is host else other
        for other in waiting if other is not prune
    ]


def reattach_prunes(
    blob: Network,
    prunes: Sequence[PrunedSubtree],
    scaffold: Optional[Network] = None,
) -> Network:
    """Hang each pruned clade back on ``blob``.

    A clade is a sibling of the MRCA of its cut-time siblings. If that node
    is a reticulation, the clade goes under its child, inside the hybrid.
    The anchor being the root creates a new root rather than dropping the
    clade. A blob-item anchor that shares no leaf with the blob, or an empty
    anchor, raises. Two clades on the same anchor attach outer-first, so the
    cut closer to the anchor on ``scaffold`` ends up the inner sibling. When
    a pass attaches nothing, DIMPLE's host search (:func:`_place_stuck_prune`)
    places one clade wherever most of its anchor is found, on the blob or
    inside another waiting clade.

    Args:
        blob (Network): Merged blob network.
        prunes (Sequence[PrunedSubtree]): Clades to put back.
        scaffold (Network | None): Tree of blobs, used only to order clades
            that claim the same anchor.

    Returns:
        Network: ``blob`` plus every pruned clade.

    Raises:
        DimpleError: If a clade cannot be attached, or its anchor is empty
            or disjoint from the blob when it was supposed to land there.
    """
    net = _copy(blob)
    if not prunes:
        return net
    # A leaf that inference put in the blob and that a prune also owns would
    # be duplicated. Drop it from the blob; the prune brings it back.
    overlap = _labels(net) & set().union(*(set(prune.leaves) for prune in prunes))
    for label in overlap:
        node = net.has_node_named(label)
        if node is not None:
            net.remove_nodes(node)
    net.clean()
    initial = _labels(net)
    ordered = sorted(prunes, key=lambda prune: _attachment_rank(scaffold, prune))
    remaining = list(ordered)

    def attach(prune: PrunedSubtree, host: Network, anchor: Node) -> Network:
        """Graft ``prune`` as the sibling of ``anchor`` (moved below any reticulation chain)."""
        anchor = _below_retics(host, anchor)
        # graft_subtree returns the image and, when anchor is the root,
        # builds a new root so the clade is not dropped.
        host.graft_subtree(prune.subtree, sibling_of=anchor)
        return host

    while remaining:
        progress = False
        waiting: list[PrunedSubtree] = []
        for prune in remaining:
            if not prune.anchor_leaves:
                raise DimpleError(
                    "A pruned subtree has an empty anchor. The division "
                    "must record the cut's siblings."
                )
            if prune.source_type == "blob_item":
                if not (set(prune.anchor_leaves) & set(initial)):
                    raise DimpleError(
                        "A pruned subtree's anchor shares no leaf with the "
                        "blob, so it cannot be attached."
                    )
                # DIMPLE anchors a blob-item cut on the blob's own leaves,
                # not on leaves another prune has brought back since.
                target = set(prune.anchor_leaves) & set(initial)
                anchor = _mrca_or_none(net, frozenset(target))
                if anchor is None:
                    waiting.append(prune)
                    continue
                net = attach(prune, net, anchor)
                progress = True
                continue
            # source_type == "prune": every anchor leaf must already be back.
            if not set(prune.anchor_leaves) <= set(_labels(net)):
                waiting.append(prune)
                continue
            anchor = _mrca_or_none(net, prune.anchor_leaves)
            if anchor is None:
                waiting.append(prune)
                continue
            net = attach(prune, net, anchor)
            progress = True
        if not progress:
            net, waiting = _place_stuck_prune(net, waiting, attach)
        remaining = waiting
    return net


# ── Scaffold graft ────────────────────────────────────────────────────


def _remove_clade(net: Network, leaves: frozenset[str]) -> None:
    """Delete the clade whose leaf set is ``leaves``, then contract degree-2 nodes."""
    node = _node_with_leaves(net, leaves)
    if node is None:
        return
    parents = net.get_parents(node)
    doomed = list(net.get_subtree_at(node))
    if parents:
        net.remove_edge(net.get_edge(parents[0], node))
    for item in doomed:
        if item in net.V():
            net.remove_nodes(item)
    net.clean(_CONTRACT_DEGREE2)


def _rehang_inner(result: Network, graft_root: Node, saved: list[GraftPiece]) -> None:
    """Put detached inner blobs back inside a freshly grafted outer blob.

    Mirrors ``merge_to_tob.merge``: at each step the inner blob with the most
    anchor leaves currently on ``result`` goes first, since another inner
    blob may hold the leaves this one anchors on. It hangs as a sibling of
    the MRCA of the anchor leaves that are present (not necessarily all of
    them). With none present it hangs beside the first child of the graft
    root, or under the root when it has no children, so the leaves are kept.

    Args:
        result (Network): Network being assembled, mutated.
        graft_root (Node): Image of the outer blob's root on ``result``.
        saved (list[GraftPiece]): Inner blobs that were inside the replaced
            clade.
    """
    remaining = list(saved)
    while remaining:
        present = set(_labels(result))
        remaining.sort(key=lambda inner: -len(set(inner.anchor_leaves) & present))
        inner = remaining.pop(0)
        available = frozenset(set(inner.anchor_leaves) & present)
        anchor = _mrca_or_none(result, available) if available else None
        if anchor is not None:
            result.graft_subtree(
                inner.network, sibling_of=_below_retics(result, anchor),
            )
            continue
        children = result.get_children(graft_root) if graft_root in result.V() else []
        if children:
            result.graft_subtree(inner.network, sibling_of=_below_retics(result, children[0]))
        elif graft_root in result.V():
            result.graft_subtree(inner.network, under=graft_root)
        else:
            raise DimpleError(
                "An inner blob has no anchor leaf on the outer blob and the "
                "graft root is gone, so it cannot be reattached."
            )


def graft_blobs(scaffold: Network, pieces: Sequence[GraftPiece]) -> Network:
    """Replace each blob clade on ``scaffold`` with its combined network.

    The deepest blob is replaced first. An inner piece already grafted inside
    an outer clade is detached, the outer clade is replaced, and the inner
    piece is hung back on its anchor inside the new clade by
    :func:`_rehang_inner`, most-available anchor first, as DIMPLE's
    ``merge_to_tob`` does. A leaf of a piece that is still on the scaffold
    after the clade is removed is an error.

    Args:
        scaffold (Network): Tree of blobs, outgroup already removed.
        pieces (Sequence[GraftPiece]): Combined blobs.

    Returns:
        Network: The full-taxon network.

    Raises:
        DimpleError: If a piece repeats a leaf that is still on the scaffold,
            or its clade cannot be found.
    """
    result = _copy(scaffold)

    def depth_of(leaves: frozenset[str]) -> int:
        """Root distance of the scaffold node holding exactly ``leaves``; 0 when absent."""
        node = _clade_node(scaffold, leaves)
        if node is None:
            return 0
        return scaffold.dist_from_root(node)

    ordered = sorted(pieces, key=lambda piece: -depth_of(piece.tob_leaves))
    placed: list[GraftPiece] = []
    for piece in ordered:
        node = _clade_node(result, piece.tob_leaves)
        if node is None:
            raise DimpleError(
                "No scaffold clade has the leaves of blob "
                f"{sorted(piece.tob_leaves)[:5]}."
            )
        saved = [
            earlier for earlier in placed
            if earlier.tob_leaves < piece.tob_leaves
            and _node_with_leaves(result, earlier.tob_leaves) is not None
            and _node_with_leaves(result, earlier.tob_leaves) in result.get_subtree_at(node)
        ]
        for earlier in saved:
            _remove_clade(result, earlier.tob_leaves)
        # The outer node may have moved if a removal contracted a degree-2
        # chain. Find whatever still holds the remaining outer leaves.
        remaining = _labels(result, node) if node in result.V() else frozenset()
        if node not in result.V() or not remaining:
            node = _node_with_leaves(result, piece.tob_leaves - set().union(
                *(set(item.tob_leaves) for item in saved)
            ) if saved else piece.tob_leaves)
        if node is None:
            raise DimpleError("The blob clade disappeared while detaching an inner graft.")
        parents = result.get_parents(node)
        parent = parents[0] if parents else None
        doomed = list(result.get_subtree_at(node))
        if parent is not None:
            result.remove_edge(result.get_edge(parent, node))
        for item in doomed:
            if item in result.V():
                result.remove_nodes(item)
        if parent is None and len(result.V()) == 0:
            result = _copy(piece.network)
            graft_root = next(
                vertex for vertex in result.V() if result.in_degree(vertex) == 0
            )
        else:
            if parent is None:
                raise DimpleError(
                    "Replacing the root left other nodes behind. The blob "
                    "does not cover the scaffold."
                )
            # Leaves of the piece must not still be somewhere else.
            collision = _labels(piece.network) & _labels(result)
            if collision:
                sample = ", ".join(sorted(collision)[:5])
                raise DimpleError(
                    f"Grafting would duplicate taxa already on the scaffold: {sample}."
                )
            graft_root = result.graft_subtree(piece.network, under=parent)
        _rehang_inner(result, graft_root, saved)
        placed.append(piece)
    result.clean(_CONTRACT_DEGREE2)
    return result


# ── Per-blob merge ────────────────────────────────────────────────────


def _msc_scorer(gene_trees: GeneTrees, taxa: frozenset[str]) -> Callable[[Network], float]:
    """Log pseudo-likelihood on ``taxa``. Higher is better.

    A non-finite value is returned as ``-inf``. Any other failure from the
    triplet scorer propagates, so a broken reticulation is not reported as
    a topology that simply scored badly.

    Args:
        gene_trees (GeneTrees): Full gene trees. Trees that omit taxa are
            restricted to ``taxa``.
        taxa (frozenset[str]): Leaves of the network being scored.

    Returns:
        Callable[[Network], float]: The scorer.
    """
    from .infer import compute_gene_tree_triplets, score_species_network_triplets
    from ._mpl import _HAS_CYTHON_MPL, _TripleDPEngine, _score_with_cython

    subset = restrict_gene_trees(gene_trees, sorted(taxa))
    table = compute_gene_tree_triplets(
        subset, subset.resolved_mapping(), species_labels=sorted(taxa),
    )

    def score(net: Network) -> float:
        """Log pseudo-likelihood of ``net`` on the subset's triplet table."""
        _populate_default_branch_lengths(net)
        value = None
        if _HAS_CYTHON_MPL:
            # The compiled scorer the subset searches use. It refuses
            # networks past its compiled limits (nodes, reticulations);
            # the Python scorer takes those.
            try:
                value = float(_score_with_cython(
                    _TripleDPEngine(net), table.triplets, table.rho_by_triplet,
                ))
            except ValueError as exc:
                if "compiled limit" not in str(exc):
                    raise
        if value is None:
            value = float(score_species_network_triplets(net, table).log_pseudo_likelihood)
        if not math.isfinite(value):
            return float("-inf")
        return value

    return score


def merge_blob_networks(
    networks: Sequence[Network],
    gene_trees: GeneTrees,
    external_siblings: frozenset[str],
    required_taxa: frozenset[str] = frozenset(),
    model_selection: Optional[str] = None,
    trace: Optional[dict] = None,
    tree_sources: Optional[Sequence[Network]] = None,
    backbone: Optional[Network] = None,
) -> tuple[Network, int, list[ReticTrial]]:
    """Displayed trees, constrained neighbor-joining, then reticulations.

    A single network that already carries every required taxon is returned
    as itself: there is nothing to reconcile. The returned relaxation count
    is zero in that case.

    Args:
        networks (Sequence[Network]): Inferred subnetworks of one blob.
        gene_trees (GeneTrees): Full gene trees.
        external_siblings (frozenset[str]): Scaffold leaves outside the blob,
            used only to place the virtual outgroup.
        required_taxa (frozenset[str]): Blob taxa that must be in the result
            even if no subnetwork carries them (a subset too small to
            infer). They join neighbor-joining as unconstrained leaves.
        model_selection (str | None): ``gain``, ``aic`` or ``bic`` make
            the merger require that gain of each reticulation
            (:func:`selection_min_gain`); ``elbow`` and None keep DIMPLE's
            any-improvement rule.
        trace (dict | None): When given, filled with the intermediate
            results for diagnostics: ``displayed`` (one tree per input),
            ``compatible`` (the trees that reached neighbor-joining),
            ``backbone`` (the merged tree before reticulations), and
            ``candidates`` (the reticulation records tried).
        tree_sources (Sequence[Network] | None): Diagnostic override. Trees
            to use as the displayed trees instead of picking one per
            network by triple support; reticulations are still taken from
            ``networks``.
        backbone (Network | None): Diagnostic override. A tree on the blob
            taxa to place reticulations on, skipping displayed-tree
            selection, the compatibility filter and neighbor-joining.

    Returns:
        tuple[Network, int, list[ReticTrial]]: Merged tree-plus-reticulations,
        neighbor-joining relaxations, and reticulation trials.
    """
    taxa: set[str] = set()
    for net in networks:
        taxa |= set(_labels(net))
    if len(networks) == 1 and set(required_taxa) <= taxa:
        return networks[0], 0, []
    taxa |= set(required_taxa)
    # One restriction to the blob serves the triple counts and the scorer.
    # The distance matrix reads the full trees: it also needs the siblings.
    blob_trees = restrict_gene_trees(gene_trees, sorted(taxa))
    counts = observed_triples(blob_trees, sorted(taxa))
    if backbone is not None:
        if frozenset(_labels(backbone)) != frozenset(taxa):
            raise DimpleError("The supplied backbone is not on the blob's taxa.")
        merged = _copy(backbone)
        if trace is not None:
            trace["displayed"] = []
            trace["compatible"] = []
            trace["backbone"] = _copy(merged)
            trace["candidates"] = collect_reticulation_candidates(networks)
        scorer = _msc_scorer(blob_trees, frozenset(sorted(taxa)))
        min_gain = 0.0
        if model_selection in ("gain", "aic", "bic"):
            min_gain = selection_min_gain(model_selection, _selection_data_size(blob_trees))
        with_retics, trials = place_reticulations(merged, networks, scorer, min_gain=min_gain)
        return with_retics, 0, trials
    references: Optional[list[Optional[dict[Node, Edge]]]] = None
    if tree_sources is not None:
        displayed = list(tree_sources)
    else:
        displayed, references = select_displayed_trees_with_choices(networks, counts)
    # DIMPLE's default filter keeps the largest clique of pairwise
    # compatible trees, not the first-fit subset.
    compatible = max_clique_compatible(displayed)
    assert compatible, "max_clique_compatible returned no trees"
    if trace is not None:
        trace["displayed"] = displayed
        trace["compatible"] = compatible
    needed = taxa | set(external_siblings)
    # Sibling–sibling pairs are not read. A missing one must not fail the
    # matrix; blob–blob pairs are checked in overlap_nj, and a blob taxon
    # with no finite sibling distance fails when the outgroup is placed.
    labels, matrix = gene_tree_distance_matrix(gene_trees, sorted(needed), strict=False)
    index = {label: position for position, label in enumerate(labels)}
    blob_labels = [label for label in labels if label in taxa]
    positions = [index[label] for label in blob_labels]
    submatrix = matrix[np.ix_(positions, positions)]
    outgroup = None
    if external_siblings:
        outgroup = _outgroup_distances(blob_labels, labels, matrix, external_siblings)
        if any(label not in outgroup for label in blob_labels):
            raise DimpleError(
                "Cannot place the virtual outgroup: a blob taxon has no "
                "distance to the scaffold siblings."
            )
    # Taxa that survived in no constraint tree are still required. They are
    # unconstrained leaves of the distance matrix, not dropped.
    merged, relaxations = overlap_nj(
        compatible, blob_labels, submatrix, outgroup_distances=outgroup,
    )
    # No scaffold sister: NJ's last join is not a root. Pick the edge whose
    # rooting displays the most observed triples.
    if outgroup is None:
        merged = _root_by_triples(merged, counts)
    if trace is not None:
        trace["backbone"] = _copy(merged)
        trace["candidates"] = collect_reticulation_candidates(networks, references)
    scorer = _msc_scorer(blob_trees, frozenset(blob_labels))
    min_gain = 0.0
    if model_selection in ("gain", "aic", "bic"):
        min_gain = selection_min_gain(model_selection, _selection_data_size(blob_trees))
    with_retics, trials = place_reticulations(
        merged, networks, scorer, min_gain=min_gain, references=references,
    )
    return with_retics, relaxations, trials


# ── Entry point ───────────────────────────────────────────────────────


def infer_dimple(
    gene_trees: GeneTrees,
    tree_of_blobs: Network,
    start_tree: Network,
    *,
    max_subset_size: int = 12,
    n_divisions: int = 15,
    max_reticulations: int = 1,
    seed: int = 0,
    outgroup: Optional[str] = None,
    model_selection: Optional[str] = None,
    true_network: Optional[Network] = None,
    score_final: bool = True,
    **search,
):
    """Infer a network by dividing a tree of blobs and stitching the pieces.

    The scaffold is partitioned at its polytomies. Each subset is inferred
    with MSC triplet pseudo-likelihood (:func:`phynetpy.infer.infer`), and
    the subset networks are merged back onto the scaffold. This is not a
    criterion and it is not dispatched through the inference registry.

    Each subset is inferred by simulated annealing: :data:`SUBSET_RUNS`
    runs of :data:`SUBSET_ITERS` iterations, and the highest log
    pseudo-likelihood is kept. Extra keyword arguments are forwarded to
    each subset ``infer`` call and override those defaults (``num_iter``,
    ``n_runs``, ``method``, ``t_start``). ``seed`` seeds only the
    partition shuffle. The search seeds are ``0, 1, 2`` unless a
    ``seed`` keyword is passed in the forwarded arguments.

    Args:
        gene_trees (GeneTrees): Rooted gene trees on the full taxon set.
            Individual trees may omit taxa. They are not rerooted, and no
            outgroup leaf is added to them.
        tree_of_blobs (Network): Rooted scaffold. A node with more than two
            children is a blob.
        start_tree (Network): Binary tree on the same taxa, used as the
            pseudo-likelihood start on each subset.
        max_subset_size (int): Largest subset. Default 12, minimum 4.
        n_divisions (int): Alternative partitions of each blob. Default 15.
        max_reticulations (int): Cap on each subset search. The merge can
            still keep every reticulation the subsets returned, so the
            stitched network can contain more than this many.
        seed (int): Seed for the partition packings and the enumeration
            shuffle. The first division of every blob is deterministic.
        outgroup (str | None): Leaf to drop from the scaffold before
            partitioning. Not inserted into the gene trees. ``None`` leaves
            the scaffold unchanged.
        model_selection (str | None): Choose each subset's reticulation
            count by ``gain`` (recommended; a step must add more than
            :data:`SELECTION_GAIN_THRESHOLD`), ``aic``, ``bic``, or
            ``elbow`` over caps 0 to ``max_reticulations``, each cap
            warm-started from the one below, and, for ``gain``, ``aic`` and
            ``bic``, make the merger require the same gain of each
            reticulation. None keeps the fixed cap and DIMPLE's
            any-improvement rule.
        true_network (Network | None): Ground-truth network. When given,
            each subset's cap is the number of detectable reticulations in
            the true network restricted to the subset
            (:func:`true_reticulation_bound`), the protocol of the DIMPLE
            paper, and ``max_reticulations`` and the subset part of
            ``model_selection`` are not used. For benchmarks only.
        score_final (bool): Score the stitched network (a triplet table on
            every taxon; seconds to minutes on large data). False leaves
            ``score`` None.
        **search: Subset search settings, see :func:`subset_chains`.
            Defaults are simulated annealing, :data:`SUBSET_ITERS` iterations,
            and :data:`SUBSET_RUNS` runs. A passed value replaces the default.
            ``chains=[{"method": "sa", "num_iter": 20000, "n_runs": 5},
            {"method": "hc", "num_iter": 20000, "n_runs": 3}]`` runs a
            portfolio per subset and keeps the best-scoring network.

    Returns:
        InferenceResult: ``best`` is the full-taxon network, ``score`` is its
        log pseudo-likelihood, ``method`` is ``"DIMPLE"``, and ``raw`` is a
        :class:`DimpleTrace`.

    Raises:
        DimpleError: If the inputs are illegal or a stitch step cannot place
            a clade.
        RuntimeError: If a blob's base tree has a non-finite pseudo-likelihood
            while reticulations are being placed.
    """
    from .infer import InferenceResult, score

    validate_inputs(
        gene_trees, tree_of_blobs, start_tree,
        max_subset_size=max_subset_size,
        n_divisions=n_divisions,
        outgroup=outgroup,
    )
    if model_selection is not None and model_selection not in SELECTION_RULES:
        raise DimpleError(
            f"model_selection must be one of {SELECTION_RULES} or None, "
            f"not {model_selection!r}."
        )
    scaffold = strip_outgroup(tree_of_blobs, outgroup)
    # Every later step reads the gene trees on scaffold taxa. Drop the
    # outgroup (and any taxon not on the scaffold) once here instead of
    # inside every subset, blob, and the final score.
    gene_trees = restrict_gene_trees(gene_trees, sorted(_labels(scaffold)))
    if true_network is not None:
        absent = sorted(_labels(scaffold) - _labels(true_network))
        if absent:
            raise DimpleError(
                "The true network lacks scaffold taxa: " + ", ".join(absent[:5]) + "."
            )
    jobs = partition_scaffold(
        scaffold,
        max_subset_size=max_subset_size,
        n_divisions=n_divisions,
        seed=seed,
    )
    blob_results: list[BlobResult] = []
    pieces: list[GraftPiece] = []
    for job in jobs:
        subsets: list[SubsetRecord] = []
        seen_subsets: set[frozenset[str]] = set()
        for division in job.divisions:
            for taxa in division:
                if taxa in seen_subsets or len(taxa) < 3:
                    continue
                seen_subsets.add(taxa)
                if true_network is not None:
                    bound = true_reticulation_bound(true_network, sorted(taxa))
                    found = _infer_subset(gene_trees, start_tree, taxa, bound, search)
                else:
                    found = _infer_subset(
                        gene_trees, start_tree, taxa, max_reticulations, search,
                        model_selection=model_selection,
                    )
                subsets.append(replace(found, blob_name=job.name))
        if not subsets:
            raise DimpleError(f"Blob {job.name} produced no subset to infer.")
        # Every taxon the division gave this blob, including those in a
        # subset too small to infer. They must not fall off the blob.
        required = frozenset().union(
            *(subset for division in job.divisions for subset in division)
        )
        if len(subsets) == 1 and required <= _labels(subsets[0].network):
            merged: Network = subsets[0].network
            relaxations = 0
            trials: list[ReticTrial] = []
        else:
            merged, relaxations, trials = merge_blob_networks(
                [record.network for record in subsets],
                gene_trees,
                job.external_siblings,
                required_taxa=required,
                model_selection=model_selection,
            )
        combined = reattach_prunes(merged, job.pruned, scaffold)
        blob_results.append(BlobResult(job, subsets, combined, relaxations, trials))
        pieces.append(GraftPiece(job.tob_leaves, combined, job.external_siblings))

    final = graft_blobs(scaffold, pieces)
    final.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
    # Score on the leaves the network actually has, so an outgroup that was
    # stripped from the scaffold and is still sitting in a gene tree does not
    # make the score undefined.
    final_score: Optional[float] = None
    if score_final:
        scoring_trees = restrict_gene_trees(gene_trees, sorted(_labels(final)))
        final_score = float(score(
            final, scoring_trees, model=MSC(), criterion=PseudoLikelihood(),
        ))
    trace = DimpleTrace(blob_results, final, final_score)
    return InferenceResult(
        best=final,
        score=final_score,
        method="DIMPLE",
        raw=trace,
    )
