#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author : Mark Kessler
Last Edit : 10/29/26
First Included in Version : 0.7.0

JointDissimilarity -- a joint reticulation-and-tree dissimilarity for rooted
phylogenetic networks.

For two networks N1, N2 on the same taxon set, with r1 and r2 reticulations
and ``m = max(r1, r2)``:

* **Tripartitions.** Every reticulation h gives ``(A, B, C)``: B is the block
  below h, A and C the blocks below h's two parents once h is deleted, pruned
  to the vertices on a path from that parent to a taxon.  Two tripartitions
  cost ``delta(t, t') = d(B, B') + min(d(A, A') + d(C, C'), d(A, C') + d(C, A'))``
  in ``[0, 3]``.
* **Reticulation term.** The smaller collection is padded with dummies to m
  (real--dummy cost 3) and ``D_ret`` is the minimum over bijections of the
  summed cost, by one Hungarian solve.
* **Remainder.** ``T(N)`` deletes the union of the blocks below all
  reticulations and hangs one marker leaf per surviving parent edge, labelled
  by the reticulation's descendant taxa.  ``D_tree = d(T(N1), T(N2))``.
* **Joint.** ``D_joint = D_ret + 3 D_tree``; normalised forms ``D_ret / 3m``
  (0 when m = 0) and ``D_joint / 3(m + 1)``, both in ``[0, 1]``.

The structure dissimilarity ``d(S, S')`` in ``[0, 1]`` is one of three
instantiations, each comparing the COMPLETE structures (no restriction to
shared leaves, no weighting; both empty -> 0, exactly one empty -> 1):

===========  ==========================================================
``leaf``     Jaccard distance of the leaf-label sets
``cluster``  multiset hardwired-cluster dissimilarity, one cluster per
             vertex
``nakhleh``  ``L / U`` over Nakhleh's vertex-equivalence classes, with
             ``L = sum_q |c_S(q) - c_S'(q)|`` and
             ``U = sum_q max(c_S(q), c_S'(q))``
===========  ==========================================================

Nothing is ever suppressed (a parent left unary by a deletion stays), and two
marker leaves may share a label, which is what makes the structural
instantiations well defined on the multi-labelled remainder.  Everything after
validation runs on :class:`_Graph`, a small rooted DAG detached from
:class:`~.Network.Network`, so no network is mutated or copied.

Reference: Nakhleh, *Reticulation-Aware Dissimilarity for Phylogenetic
Networks* (2026 manuscript, joint reticulation-and-tree form); Nakhleh,
*A metric on the space of reduced phylogenetic networks*, IEEE/ACM TCBB 7
(2010) 218--222 for the vertex-equivalence classes.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

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

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .Network import Network, Node, NetworkError

__all__ = ["JOINT_KINDS", "JointDissimilarityResult", "joint_dissimilarity"]

#: The three instantiations of the structure dissimilarity ``d(S, S')``.
JOINT_KINDS = ("leaf", "cluster", "nakhleh")

#: Cost of pairing a real tripartition with a padding dummy: the maximum of
#: ``delta``, three block comparisons each in ``[0, 1]``.
_DUMMY_COST = 3.0


# ── The light rooted-DAG type every block and remainder is built on ──────

class _Graph:
    """
    A rooted DAG with labelled leaves, detached from :class:`Network`.

    ``children[v]`` is a tuple of child ids without repeats and ``label[v]``
    the label of leaf ``v``.  Two leaves may share a label (two markers of
    reticulations with the same descendant taxa).  ``root`` is None only for
    the empty graph.

    Args:
        root (int | None): Id of the root vertex, None for the empty graph.
        children (dict[int, tuple[int, ...]]): Adjacency, every vertex a key.
        label (dict[int, object]): Label of every leaf vertex.
    """

    __slots__ = ("root", "children", "label", "_leaves", "_order", "_clusters")

    def __init__(self, root, children: dict, label: dict) -> None:
        self.root, self.children, self.label = root, children, label
        self._leaves = self._order = self._clusters = None

    @property
    def leaves(self) -> frozenset:
        """
        The set of distinct leaf labels ``L(S)`` (cached: the leaf
        instantiation reads it once per block comparison).

        Returns:
            frozenset: Every label carried by a leaf, each once.
        """
        if self._leaves is None:
            self._leaves = frozenset(self.label.values())
        return self._leaves

    def order(self) -> list:
        """
        Vertices root first, every vertex before its children (Kahn's sort,
        cached).

        Returns:
            list[int]: A topological order of the vertex ids.
        """
        if self._order is None:
            indeg = Counter(c for cs in self.children.values() for c in cs)
            out, ready = [], [] if self.root is None else [self.root]
            # Pop a vertex once every parent has been emitted, so a vertex
            # reached along two paths (below a reticulation) appears once.
            while ready:
                v = ready.pop()
                out.append(v)
                for c in self.children[v]:
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        ready.append(c)
            self._order = out
        return self._order

    def cluster_multiset(self) -> Counter:
        """
        Hardwired clusters of the FULL structure, one record per vertex.

        A cluster is the multiset of labels of the leaf vertices reachable
        from the vertex: a leaf reached along two paths counts once, two
        leaves with the same label count twice.  Records are keyed by the
        frozen set of ``(label, multiplicity)`` pairs so labels need not be
        sortable.

        Returns:
            Counter: Cluster record -> number of vertices carrying it.
        """
        if self._clusters is None:
            reach, out = {}, Counter()
            for v in reversed(self.order()):
                if v in self.label:
                    reach[v] = frozenset([v])
                else:
                    reach[v] = frozenset().union(*(reach[c] for c in self.children[v]))
                out[frozenset(Counter(self.label[leaf] for leaf in reach[v]).items())] += 1
            self._clusters = out
        return self._clusters

    def node_classes(self, interner: dict) -> Counter:
        """
        Nakhleh's vertex-equivalence classes of the FULL structure.

        Two vertices are equivalent when they are leaves with the same label,
        or when their children can be matched one-to-one into equivalent
        vertices (Nakhleh 2010, computed bottom-up as in Aho--Hopcroft--Ullman
        tree isomorphism).  Classes are interned in ``interner`` so two graphs
        keyed through the SAME dict share class ids.

        Args:
            interner (dict): Signature -> class id table, mutated in place.

        Returns:
            Counter: Class id -> number of vertices in that class.
        """
        key, out = {}, Counter()
        for v in reversed(self.order()):
            if v in self.label:
                sig = ("leaf", self.label[v])
            else:
                sig = ("node", tuple(sorted(key[c] for c in self.children[v])))
            key[v] = interner.setdefault(sig, len(interner))
            out[key[v]] += 1
        return out


_EMPTY = _Graph(None, {}, {})


def _restrict(G: _Graph, root, keep: set) -> _Graph:
    """
    The graph induced on the vertex set ``keep`` of ``G``, rooted at ``root``.

    Args:
        G (_Graph): The graph to restrict.
        root (int): Id of the root of the result; must lie in ``keep``.
        keep (set[int]): Vertex ids to retain.

    Returns:
        _Graph: ``G`` with every vertex and edge outside ``keep`` dropped.
    """
    return _Graph(root,
                  {v: tuple(c for c in G.children[v] if c in keep) for v in keep},
                  {v: G.label[v] for v in keep if v in G.label})


def _subgraph(G: _Graph, root, exclude: set = frozenset()) -> _Graph:
    """
    Everything reachable from ``root`` without entering a vertex of
    ``exclude``.  The block B below a reticulation h is ``_subgraph(G, h)``.

    Args:
        G (_Graph): The whole network.
        root (int): Vertex to start from.
        exclude (set[int]): Vertices that are deleted first.

    Returns:
        _Graph: The reachable sub-DAG, or the empty graph if ``root`` is
                excluded.
    """
    if root in exclude:
        return _EMPTY
    seen, stack = {root}, [root]
    while stack:
        for c in G.children[stack.pop()]:
            if c not in exclude and c not in seen:
                seen.add(c)
                stack.append(c)
    return _restrict(G, root, seen)


def _parent_block(G: _Graph, root, exclude: set) -> _Graph:
    """
    The vertices on a path from ``root`` to a taxon leaf once ``exclude`` is
    deleted: the blocks A and C of a tripartition.  Nothing is suppressed, so a
    parent left unary by the deletion stays.

    Args:
        G (_Graph): The whole network.
        root (int): A parent of the deleted reticulation.
        exclude (set[int]): The reticulation being deleted.

    Returns:
        _Graph: The pruned block, empty when no path from ``root`` to a taxon
                leaf survives the deletion.
    """
    S = _subgraph(G, root, exclude)
    if S.root is None:
        return _EMPTY
    # Bottom-up: keep a leaf, and keep an internal vertex iff one of its
    # children is kept, so exactly the vertices on a root-to-taxon path stay.
    keep = set()
    for v in reversed(S.order()):
        if v in S.label or any(c in keep for c in S.children[v]):
            keep.add(v)
    return _restrict(S, S.root, keep) if S.root in keep else _EMPTY


# ── The three instantiations of d(S, S') ──────────────────────────────────

def _multiset_dissimilarity(C1: Counter, C2: Counter) -> float:
    """
    ``sum_k |C1[k] - C2[k]| / sum_k max(C1[k], C2[k])``, the multiset form of
    the Jaccard distance; 0 when both multisets are empty.

    Args:
        C1 (Counter): First multiset.
        C2 (Counter): Second multiset.

    Returns:
        float: A value in ``[0, 1]``.
    """
    keys = C1.keys() | C2.keys()
    union = sum(max(C1[k], C2[k]) for k in keys)
    return sum(abs(C1[k] - C2[k]) for k in keys) / union if union else 0.0


def _d_leaf(S: _Graph, T: _Graph) -> float:
    """
    Leaf instantiation: Jaccard distance of the two label sets.

    Args:
        S (_Graph): First structure.
        T (_Graph): Second structure.

    Returns:
        float: ``|L xor L'| / |L u L'|`` in ``[0, 1]``; 0 when both are empty.
    """
    P, Q = S.leaves, T.leaves
    return len(P ^ Q) / len(P | Q) if (P or Q) else 0.0


def _d_cluster(S: _Graph, T: _Graph) -> float:
    """
    Cluster instantiation: multiset dissimilarity of the hardwired-cluster
    records of the two full structures (one record per vertex).

    Args:
        S (_Graph): First structure.
        T (_Graph): Second structure.

    Returns:
        float: A value in ``[0, 1]``.
    """
    return _multiset_dissimilarity(S.cluster_multiset(), T.cluster_multiset())


def _d_nakhleh(S: _Graph, T: _Graph) -> float:
    """
    Nakhleh instantiation: ``L / U`` over the vertex-equivalence classes of
    the two full structures (equivalently ``4 D / (|V| + |V'| + 2 D)`` with
    ``D = L / 2`` Nakhleh's 2010 metric).

    Args:
        S (_Graph): First structure.
        T (_Graph): Second structure.

    Returns:
        float: A value in ``[0, 1]``.
    """
    interner: dict = {}
    return _multiset_dissimilarity(S.node_classes(interner), T.node_classes(interner))


_DSET: dict[str, Callable[[_Graph, _Graph], float]] = {
    "leaf": _d_leaf, "cluster": _d_cluster, "nakhleh": _d_nakhleh,
}


# ── From a Network to its tripartitions and remainder ─────────────────────

def _validate(net: Network) -> list[Node]:
    """
    Refuse a network the measure is not defined on, and return its
    topological order.

    Checked in order: nonempty with exactly one root; acyclic; every leaf
    labelled and no label used twice; every reticulation (in-degree >= 2,
    whatever its flag) has exactly two distinct parents and exactly one child.

    Args:
        net (Network): The network to check.

    Raises:
        NetworkError: On the first violated requirement, naming it.

    Returns:
        list[Node]: The nodes root first, every node before its children.
    """
    nodes = net.V()
    if not nodes:
        raise NetworkError("joint_dissimilarity: empty network")
    order = net.topological_order()                  # raises on a cycle
    roots = [v for v in nodes if net.in_degree(v) == 0]
    if len(roots) != 1:
        raise NetworkError(f"joint_dissimilarity: network has {len(roots)} roots; "
                           "a rooted network has exactly one")
    labels = [v.label for v in nodes if net.out_degree(v) == 0]
    if any(not l for l in labels):
        raise NetworkError("joint_dissimilarity: network has an unlabelled leaf")
    dup = sorted(l for l, c in Counter(labels).items() if c > 1)
    if dup:
        raise NetworkError(f"joint_dissimilarity: duplicate leaf labels {dup}")
    # The tripartition of a reticulation needs two distinct parent blocks and
    # one block below it, so parallel edges, three parents, a reticulate leaf
    # (usually a duplicated taxon merged by the parser) and a reticulation
    # with two children are all refused rather than silently scored.
    for h in nodes:
        if net.in_degree(h) < 2:
            continue
        parents = net.get_parents(h)
        if len(parents) != 2 or parents[0] is parents[1]:
            raise NetworkError(f"joint_dissimilarity: reticulation {h.label} has "
                               f"{len(parents)} parent edges; exactly two distinct "
                               "parents are required")
        if net.out_degree(h) != 1:
            raise NetworkError(f"joint_dissimilarity: reticulation {h.label} has "
                               f"{net.out_degree(h)} children; exactly one is required")
    return order


def _network_graph(net: Network) -> tuple[_Graph, dict[Node, int]]:
    """
    A validated network as a :class:`_Graph` with small-integer ids (0 is the
    root), plus the node-to-id map.

    Args:
        net (Network): The network to convert.

    Raises:
        NetworkError: See :func:`_validate`.

    Returns:
        tuple[_Graph, dict[Node, int]]: The graph and the id of every node.
    """
    order = _validate(net)
    id_of = {v: i for i, v in enumerate(order)}
    children = {id_of[v]: tuple(dict.fromkeys(id_of[c] for c in net.get_children(v)))
                for v in order}
    label = {id_of[v]: v.label for v in order if net.out_degree(v) == 0}
    return _Graph(0, children, label), id_of


def _remainder(G: _Graph, below: dict) -> _Graph:
    """
    The remainder ``T(N)``: delete the union of the blocks below all
    reticulations and, for every surviving vertex v and original edge (v, h)
    into a reticulation h, hang one marker leaf on v labelled by the taxa
    below h in the ORIGINAL network.  Taxa are relabelled ``("taxon", name)``
    and markers ``("marker", sorted taxa)`` so the two namespaces never
    collide.  Unary vertices are kept.

    Args:
        G (_Graph): The whole network.
        below (dict[int, _Graph]): Reticulation id -> the block below it.

    Returns:
        _Graph: ``T(N)``.
    """
    deleted = set().union(*(B.children.keys() for B in below.values()))
    marker = {h: ("marker", tuple(sorted(B.leaves))) for h, B in below.items()}
    children, label = {}, {}
    next_id = len(G.children)
    # Walk the survivors; a surviving vertex's child is either a survivor or a
    # reticulation (a deleted non-reticulation child would need a second
    # parent, making it a reticulation), so one marker per edge into H.
    for v in G.order():
        if v in deleted:
            continue
        kids = []
        for c in G.children[v]:
            if c in marker:
                children[next_id], label[next_id] = (), marker[c]
                kids.append(next_id)
                next_id += 1
            else:
                kids.append(c)
        children[v] = tuple(kids)
        if v in G.label:
            label[v] = ("taxon", G.label[v])
    return _Graph(G.root, children, label)


def _prepare(net: Network) -> tuple[list, _Graph, frozenset]:
    """
    Reduce a network to what the measure needs: its tripartitions
    ``(A, B, C)`` (one per reticulation), its remainder ``T(N)`` and its taxon
    set.

    Args:
        net (Network): The network.

    Raises:
        NetworkError: See :func:`_validate`.

    Returns:
        tuple[list[tuple[_Graph, _Graph, _Graph]], _Graph, frozenset[str]]:
            Tripartitions, remainder, taxa.
    """
    G, id_of = _network_graph(net)
    parents = {id_of[h]: [id_of[p] for p in net.get_parents(h)]
               for h in id_of if net.in_degree(h) >= 2}
    below = {h: _subgraph(G, h) for h in parents}
    tri = [(_parent_block(G, u, {h}), below[h], _parent_block(G, v, {h}))
           for h, (u, v) in parents.items()]
    return tri, _remainder(G, below), G.leaves


def _d_ret(t1: list, t2: list, d: Callable) -> float:
    """
    ``D_ret``: pad the shorter tripartition list with dummies to
    ``m = max(r1, r2)`` (real--dummy cost 3, dummy--dummy 0) and return the
    minimum-cost bijection value by one Hungarian solve.

    Args:
        t1 (list): Tripartitions of the first network.
        t2 (list): Tripartitions of the second network.
        d (Callable): The structure dissimilarity ``d(S, S')``.

    Returns:
        float: ``D_ret`` in ``[0, 3m]``; 0 when both networks are trees.
    """
    m = max(len(t1), len(t2))
    if m == 0:
        return 0.0
    cost = np.full((m, m), _DUMMY_COST)
    cost[len(t1):, len(t2):] = 0.0
    # delta compares the hybrid blocks and the cheaper of the two pairings of
    # the unordered parent blocks.
    for i, (A, B, C) in enumerate(t1):
        for j, (D, E, F) in enumerate(t2):
            cost[i, j] = d(B, E) + min(d(A, D) + d(C, F), d(A, F) + d(C, D))
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


# ── Public surface ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class JointDissimilarityResult:
    """
    The result of :func:`joint_dissimilarity` under one instantiation.

    Args:
        kind (str): The instantiation used, one of :data:`JOINT_KINDS`.
        r1 (int): Reticulation number of the first network.
        r2 (int): Reticulation number of the second network.
        D_ret (float): Reticulation term, raw, in ``[0, 3m]``.
        D_tree (float): Remainder term ``d(T(N1), T(N2))`` in ``[0, 1]``.
    """

    kind: str
    r1: int
    r2: int
    D_ret: float
    D_tree: float

    @property
    def m(self) -> int:
        """
        Returns:
            int: ``max(r1, r2)``, the size of the padded matching.
        """
        return max(self.r1, self.r2)

    @property
    def D_joint(self) -> float:
        """
        Returns:
            float: ``D_ret + 3 D_tree``, raw, in ``[0, 3(m + 1)]``.
        """
        return self.D_ret + 3.0 * self.D_tree

    @property
    def D_ret_normalized(self) -> float:
        """
        Returns:
            float: ``D_ret / 3m`` in ``[0, 1]``; 0 when both are trees.
        """
        return self.D_ret / (3.0 * self.m) if self.m else 0.0

    @property
    def D_joint_normalized(self) -> float:
        """
        Returns:
            float: ``D_joint / 3(m + 1)`` in ``[0, 1]``.
        """
        return self.D_joint / (3.0 * (self.m + 1))


def joint_dissimilarity(net1: Network,
                        net2: Network,
                        kind: str = "nakhleh") -> JointDissimilarityResult:
    """
    The joint reticulation-and-tree dissimilarity of two rooted phylogenetic
    networks on the same taxon set (see the module docstring for the
    definition).  Symmetric, 0 on identical networks, and a metric on the
    tripartition and remainder representations under each instantiation.

    Both networks must have one root, be acyclic, have labelled leaves with no
    label used twice, and give every reticulation exactly two distinct parents
    and one child.  Internal-node labels and branch lengths are ignored.

    Args:
        net1 (Network): First network.
        net2 (Network): Second network.
        kind (str): Structure dissimilarity: ``"leaf"`` (Jaccard of leaf
                    sets), ``"cluster"`` (multiset hardwired clusters) or
                    ``"nakhleh"`` (vertex-equivalence classes; the default).

    Raises:
        ValueError: If ``kind`` is not one of :data:`JOINT_KINDS`.
        NetworkError: If either network violates the requirements above, or
                      the two taxon sets differ.

    Returns:
        JointDissimilarityResult: ``r1``, ``r2``, ``D_ret`` and ``D_tree``, with
                            ``m``, ``D_joint`` and the normalised forms as
                            properties.
    """
    if kind not in _DSET:
        raise ValueError(f"unknown instantiation {kind!r}; choose from {JOINT_KINDS}")
    d = _DSET[kind]
    t1, T1, X1 = _prepare(net1)
    t2, T2, X2 = _prepare(net2)
    if X1 != X2:
        raise NetworkError("joint_dissimilarity requires identical leaf label sets. "
                           f"Got {sorted(X1)} vs {sorted(X2)}")
    return JointDissimilarityResult(kind, len(t1), len(t2), _d_ret(t1, t2, d), d(T1, T2))
