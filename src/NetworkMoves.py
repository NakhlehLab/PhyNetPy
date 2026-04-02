#! /usr/bin/env python
# -*- coding: utf-8 -*-
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

"""
NetworkMoves -- Topology-altering operations for phylogenetic networks.

This module provides the core set of "moves" used during phylogenetic
network search (e.g. MCMC or hill-climbing). Each move modifies a
:class:`Network` **in place** and is designed to be reversible so that a
search algorithm can propose a candidate topology, score it, and accept
or reject it.

Available moves
---------------
add_hybrid        Increase network complexity by inserting a reticulation
                  (hybrid) edge between two existing edges, raising the
                  network level by one. Also supports creating "bubble"
                  edges (genome duplication) when both edges are the same.

remove_hybrid     Decrease network complexity by removing a reticulation
                  edge and suppressing the resulting degree-2 nodes,
                  lowering the network level by one.

nni               Nearest-neighbor interchange -- swap two subtrees across
                  an internal edge to explore alternative topologies at the
                  same network level.

node_height_change  Move a node up or down the time axis while keeping the
                    tree ultrametric. Optionally shifts the entire subtree
                    to preserve branch-length ratios.

spr               Subtree prune-and-regraft -- detach a subtree and
                  reattach it elsewhere. Currently restricted to level-0
                  networks (trees).

permute_leaves    Enumerate every relabeling of the leaf taxa, yielding a
                  new network copy for each permutation.
"""

from .Network import Network, Node, Edge, NetworkError
from .GraphUtils import level
import random
import warnings
from itertools import permutations
from typing import Iterator


# ---------------------------------------------------------------------------
#  Topology-changing moves
# ---------------------------------------------------------------------------

def add_hybrid(net: Network,
               source: Edge, 
               destination: Edge, 
               t_src: float = None, 
               t_dest: float = None) -> None:
    """Add a hybrid (reticulation) edge between two existing edges.

    This is the primary move for *increasing* network complexity. A new
    directed edge is created that represents a hybridization or
    introgression event between two lineages.

    How it works (normal case, source != destination)::

        BEFORE                        AFTER
        a           x                a           x
        |           |                |           |
        |           |                n1 -------> n2   (new hybrid edge)
        |           |                |           |
        b           y                b           y

    Two fresh internal nodes (n1, n2) are inserted by splitting the
    *source* and *destination* edges respectively. n2 is flagged as a
    reticulation node, and a directed edge n1 -> n2 is added.

    **Bubble (genome-duplication) case** -- when *source* and
    *destination* are the same edge, two parallel edges with equal
    inheritance probabilities (gamma = 0.5) are created between two new
    internal nodes, modeling a whole-genome duplication event.

    Args:
        net:     The network to modify **in place**.
        source:  The edge that will donate the new outgoing lineage.
        destination: The edge that will receive the incoming hybrid lineage.
        t_src:   Optional divergence time for the new source-side node.
        t_dest:  Optional divergence time for the new reticulation node.
    """
    if source == destination:
        # --- Bubble / genome-duplication special case ---
        warnings.warn("Source and destination edges are the same. Bubble edge will be created.")
        
        n1: Node = source.src   # existing parent (top of the original edge)
        n2: Node = source.dest  # existing child  (bottom of the original edge)
        n3: Node = net.add_uid_node()  # new upper internal node
        n4: Node = net.add_uid_node()  # new lower internal node
        
        if t_src is not None:
            n3.set_time(t_src)
        if t_dest is not None:
            n4.set_time(t_dest)
        
        # Replace the single edge with a diamond/bubble: n1 -> n3 => n4 -> n2
        net.remove_edge(source)
        
        # Two parallel edges between n3 and n4, each carrying half the
        # inheritance probability. Tags 'left'/'right' let downstream code
        # distinguish the two copies.
        bubble_right: Edge = Edge(n3, n4, gamma=.5, tag='right')
        bubble_left:  Edge = Edge(n3, n4, gamma=.5, tag='left')
        top_to_n3:    Edge = Edge(n1, n3)
        n4_to_bot:    Edge = Edge(n4, n2)
        
        net.add_edges([bubble_right, bubble_left, top_to_n3, n4_to_bot])
    else:
        # --- Standard hybrid-edge insertion ---
        a: Node = source.src        # parent of source edge
        b: Node = source.dest       # child  of source edge
        x: Node = destination.src   # parent of destination edge
        y: Node = destination.dest  # child  of destination edge
        
        n1: Node = net.add_uid_node()  # splits the source edge
        n2: Node = net.add_uid_node()  # splits the destination edge
        n2.set_is_reticulation(True)   # mark as reticulation / hybrid node
        
        if t_src is not None:
            n1.set_time(t_src)
        if t_dest is not None:
            n2.set_time(t_dest)
        
        # Remove the two original edges that are being subdivided
        net.remove_edge(source)
        net.remove_edge(destination)
        
        # Wire up: a->n1->b, x->n2->y, and the new hybrid edge n1->n2
        net.add_edges([Edge(a, n1),
                       Edge(n1, b), 
                       Edge(x, n2), 
                       Edge(n2, y),
                       Edge(n1, n2)])
        
def remove_hybrid(net: Network, 
                  hybrid_edge: Edge) -> None:
    """Remove a hybrid (reticulation) edge and clean up the two degree-2 nodes.

    This is the inverse of :func:`add_hybrid`. It lowers the network level
    by one by deleting the hybrid edge and the two internal nodes that were
    created when the hybrid was originally added.

    Expected local topology around the hybrid edge::

        a           x
        |           |
        v           v
        n1 -------> n2   <-- hybrid_edge (n1 -> n2, n2 is reticulation)
        |           |
        v           v
        b           y

    After removal, n1 and n2 are deleted and replaced by two simple edges::

        a           x
        |           |
        v           v
        b           y

    **Preconditions** -- the move only makes sense when n1 and n2 were
    originally inserted by :func:`add_hybrid`, meaning:

    * n1 has exactly one parent (a) and one non-hybrid child (b).
    * n2 has exactly one other parent (x) and one child (y).

    Args:
        net:         The network to modify **in place**.
        hybrid_edge: The reticulation edge to remove. Its destination (n2)
                     must be flagged as a reticulation node.

    Raises:
        NetworkError: If the edge is not a valid hybrid edge, or if the
            surrounding topology does not allow clean removal (i.e. the
            nodes don't satisfy the degree constraints listed above).
    """
    # Guard: the edge must actually point to a reticulation node
    if not hybrid_edge.dest.is_reticulation():
        raise NetworkError("Given edge parameter is not a hybrid edge!")

    # Guard: we don't support removing edges between two reticulations
    if hybrid_edge.src.is_reticulation():
        raise NetworkError("The source of the given edge is a reticulation!")
    
    n1: Node = hybrid_edge.src   # non-reticulation internal node
    n2: Node = hybrid_edge.dest  # reticulation node
    
    # Gather the neighborhood around n1 and n2, excluding the hybrid edge
    parents_n1 = net.get_parents(n1)
    children_n1 = [node for node in net.get_children(n1) if node != n2]
    other_parents_n2 = [node for node in net.get_parents(n2) if node != n1]
    children_n2 = net.get_children(n2)
    
    # Verify that both nodes are simple pass-through (degree-2) once the
    # hybrid edge is conceptually removed -- otherwise we can't cleanly
    # suppress them without losing topology.
    if len(parents_n1) != 1 or len(children_n1) != 1:
        raise NetworkError(
            "Hybrid edge source must have exactly one parent and one "
            "non-hybrid child for clean removal."
        )
    if len(other_parents_n2) != 1 or len(children_n2) != 1:
        raise NetworkError(
            "Hybrid edge destination must have exactly one other parent "
            "and one child for clean removal."
        )
    
    a: Node = parents_n1[0]        # grandparent on the source side
    b: Node = children_n1[0]       # grandchild  on the source side
    x: Node = other_parents_n2[0]  # other parent of the reticulation
    y: Node = children_n2[0]       # child of the reticulation

    # Delete the two internal nodes (and all their incident edges)
    net.remove_nodes(n2)
    net.remove_nodes(n1)

    # Reconnect: a->b restores the source lineage, x->y restores the dest
    net.add_edges([Edge(a, b), Edge(x, y)])       

def nni(net: Network) -> None:
    """Perform a random nearest-neighbor interchange (NNI) on the network.

    NNI is the workhorse topology-exploration move. It swaps two subtrees
    across an internal edge, producing a "neighboring" topology that differs
    by exactly one bipartition. Because it never adds or removes
    reticulations, the network level is unchanged.

    Diagram of the swap::

        BEFORE              AFTER
          a                   a
         / \\                 / \\
        c   b               d   b
           / \\                 / \\
          d   ...             c   ...

    A child *c* of *a* (that is not *b*) and a child *d* of *b* are chosen
    uniformly at random and exchanged. All edge metadata (length, gamma,
    weight, tag) travel with their respective subtrees.

    **Eligible edges** -- only internal edges whose endpoints are both
    non-reticulation and non-leaf are considered, so the move never
    disrupts reticulation structure.

    Args:
        net: The network to modify **in place**.

    Raises:
        NetworkError: If no eligible internal edge exists, or if the
            randomly selected edge lacks sufficient children on both sides.
    """
    leaves = set(net.get_leaves())

    # An "internal edge" for NNI purposes: neither endpoint is a leaf or
    # reticulation, so both sides have freely-swappable children.
    internal_edges = [
        e for e in net.E()
        if e.dest not in leaves
        and not e.src.is_reticulation()
        and not e.dest.is_reticulation()
    ]
    if not internal_edges:
        raise NetworkError("No internal edges available for NNI.")
    
    # Pick the pivot edge at random
    edge = random.choice(internal_edges)
    a, b = edge.src, edge.dest
    
    # Children of a (excluding b, which is the other end of the pivot)
    a_neighbors = [node for node in net.get_children(a) if node != b]
    # All children of b are candidates on the lower side
    b_neighbors = [node for node in net.get_children(b)]
    
    if not a_neighbors or not b_neighbors:
        raise NetworkError("Not enough neighbors for NNI.")
    
    # Uniformly pick one subtree from each side to swap
    c = random.choice(a_neighbors)
    d = random.choice(b_neighbors)
    
    # Snapshot edge attributes before removal so we can reattach them
    e_ac = net.get_edge(a, c)
    e_bd = net.get_edge(b, d)
    
    len_ac, gamma_ac = e_ac.get_length(), e_ac.get_gamma()
    wt_ac, tag_ac = e_ac.get_weight(), e_ac.get_tag()
    len_bd, gamma_bd = e_bd.get_length(), e_bd.get_gamma()
    wt_bd, tag_bd = e_bd.get_weight(), e_bd.get_tag()
    
    # Detach the two subtrees from their current parents
    net.remove_edge(e_ac)
    net.remove_edge(e_bd)
    
    # Re-attach them in swapped positions, preserving original edge metadata
    net.add_edges([
        Edge(a, d, length=len_bd, gamma=gamma_bd, weight=wt_bd, tag=tag_bd),
        Edge(b, c, length=len_ac, gamma=gamma_ac, weight=wt_ac, tag=tag_ac),
    ])

def node_height_change(n: Node, 
                       net: Network, 
                       height: float,
                       extend: bool = False) -> None:
    """Change the divergence time (height) of an internal node.

    In a time-consistent (ultrametric) network, every node has a "time"
    value that increases from root toward the tips. This move slides a
    single node up or down the time axis while preserving the parent >
    child ordering on both sides.

    **Simple mode** (``extend=False``, default):
        Only the time of *n* is changed. Parent and child times are
        untouched, so the branch lengths into and out of *n* change
        accordingly. The new height must fall strictly between the
        nearest parent time and the nearest child time.

    **Extend mode** (``extend=True``):
        The node *and all of its immediate children* are shifted by the
        same delta, preserving the branch lengths between *n* and its
        children. This is useful when you want to move a speciation
        event without distorting the subtree below it. The move is
        rejected if any shifted child would collide with or pass its
        own grandchildren in time.

    Args:
        n:      The internal node whose height should change.
        net:    The network containing *n*.
        height: The proposed new time value for *n*.
        extend: If True, shift children by the same delta to keep
                branch lengths within the subtree constant.

    Raises:
        NetworkError: If *n* is a root or leaf (no parents or children),
            or if the proposed height violates time-ordering constraints.
    """
    parents = net.get_parents(n)
    children = net.get_children(n)

    if not parents or not children:
        raise NetworkError("Node must have both parents and children.")

    # The valid interval for the new height is open:
    #   (max parent time, min child time)
    max_parent_height = max(parent.get_time() for parent in parents)
    min_child_height = min(child.get_time() for child in children)

    if not (max_parent_height < height < min_child_height):
        raise NetworkError("New height is out of bounds.")

    if extend:
        # How far we're moving n -- children will shift by the same amount
        delta = height - n.get_time()

        # Pre-check: make sure every child can absorb the shift without
        # crashing into its own children (grandchildren of n).
        for child in children:
            grandchildren = net.get_children(child)
            if grandchildren:
                min_gc_height = min(gc.get_time() for gc in grandchildren)
                if child.get_time() + delta >= min_gc_height:
                    raise NetworkError(
                        "Extending subtree would violate time ordering "
                        f"for child {child.label}."
                    )

        # All checks passed -- apply the shift to every child
        for child in children:
            child.set_time(child.get_time() + delta)

    n.set_time(height)

# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _suppress_chain_node(net: Network, x: Node) -> None:
    """Suppress a degree-2 ("chain") node by merging its two incident edges.

    After certain moves (e.g. pruning in SPR), a node may be left with
    exactly one parent and one child, making it a redundant pass-through.
    This helper removes *x* and replaces the two edges (parent->x, x->child)
    with a single direct edge (parent->child). Branch lengths and weights
    are summed; gamma is inherited from the downstream edge when available.

    If *x* does **not** have in-degree 1 and out-degree 1, the function
    is a no-op (safe to call speculatively).

    Args:
        net: The network to modify **in place**.
        x:   The candidate node to suppress.
    """
    if net.in_degree(x) != 1 or net.out_degree(x) != 1:
        return  # not a chain node -- nothing to do

    p = net.get_parents(x)[0]   # sole parent
    c = net.get_children(x)[0]  # sole child

    e_up = net.get_edge(p, x)  # parent -> x
    e_dn = net.get_edge(x, c)  # x -> child

    # Merge edge attributes: lengths and weights sum, gamma prefers downstream
    new_len = e_up.get_length() + e_dn.get_length()
    new_wt = e_up.get_weight() + e_dn.get_weight()
    new_gamma = e_dn.get_gamma() if e_dn.get_gamma() else e_up.get_gamma()

    # Remove the two edges and the node, then add the merged bypass edge
    net.remove_edge(e_up)
    net.remove_edge(e_dn)
    net.remove_nodes(x)
    net.add_edges(Edge(p, c, length=new_len, weight=new_wt, gamma=new_gamma))


def spr(net: Network) -> None:
    """Perform a random subtree prune-and-regraft (SPR) on a tree.

    SPR is a larger-radius topology move than NNI: it detaches an entire
    subtree and reattaches it at a different location in the tree, allowing
    the search to jump further across tree space in a single step.

    **Restriction:** currently limited to level-0 networks (i.e. trees
    with no reticulations). This is checked up front.

    Algorithm outline::

        1. Pick a random "prune edge" (u, v) where v is not the root and
           at least one edge exists entirely outside the subtree rooted
           at v (so there is somewhere to reattach).
        2. Remove the prune edge (u, v).
        3. If u is now a degree-2 pass-through node, suppress it by
           merging its parent and child edges (see _suppress_chain_node).
        4. Pick a random "host edge" (a, b) that lies completely outside
           the pruned subtree.
        5. Insert a new internal node k on (a, b), splitting it into
           (a, k) and (k, b), and attach the pruned subtree as (k, v).

    The host edge's length is split randomly between (a, k) and (k, b),
    and the original prune-edge length is assigned to (k, v). The result
    is always a bifurcating tree with the same leaf set.

    Args:
        net: The tree to modify **in place**.

    Raises:
        NetworkError: If the network is not level-0, or if no valid prune
            or reattachment edge can be found.
    """
    root = net.root()
    if level(net) != 0:
        raise NetworkError("SPR requires a level-0 network (tree).")

    def _has_host_outside(subtree: set[Node]) -> bool:
        """Check whether at least one edge lies entirely outside *subtree*."""
        return any(
            e.src not in subtree and e.dest not in subtree for e in net.E()
        )

    # --- Step 1: Find edges eligible for pruning ---
    # We can't prune at the root (nothing above it), and we need at least
    # one reattachment site outside the pruned subtree.
    prune_candidates = [
        e
        for e in net.E()
        if e.src is not root
        and e.dest is not root
        and _has_host_outside(net.get_subtree_at(e.dest))
    ]
    if not prune_candidates:
        raise NetworkError(
            "No valid SPR prune edge (tree may be too small or degenerate)."
        )

    # --- Step 2: Prune the subtree ---
    prune_edge = random.choice(prune_candidates)
    u, v = prune_edge.src, prune_edge.dest
    len_uv = prune_edge.get_length()
    subtree_nodes = net.get_subtree_at(v)

    net.remove_edge(prune_edge)

    # --- Step 3: Clean up the stump (suppress u if it became degree-2) ---
    _suppress_chain_node(net, u)

    # --- Step 4: Find edges eligible for reattachment ---
    # Must be completely disjoint from the pruned subtree
    reattach_candidates = [
        e
        for e in net.E()
        if e.src not in subtree_nodes and e.dest not in subtree_nodes
    ]
    if not reattach_candidates:
        raise NetworkError("No SPR reattachment edge disjoint from the pruned subtree.")

    # --- Step 5: Regraft ---
    host = random.choice(reattach_candidates)
    a, b = host.src, host.dest
    len_ab = host.get_length()

    net.remove_edge(host)

    # Split the host edge at a uniformly random point
    r = random.random()
    len_ak = r * len_ab
    len_kb = (1.0 - r) * len_ab

    # Insert a new internal node k that connects a->k->b and k->v
    k = net.add_uid_node()
    net.add_edges(
        [
            Edge(a, k, length=len_ak),   # upper portion of old host edge
            Edge(k, b, length=len_kb),   # lower portion of old host edge
            Edge(k, v, length=len_uv),   # reattach pruned subtree
        ]
    )


def permute_leaves(net: Network) -> Iterator[Network]:
    """Yield every possible relabeling of the leaf taxa on the network.

    This is useful for exhaustive searches over small networks: given a
    fixed internal topology, the function produces one network copy for
    every permutation of the leaf labels across the existing pendant
    (leaf-edge) positions.

    For *n* leaves this yields *n!* networks, so use with caution on
    networks with more than ~10 leaves.

    How it works:

    1. Record the canonical leaf ordering and each leaf's parent node.
    2. For each permutation of the leaf list, make a deep copy of the
       network, detach all leaves from their parents, and rewire them
       according to the permutation -- leaf ``perm[i]`` gets attached
       to the parent that originally held leaf ``i``.
    3. Edge metadata (length, gamma, weight, tag) travel with the leaf
       they were originally attached to, not with the slot.

    The original network is **never modified**.

    Args:
        net: The template network to permute.

    Yields:
        A deep copy of *net* with leaves rewired according to one
        permutation.

    Raises:
        NetworkError: If any leaf has zero or multiple parents (each leaf
            must have exactly one parent for the rewiring to be well-defined).
    """
    leaves_orig = list(net.get_leaves())
    if not leaves_orig:
        return

    # Record each leaf's unique parent in the original network
    parents_orig: list[Node] = []
    for lf in leaves_orig:
        pars = net.get_parents(lf)
        if len(pars) != 1:
            raise NetworkError(
                "permute_leaves expects each leaf to have exactly one parent."
            )
        parents_orig.append(pars[0])

    for perm in permutations(leaves_orig):
        # Deep-copy the network; old_new maps original nodes -> copy nodes
        copy_net, old_new = net.copy()
        leaves_copy = [old_new[lf] for lf in leaves_orig]
        parents_copy = [old_new[p] for p in parents_orig]
        perm_copy = tuple(old_new[lf] for lf in perm)

        # Snapshot every leaf's pendant edge (the edge connecting it to its
        # parent) so we can reuse its metadata after detaching.
        pendant_ref: dict[Node, Edge] = {}
        for lf in leaves_copy:
            p = copy_net.get_parents(lf)[0]
            pendant_ref[lf] = copy_net.get_edge(p, lf)

        # Detach all leaves from their current parents
        for lf in leaves_copy:
            for par in list(copy_net.get_parents(lf)):
                copy_net.remove_edge(copy_net.get_edge(par, lf))

        # Rewire: leaf perm[i] goes to slot i (the parent that originally
        # held leaf i). Metadata travels with the leaf, not the slot.
        for i, leaf in enumerate(perm_copy):
            ref = pendant_ref[leaf]
            copy_net.add_edges(
                Edge(
                    parents_copy[i],
                    leaf,
                    length=ref.get_length(),
                    gamma=ref.get_gamma(),
                    weight=ref.get_weight(),
                    tag=ref.get_tag(),
                )
            )
        yield copy_net

