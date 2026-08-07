#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author : Mark Kessler
Last Edit : 5/12/26
First Included in Version : 1.0.0

GraphUtils -- topology analysis, manipulation, and rendering utilities
for phylogenetic networks.

This module is the catch-all for graph-theoretic operations on
:class:`~.Network.Network` objects, including:

* topology summaries (network level, blob/biconnected-component
  decomposition, tree-child / tree-based predicates);
* enumeration of displayed trees and subnetworks
  (``get_all_subtrees``, ``subnet_given_leaves``,
  ``induced_subnetwork_by_taxa``);
* distance metrics on networks
  (mu-distance, hardwired / softwired / Robinson--Foulds /
  tripartition / displayed-tree / average-path / weighted-APD);
* convenience helpers for branch lengths, clades, and ASCII rendering.

Docs   - [x]
Tests  - [ ]
Design - [ ]
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

from .Network import Network, Node, Edge, NetworkError
from collections import Counter, deque
from itertools import product as _product
from typing import Dict, Set, Union
import heapq
import warnings

#: Public surface of this module.  Declared explicitly so that
#: ``from .GraphUtils import *`` exports only these helpers and not the
#: ``Network`` names re-imported above.  The reticulation-comparison metrics
#: appended at the bottom of this file are listed here too.
__all__ = [
    # Structural edits
    "add_hybrid",
    # Topology extraction / sub-structures
    "subnet_given_leaves",
    "induced_subnetwork_by_taxa",
    "get_all_subtrees",
    "get_all_clusters",
    "network_clusters",
    "dominant_tree",
    "count_displayed_trees",
    # Decomposition
    "blobs",
    "tree_of_blobs",
    "bridges_and_articulations",
    # Descriptive properties
    "level",
    "is_tree",
    "count_reticulations",
    "is_isomorphic",
    # Distances / dissimilarities
    "hardwired_cluster_distance",
    "softwired_cluster_distance",
    "displayed_tree_distance",
    "robinson_foulds_distance",
    "tripartition_distance",
    "mu_distance",
    "pairwise_leaf_distance",
    "average_path_distance",
    "weighted_average_path_distance",
    # Reticulation-aware comparison (re-exported from ReticulationComparison)
    "JACCARD",
    "SYMMETRIC",
    "ReticulationTripartition",
    "NetworkComparison",
    "reticulation_tripartitions",
    "set_distance",
    "tripartition_dissimilarity",
    "nakhleh_metric",
    "reticulation_dissimilarity",
    "reticulation_precision_recall",
    "combined_dissimilarity",
    "compare_networks",
    # Rendering
    "ascii",
]

def get_all_clusters(net : Network,
                     include_trivial : bool = False)\
                     -> set[frozenset[Node]]:
    """
    Compile the set of clusters that make up this network.
    Ie: for a graph ((A, B)C, D); , set of all clusters is {(A,B), (A,B,C)}.
    
    A cluster is the set of leaves below some non-root node. Clusters of
    size 1 can optionally be included.

    Note that clusters are sets of :class:`~phynetpy.Network.Node` objects
    belonging to *this* network, so they cannot be compared across networks;
    use :func:`network_clusters` for that.

    Args:
        net (Network): the network to operate on
        include_trivial (bool): If set to True, includes clusters of size 1. 
                                Defaults to False.

    Returns:
        set: A set of all clusters in this graph, each a frozenset of Nodes.
    """
    root = net.root()

    # leaf_descendants_all memoizes over the DAG, so a subnetwork shared by
    # several reticulation parents is walked once rather than once per path.
    cluster_set = {frozenset(below)
                   for node, below in net.leaf_descendants_all().items()
                   if node is not root and len(below) > 1}

    if include_trivial:
        cluster_set |= {frozenset([leaf]) for leaf in net.get_leaves()}

    return cluster_set

def _induced_subnetwork(net : Network, targets : set[Node]) -> Network:
    """
    Copy the part of 'net' at or below MRCA(targets) that leads to a target.

    Shared core of :func:`subnet_given_leaves` and
    :func:`induced_subnetwork_by_taxa`; the caller decides which
    :meth:`Network.clean` passes to run on the result.

    Each retained node is enqueued exactly once. A reticulation reachable
    from two retained parents would otherwise have its whole subtree walked
    once per parent, and since Edge equality is by identity, every edge below
    it would be copied twice into a spurious bubble.

    Args:
        net (Network): A network object.
        targets (set[Node]): The leaves the subnetwork must retain.
    Returns:
        Network: An uncleaned copy of the induced subnetwork.
    """
    sub_root = net.mrca(targets)
    root_copy = sub_root.copy()

    sub : Network = Network()
    sub.add_nodes(root_copy)
    old_new : dict[Node, Node] = {sub_root : root_copy}
    q : deque[Node] = deque([sub_root])

    while q:
        cur = q.popleft()

        for child in net.get_children(cur):
            # ``leaf_descendants`` of a leaf is the leaf itself, so this also
            # covers the case of `child` being a target.
            if not net.leaf_descendants(child) & targets:
                continue

            if child not in old_new:
                old_new[child] = child.copy()
                sub.add_nodes(old_new[child])
                q.append(child)

            old_edge = net.get_edge(cur, child)
            new_edge = Edge(old_new[cur], old_new[child])
            new_edge.set_gamma(old_edge.get_gamma())
            new_edge.set_length(old_edge.get_length())
            sub.add_edges(new_edge)

    return sub

def subnet_given_leaves(net : Network, leaf_set : list[Node]) -> Network:
    """
    Compute the minimally sized subnetwork of a network such that the leaf set 
    of the subnetwork is a subset of the original network's leaf set.

    Args:
        net (Network): A network
        leaf_set (list[Node]): A set of leaf nodes of the given network

    Returns:
        Network: A new Network object with node and edge copies of the original.
    """
    subnet = _induced_subnetwork(net, set(leaf_set))

    if len(subnet.V()) > 1:
        subnet.clean()

    return subnet
    
def _displayed_trees_with_probs(net : Network) -> list[tuple[Network, float]]:
    """
    Enumerate every displayed tree together with its probability.

    A displayed tree is obtained by keeping exactly one in-edge at each
    reticulation; its probability is the product of the gammas of the kept
    edges. When gamma is unset for an edge, a uniform 1/k weight is assumed
    (k = in-degree of the reticulation node).

    Reticulations and their in-edges are both sorted by label so that the
    enumeration order is deterministic across runs.

    Args:
        net (Network): A phylogenetic network.
    Returns:
        list[tuple[Network, float]]: (tree copy, probability) pairs.
    """
    retics = sorted(
        (node for node in net.V() if node.is_reticulation()),
        key=lambda n: n.label,
    )
    if not retics:
        copy, _ = net.copy()
        return [(copy, 1.0)]

    retic_in_edges = [
        sorted(list(net.in_edges(node)), key=lambda e: e.src.label)
        for node in retics
    ]

    results : list[tuple[Network, float]] = []
    for combo in _product(*retic_in_edges):
        tree, old_new = net.copy()
        prob = 1.0
        for retic_node, kept_edge in zip(retics, combo):
            gamma = kept_edge.get_gamma()
            if gamma is not None and gamma > 0:
                prob *= gamma
            else:
                k = len(net.in_edges(retic_node))
                prob *= 1.0 / k if k > 0 else 1.0
            for e in net.in_edges(retic_node):
                if e is not kept_edge:
                    tree.remove_edge([old_new[e.src], old_new[e.dest]])
        tree.clean()
        results.append((tree, prob))

    return results

def get_all_subtrees(net : Network) -> list[Network]:
    """
    Generate all possible trees that can be derived from the given network by
    removing hybrid edges and creating copies with subtrees that start at each 
    non-reticulation node.

    Args:
        net (Network): A network object
    Returns:
        list[Network]: A list of network objects, each representing a tree that
                       is derived from the original network.
    """
    return [tree for tree, _ in _displayed_trees_with_probs(net)]

def dominant_tree(net : Network) -> Network:
    """
    Generate the dominant tree from a given network by retaining only the 
    reticulation edges with the highest inheritance probability and removing 
    all other reticulation edges.

    Args:
        net (Network): A network object
    Returns:
        Network: A new network object representing the dominant tree derived 
                 from the original network.
    """
    dom : Network = Network()
    
    edges_2_remove = []
    old_new_node_map = {}
    
    #Only include reticulation edges that are the maximum inheritance prob
    for node in [retic for retic in net.V() if retic.is_reticulation()]:
        retic_edges = [e for e in net.in_edges(node)]
        # Sort by gamma ascending; keep only the maximum, remove the rest
        retic_edges.sort(key=lambda e: e.get_gamma())
        edges_2_remove.extend(retic_edges[:-1])
    
    #Add all nodes from original network
    for node in net.V():
        new_node = node.copy()
        dom.add_nodes(new_node)
        old_new_node_map[node] = new_node
    
    #Add only dominant reticulation edges and all other normal edges
    for edge in net.E():
        if edge not in edges_2_remove:
            new_src = old_new_node_map[edge.src]
            new_dest = old_new_node_map[edge.dest]
            dom.add_edges(edge.copy(new_src, new_dest))
      
    #Clean artifacts created by removing some of the retic edges      
    dom.clean()
    
    return dom

def count_reticulations(net: Network) -> int:
    """
    Count the number of reticulation nodes in a network.

    Args:
        net (Network): A network object.

    Returns:
        int: Number of nodes with indegree >= 2 (reticulations by flag).
    """
    return sum(1 for n in net.V() if n.is_reticulation())

def add_hybrid(net : Network,
               source : Edge,
               destination : Edge,
               t_src : float = None,
               t_dest : float = None) -> None:
    """
    Add a hybrid (reticulation) edge between two existing edges, raising the
    network level by one.

    This is a structural edit, not a Metropolis-Hastings proposal: it mutates
    ``net`` in place and reports no Hastings ratio. Samplers should use the
    :class:`~.ModelMove.AddReticulation` move instead, which carries the
    reversible-jump proposal math.

    How it works (normal case, ``source != destination``)::

        BEFORE                        AFTER
        a           x                a           x
        |           |                |           |
        |           |                n1 -------> n2   (new hybrid edge)
        |           |                |           |
        b           y                b           y

    Two fresh internal nodes (n1, n2) are inserted by splitting the ``source``
    and ``destination`` edges respectively. n2 is flagged as a reticulation
    node, and a directed edge n1 -> n2 is added.

    When ``source`` and ``destination`` are the *same* edge, two parallel edges
    with equal inheritance probabilities (gamma = 0.5) are created between two
    new internal nodes, modeling a whole-genome duplication ("bubble") event.

    Args:
        net (Network): The network to modify in place.
        source (Edge): The edge that will donate the new outgoing lineage.
        destination (Edge): The edge that will receive the incoming hybrid
                            lineage.
        t_src (float, optional): Divergence time for the new source-side node.
                                 Defaults to None.
        t_dest (float, optional): Divergence time for the new reticulation
                                  node. Defaults to None.
    Returns:
        N/A
    """
    if source == destination:
        # --- Bubble / genome-duplication special case ---
        warnings.warn("Source and destination edges are the same. Bubble edge "
                      "will be created.")
        
        n1 : Node = source.src   # existing parent (top of the original edge)
        n2 : Node = source.dest  # existing child  (bottom of the original edge)
        n3 : Node = net.add_uid_node()  # new upper internal node
        n4 : Node = net.add_uid_node()  # new lower internal node
        
        if t_src is not None:
            n3.set_time(t_src)
        if t_dest is not None:
            n4.set_time(t_dest)
        
        # Replace the single edge with a diamond/bubble: n1 -> n3 => n4 -> n2
        net.remove_edge(source)
        
        # Two parallel edges between n3 and n4, each carrying half the
        # inheritance probability. Tags 'left'/'right' let downstream code
        # distinguish the two copies.
        bubble_right : Edge = Edge(n3, n4, gamma = .5, tag = 'right')
        bubble_left : Edge = Edge(n3, n4, gamma = .5, tag = 'left')
        top_to_n3 : Edge = Edge(n1, n3)
        n4_to_bot : Edge = Edge(n4, n2)
        
        net.add_edges([bubble_right, bubble_left, top_to_n3, n4_to_bot])
    else:
        # --- Standard hybrid-edge insertion ---
        a : Node = source.src        # parent of source edge
        b : Node = source.dest       # child  of source edge
        x : Node = destination.src   # parent of destination edge
        y : Node = destination.dest  # child  of destination edge
        
        n1 : Node = net.add_uid_node()  # splits the source edge
        n2 : Node = net.add_uid_node()  # splits the destination edge
        n2.set_is_reticulation(True)    # mark as reticulation / hybrid node
        
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

def _valid_network_degrees(net : Network) -> bool:
    """
    Check the degree invariants a rooted phylogenetic network must satisfy.

    Unlike :func:`validate_binary`, this tolerates polytomies (internal
    out-degree >= 2 rather than exactly 2), insists on a *single* root, and
    reports one boolean instead of a violation map.  Acyclicity is not
    checked here -- callers pair this with :meth:`Network.is_acyclic`.

    Args:
        net (Network): A network object.

    Returns:
        bool: True when there is exactly one root (in-degree 0, out-degree
              >= 2), every leaf has in-degree 1, every reticulation has
              in-degree 2 and out-degree 1, and every other internal node
              has in-degree 1 and out-degree >= 2.
    """
    root_count = 0
    for node in net.V():
        indeg = net.in_degree(node)
        outdeg = net.out_degree(node)
        if indeg == 0:
            root_count += 1
            if outdeg < 2:
                return False
        elif outdeg == 0:
            if indeg != 1:
                return False
        elif node.is_reticulation():
            if indeg != 2 or outdeg != 1:
                return False
        else:
            if indeg != 1 or outdeg < 2:
                return False
    return root_count == 1

def pairwise_leaf_distance(net: Network,
                           use_branch_lengths: bool = True
                           ) -> dict[tuple[str, str], float]:
    """
    Compute pairwise distances between leaves on the underlying undirected graph.

    If use_branch_lengths is True, sum edge lengths; otherwise, use unit weights.

    Args:
        net (Network): A network object.
        use_branch_lengths (bool): Whether to sum edge lengths.

    Returns:
        dict[tuple[str, str], float]: Map from sorted (leaf_i, leaf_j) to distance.
    """
    # Build undirected weighted adjacency
    adj: Dict[Node, list[tuple[Node, float]]] = {n: [] for n in net.V()}
    for e in net.E():
        w = e.get_length() if use_branch_lengths else 1.0
        adj[e.src].append((e.dest, w))
        adj[e.dest].append((e.src, w))

    leaves = net.get_leaves()
    name_dist: dict[tuple[str, str], float] = {}

    def dijkstra(start: Node) -> Dict[Node, float]:
        dist: Dict[Node, float] = {n: float("inf") for n in net.V()}
        dist[start] = 0.0
        heap: list[tuple[float, Node]] = [(0.0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    for i in range(len(leaves)):
        s = leaves[i]
        dist_map = dijkstra(s)
        for j in range(i + 1, len(leaves)):
            t = leaves[j]
            key = tuple(sorted([s.label, t.label]))  # type: ignore
            name_dist[key] = dist_map[t]

    return name_dist

def _undirected_adjacency(net : Network) -> Dict[Node, list[Node]]:
    """
    Build neighbour lists for the undirected view of a directed network.

    Parallel edges collapse to a single neighbour entry, so that a bubble does
    not read as a cycle to the Tarjan traversals that consume this.

    Args:
        net (Network): A network object.
    Returns:
        Dict[Node, list[Node]]: Each node mapped to its distinct neighbours,
                                in first-encountered order.
    """
    adj : Dict[Node, list[Node]] = {n: [] for n in net.V()}
    seen : Dict[Node, Set[Node]] = {n: set() for n in net.V()}

    for e in net.E():
        if e.dest not in seen[e.src]:
            seen[e.src].add(e.dest)
            adj[e.src].append(e.dest)
        if e.src not in seen[e.dest]:
            seen[e.dest].add(e.src)
            adj[e.dest].append(e.src)

    return adj

def bridges_and_articulations(net: Network) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Compute bridges and articulation points on the underlying undirected graph
    using Tarjan's algorithm.

    Args:
        net (Network): A network object.

    Returns:
        tuple[list[tuple[str, str]], list[str]]:
            A pair (bridges, articulations) where bridges are (u, v) tuples
            with u < v by node label, and articulations are node labels.
    """
    adj = _undirected_adjacency(net)

    # Tarjan's algorithm state:
    #   disc[u]   – discovery time of node u
    #   low[u]    – lowest disc reachable from the subtree rooted at u
    #   parent[u] – DFS-tree parent of u (None for DFS roots)
    time = 0
    disc: Dict[Node, int] = {}
    low: Dict[Node, int] = {}
    parent: Dict[Node, Union[Node, None]] = {n: None for n in net.V()}
    visited: Set[Node] = set()
    bridges: Set[tuple[str, str]] = set()
    arts: Set[str] = set()

    def dfs(u: Node) -> None:
        nonlocal time
        visited.add(u)
        disc[u] = time
        low[u] = time
        time += 1
        child_count = 0
        is_art = False

        for v in adj[u]:
            if v not in visited:
                parent[v] = u
                child_count += 1
                dfs(v)
                # Propagate the lowest-reachable discovery time upward
                low[u] = min(low[u], low[v])
                # Bridge: no back-edge from v's subtree reaches u or above
                if low[v] > disc[u]:
                    bridges.add(tuple(sorted((u.label, v.label))))  # type: ignore
                # Articulation (non-root): removing u disconnects v's subtree
                if parent[u] is not None and low[v] >= disc[u]:
                    is_art = True
            elif v != parent[u]:
                # Back edge: update low-link to reflect the cycle
                low[u] = min(low[u], disc[v])

        # A DFS root is an articulation point iff it has >1 DFS children
        if parent[u] is None and child_count > 1:
            is_art = True
        if is_art:
            arts.add(u.label)  # type: ignore

    for n in net.V():
        if n not in visited:
            dfs(n)

    return list(sorted(bridges)), list(sorted(arts))

def blobs(net: Network) -> list[set[Node]]:
    """
    Return the biconnected components ("blobs") of the underlying undirected graph.

    Args:
        net (Network): A network object.

    Returns:
        list[set[Node]]: A list where each element is the node set of one
                         biconnected component in the undirected view of the
                         network.
    """
    adj = _undirected_adjacency(net)

    # Tarjan's biconnected-component algorithm.
    # An edge stack tracks the DFS-tree and back edges belonging to the
    # current component.  When an articulation point is detected
    # (low[v] >= disc[u]), all edges from the stack down to (u,v) form
    # one biconnected component.
    time = 0
    disc: Dict[Node, int] = {n: -1 for n in net.V()}
    low: Dict[Node, int] = {n: -1 for n in net.V()}
    parent: Dict[Node, Union[Node, None]] = {n: None for n in net.V()}
    edge_stack: list[tuple[Node, Node]] = []
    components: list[set[Node]] = []

    def push_edge(u: Node, v: Node) -> None:
        edge_stack.append((u, v))

    def pop_component(until: tuple[Node, Node]) -> set[Node]:
        """Pop edges until *until* is reached; return the node set."""
        comp_nodes: set[Node] = set()
        while edge_stack:
            a, b = edge_stack.pop()
            comp_nodes.add(a)
            comp_nodes.add(b)
            if (a, b) == until or (b, a) == until:
                break
        return comp_nodes

    def dfs(u: Node) -> None:
        nonlocal time
        disc[u] = time
        low[u] = time
        time += 1
        for v in adj[u]:
            if disc[v] == -1:
                parent[v] = u
                push_edge(u, v)
                dfs(v)
                low[u] = min(low[u], low[v])
                # Articulation detected: v's subtree cannot reach above u
                if low[v] >= disc[u]:
                    comp = pop_component((u, v))
                    if len(comp) > 0:
                        components.append(comp)
            elif v != parent[u] and disc[v] < disc[u]:
                # Back edge → part of the current component
                push_edge(u, v)
                low[u] = min(low[u], disc[v])

    for n in net.V():
        if disc[n] == -1:
            dfs(n)
            # Flush any remaining edges into a final component
            if edge_stack:
                comp = pop_component(edge_stack[-1])
                if len(comp) > 0:
                    components.append(comp)

    return components

def tree_of_blobs(net: Network) -> list[Network]:
    """
    Decompose a phylogenetic network into its biconnected components (blobs).

    Each blob is a maximal biconnected subgraph of the underlying undirected
    graph. Returns a list of Network objects, one per biconnected component,
    containing copies of the original nodes and directed edges. Node names
    are preserved but all objects are distinct from the original network.

    Args:
        net (Network): A phylogenetic network.

    Returns:
        list[Network]: One Network per biconnected component (blob).
    """
    result: list[Network] = []

    for comp_nodes in blobs(net):
        blob = Network()
        old_new: dict[Node, Node] = {}

        for node in comp_nodes:
            new_node = node.copy()
            old_new[node] = new_node
            blob.add_nodes(new_node)

        for e in net.E():
            if e.src in comp_nodes and e.dest in comp_nodes:
                new_edge = e.copy(old_new[e.src], old_new[e.dest])
                blob.add_edges(new_edge)

        result.append(blob)

    return result

def level(net: Network) -> int:
    """
    Compute the level of a phylogenetic network.

    The level is the maximum number of reticulation nodes contained in any
    biconnected component ("blob") of the underlying undirected graph.

    Returns 0 for trees.

    Args:
        net (Network): Input network.

    Returns:
        int: The network level.
    """
    if len(net.V()) == 0:
        return 0
    components = blobs(net)
    if not components:
        return 0
    return max((sum(1 for node in comp if node.is_reticulation()) for comp in components), default=0)

def count_displayed_trees(net: Network) -> int:
    """
    Estimate the number of displayed trees of a network.

    Computed as the product, over reticulation nodes, of their inbound edge
    counts (typically 2). For general networks, this is an upper bound when
    some choices may be incompatible; for level-1 networks this often matches
    the exact count.

    Args:
        net (Network): A network object.

    Returns:
        int: Estimated number of displayed trees.
    """
    prod = 1
    for n in net.V():
        if n.is_reticulation():
            prod *= max(1, len(net.in_edges(n)))
    return prod

def is_tree(net: Network) -> bool:
    """
    Check whether a network is a tree (i.e., has no reticulation nodes).

    Args:
        net (Network): A network object.

    Returns:
        bool: True if there are no reticulation nodes, False otherwise.
    """
    return count_reticulations(net) == 0

def _edge_between(net : Network, parent : Node, child : Node) -> Edge:
    """
    The single directed edge ``parent -> child``.

    ``Network.get_edge`` returns either an ``Edge`` or a list of them
    (bubbles put two edges on the same node pair); this normalises both
    shapes to one edge.

    Args:
        net (Network): A network object.
        parent (Node): Source of the edge.
        child (Node): Destination of the edge.

    Returns:
        Edge: The edge running from parent to child.
    """
    edges = net.get_edge(parent, child)
    return edges[0] if isinstance(edges, list) else edges

def _node_height(net : Network,
                 node : Node,
                 cache : dict[Node, float]) -> float:
    """
    Ultrametric height above the present, from child branch lengths.

    Derived from child branch lengths by a memoised post-order recursion
    rather than read from ``Node.get_time()``, so it is correct even when
    explicit node times were never populated. Leaves are height 0, and a
    missing branch length counts as 0.

    Args:
        net (Network): The tree or network 'node' belongs to.
        node (Node): Node whose height is requested.
        cache (dict[Node, float]): Memo of already-computed heights, mutated
                                   in place.

    Returns:
        float: The node's height, in the branch lengths' own units.
    """
    cached = cache.get(node)
    if cached is not None:
        return cached
    children = net.get_children(node)
    if not children:
        cache[node] = 0.0
        return 0.0
    best = 0.0
    for child in children:
        length = _edge_between(net, node, child).get_length()
        if length is None:
            length = 0.0
        best = max(best, _node_height(net, child, cache) + float(length))
    cache[node] = best
    return best

def _node_heights(net : Network) -> dict[Node, float]:
    """
    Map every node of a network to its height above the present.

    Args:
        net (Network): A network object.

    Returns:
        dict[Node, float]: Height of every node, in branch-length units.
    """
    cache : dict[Node, float] = {}
    for node in net.V():
        _node_height(net, node, cache)
    return cache

def _clone_net(net : Network) -> Network:
    """
    Fast structural clone of a network, for cheap whole-network rollback.

    Topology-changing moves and samplers snapshot a network before mutating
    it in place so the mutation can be undone wholesale.
    :func:`copy.deepcopy` walks the entire object graph reflectively --
    nested attribute values, sequence data, and back-references -- which
    profiling showed to dominate per-iteration cost.
    :meth:`Network.copy` instead builds fresh ``Node`` / ``Edge`` objects
    directly in O(V+E), preserving topology, branch lengths, gammas, node
    times, and reticulation flags: everything the coalescent / Felsenstein
    scorers read.

    Node attribute dictionaries are shared with the source by reference,
    which is safe because no move or scorer mutates them in place -- the
    per-node state moves change (time, reticulation flag) and per-edge state
    (length, gamma) are copied by value, and ultrametric heights live in
    external ``dict[Node, float]`` maps.

    Args:
        net (Network): A network object.

    Returns:
        Network: A structural copy of 'net'.
    """
    return net.copy()[0]

def induced_subnetwork_by_taxa(net: Network, taxa: list[str]) -> Network:
    """
    Construct the subnetwork induced by a set of leaf labels.

    The induced subnetwork is formed by taking the MRCA of the target leaves
    and retaining only those descendant branches that lead to at least one
    target leaf. Node/edge attributes (gamma, length) are copied where
    applicable.

    Args:
        net (Network): A network object.
        taxa (list[str]): List of leaf labels to induce on.

    Raises:
        NetworkError: If any leaf name is not found in the network.

    Returns:
        Network: The induced subnetwork as a new Network object.
    """
    targets : set[Node] = set()
    for name in taxa:
        node = net.has_node_named(name)
        if node is None:
            raise NetworkError(f"Leaf '{name}' not found in network")
        targets.add(node)

    sub = _induced_subnetwork(net, targets)

    # Consolidate the degree-1 chains left behind by the pruned branches, but
    # keep any spurious root edge: callers compare these against each other
    # and rely on the induced root staying put.
    sub.clean([False, False, True])
    return sub

def is_isomorphic(net1 : Network, net2 : Network) -> bool:
    """
    Returns True, if net1 and net2 are topologically identical networks. Even if 
    branch lengths are different, or node labels are different, networks can
    be isomorphic.
    
    ie-- all trees with 3 taxa are isomorphic. quartet trees have only a few 
    different variations. Networks of any size, have infinitely many possible 
    topologies.
    
    (A,B)C; is isomorphic with (D,E)F; 
    (A,B),(C,D); is not isomorphic with ((A,B)C, D);

    Args:
        net1 (Network): A Network object
        net2 (Network): Another Network object
    
    Returns:
        bool: True if the networks are topologically isomorphic, False otherwise
    """
    # Quick checks: basic invariants must match
    leaves1 = net1.get_leaves()
    leaves2 = net2.get_leaves()
    
    if len(leaves1) != len(leaves2):
        return False
    
    if len(net1.V()) != len(net2.V()):
        return False
    
    if len(net1.E()) != len(net2.E()):
        return False
    
    # Check number of reticulation nodes
    retics1 = [n for n in net1.V() if n.is_reticulation()]
    retics2 = [n for n in net2.V() if n.is_reticulation()]
    if len(retics1) != len(retics2):
        return False
    
    def normalized_clusters(net: Network) -> set[frozenset[str]]:
        """Create a normalized copy with generic leaf labels and return cluster set."""
        net_copy, old_new = net.copy()
        
        leaves = sorted(net.get_leaves(), key=lambda n: n.label)
        
        for i, leaf in enumerate(leaves):
            net_copy.update_node_name(old_new[leaf], f"L{i+1}")
        
        clusters = get_all_clusters(net_copy, include_trivial=False)
        return {frozenset(n.label for n in cluster) for cluster in clusters}
    
    return normalized_clusters(net1) == normalized_clusters(net2)


# ── Network Distance / Comparison Helpers ──────────────────────────────

def _leaf_label_set(net: Network) -> frozenset[str]:
    """Return the frozenset of leaf labels in the network."""
    return frozenset(n.label for n in net.get_leaves())


def _hardwired_clusters_by_label(
    net: Network,
    include_trivial: bool = False
) -> set[frozenset[str]]:
    """
    Compute hardwired clusters expressed as frozensets of leaf labels.

    A hardwired cluster of node v is the full set of leaf descendants
    reachable from v by following *all* directed edges (ignoring the
    reticulation / switching semantics).

    Args:
        net: A phylogenetic network.
        include_trivial: If True, include singleton leaf clusters.

    Returns:
        The set of hardwired clusters (each a frozenset of leaf labels).
    """
    clusters: set[frozenset[str]] = set()

    for labels in _leaf_labels_below(net).values():
        if len(labels) > 1 or (include_trivial and len(labels) == 1):
            clusters.add(labels)

    return clusters

def _leaf_labels_below(net : Network) -> dict[Node, frozenset[str]]:
    """
    Map every node of a network to the leaf *labels* below it.

    The label-valued form of :meth:`Network.leaf_descendants_all`, which
    returns :class:`~phynetpy.Network.Node` objects belonging to one
    particular graph and so cannot be compared across networks.

    Args:
        net (Network): A network object.

    Returns:
        dict[Node, frozenset[str]]: Hardwired cluster of every node
                                    reachable from the root.
    """
    return {node : frozenset(leaf.label for leaf in desc)
            for node, desc in net.leaf_descendants_all().items()}

def network_clusters(net: Network) -> set[frozenset[str]]:
    """
    The set of non-trivial leaf-label clusters induced by a network's nodes.

    Label-based rather than node-based, so clusters are comparable *across*
    networks -- which :func:`get_all_clusters` is not, since it returns sets
    of :class:`~phynetpy.Network.Node` objects belonging to one graph.

    Args:
        net (Network): The network to summarise.

    Returns:
        set[frozenset[str]]: One frozenset of leaf labels per non-trivial
                             cluster: size 2 or more, excluding the full
                             leaf set.
    """
    leaf_count = len(net.get_leaves())
    return {c for c in _hardwired_clusters_by_label(net) if len(c) < leaf_count}


def _softwired_clusters_by_label(
    net: Network,
    include_trivial: bool = False
) -> set[frozenset[str]]:
    """
    Compute softwired clusters expressed as frozensets of leaf labels.

    The softwired cluster set is the union of cluster sets taken over
    every displayed tree of the network.

    Args:
        net: A phylogenetic network.
        include_trivial: If True, include singleton leaf clusters.

    Returns:
        The set of softwired clusters (each a frozenset of leaf labels).
    """
    if is_tree(net):
        return _hardwired_clusters_by_label(net, include_trivial)

    clusters: set[frozenset[str]] = set()
    for tree in get_all_subtrees(net):
        clusters.update(_hardwired_clusters_by_label(tree, include_trivial))

    return clusters


def _mu_vectors(
    net: Network,
    leaf_order: list[str] | None = None
) -> list[tuple[int, ...]]:
    """
    Compute the mu-representation of a network.

    For every node v in the network whose leaf set is
    X = {l_1, ..., l_n}, the mu-vector is

        mu(v) = (m(v, l_1), ..., m(v, l_n))

    where m(v, l_i) counts the number of distinct directed paths from v
    to leaf l_i.

    Args:
        net: A phylogenetic network (must be acyclic).
        leaf_order: Fixed ordering of leaf labels.  When comparing two
            networks the same ordering must be used.  If *None*, leaves
            are sorted alphabetically.

    Returns:
        A list of mu-vectors (one per node in V).
    """
    leaves = net.get_leaves()
    if leaf_order is None:
        leaf_order = sorted(n.label for n in leaves)

    leaf_set = set(leaves)
    leaf_index = {label: i for i, label in enumerate(leaf_order)}
    n_leaves = len(leaf_order)

    topo = net.topological_order()
    path_counts: dict[Node, list[int]] = {}

    for node in reversed(topo):
        counts = [0] * n_leaves
        if node in leaf_set:
            idx = leaf_index.get(node.label)
            if idx is not None:
                counts[idx] = 1
        for child in net.get_children(node):
            if child in path_counts:
                for i in range(n_leaves):
                    counts[i] += path_counts[child][i]
        path_counts[node] = counts
    
    reduced = []
    for node in net.V():
        if (not node.is_reticulation()
                and net.in_degree(node) == 1
                and net.out_degree(node) == 1):
            continue  # this node would be suppressed in a reduced network
        reduced.append(tuple(path_counts[node]))
    return reduced
    #return [tuple(path_counts[node]) for node in net.V()]


def _tripartitions(
    net: Network
) -> set[tuple[frozenset[str], frozenset[str], frozenset[str]]]:
    """
    Compute tripartitions induced by tree nodes with exactly two children.

    For each such node u with children c_1, c_2 the tripartition is

        (desc(c_1),  desc(c_2),  X \\ (desc(c_1) | desc(c_2)))

    where X is the full leaf label set and desc(c_i) are the hardwired
    leaf descendants of child c_i.

    Args:
        net: A phylogenetic network.

    Returns:
        A set of tripartitions, each in canonical (sorted) form.
    """
    all_leaves = _leaf_label_set(net)
    triparts: set[tuple[frozenset[str], frozenset[str], frozenset[str]]] = set()

    for node in net.V():
        children = net.get_children(node)
        if len(children) != 2 or node.is_reticulation():
            continue

        desc1 = frozenset(n.label for n in net.leaf_descendants(children[0]))
        desc2 = frozenset(n.label for n in net.leaf_descendants(children[1]))
        rest = all_leaves - desc1 - desc2

        parts = tuple(sorted(
            [desc1, desc2, rest],
            key=lambda s: (len(s), tuple(sorted(s)))
        ))
        triparts.add(parts)

    return triparts


def _displayed_tree_topology_set(
    net: Network
) -> set[frozenset[frozenset[str]]]:
    """
    Enumerate the set of unique displayed-tree topologies.

    Each topology is identified by its cluster set (frozenset of
    frozensets of leaf labels), so two trees that share the same
    cluster set are considered identical.

    Args:
        net: A phylogenetic network.

    Returns:
        A set of tree topologies, each encoded as a frozenset of clusters.
    """
    if is_tree(net):
        clusters = _hardwired_clusters_by_label(net, include_trivial=True)
        return {frozenset(clusters)}

    topos: set[frozenset[frozenset[str]]] = set()
    for tree in get_all_subtrees(net):
        clusters = _hardwired_clusters_by_label(tree, include_trivial=True)
        topos.add(frozenset(clusters))

    return topos


# ── Network Distance Metrics ──────────────────────────────────────────

def _symmetric_difference_distance(
    s1: set,
    s2: set,
    normalize: bool
) -> Union[int, float]:
    """
    Size of the symmetric difference of two feature sets.

    Shared tail of the cluster, tripartition, and displayed-tree metrics:
    each builds a set of structural features per network and then differs
    only in how that set is built.

    Args:
        s1: Feature set of the first network.
        s2: Feature set of the second network.
        normalize: Divide by |s1 | s2| to get a value in [0, 1].

    Returns:
        The distance (int, or float when normalized).
    """
    sym_diff = len(s1.symmetric_difference(s2))

    if normalize:
        union_size = len(s1 | s2)
        return sym_diff / union_size if union_size > 0 else 0.0

    return sym_diff


def hardwired_cluster_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> Union[int, float]:
    """
    Hardwired cluster distance between two phylogenetic networks.

    The hardwired cluster of a node v is the set of all leaf labels
    reachable from v by following every directed edge.  The distance is
    the cardinality of the symmetric difference of the two hardwired
    cluster sets:

        d_HC(N1, N2) = |HC(N1)  triangle  HC(N2)|

    When *normalize* is True the result is divided by |HC(N1) | HC(N2)|,
    yielding a value in [0, 1].

    Reference
    ---------
    Huson, Rupp & Scornavacca.  *Phylogenetic Networks: Concepts,
    Algorithms and Applications* (2010), Chapter 7.

    Args:
        net1: First network.
        net2: Second network.
        normalize: Return a value in [0, 1] instead of a raw count.

    Returns:
        The hardwired cluster distance (int, or float when normalized).
    """
    return _symmetric_difference_distance(
        _hardwired_clusters_by_label(net1),
        _hardwired_clusters_by_label(net2),
        normalize,
    )


def softwired_cluster_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> Union[int, float]:
    """
    Softwired cluster distance between two phylogenetic networks.

    The softwired cluster set of a network is the union of cluster sets
    across all of its displayed trees.  The distance is the cardinality
    of the symmetric difference:

        d_SC(N1, N2) = |SC(N1)  triangle  SC(N2)|

    .. note::
       This enumerates all 2^k displayed trees where k is the number of
       reticulations.  For networks with many reticulations the cost is
       exponential.

    When *normalize* is True the result is divided by |SC(N1) | SC(N2)|,
    yielding a value in [0, 1].

    Reference
    ---------
    Huson, Rupp & Scornavacca.  *Phylogenetic Networks: Concepts,
    Algorithms and Applications* (2010), Chapter 7.

    Args:
        net1: First network.
        net2: Second network.
        normalize: Return a value in [0, 1] instead of a raw count.

    Returns:
        The softwired cluster distance (int, or float when normalized).
    """
    return _symmetric_difference_distance(
        _softwired_clusters_by_label(net1),
        _softwired_clusters_by_label(net2),
        normalize,
    )


def mu_distance(net1: Network, net2: Network) -> int:
    """
    Mu-representation distance between two phylogenetic networks.

    For every node v in a network with leaf set X = {l_1, ..., l_n} the
    *mu-vector* is

        mu(v) = (m(v, l_1), ..., m(v, l_n))

    where m(v, l_i) is the number of directed paths from v to leaf l_i.
    The *mu-representation* of a network is the multiset of mu-vectors
    over all nodes.  The distance is the size of the multiset symmetric
    difference of the two mu-representations.

    This is a metric on the space of *reduced* phylogenetic networks
    (networks with no degree-2 tree nodes and no parallel edges).  It
    is also referred to as the *topological distance for reduced
    networks*.  For trees it reduces to the Robinson--Foulds distance.

    Both networks must share the same leaf label set.

    Reference
    ---------
    Cardona, Llabres, Rossello & Valiente.  *Metrics for Phylogenetic
    Networks I: Generalizations of the Robinson-Foulds Metric*.
    IEEE/ACM Trans. Comput. Biol. Bioinformatics **6** (2009), 46--61.

    Args:
        net1: First network.
        net2: Second network.

    Raises:
        NetworkError: If the two networks have different leaf label sets.

    Returns:
        The mu-distance (non-negative integer).
    """
    leaves1 = _leaf_label_set(net1)
    leaves2 = _leaf_label_set(net2)

    if leaves1 != leaves2:
        raise NetworkError(
            "mu_distance requires identical leaf label sets. "
            f"Got {sorted(leaves1)} vs {sorted(leaves2)}"
        )

    leaf_order = sorted(leaves1)
    mu1 = _mu_vectors(net1, leaf_order)
    mu2 = _mu_vectors(net2, leaf_order)

    counter1 = Counter(mu1)
    counter2 = Counter(mu2)

    all_keys = set(counter1) | set(counter2)
    return sum(abs(counter1.get(k, 0) - counter2.get(k, 0)) for k in all_keys)


def tripartition_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> Union[int, float]:
    """
    Tripartition-based distance between two phylogenetic networks.

    For each tree node u with exactly two children c_1, c_2 the
    tripartition is

        (desc(c_1),  desc(c_2),  X \\ (desc(c_1) | desc(c_2)))

    where X is the full leaf set.  The distance is the cardinality of
    the symmetric difference of the tripartition sets:

        d_T(N1, N2) = |T(N1)  triangle  T(N2)|

    Reticulation nodes (in-degree >= 2) are excluded from tripartition
    generation.

    When *normalize* is True the result is divided by |T(N1) | T(N2)|,
    yielding a value in [0, 1].

    Reference
    ---------
    Moret *et al.*; Huson, Rupp & Scornavacca.  *Phylogenetic Networks:
    Concepts, Algorithms and Applications* (2010), Chapter 7.

    Args:
        net1: First network.
        net2: Second network.
        normalize: Return a value in [0, 1] instead of a raw count.

    Returns:
        The tripartition distance (int, or float when normalized).
    """
    return _symmetric_difference_distance(
        _tripartitions(net1), _tripartitions(net2), normalize
    )


def displayed_tree_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> Union[int, float]:
    """
    Tree-based distance between two phylogenetic networks.

    Each network's set of unique displayed-tree topologies (identified
    by cluster representation) is computed.  The distance is the
    cardinality of the symmetric difference:

        d_DT(N1, N2) = |DT(N1)  triangle  DT(N2)|

    .. note::
       This enumerates all 2^k displayed trees per network.  For
       networks with many reticulations the cost is exponential.

    When *normalize* is True the result is divided by |DT(N1) | DT(N2)|,
    yielding a value in [0, 1].

    Args:
        net1: First network.
        net2: Second network.
        normalize: Return a value in [0, 1] instead of a raw count.

    Returns:
        The displayed-tree distance (int, or float when normalized).
    """
    return _symmetric_difference_distance(
        _displayed_tree_topology_set(net1),
        _displayed_tree_topology_set(net2),
        normalize,
    )


def robinson_foulds_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> Union[int, float]:
    """
    Robinson--Foulds distance between two phylogenetic trees or networks.

    For rooted trees the RF distance equals the cardinality of the
    symmetric difference of the non-trivial cluster sets (clusters of
    size >= 2 and < n, where n is the number of leaves).

    For networks the function uses hardwired clusters, which is the
    natural generalization of RF to DAGs.

    Both inputs must share the same leaf label set.

    When *normalize* is True the result is divided by
    |C(N1)| + |C(N2)| (the maximum possible RF for the given cluster
    counts), yielding a value in [0, 1].

    Reference
    ---------
    Robinson & Foulds.  *Comparison of Phylogenetic Trees*.
    Mathematical Biosciences **53** (1981), 131--147.

    Cardona, Llabres, Rossello & Valiente.  *Metrics for Phylogenetic
    Networks I: Generalizations of the Robinson-Foulds Metric*.
    IEEE/ACM Trans. Comput. Biol. Bioinformatics **6** (2009), 46--61.

    Args:
        net1: First tree / network.
        net2: Second tree / network.
        normalize: Return a value in [0, 1] instead of a raw count.

    Raises:
        NetworkError: If the two inputs have different leaf label sets.

    Returns:
        The RF distance (int, or float when normalized).
    """
    leaves1 = _leaf_label_set(net1)
    leaves2 = _leaf_label_set(net2)
    if leaves1 != leaves2:
        raise NetworkError(
            "Robinson-Foulds distance requires identical leaf label sets. "
            f"Got {sorted(leaves1)} vs {sorted(leaves2)}"
        )

    n = len(leaves1)
    hc1 = {c for c in _hardwired_clusters_by_label(net1) if 1 < len(c) < n}
    hc2 = {c for c in _hardwired_clusters_by_label(net2) if 1 < len(c) < n}
    sym_diff = len(hc1.symmetric_difference(hc2))

    if normalize:
        denom = len(hc1) + len(hc2)
        return sym_diff / denom if denom > 0 else 0.0

    return sym_diff


# ── Branch-Length-Aware Distance Helpers ───────────────────────────────

def _average_pairwise_matrix(
    net: Network,
    weighted: bool = False
) -> dict[tuple[str, str], float]:
    """
    Compute the (weighted) average pairwise leaf distance matrix.

    For each pair of *original* leaves, the distance is averaged across
    all displayed trees.  If *weighted* is True the average uses each
    displayed tree's probability (product of kept gammas); otherwise a
    uniform average is used.

    Branch lengths on the network edges are required.  Edges whose
    length is *None* are treated as having length 0.

    Args:
        net: A phylogenetic network.
        weighted: Use probability-weighted averaging.

    Returns:
        A dictionary mapping sorted (label_i, label_j) pairs to their
        averaged distance.
    """
    original_leaves = sorted(n.label for n in net.get_leaves())
    n_leaves = len(original_leaves)
    pairs = [
        tuple(sorted((original_leaves[i], original_leaves[j])))
        for i in range(n_leaves)
        for j in range(i + 1, n_leaves)
    ]

    pair_sum: dict[tuple[str, str], float] = {p: 0.0 for p in pairs}
    weight_sum = 0.0

    trees_with_probs = _displayed_trees_with_probs(net)
    for tree, prob in trees_with_probs:
        w = prob if weighted else 1.0
        weight_sum += w
        dist_map = pairwise_leaf_distance(tree, use_branch_lengths=True)
        for pair in pairs:
            if pair in dist_map:
                pair_sum[pair] += w * dist_map[pair]

    if weight_sum > 0:
        for pair in pairs:
            pair_sum[pair] /= weight_sum

    return pair_sum


def average_path_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> float:
    """
    Average Path Distance (APD) between two phylogenetic networks.

    For each network a pairwise leaf distance matrix is computed by
    averaging the path lengths across all displayed trees (uniform
    weighting).  The APD is the L1 norm of the difference between the
    two matrices:

        APD(N1, N2) = sum_{i<j} |D_N1(l_i,l_j) - D_N2(l_i,l_j)|

    When *normalize* is True the sum is divided by the number of leaf
    pairs C(n, 2).

    Branch lengths on every edge are required.

    .. note::
       This enumerates all 2^k displayed trees per network.  For
       networks with many reticulations the cost is exponential.

    Reference
    ---------
    Yakici, Ogilvie & Nakhleh.  *Phylogenetic Network Dissimilarity
    Measures that Take Branch Lengths into Account*.  RECOMB-CG 2022,
    LNCS 13234, pp. 86--102.

    Args:
        net1: First network.
        net2: Second network.
        normalize: Divide by C(n, 2) to get a per-pair average.

    Raises:
        NetworkError: If the two networks have different leaf label sets.

    Returns:
        The APD (float).
    """
    leaves1 = _leaf_label_set(net1)
    leaves2 = _leaf_label_set(net2)
    if leaves1 != leaves2:
        raise NetworkError(
            "Average path distance requires identical leaf label sets. "
            f"Got {sorted(leaves1)} vs {sorted(leaves2)}"
        )

    m1 = _average_pairwise_matrix(net1, weighted=False)
    m2 = _average_pairwise_matrix(net2, weighted=False)

    total = sum(abs(m1[p] - m2.get(p, 0.0)) for p in m1)

    if normalize:
        n_pairs = len(m1)
        return total / n_pairs if n_pairs > 0 else 0.0

    return total


def weighted_average_path_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> float:
    """
    Weighted Average Path Distance (WAPD) between two networks.

    Identical to :func:`average_path_distance` except that the average
    over displayed trees is weighted by each tree's probability (the
    product of the gamma values of the kept reticulation edges).

        WAPD(N1, N2) = sum_{i<j} |WD_N1(l_i,l_j) - WD_N2(l_i,l_j)|

    When *normalize* is True the sum is divided by the number of leaf
    pairs C(n, 2).

    Branch lengths and gamma values on every reticulation edge are
    required.

    .. note::
       This enumerates all 2^k displayed trees per network.  For
       networks with many reticulations the cost is exponential.

    Reference
    ---------
    Yakici, Ogilvie & Nakhleh.  *Phylogenetic Network Dissimilarity
    Measures that Take Branch Lengths into Account*.  RECOMB-CG 2022,
    LNCS 13234, pp. 86--102.

    Args:
        net1: First network.
        net2: Second network.
        normalize: Divide by C(n, 2) to get a per-pair average.

    Raises:
        NetworkError: If the two networks have different leaf label sets.

    Returns:
        The WAPD (float).
    """
    leaves1 = _leaf_label_set(net1)
    leaves2 = _leaf_label_set(net2)
    if leaves1 != leaves2:
        raise NetworkError(
            "Weighted average path distance requires identical leaf label sets. "
            f"Got {sorted(leaves1)} vs {sorted(leaves2)}"
        )

    m1 = _average_pairwise_matrix(net1, weighted=True)
    m2 = _average_pairwise_matrix(net2, weighted=True)

    total = sum(abs(m1[p] - m2.get(p, 0.0)) for p in m1)

    if normalize:
        n_pairs = len(m1)
        return total / n_pairs if n_pairs > 0 else 0.0

    return total


# ── Shared helpers for ascii() / ascii_extended() ──────────────────────

def _ascii_compute_depths(
    net: Network, root: Node
) -> tuple[dict[Node, int], int]:
    """BFS depth assignment. Reticulation nodes keep the maximum depth."""
    node_depth: dict[Node, int] = {}
    queue: deque[tuple[Node, int]] = deque([(root, 0)])
    max_depth = 0
    while queue:
        node, depth = queue.popleft()
        if node in node_depth:
            node_depth[node] = max(node_depth[node], depth)
        else:
            node_depth[node] = depth
        max_depth = max(max_depth, depth)
        for child in net.get_children(node):
            queue.append((child, depth + 1))
    return node_depth, max_depth


def _ascii_ordered_leaves(net: Network, node: Node) -> list[Node]:
    """Left-to-right DFS collection of leaf nodes."""
    children = net.get_children(node)
    if len(children) == 0:
        return [node]
    result: list[Node] = []
    for child in children:
        result.extend(_ascii_ordered_leaves(net, child))
    return result


def _ascii_compute_x(
    net: Network, node: Node, node_x: dict[Node, float]
) -> float:
    """Recursive x-position: leaves keep their preset value; internal
    nodes are centred on the mean of their children."""
    if node in node_x:
        return node_x[node]
    children = net.get_children(node)
    if len(children) == 0:
        return node_x.get(node, 0.0)
    child_xs = [_ascii_compute_x(net, child, node_x) for child in children]
    node_x[node] = sum(child_xs) / len(child_xs)
    return node_x[node]


def _ascii_group_by_depth(
    node_depth: dict[Node, int], node_x: dict[Node, float]
) -> dict[int, list[Node]]:
    """Group nodes by depth and sort each group by x-position."""
    nodes_by_depth: dict[int, list[Node]] = {}
    for node, depth in node_depth.items():
        nodes_by_depth.setdefault(depth, []).append(node)
    for depth in nodes_by_depth:
        nodes_by_depth[depth].sort(key=lambda n: node_x[n])
    return nodes_by_depth


def _ascii_prepare_layout(
    net: Network,
) -> tuple[Node, dict[Node, int], int, list[Node], dict[Node, float], None] | str:
    """Shared preamble for both ascii renderers.

    Returns either a descriptive error string or a tuple of
    ``(root, node_depth, max_depth, ordered_leaves, node_x, None)``.
    """
    if len(net.V()) == 0:
        return "(empty network)"
    try:
        root = net.root()
    except Exception:
        return "(network has no valid root)"

    node_depth, max_depth = _ascii_compute_depths(net, root)
    ordered_leaves = _ascii_ordered_leaves(net, root)
    if len(ordered_leaves) == 0:
        return f"  {root.label if root.label else '?'}"

    node_x: dict[Node, float] = {leaf: float(i) for i, leaf in enumerate(ordered_leaves)}
    _ascii_compute_x(net, root, node_x)
    return root, node_depth, max_depth, ordered_leaves, node_x, None


# ── Public ASCII renderers ─────────────────────────────────────────────

def ascii(net: Network, show_edge_lengths: bool = False) -> str:
    """
    Prints out an ascii art depiction of this Network object as a vertical
    tree/network with the root at the top and leaves at the bottom.

    Args:
        net (Network): A Network
        show_edge_lengths (bool): If True, display edge lengths. Defaults to False.

    Returns:
        str: The ASCII art representation of the network

    Example:
        For newick string "((C,D)A, E)Root;", outputs::

                  Root
                 /    \\
                A      E
               / \\
              C   D
    """
    layout = _ascii_prepare_layout(net)
    if isinstance(layout, str):
        return layout
    root, node_depth, max_depth, ordered_leaves, node_x, _ = layout

    char_width = 6
    max_label_len = max(len(n.label) if n.label else 1 for n in net.V())
    char_width = max(char_width, max_label_len + 2)
    total_width = int(len(ordered_leaves) * char_width + max_label_len)
    nodes_by_depth = _ascii_group_by_depth(node_depth, node_x)

    def _center(node: Node) -> int:
        return int(node_x[node] * char_width) + char_width // 2

    lines: list[str] = []
    for depth in range(max_depth + 1):
        if depth not in nodes_by_depth:
            continue
        nodes_at_depth = nodes_by_depth[depth]

        label_line = [' '] * total_width
        for node in nodes_at_depth:
            x_pos = int(node_x[node] * char_width)
            label = node.label if node.label else "?"
            start = max(0, x_pos - len(label) // 2 + char_width // 2)
            for i, ch in enumerate(label):
                if start + i < total_width:
                    label_line[start + i] = ch
        lines.append(''.join(label_line).rstrip())

        if depth < max_depth:
            branch_line = [' '] * total_width
            for node in nodes_at_depth:
                children = net.get_children(node)
                if len(children) == 0:
                    continue
                parent_x = _center(node)
                child_positions = sorted(
                    [(_center(c), c) for c in children], key=lambda t: t[0]
                )
                if len(child_positions) == 1:
                    cx = child_positions[0][0]
                    if cx < parent_x:
                        if 0 <= parent_x - 1 < total_width:
                            branch_line[parent_x - 1] = '/'
                    elif cx > parent_x:
                        if 0 <= parent_x + 1 < total_width:
                            branch_line[parent_x + 1] = '\\'
                    else:
                        if 0 <= parent_x < total_width:
                            branch_line[parent_x] = '|'
                elif len(child_positions) == 2:
                    if child_positions[0][0] < parent_x:
                        if 0 <= parent_x - 1 < total_width:
                            branch_line[parent_x - 1] = '/'
                    if child_positions[1][0] > parent_x:
                        if 0 <= parent_x + 1 < total_width:
                            branch_line[parent_x + 1] = '\\'
                else:
                    leftmost = child_positions[0][0]
                    rightmost = child_positions[-1][0]
                    if leftmost < parent_x and 0 <= parent_x - 1 < total_width:
                        branch_line[parent_x - 1] = '/'
                    if rightmost > parent_x and 0 <= parent_x + 1 < total_width:
                        branch_line[parent_x + 1] = '\\'
                    for cx, _ in child_positions:
                        if cx == parent_x and 0 <= parent_x < total_width:
                            branch_line[parent_x] = '|'
            lines.append(''.join(branch_line).rstrip())

    return '\n'.join(lines)


# ── Reticulation-Aware Comparison (re-export) ──────────────────────────
#
# The tripartition-matching reticulation dissimilarity lives in its own
# module (:mod:`~.ReticulationComparison`) but is surfaced here alongside the
# other network distance metrics for discoverability.  The import is placed at
# the bottom of this file, and ``ReticulationComparison`` imports the induced
# sub-network helper lazily, so there is no circular import at load time.
from .ReticulationComparison import (  # noqa: E402
    JACCARD,
    SYMMETRIC,
    ReticulationTripartition,
    NetworkComparison,
    reticulation_tripartitions,
    set_distance,
    tripartition_dissimilarity,
    nakhleh_metric,
    reticulation_dissimilarity,
    reticulation_precision_recall,
    combined_dissimilarity,
    compare_networks,
)