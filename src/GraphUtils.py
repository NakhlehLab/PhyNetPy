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
Last Edit : 8/18/25
First Included in Version : 1.0.0

Docs   - [x]
Tests  - [ ]
Design - [ ]
"""

from .Network import *
from collections import Counter, deque
from itertools import product as _product
from typing import Dict, List, Tuple, Set, Union
import numpy as np
import heapq

def _retic_edge_choice(retic_map : dict[Node, list[Edge]], 
                      funcs : list[dict[Node, Edge]]) -> list[dict[Node, Edge]]:
    """
    Recursive helper function that generates all combinations of hybrid edges
    to remove from a network to create an underlying tree structure.

    Args:
        retic_map (dict[Node, list[Edge]]): dictionary of reticulation nodes to 
                                            their hybrid in-edges
        funcs (list[dict[Node, Edge]]): list of dictionaries that map reticulation
                                        nodes to their chosen hybrid in-edges   
    Returns:
        list[dict[Node, Edge]]: list of dictionaries that map reticulation nodes
                                to their chosen hybrid in-edges
    """
    if retic_map != {}:
        node = list(retic_map.keys())[0]
        hybrid_edges = retic_map[node]
       
        for index in range(len(funcs) // 2):
            funcs[index][node] = hybrid_edges[0]
        for index in range(len(funcs) // 2, len(funcs)):
            funcs[index][node] = hybrid_edges[1]
         
        del retic_map[node]
        
        _retic_edge_choice(retic_map, funcs)
        
    return funcs

def get_all_clusters(net : Network,
                     node : Union[Node, None] = None, 
                     include_trivial : bool = False)\
                     -> set[frozenset[Node]]:
    """
    Compile a list of clusters that make up this graph.
    Ie: for a graph ((A, B)C, D); , set of all clusters is {(A,B), (A,B,C)}.
    
    Can optionally allow the trivial leaf clusters in the set as desired.

    Args:
        net (Network): the network to operate on
        node (Node): For any user call, this should be the root. 
                     For internal calls, it is the starting point for search.
        include_trivial (bool): If set to True, includes clusters of size 1. 
                                Defaults to False.

    Returns:
        set: A set of all clusters in this graph. Each cluster is represented 
             as a set of either names or nodes.
    """
    if node is None:
        node = net.root()
    
    cluster_set : set[frozenset[Node]] = set()
    graph_leaves = net.get_leaves()
    children = net.get_children(node)
    
    # Each leaf_descendant set of a child is a cluster, 
    # so long as it is not trivial
    for child in children:
        if child not in graph_leaves:
            #Get potential cluster
            leaf_descendant_set = net.leaf_descendants(child)
            
            #Check for size 
            if len(leaf_descendant_set) > 1: 
                cluster_set.add(frozenset(leaf_descendant_set))
            
            #Recurse over the next subtree
            cluster_set = cluster_set.union(get_all_clusters(net, child))
    
    if include_trivial:
        for leaf in graph_leaves:
            cluster_set.add(frozenset([leaf]))
        
    return cluster_set

def merge_networks(left : Network, right : Network) -> Network:
    """
    Combine two networks into a single
    network object by making the roots of each network
    the children of a new root node.

    Args:
        left (Network): Left subnetwork
        right (Network): Right subnetwork

    Returns:
        Network: The resulting network object, containing copies of the nodes
                 and edges of the original networks.
    """
    merger = Network()
    
    left_copy, oldnew_left = left.copy()
    right_copy, oldnew_right = right.copy()
    
    # Add all nodes and edges from left_copy and right_copy to merger first
    merger.add_nodes(left_copy.V())
    for edge in left_copy.E():
        merger.add_edges(edge)
    merger.add_nodes(right_copy.V())
    for edge in right_copy.E():
        merger.add_edges(edge)
    
    # Create a new root node with a unique id in this merger
    new_root = merger.add_uid_node()
    
    # Add left and right roots as children of the new root
    left_root = oldnew_left[left.root()]
    right_root = oldnew_right[right.root()]
    
    merger.add_edges(Edge(new_root, left_root))
    merger.add_edges(Edge(new_root, right_root))
    
    return merger

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
    subnet : Network = Network()
    target_set : set[Node] = set(leaf_set)
    
    sub_root = net.mrca(set(leaf_set))
    new_sub_root = sub_root.copy()
    subnet.add_nodes(new_sub_root)
    old_new_map = {sub_root : new_sub_root}
    q = deque([sub_root])
    
    while len(q) != 0:
        cur = q.popleft()
    
        for child in net.get_children(cur):
            if child not in target_set \
               and len(net.leaf_descendants(child).intersection(target_set)) == 0:
                continue
            
            if child not in old_new_map:
                new_child = child.copy()
                old_new_map[child] = new_child
                subnet.add_nodes(new_child)
                q.appendleft(child)
            
            old_edge = net.get_edge(cur, child)
            new_edge = Edge(old_new_map[cur], old_new_map[child])
            new_edge.set_gamma(old_edge.get_gamma())
            new_edge.set_length(old_edge.get_length())
            subnet.add_edges(new_edge)
    
    if len(list(subnet.V())) > 1:
        subnet.clean()
    
    return subnet
    
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
    retics = sorted(
        (node for node in net.V() if node.is_reticulation()),
        key=lambda n: n.label,
    )
    if not retics:
        copy, _ = net.copy()
        return [copy]

    retic_in_edges = [
        sorted(list(net.in_edges(node)), key=lambda e: e.src.label)
        for node in retics
    ]

    trees: list[Network] = []
    for combo in _product(*retic_in_edges):
        preop, old_new = net.copy()
        for retic_node, kept_edge in zip(retics, combo):
            for e in list(net.in_edges(retic_node)):
                if e is not kept_edge:
                    src_new = old_new[e.src]
                    dst_new = old_new[e.dest]
                    preop.remove_edge([src_new, dst_new])
        preop.clean()
        trees.append(preop)

    return trees

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

def validate_binary(net: Network,
                    require_tree_nodes: bool = True,
                    require_retic_nodes: bool = True
                    ) -> tuple[bool, dict[str, list[Node]]]:
    """
    Validate standard binary constraints.

    - Tree/internal nodes: indegree=1 (except root with 0), outdegree=2
    - Reticulation nodes: indegree=2, outdegree=1
    - Leaves: outdegree=0

    Args:
        net (Network): A network object.
        require_tree_nodes (bool): Enforce tree node constraints.
        require_retic_nodes (bool): Enforce retic node constraints.

    Returns:
        tuple[bool, dict[str, list[Node]]]: (is_valid, violations)
    """
    violations: dict[str, list[Node]] = {
        "tree_in": [],
        "tree_out": [],
        "retic_in": [],
        "retic_out": [],
        "leaf_in": [],
        "root_out": [],
    }

    roots = set(net.roots())
    leaves = set(net.get_leaves())

    for node in net.V():
        indeg = net.in_degree(node)
        outdeg = net.out_degree(node)
        if node.is_reticulation():
            if require_retic_nodes:
                if indeg != 2:
                    violations["retic_in"].append(node)
                if outdeg != 1:
                    violations["retic_out"].append(node)
        else:
            if node in leaves:
                if outdeg != 0:
                    violations["tree_out"].append(node)
                if indeg < 1:
                    violations["leaf_in"].append(node)
            elif node in roots:
                if require_tree_nodes and outdeg == 0:
                    violations["root_out"].append(node)
            else:
                if require_tree_nodes:
                    if indeg != 1:
                        violations["tree_in"].append(node)
                    if outdeg != 2:
                        violations["tree_out"].append(node)

    is_valid = all(len(v) == 0 for v in violations.values())
    return is_valid, violations

def detect_bubbles(net: Network) -> list[tuple[Edge, Edge]]:
    """
    Detect bubble structures as parallel directed edges (same (src, dest)).

    Args:
        net (Network): A network object.

    Returns:
        list[tuple[Edge, Edge]]: List of edge pairs representing bubbles.
    """
    pair_map: Dict[tuple[Node, Node], list[Edge]] = {}
    for e in net.E():
        pair = (e.src, e.dest)
        if pair not in pair_map:
            pair_map[pair] = [e]
        else:
            pair_map[pair].append(e)

    bubbles: list[tuple[Edge, Edge]] = []
    for edges in pair_map.values():
        if len(edges) == 2:
            bubbles.append((edges[0], edges[1]))
    return bubbles

def sample_displayed_tree(net: Network,
                          rng: Union[np.random.Generator, None] = None) -> Network:
    """
    Sample one displayed tree by selecting, for each reticulation, a single
    incoming edge with probability proportional to its inheritance probability.

    Args:
        net (Network): A network object.
        rng (Union[np.random.Generator, None]): Optional RNG; defaults to new generator.

    Returns:
        Network: A cleaned, displayed tree copy of the input network.
    """
    if rng is None:
        rng = np.random.default_rng()

    tree_copy, old_new = net.copy()

    # Iterate reticulation nodes in the copy
    for node in [n for n in tree_copy.V() if n.is_reticulation()]:
        in_edges = list(tree_copy.in_edges(node))
        if len(in_edges) <= 1:
            continue
        gammas = np.array([e.get_gamma() for e in in_edges], dtype=float)
        # Normalize, handle all-zero by uniform
        if gammas.sum() <= 0:
            probs = np.ones(len(in_edges)) / len(in_edges)
        else:
            probs = gammas / gammas.sum()
        keep_index = int(rng.choice(len(in_edges), p=probs.tolist()))
        # Remove all others
        for idx, e in enumerate(in_edges):
            if idx != keep_index:
                tree_copy.remove_edge(e)

    tree_copy.clean()
    return tree_copy

def _copy_edge_attributes(src_edge: Edge, dest_edge: Edge) -> None:
    """Copy gamma, weight, length, and tag from *src_edge* to *dest_edge*."""
    dest_edge.set_gamma(src_edge.get_gamma())
    dest_edge.set_weight(src_edge.get_weight())
    dest_edge.set_length(src_edge.get_length())
    dest_edge.set_tag(src_edge.get_tag())


def contract_edge(net: Network, edge: Edge, keep: str = "parent") -> None:
    """
    Contract an internal edge by merging its endpoints.

    Args:
        net (Network): Network to modify in place.
        edge (Edge): The edge (u->v) to contract.
        keep (str): Which endpoint to keep as the merged node: "parent" keeps u,
                    "child" keeps v.

    Raises:
        NetworkError: On invalid keep value.
    """
    if keep not in ("parent", "child"):
        raise NetworkError("keep must be one of ['parent','child']")

    keep_node = edge.src if keep == "parent" else edge.dest
    drop_node = edge.dest if keep == "parent" else edge.src

    existing_pairs: set[tuple[int, int]] = {(id(e.src), id(e.dest)) for e in net.E()}

    for e in list(net.in_edges(drop_node)):
        if e.src is keep_node:
            net.remove_edge(e)
            continue
        if (id(e.src), id(keep_node)) not in existing_pairs:
            new_e = Edge(e.src, keep_node)
            _copy_edge_attributes(e, new_e)
            net.add_edges(new_e)
            existing_pairs.add((id(e.src), id(keep_node)))
        net.remove_edge(e)

    for e in list(net.out_edges(drop_node)):
        if e.dest is keep_node:
            net.remove_edge(e)
            continue
        if (id(keep_node), id(e.dest)) not in existing_pairs:
            new_e = Edge(keep_node, e.dest)
            _copy_edge_attributes(e, new_e)
            net.add_edges(new_e)
            existing_pairs.add((id(keep_node), id(e.dest)))
        net.remove_edge(e)

    if edge in net.E():
        net.remove_edge(edge)
    net.remove_nodes(drop_node)

def reroot(net: Network, new_root: Node) -> Network:
    """
    Return a copy of the network re-rooted at `new_root` by orienting edges
    away from the new root using undirected BFS levels.

    This preserves node/edge attributes; reticulation indegrees may change
    depending on topology.

    Args:
        net (Network): A network object.
        new_root (Node): Node in `net` to be the new root.

    Raises:
        NetworkError: If `new_root` is not a node in the network.

    Returns:
        Network: Re-rooted copy.
    """
    if new_root not in net.V():
        raise NetworkError("new_root must be a node in the network")

    # Build undirected adjacency for BFS levels
    adj: Dict[Node, list[Node]] = {n: [] for n in net.V()}
    for e in net.E():
        adj[e.src].append(e.dest)
        adj[e.dest].append(e.src)

    # Distances from new_root
    dist: Dict[Node, int] = {n: -1 for n in net.V()}
    q: deque[Node] = deque()
    dist[new_root] = 0
    q.append(new_root)
    while q:
        cur = q.popleft()
        for nei in adj[cur]:
            if dist[nei] == -1:
                dist[nei] = dist[cur] + 1
                q.append(nei)

    # Copy nodes
    new_net = Network()
    old_new: Dict[Node, Node] = {}
    for n in net.V():
        cpy = n.copy()
        new_net.add_nodes(cpy)
        old_new[n] = cpy

    # Orient edges from lower dist to higher dist (ties keep original)
    for e in net.E():
        a = e.src
        b = e.dest
        if dist[a] < dist[b] or (dist[a] == dist[b] and a == new_root):
            src, dst = a, b
        elif dist[b] < dist[a] or (dist[a] == dist[b] and b == new_root):
            src, dst = b, a
        else:
            # Tie: keep original
            src, dst = a, b
        ne = Edge(old_new[src], old_new[dst])
        ne.set_gamma(e.get_gamma())
        ne.set_weight(e.get_weight())
        ne.set_length(e.get_length())
        ne.set_tag(e.get_tag())
        new_net.add_edges(ne)

    new_net.clean([True, False, False])
    return new_net

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
    # Build an undirected adjacency list from the directed edge set.
    adj: Dict[Node, list[Node]] = {n: [] for n in net.V()}
    for e in net.E():
        if e.dest not in adj[e.src]:
            adj[e.src].append(e.dest)
        if e.src not in adj[e.dest]:
            adj[e.dest].append(e.src)

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

def transitive_reduction(net: Network) -> Network:
    """
    Compute the transitive reduction of an acyclic network: remove edges (u,v)
    for which an alternate directed path u -> ... -> v exists.

    Args:
        net (Network): Input network (must be acyclic).

    Raises:
        NetworkError: If the network contains a directed cycle.

    Returns:
        Network: A reduced network with identical reachability.
    """
    if not net.is_acyclic():
        raise NetworkError("transitive_reduction requires an acyclic network")

    reduced, _ = net.copy()

    # For each edge, test reachability with the edge removed
    to_remove: list[Edge] = []

    for e in list(reduced.E()):
        src = e.src
        dst = e.dest
        # Temporarily remove e
        reduced.remove_edge(e)
        # BFS from src to see if dst is reachable
        q: deque[Node] = deque([src])
        seen: Set[Node] = {src}
        reachable = False
        while q and not reachable:
            cur = q.popleft()
            for out_e in reduced.out_edges(cur):
                v = out_e.dest
                if v in seen:
                    continue
                if v == dst:
                    reachable = True
                    break
                seen.add(v)
                q.append(v)
        if reachable:
            to_remove.append(e)
        # Add the edge back for the next test
        reduced.add_edges(e)

    for e in to_remove:
        if e in reduced.E():
            reduced.remove_edge(e)

    return reduced

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
    # Build undirected adjacency list from directed edges
    adj: Dict[Node, list[Node]] = {n: [] for n in net.V()}
    for e in net.E():
        if e.dest not in adj[e.src]:
            adj[e.src].append(e.dest)
        if e.src not in adj[e.dest]:
            adj[e.dest].append(e.src)

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

def validate_times_and_lengths(net: Network, tol: float = 1e-5) -> tuple[bool, list[str]]:
    """
    Validate node times and branch lengths consistency across the network.

    For each edge (parent u -> child v), verify:
    - child_time - parent_time ≈ edge.length within tolerance.
    - child_time >= parent_time (monotonicity).

    Args:
        net (Network): A network object.
        tol (float): Allowed absolute tolerance for length mismatch.

    Returns:
        tuple[bool, list[str]]: (is_valid, diagnostics) where diagnostics is a
                                 list of messages describing any violations or
                                 missing values encountered.
    """
    msgs: list[str] = []
    ok = True
    for e in net.E():
        try:
            pt = e.src.get_time()
            ct = e.dest.get_time()
            el = e.get_length()
            if ct < pt - tol:
                ok = False
                msgs.append(f"Time monotonicity violated on {e.src.label}->{e.dest.label}: {pt} > {ct}")
            if abs((ct - pt) - el) > tol:
                ok = False
                msgs.append(f"Length mismatch on {e.src.label}->{e.dest.label}: len={el} vs dt={ct-pt}")
        except Exception:
            # Missing times; note and continue
            msgs.append(f"Missing times on {e.src.label}->{e.dest.label}; skipped length check")
    return ok, msgs

def ancestors_descendants(net: Network) -> tuple[dict[Node, set[Node]], dict[Node, set[Node]]]:
    """
    Compute the ancestor and descendant sets for each node in a network.

    Args:
        net (Network): A network object.

    Returns:
        tuple[dict[Node, set[Node]], dict[Node, set[Node]]]:
            A pair (ancestors, descendants) where:
            - ancestors[n] is the set of all nodes with a directed path to n.
            - descendants[n] is the set of all nodes reachable from n.
    """
    ancestors: dict[Node, set[Node]] = {n: set() for n in net.V()}
    descendants: dict[Node, set[Node]] = {n: set() for n in net.V()}

    # Descendants via forward BFS per node
    for n in net.V():
        q: deque[Node] = deque([n])
        seen: set[Node] = set([n])
        while q:
            cur = q.popleft()
            for e in net.out_edges(cur):
                v = e.dest
                if v not in seen:
                    seen.add(v)
                    q.append(v)
                    descendants[n].add(v)

    # Ancestors via reverse BFS per node
    for n in net.V():
        q: deque[Node] = deque([n])
        seen: set[Node] = set([n])
        while q:
            cur = q.popleft()
            for e in net.in_edges(cur):
                u = e.src
                if u not in seen:
                    seen.add(u)
                    q.append(u)
                    ancestors[n].add(u)

    return ancestors, descendants

def topological_order(net: Network) -> list[Node]:
    """
    Compute a topological order of nodes in an acyclic network.

    Delegates to :pymeth:`Network.topological_order`.

    Args:
        net (Network): A network object.

    Raises:
        NetworkError: If the network contains a directed cycle.

    Returns:
        list[Node]: Nodes in topological order from roots to leaves.
    """
    return net.topological_order()

def break_cycles(net: Network, strategy: str = "greedy") -> Network:
    """
    Produce an acyclic copy by removing a small set of cycle-forming edges.

    Strategy 'greedy' removes the first encountered back edge per DFS pass
    until the graph is acyclic. This is not guaranteed minimal but is fast
    and effective for cleaning up accidental cycles.

    Args:
        net (Network): A network object.
        strategy (str): Currently only 'greedy' is supported.

    Raises:
        NetworkError: If an unsupported strategy is specified.

    Returns:
        Network: An acyclic copy of the input network.
    """
    if strategy != "greedy":
        raise NetworkError("Unsupported strategy; use 'greedy'")

    ac, _ = net.copy()

    def find_back_edge() -> Edge | None:
        state: Dict[Node, int] = {n: 0 for n in ac.V()}  # 0=unseen,1=visiting,2=done
        parent_edge: Dict[Node, Edge | None] = {n: None for n in ac.V()}
        def dfs(u: Node) -> Edge | None:
            state[u] = 1
            for e in ac.out_edges(u):
                v = e.dest
                if state[v] == 0:
                    parent_edge[v] = e
                    be = dfs(v)
                    if be is not None:
                        return be
                elif state[v] == 1:
                    return e
            state[u] = 2
            return None
        for n in ac.V():
            if state[n] == 0:
                be = dfs(n)
                if be is not None:
                    return be
        return None

    while not ac.is_acyclic():
        be = find_back_edge()
        if be is None:
            break
        ac.remove_edge(be)
    return ac

def reticulation_parent_pairs(net: Network) -> dict[Node, list[Edge]]:
    """
    Map each reticulation node to its inbound edges, sorted by gamma.

    Args:
        net (Network): A network object.

    Returns:
        dict[Node, list[Edge]]: A mapping from each reticulation node to the list
                                 of inbound edges sorted by decreasing inheritance
                                 probability (gamma).
    """
    mapping: dict[Node, list[Edge]] = {}
    for n in net.V():
        if n.is_reticulation():
            ies = list(net.in_edges(n))
            ies.sort(key=lambda e: e.get_gamma(), reverse=True)
            mapping[n] = ies
    return mapping

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
    # Resolve leaf nodes by name
    leaf_nodes: list[Node] = []
    for name in taxa:
        node = net.has_node_named(name)
        if node is None:
            raise NetworkError(f"Leaf '{name}' not found in network")
        leaf_nodes.append(node)

    target_set: set[Node] = set(leaf_nodes)
    mrca_node = net.mrca(set(taxa))

    # Decide which child branches to include by checking if they lead to target leaves
    def contains_target_descendant(x: Node) -> bool:
        desc = net.leaf_descendants(x)
        return len(desc.intersection(target_set)) > 0

    # Build subnetwork by copying nodes/edges along retained branches
    sub = Network()
    old_new: dict[Node, Node] = {}
    root_copy = mrca_node.copy()
    sub.add_nodes(root_copy)
    old_new[mrca_node] = root_copy

    q: deque[Node] = deque([mrca_node])
    while q:
        cur = q.popleft()
        for child in net.get_children(cur):
            if not contains_target_descendant(child):
                continue
            if child not in old_new:
                child_copy = child.copy()
                sub.add_nodes(child_copy)
                old_new[child] = child_copy
            e = net.get_edge(cur, child)
            ne = Edge(old_new[cur], old_new[child])
            ne.set_gamma(e.get_gamma())
            ne.set_length(e.get_length())
            sub.add_edges(ne)
            q.append(child)

    # Clean to remove spurious degree-1 chains, etc.
    sub.clean([False, False, True])
    return sub

def extract_topology(newick_str: str) -> str:  
    """
    Extract the topology from a newick string by removing branch lengths,
    inheritance probabilities, and other metadata.
    
    This function handles PhyloNet-style newick strings that may contain:
    - Branch lengths after single colons (:)
    - Inheritance probabilities after double colons (::) 
    - Additional metadata after triple colons (:::)
    - Comments in square brackets [...]
    
    Args:
        newick_str (str): A newick string potentially with PhyloNet extensions
        
    Returns:
        str: A cleaned newick string showing only the topology
        
    Examples:
        >>> extract_topology("(A:0.1,B:0.2)C:0.3;")
        '(A,B)C;'
        
        >>> extract_topology("((A:1.0,B:2.0)#H1:3.0::0.4,C:4.0)D;")
        '((A,B)#H1,C)D;'
        
        >>> extract_topology("(A:1.0[&comment],B:2.0)C;")
        '(A,B)C;'
    """
    if not newick_str:
        return ""
    
    result = []
    i = 0
    in_comment = False
    
    while i < len(newick_str):
        char = newick_str[i]
        
        # Handle comments in square brackets
        if char == '[':
            in_comment = True
            i += 1
            continue
        elif char == ']':
            in_comment = False
            i += 1
            continue
        elif in_comment:
            i += 1
            continue
        
        # Handle colons - skip everything until next topology character
        if char == ':':
            # Count consecutive colons (for ::, :::)
            colon_count = 0
            while i < len(newick_str) and newick_str[i] == ':':
                colon_count += 1
                i += 1
            
            # Skip the value after the colon(s)
            # Values end at: comma, closing paren, opening paren, semicolon, or bracket
            while i < len(newick_str):
                if newick_str[i] in ',();[':
                    break
                i += 1
            continue
        
        # Keep topology-relevant characters
        if char in '(),;':
            result.append(char)
            i += 1
        # Keep node labels (including reticulation markers like #H1)
        elif char not in ' \t\n\r':
            # Collect the full label
            label_chars = []
            while i < len(newick_str):
                if newick_str[i] in '(),:;[':
                    break
                if newick_str[i] not in ' \t\n\r':
                    label_chars.append(newick_str[i])
                i += 1
            
            if label_chars:
                result.append(''.join(label_chars))
        else:
            # Skip whitespace
            i += 1
    
    return ''.join(result)

def extract_topology_from_file(filename: str, output_filename: str = None) -> list[str]:
    """
    Extract topology from all newick strings in a file.
    
    Args:
        filename (str): Path to file containing newick strings.
        output_filename (str, optional): Path to save cleaned newick strings.
            If None, results are only returned without writing to disk.

    Raises:
        FileNotFoundError: If `filename` does not exist.
        IOError: If reading or writing the file fails.
        
    Returns:
        list[str]: List of topology-only newick strings.
    """
    cleaned_newicks = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                cleaned = extract_topology(line)
                if cleaned:
                    cleaned_newicks.append(cleaned)
    
    if output_filename:
        with open(output_filename, 'w') as f:
            for newick in cleaned_newicks:
                f.write(newick + '\n')
    
    return cleaned_newicks

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
    leaf_nodes = set(net.get_leaves())

    for node in net.V():
        if node in leaf_nodes and not include_trivial:
            continue
        desc = net.leaf_descendants(node)
        if len(desc) > 1 or (include_trivial and len(desc) >= 1):
            clusters.add(frozenset(n.label for n in desc))

    return clusters


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
    hc1 = _hardwired_clusters_by_label(net1)
    hc2 = _hardwired_clusters_by_label(net2)
    sym_diff = len(hc1.symmetric_difference(hc2))

    if normalize:
        union_size = len(hc1 | hc2)
        return sym_diff / union_size if union_size > 0 else 0.0

    return sym_diff


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
    sc1 = _softwired_clusters_by_label(net1)
    sc2 = _softwired_clusters_by_label(net2)
    sym_diff = len(sc1.symmetric_difference(sc2))

    if normalize:
        union_size = len(sc1 | sc2)
        return sym_diff / union_size if union_size > 0 else 0.0

    return sym_diff


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
    t1 = _tripartitions(net1)
    t2 = _tripartitions(net2)
    sym_diff = len(t1.symmetric_difference(t2))

    if normalize:
        union_size = len(t1 | t2)
        return sym_diff / union_size if union_size > 0 else 0.0

    return sym_diff


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
    dt1 = _displayed_tree_topology_set(net1)
    dt2 = _displayed_tree_topology_set(net2)
    sym_diff = len(dt1.symmetric_difference(dt2))

    if normalize:
        union_size = len(dt1 | dt2)
        return sym_diff / union_size if union_size > 0 else 0.0

    return sym_diff


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

def _displayed_trees_with_probs(
    net: Network
) -> list[tuple[Network, float]]:
    """
    Enumerate every displayed tree together with its probability.

    The probability of a displayed tree is the product of gamma values
    of the reticulation edges that were *kept*.  When gamma is unset
    for an edge, a uniform 1/k weight is assumed (k = in-degree of the
    reticulation node).

    Args:
        net: A phylogenetic network.

    Returns:
        List of (tree_copy, probability) pairs.
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

    results: list[tuple[Network, float]] = []
    for combo in _product(*retic_in_edges):
        preop, old_new = net.copy()
        prob = 1.0
        for retic_node, kept_edge in zip(retics, combo):
            gamma = kept_edge.get_gamma()
            if gamma is not None and gamma > 0:
                prob *= gamma
            else:
                k = len(list(net.in_edges(retic_node)))
                prob *= 1.0 / k if k > 0 else 1.0
            for e in list(net.in_edges(retic_node)):
                if e is not kept_edge:
                    preop.remove_edge([old_new[e.src], old_new[e.dest]])
        preop.clean()
        results.append((preop, prob))

    return results


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


def ascii_extended(net: Network) -> str:
    """
    Prints a more detailed vertical ASCII art depiction of the Network
    with extended connecting lines between levels for wide trees.

    Args:
        net (Network): A Network

    Returns:
        str: The ASCII art representation of the network
    """
    layout = _ascii_prepare_layout(net)
    if isinstance(layout, str):
        return layout
    root, node_depth, max_depth, ordered_leaves, node_x, _ = layout

    max_label_len = max(len(n.label) if n.label else 1 for n in net.V())
    spacing = max(4, max_label_len + 2)
    total_width = int(len(ordered_leaves) * spacing) + spacing
    nodes_by_depth = _ascii_group_by_depth(node_depth, node_x)

    def _center(node: Node) -> int:
        return int(node_x[node] * spacing) + spacing // 2

    lines: list[str] = []
    for depth in range(max_depth + 1):
        if depth not in nodes_by_depth:
            continue
        nodes_at_depth = nodes_by_depth[depth]

        label_line = [' '] * total_width
        node_positions: dict[Node, int] = {}
        for node in nodes_at_depth:
            x_pos = _center(node)
            label = node.label if node.label else "?"
            start = max(0, min(x_pos - len(label) // 2, total_width - len(label)))
            node_positions[node] = x_pos
            for i, ch in enumerate(label):
                if start + i < total_width:
                    label_line[start + i] = ch
        lines.append(''.join(label_line).rstrip())

        if depth < max_depth:
            branch_line = [' '] * total_width
            extend_line = [' '] * total_width
            for node in nodes_at_depth:
                children = net.get_children(node)
                if len(children) == 0:
                    continue
                parent_pos = node_positions.get(node, _center(node))
                child_positions = sorted(
                    [(_center(c), c) for c in children], key=lambda t: t[0]
                )
                leftmost = child_positions[0][0]
                rightmost = child_positions[-1][0]

                if leftmost < parent_pos:
                    if 0 <= parent_pos - 1 < total_width:
                        branch_line[parent_pos - 1] = '/'
                    if leftmost < parent_pos - 2:
                        for x in range(leftmost, parent_pos - 1):
                            if 0 <= x < total_width:
                                extend_line[x] = '_'
                        if 0 <= leftmost < total_width:
                            extend_line[leftmost] = '/'

                if rightmost > parent_pos:
                    if 0 <= parent_pos + 1 < total_width:
                        branch_line[parent_pos + 1] = '\\'
                    if rightmost > parent_pos + 2:
                        for x in range(parent_pos + 2, rightmost + 1):
                            if 0 <= x < total_width:
                                extend_line[x] = '_'
                        if 0 <= rightmost < total_width:
                            extend_line[rightmost] = '\\'

                for cx, _ in child_positions:
                    if cx == parent_pos and 0 <= parent_pos < total_width:
                        branch_line[parent_pos] = '|'

            line1_str = ''.join(branch_line).rstrip()
            line2_str = ''.join(extend_line).rstrip()
            if line1_str.strip():
                lines.append(line1_str)
            if line2_str.strip():
                lines.append(line2_str)

    return '\n'.join(lines)