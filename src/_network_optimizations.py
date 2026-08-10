"""
Author : Mark Kessler

Backdoor optimization module for :mod:`.Network`.

This module houses the *algorithmic heavy lifting* for phylogenetic networks so
that :mod:`.Network` can stay a thin, readable data-structure + public-API
layer. Everything here operates on a ``Network`` through its public interface
(plus the ``_nodes`` / ``_edges`` structural containers), so the functions are
implementation details -- import them via the delegating ``Network`` methods,
not directly.

Design notes
------------
* These functions are the natural landing zone for Cython acceleration: they
  are pure graph traversals over the ``CNodeSet`` / ``CEdgeSet`` adjacency
  maps, with no dependence on the rest of the ``Network`` object state.
* Node/Edge equality + hashing is name-based (see ``Network.Node``); the
  memoized traversals below rely on that contract for their ``dict``/``set``
  bookkeeping.
* ``from __future__ import annotations`` keeps ``Node``/``Edge`` type hints as
  strings, so we only need a runtime import of ``NetworkError``. ``Network``
  imports this module at the *bottom* of its file (after the exception classes
  are defined), which keeps the circular import safe.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import combinations
from typing import TYPE_CHECKING, Any, Callable, Optional, Set, Tuple

import networkx as nx

# Runtime imports. ``Network`` imports this module at the *bottom* of its file,
# so by the time this line executes every one of these names is already bound
# in the ``Network`` module namespace -- no circular-import failure. We need the
# concrete classes here (not just for typing) because functions like ``copy``,
# ``subnet`` and ``clean`` construct new ``Node`` / ``Edge`` / ``Network``
# objects.
from .Network import Network, Node, Edge, NetworkError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


# =============================================================================
# Subgenome counting
# =============================================================================

def subgenome_count(net: "Network", n: "Node") -> int:
    """Subgenome count of ``n`` (number of ancestral lineages reaching it).

    Memoized DP up the DAG: the naive recursion recomputes shared ancestors
    once per downstream path, which is exponential in the number of
    reticulations. Caching per call makes it O(V + E), and the root is
    resolved once instead of being rebuilt on every recursive frame.
    """
    if n not in net._nodes:
        raise NetworkError("Input node is not in the graph")

    root = net.root()
    # Memo is keyed by ``id(node)`` rather than the node itself: the node keys
    # would otherwise route every lookup through the (name-based) ``Node.__hash__``
    # and ``Node.__eq__``, which profiling showed dominates this traversal.
    # ``id`` is a plain C-int hash, and every node is a distinct live object for
    # the duration of the call, so identity keying is exact. For the same reason
    # the base case uses ``is`` (object identity) instead of ``==``.
    memo: dict[int, int] = {}
    get_parents = net.get_parents

    def _count(node: "Node") -> int:
        key = id(node)
        cached = memo.get(key)
        if cached is not None:
            return cached
        if node is root:
            result = 1
        else:
            result = sum(_count(parent) for parent in get_parents(node))
        memo[key] = result
        return result

    return _count(n)


def edges_downstream_of_node(net: "Network", n: "Node") -> list["Edge"]:
    """All edges in the subgraph reachable from ``n`` (forward traversal)."""
    if n not in net._nodes:
        raise NetworkError("Input node is not in the graph.")

    q: deque[Node] = deque()
    q.appendleft(n)
    visited: set[Node] = set()
    edges: list[Edge] = []

    while len(q) != 0:
        cur = q.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for edge in net.out_edges(cur):
            edges.append(edge)
            q.append(edge.dest)

    return edges


def edges_upstream_of_node(net: "Network", n: "Node") -> list["Edge"]:
    """All edges on paths from ancestors up to ``n`` (backward traversal)."""
    if n not in net._nodes:
        raise NetworkError("Input node is not in the graph.")

    q: deque[Node] = deque()
    q.appendleft(n)
    visited: set[Node] = set()
    edges: list[Edge] = []

    while len(q) != 0:
        cur = q.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for edge in net.in_edges(cur):
            edges.append(edge)
            q.append(edge.src)

    return edges


def edges_to_subgenome_count(
    net: "Network",
    downstream_node: Optional["Node"] = None,
    delta: float = math.inf,
    start_node: Optional["Node"] = None,
) -> dict[int, list["Edge"]]:
    """Partition edges by their subgenome count.

    Propagates counts forward from ``start_node`` (root by default) in a single
    BFS, then buckets edges by count. When ``downstream_node`` is given, the
    downstream edge set is computed *once* (the original recomputed the full
    traversal inside the per-edge loop, making the filter O(E * (V + E))).
    """
    if start_node is None:
        start_node = net.root()

    q: deque[Node] = deque()
    q.appendleft(start_node)

    edges_2_sub = {edge: 0 for edge in net.E()}

    while len(q) != 0:
        cur = q.pop()  # pop right for bfs
        for neighbor in net.get_children(cur):
            edges_2_sub[net._edges.get(cur, neighbor)] += 1
            q.append(neighbor)

    partition: dict[int, list[Edge]] = {}
    for edge, value in edges_2_sub.items():
        if value not in partition:
            partition[value] = [edge]
        else:
            partition[value].append(edge)

    # Filter out invalid keys
    filter1 = {key: value for (key, value) in partition.items()
               if key <= delta}

    # Filter out edges that would create a cycle from param edge
    if downstream_node is not None:
        downstream_edges = set(edges_downstream_of_node(net, downstream_node))
        filter2: dict[int, list[Edge]] = {}
        for subct, edges in filter1.items():
            for target in edges:
                if target not in downstream_edges:
                    if subct not in filter2:
                        filter2[subct] = [target]
                    else:
                        filter2[subct].append(target)
        return filter2
    return filter1


def subgenome_ct_edges(
    net: "Network",
    downstream_node: Optional["Node"] = None,
    delta: float = math.inf,
    start_node: Optional["Node"] = None,
) -> dict["Edge", int]:
    """Inverted view of :func:`edges_to_subgenome_count`: edge -> count."""
    old_map = edges_to_subgenome_count(net, downstream_node, delta, start_node)
    rev_map: dict[Edge, int] = {}
    for key, edges in old_map.items():
        for edge in edges:
            rev_map[edge] = key
    return rev_map


# =============================================================================
# Leaf descendants
# =============================================================================

def leaf_descendants(net: "Network", node: "Node") -> set["Node"]:
    """Set of all leaves reachable from ``node``.

    Uses a ``visited`` guard: in a reticulate DAG a node can be reached by
    multiple paths, and without the guard those subtrees get re-expanded
    (exponential worst case).
    """
    if node not in net._nodes:
        raise NetworkError("Node not found in graph.")

    q: deque[Node] = deque()
    q.appendleft(node)
    leaves: set[Node] = set()
    visited: set[Node] = set()

    while len(q) != 0:
        cur = q.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        if net.out_degree(cur) == 0:
            leaves.add(cur)
        for neighbor in net.get_children(cur):
            q.append(neighbor)

    return leaves


def _leaf_desc_help(
    net: "Network",
    node: "Node",
    leaves: set["Node"],
    desc_map: dict["Node", set["Node"]],
) -> set["Node"]:
    """Memoized DFS helper for :func:`leaf_descendants_all`.

    ``leaves`` is a ``set`` so leaf membership is O(1); the previous code
    passed the leaf *list*, turning ``node in leaves`` into an O(L) scan at
    every node (O(V*L) overall).
    """
    if node not in desc_map:
        if node in leaves:
            desc_map[node] = {node}
        else:
            acc: set[Node] = set()
            for child in net.get_children(node):
                acc |= _leaf_desc_help(net, child, leaves, desc_map)
            desc_map[node] = acc
    return desc_map[node]


def leaf_descendants_all(net: "Network") -> dict["Node", set["Node"]]:
    """Map every reachable node to its set of leaf descendants (memoized DFS)."""
    desc_map: dict[Node, set[Node]] = {}
    _leaf_desc_help(net, net.root(), set(net.get_leaves()), desc_map)
    return desc_map


# =============================================================================
# Serialization
# =============================================================================

def _newick_help(
    net: "Network",
    node: "Node",
    processed_retics: set["Node"],
    leaf_set: set["Node"],
) -> str:
    """Recursive Newick builder for the subnetwork rooted at ``node``.

    ``leaf_set`` is precomputed and threaded down so leaf membership is an
    O(1) test per node instead of rebuilding the leaf list (O(V)) at every
    frame -- the naive version was O(V^2) over the whole tree.
    """
    if node in leaf_set:
        return node.label

    if net.in_degree(node) >= 2 and node in processed_retics:
        if node.label[0] != "#":
            return "#" + node.label
        return node.label

    if net.in_degree(node) >= 2:
        processed_retics.add(node)
        if node.label[0] != "#":
            node_name = "#" + node.label
        else:
            node_name = node.label
    else:
        node_name = node.label

    substr = "("
    for child in net.get_children(node):
        substr += _newick_help(net, child, processed_retics, leaf_set)
        substr += ","
    substr = substr[0:-1]
    substr += ")"
    substr += node_name
    return substr


def newick(net: "Network") -> str:
    """Extended-Newick string for ``net`` (with trailing semicolon)."""
    return _newick_help(net, net.root(), set(), set(net.get_leaves())) + ";"


# =============================================================================
# Ancestry
# =============================================================================

def mrca(net: "Network", set_of_nodes: "set[Node] | set[str]") -> "Node":
    """Least common ancestor of ``set_of_nodes`` (nodes or node-name strings).

    The LCA is the common ancestor minimizing the summed upward hop-distance to
    every input node.
    """
    format_set: set[Node] = set()
    for item in set_of_nodes:
        if type(item) is str:
            node_version = net.has_node_named(item)
            if node_version is None:
                raise NetworkError("A node in 'set_of_nodes' is not "
                                   "in the graph")
            format_set.add(node_version)
        elif type(item) is Node:
            if item in net._nodes:
                format_set.add(item)
            else:
                raise NetworkError("A node in 'set_of_nodes' is not "
                                   "in the graph")
        else:
            raise NetworkError("Wrong type for parameter set_of_nodes. "
                               "Expected set[Node] or set[str].")

    set_of_nodes = format_set

    # For each input node, the map of every ancestor -> upward distance.
    leaf_2_parents: dict[Node, dict[Node, int]] = {}

    for leaf in set_of_nodes:
        node_2_lvl: dict[Node, int] = {}
        q: deque[Node] = deque()
        q.append(leaf)
        visited: set[Node] = set()
        node_2_lvl[leaf] = 0

        while len(q) != 0:
            cur = q.popleft()
            for neighbor in net.get_parents(cur):
                if neighbor not in visited:
                    node_2_lvl[neighbor] = node_2_lvl[cur] + 1
                    q.append(neighbor)
                    visited.add(neighbor)

        leaf_2_parents[leaf] = node_2_lvl

    # Candidate LCAs are ancestors common to every input node.
    intersection = net._nodes.get_set()
    for leaf, par_level in leaf_2_parents.items():
        intersection = intersection.intersection(set(par_level.keys()))

    additive_level: dict[Node, int] = {}
    for node in intersection:
        lvl = 0
        for leaf in set_of_nodes:
            try:
                lvl += leaf_2_parents[leaf][node]
            except KeyError:
                continue
        additive_level[node] = lvl

    return min(additive_level, key=additive_level.get)  # type: ignore


def _random_object(mylist: list[Any], rng: "np.random.Generator") -> Any:
    """Pick a uniformly-random element from ``mylist`` using ``rng``."""
    return mylist[rng.integers(0, len(mylist))]


def diff_subtree_edges(net: "Network", rng: "np.random.Generator") -> list["Edge"]:
    """Two random edges whose source subtrees do not overlap.

    (Neither edge's source is reachable from the other's.)
    """
    first_edge = _random_object(net.E(), rng)
    assert type(first_edge) is Edge

    first_edge_subtree = net.leaf_descendants(first_edge.dest)

    valid_edges: list[Edge] = []
    for edge in net.E():
        leaf_desc_edge = net.leaf_descendants(edge.dest)
        if len(leaf_desc_edge.intersection(first_edge_subtree)) == 0:
            valid_edges.append(edge)

    second_edge = _random_object(valid_edges, rng)
    assert type(second_edge) is Edge

    return [first_edge, second_edge]


# =============================================================================
# Cycles / traversal / ordering
# =============================================================================

def _is_cyclic_util(
    net: "Network",
    v: "Node",
    visited: dict["Node", bool],
    rec_stack: dict["Node", bool],
) -> bool:
    """Recursive DFS helper for :func:`is_acyclic`."""
    visited[v] = True
    rec_stack[v] = True

    for neighbor in net.get_children(v):
        if not visited[neighbor]:
            if _is_cyclic_util(net, neighbor, visited, rec_stack):
                return True
        elif rec_stack[neighbor]:
            return True

    rec_stack[v] = False
    return False


def is_acyclic(net: "Network") -> bool:
    """True if every connected component of ``net`` is acyclic."""
    visited = {node: False for node in net.V()}
    rec_stack = {node: False for node in net.V()}

    for node in net.roots():
        if not visited[node]:
            if _is_cyclic_util(net, node, visited, rec_stack):
                return False

    return True


def bfs_dfs(
    net: "Network",
    start_node: Optional["Node"] = None,
    dfs: bool = False,
    is_connected: bool = False,
    accumulator: Optional[Callable[..., Any]] = None,
    accumulated: Any = None,
) -> tuple[dict["Node", int], Any]:
    """General BFS/DFS from ``start_node`` (root by default).

    Returns ``(dist, accumulated)`` where ``dist`` maps each visited node to its
    hop-distance from the start. The ``visited`` guard keeps reticulate DAGs
    (and any cycles) from re-expanding nodes.
    """
    q: deque[Node] = deque()
    visited: set[Node] = set()

    if start_node is not None:
        q.append(start_node)
        dist = {start_node: 0}
        visited.add(start_node)
    else:
        root: Node = net.root()
        q.append(root)
        dist = {root: 0}
        visited.add(root)

    while len(q) != 0:
        if dfs:
            cur = q.popleft()  # add-left + pop-left => LIFO
        else:
            cur = q.pop()      # add-left + pop-right => FIFO

        if accumulator is not None and accumulated is not None:
            accumulated = accumulator(cur, accumulated)

        for neighbor in net.get_children(cur):
            if neighbor in visited:
                continue
            dist[neighbor] = dist[cur] + 1
            q.appendleft(neighbor)
            visited.add(neighbor)

    if is_connected:
        if len(set(net.V()).difference(visited)) != 0:
            print("GRAPH HAS MORE THAN 1 CONNECTED COMPONENT")
        else:
            print("GRAPH IS FULLY CONNECTED")

    return dist, accumulated


def get_subtree_at(net: "Network", node: "Node") -> set["Node"]:
    """All nodes in the subtree reachable from ``node`` (inclusive)."""
    subtree_nodes: set[Node] = set()
    q: deque[Node] = deque([node])
    while q:
        current = q.popleft()
        if current in subtree_nodes:
            continue
        subtree_nodes.add(current)
        for child in net.get_children(current):
            q.append(child)
    return subtree_nodes


def dist_from_root(net: "Network", node: "Node") -> int:
    """Hop-count distance from the root to ``node`` (via BFS)."""
    return bfs_dfs(net)[0][node]


def topological_order(net: "Network") -> list["Node"]:
    """Kahn's-algorithm topological order (roots -> leaves).

    Raises ``NetworkError`` if the network contains a directed cycle.
    """
    indeg: dict[Node, int] = {n: net.in_degree(n) for n in net.V()}
    q: deque[Node] = deque([n for n, d in indeg.items() if d == 0])

    order: list[Node] = []
    while q:
        u = q.popleft()
        order.append(u)
        for e in net.out_edges(u):
            v = e.dest
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(net.V()):
        raise NetworkError("Graph has cycles; no topological order exists")
    return order


def distance_from_root(
    net: "Network", node: "Node", use_time: bool = True
) -> float:
    """Distance from root to ``node`` -- branch-length sum or edge count.

    Uses the O(1) node-time fast path when times are set; otherwise a targeted
    BFS from the root.
    """
    if node not in net._nodes:
        raise NetworkError("Node not found in network")

    if use_time and node.get_time() is not None:
        try:
            root_time = net.root().get_time()
            if root_time is not None:
                return node.get_time() - root_time
        except (NetworkError, Exception):
            pass  # Fall through to BFS approach

    root = net.root()
    if root == node:
        return 0.0

    visited: set[Node] = set()
    queue: deque[tuple[Node, float]] = deque([(root, 0.0)])
    visited.add(root)

    while queue:
        current, dist = queue.popleft()
        for child in net.get_children(current):
            if child == node:
                if use_time:
                    edge = net.get_edge(current, child)
                    return dist + edge.get_length()
                return dist + 1.0

            if child not in visited:
                visited.add(child)
                if use_time:
                    edge = net.get_edge(current, child)
                    new_dist = dist + edge.get_length()
                else:
                    new_dist = dist + 1.0
                queue.append((child, new_dist))

    raise NetworkError(f"No path found from root to node {node.label}")


def set_node_times_from_root(net: "Network") -> None:
    """Set every node's time to its cumulative branch length from the root."""
    try:
        root = net.root()
    except NetworkError:
        return  # No root available

    root.set_time(0.0)
    visited: set[Node] = {root}
    queue: deque[tuple[Node, float]] = deque([(root, 0.0)])

    while queue:
        current, current_time = queue.popleft()
        for child in net.get_children(current):
            if child not in visited:
                visited.add(child)
                edge = net.get_edge(current, child)
                child_time = current_time + edge.get_length()
                child.set_time(child_time)
                queue.append((child, child_time))


def set_node_times_by_edge_count(net: "Network") -> None:
    """Set every node's time to its edge-count depth from the root."""
    try:
        root = net.root()
    except NetworkError:
        return  # No root available

    root.set_time(0.0)
    visited: set[Node] = {root}
    queue: deque[tuple[Node, float]] = deque([(root, 0.0)])

    while queue:
        current, current_time = queue.popleft()
        for child in net.get_children(current):
            if child not in visited:
                visited.add(child)
                child_time = current_time + 1.0
                child.set_time(child_time)
                queue.append((child, child_time))


# =============================================================================
# Distance measures
# =============================================================================

def rooted_triplet_distance(net: "Network", other: "Network") -> float:
    """Normalized rooted-triplet distance between ``net`` and ``other``.

    For each triple of leaves the sibling pair (deepest shared MRCA) gives a
    canonical encoding; the distance is the normalized symmetric difference of
    the two triplet sets (0 = identical, 1 = disjoint).
    """

    def get_rooted_triplets(network: "Network") -> Set[Tuple[Any, ...]]:
        """Canonical ``(sibling_pair, outgroup)`` encoding of every leaf triple in ``network``."""
        triplets: Set[Tuple[Any, ...]] = set()
        leaves = network.get_leaves()

        for x, y, z in combinations(leaves, 3):
            x_label = x.label
            y_label = y.label
            z_label = z.label

            try:
                mrca_xy = network.mrca({x, y})
                mrca_xz = network.mrca({x, z})
                mrca_yz = network.mrca({y, z})

                dist_xy = network.dist_from_root(mrca_xy)
                dist_xz = network.dist_from_root(mrca_xz)
                dist_yz = network.dist_from_root(mrca_yz)

                if dist_xy >= dist_xz and dist_xy >= dist_yz:
                    triplet = tuple(sorted(
                        [tuple(sorted([x_label, y_label])), z_label], key=str))
                elif dist_xz >= dist_xy and dist_xz >= dist_yz:
                    triplet = tuple(sorted(
                        [tuple(sorted([x_label, z_label])), y_label], key=str))
                else:
                    triplet = tuple(sorted(
                        [tuple(sorted([y_label, z_label])), x_label], key=str))

                triplets.add(triplet)
            except Exception as e:
                print(f"Warning: Could not compute MRCA for triplet "
                      f"{x_label}, {y_label}, {z_label}: {e}")
                continue

        return triplets

    def triplet_distance(triplets1: Set[Tuple], triplets2: Set[Tuple]) -> float:
        """Normalized symmetric difference between two triplet sets (0 = identical, 1 = disjoint)."""
        if not triplets1 and not triplets2:
            return 0.0
        if not triplets1 or not triplets2:
            return 1.0
        common = triplets1.intersection(triplets2)
        total = triplets1.union(triplets2)
        if len(total) == 0:
            return 0.0
        return 1.0 - (len(common) / len(total))

    triplets_self = get_rooted_triplets(net)
    triplets_other = get_rooted_triplets(other)
    return triplet_distance(triplets_self, triplets_other)


# =============================================================================
# Structural copy / conversion / cleanup
# =============================================================================

def subnet(net: "Network", retic_node: "Node") -> "Network":
    """Copy the subnetwork rooted at ``retic_node`` with freshly-named nodes."""
    uid_suffix = 0
    q: deque[Node] = deque()
    q.appendleft(retic_node)
    net_copy = Network()

    new_node = Node(name=retic_node.label + "_copy" + str(uid_suffix))
    uid_suffix += 1
    net_copy.add_nodes(new_node)
    net_2_mul = {retic_node: new_node}

    while len(q) != 0:
        cur = q.pop()
        for neighbor in net.get_children(cur):
            new_node = Node(name=neighbor.label + "_copy" + str(uid_suffix))
            uid_suffix += 1
            net_copy.add_nodes(new_node)
            net_2_mul[neighbor] = new_node
            net_copy.add_edges(Edge(net_2_mul[cur], new_node))
            q.append(neighbor)

    return net_copy


def copy(net: "Network") -> "tuple[Network, dict[Node, Node]]":
    """Independent structural copy of ``net`` (new Node/Edge objects).

    Returns ``(copy, old_to_new_node_map)``. The unique-id counter is carried
    forward so later ``add_uid_node()`` calls cannot regenerate ``UID_*`` names
    that collide with existing nodes (which would corrupt the graph).
    """
    net_copy: Network = Network(
        branch_length_unit=net.get_branch_length_unit()
    )
    old_new: dict[Node, Node] = {}

    for node in net.V():
        new = node.copy()
        old_new[node] = new
        net_copy.add_nodes(new)

    for edge in net.E():
        new_src = old_new[edge.src]
        new_dest = old_new[edge.dest]
        net_copy.add_edges(edge.copy(new_src, new_dest))

    net_copy.set_uid_count(net.uid_count())
    return net_copy, old_new


def to_networkx(net: "Network") -> "nx.MultiDiGraph":
    """Export ``net`` to a :class:`networkx.MultiDiGraph`."""
    nx_network = nx.MultiDiGraph()
    nx_network.add_nodes_from([node.label for node in net.V()])
    nx_network.add_edges_from([edge.to_names() for edge in net.E()])
    return nx_network


def clean(net: "Network", options: list[bool]) -> None:
    """In-place topology-preserving cleanup of ``net``.

    ``options`` selects which routines run:
    ``[0]`` remove floater (degree-0) nodes; ``[1]`` remove a spurious
    single-child root; ``[2]`` contract degree-2 "spacer" chains into one edge.
    """
    if options[0]:
        floaters = [node for node in net.V()
                    if net.in_degree(node) == 0 and net.out_degree(node) == 0]
        for floater in floaters:
            net._nodes.remove(floater)

    if options[1]:
        root = net.root()
        if net.out_degree(root) == 1:
            spurious_edge = net.get_edge(root, net.get_children(root)[0])
            net.remove_edge(spurious_edge)
            net.remove_nodes(root)

    if options[2]:
        spacers = [n for n in net.V()
                   if net.in_degree(n) == 1 and net.out_degree(n) == 1]
        while len(spacers) != 0:
            cur = spacers[0]
            spacer_par = net.get_parents(cur)[0]
            spacer_child = net.get_children(cur)[0]

            edge_in = net.get_edge(spacer_par, cur)
            edge_out = net.get_edge(cur, spacer_child)

            len_in = edge_in.get_length() if edge_in.get_length() else 0.0
            len_out = edge_out.get_length() if edge_out.get_length() else 0.0
            gamma_out = edge_out.get_gamma()

            net.remove_edge(edge_in)
            net.remove_edge(edge_out)
            net.remove_nodes(cur)

            merged = Edge(spacer_par, spacer_child, length=len_in + len_out)
            if gamma_out:
                merged.set_gamma(gamma_out)
            net.add_edges(merged)

            spacers = [n for n in net.V()
                       if net.in_degree(n) == 1 and net.out_degree(n) == 1]
