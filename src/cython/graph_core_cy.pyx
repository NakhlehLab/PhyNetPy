# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
The adjacency structures that back every :class:`~PhyNetPy.Network.Network`.

``NodeSet`` owns V and the in/out edge maps; ``EdgeSet`` owns E and the
``(src, dest) -> [Edge]`` lookup that makes bubble edges addressable. Both are
compiled and are a hard requirement of the library -- ``Network.py`` imports
them directly rather than falling back to a Python twin.

To (re)build, from the repository root::

    pip install -e .

Neither class is meant to be used on its own; ``Network`` is the supported
surface. They deliberately do no validation beyond what ``Network`` cannot
cheaply do itself, since every method here sits in a hot path.
"""

import warnings
from collections import defaultdict


cdef class NodeSet:
    """
    The node set V of a Network, plus the in/out edge maps that make
    degree and incidence queries O(1).
    """
    
    # Typed attributes - stored as C-level data
    cdef set _nodes
    cdef object _in_map      # defaultdict[Node, set[Edge]]
    cdef object _out_map     # defaultdict[Node, set[Edge]]
    cdef dict _node_names    # dict[str, Node]
    
    def __init__(self):
        """Initialize an empty node set."""
        self._nodes = set()
        self._in_map = defaultdict(set)
        self._out_map = defaultdict(set)
        self._node_names = {}
    
    def __contains__(self, n) -> bool:
        """Check if a node is in the node set. O(1)"""
        return n in self._nodes
    
    def add(self, *nodes):
        """
        Add nodes to V.
        
        Args:
            *nodes: Node objects, lists of Node objects, or a mixture.
        """
        for node in nodes:
            self._add_single(node)
    
    cdef void _add_single(self, object node):
        """Internal: add one node, or every node in one list."""
        cdef object n
        
        if isinstance(node, (list, tuple, set)):
            for n in node:
                self._add_single(n)
        elif node not in self._nodes:
            self._nodes.add(node)
            self._node_names[node.label] = node
    
    cpdef bint ready(self, edge):
        """
        Check whether an edge can be safely added (both endpoints in V).
        
        Args:
            edge: An Edge object.
        Returns:
            bool: True if the edge's endpoints are both in V.
        """
        if type(edge).__name__ != 'Edge':
            return False
        return edge.src in self._nodes and edge.dest in self._nodes
    
    cpdef int in_deg(self, node):
        """Get the in-degree of a node, or 0 if it has none. O(1)"""
        cdef set edges = self._in_map.get(node)
        if edges is None:
            return 0
        return len(edges)
    
    cpdef int out_deg(self, node):
        """Get the out-degree of a node, or 0 if it has none. O(1)"""
        cdef set edges = self._out_map.get(node)
        if edges is None:
            return 0
        return len(edges)
    
    cdef set _edge_set(self, object mp, object node):
        """Return the live edge set for ``node`` without leaking absent keys.

        ``_in_map`` / ``_out_map`` are ``defaultdict(set)``, so a naive
        ``mp[node]`` access silently *inserts* an empty set for any node that
        is queried -- including transient and already-removed nodes. Over a
        long MCMC chain those phantom keys (and the ``Node`` objects they key
        on) accumulate without bound, ballooning every ``copy.deepcopy`` of
        the network in the undo hot path. Only auto-create the entry for nodes
        that are genuinely in V; everything else gets a fresh empty set that
        is not retained.
        """
        cdef set entry = mp.get(node)
        cdef set fresh
        if entry is not None:
            return entry
        if node in self._nodes:
            fresh = set()
            mp[node] = fresh
            return fresh
        return set()
    
    cpdef set in_edges(self, node):
        """
        Get the incoming edges of a node, or an empty set if it has none. O(1)
        
        Note: Returns a reference to the internal set. Copy before modifying
        the graph while iterating.
        """
        return self._edge_set(self._in_map, node)
    
    cpdef set out_edges(self, node):
        """
        Get the outgoing edges of a node, or an empty set if it has none. O(1)
        
        Note: Returns a reference to the internal set. Copy before modifying
        the graph while iterating.
        """
        return self._edge_set(self._out_map, node)
    
    cpdef void process(self, edge, bint removal=False):
        """
        Update the in/out maps to reflect an edge being added or removed.
        
        Raises:
            TypeError: If ``edge`` is not an Edge object.
            ValueError: If either endpoint is not in V.
        Args:
            edge: The Edge being added or removed.
            removal: True if removing, False if adding.
        """
        if type(edge).__name__ != 'Edge':
            raise TypeError("Tried to process wrong type of edge!")
        if not self.ready(edge):
            raise ValueError("Edge contains node not in network. "
                             "Add node first!")
        
        if not removal:
            self._out_map[edge.src].add(edge)
            self._in_map[edge.dest].add(edge)
        else:
            self._out_map[edge.src].discard(edge)
            self._in_map[edge.dest].discard(edge)
    
    cpdef set get_set(self):
        """Return the set of nodes (V)."""
        return self._nodes
    
    cpdef void remove(self, node):
        """Remove a node from V and drop its entries in the edge maps."""
        if node in self._nodes:
            self._nodes.discard(node)
            
            if node in self._out_map:
                del self._out_map[node]
            if node in self._in_map:
                del self._in_map[node]
            if node.label in self._node_names:
                del self._node_names[node.label]
    
    cpdef void update(self, node, str new_name):
        """
        Rename a node, rehashing it in every collection.
        
        ``Node.__hash__`` is name-based, so the node has to leave and re-enter
        each hash-keyed collection around the rename.
        
        Raises:
            ValueError: If ``node`` is not in V.
        Args:
            node: The Node to rename.
            new_name: The node's new name.
        """
        if node not in self._nodes:
            raise ValueError(f"Node {node.label} not found in NodeSet")
        
        cdef set in_edges = self._in_map.get(node, set()).copy()
        cdef set out_edges = self._out_map.get(node, set()).copy()
        
        self._nodes.discard(node)
        if node in self._in_map:
            del self._in_map[node]
        if node in self._out_map:
            del self._out_map[node]
        if node.label in self._node_names:
            del self._node_names[node.label]
        
        node.set_name(new_name)
        
        self._nodes.add(node)
        self._in_map[node] = in_edges
        self._out_map[node] = out_edges
        self._node_names[new_name] = node
    
    cpdef object get(self, str name):
        """Get a node by name, or None if there is no such node. O(1)"""
        return self._node_names.get(name)


cdef class EdgeSet:
    """
    The edge set E of a Network, plus a ``(src, dest) -> [Edge]`` lookup.
    
    The lookup maps to a *list* rather than a single edge because a bubble
    (two reticulation edges between the same pair of nodes) puts two distinct
    Edge objects under one key; :meth:`get` disambiguates them by gamma/tag.
    """
    
    cdef dict _hash      # dict[tuple[Node, Node], list[Edge]]
    cdef set _edges      # set[Edge]
    
    def __init__(self):
        """Initialize an empty edge set E."""
        self._hash = {}
        self._edges = set()
    
    def __contains__(self, e) -> bool:
        """Check if an edge is in E. O(1)"""
        return e in self._edges
    
    def add(self, *edges):
        """
        Add edges to E.
        
        Raises:
            TypeError: If any argument is not an Edge object.
        Args:
            *edges: Edge objects.
        """
        cdef tuple key
        for edge in edges:
            if type(edge).__name__ != 'Edge':
                raise TypeError("Networks are directed and hold only Edge "
                                f"objects. Got a {type(edge).__name__}.")
            if edge in self._edges:
                continue
            key = (edge.src, edge.dest)
            if key in self._hash:
                self._hash[key].append(edge)
            else:
                self._hash[key] = [edge]
            self._edges.add(edge)
    
    cpdef void remove(self, edge):
        """Remove an edge from E. Has no effect if it is not in E."""
        cdef tuple key
        
        if edge in self._edges:
            key = (edge.src, edge.dest)
            self._hash[key].remove(edge)
            if not self._hash[key]:
                del self._hash[key]
            self._edges.discard(edge)
    
    cpdef void rehash_node(self, node, affected_edges):
        """Rebuild the (src, dest) lookup after a node was renamed.

        A rename changes ``Node.__hash__`` (name-based), so every hash key
        that references *node* is now stale and unreachable. The set of live
        Edge objects is unaffected, so simply rebuild the map from it.
        ``affected_edges`` is accepted for call-site compatibility and is not
        needed here.
        """
        cdef dict rebuilt = {}
        cdef tuple key
        for edge in self._edges:
            key = (edge.src, edge.dest)
            if key in rebuilt:
                rebuilt[key].append(edge)
            else:
                rebuilt[key] = [edge]
        self._hash = rebuilt

    cpdef object get(self, n1, n2, gamma=None, tag=None):
        """
        Get the edge from n1 to n2.
        
        Raises:
            ValueError: If there is no such edge, or if a bubble cannot be
                        disambiguated by the given gamma/tag.
        Args:
            n1: Source node.
            n2: Destination node.
            gamma: Inheritance probability, to disambiguate a bubble.
            tag: Edge tag, to disambiguate a bubble whose gammas are equal.
        Returns:
            Edge: The matching edge.
        """
        cdef list valid_edges = self._hash.get((n1, n2))
        cdef int num_edges
        
        if valid_edges is None:
            raise ValueError("No matching edges found")
        num_edges = len(valid_edges)
        
        if num_edges == 1:
            return valid_edges[0]
        elif num_edges == 2:
            # Bubble: two edges share this (src, dest) key.
            if gamma is None:
                warnings.warn("Bubble lookup without gamma - returning first "
                              "edge")
                return valid_edges[0]
            
            if (valid_edges[0].get_gamma() == gamma
                    and valid_edges[1].get_gamma() == gamma):
                # Identical gammas, so only the tag can tell them apart.
                if valid_edges[0].get_tag() == tag:
                    return valid_edges[0]
                elif valid_edges[1].get_tag() == tag:
                    return valid_edges[1]
                else:
                    raise ValueError(f"Tags don't match: {tag}")
            elif valid_edges[0].get_gamma() == gamma:
                return valid_edges[0]
            elif valid_edges[1].get_gamma() == gamma:
                return valid_edges[1]
            else:
                raise ValueError("Gamma doesn't match any edge")
        else:
            raise ValueError("More than 2 edges found - invalid topology")
    
    cpdef set get_set(self):
        """Return the edge set E."""
        return self._edges
