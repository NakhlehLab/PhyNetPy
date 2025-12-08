#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --                                                              
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##############################################################################

"""
Optimized Network module with inner class architecture.
Author: Mark Kessler
Last Edit: Optimized version
"""

from __future__ import annotations
from collections import defaultdict, deque
import copy
import math
from typing import Any, Callable, Union, Optional, Set, Tuple
import warnings
import numpy as np
import sys
import networkx as nx
from MSA import DataSequence
from itertools import combinations
import tempfile

sys.setrecursionlimit(300)

#############################
####  EXCEPTION CLASSES  ####
#############################

class NetworkError(Exception):
    """Exception raised when a network operation fails."""
    def __init__(self, message: str = "Error operating on a Network") -> None:
        self.message = message
        super().__init__(self.message)

##########################
#### HELPER FUNCTIONS ####
##########################

def random_object(mylist: list[Any], rng: np.random.Generator) -> Any:
    """Select a random item from a list using an rng object."""
    rand_index = rng.integers(0, len(mylist))
    return mylist[rand_index]

#########################
#### NETWORK CLASS ####
#########################

class Network:
    """
    Directed phylogenetic network with inner Node and Edge classes.
    Nodes and edges are bound to their parent network, preventing bookkeeping errors.
    """
    
    class Node:
        """Node that only exists within a Network context."""
        
        __slots__ = ('_network', '_label', '_time', '_is_retic', '_attributes', '_seq')
        
        def __init__(self, network: 'Network', label: str, **kwargs):
            self._network = network
            self._label = label
            self._time = kwargs.get('time', None)
            self._is_retic = kwargs.get('is_reticulation', False)
            self._attributes = kwargs.get('attributes', {})
            self._seq = kwargs.get('seq', None)
        
        @property
        def label(self) -> str:
            return self._label
        
        @property
        def network(self) -> 'Network':
            return self._network
        
        def get_parents(self) -> list['Network.Node']:
            """Get parent nodes efficiently from network topology."""
            return self._network._get_parents(self)
        
        def get_children(self) -> list['Network.Node']:
            """Get child nodes efficiently from network topology."""
            return self._network._get_children(self)
        
        @property
        def in_degree(self) -> int:
            """Get in-degree in O(1) time."""
            return len(self._network._adjacency[self]['in'])
        
        @property
        def out_degree(self) -> int:
            """Get out-degree in O(1) time."""
            return len(self._network._adjacency[self]['out'])
        
        @property
        def in_edges(self) -> list['Network.Edge']:
            """Get incoming edges in O(1) time."""
            return self._network._adjacency[self]['in']
        
        @property
        def out_edges(self) -> list['Network.Edge']:
            """Get outgoing edges in O(1) time."""
            return self._network._adjacency[self]['out']
        
        def is_reticulation(self) -> bool:
            return self._is_retic
        
        def set_is_reticulation(self, is_retic: bool) -> None:
            self._is_retic = is_retic
        
        def get_time(self) -> float:
            if self._time is None:
                raise NetworkError("No time has been set for this node!")
            return self._time
        
        def set_time(self, new_t: float) -> None:
            if new_t < 0:
                raise NetworkError("Time must be non-negative!")
            self._time = new_t
        
        def get_attributes(self) -> dict:
            return self._attributes
        
        def set_attributes(self, new_attr: dict) -> None:
            self._attributes = new_attr
        
        def add_attribute(self, key: Any, value: Any, append: bool = False) -> None:
            if append and key in self._attributes:
                content = self._attributes[key]
                if isinstance(content, dict):
                    self._attributes[key] = {**content, **value}
                elif isinstance(content, list):
                    content.extend(value)
            else:
                self._attributes[key] = value
        
        def attribute_value(self, key: Any) -> Any:
            return self._attributes.get(key, None)
        
        def set_seq(self, new_sequence: DataSequence) -> None:
            self._seq = new_sequence
        
        def get_seq(self) -> DataSequence:
            if self._seq is None:
                raise NetworkError("No sequence associated with this node!")
            return self._seq
        
        def __str__(self) -> str:
            return self._label
        
        def __repr__(self) -> str:
            return f"Node({self._label}@{id(self._network):#x})"
        
        def __eq__(self, other) -> bool:
            if not isinstance(other, Network.Node):
                return False
            return self._label == other._label and self._network is other._network
        
        def __hash__(self) -> int:
            return hash((self._label, id(self._network)))
    
    class Edge:
        """Edge that only exists within a Network context."""
        
        __slots__ = ('_network', '_src', '_dest', '_length', '_gamma', '_weight', '_tag')
        
        def __init__(self, network: 'Network', src: 'Network.Node', 
                     dest: 'Network.Node', **kwargs):
            self._network = network
            self._src = src
            self._dest = dest
            self._length = kwargs.get('length', 1.0)
            self._gamma = kwargs.get('gamma', 0.0)
            self._weight = kwargs.get('weight', 0.0)
            self._tag = kwargs.get('tag', "")
            
            # Validate gamma
            if not 0 <= self._gamma <= 1:
                raise ValueError("Gamma must be between 0 and 1!")
        
        @property
        def src(self) -> 'Network.Node':
            return self._src
        
        @property
        def dest(self) -> 'Network.Node':
            return self._dest
        
        def get_length(self) -> float:
            return self._length
        
        def set_length(self, length: float, enforce_times: bool = False) -> None:
            if enforce_times:
                try:
                    expected = self._dest.get_time() - self._src.get_time()
                    if abs(length - expected) >= 1e-5:
                        raise NetworkError("Length doesn't match node times!")
                except NetworkError:
                    pass
            self._length = length
        
        def get_gamma(self) -> float:
            return self._gamma
        
        def set_gamma(self, gamma: float) -> None:
            if not 0 <= gamma <= 1:
                raise ValueError("Gamma must be between 0 and 1!")
            self._gamma = gamma
        
        def get_weight(self) -> float:
            return self._weight
        
        def set_weight(self, weight: float) -> None:
            self._weight = weight
        
        def get_tag(self) -> str:
            return self._tag
        
        def set_tag(self, tag: str) -> None:
            self._tag = tag
        
        def to_names(self) -> tuple[str, str]:
            return (self._src.label, self._dest.label)
        
        def __repr__(self) -> str:
            return f"Edge({self._src.label}->{self._dest.label})"
        
        def __eq__(self, other) -> bool:
            if not isinstance(other, Network.Edge):
                return False
            return (self._src == other._src and self._dest == other._dest 
                   and self._network is other._network)
        
        def __hash__(self) -> int:
            return hash((self._src, self._dest, id(self._network)))
    
    def __init__(self):
        """Initialize empty network with optimized data structures."""
        self._nodes: dict[str, Network.Node] = {}
        self._edges: set[Network.Edge] = set()
        
        # Single source of truth for topology - O(1) lookups
        self._adjacency: dict[Network.Node, dict[str, list[Network.Edge]]] = defaultdict(
            lambda: {'in': [], 'out': []}
        )
        
        # Cache frequently accessed node sets
        self._leaves_cache: Optional[list[Network.Node]] = None
        self._roots_cache: Optional[list[Network.Node]] = None
        self._cache_valid = True
        
        # For unique ID generation
        self._uid_counter = 0
    
    @classmethod
    def empty(cls) -> 'Network':
        """Create an empty network."""
        return cls()
    
    @classmethod
    def from_newick(cls, newick_string: str) -> 'Network':
        """Parse a newick string into a network."""
        network = cls()
        # Parse newick and build network
        # This would use your existing newick parsing logic
        network._parse_newick(newick_string)
        return network
    
    @classmethod
    def from_nexus(cls, filepath: str, tree_index: int = 0) -> 'Network':
        """Load a network from a nexus file."""
        # This would use your existing nexus parsing logic
        with open(filepath, 'r') as f:
            # Parse nexus and extract newick
            newick = ""  # Extract from nexus
            return cls.from_newick(newick)
    
    def _invalidate_cache(self):
        """Invalidate cached values when topology changes."""
        self._cache_valid = False
        self._leaves_cache = None
        self._roots_cache = None
    
    def _update_caches(self):
        """Update cached values - O(n) but only when needed."""
        if not self._cache_valid:
            self._leaves_cache = [n for n in self._nodes.values() if n.out_degree == 0]
            self._roots_cache = [n for n in self._nodes.values() if n.in_degree == 0]
            self._cache_valid = True
    
    def create_node(self, label: str, **kwargs) -> Node:
        """Create a node bound to this network - O(1)."""
        if label in self._nodes:
            raise NetworkError(f"Node {label} already exists")
        
        node = self.Node(self, label, **kwargs)
        self._nodes[label] = node
        self._adjacency[node] = {'in': [], 'out': []}
        self._invalidate_cache()
        return node
    
    def create_edge(self, src_label: str, dest_label: str, **kwargs) -> Edge:
        """Create an edge bound to this network - O(1)."""
        src = self.get_node(src_label)
        dest = self.get_node(dest_label)
        
        if src is None or dest is None:
            raise NetworkError("Both nodes must exist before creating edge")
        
        # Check for duplicate edges (bubble detection)
        existing = [e for e in self._adjacency[src]['out'] if e.dest == dest]
        if len(existing) >= 2:
            raise NetworkError("Cannot add more than 2 edges between same nodes (bubble limit)")
        
        edge = self.Edge(self, src, dest, **kwargs)
        self._edges.add(edge)
        
        # Update topology - O(1)
        self._adjacency[src]['out'].append(edge)
        self._adjacency[dest]['in'].append(edge)
        
        # Update reticulation status
        if len(self._adjacency[dest]['in']) > 1:
            dest.set_is_reticulation(True)
        
        self._invalidate_cache()
        return edge
    
    def add_node(self, label: str, **kwargs) -> Node:
        """Add or get existing node - O(1)."""
        if label in self._nodes:
            return self._nodes[label]
        return self.create_node(label, **kwargs)
    
    def add_edge(self, src_label: str, dest_label: str, **kwargs) -> Edge:
        """Add edge, creating nodes if needed - O(1)."""
        self.add_node(src_label)
        self.add_node(dest_label)
        return self.create_edge(src_label, dest_label, **kwargs)
    
    def remove_node(self, node: Node) -> None:
        """Remove node and all its edges - O(degree)."""
        if node not in self._adjacency:
            return
        
        # Remove all incident edges
        for edge in list(self._adjacency[node]['in']):
            self.remove_edge(edge)
        for edge in list(self._adjacency[node]['out']):
            self.remove_edge(edge)
        
        # Remove node
        del self._nodes[node.label]
        del self._adjacency[node]
        self._invalidate_cache()
    
    def remove_edge(self, edge: Edge) -> None:
        """Remove edge from network - O(1) average."""
        if edge not in self._edges:
            return
        
        self._edges.remove(edge)
        self._adjacency[edge.src]['out'].remove(edge)
        self._adjacency[edge.dest]['in'].remove(edge)
        
        # Update reticulation status
        if len(self._adjacency[edge.dest]['in']) <= 1:
            edge.dest.set_is_reticulation(False)
        
        self._invalidate_cache()
    
    def get_node(self, label: str) -> Optional[Node]:
        """Get node by label - O(1)."""
        return self._nodes.get(label)
    
    def get_edge(self, src_label: str, dest_label: str, 
                 gamma: Optional[float] = None, tag: Optional[str] = None) -> Optional[Edge]:
        """Get edge between nodes - O(degree)."""
        src = self.get_node(src_label)
        dest = self.get_node(dest_label)
        
        if src is None or dest is None:
            return None
        
        candidates = [e for e in self._adjacency[src]['out'] if e.dest == dest]
        
        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]
        else:  # Bubble - need to disambiguate
            if gamma is not None:
                for e in candidates:
                    if abs(e.get_gamma() - gamma) < 1e-10:
                        return e
            if tag is not None:
                for e in candidates:
                    if e.get_tag() == tag:
                        return e
            # Return first if no disambiguation
            return candidates[0]
    
    def _get_parents(self, node: Node) -> list[Node]:
        """Internal method for Node.get_parents() - O(in_degree)."""
        return [e.src for e in self._adjacency[node]['in']]
    
    def _get_children(self, node: Node) -> list[Node]:
        """Internal method for Node.get_children() - O(out_degree)."""
        return [e.dest for e in self._adjacency[node]['out']]
    
    def V(self) -> list[Node]:
        """Get all nodes - O(1)."""
        return list(self._nodes.values())
    
    def E(self) -> list[Edge]:
        """Get all edges - O(1)."""
        return list(self._edges)
    
    @property
    def roots(self) -> list[Node]:
        """Get root nodes with caching - O(1) after first call."""
        self._update_caches()
        return self._roots_cache or []
    
    @property
    def leaves(self) -> list[Node]:
        """Get leaf nodes with caching - O(1) after first call."""
        self._update_caches()
        return self._leaves_cache or []
    
    def root(self) -> Node:
        """Get the root (assumes single root) - O(1) after first call."""
        roots_list = self.roots
        if not roots_list:
            raise NetworkError("No root found (cycle or empty network)")
        if len(roots_list) > 1:
            warnings.warn("Multiple roots found, returning first")
        return roots_list[0]
    
    def get_leaves(self) -> list[Node]:
        """Get connected leaves - O(1) after first call."""
        return [leaf for leaf in self.leaves if leaf.in_degree > 0]
    
    def add_uid_node(self, node: Optional[Node] = None) -> Node:
        """Add node with unique ID - O(1)."""
        if node is None:
            label = f"UID_{self._uid_counter}"
            self._uid_counter += 1
            return self.create_node(label)
        else:
            new_label = f"UID_{self._uid_counter}"
            self._uid_counter += 1
            if node.label in self._nodes:
                # Rename existing node
                del self._nodes[node.label]
                node._label = new_label
                self._nodes[new_label] = node
            else:
                node._label = new_label
                self._nodes[new_label] = node
                self._adjacency[node] = {'in': [], 'out': []}
            return node
    
    def in_degree(self, node: Node) -> int:
        """Get in-degree - O(1)."""
        return len(self._adjacency[node]['in'])
    
    def out_degree(self, node: Node) -> int:
        """Get out-degree - O(1)."""
        return len(self._adjacency[node]['out'])
    
    def in_edges(self, node: Node) -> list[Edge]:
        """Get incoming edges - O(1)."""
        return self._adjacency[node]['in']
    
    def out_edges(self, node: Node) -> list[Edge]:
        """Get outgoing edges - O(1)."""
        return self._adjacency[node]['out']
    
    def get_parents(self, node: Node) -> list[Node]:
        """Get parent nodes - O(in_degree)."""
        return self._get_parents(node)
    
    def get_children(self, node: Node) -> list[Node]:
        """Get child nodes - O(out_degree)."""
        return self._get_children(node)
    
    def is_acyclic(self) -> bool:
        """Check if network is acyclic using DFS - O(V+E)."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: Node) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for child in self.get_children(node):
                if child not in visited:
                    if has_cycle(child):
                        return True
                elif child in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for root in self.roots:
            if root not in visited:
                if has_cycle(root):
                    return False
        
        return True
    
    def topological_order(self) -> list[Node]:
        """Get topological ordering using Kahn's algorithm - O(V+E)."""
        in_degree = {n: len(self._adjacency[n]['in']) for n in self._nodes.values()}
        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []
        
        while queue:
            node = queue.popleft()
            order.append(node)
            
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        if len(order) != len(self._nodes):
            raise NetworkError("Graph has cycles")
        
        return order
    
    def mrca(self, nodes: Set[Union[Node, str]]) -> Node:
        """Find MRCA using efficient LCA algorithm - O(n log n)."""
        # Convert strings to nodes
        node_set = set()
        for item in nodes:
            if isinstance(item, str):
                node = self.get_node(item)
                if node is None:
                    raise NetworkError(f"Node {item} not found")
                node_set.add(node)
            else:
                node_set.add(item)
        
        if not node_set:
            raise NetworkError("Empty node set")
        
        # Use path intersection method
        ancestors = None
        for node in node_set:
            node_ancestors = set()
            current = [node]
            visited = set()
            
            while current:
                next_level = []
                for n in current:
                    if n not in visited:
                        visited.add(n)
                        node_ancestors.add(n)
                        next_level.extend(self.get_parents(n))
                current = next_level
            
            if ancestors is None:
                ancestors = node_ancestors
            else:
                ancestors &= node_ancestors
        
        # Find lowest ancestor
        if not ancestors:
            raise NetworkError("No common ancestor found")
        
        # Find the ancestor furthest from root
        best = None
        best_dist = -1
        for anc in ancestors:
            dist = self.distance_from_root(anc, use_time=False)
            if dist > best_dist:
                best_dist = dist
                best = anc
        
        return best
    
    def leaf_descendants(self, node: Node) -> set[Node]:
        """Get all leaf descendants using DFS - O(subtree size)."""
        leaves = set()
        stack = [node]
        visited = set()
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            children = self.get_children(current)
            if not children:
                leaves.add(current)
            else:
                stack.extend(children)
        
        return leaves
    
    def distance_from_root(self, node: Node, use_time: bool = True) -> float:
        """Compute distance from root - O(path length)."""
        if node not in self._nodes.values():
            raise NetworkError("Node not in network")
        
        root = self.root()
        if root == node:
            return 0.0
        
        # BFS to find shortest path
        visited = set()
        queue = deque([(root, 0.0)])
        visited.add(root)
        
        while queue:
            current, dist = queue.popleft()
            
            for child in self.get_children(current):
                if child == node:
                    if use_time:
                        edge = next(e for e in self._adjacency[current]['out'] if e.dest == child)
                        return dist + edge.get_length()
                    else:
                        return dist + 1.0
                
                if child not in visited:
                    visited.add(child)
                    if use_time:
                        edge = next(e for e in self._adjacency[current]['out'] if e.dest == child)
                        new_dist = dist + edge.get_length()
                    else:
                        new_dist = dist + 1.0
                    queue.append((child, new_dist))
        
        raise NetworkError(f"No path from root to {node.label}")
    
    def clean(self, options: list[bool] = [True, True, True]) -> None:
        """Clean network topology - O(V+E)."""
        if options[0]:  # Remove floater nodes
            floaters = [n for n in self._nodes.values() 
                       if n.in_degree == 0 and n.out_degree == 0]
            for floater in floaters:
                self.remove_node(floater)
        
        if options[1]:  # Remove spurious root
            try:
                root = self.root()
                if root.out_degree == 1:
                    child = self.get_children(root)[0]
                    self.remove_node(root)
            except NetworkError:
                pass
        
        if options[2]:  # Consolidate degree-1 chains
            changed = True
            while changed:
                changed = False
                for node in list(self._nodes.values()):
                    if node.in_degree == 1 and node.out_degree == 1:
                        parent = self.get_parents(node)[0]
                        child = self.get_children(node)[0]
                        
                        # Get edge lengths
                        e1 = self._adjacency[parent]['out'][0]
                        e2 = self._adjacency[node]['out'][0]
                        new_length = e1.get_length() + e2.get_length()
                        
                        self.remove_node(node)
                        self.create_edge(parent.label, child.label, length=new_length)
                        changed = True
                        break
    
    def copy(self) -> tuple['Network', dict[Node, Node]]:
        """Deep copy network - O(V+E)."""
        new_net = Network()
        old_to_new = {}
        
        # Copy nodes
        for old_node in self._nodes.values():
            new_node = new_net.create_node(
                old_node.label,
                time=old_node._time,
                is_reticulation=old_node._is_retic,
                attributes=copy.deepcopy(old_node._attributes),
                seq=old_node._seq
            )
            old_to_new[old_node] = new_node
        
        # Copy edges
        for old_edge in self._edges:
            new_net.create_edge(
                old_edge.src.label,
                old_edge.dest.label,
                length=old_edge._length,
                gamma=old_edge._gamma,
                weight=old_edge._weight,
                tag=old_edge._tag
            )
        
        return new_net, old_to_new
    
    def subnet(self, root_node: Node) -> 'Network':
        """Extract subnetwork rooted at node - O(subtree size)."""
        subnet = Network()
        visited = set()
        queue = deque([root_node])
        
        # Map old to new nodes
        old_to_new = {}
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            # Create node in subnet
            if current not in old_to_new:
                new_node = subnet.create_node(
                    f"{current.label}_copy",
                    time=current._time,
                    is_reticulation=current._is_retic,
                    attributes=copy.deepcopy(current._attributes)
                )
                old_to_new[current] = new_node
            
            # Add children and edges
            for child in self.get_children(current):
                if child not in old_to_new:
                    new_child = subnet.create_node(
                        f"{child.label}_copy",
                        time=child._time,
                        is_reticulation=child._is_retic,
                        attributes=copy.deepcopy(child._attributes)
                    )
                    old_to_new[child] = new_child
                
                # Copy edge
                old_edge = next(e for e in self._adjacency[current]['out'] if e.dest == child)
                subnet.create_edge(
                    old_to_new[current].label,
                    old_to_new[child].label,
                    length=old_edge._length,
                    gamma=old_edge._gamma,
                    weight=old_edge._weight
                )
                
                queue.append(child)
        
        return subnet
    
    def newick(self) -> str:
        """Generate newick string - O(V+E)."""
        if not self._nodes:
            return ";"
        
        def build_newick(node: Node, visited_retics: set) -> str:
            if node.out_degree == 0:  # Leaf
                return node.label
            
            # Handle reticulation nodes
            if node.in_degree >= 2:
                if node in visited_retics:
                    return f"#{node.label}" if not node.label.startswith("#") else node.label
                visited_retics.add(node)
                node_label = f"#{node.label}" if not node.label.startswith("#") else node.label
            else:
                node_label = node.label
            
            # Build subtree strings
            children_newicks = []
            for child in self.get_children(node):
                children_newicks.append(build_newick(child, visited_retics))
            
            return f"({','.join(children_newicks)}){node_label}"
        
        return build_newick(self.root(), set()) + ";"
    
    def to_networkx(self) -> nx.MultiDiGraph:
        """Convert to NetworkX graph - O(V+E)."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([n.label for n in self._nodes.values()])
        G.add_edges_from([e.to_names() for e in self._edges])
        return G
    
    def bfs_dfs(self, start_node: Optional[Node] = None, dfs: bool = False,
                accumulator: Optional[Callable] = None, 
                accumulated: Any = None) -> tuple[dict[Node, int], Any]:
        """Generic graph traversal - O(V+E)."""
        if start_node is None:
            start_node = self.root()
        
        visited = set()
        queue = deque([start_node])
        distances = {start_node: 0}
        
        while queue:
            current = queue.popleft() if dfs else queue.pop()
            
            if current in visited:
                continue
            visited.add(current)
            
            if accumulator is not None:
                accumulated = accumulator(current, accumulated)
            
            for child in self.get_children(current):
                if child not in visited:
                    distances[child] = distances[current] + 1
                    queue.append(child)
        
        return distances, accumulated
    
    def subgenome_count(self, node: Node) -> int:
        """Count subgenomes for a node - O(ancestors)."""
        if node not in self._nodes.values():
            raise NetworkError("Node not in network")
        
        if node == self.root():
            return 1
        
        parents = self.get_parents(node)
        return sum(self.subgenome_count(p) for p in parents)
    
    def edges_downstream_of_node(self, node: Node) -> list[Edge]:
        """Get all edges in subgraph rooted at node - O(subtree size)."""
        edges = []
        visited = set()
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            for edge in self._adjacency[current]['out']:
                edges.append(edge)
                queue.append(edge.dest)
        
        return edges
    
    def edges_upstream_of_node(self, node: Node) -> list[Edge]:
        """Get all edges from root to node - O(ancestors)."""
        edges = []
        visited = set()
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            for edge in self._adjacency[current]['in']:
                edges.append(edge)
                queue.append(edge.src)
        
        return edges
    
    def diff_subtree_edges(self, rng: np.random.Generator) -> list[Edge]:
        """Get 2 random edges from different subtrees - O(E)."""
        if len(self._edges) < 2:
            raise NetworkError("Need at least 2 edges")
        
        edges_list = list(self._edges)
        first_edge = random_object(edges_list, rng)
        
        # Find edges not reachable from first edge
        first_subtree_leaves = self.leaf_descendants(first_edge.dest)
        
        valid_edges = []
        for edge in edges_list:
            if edge != first_edge:
                edge_subtree_leaves = self.leaf_descendants(edge.dest)
                if not first_subtree_leaves.intersection(edge_subtree_leaves):
                    valid_edges.append(edge)
        
        if not valid_edges:
            raise NetworkError("No valid second edge found")
        
        return [first_edge, random_object(valid_edges, rng)]
    
    def set_node_times_from_root(self) -> None:
        """Set node times based on cumulative branch lengths - O(V+E)."""
        root = self.root()
        root.set_time(0.0)
        
        visited = set([root])
        queue = deque([(root, 0.0)])
        
        while queue:
            current, current_time = queue.popleft()
            
            for edge in self._adjacency[current]['out']:
                child = edge.dest
                if child not in visited:
                    visited.add(child)
                    child_time = current_time + edge.get_length()
                    child.set_time(child_time)
                    queue.append((child, child_time))
    
    def _parse_newick(self, newick: str) -> None:
        """Parse newick string into network - implementation needed."""
        # This would contain your newick parsing logic
        pass
    
    def __contains__(self, item: Union[Node, Edge]) -> bool:
        """Check if item is in network - O(1)."""
        if isinstance(item, Network.Node):
            return item in self._adjacency
        elif isinstance(item, Network.Edge):
            return item in self._edges
        return False
    
    def __str__(self) -> str:
        """String representation."""
        return f"Network(nodes={len(self._nodes)}, edges={len(self._edges)})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Network(nodes={list(self._nodes.keys())[:5]}..., edges={len(self._edges)})"