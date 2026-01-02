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
Last Edit : 12/11/25
First Included in Version : 1.1.0
Docs   - [x]
Tests  - [ ]
Design - [x]

This module provides the NetworkAdapter class, which bridges the gap between
Network objects and Model computation without tight coupling.

The key insight is that we don't need every Network node to BE a ModelNode.
Instead, we can maintain a lightweight adapter that:
1. Holds per-node computation data (cached partials, etc.)
2. Tracks which nodes have been modified
3. Provides traversal methods for scoring algorithms

This approach solves the "Network-Model Synchronization Problem" by keeping
the Network's structure separate from computation concerns.
"""

from __future__ import annotations
from typing import Any, Callable, Iterator, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .Network import Network, Edge, Node


class NetworkAdapter:
    """
    Bridges Network and Model computation without tight coupling.
    
    Instead of making every Network Node inherit from ModelNode (which creates
    synchronization issues when topology changes), this adapter maintains
    computation data externally to the network structure.
    
    Benefits:
    - Network topology changes don't break model node references
    - Computation data is cleanly separated from network structure
    - Easy to swap networks or clear all cached data
    - Works with both caching and non-caching scoring strategies
    """
    
    def __init__(self, network = None) -> None:
        """
        Create a NetworkAdapter for the given network.
        
        Args:
            network (Network, optional): The phylogenetic network to adapt.
                                         Can be set later via set_network().
        Returns:
            N/A
        """
        self._network = network
        
        # Per-node cached computations (partial likelihoods, etc.)
        self._node_data: dict = defaultdict(dict)
        
        # Per-edge cached computations (transition matrices, etc.)
        self._edge_data: dict = defaultdict(dict)
        
        # Track which nodes have been modified since last computation
        self._dirty_nodes: set = set()
        
        # Callbacks for topology changes
        self._topology_listeners: list[Callable] = []
    
    @property
    def network(self):
        """
        Get the underlying network.
        
        Args:
            N/A
        Returns:
            Network: The adapted network.
        """
        return self._network
    
    def set_network(self, network) -> None:
        """
        Set or replace the underlying network.
        
        Clears all cached data when the network changes.
        
        Args:
            network (Network): The new network to adapt.
        Returns:
            N/A
        """
        self._network = network
        self.clear_all_data()
    
    def get_node_data(self, node, key: str, default: Any = None) -> Any:
        """
        Retrieve cached data for a node.
        
        Args:
            node (Node): The network node.
            key (str): The data key (e.g., "partial_likelihood", "time").
            default (Any, optional): Default if not found.
        Returns:
            Any: The cached value, or default.
        """
        return self._node_data[node].get(key, default)
    
    def set_node_data(self, node, key: str, value: Any) -> None:
        """
        Store cached data for a node.
        
        Args:
            node (Node): The network node.
            key (str): The data key.
            value (Any): The value to cache.
        Returns:
            N/A
        """
        self._node_data[node][key] = value
        self._dirty_nodes.discard(node)
    
    def has_node_data(self, node, key: str) -> bool:
        """
        Check if cached data exists for a node.
        
        Args:
            node (Node): The network node.
            key (str): The data key.
        Returns:
            bool: True if data exists.
        """
        return key in self._node_data[node]
    
    def get_edge_data(self, edge, key: str, default: Any = None) -> Any:
        """
        Retrieve cached data for an edge.
        
        Args:
            edge (Edge): The network edge.
            key (str): The data key.
            default (Any, optional): Default if not found.
        Returns:
            Any: The cached value, or default.
        """
        return self._edge_data[edge].get(key, default)
    
    def set_edge_data(self, edge, key: str, value: Any) -> None:
        """
        Store cached data for an edge.
        
        Args:
            edge (Edge): The network edge.
            key (str): The data key.
            value (Any): The value to cache.
        Returns:
            N/A
        """
        self._edge_data[edge][key] = value
    
    def mark_dirty(self, nodes: list) -> None:
        """
        Mark nodes as needing recomputation.
        
        Also marks all ancestor nodes as dirty (upstream propagation).
        
        Args:
            nodes (list[Node]): Nodes that have been modified.
        Returns:
            N/A
        """
        for node in nodes:
            self._mark_dirty_recursive(node)
        
        # Notify listeners
        for listener in self._topology_listeners:
            listener(nodes)
    
    def _mark_dirty_recursive(self, node) -> None:
        """
        Recursively mark a node and its ancestors as dirty.
        
        Args:
            node (Node): The starting node.
        Returns:
            N/A
        """
        if node in self._dirty_nodes:
            return  # Already marked, ancestors must be too
        
        self._dirty_nodes.add(node)
        
        # Clear cached data for this node
        if node in self._node_data:
            self._node_data[node].clear()
        
        # Propagate to parents
        if self._network is not None:
            parents = self._network.get_parents(node)
            if parents:
                for parent in parents:
                    self._mark_dirty_recursive(parent)
    
    def is_dirty(self, node) -> bool:
        """
        Check if a node needs recomputation.
        
        Args:
            node (Node): The node to check.
        Returns:
            bool: True if the node needs recomputation.
        """
        return node in self._dirty_nodes
    
    def clear_dirty(self, node = None) -> None:
        """
        Clear the dirty flag for a node (or all nodes).
        
        Args:
            node (Node, optional): Specific node to clear. If None, clears all.
        Returns:
            N/A
        """
        if node is None:
            self._dirty_nodes.clear()
        else:
            self._dirty_nodes.discard(node)
    
    def on_topology_change(self, affected: list) -> None:
        """
        Handle network topology changes.
        
        Called when nodes/edges are added or removed from the network.
        Clears caches for affected nodes and their ancestors.
        
        Args:
            affected (list[Node]): Nodes affected by the topology change.
        Returns:
            N/A
        """
        if self._network is None:
            return
            
        # Remove data for nodes that may no longer exist
        network_nodes = set(self._network.V())
        nodes_to_remove = [n for n in self._node_data.keys() 
                          if n not in network_nodes]
        for node in nodes_to_remove:
            del self._node_data[node]
            self._dirty_nodes.discard(node)
        
        # Remove data for edges that may no longer exist
        network_edges = set(self._network.E())
        edges_to_remove = [e for e in self._edge_data.keys()
                          if e not in network_edges]
        for edge in edges_to_remove:
            del self._edge_data[edge]
        
        # Mark affected nodes as dirty
        self.mark_dirty(affected)
    
    def clear_all_data(self) -> None:
        """
        Clear all cached computation data.
        
        Args:
            N/A
        Returns:
            N/A
        """
        self._node_data.clear()
        self._edge_data.clear()
        self._dirty_nodes.clear()
    
    def add_topology_listener(self, callback: Callable) -> None:
        """
        Register a callback for topology changes.
        
        Args:
            callback (Callable): Function called with list of affected nodes.
        Returns:
            N/A
        """
        self._topology_listeners.append(callback)
    
    def remove_topology_listener(self, callback: Callable) -> None:
        """
        Remove a topology change callback.
        
        Args:
            callback (Callable): The callback to remove.
        Returns:
            N/A
        """
        if callback in self._topology_listeners:
            self._topology_listeners.remove(callback)
    
    # Convenience traversal methods
    
    def postorder(self) -> Iterator:
        """
        Iterate through nodes in postorder (leaves first, root last).
        
        Useful for bottom-up computations like likelihood.
        
        Args:
            N/A
        Yields:
            Node: Nodes in postorder.
        """
        if self._network is None:
            return
        
        visited = set()
        
        def visit(node) -> Iterator:
            if node in visited:
                return
            visited.add(node)
            
            # Visit children first
            children = self._network.get_children(node)
            if children:
                for child in children:
                    yield from visit(child)
            
            yield node
        
        root = self._network.root()
        if root is not None:
            yield from visit(root)
    
    def preorder(self) -> Iterator:
        """
        Iterate through nodes in preorder (root first, leaves last).
        
        Useful for top-down computations like simulation.
        
        Args:
            N/A
        Yields:
            Node: Nodes in preorder.
        """
        if self._network is None:
            return
        
        visited = set()
        
        def visit(node) -> Iterator:
            if node in visited:
                return
            visited.add(node)
            
            yield node
            
            # Visit children after
            children = self._network.get_children(node)
            if children:
                for child in children:
                    yield from visit(child)
        
        root = self._network.root()
        if root is not None:
            yield from visit(root)
    
    def leaves(self) -> list:
        """
        Get all leaf nodes.
        
        Args:
            N/A
        Returns:
            list[Node]: The leaf nodes.
        """
        if self._network is None:
            return []
        return self._network.get_leaves()
    
    def root(self):
        """
        Get the root node.
        
        Args:
            N/A
        Returns:
            Node: The root node, or None.
        """
        if self._network is None:
            return None
        return self._network.root()


class NetworkObserver:
    """
    Mixin or wrapper that can be used to make Network notify adapters of changes.
    
    This can be used to automatically keep adapters in sync with network changes.
    """
    
    def __init__(self, network):
        """
        Wrap a network to observe changes.
        
        Args:
            network (Network): The network to observe.
        Returns:
            N/A
        """
        self._network = network
        self._adapters: list[NetworkAdapter] = []
    
    def register_adapter(self, adapter: NetworkAdapter) -> None:
        """
        Register an adapter to be notified of changes.
        
        Args:
            adapter (NetworkAdapter): The adapter to register.
        Returns:
            N/A
        """
        self._adapters.append(adapter)
    
    def unregister_adapter(self, adapter: NetworkAdapter) -> None:
        """
        Unregister an adapter.
        
        Args:
            adapter (NetworkAdapter): The adapter to unregister.
        Returns:
            N/A
        """
        if adapter in self._adapters:
            self._adapters.remove(adapter)
    
    def notify_change(self, affected: list) -> None:
        """
        Notify all registered adapters of a topology change.
        
        Args:
            affected (list[Node]): Nodes affected by the change.
        Returns:
            N/A
        """
        for adapter in self._adapters:
            adapter.on_topology_change(affected)

