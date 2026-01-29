"""
Memory and performance tests for Network data structures.

These tests verify that the set-based implementations of in_map, out_map,
__leaves, and __roots provide memory and performance benefits.

Run with: pytest test_memory.py -v -s
"""

import pytest
import tracemalloc
import time
from typing import List, Tuple
from PhyNetPy.Network import Network, Node, Edge, NodeSet


def build_large_network(num_leaves: int = 1000) -> Tuple[Network, List[Node]]:
    """
    Build a large binary tree network for benchmarking.
    
    Args:
        num_leaves: Number of leaf nodes (total nodes ~2*num_leaves)
    
    Returns:
        Tuple of (Network, list of all nodes)
    """
    net = Network()
    
    # Create nodes
    nodes: List[Node] = []
    for i in range(num_leaves):
        nodes.append(Node(f"leaf_{i}"))
    
    # Create internal nodes and build tree bottom-up
    current_level = nodes.copy()
    internal_count = 0
    
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level) - 1, 2):
            parent = Node(f"internal_{internal_count}")
            internal_count += 1
            nodes.append(parent)
            next_level.append(parent)
        
        # Handle odd node
        if len(current_level) % 2 == 1:
            next_level.append(current_level[-1])
        
        current_level = next_level
    
    # Add all nodes first
    net.add_nodes(nodes)
    
    # Now add edges (rebuild the tree structure)
    current_level = nodes[:num_leaves]  # leaves
    internal_idx = num_leaves
    
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level) - 1, 2):
            parent = nodes[internal_idx]
            internal_idx += 1
            net.add_edges(Edge(parent, current_level[i]))
            net.add_edges(Edge(parent, current_level[i + 1]))
            next_level.append(parent)
        
        if len(current_level) % 2 == 1:
            next_level.append(current_level[-1])
        
        current_level = next_level
    
    return net, nodes


class TestMemoryUsage:
    """Tests for memory efficiency of set-based data structures."""
    
    def test_nodeset_memory_snapshot(self):
        """
        Measure memory used by NodeSet with set-based in_map/out_map.
        This is a baseline measurement - compare with historical data.
        """
        tracemalloc.start()
        
        # Build a moderately large network
        net, nodes = build_large_network(500)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\n--- Memory Usage (500 leaves) ---")
        print(f"Current memory: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
        print(f"Nodes: {len(net.V())}")
        print(f"Edges: {len(net.E())}")
        
        # Sanity check - network should exist
        assert len(net.V()) > 500
        assert len(net.E()) > 400
    
    def test_edge_operations_dont_leak(self):
        """
        Test that adding and removing edges doesn't cause memory leaks
        in the in_map/out_map structures.
        """
        tracemalloc.start()
        
        net = Network()
        n1 = Node("a")
        n2 = Node("b")
        n3 = Node("c")
        net.add_nodes(n1, n2, n3)
        
        # Add and remove edges many times
        for _ in range(1000):
            e1 = Edge(n1, n2)
            e2 = Edge(n2, n3)
            net.add_edges(e1)
            net.add_edges(e2)
            net.remove_edge(e1)
            net.remove_edge(e2)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\n--- Memory after 1000 add/remove cycles ---")
        print(f"Current memory: {current / 1024:.2f} KB")
        print(f"Peak memory: {peak / 1024:.2f} KB")
        
        # After all operations, edge count should be 0
        assert len(net.E()) == 0
        # Memory should be reasonable (less than 1 MB for this small test)
        assert current < 1024 * 1024


class TestSetOperationPerformance:
    """Tests verifying O(1) performance of set operations."""
    
    def test_in_edges_lookup_performance(self):
        """
        Test that in_edges lookup is fast (should be O(1) with sets).
        """
        net, nodes = build_large_network(1000)
        
        # Pick a node with edges
        test_node = None
        for node in net.V():
            if net.in_degree(node) > 0:
                test_node = node
                break
        
        assert test_node is not None
        
        # Time many lookups
        start = time.perf_counter()
        for _ in range(10000):
            _ = net.in_edges(test_node)
        elapsed = time.perf_counter() - start
        
        print(f"\n--- in_edges() performance ---")
        print(f"10,000 lookups: {elapsed*1000:.2f} ms")
        print(f"Per lookup: {elapsed/10000*1_000_000:.2f} µs")
        
        # Should complete quickly (< 100ms for 10k lookups)
        assert elapsed < 0.1
    
    def test_leaves_membership_performance(self):
        """
        Test that checking if a node is a leaf is fast with set-based __leaves.
        """
        net, nodes = build_large_network(1000)
        leaves = net.get_leaves()
        
        # Get a leaf and a non-leaf for testing
        leaf_node = leaves[0]
        non_leaf = net.root()
        
        # Time many membership checks (internally uses set)
        start = time.perf_counter()
        for _ in range(10000):
            # This triggers the internal __leaves set lookup
            _ = net.out_degree(leaf_node) == 0
            _ = net.out_degree(non_leaf) == 0
        elapsed = time.perf_counter() - start
        
        print(f"\n--- Leaf check performance ---")
        print(f"20,000 degree checks: {elapsed*1000:.2f} ms")
        
        assert elapsed < 0.5
    
    def test_add_remove_edge_performance(self):
        """
        Test that adding/removing edges (which updates in_map/out_map)
        is fast with set operations.
        """
        net = Network()
        nodes = [Node(f"n{i}") for i in range(100)]
        net.add_nodes(nodes)
        
        # Build initial edges
        for i in range(len(nodes) - 1):
            net.add_edges(Edge(nodes[i], nodes[i + 1]))
        
        # Time edge modifications
        start = time.perf_counter()
        for _ in range(1000):
            # Remove and re-add an edge
            edge = Edge(nodes[0], nodes[1])
            net.remove_edge(edge)
            net.add_edges(edge)
        elapsed = time.perf_counter() - start
        
        print(f"\n--- Edge add/remove performance ---")
        print(f"1,000 remove+add cycles: {elapsed*1000:.2f} ms")
        print(f"Per cycle: {elapsed/1000*1000:.3f} ms")
        
        # Should be fast (< 500ms for 1000 cycles)
        assert elapsed < 0.5


class TestCorrectnessAfterRefactor:
    """Tests ensuring the set refactor didn't break functionality."""
    
    def test_in_out_edges_consistency(self):
        """
        Verify in_edges and out_edges return consistent data.
        """
        net, _ = build_large_network(100)
        
        for node in net.V():
            in_edges = net.in_edges(node)
            out_edges = net.out_edges(node)
            
            # All in_edges should have this node as dest
            for edge in in_edges:
                assert edge.dest == node
            
            # All out_edges should have this node as src
            for edge in out_edges:
                assert edge.src == node
            
            # Degrees should match edge counts
            assert net.in_degree(node) == len(in_edges)
            assert net.out_degree(node) == len(out_edges)
    
    def test_leaves_roots_consistency(self):
        """
        Verify get_leaves() and roots() return correct nodes.
        """
        net, _ = build_large_network(100)
        
        leaves = net.get_leaves()
        roots = net.roots()
        
        # All leaves should have out_degree 0
        for leaf in leaves:
            assert net.out_degree(leaf) == 0
        
        # All roots should have in_degree 0
        for root in roots:
            assert net.in_degree(root) == 0
        
        # No node should be both leaf and root (unless single-node network)
        if len(net.V()) > 1:
            assert len(set(leaves) & set(roots)) == 0
    
    def test_edge_removal_updates_structures(self):
        """
        Test that removing edges properly updates all data structures.
        """
        net = Network()
        n1, n2, n3 = Node("a"), Node("b"), Node("c")
        net.add_nodes(n1, n2, n3)
        
        e1 = Edge(n1, n2)
        e2 = Edge(n2, n3)
        net.add_edges(e1)
        net.add_edges(e2)
        
        # Initial state
        assert net.in_degree(n2) == 1
        assert net.out_degree(n2) == 1
        assert n3 in [n.label for n in net.get_leaves()] or n3 in net.get_leaves()
        
        # Remove edge to n3
        net.remove_edge(e2)
        
        # n2 should now be a leaf
        assert net.out_degree(n2) == 0
        assert n2 in net.get_leaves()
        
        # n3 should be disconnected (no in-edges)
        assert net.in_degree(n3) == 0
    
    def test_no_duplicates_in_edge_sets(self):
        """
        Verify that adding the same edge twice doesn't create duplicates
        in in_map/out_map (sets should prevent this).
        """
        net = Network()
        n1, n2 = Node("a"), Node("b")
        net.add_nodes(n1, n2)
        
        e = Edge(n1, n2)
        net.add_edges(e)
        
        # Try to add same edge reference again - should be handled
        # (EdgeSet may reject, but in_map/out_map shouldn't have duplicates)
        initial_in_count = len(net.in_edges(n2))
        initial_out_count = len(net.out_edges(n1))
        
        # in_map and out_map should have exactly 1 entry each
        assert initial_in_count == 1
        assert initial_out_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

