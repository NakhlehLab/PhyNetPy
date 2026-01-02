# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-optimized Network class.

This provides a high-performance Network implementation that eliminates
Python method call overhead for the most commonly used operations.

To compile:
    cd PhyNetPy/src/PhyNetPy
    cythonize -i network_cy.pyx

Usage:
    from PhyNetPy.network_cy import CNetwork
    net = CNetwork()
    net.add_node(node)
    net.add_edge(edge)
"""

from collections import defaultdict, deque
import warnings

# Import from compiled Cython module
try:
    from PhyNetPy.graph_core_cy import CNodeSet, CEdgeSet
except ImportError:
    # Direct import when running from same directory
    from graph_core_cy import CNodeSet, CEdgeSet


cdef class CNetwork:
    """
    Cython-optimized directed Network (DAG) implementation.
    
    Provides fast access to common phylogenetic network operations:
    - Node/edge addition and removal
    - Degree queries (in_degree, out_degree)
    - Neighbor queries (get_children, get_parents)
    - Membership tests
    
    For full phylogenetic functionality, use the Python Network class.
    This class focuses on performance-critical operations.
    """
    
    # Typed attributes for C-level speed
    cdef object _nodes      # CNodeSet
    cdef object _edges      # CEdgeSet  
    cdef set _leaves
    cdef set _roots
    cdef dict _items        # blob storage
    
    def __init__(self):
        """Initialize an empty directed network."""
        self._nodes = CNodeSet(directed=True)
        self._edges = CEdgeSet(directed=True)
        self._leaves = set()
        self._roots = set()
        self._items = {}
    
    # =========================================================================
    # CORE NODE OPERATIONS
    # =========================================================================
    
    cpdef void add_node(self, node):
        """Add a single node to the network. O(1)"""
        self._nodes.add(node)
        # New node with no edges is both a leaf and a root
        self._leaves.add(node)
        self._roots.add(node)
    
    def add_nodes(self, *nodes):
        """Add multiple nodes to the network."""
        for node in nodes:
            self.add_node(node)
    
    cpdef void remove_node(self, node):
        """Remove a node and all its edges. O(degree)"""
        if node not in self._nodes:
            return
        
        # Remove all edges connected to this node
        cdef set in_edges = self._nodes.in_edges(node).copy()
        cdef set out_edges = self._nodes.out_edges(node).copy()
        
        for edge in in_edges:
            self.remove_edge(edge)
        for edge in out_edges:
            self.remove_edge(edge)
        
        self._nodes.remove(node)
        self._leaves.discard(node)
        self._roots.discard(node)
    
    def __contains__(self, obj) -> bool:
        """Check if node or edge is in network. O(1)"""
        cdef str obj_type = type(obj).__name__
        if obj_type == 'Node':
            return obj in self._nodes
        elif obj_type == 'Edge':
            return obj in self._edges
        return False
    
    # =========================================================================
    # CORE EDGE OPERATIONS  
    # =========================================================================
    
    cpdef void add_edge(self, edge):
        """
        Add a directed edge to the network. O(1)
        Both nodes must already be in the network.
        """
        cdef object src = edge.src
        cdef object dest = edge.dest
        
        # Ensure nodes exist
        if src not in self._nodes:
            self.add_node(src)
        if dest not in self._nodes:
            self.add_node(dest)
        
        # Add edge
        self._edges.add(edge)
        self._nodes.process(edge)
        
        # Update leaf/root tracking
        self._reclassify(src, is_parent=True, is_addition=True)
        self._reclassify(dest, is_parent=False, is_addition=True)
    
    def add_edges(self, *edges):
        """Add multiple edges to the network."""
        for edge in edges:
            self.add_edge(edge)
    
    cpdef void remove_edge(self, edge):
        """Remove an edge from the network. O(1)"""
        if edge not in self._edges:
            return
        
        cdef object src = edge.src
        cdef object dest = edge.dest
        
        self._edges.remove(edge)
        self._nodes.process(edge, removal=True)
        
        # Update leaf/root tracking
        self._reclassify(src, is_parent=True, is_addition=False)
        self._reclassify(dest, is_parent=False, is_addition=False)
    
    cdef void _reclassify(self, node, bint is_parent, bint is_addition):
        """Update leaf/root status after edge change."""
        if is_addition:
            if is_parent:
                # Parent got a child -> no longer a leaf
                if self._nodes.out_deg(node) == 1:
                    self._leaves.discard(node)
                if self._nodes.in_deg(node) == 0:
                    self._roots.add(node)
            else:
                # Child got a parent -> no longer a root
                if self._nodes.in_deg(node) == 1:
                    self._roots.discard(node)
                if self._nodes.out_deg(node) == 0:
                    self._leaves.add(node)
        else:
            if is_parent:
                # Lost a child -> might be a leaf now
                if self._nodes.out_deg(node) == 0:
                    self._leaves.add(node)
            else:
                # Lost a parent -> might be a root now
                if self._nodes.in_deg(node) == 0:
                    self._roots.add(node)
    
    # =========================================================================
    # DEGREE QUERIES (Hot path - fully optimized)
    # =========================================================================
    
    cpdef int in_degree(self, node):
        """Get in-degree of a node. O(1)"""
        return self._nodes.in_deg(node)
    
    cpdef int out_degree(self, node):
        """Get out-degree of a node. O(1)"""
        return self._nodes.out_deg(node)
    
    # =========================================================================
    # NEIGHBOR QUERIES
    # =========================================================================
    
    cpdef set in_edges(self, node):
        """Get incoming edges. O(1)"""
        return self._nodes.in_edges(node)
    
    cpdef set out_edges(self, node):
        """Get outgoing edges. O(1)"""
        return self._nodes.out_edges(node)
    
    cpdef list get_parents(self, node):
        """Get parent nodes. O(in_degree)"""
        return [e.src for e in self._nodes.in_edges(node)]
    
    cpdef list get_children(self, node):
        """Get child nodes. O(out_degree)"""
        return [e.dest for e in self._nodes.out_edges(node)]
    
    # =========================================================================
    # GRAPH STRUCTURE QUERIES
    # =========================================================================
    
    cpdef list V(self):
        """Get all nodes. O(n)"""
        return list(self._nodes.get_set())
    
    cpdef list E(self):
        """Get all edges. O(m)"""
        return list(self._edges.get_set())
    
    cpdef list get_leaves(self):
        """Get leaf nodes (out-degree 0). O(1) + copy"""
        return [leaf for leaf in self._leaves 
                if self._nodes.in_deg(leaf) != 0]
    
    cpdef list roots(self):
        """Get root nodes (in-degree 0). O(1) + copy"""
        return [root for root in self._roots
                if self._nodes.out_deg(root) != 0]
    
    cpdef object root(self):
        """Get the single root (raises if multiple). O(1)"""
        cdef list roots_list = self.roots()
        if len(roots_list) == 0:
            raise ValueError("Network has no root")
        if len(roots_list) > 1:
            warnings.warn("Multiple roots found, returning first")
        return roots_list[0]
    
    cpdef int num_nodes(self):
        """Get number of nodes. O(1)"""
        return len(self._nodes.get_set())
    
    cpdef int num_edges(self):
        """Get number of edges. O(1)"""
        return len(self._edges.get_set())
    
    # =========================================================================
    # TRAVERSAL
    # =========================================================================
    
    cpdef list bfs(self, start_node=None):
        """
        Breadth-first traversal from start_node (or root).
        Returns list of nodes in BFS order.
        """
        cdef list result = []
        cdef set visited = set()
        cdef object cur
        
        if start_node is None:
            start_node = self.root()
        
        queue = deque([start_node])
        
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            result.append(cur)
            
            for child in self.get_children(cur):
                if child not in visited:
                    queue.append(child)
        
        return result
    
    cpdef list dfs(self, start_node=None):
        """
        Depth-first traversal from start_node (or root).
        Returns list of nodes in DFS order.
        """
        cdef list result = []
        cdef set visited = set()
        cdef object cur
        
        if start_node is None:
            start_node = self.root()
        
        stack = [start_node]
        
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            result.append(cur)
            
            for child in self.get_children(cur):
                if child not in visited:
                    stack.append(child)
        
        return result


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark():
    """
    Benchmark CNetwork vs Python Network.
    """
    import time
    
    # Mock Node and Edge classes
    class Node:
        __slots__ = ('label',)
        def __init__(self, name):
            self.label = name
        def __hash__(self):
            return hash(self.label)
        def __eq__(self, other):
            return isinstance(other, Node) and self.label == other.label
    
    class Edge:
        __slots__ = ('src', 'dest')
        def __init__(self, src, dest):
            self.src = src
            self.dest = dest
        def __hash__(self):
            return hash((self.src.label, self.dest.label))
        def __eq__(self, other):
            return (isinstance(other, Edge) and 
                    self.src == other.src and self.dest == other.dest)
    
    print("=" * 60)
    print("CYTHON CNetwork BENCHMARK")
    print("=" * 60)
    
    N = 1000
    ITERATIONS = 100000
    
    nodes = [Node(f"n{i}") for i in range(N)]
    
    # Build network
    net = CNetwork()
    
    start = time.perf_counter()
    for node in nodes:
        net.add_node(node)
    add_node_time = time.perf_counter() - start
    print(f"\nadd_node() x{N}:        {add_node_time*1000:.2f} ms")
    
    start = time.perf_counter()
    for i in range(N-1):
        net.add_edge(Edge(nodes[i], nodes[i+1]))
    add_edge_time = time.perf_counter() - start
    print(f"add_edge() x{N-1}:        {add_edge_time*1000:.2f} ms")
    
    test_node = nodes[N // 2]
    
    # Benchmark queries
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = net.out_degree(test_node)
    out_deg_time = time.perf_counter() - start
    print(f"out_degree() x{ITERATIONS//1000}k:   {out_deg_time*1000:.2f} ms ({out_deg_time/ITERATIONS*1e6:.2f} µs/call)")
    
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = net.in_degree(test_node)
    in_deg_time = time.perf_counter() - start
    print(f"in_degree() x{ITERATIONS//1000}k:    {in_deg_time*1000:.2f} ms ({in_deg_time/ITERATIONS*1e6:.2f} µs/call)")
    
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = net.get_children(test_node)
    children_time = time.perf_counter() - start
    print(f"get_children() x{ITERATIONS//1000}k: {children_time*1000:.2f} ms ({children_time/ITERATIONS*1e6:.2f} µs/call)")
    
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = test_node in net
    contains_time = time.perf_counter() - start
    print(f"__contains__ x{ITERATIONS//1000}k:   {contains_time*1000:.2f} ms ({contains_time/ITERATIONS*1e6:.2f} µs/call)")
    
    # Traversal
    start = time.perf_counter()
    for _ in range(1000):
        _ = net.bfs()
    bfs_time = time.perf_counter() - start
    print(f"bfs() x1000:            {bfs_time*1000:.2f} ms ({bfs_time/1000*1000:.2f} ms/call)")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

