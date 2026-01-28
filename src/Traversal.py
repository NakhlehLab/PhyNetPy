from enum import Enum, auto
from typing import Any, Iterator, Generator, Optional
from collections import deque
from .ModelGraph2 import ModelNode

class TraversalOrder(Enum):
    """Traversal order options."""
    PRE_ORDER = auto()   # Parent before children (root → leaves) - good for simulation
    POST_ORDER = auto()  # Children before parent (leaves → root) - good for likelihood
    LEVEL_ORDER = auto() # Breadth-first by level - good for parallel scheduling


class Traversal:
    """
    Iterator-based graph traversal for model nodes.
    
    Decouples traversal order from visitation logic, allowing:
    - Easy switching between pre/post/level order
    - Use in for-loops or manual next() calls
    - Clean separation of concerns
    
    Example:
        >>> # Likelihood computation (post-order)
        >>> for node in Traversal(root, TraversalOrder.POST_ORDER):
        ...     visitor.visit(node)
        
        >>> # Simulation (pre-order, root to leaves)
        >>> for node in ModelTraversal(root, TraversalOrder.PRE_ORDER):
        ...     simulator.simulate_at(node)
        
        >>> # Manual iteration
        >>> traversal = ModelTraversal(root, TraversalOrder.POST_ORDER)
        >>> first_node = next(traversal)
        >>> second_node = next(traversal)
    """
    
    def __init__(self, 
                 root: Any, 
                 order: TraversalOrder = TraversalOrder.POST_ORDER):
        """
        Initialize traversal.
        
        Args:
            root: Starting node (root of the network)
            order: Traversal order (PRE_ORDER, POST_ORDER, or LEVEL_ORDER)
        """
        self.root = root
        self.order = order
        self._iterator: Optional[Iterator[ModelNode]] = None
    
    def __iter__(self) -> Iterator[ModelNode]:
        """Return fresh iterator each time."""
        if self.order == TraversalOrder.PRE_ORDER:
            self._iterator = self._preorder(self.root)
        elif self.order == TraversalOrder.POST_ORDER:
            self._iterator = self._postorder(self.root)
        elif self.order == TraversalOrder.LEVEL_ORDER:
            self._iterator = self._levelorder(self.root)
        else:
            raise ValueError(f"Unknown traversal order: {self.order}")
        return self
    
    def __next__(self) -> ModelNode:
        """Get next node in traversal."""
        if self._iterator is None:
            self.__iter__()
        return next(self._iterator)
    
    # ───────────────────────────────────────────────────────────────
    # Traversal Generators
    # ───────────────────────────────────────────────────────────────
    
    def _preorder(self, root: ModelNode) -> Generator[ModelNode, None, None]:
        """
        Pre-order: Visit parent BEFORE children.
        
        Order: Root → Internal → Leaves
        Use for: Simulation (evolve sequences down the tree)
        
                    R (1st)
                   /   \\
                 I1   I2 (2nd, 5th)
                / \\      \\
               A   B    C (3rd, 4th, 6th)
        """
        visited: set[int] = set()
        stack: list[ModelNode] = [root]
        
        while stack:
            node = stack.pop()
            node_id = id(node)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            yield node
            
            # Add children to stack (reverse order so left child is processed first)
            
            children = node.get_model_children()
            
            if children:
                stack.extend(reversed(children))
    
    def _postorder(self, root: ModelNode) -> Generator[ModelNode, None, None]:
        """
        Post-order: Visit children BEFORE parent.
        
        Order: Leaves → Internal → Root
        Use for: Likelihood (need child VPIs before computing parent)
        
                    R (6th)
                   / \\
                 I1   I2 (3rd, 5th)
                / \\     \\
               A   B     C (1st, 2nd, 4th)
        """
        visited: set[int] = set()
        
        def _visit(node: ModelNode) -> Generator[ModelNode, None, None]:
            node_id = id(node)
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            
            # Visit children first
            children = node.get_model_children()
            if children:
                for child in children:
                    yield from _visit(child)
            
            # Then yield self
            yield node
        
        yield from _visit(root)
    
    def _levelorder(self, root: ModelNode) -> Generator[ModelNode, None, None]:
        """
        Level-order (BFS): Visit by depth level.
        
        Order: Level 0, Level 1, Level 2, ...
        Use for: Parallel scheduling (nodes at same level are independent)
        
        Note: This goes TOP-DOWN. For bottom-up level order 
              (leaves first), use reversed(list(ModelTraversal(root, LEVEL_ORDER)))
              or use LevelParallelTraversal below.
        
                    R (1st)         Level 0
                   / \\
                 I1   I2 (2nd, 3rd) Level 1
                / \\     \\
               A   B     C (4-6th)  Level 2
        """
        visited: set[int] = set()
        queue: deque[ModelNode] = deque([root])
        
        while queue:
            node = queue.popleft()
            node_id = id(node)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            yield node
            
            # Add children to queue
            children = node.get_model_children()
            if children:
                queue.extend(children)
    
    # ───────────────────────────────────────────────────────────────
    # Utility Methods
    # ───────────────────────────────────────────────────────────────
    
    def as_list(self) -> list[ModelNode]:
        """Return traversal as a list."""
        return list(self)
    
    def count(self) -> int:
        """Count nodes in traversal."""
        return sum(1 for _ in self)


class LevelParallelTraversal:
    """
    Traversal that yields nodes grouped by level (for parallel execution).
    
    Unlike ModelTraversal which yields one node at a time, this yields
    entire levels - all nodes in a level can be processed in parallel.
    
    Example:
        >>> for level_num, nodes in LevelParallelTraversal(root):
        ...     # Process all nodes in this level in parallel
        ...     with ThreadPoolExecutor() as pool:
        ...         pool.map(visitor.visit, nodes)
    """
    
    def __init__(self, root: ModelNode, bottom_up: bool = True):
        """
        Initialize level-parallel traversal.
        
        Args:
            root: Root node of the network
            bottom_up: If True, yield leaves first (for likelihood).
                      If False, yield root first (for simulation).
        """
        self.root = root
        self.bottom_up = bottom_up
    
    def __iter__(self) -> Iterator[tuple[int, list[ModelNode]]]:
        """Yield (level_number, nodes_at_level) tuples."""
        levels = self._compute_levels()
        
        if self.bottom_up:
            # Leaves (highest level number) first
            for level_num in sorted(levels.keys(), reverse=True):
                yield level_num, levels[level_num]
        else:
            # Root (level 0) first
            for level_num in sorted(levels.keys()):
                yield level_num, levels[level_num]
    
    def _compute_levels(self) -> dict[int, list[ModelNode]]:
        """
        Assign each node to a level based on distance from root.
        
        Returns:
            Dict mapping level_number → list of nodes at that level
        """
        levels: dict[int, list[ModelNode]] = {}
        visited: set[int] = set()
        
        # BFS from root, tracking depth
        queue: deque[tuple[ModelNode, int]] = deque([(self.root, 0)])
        
        while queue:
            node, depth = queue.popleft()
            node_id = id(node)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(node)
            
            children = node.get_model_children()
            if children:
                for child in children:
                    queue.append((child, depth + 1))
        
        return levels
    
    @property
    def num_levels(self) -> int:
        """Number of levels in the traversal."""
        return len(self._compute_levels())
    
    @property
    def max_parallelism(self) -> int:
        """Maximum nodes at any single level."""
        levels = self._compute_levels()
        return max(len(nodes) for nodes in levels.values()) if levels else 0