"""
Visitor pattern interface for :class:`~.ModelGraph.ModelNode` traversals.

Use a :class:`Visitor` when the operation is a *side-effect* (e.g.
printing, accumulating statistics) and does not need to propagate a
return value upward through the model graph.  For computations that
return partial results, see :class:`~.Strategy.Strategy`.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ModelGraph import ModelNode

class Visitor(ABC):
    """Abstract base for model-node visitors.

    Subclasses implement one ``visit_*`` method per node kind.
    :meth:`visit` provides automatic dispatch by calling
    :pymeth:`ModelNode.get_node_type`.
    """
    
    @abstractmethod
    def visit_leaf(self, n: ModelNode) -> None:
        ...
    
    @abstractmethod
    def visit_internal(self, n: ModelNode) -> None:
        ...

    @abstractmethod
    def visit_reticulation(self, n: ModelNode) -> None:
        ...

    @abstractmethod
    def visit_root(self, n: ModelNode) -> None:
        ...
    
    def visit(self, n: ModelNode) -> None:
        """Dispatch to the correct ``visit_*`` method based on node type."""
        dispatch = {
            "leaf": self.visit_leaf,
            "internal": self.visit_internal,
            "root": self.visit_root,
            "reticulation": self.visit_reticulation,
        }
        return dispatch[n.get_node_type()](n)

