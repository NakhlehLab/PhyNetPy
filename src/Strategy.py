"""
Strategy pattern interface for node-level computations on a
:class:`~.ModelGraph.Model`.

Implementations supply concrete ``compute_at_*`` methods that are
dispatched by the :class:`~.ModelGraph.ModelNode` during a bottom-up
traversal.  Use this when computations need *return values* propagated
upward (e.g. partial-likelihood vectors).  For fire-and-forget side
effects, see the sibling :class:`~.Visitor.Visitor` interface.
"""

from abc import abstractmethod, ABC
from .ModelGraph import ModelNode


class Strategy(ABC):
    """Abstract base for model-node computation strategies.

    Subclasses must implement one method per node kind.  The ``ModelNode``
    dispatch layer calls the matching method automatically during traversal.
    """
    
    @abstractmethod
    def compute_at_leaf(self, n: ModelNode) -> None:
        ...
    
    @abstractmethod
    def compute_at_internal(self, n: ModelNode) -> None:
        ...

    @abstractmethod
    def compute_at_reticulation(self, n: ModelNode) -> None:
        ...

    @abstractmethod
    def compute_at_root(self, n: ModelNode) -> None:
        ...
