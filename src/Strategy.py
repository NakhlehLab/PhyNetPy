"""
Author : Mark Kessler
Last Edit : 5/12/26
First Included in Version : 0.3.0

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
        """Compute the strategy's quantity at a leaf node.

        Args:
            n (ModelNode): The leaf node currently being visited.
        """
        ...

    @abstractmethod
    def compute_at_internal(self, n: ModelNode) -> None:
        """Compute the strategy's quantity at an internal (non-root, non-leaf) node.

        Implementations may assume that ``compute_at_*`` has already been
        called for every child of ``n``.

        Args:
            n (ModelNode): The internal node currently being visited.
        """
        ...

    @abstractmethod
    def compute_at_reticulation(self, n: ModelNode) -> None:
        """Compute the strategy's quantity at a reticulation (hybrid) node.

        Reticulation nodes have two or more parents and a single child; the
        strategy is responsible for combining the contributions from each
        incoming branch (e.g. via the inheritance probabilities).

        Args:
            n (ModelNode): The reticulation node currently being visited.
        """
        ...

    @abstractmethod
    def compute_at_root(self, n: ModelNode) -> None:
        """Compute the strategy's quantity at the root node.

        This is the terminal step of a bottom-up traversal; the return value
        (stored on the node by the implementation) typically represents the
        whole-network result, e.g. the marginal likelihood.

        Args:
            n (ModelNode): The root node currently being visited.
        """
        ...
