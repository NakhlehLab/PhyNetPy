"""
Graph core data structures with automatic Cython acceleration.

This module provides NodeSet and EdgeSet classes, automatically using
the Cython-optimized versions if available, falling back to the pure
Python implementations defined in :mod:`.Network`.

Usage::

    from PhyNetPy.graph_core import NodeSet, EdgeSet

    # These will be Cython versions if compiled, Python otherwise
    ns = NodeSet(directed=True)
    es = EdgeSet(directed=True)

To check which version is active::

    from PhyNetPy.graph_core import USING_CYTHON
    print(f"Using Cython: {USING_CYTHON}")
"""

USING_CYTHON = False

try:
    from .cython.graph_core_cy import CNodeSet as NodeSet, CEdgeSet as EdgeSet  # type: ignore[import-not-found]
    USING_CYTHON = True
except ImportError:
    from .Network import NodeSet, EdgeSet  # type: ignore[assignment]


def get_implementation_info() -> dict:
    """
    Get information about which implementation is being used.
    
    Returns:
        dict with keys:
            - 'using_cython': bool
            - 'nodeset_class': class object
            - 'edgeset_class': class object
    """
    return {
        'using_cython': USING_CYTHON,
        'nodeset_class': NodeSet,
        'edgeset_class': EdgeSet,
    }


if __name__ == "__main__":
    info = get_implementation_info()
    print(f"Using Cython: {info['using_cython']}")
    if info['using_cython']:
        print("✓ Cython-optimized graph operations enabled")
    else:
        print("○ Using pure Python (install Cython for 2-5x speedup)")

