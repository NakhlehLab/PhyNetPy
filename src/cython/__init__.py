"""Compiled Cython acceleration extensions for PhyNetPy.

Marker package for the ``.pyx``-derived extension modules (built via
``pip install -e .``); imported from throughout :mod:`phynetpy` wherever a
hot loop has a compiled fallback (e.g. graph traversals, the MSNC
ancestral-configurations DP). See :mod:`phynetpy.Network` for the
mandatory ``NodeSet`` / ``EdgeSet`` import and its build-error message.
"""
