"""Backwards-compatibility shim for ``phynetpy.Infer_MP_Allop``.

The implementation now lives in :mod:`phynetpy._infer_mp_allop` and the
curated public inference surface is exposed through
:mod:`phynetpy.infer`.  New code should prefer::

    from phynetpy.infer import INFER_MP_ALLOP

This shim forwards every attribute lookup (including underscore-prefixed
helpers and lower-level utilities used by older tests, e.g.
``allele_map_set``, ``partition_gene_trees``) to
``phynetpy._infer_mp_allop`` so existing imports keep working unchanged.
"""

from . import _infer_mp_allop as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
