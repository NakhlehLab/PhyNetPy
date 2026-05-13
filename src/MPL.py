"""Backwards-compatibility shim for ``phynetpy.MPL``.

The implementation now lives in :mod:`phynetpy._mpl` and the curated
public inference surface is exposed through :mod:`phynetpy.infer`.  New
code should prefer::

    from phynetpy.infer import MPL

This shim forwards every attribute lookup (including underscore-prefixed
helpers used by older tests, e.g. ``_TripleDPEngine``) to
``phynetpy._mpl`` so existing imports keep working unchanged.
"""

from . import _mpl as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
