"""Backwards-compatibility shim for ``phynetpy.MCMC_GT``.

The implementation now lives in :mod:`phynetpy._mcmc_gt` and the curated
public inference surface is exposed through :mod:`phynetpy.infer`.  New
code should prefer::

    from phynetpy.infer import MCMC_GT

This shim forwards every attribute lookup (including underscore-prefixed
helpers used by older tests, e.g. ``_GTLikelihoodEngine``) to
``phynetpy._mcmc_gt`` so existing imports keep working unchanged.
"""

from . import _mcmc_gt as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
