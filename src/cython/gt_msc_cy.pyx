# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
C-level hot-loop helpers for the MCMC_GT MSC / MSNC DP.

This module accelerates the two innermost functions of the
gene-tree-likelihood DP in ``MCMC_GT.py``:

  * :func:`apply_branch_coalescent_cy` -- replaces
    ``_apply_branch_coalescent_int`` (called ~13/network-edge x
    ~1000 gene trees per likelihood, i.e. tens of thousands of
    times per MH iteration on the 7-tax bench; profile shows it
    as the hot 70% of total scoring time).
  * :func:`combine_configs_cy` -- replaces
    ``_combine_configs_int`` (the outer-product merge of two
    child distributions at a tree-internal node).

Both functions keep the *same* dict-of-int -> log-prob protocol the
Python implementation uses, so the fallback path is a drop-in
replacement.  The speedup comes from:

  * Streaming pairwise ``logaddexp`` directly into the output
    dict (no per-cfg-out list allocation, no final list-then-
    logsumexp pass).
  * Pure-C ``log1p``/``exp`` math on the inner loop.
  * Cython-typed locals for the popcount and arithmetic inside
    the per-row loop.

The logaddexp identity used:

    logaddexp(a, b) = max(a,b) + log1p(exp(-|a-b|))

which is numerically stable for all finite inputs.

Importantly we still call back into Python for ``gti.coarsenings``
and ``engine._log_gij`` / ``engine._log_denom`` -- those are
already memoised in Python and the per-call overhead is dominated
by what we *do* save (the ~6.1s of dict iteration / list-building
seen on the profile).
"""

from libc.math cimport log, exp, log1p


cdef double _LOG_FLOOR = log(1e-200)


cdef inline double _logaddexp_pair(double a, double b) noexcept nogil:
    """Numerically stable ``log(exp(a) + exp(b))``.

    Branchless under the assumption neither input is -inf; for our
    DP, both inputs are bounded below by ``_LOG_FLOOR`` so we don't
    need to handle -inf explicitly.
    """
    cdef double diff
    if a > b:
        diff = b - a
        return a + log1p(exp(diff))
    diff = a - b
    return b + log1p(exp(diff))


cdef inline int _popcount(long mask) noexcept nogil:
    """Hamming weight of ``mask`` (Brian-Kernighan loop)."""
    cdef int c = 0
    while mask:
        mask &= mask - 1
        c += 1
    return c


def apply_branch_coalescent_cy(
    dict config_in,
    object coarsen_fn,
    object log_gij_fn,
    object log_denom_fn,
    object length,
):
    """Apply the coalescent transition over one species-network branch.

    Equivalent to ``MCMC_GT._apply_branch_coalescent_int`` but uses
    streaming pairwise ``logaddexp`` instead of per-cfg-out lists.

    Args:
        config_in: ``dict`` mapping ``int`` lineage-bitmask keys to
            ``float`` log-probabilities entering the branch.
        coarsen_fn: Bound ``_GeneTreeIndex.coarsenings`` -- maps a
            cfg_in mask to a tuple of ``(cfg_out, merge_mask, m,
            k, log_le)`` rows.
        log_gij_fn: Bound ``_GTLikelihoodEngine._log_gij`` -- maps
            ``(length, n, m)`` to ``log g_{n,m}(length)``.
        log_denom_fn: Bound ``_GTLikelihoodEngine._log_denom`` --
            maps ``(n, k)`` to ``log prod_{i=1..k} C(n-i+1, 2)``.
        length: Branch length (float or ``None`` for the infinite
            root edge); passed straight through to ``log_gij_fn``.

    Returns:
        ``dict`` of cfg_out -> log-prob (out-of-the-branch).
    """
    cdef dict out = {}
    cdef long cfg_in_key, cfg_out_key
    cdef double lp_in, log_branch, log_total, log_le, existing
    cdef int n, m, k
    cdef tuple rows, row
    cdef object cfg_in_obj, lp_in_obj, existing_obj

    for cfg_in_obj, lp_in_obj in config_in.items():
        cfg_in_key = cfg_in_obj
        lp_in = lp_in_obj
        n = _popcount(cfg_in_key)
        rows = coarsen_fn(cfg_in_obj)
        for row in rows:
            cfg_out_key = row[0]
            m = row[2]
            k = row[3]
            log_branch = log_gij_fn(length, n, m)
            if log_branch <= _LOG_FLOOR:
                continue
            if k > 0:
                log_le = row[4]
                log_branch = log_branch + log_le - log_denom_fn(n, k)
            log_total = lp_in + log_branch
            existing_obj = out.get(cfg_out_key)
            if existing_obj is None:
                out[cfg_out_key] = log_total
            else:
                existing = existing_obj
                out[cfg_out_key] = _logaddexp_pair(existing, log_total)
    return out


def combine_configs_cy(dict left, dict right):
    """Outer-product disjoint-union merge of two child distributions.

    Equivalent to ``MCMC_GT._combine_configs_int`` -- if the two
    children's lineage-bitmasks overlap, the pairing is dropped
    (ill-formed); otherwise the union is the cfg_out and the
    log-probabilities add.  Same-union duplicates are merged via
    streaming pairwise ``logaddexp``.
    """
    cdef dict out = {}
    cdef long cfg_l, cfg_r, union
    cdef double lp_l, lp_r, total, existing
    cdef object cfg_l_obj, lp_l_obj, cfg_r_obj, lp_r_obj, existing_obj

    for cfg_l_obj, lp_l_obj in left.items():
        cfg_l = cfg_l_obj
        lp_l = lp_l_obj
        for cfg_r_obj, lp_r_obj in right.items():
            cfg_r = cfg_r_obj
            if cfg_l & cfg_r:
                continue
            lp_r = lp_r_obj
            union = cfg_l | cfg_r
            total = lp_l + lp_r
            existing_obj = out.get(union)
            if existing_obj is None:
                out[union] = total
            else:
                existing = existing_obj
                out[union] = _logaddexp_pair(existing, total)
    return out
