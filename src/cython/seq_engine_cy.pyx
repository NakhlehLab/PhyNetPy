# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
C-level hot-loop helpers for the MCMC_SEQ timed MSNC density.

Accelerates the two innermost functions of the gene-tree density DP in
``phynetpy._seq_likelihood``:

  * :func:`apply_branch_coalescent_seq_cy` -- the per-species-branch
    coalescent kernel (PhyloNet's
    ``GeneTreeBrSpeciesNetDistribution.calculateProbability``).  Called once
    per (species branch x configuration x locus) per likelihood evaluation,
    which dominates the network-density runtime.
  * :func:`combine_configs_seq_cy` -- the disjoint-union outer-product merge
    of two child configuration distributions at a tree-internal species node.

Both keep the exact ``dict[int] -> float`` log-probability protocol the
Python implementation uses, so they are drop-in replacements: the
pure-Python fallbacks in ``_seq_likelihood`` produce numerically identical
results.

The coalescent kernel per branch is::

    factor  = gamma^u                                  (u = entering lineages)
    for each ready coalescence at time t (u lineages just below it):
        factor *= (2/theta) * exp(-(t - prev) * u*(u-1) / theta)
    if branch top is finite and u_final > 1:
        factor *= exp(-(tau_high - prev) * u_final*(u_final-1) / theta)

Events are the gene-tree coalescences ``(time, parent_bit_id, child0_id,
child1_id)`` sorted ascending by time; an event is realised on this branch
iff both its child lineages are present in the current configuration.

The numerically stable log-sum-exp identity used when two configurations
collide is ``logaddexp(a, b) = max(a,b) + log1p(exp(-|a-b|))``.
"""

from libc.math cimport log, exp, log1p, INFINITY


cdef inline double _logaddexp_pair(double a, double b) noexcept nogil:
    """Numerically stable ``log(exp(a) + exp(b))`` for finite inputs."""
    cdef double diff
    if a == -INFINITY:
        return b
    if b == -INFINITY:
        return a
    if a > b:
        diff = b - a
        return a + log1p(exp(diff))
    diff = a - b
    return b + log1p(exp(diff))


cdef inline int _popcount(unsigned long long mask) noexcept nogil:
    """Hamming weight of ``mask`` (Brian-Kernighan loop)."""
    cdef int c = 0
    while mask:
        mask &= mask - 1
        c += 1
    return c


def apply_branch_coalescent_seq_cy(
    dict config_in,
    double tau_low,
    object tau_high,
    double theta,
    list events,
    double log_gamma,
):
    """Propagate a configuration distribution up one species branch.

    Equivalent to ``_seq_likelihood._apply_branch_seq_py``.

    Args:
        config_in: ``dict`` mapping ``int`` lineage-bitmask -> ``float``
            log-prob entering the bottom of the branch.
        tau_low: Branch bottom height (younger endpoint).
        tau_high: Branch top height (``float``), or ``None`` for the infinite
            root branch.
        theta: Population mutation rate on this branch (``> 0``).
        events: List of ``(time, parent_id, child0_id, child1_id)`` tuples,
            sorted ascending by time.
        log_gamma: ``log(gamma)`` for a reticulation edge (``-inf`` allowed),
            or ``0.0`` for an ordinary edge.

    Returns:
        ``dict`` mapping ``int`` bitmask -> ``float`` log-prob exiting the top.
    """
    cdef bint finite_top = tau_high is not None
    cdef double th = 0.0
    if finite_top:
        th = <double> tau_high

    cdef double inv_theta = 1.0 / theta
    cdef double log_two_over_theta = log(2.0 * inv_theta)

    cdef dict out = {}
    # Lineage bitmasks are packed into a 64-bit word.  The caller
    # (``_seq_likelihood.gene_tree_msnc_log_density``) routes any gene tree
    # whose node count exceeds 64 to the arbitrary-precision pure-Python
    # kernel, so the ``<unsigned long long>`` casts below never overflow.
    cdef unsigned long long cfg, cfg_cur, b0, b1, pid_bit
    cdef unsigned long long one = 1
    cdef double lp, cur, factor, total, existing
    cdef int u
    cdef Py_ssize_t i, n_events = len(events)
    cdef tuple ev
    cdef double t
    cdef int c0, c1, pid
    cdef object cfg_obj, lp_obj, existing_obj

    for cfg_obj, lp_obj in config_in.items():
        cfg = <unsigned long long> cfg_obj
        lp = <double> lp_obj
        cur = tau_low
        u = _popcount(cfg)
        cfg_cur = cfg
        if u == 0:
            factor = 0.0
        else:
            factor = log_gamma * u

        for i in range(n_events):
            ev = <tuple> events[i]
            t = <double> ev[0]
            if t < tau_low:
                continue
            if finite_top and t >= th:
                break
            pid = <int> ev[1]
            c0 = <int> ev[2]
            c1 = <int> ev[3]
            b0 = one << c0
            b1 = one << c1
            if (cfg_cur & b0) and (cfg_cur & b1):
                factor += -(t - cur) * u * (u - 1) * inv_theta
                factor += log_two_over_theta
                cur = t
                pid_bit = one << pid
                cfg_cur = (cfg_cur & ~b0 & ~b1) | pid_bit
                u -= 1

        if finite_top and u > 1:
            factor += -(th - cur) * u * (u - 1) * inv_theta

        total = lp + factor
        existing_obj = out.get(cfg_cur)
        if existing_obj is None:
            out[cfg_cur] = total
        else:
            existing = <double> existing_obj
            out[cfg_cur] = _logaddexp_pair(existing, total)
    return out


def combine_configs_seq_cy(dict left, dict right):
    """Disjoint-union outer-product merge of two child config distributions.

    Equivalent to ``_seq_likelihood._combine_configs_py``: overlapping
    bitmask pairs are dropped, the union is the merged configuration, and
    log-probabilities add (duplicate unions combined with ``logaddexp``).
    """
    cdef dict out = {}
    cdef unsigned long long cfg_l, cfg_r, union
    cdef double lp_l, lp_r, total, existing
    cdef object cfg_l_obj, lp_l_obj, cfg_r_obj, lp_r_obj, existing_obj

    for cfg_l_obj, lp_l_obj in left.items():
        cfg_l = <unsigned long long> cfg_l_obj
        lp_l = <double> lp_l_obj
        for cfg_r_obj, lp_r_obj in right.items():
            cfg_r = <unsigned long long> cfg_r_obj
            if cfg_l & cfg_r:
                continue
            lp_r = <double> lp_r_obj
            union = cfg_l | cfg_r
            total = lp_l + lp_r
            existing_obj = out.get(union)
            if existing_obj is None:
                out[union] = total
            else:
                existing = <double> existing_obj
                out[union] = _logaddexp_pair(existing, total)
    return out
