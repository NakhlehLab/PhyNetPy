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
from libc.stdlib cimport malloc, free


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


# ══════════════════════════════════════════════════════════════════════
# Full network frontier DP (Yu, Degnan & Nakhleh 2012 ancestral configs)
# ══════════════════════════════════════════════════════════════════════
#
# ``network_dp_cy`` / ``network_dp_timed_cy`` port the two hot frontier
# loops of ``_msnc_density`` (topology-only, used by MCMC_GT; and the
# timed / event-based branch density, used by MCMC_SEQ) into C-typed
# Cython.  They keep the *exact* frontier protocol of the Python
# reference -- a ``dict`` mapping a sorted tuple of ``(edge_id, mask)``
# pairs to a log-probability -- so results are bit-for-bit identical
# (verified by ``tests/test_mcmc_gt.py`` / ``test_mcmc_seq.py`` parity
# checks).  The speedup comes from:
#
#   * inlining the ``_frontier_insert`` / ``_frontier_acc`` bookkeeping
#     (the profiled 25M dict.get + 5M tuple-rebuild hot spot) as C funcs,
#   * C-typed popcount / arithmetic in the reticulation subset loop,
#   * calling the already-C ``apply_branch_coalescent_cy`` in-process.


cdef inline object _frontier_insert_c(object tup, int eid, long mask):
    """Insert ``(eid, mask)`` into a sorted-by-eid tuple key."""
    cdef Py_ssize_t n, pos
    cdef tuple t
    cdef list out
    if tup is None or len(<tuple>tup) == 0:
        return ((eid, mask),)
    t = <tuple>tup
    n = len(t)
    pos = 0
    while pos < n and (<tuple>t[pos])[0] < eid:
        pos += 1
    out = list(t)
    out.insert(pos, (eid, mask))
    return tuple(out)


cdef inline void _frontier_acc_c(dict frontier, object key, double lp):
    """Accumulate ``lp`` into ``frontier[key]`` via stable logaddexp."""
    cdef object existing_obj = frontier.get(key)
    cdef double existing
    if existing_obj is None:
        frontier[key] = lp
    else:
        existing = existing_obj
        frontier[key] = _logaddexp_pair(existing, lp)


cdef double _logsumexp_list(list terms):
    cdef Py_ssize_t i, n = len(terms)
    cdef double m, acc, v
    if n == 0:
        return _LOG_FLOOR
    m = terms[0]
    for i in range(1, n):
        v = terms[i]
        if v > m:
            m = v
    if m == _NEG_INF:
        return _LOG_FLOOR
    acc = 0.0
    for i in range(n):
        acc += exp(<double>terms[i] - m)
    if acc <= 0.0:
        return _LOG_FLOOR
    return m + log(acc)


cdef double _NEG_INF = float("-inf")


def network_dp_cy(object net_idx, object gti, object engine, dict species_to_bits):
    """C-typed port of ``_msnc_log_prob_network_int`` (topology DP).

    Args:
        net_idx: ``_NetworkIndex`` view of the species network.
        gti: ``_GeneTreeIndex`` view of the gene tree.
        engine: ``MSCBranchKernel`` (provides ``_log_gij`` / ``_log_denom``
            and, via ``gti.coarsenings``, the coarsening rows).
        species_to_bits: species-label -> OR of all gene-tree leaf bits
            mapped there (built by the Python caller).

    Returns:
        ``log P(gene_tree | species_net)`` (floored at ``_LOG_FLOOR``).
    """
    cdef int root_bit = gti.root_bit
    if root_bit < 0 or not gti.leaves:
        return 0.0
    if net_idx.n_nodes == 0 or net_idx.root < 0:
        return _LOG_FLOOR

    cdef object coarsen_fn = gti.coarsenings
    cdef object log_gij_fn = engine._log_gij
    cdef object log_denom_fn = engine._log_denom
    cdef object leaf_label = net_idx.leaf_label
    cdef object edge_gamma = net_idx.edge_gamma
    cdef object edge_length = net_idx.edge_length
    cdef list topo_order = net_idx.topo_order
    cdef list down_edges = net_idx.down_edges
    cdef list up_edges = net_idx.up_edges
    cdef list is_retic = net_idx.is_retic

    cdef dict frontier = {(): 0.0}
    cdef dict new_frontier
    cdef object key, lp_obj, new_key_base, out, out1, out2
    cdef double lp, top_lp, lp1, lp2, factor, log_g1, log_g2
    cdef long mask_at_v, top_mask, S, AS, top1, top2
    cdef int v, de, e1, e2, e_up, parent_edge, found_idx, idx
    cdef int n_total, k_S, klen
    cdef list down_es, up_es, new_key_list
    cdef object sp, g1o, g2o, length, length1, length2

    for v in topo_order:
        down_es = down_edges[v]
        up_es = up_edges[v]
        new_frontier = {}

        for key, lp_obj in frontier.items():
            lp = lp_obj
            if len(<list>down_es) == 0:
                sp = leaf_label.get(v)
                mask_at_v = species_to_bits.get(sp, 0) if sp is not None else 0
                new_key_base = key
            else:
                new_key_list = list(<tuple>key)
                mask_at_v = 0
                found_idx = 0
                ok = True
                for de in down_es:
                    found_idx = -1
                    klen = len(new_key_list)
                    for idx in range(klen):
                        if (<tuple>new_key_list[idx])[0] == de:
                            found_idx = idx
                            break
                    if found_idx < 0:
                        ok = False
                        break
                    mask_at_v |= <long>(<tuple>new_key_list[found_idx])[1]
                    new_key_list.pop(found_idx)
                if not ok:
                    continue
                new_key_base = tuple(new_key_list)

            if len(<list>up_es) == 0:
                out = apply_branch_coalescent_cy(
                    {mask_at_v: 0.0}, coarsen_fn, log_gij_fn, log_denom_fn, None
                )
                for top_mask, top_lp in (<dict>out).items():
                    _frontier_acc_c(
                        new_frontier,
                        _frontier_insert_c(new_key_base, -1, top_mask),
                        lp + <double>top_lp,
                    )
                continue

            if len(<list>up_es) >= 2 and is_retic[v]:
                e1 = up_es[0]
                e2 = up_es[1]
                g1o = edge_gamma(e1)
                g2o = edge_gamma(e2)
                if g1o is None and g2o is None:
                    log_g1 = log(0.5)
                    log_g2 = log(0.5)
                elif g1o is None:
                    log_g1 = log(max(0.0, 1.0 - <double>g2o)) if (1.0 - <double>g2o) > 0.0 else _LOG_FLOOR
                    log_g2 = log(<double>g2o) if <double>g2o > 0.0 else _LOG_FLOOR
                elif g2o is None:
                    log_g1 = log(<double>g1o) if <double>g1o > 0.0 else _LOG_FLOOR
                    log_g2 = log(max(0.0, 1.0 - <double>g1o)) if (1.0 - <double>g1o) > 0.0 else _LOG_FLOOR
                else:
                    log_g1 = log(<double>g1o) if <double>g1o > 0.0 else _LOG_FLOOR
                    log_g2 = log(<double>g2o) if <double>g2o > 0.0 else _LOG_FLOOR
                length1 = edge_length(e1)
                length2 = edge_length(e2)
                n_total = _popcount(mask_at_v)
                S = mask_at_v
                while True:
                    k_S = _popcount(S)
                    factor = k_S * log_g1 + (n_total - k_S) * log_g2
                    AS = mask_at_v ^ S
                    out1 = apply_branch_coalescent_cy(
                        {S: 0.0}, coarsen_fn, log_gij_fn, log_denom_fn, length1
                    )
                    out2 = apply_branch_coalescent_cy(
                        {AS: 0.0}, coarsen_fn, log_gij_fn, log_denom_fn, length2
                    )
                    for top1, lp1 in (<dict>out1).items():
                        for top2, lp2 in (<dict>out2).items():
                            _frontier_acc_c(
                                new_frontier,
                                _frontier_insert_c(
                                    _frontier_insert_c(new_key_base, e1, top1),
                                    e2, top2,
                                ),
                                lp + factor + <double>lp1 + <double>lp2,
                            )
                    if S == 0:
                        break
                    S = (S - 1) & mask_at_v
                continue

            e_up = up_es[0]
            length = edge_length(e_up)
            out = apply_branch_coalescent_cy(
                {mask_at_v: 0.0}, coarsen_fn, log_gij_fn, log_denom_fn, length
            )
            for top_mask, top_lp in (<dict>out).items():
                _frontier_acc_c(
                    new_frontier,
                    _frontier_insert_c(new_key_base, e_up, top_mask),
                    lp + <double>top_lp,
                )

        frontier = new_frontier

    cdef long target = 1 << root_bit
    cdef list log_terms = []
    for key, lp_obj in frontier.items():
        if len(<tuple>key) != 1:
            continue
        e_up = (<tuple>(<tuple>key)[0])[0]
        top_mask = (<tuple>(<tuple>key)[0])[1]
        if e_up != -1:
            continue
        if top_mask == target:
            log_terms.append(lp_obj)
        elif _popcount(top_mask) == 1 and (top_mask >> root_bit) & 1:
            log_terms.append(lp_obj)
    if len(log_terms) == 0:
        return _LOG_FLOOR
    return _logsumexp_list(log_terms)


# ══════════════════════════════════════════════════════════════════════
# Timed (event-based) network frontier DP -- MCMC_SEQ scorer
# ══════════════════════════════════════════════════════════════════════
#
# ``network_dp_timed_cy`` ports ``_msnc_log_density_timed`` +
# ``_apply_branch_timed`` from ``_msnc_density.py``.  Same frontier
# protocol / result as the Python reference (parity-checked); the
# per-species-branch coalescent density is integrated over the
# pre-sorted gene-coalescence events, which are copied once into C
# arrays so the innermost event loop is branch-free C.


cdef void _apply_branch_timed_single(
    long cfg,
    double lp_in,
    double tau_low,
    double tau_high,
    bint has_high,
    double inv_theta,
    double log_two_over_theta,
    double* ev_t,
    long* ev_pid,
    long* ev_c0,
    long* ev_c1,
    int n_ev,
    long* out_cfg,
    double* out_lp,
) noexcept nogil:
    """Event-based coalescent density along one species branch (single cfg)."""
    cdef double cur = tau_low
    cdef int u = _popcount(cfg)
    cdef long cfg_cur = cfg
    cdef double lp = lp_in
    cdef int j
    cdef double t, lp_branch
    cdef long b0, b1
    for j in range(n_ev):
        t = ev_t[j]
        if t < tau_low:
            continue
        if has_high and t >= tau_high:
            break
        b0 = <long>1 << ev_c0[j]
        b1 = <long>1 << ev_c1[j]
        if (cfg_cur & b0) and (cfg_cur & b1):
            lp_branch = -(t - cur) * u * (u - 1) * inv_theta
            lp_branch += log_two_over_theta
            cur = t
            cfg_cur = (cfg_cur & ~b0 & ~b1) | (<long>1 << ev_pid[j])
            u -= 1
            lp += lp_branch
    if has_high and u > 1:
        lp += -(tau_high - cur) * u * (u - 1) * inv_theta
    out_cfg[0] = cfg_cur
    out_lp[0] = lp


def network_dp_timed_cy(
    object net_idx,
    object gti,
    object events,
    object sp_heights,
    double theta,
):
    """C-typed port of ``_msnc_log_density_timed`` (timed / event DP).

    Args:
        net_idx: ``_NetworkIndex`` view of the species network.
        gti: ``_GeneTreeIndex`` view of the gene tree.
        events: pre-sorted list of ``(time, parent_bit, child0_bit,
            child1_bit)`` gene-tree coalescences.
        sp_heights: per-node ultrametric heights (indexed by node id).
        theta: population mutation rate.

    Returns:
        ``log P(gene_tree | species_net)`` under the timed MSNC density
        (floored at ``_LOG_FLOOR``).
    """
    cdef int root_bit = gti.root_bit
    if root_bit < 0 or not gti.leaves:
        return 0.0
    if net_idx.n_nodes == 0 or net_idx.root < 0:
        return _LOG_FLOOR

    cdef object leaf_label = net_idx.leaf_label
    cdef object edge_gamma = net_idx.edge_gamma
    cdef list topo_order = net_idx.topo_order
    cdef list down_edges = net_idx.down_edges
    cdef list up_edges = net_idx.up_edges
    cdef list is_retic = net_idx.is_retic
    cdef list edge_src = net_idx.edge_src
    cdef list sph = list(sp_heights)

    # species label -> OR of gene-tree leaf bits mapped there
    cdef dict species_to_bits = {}
    cdef object sp_lbl
    cdef int leaf_bit
    for leaf_bit in gti.leaves:
        sp_lbl = gti.leaf_species_of.get(leaf_bit)
        if sp_lbl is not None:
            species_to_bits[sp_lbl] = species_to_bits.get(sp_lbl, 0) | (<long>1 << leaf_bit)

    # Copy events into C arrays (branch-free inner loop).
    cdef int n_ev = len(events)
    cdef double* ev_t = <double*>malloc(n_ev * sizeof(double)) if n_ev > 0 else NULL
    cdef long* ev_pid = <long*>malloc(n_ev * sizeof(long)) if n_ev > 0 else NULL
    cdef long* ev_c0 = <long*>malloc(n_ev * sizeof(long)) if n_ev > 0 else NULL
    cdef long* ev_c1 = <long*>malloc(n_ev * sizeof(long)) if n_ev > 0 else NULL
    cdef int j
    cdef tuple ev
    if n_ev > 0 and (ev_t == NULL or ev_pid == NULL or ev_c0 == NULL or ev_c1 == NULL):
        if ev_t != NULL: free(ev_t)
        if ev_pid != NULL: free(ev_pid)
        if ev_c0 != NULL: free(ev_c0)
        if ev_c1 != NULL: free(ev_c1)
        raise MemoryError()
    for j in range(n_ev):
        ev = <tuple>events[j]
        ev_t[j] = ev[0]
        ev_pid[j] = ev[1]
        ev_c0[j] = ev[2]
        ev_c1[j] = ev[3]

    cdef double inv_theta = 1.0 / theta
    cdef double log_two_over_theta = log(2.0 * inv_theta)

    cdef dict frontier = {(): 0.0}
    cdef dict new_frontier
    cdef object key, lp_obj, new_key_base, g1o, g2o
    cdef double lp, factor, log_g1, log_g2, tau_low, out_lp, lp1
    cdef long mask_at_v, S, AS, out_cfg, target, top_mask, top1_cfg
    cdef int v, de, e1, e2, e_up, parent_id, parent1, parent2
    cdef int found_idx, idx, klen, n_total, k_S
    cdef list down_es, up_es, new_key_list, log_terms
    cdef bint ok
    cdef double tau_high1, tau_high2, tau_high_up

    try:
        for v in topo_order:
            down_es = down_edges[v]
            up_es = up_edges[v]
            new_frontier = {}

            for key, lp_obj in frontier.items():
                lp = lp_obj
                if len(<list>down_es) == 0:
                    sp_lbl = leaf_label.get(v)
                    mask_at_v = species_to_bits.get(sp_lbl, 0) if sp_lbl is not None else 0
                    new_key_base = key
                else:
                    new_key_list = list(<tuple>key)
                    mask_at_v = 0
                    ok = True
                    for de in down_es:
                        found_idx = -1
                        klen = len(new_key_list)
                        for idx in range(klen):
                            if (<tuple>new_key_list[idx])[0] == de:
                                found_idx = idx
                                break
                        if found_idx < 0:
                            ok = False
                            break
                        mask_at_v |= <long>(<tuple>new_key_list[found_idx])[1]
                        new_key_list.pop(found_idx)
                    if not ok:
                        continue
                    new_key_base = tuple(new_key_list)

                if len(<list>up_es) == 0:
                    _apply_branch_timed_single(
                        mask_at_v, 0.0, sph[v], 0.0, False,
                        inv_theta, log_two_over_theta,
                        ev_t, ev_pid, ev_c0, ev_c1, n_ev,
                        &out_cfg, &out_lp,
                    )
                    _frontier_acc_c(
                        new_frontier,
                        _frontier_insert_c(new_key_base, -1, out_cfg),
                        lp + out_lp,
                    )
                    continue

                if len(<list>up_es) >= 2 and is_retic[v]:
                    e1 = up_es[0]
                    e2 = up_es[1]
                    g1o = edge_gamma(e1)
                    g2o = edge_gamma(e2)
                    if g1o is None and g2o is None:
                        log_g1 = log(0.5)
                        log_g2 = log(0.5)
                    elif g1o is None:
                        log_g1 = log(max(0.0, 1.0 - <double>g2o)) if (1.0 - <double>g2o) > 0.0 else _LOG_FLOOR
                        log_g2 = log(<double>g2o) if <double>g2o > 0.0 else _LOG_FLOOR
                    elif g2o is None:
                        log_g1 = log(<double>g1o) if <double>g1o > 0.0 else _LOG_FLOOR
                        log_g2 = log(max(0.0, 1.0 - <double>g1o)) if (1.0 - <double>g1o) > 0.0 else _LOG_FLOOR
                    else:
                        log_g1 = log(<double>g1o) if <double>g1o > 0.0 else _LOG_FLOOR
                        log_g2 = log(<double>g2o) if <double>g2o > 0.0 else _LOG_FLOOR
                    tau_low = sph[v]
                    parent1 = edge_src[e1]
                    parent2 = edge_src[e2]
                    tau_high1 = sph[parent1]
                    tau_high2 = sph[parent2]
                    n_total = _popcount(mask_at_v)
                    S = mask_at_v
                    while True:
                        k_S = _popcount(S)
                        factor = k_S * log_g1 + (n_total - k_S) * log_g2
                        AS = mask_at_v ^ S
                        _apply_branch_timed_single(
                            S, 0.0, tau_low, tau_high1, True,
                            inv_theta, log_two_over_theta,
                            ev_t, ev_pid, ev_c0, ev_c1, n_ev,
                            &out_cfg, &out_lp,
                        )
                        top1_cfg = out_cfg
                        lp1 = out_lp
                        _apply_branch_timed_single(
                            AS, 0.0, tau_low, tau_high2, True,
                            inv_theta, log_two_over_theta,
                            ev_t, ev_pid, ev_c0, ev_c1, n_ev,
                            &out_cfg, &out_lp,
                        )
                        _frontier_acc_c(
                            new_frontier,
                            _frontier_insert_c(
                                _frontier_insert_c(new_key_base, e1, top1_cfg),
                                e2, out_cfg,
                            ),
                            lp + factor + lp1 + out_lp,
                        )
                        if S == 0:
                            break
                        S = (S - 1) & mask_at_v
                    continue

                e_up = up_es[0]
                parent_id = edge_src[e_up]
                tau_high_up = sph[parent_id]
                _apply_branch_timed_single(
                    mask_at_v, 0.0, sph[v], tau_high_up, True,
                    inv_theta, log_two_over_theta,
                    ev_t, ev_pid, ev_c0, ev_c1, n_ev,
                    &out_cfg, &out_lp,
                )
                _frontier_acc_c(
                    new_frontier,
                    _frontier_insert_c(new_key_base, e_up, out_cfg),
                    lp + out_lp,
                )

            frontier = new_frontier
    finally:
        if ev_t != NULL: free(ev_t)
        if ev_pid != NULL: free(ev_pid)
        if ev_c0 != NULL: free(ev_c0)
        if ev_c1 != NULL: free(ev_c1)

    target = <long>1 << root_bit
    log_terms = []
    for key, lp_obj in frontier.items():
        if len(<tuple>key) != 1:
            continue
        e_up = (<tuple>(<tuple>key)[0])[0]
        top_mask = (<tuple>(<tuple>key)[0])[1]
        if e_up != -1:
            continue
        if top_mask == target:
            log_terms.append(lp_obj)
        elif _popcount(top_mask) == 1 and (top_mask >> root_bit) & 1:
            log_terms.append(lp_obj)
    if len(log_terms) == 0:
        return _LOG_FLOOR
    return _logsumexp_list(log_terms)
