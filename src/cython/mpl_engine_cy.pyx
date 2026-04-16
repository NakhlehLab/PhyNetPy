# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
C-level DP engine for MPL triplet probability computation.

Replaces the hot inner loop of _TripleDPEngine.calculate_triple_probability
with typed C structs and arithmetic, eliminating Python dict/object overhead.
"""

from libc.math cimport exp, log, sqrt
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy

DEF MAX_RETICS = 8
DEF MAX_CONFIGS = 64
DEF NUM_STATES = 11
DEF MAX_CHILDREN = 4
DEF MAX_PARENTS = 2
DEF MAX_NODES = 128
DEF MAX_SPLITS = 4
DEF MAX_COAL = 3

cdef double _LOG_FLOOR = log(1e-200)

# ===================================================================
# Structs
# ===================================================================

cdef struct Config:
    double total_prob
    int idx[MAX_RETICS]

cdef struct StateSlot:
    int n
    Config items[MAX_CONFIGS]

cdef struct StateMap:
    StateSlot slots[NUM_STATES]

# ===================================================================
# Lookup tables (initialized lazily)
# ===================================================================

cdef int _COAL_N[NUM_STATES]
cdef int _COAL_TGT[NUM_STATES][MAX_COAL]
cdef int _COAL_LI[NUM_STATES][MAX_COAL]
cdef int _COAL_LO[NUM_STATES][MAX_COAL]

cdef int _SPLIT_N[NUM_STATES]
cdef int _SPLIT_S1[NUM_STATES][MAX_SPLITS]
cdef int _SPLIT_S2[NUM_STATES][MAX_SPLITS]

cdef int _tables_ready = 0

cdef void _init_tables() noexcept nogil:
    global _tables_ready
    if _tables_ready:
        return

    _COAL_N[0]=1;  _COAL_TGT[0][0]=0;  _COAL_LI[0][0]=0; _COAL_LO[0][0]=0
    _COAL_N[1]=1;  _COAL_TGT[1][0]=1;  _COAL_LI[1][0]=1; _COAL_LO[1][0]=1
    _COAL_N[2]=1;  _COAL_TGT[2][0]=2;  _COAL_LI[2][0]=1; _COAL_LO[2][0]=1
    _COAL_N[3]=1;  _COAL_TGT[3][0]=3;  _COAL_LI[3][0]=1; _COAL_LO[3][0]=1
    _COAL_N[4]=2;  _COAL_TGT[4][0]=4;  _COAL_LI[4][0]=2; _COAL_LO[4][0]=2
    _COAL_TGT[4][1]=8;  _COAL_LI[4][1]=2; _COAL_LO[4][1]=1
    _COAL_N[5]=1;  _COAL_TGT[5][0]=5;  _COAL_LI[5][0]=2; _COAL_LO[5][0]=2
    _COAL_N[6]=1;  _COAL_TGT[6][0]=6;  _COAL_LI[6][0]=2; _COAL_LO[6][0]=2
    _COAL_N[7]=3;  _COAL_TGT[7][0]=7;  _COAL_LI[7][0]=3; _COAL_LO[7][0]=3
    _COAL_TGT[7][1]=9;  _COAL_LI[7][1]=3; _COAL_LO[7][1]=2
    _COAL_TGT[7][2]=10; _COAL_LI[7][2]=3; _COAL_LO[7][2]=1
    _COAL_N[8]=1;  _COAL_TGT[8][0]=8;  _COAL_LI[8][0]=1; _COAL_LO[8][0]=1
    _COAL_N[9]=2;  _COAL_TGT[9][0]=9;  _COAL_LI[9][0]=2; _COAL_LO[9][0]=2
    _COAL_TGT[9][1]=10; _COAL_LI[9][1]=2; _COAL_LO[9][1]=1
    _COAL_N[10]=1; _COAL_TGT[10][0]=10; _COAL_LI[10][0]=1; _COAL_LO[10][0]=1

    _SPLIT_N[0]=1;  _SPLIT_S1[0][0]=0; _SPLIT_S2[0][0]=0
    _SPLIT_N[1]=1;  _SPLIT_S1[1][0]=0; _SPLIT_S2[1][0]=1
    _SPLIT_N[2]=1;  _SPLIT_S1[2][0]=0; _SPLIT_S2[2][0]=2
    _SPLIT_N[3]=1;  _SPLIT_S1[3][0]=0; _SPLIT_S2[3][0]=3
    _SPLIT_N[4]=2;  _SPLIT_S1[4][0]=0; _SPLIT_S2[4][0]=4
    _SPLIT_S1[4][1]=1; _SPLIT_S2[4][1]=2
    _SPLIT_N[5]=2;  _SPLIT_S1[5][0]=0; _SPLIT_S2[5][0]=5
    _SPLIT_S1[5][1]=1; _SPLIT_S2[5][1]=3
    _SPLIT_N[6]=2;  _SPLIT_S1[6][0]=0; _SPLIT_S2[6][0]=6
    _SPLIT_S1[6][1]=2; _SPLIT_S2[6][1]=3
    _SPLIT_N[7]=4;  _SPLIT_S1[7][0]=0; _SPLIT_S2[7][0]=7
    _SPLIT_S1[7][1]=1; _SPLIT_S2[7][1]=6
    _SPLIT_S1[7][2]=2; _SPLIT_S2[7][2]=5
    _SPLIT_S1[7][3]=3; _SPLIT_S2[7][3]=4
    _SPLIT_N[8]=1;  _SPLIT_S1[8][0]=0; _SPLIT_S2[8][0]=8
    _SPLIT_N[9]=2;  _SPLIT_S1[9][0]=0; _SPLIT_S2[9][0]=9
    _SPLIT_S1[9][1]=3; _SPLIT_S2[9][1]=8
    _SPLIT_N[10]=1; _SPLIT_S1[10][0]=0; _SPLIT_S2[10][0]=10

    _tables_ready = 1

# ===================================================================
# Utility helpers
# ===================================================================

cdef inline void clear_sm(StateMap* sm) noexcept nogil:
    cdef int i
    for i in range(NUM_STATES):
        sm.slots[i].n = 0

cdef inline void cfg_copy(Config* dst, Config* src, int nr) noexcept nogil:
    dst.total_prob = src.total_prob
    cdef int i
    for i in range(nr):
        dst.idx[i] = src.idx[i]

cdef inline int cfg_match(Config* a, Config* b, int nr) noexcept nogil:
    cdef int i
    for i in range(nr):
        if a.idx[i] != b.idx[i]:
            return 0
    return 1

cdef inline int cfg_compatible(Config* a, Config* b, int nr) noexcept nogil:
    cdef int i
    for i in range(nr):
        if a.idx[i] != b.idx[i] and a.idx[i] != 0 and b.idx[i] != 0:
            return 0
    return 1

cdef inline void cfg_merge(Config* out, Config* a, Config* b, int nr) noexcept nogil:
    cdef double p = a.total_prob * b.total_prob
    out.total_prob = p if p > 0.0 else 0.0
    cdef int i
    for i in range(nr):
        out.idx[i] = a.idx[i] if a.idx[i] > b.idx[i] else b.idx[i]

cdef inline void cfg_clear(Config* c, int nr) noexcept nogil:
    cdef int i
    for i in range(nr):
        c.idx[i] = 0

cdef inline void cfg_zero(Config* c, int nr) noexcept nogil:
    c.total_prob = 0.0
    cdef int i
    for i in range(nr):
        c.idx[i] = 0

cdef void slot_add_dedup(StateSlot* sl, Config* cfg, int nr) noexcept nogil:
    """Add config to slot, summing prob if idx vector already exists."""
    cdef int i
    for i in range(sl.n):
        if cfg_match(&sl.items[i], cfg, nr):
            sl.items[i].total_prob = sl.items[i].total_prob + cfg.total_prob
            return
    if sl.n < MAX_CONFIGS:
        cfg_copy(&sl.items[sl.n], cfg, nr)
        sl.n = sl.n + 1

cdef inline void slot_append(StateSlot* sl, Config* cfg, int nr) noexcept nogil:
    """Append without dedup."""
    if sl.n < MAX_CONFIGS:
        cfg_copy(&sl.items[sl.n], cfg, nr)
        sl.n = sl.n + 1

# ===================================================================
# gij - coalescent transition probability
# ===================================================================

cdef double _fact(int start, int end) noexcept nogil:
    cdef double r = 1.0
    cdef int i
    for i in range(start, end + 1):
        r = r * <double>i
    return r

cdef double gij(double length, int i, int j) noexcept nogil:
    if length < 0.0:
        return 1.0 if j == 1 else 0.0
    if length == 0.0:
        return 1.0 if i == j else 0.0
    if i == 0:
        return 1.0

    cdef double result = 0.0
    cdef int k
    cdef double sign, tmp, denom
    for k in range(j, i + 1):
        sign = 1.0 if ((k - j) % 2 == 0) else -1.0
        tmp = (
            exp(0.5 * <double>k * (1.0 - <double>k) * length)
            * (2.0 * <double>k - 1.0)
            * sign
            * _fact(j, j + k - 2)
            * _fact(i - k + 1, i)
        )
        denom = _fact(1, j) * _fact(1, k - j) * _fact(i, i + k - 1)
        result = result + tmp / denom
    return result

# ===================================================================
# Merging lookup
# ===================================================================

cdef inline int merge_lookup(int s1, int s2) noexcept nogil:
    cdef int a, b
    if s1 < s2:
        a = s1; b = s2
    else:
        a = s2; b = s1
    if a == 1:
        if b == 2: return 4
        if b == 3: return 5
        if b == 6: return 7
    elif a == 2:
        if b == 3: return 6
        if b == 5: return 7
    elif a == 3:
        if b == 4: return 7
        if b == 8: return 9
    return -1

# ===================================================================
# compute_ac_minus  (branch transition)
# ===================================================================

cdef void compute_ac_minus(
    StateMap* result,
    StateMap* cacs,
    double branch_length,
    double inheritance_prob,
    int nr,
) noexcept nogil:
    clear_sm(result)

    cdef int sid, ci, coal_idx
    cdef int tgt, lin_in, lin_out
    cdef double prob, gij_val, new_prob
    cdef Config tmp

    for sid in range(NUM_STATES):
        if cacs.slots[sid].n == 0:
            continue
        for coal_idx in range(_COAL_N[sid]):
            tgt = _COAL_TGT[sid][coal_idx]
            lin_in = _COAL_LI[sid][coal_idx]
            lin_out = _COAL_LO[sid][coal_idx]

            prob = 1.0
            if inheritance_prob != 1.0:
                if lin_in == 1:
                    prob = inheritance_prob
                elif lin_in == 2:
                    prob = inheritance_prob * inheritance_prob
                elif lin_in == 3:
                    prob = inheritance_prob * inheritance_prob * inheritance_prob

            if lin_in > 1:
                gij_val = gij(branch_length, lin_in, lin_out)
                prob = prob * gij_val
                if lin_in == 3 and lin_out != 3:
                    prob = prob / 3.0

            if prob == 0.0:
                continue

            for ci in range(cacs.slots[sid].n):
                cfg_copy(&tmp, &cacs.slots[sid].items[ci], nr)
                new_prob = tmp.total_prob * prob
                tmp.total_prob = new_prob if new_prob > 0.0 else 0.0
                slot_add_dedup(&result.slots[tgt], &tmp, nr)

# ===================================================================
# split_at_retic  (reticulation node split)
# ===================================================================

cdef int split_at_retic(
    StateMap* ap1,
    StateMap* ap2,
    StateMap* cacs,
    int net_node_id,
    int nr,
) noexcept nogil:
    """Returns updated net_index after all splits."""
    clear_sm(ap1)
    clear_sm(ap2)

    cdef int sid, sp_idx, ci
    cdef int s1, s2
    cdef double prob
    cdef Config cfg1, cfg2
    cdef int net_index = 1

    for sid in range(NUM_STATES):
        if cacs.slots[sid].n == 0:
            continue
        for sp_idx in range(_SPLIT_N[sid]):
            s1 = _SPLIT_S1[sid][sp_idx]
            s2 = _SPLIT_S2[sid][sp_idx]

            for ci in range(cacs.slots[sid].n):
                prob = sqrt(cacs.slots[sid].items[ci].total_prob)

                # Forward: cfg1 -> ap1[s1],  cfg2 -> ap2[s2]
                cfg_copy(&cfg1, &cacs.slots[sid].items[ci], nr)
                cfg1.idx[net_node_id] = net_index
                cfg1.total_prob = prob
                cfg_copy(&cfg2, &cfg1, nr)
                slot_append(&ap1.slots[s1], &cfg1, nr)
                slot_append(&ap2.slots[s2], &cfg2, nr)
                net_index = net_index + 1

                # Opposite (only for non-empty states)
                if sid != 0:
                    cfg_copy(&cfg1, &cacs.slots[sid].items[ci], nr)
                    cfg1.idx[net_node_id] = net_index
                    cfg1.total_prob = prob
                    cfg_copy(&cfg2, &cfg1, nr)
                    slot_append(&ap1.slots[s2], &cfg1, nr)
                    slot_append(&ap2.slots[s1], &cfg2, nr)
                    net_index = net_index + 1

    return net_index

# ===================================================================
# Articulation compression helpers
# ===================================================================

cdef void copy_with_art(
    StateMap* dst, StateMap* src, int is_low_art, int nr
) noexcept nogil:
    """Copy src to dst; compress configs at lowest articulation nodes."""
    cdef int s, i
    if not is_low_art:
        memcpy(dst, src, sizeof(StateMap))
        return
    clear_sm(dst)
    for s in range(NUM_STATES):
        if src.slots[s].n == 0:
            continue
        cfg_copy(&dst.slots[s].items[0], &src.slots[s].items[0], nr)
        for i in range(1, src.slots[s].n):
            dst.slots[s].items[0].total_prob = (
                dst.slots[s].items[0].total_prob + src.slots[s].items[i].total_prob
            )
        cfg_clear(&dst.slots[s].items[0], nr)
        dst.slots[s].n = 1

cdef void merge_children(
    StateMap* dst,
    StateMap* ac1,
    StateMap* ac2,
    int is_low_art,
    int nr,
) noexcept nogil:
    """Merge two children state maps at a tree node."""
    clear_sm(dst)

    cdef int s1, s2, ci1, ci2
    cdef int can_merge, tgt, mr
    cdef Config merged
    cdef double p

    for s1 in range(NUM_STATES):
        if ac1.slots[s1].n == 0:
            continue
        for s2 in range(NUM_STATES):
            if ac2.slots[s2].n == 0:
                continue

            can_merge = 0
            tgt = 0
            if s1 == 0 or s2 == 0:
                can_merge = 1
                tgt = s1 if s1 != 0 else s2
            else:
                mr = merge_lookup(s1, s2)
                if mr >= 0:
                    can_merge = 1
                    tgt = mr

            if not can_merge:
                continue

            for ci1 in range(ac1.slots[s1].n):
                for ci2 in range(ac2.slots[s2].n):
                    if not cfg_compatible(
                        &ac1.slots[s1].items[ci1],
                        &ac2.slots[s2].items[ci2],
                        nr,
                    ):
                        continue

                    if is_low_art:
                        if dst.slots[tgt].n == 0:
                            cfg_merge(
                                &merged,
                                &ac1.slots[s1].items[ci1],
                                &ac2.slots[s2].items[ci2],
                                nr,
                            )
                            cfg_clear(&merged, nr)
                            cfg_copy(&dst.slots[tgt].items[0], &merged, nr)
                            dst.slots[tgt].n = 1
                        else:
                            p = (ac1.slots[s1].items[ci1].total_prob
                                 * ac2.slots[s2].items[ci2].total_prob)
                            if p < 0.0:
                                p = 0.0
                            dst.slots[tgt].items[0].total_prob = (
                                dst.slots[tgt].items[0].total_prob + p
                            )
                    else:
                        cfg_merge(
                            &merged,
                            &ac1.slots[s1].items[ci1],
                            &ac2.slots[s2].items[ci2],
                            nr,
                        )
                        slot_append(&dst.slots[tgt], &merged, nr)

# ===================================================================
# Main per-triple DP
# ===================================================================

cdef double calc_triple(
    int n_nodes,
    int nr,
    int* is_leaf,
    int* is_retic,
    int* in_art,
    int* in_low_art,
    int* n_ch,
    int* ch,
    int* n_pa,
    int* pa,
    double* pa_bl,
    double* pa_gamma,
    int* ch_pa_slot,
    int triple[3],
    StateMap* store,
    int* active,
    StateMap* tmp_cacs,
    StateMap* tmp_ap1,
    StateMap* tmp_ap2,
) noexcept nogil:
    cdef int i, j, k
    cdef int node_idx, c0, c1, slot0, slot1
    cdef int has_ac1, has_ac2, has_cacs
    cdef StateMap* cacs
    cdef StateMap* ac1_ptr
    cdef StateMap* ac2_ptr
    cdef double bl, gamma, total_prob
    cdef int net_node_id = 0

    memset(active, 0, n_nodes * MAX_PARENTS * sizeof(int))
    total_prob = 0.0

    for i in range(n_nodes):
        node_idx = i
        has_cacs = 0

        if is_leaf[node_idx]:
            for j in range(3):
                if triple[j] == node_idx:
                    clear_sm(tmp_cacs)
                    tmp_cacs.slots[j + 1].n = 1
                    tmp_cacs.slots[j + 1].items[0].total_prob = 1.0
                    for k in range(nr):
                        tmp_cacs.slots[j + 1].items[0].idx[k] = 0
                    cacs = tmp_cacs
                    has_cacs = 1
                    break

        elif is_retic[node_idx]:
            if n_ch[node_idx] > 0:
                c0 = ch[node_idx * MAX_CHILDREN]
                slot0 = ch_pa_slot[node_idx * MAX_CHILDREN]
                if active[c0 * MAX_PARENTS + slot0]:
                    cacs = &store[c0 * MAX_PARENTS + slot0]
                    has_cacs = 1

        else:
            if n_ch[node_idx] >= 2:
                c0 = ch[node_idx * MAX_CHILDREN]
                c1 = ch[node_idx * MAX_CHILDREN + 1]
                slot0 = ch_pa_slot[node_idx * MAX_CHILDREN]
                slot1 = ch_pa_slot[node_idx * MAX_CHILDREN + 1]

                has_ac1 = active[c0 * MAX_PARENTS + slot0]
                has_ac2 = active[c1 * MAX_PARENTS + slot1]

                if has_ac1 and not has_ac2:
                    ac1_ptr = &store[c0 * MAX_PARENTS + slot0]
                    copy_with_art(tmp_cacs, ac1_ptr, in_low_art[node_idx], nr)
                    cacs = tmp_cacs
                    has_cacs = 1
                elif has_ac2 and not has_ac1:
                    ac2_ptr = &store[c1 * MAX_PARENTS + slot1]
                    copy_with_art(tmp_cacs, ac2_ptr, in_low_art[node_idx], nr)
                    cacs = tmp_cacs
                    has_cacs = 1
                elif has_ac1 and has_ac2:
                    ac1_ptr = &store[c0 * MAX_PARENTS + slot0]
                    ac2_ptr = &store[c1 * MAX_PARENTS + slot1]
                    merge_children(
                        tmp_cacs, ac1_ptr, ac2_ptr,
                        in_low_art[node_idx], nr,
                    )
                    cacs = tmp_cacs
                    has_cacs = 1

        if not has_cacs:
            continue

        # Check articulation extraction (all 3 lineages present)
        if cacs.slots[7].n > 0 and in_art[node_idx]:
            total_prob = cacs.slots[7].items[0].total_prob / 3.0
            if cacs.slots[9].n > 0:
                total_prob = total_prob + cacs.slots[9].items[0].total_prob
            if cacs.slots[10].n > 0:
                total_prob = total_prob + cacs.slots[10].items[0].total_prob
            return total_prob

        # Propagate to parent edges
        if is_retic[node_idx]:
            split_at_retic(tmp_ap1, tmp_ap2, cacs, net_node_id, nr)
            net_node_id = net_node_id + 1

            for j in range(n_pa[node_idx]):
                bl = pa_bl[node_idx * MAX_PARENTS + j]
                gamma = pa_gamma[node_idx * MAX_PARENTS + j]
                if j == 0:
                    compute_ac_minus(
                        &store[node_idx * MAX_PARENTS + j],
                        tmp_ap1, bl, gamma, nr,
                    )
                else:
                    compute_ac_minus(
                        &store[node_idx * MAX_PARENTS + j],
                        tmp_ap2, bl, gamma, nr,
                    )
                active[node_idx * MAX_PARENTS + j] = 1
        else:
            if n_pa[node_idx] > 0:
                bl = pa_bl[node_idx * MAX_PARENTS]
                compute_ac_minus(
                    &store[node_idx * MAX_PARENTS],
                    cacs, bl, 1.0, nr,
                )
                active[node_idx * MAX_PARENTS] = 1

    return total_prob

# ===================================================================
# Python entry point
# ===================================================================

def score_all_triplets(
    int n_nodes,
    int net_node_num,
    int[::1] is_leaf,
    int[::1] is_retic,
    int[::1] in_art,
    int[::1] in_low_art,
    int[::1] n_children,
    int[:, ::1] children,
    int[::1] n_parents,
    int[:, ::1] parents,
    double[:, ::1] pa_bl,
    double[:, ::1] pa_gamma,
    int[:, ::1] ch_pa_slot,
    list triplets_indexed,
    list rho_list,
):
    """Score all triplets on the pre-extracted network topology.

    Parameters are flat numpy arrays produced by _extract_topology_for_cython
    in MPL.py.  Returns the total log pseudo-likelihood (float).
    """
    _init_tables()

    if n_nodes > MAX_NODES:
        raise ValueError(
            f"Network has {n_nodes} nodes, exceeds compiled limit {MAX_NODES}"
        )
    if net_node_num > MAX_RETICS:
        raise ValueError(
            f"Network has {net_node_num} reticulations, exceeds compiled limit {MAX_RETICS}"
        )

    cdef StateMap* store = <StateMap*>malloc(
        MAX_NODES * MAX_PARENTS * sizeof(StateMap)
    )
    cdef int* act = <int*>malloc(MAX_NODES * MAX_PARENTS * sizeof(int))

    if store == NULL or act == NULL:
        if store != NULL:
            free(store)
        if act != NULL:
            free(act)
        raise MemoryError("Failed to allocate DP workspace")

    cdef StateMap tmp_cacs, tmp_ap1, tmp_ap2
    cdef double total = 0.0
    cdef double p_xy, p_xz, p_yz
    cdef int triple[3]
    cdef double rho[3]
    cdef int ti, n_trips
    cdef double log_p

    cdef int* c_is_leaf = &is_leaf[0]
    cdef int* c_is_retic = &is_retic[0]
    cdef int* c_in_art = &in_art[0]
    cdef int* c_in_low_art = &in_low_art[0]
    cdef int* c_n_ch = &n_children[0]
    cdef int* c_ch = &children[0, 0]
    cdef int* c_n_pa = &n_parents[0]
    cdef int* c_pa = &parents[0, 0]
    cdef double* c_pa_bl = &pa_bl[0, 0]
    cdef double* c_pa_gamma = &pa_gamma[0, 0]
    cdef int* c_ch_pa_slot = &ch_pa_slot[0, 0]

    n_trips = len(triplets_indexed)

    try:
        for ti in range(n_trips):
            trip = triplets_indexed[ti]
            r = rho_list[ti]

            triple[0] = trip[0]
            triple[1] = trip[1]
            triple[2] = trip[2]
            rho[0] = r[0]
            rho[1] = r[1]
            rho[2] = r[2]

            # P(xy|z) with ordering (x, y, z)
            p_xy = calc_triple(
                n_nodes, net_node_num,
                c_is_leaf, c_is_retic, c_in_art, c_in_low_art,
                c_n_ch, c_ch, c_n_pa, c_pa,
                c_pa_bl, c_pa_gamma, c_ch_pa_slot,
                triple, store, act,
                &tmp_cacs, &tmp_ap1, &tmp_ap2,
            )

            # P(xz|y) with ordering (x, z, y)
            triple[1] = trip[2]
            triple[2] = trip[1]
            p_xz = calc_triple(
                n_nodes, net_node_num,
                c_is_leaf, c_is_retic, c_in_art, c_in_low_art,
                c_n_ch, c_ch, c_n_pa, c_pa,
                c_pa_bl, c_pa_gamma, c_ch_pa_slot,
                triple, store, act,
                &tmp_cacs, &tmp_ap1, &tmp_ap2,
            )

            p_yz = 1.0 - p_xy - p_xz
            if p_yz < 0.0:
                p_yz = 0.0

            if rho[0] > 0.0:
                log_p = log(p_xy) if p_xy > 0.0 else _LOG_FLOOR
                total = total + rho[0] * log_p
            if rho[1] > 0.0:
                log_p = log(p_xz) if p_xz > 0.0 else _LOG_FLOOR
                total = total + rho[1] * log_p
            if rho[2] > 0.0:
                log_p = log(p_yz) if p_yz > 0.0 else _LOG_FLOOR
                total = total + rho[2] * log_p
    finally:
        free(store)
        free(act)

    return total
