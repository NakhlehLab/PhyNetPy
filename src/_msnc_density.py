#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared MSNC density: ancestral-configurations DP (Yu, Degnan and Nakhleh 2012).

Used by :mod:`phynetpy._mcmc_gt` and :mod:`phynetpy._seq_likelihood` / MCMC_SEQ.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional, Sequence

import numpy as np

from .Network import Network, Node

try:
    from .cython.gt_msc_cy import (
        apply_branch_coalescent_cy as _apply_branch_coalescent_cy,
        combine_configs_cy as _combine_configs_cy,
    )
    _CYTHON_AVAILABLE = True
except ImportError:
    _apply_branch_coalescent_cy = None
    _combine_configs_cy = None
    _CYTHON_AVAILABLE = False


class MSCBranchKernel:
    """Kingman branch-coalescent kernel with explicit theta (4 N mu)."""

    def __init__(self, theta: float = 2.0) -> None:
        self.theta = float(theta)
        self._log_gij_cache: dict[tuple, float] = {}
        self._log_denom_cache: dict[tuple[int, int], float] = {}

    def _scaled_length(self, length: float | None) -> float | None:
        if length is None:
            return None
        return length * 2.0 / self.theta

    @staticmethod
    def _fact_range(start: int, end: int) -> float:
        if end < start:
            return 1.0
        result = 1.0
        for i in range(start, end + 1):
            result *= i
        return result

    @staticmethod
    def _gij_raw(length: float | None, i: int, j: int) -> float:
        if length is None or length == -1:
            return 1.0 if j == 1 else 0.0
        if length == 0:
            return 1.0 if i == j else 0.0
        if i == 0:
            return 1.0
        _fact = MSCBranchKernel._fact_range
        result = 0.0
        for k in range(j, i + 1):
            temp = (
                math.exp(0.5 * k * (1.0 - k) * length)
                * (2.0 * k - 1.0)
                * ((-1.0) ** (k - j))
                * _fact(j, j + k - 2)
                * _fact(i - k + 1, i)
            )
            denom = _fact(1, j) * _fact(1, k - j) * _fact(i, i + k - 1)
            result += temp / denom
        if result < 0.0:
            return 0.0
        if result > 1.0:
            return 1.0
        return result

    def _log_gij(self, length: float | None, i: int, j: int) -> float:
        key = (length, i, j, self.theta)
        cached = self._log_gij_cache.get(key)
        if cached is not None:
            return cached
        val = self._gij_raw(self._scaled_length(length), i, j)
        log_val = _LOG_FLOOR if val <= 0.0 else math.log(val)
        self._log_gij_cache[key] = log_val
        return log_val

    def _log_denom(self, n: int, k: int) -> float:
        if k <= 0:
            return 0.0
        key = (n, k)
        cached = self._log_denom_cache.get(key)
        if cached is not None:
            return cached
        denom = 1
        for i in range(1, k + 1):
            denom *= ((n - i + 1) * (n - i)) >> 1
        val = _LOG_FLOOR if denom <= 0 else math.log(denom)
        self._log_denom_cache[key] = val
        return val


def _node_height(net: Network, node: Node, cache: dict[Node, float]) -> float:
    """Ultrametric height above the present from child branch lengths."""
    cached = cache.get(node)
    if cached is not None:
        return cached
    children = net.get_children(node)
    if not children:
        cache[node] = 0.0
        return 0.0
    best = 0.0
    for child in children:
        edges = net.get_edge(node, child)
        edge = edges[0] if isinstance(edges, list) else edges
        length = edge.get_length()
        if length is None:
            length = 0.0
        best = max(best, _node_height(net, child, cache) + float(length))
    cache[node] = best
    return best


def _gene_coalescence_events(
    gene_tree: Network, gti: "_GeneTreeIndex"
) -> list[tuple[float, int, int, int]]:
    """Sorted gene-tree coalescences as (time, parent_bit, child0_bit, child1_bit)."""
    height_cache: dict[Node, float] = {}
    events: list[tuple[float, int, int, int]] = []
    for node in gene_tree.V():
        kids = gene_tree.get_children(node)
        if len(kids) == 2:
            events.append(
                (
                    _node_height(gene_tree, node, height_cache),
                    gti.bit_of[node],
                    gti.bit_of[kids[0]],
                    gti.bit_of[kids[1]],
                )
            )
    events.sort(key=lambda e: e[0])
    return events


def _apply_branch_timed(
    config_in: dict[int, float],
    tau_low: float,
    tau_high: Optional[float],
    theta: float,
    events: list[tuple[float, int, int, int]],
) -> dict[int, float]:
    """Event-based coalescent density along one species branch (PhyloNet MCMC_SEQ)."""
    inv_theta = 1.0 / theta
    log_two_over_theta = math.log(2.0 * inv_theta)
    out: dict[int, float] = {}
    for cfg, lp in config_in.items():
        cur = tau_low
        u = _popcount(cfg)
        cfg_cur = cfg
        for (t, pid, c0, c1) in events:
            if t < tau_low:
                continue
            if tau_high is not None and t >= tau_high:
                break
            b0, b1 = 1 << c0, 1 << c1
            if (cfg_cur & b0) and (cfg_cur & b1):
                lp_branch = -(t - cur) * u * (u - 1) * inv_theta
                lp_branch += log_two_over_theta
                cur = t
                cfg_cur = (cfg_cur & ~b0 & ~b1) | (1 << pid)
                u -= 1
                lp += lp_branch
        if tau_high is not None and u > 1:
            lp += -(tau_high - cur) * u * (u - 1) * inv_theta
        prev = out.get(cfg_cur)
        if prev is None:
            out[cfg_cur] = lp
        else:
            out[cfg_cur] = float(np.logaddexp(prev, lp))
    return out


def _msnc_log_density_timed(
    net_idx: "_NetworkIndex",
    gti: "_GeneTreeIndex",
    events: list[tuple[float, int, int, int]],
    sp_heights: list[float],
    theta: float,
) -> float:
    """Joint-edge AC DP with timed (event-based) branch coalescent factors."""
    if gti.root_bit < 0 or not gti.leaves:
        return 0.0
    if net_idx.n_nodes == 0 or net_idx.root < 0:
        return _LOG_FLOOR

    species_to_bits: dict[str, int] = {}
    for leaf_bit in gti.leaves:
        sp = gti.leaf_species_of.get(leaf_bit)
        if sp is not None:
            species_to_bits[sp] = species_to_bits.get(sp, 0) | (1 << leaf_bit)

    apply_branch = _apply_branch_timed
    log_floor = _LOG_FLOOR
    frontier: dict[tuple[tuple[int, int], ...], float] = {(): 0.0}

    for v in net_idx.topo_order:
        down_es = net_idx.down_edges[v]
        up_es = net_idx.up_edges[v]
        is_retic_v = net_idx.is_retic[v]
        new_frontier: dict[tuple[tuple[int, int], ...], float] = {}

        for key, lp in frontier.items():
            if not down_es:
                sp = net_idx.leaf_label.get(v)
                mask_at_v = species_to_bits.get(sp, 0) if sp is not None else 0
                new_key_base = key
            else:
                new_key_list = list(key)
                mask_at_v = 0
                ok = True
                for de in down_es:
                    found_idx = -1
                    for idx in range(len(new_key_list)):
                        if new_key_list[idx][0] == de:
                            found_idx = idx
                            break
                    if found_idx < 0:
                        ok = False
                        break
                    mask_at_v |= new_key_list[found_idx][1]
                    new_key_list.pop(found_idx)
                if not ok:
                    continue
                new_key_base = tuple(new_key_list)

            if not up_es:
                out = apply_branch(
                    {mask_at_v: 0.0}, sp_heights[v], None, theta, events
                )
                for top_mask, top_lp in out.items():
                    new_key = _frontier_insert(new_key_base, (-1, top_mask))
                    _frontier_acc(new_frontier, new_key, lp + top_lp)
                continue

            if len(up_es) >= 2 and is_retic_v:
                e1, e2 = up_es[0], up_es[1]
                gamma1 = net_idx.edge_gamma(e1)
                gamma2 = net_idx.edge_gamma(e2)
                if gamma1 is None and gamma2 is None:
                    gamma1 = 0.5
                    gamma2 = 0.5
                elif gamma1 is None:
                    gamma1 = max(0.0, 1.0 - float(gamma2))
                elif gamma2 is None:
                    gamma2 = max(0.0, 1.0 - float(gamma1))
                log_g1 = math.log(gamma1) if gamma1 > 0.0 else log_floor
                log_g2 = math.log(gamma2) if gamma2 > 0.0 else log_floor
                tau_low = sp_heights[v]
                n_total = _popcount(mask_at_v)
                S = mask_at_v
                while True:
                    k_s = _popcount(S)
                    factor = k_s * log_g1 + (n_total - k_s) * log_g2
                    as_mask = mask_at_v ^ S
                    parent1 = net_idx.edge_src[e1]
                    parent2 = net_idx.edge_src[e2]
                    out1 = apply_branch(
                        {S: 0.0}, tau_low, sp_heights[parent1], theta, events
                    )
                    out2 = apply_branch(
                        {as_mask: 0.0}, tau_low, sp_heights[parent2], theta, events
                    )
                    for top1, lp1 in out1.items():
                        for top2, lp2 in out2.items():
                            new_key = _frontier_insert(
                                new_key_base, (e1, top1)
                            )
                            new_key = _frontier_insert(new_key, (e2, top2))
                            _frontier_acc(
                                new_frontier, new_key, lp + factor + lp1 + lp2
                            )
                    if S == 0:
                        break
                    S = (S - 1) & mask_at_v
                continue

            e_up = up_es[0]
            parent_id = net_idx.edge_src[e_up]
            out = apply_branch(
                {mask_at_v: 0.0},
                sp_heights[v],
                sp_heights[parent_id],
                theta,
                events,
            )
            for top_mask, top_lp in out.items():
                new_key = _frontier_insert(new_key_base, (e_up, top_mask))
                _frontier_acc(new_frontier, new_key, lp + top_lp)

        frontier = new_frontier

    target = 1 << gti.root_bit
    log_terms: list[float] = []
    for key, lp in frontier.items():
        if len(key) != 1:
            continue
        eid, mask = key[0]
        if eid != -1:
            continue
        if mask == target:
            log_terms.append(lp)
        elif _popcount(mask) == 1 and (mask >> gti.root_bit) & 1:
            log_terms.append(lp)
    if not log_terms:
        return _LOG_FLOOR
    return _logsumexp(log_terms)


def gene_tree_msnc_log_density(
    gene_tree: Network,
    species_net: Network,
    species_of: dict[str, str],
    *,
    theta: float = 0.02,
    pop_sizes: Optional[dict] = None,
) -> float:
    """Log timed MSNC density log P(g | Psi) via joint-edge AC DP."""
    if pop_sizes is not None:
        raise NotImplementedError(
            "per-branch pop_sizes not yet supported in shared MSNC DP"
        )
    gti = _GeneTreeIndex(gene_tree, species_of)
    net_idx = _NetworkIndex(species_net)
    if net_idx.n_nodes == 0 or net_idx.root < 0:
        return float("-inf")
    height_cache: dict[Node, float] = {}
    sp_heights = [
        _node_height(species_net, net_idx._node_objs[i], height_cache)
        for i in range(net_idx.n_nodes)
    ]
    events = _gene_coalescence_events(gene_tree, gti)
    score = _msnc_log_density_timed(net_idx, gti, events, sp_heights, theta)
    if score <= _LOG_FLOOR + 1:
        return float("-inf")
    return float(score)


__all__ = [
    "MSCBranchKernel",
    "gene_tree_msnc_log_density",
    "_LOG_FLOOR",
    "_GeneTreeIndex",
    "_NetworkIndex",
    "_msnc_log_prob_network_int",
    "_msc_log_prob_tree_int",
    "_apply_branch_coalescent_int",
    "_combine_configs_int",
    "_logsumexp",
    "_popcount",
]

_LOG_FLOOR: float = math.log(1e-200)


class _GeneTreeIndex:
    """Per-gene-tree static-poset cache used by the bitmask MSC DP.

    Built once on first sight of a gene tree and reused for the rest
    of the engine's lifetime (gene trees are immutable inputs).  The
    DP operates exclusively on integer bitmasks keyed by ``bit``
    indices assigned here, which sidesteps the dominant cost of the
    earlier implementation: hashing ``Node`` objects into
    ``frozenset`` keys.

    The instance also memoises two purely-structural functions of the
    gene tree:

      * :meth:`linear_extensions` -- number of linear extensions of a
        sub-forest of internal-node bitmasks (hook-length formula).
        Permanent cache: depends only on the gene-tree topology.
      * :meth:`coarsenings` -- every reachable ``(active_out_mask,
        merge_mask)`` pair from a starting ``active_in_mask``.
        Permanent cache: same justification.

    Together, these two caches turn the inner DP into ``O(1)``
    table lookups for the second and subsequent calls per
    (gene tree, mask) -- exactly the regime an MH chain spends most
    of its time in once its small set of "interesting" lineage
    configurations has been enumerated.

    Attributes:
        n_total: Total number of distinct gene-tree nodes that
            received a bit index.
        bit_of: Map from gene-tree :class:`Node` to its bit index.
        leaves: List of bit indices for every gene-tree leaf.
        leaf_species_of: Map from a leaf's bit index to its species
            label (extracted from the configured ``species_of`` map).
        internals_mask: Bitmask of every internal gene-tree node.
        children: Map from an internal-node bit index to its two
            children's bit indices ``(left_bit, right_bit)``.
        parent: Map from every non-root bit index to its parent's
            bit index.
        root_bit: Bit index of the gene-tree root (or ``-1`` for
            an empty gene tree).
    """

    __slots__ = (
        "n_total",
        "bit_of",
        "leaves",
        "leaf_species_of",
        "internals_mask",
        "children",
        "parent",
        "root_bit",
        "_le_cache",
        "_coarse_cache",
    )

    def __init__(self, gene_tree: Network, species_of: dict[str, str]) -> None:
        """Build the bit-indexed view of ``gene_tree``.

        Args:
            gene_tree: The gene tree.  Reticulations in gene trees
                are not supported (assumed to be a strict rooted
                binary tree, possibly with one polytomy that we
                reduce to a binary approximation).
            species_of: Gene-copy label -> species label.  Used here
                only to populate :attr:`leaf_species_of`; the
                topology cache is unaffected by ``species_of`` so
                callers that swap species mappings can re-use the
                same index by re-binding through
                :meth:`refresh_species`.
        """
        self.bit_of: dict[Any, int] = {}
        self.leaves: list[int] = []
        self.leaf_species_of: dict[int, str] = {}
        self.children: dict[int, tuple[int, int]] = {}
        self.parent: dict[int, int] = {}
        self.root_bit: int = -1
        self._le_cache: dict[int, int] = {}
        self._coarse_cache: dict[int, tuple[tuple[int, int], ...]] = {}

        # Assign bit indices in a single pass over ``V()`` so two
        # calls on the same gene tree get the same bits (Network's
        # iteration order is deterministic per its own conventions).
        nodes = list(gene_tree.V())
        for idx, node in enumerate(nodes):
            self.bit_of[node] = idx
        self.n_total = len(nodes)

        internals_mask = 0
        for node in nodes:
            bit = self.bit_of[node]
            kids = gene_tree.get_children(node)
            if not kids:
                self.leaves.append(bit)
                lbl = species_of.get(node.label)
                if lbl is not None:
                    self.leaf_species_of[bit] = lbl
                continue
            if len(kids) > 2:
                # Soft-collapse polytomies to the first two children;
                # higher-arity nodes lower-bound the MSC likelihood
                # but at least keep the DP finite.
                kids = kids[:2]
            if len(kids) == 2:
                lb = self.bit_of[kids[0]]
                rb = self.bit_of[kids[1]]
                self.children[bit] = (lb, rb)
                self.parent[lb] = bit
                self.parent[rb] = bit
                internals_mask |= 1 << bit
        self.internals_mask = internals_mask

        if gene_tree.roots():
            try:
                root = gene_tree.root()
                self.root_bit = self.bit_of.get(root, -1)
            except Exception:
                self.root_bit = -1

    def refresh_species(self, gene_tree: Network, species_of: dict[str, str]) -> None:
        """Rebuild only :attr:`leaf_species_of` for a new mapping.

        Cheap: doesn't touch the topology / coarsening / linear-ext
        caches.  Useful when the same gene-tree set is scored under
        different allele -> species mappings.
        """
        self.leaf_species_of.clear()
        for node, bit in self.bit_of.items():
            if bit in self._cached_leaves_set():
                lbl = species_of.get(node.label)
                if lbl is not None:
                    self.leaf_species_of[bit] = lbl

    def _cached_leaves_set(self) -> set[int]:
        return set(self.leaves)

    # --- Static caches over masks ---------------------------------

    def linear_extensions(self, merge_mask: int) -> int:
        """``|L(F)|`` for the sub-forest of internals encoded by ``merge_mask``.

        Hook-length formula on the tree poset induced by gene-tree
        ancestry.  Cached permanently because the gene tree is
        immutable.
        """
        cached = self._le_cache.get(merge_mask)
        if cached is not None:
            return cached
        if merge_mask == 0:
            self._le_cache[merge_mask] = 1
            return 1
        # Single-bit fast path.
        if merge_mask & (merge_mask - 1) == 0:
            self._le_cache[merge_mask] = 1
            return 1
        # General case: |M|! / prod_v size_within(v)
        bits = _bits(merge_mask)
        k = len(bits)
        denom = 1
        for v_bit in bits:
            denom *= self._size_within(v_bit, merge_mask)
        result = math.factorial(k) // denom
        self._le_cache[merge_mask] = result
        return result

    def _size_within(self, root_bit: int, merge_mask: int) -> int:
        """``size`` of the subtree of ``merge_mask`` rooted at ``root_bit``.

        Walks descendants of ``root_bit`` in the gene tree, counting
        how many of them are also in ``merge_mask`` (including
        ``root_bit`` itself).  Bounded by ``popcount(merge_mask)``
        per call; small for the merge sets the DP actually visits.
        """
        children = self.children
        count = 0
        stack = [root_bit]
        while stack:
            cur = stack.pop()
            if (merge_mask >> cur) & 1:
                count += 1
            kids = children.get(cur)
            if kids is None:
                continue
            l, r = kids
            stack.append(l)
            stack.append(r)
        return count

    def coarsenings(
        self,
        active_mask: int,
    ) -> tuple[tuple[int, int, int, int, float], ...]:
        """Pre-computed coarsenings of ``active_mask``.

        Each row is ``(cfg_out, merge_mask, m_out, k, log_le)`` where:

          * ``cfg_out`` is the active-set bitmask after the merges,
          * ``merge_mask`` is the bitmask of internal-node bits picked
            up by the coarsening,
          * ``m_out = popcount(cfg_out)``,
          * ``k = popcount(active_mask) - m_out`` (number of merges),
          * ``log_le = log(linear_extensions(merge_mask))`` -- the
            hook-length factor in the per-coarsening branch
            probability.

        Pre-baking these scalars in the cache row eliminates 4 of
        the 6 hot-loop ops per coarsening (popcount, linear-ext
        lookup, ``math.log`` on the LE value); the inner DP only
        keeps ``log(g_ij)`` and the ``denom`` factor (both length-
        dependent and looked up against engine-level caches).
        """
        cached = self._coarse_cache.get(active_mask)
        if cached is not None:
            return cached

        # Iterative BFS over (active, merged) pairs.
        states: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        frontier = [(active_mask, 0)]
        children = self.children
        parent = self.parent
        while frontier:
            new_frontier: list[tuple[int, int]] = []
            for state in frontier:
                if state in seen:
                    continue
                seen.add(state)
                states.append(state)
                active, merged = state
                # Group active bits by parent.
                groups: dict[int, list[int]] = {}
                m = active
                while m:
                    bit = (m & -m).bit_length() - 1
                    m &= m - 1
                    p = parent.get(bit)
                    if p is None:
                        continue
                    groups.setdefault(p, []).append(bit)
                for p, kids in groups.items():
                    if (merged >> p) & 1:
                        continue
                    pc = children.get(p)
                    if pc is None:
                        continue
                    if len(kids) < 2:
                        continue
                    lb, rb = pc
                    if lb in kids and rb in kids:
                        new_active = (active & ~((1 << lb) | (1 << rb))) | (1 << p)
                        new_merged = merged | (1 << p)
                        nxt = (new_active, new_merged)
                        if nxt not in seen:
                            new_frontier.append(nxt)
            frontier = new_frontier

        # Bake popcounts and log(linear_extensions) into each row.
        n_in = _popcount(active_mask)
        rows: list[tuple[int, int, int, int, float]] = []
        for active_out, merge_mask in states:
            m_out = _popcount(active_out)
            k = n_in - m_out
            le_val = self.linear_extensions(merge_mask)
            log_le = math.log(le_val) if le_val > 0 else _LOG_FLOOR
            rows.append((active_out, merge_mask, m_out, k, log_le))
        result = tuple(rows)
        self._coarse_cache[active_mask] = result
        return result


def _bits(mask: int) -> list[int]:
    """Return the bit indices set in ``mask`` (low-to-high)."""
    out: list[int] = []
    while mask:
        lsb = mask & -mask
        out.append(lsb.bit_length() - 1)
        mask ^= lsb
    return out


def _popcount(mask: int) -> int:
    """Number of bits set in ``mask``.

    Uses :func:`int.bit_count` on Python 3.10+ when available,
    otherwise falls back to ``bin(mask).count("1")`` (still
    C-speed, just one extra string conversion per call).  Hot path
    of the bitmask MSC DP.
    """
    if mask < 0:
        mask = -mask
    return bin(mask).count("1")


class _NetworkIndex:
    """Int-indexed view of a phylogenetic species network for the AC DP.

    Built on first sight of a network and re-used as long as the
    topology is unchanged; :meth:`_GTLikelihoodEngine.update`
    invalidates and rebuilds it on a topology mutation.  Branch
    lengths and gammas are read fresh on every DP call via
    :meth:`edge_length` / :meth:`edge_gamma`, so parameter-only
    moves preserve the index.

    The DP in :func:`_msnc_log_prob_network_int` consumes this view
    instead of poking at :class:`Network` directly so the hot loop
    sees only ints (node IDs, edge IDs, lineage bitmasks).

    Attributes:
        n_nodes: Total number of network nodes.
        n_edges: Total number of network edges.
        is_retic: ``[i] -> True`` if node ``i`` is a reticulation.
        is_leaf: ``[i] -> True`` if node ``i`` has out-degree 0.
        leaf_label: Leaf node ID -> species label.
        down_edges: ``[i] -> list[edge_id]`` of edges going from
            ``i`` to ``i``'s children (= out-edges of ``i``).
        up_edges: ``[i] -> list[edge_id]`` of edges going from
            ``i``'s parents to ``i`` (= in-edges of ``i``).
        edge_src / edge_dst: Per-edge parent / child node IDs.
        topo_order: Bottom-up node-ID order (every node appears
            after all of its children).
        root: Node ID of the network root (unique node with no
            up-edges; ``-1`` for the empty-network case).
    """

    __slots__ = (
        "n_nodes",
        "n_edges",
        "is_retic",
        "is_leaf",
        "leaf_label",
        "down_edges",
        "up_edges",
        "edge_src",
        "edge_dst",
        "topo_order",
        "root",
        "_node_objs",
        "_edge_objs",
    )

    def __init__(self, net: Network) -> None:
        """Build the int-indexed view of ``net``.

        O(|V| + |E|) one-time cost; subsequent DP calls amortise this
        across every gene tree the chain scores against this network.
        """
        nodes = list(net.V())
        edges = list(net.E())
        self.n_nodes = len(nodes)
        self.n_edges = len(edges)
        self._node_objs = nodes
        self._edge_objs = edges
        node_to_id = {n: i for i, n in enumerate(nodes)}
        self.is_retic = [bool(n.is_reticulation()) for n in nodes]
        self.is_leaf = [net.out_degree(n) == 0 for n in nodes]
        self.leaf_label = {
            i: nodes[i].label
            for i in range(self.n_nodes)
            if self.is_leaf[i]
        }
        self.down_edges = [[] for _ in range(self.n_nodes)]
        self.up_edges = [[] for _ in range(self.n_nodes)]
        self.edge_src = [0] * self.n_edges
        self.edge_dst = [0] * self.n_edges
        for j, e in enumerate(edges):
            src_id = node_to_id[e.src]
            dst_id = node_to_id[e.dest]
            self.edge_src[j] = src_id
            self.edge_dst[j] = dst_id
            # Edges go src(parent) -> dst(child); "down-edge of src"
            # means the edge points from src down to its child, which
            # matches the DAG out-edge of src.
            self.down_edges[src_id].append(j)
            self.up_edges[dst_id].append(j)
        # Bottom-up topological order: a node is ready when all its
        # children (= dst of every edge in down_edges[v]) have been
        # popped from the queue.  Standard Kahn's algorithm.
        children_remaining = [
            len(self.down_edges[v]) for v in range(self.n_nodes)
        ]
        ready: deque[int] = deque(
            v for v in range(self.n_nodes) if children_remaining[v] == 0
        )
        order: list[int] = []
        while ready:
            v = ready.popleft()
            order.append(v)
            for j in self.up_edges[v]:
                p = self.edge_src[j]
                children_remaining[p] -= 1
                if children_remaining[p] == 0:
                    ready.append(p)
        self.topo_order = order
        roots = [v for v in range(self.n_nodes) if not self.up_edges[v]]
        self.root = roots[0] if roots else -1

    def edge_length(self, edge_id: int) -> "float | None":
        """Live-read the length of edge ``edge_id`` from the network."""
        return self._edge_objs[edge_id].get_length()

    def edge_gamma(self, edge_id: int) -> "float | None":
        """Live-read the inheritance prob of edge ``edge_id``."""
        return self._edge_objs[edge_id].get_gamma()


def _logsumexp(values: Sequence[float]) -> float:
    """Numerically stable ``log(sum(exp(v) for v in values))``.

    Kept local (instead of depending on scipy) because the rest of the
    PhyNetPy stack is numpy-only and we don't want to pull in scipy
    just for one helper.
    """
    if not values:
        return _LOG_FLOOR
    m = max(values)
    if m == -math.inf:
        return _LOG_FLOOR
    acc = 0.0
    for v in values:
        acc += math.exp(v - m)
    if acc <= 0.0:
        return _LOG_FLOOR
    return m + math.log(acc)


def _safe_add(a: float | None, b: float | None) -> float | None:
    """Add two possibly-``None`` branch lengths.

    Treats ``None`` as "no finite length" (the root edge / infinite
    branch); summing ``None`` with anything yields ``None``.  For
    finite inputs the usual sum applies.
    """
    if a is None or b is None:
        return None
    return float(a) + float(b)


def _msc_log_prob_tree_int(
    dt: "_DisplayedTree",
    gti: "_GeneTreeIndex",
    engine: "MSCBranchKernel",
) -> float:
    """Bitmask MSC DP: ``log P(gene_tree | dt)`` via int configs.

    Equivalent to :func:`_msc_log_prob_tree` (which it replaces in
    the hot path) but represents each lineage configuration as a
    Python ``int`` bitmask of gene-tree node bits.  This kills the
    ~60M ``Node.__hash__`` calls per chain step that dominated the
    pre-optimisation profile, and lets us reuse
    :class:`_GeneTreeIndex`'s permanent coarsening / linear-extension
    caches.

    Args:
        dt: Displayed tree (already contracted).
        gti: Per-gene-tree :class:`_GeneTreeIndex` (bit indices,
             children, parent maps, and static caches).
        engine: Backing :class:`_GTLikelihoodEngine` (for ``_gij``).

    Returns:
        Log probability; clamped at ``_LOG_FLOOR`` on incompatible
        configurations.
    """
    if gti.root_bit < 0 or not gti.leaves:
        return 0.0

    # Initial config at each species-leaf: a bitmask of all gene
    # leaves whose species maps there.
    species_to_bits: dict[str, int] = {}
    leaf_species = gti.leaf_species_of
    for leaf_bit in gti.leaves:
        sp = leaf_species.get(leaf_bit)
        if sp is None:
            continue
        species_to_bits[sp] = species_to_bits.get(sp, 0) | (1 << leaf_bit)

    n = dt._idx_n
    configs_at: list[dict[int, float] | None] = [None] * n
    leaf_species_dt = dt._idx_leaf_species
    for leaf_idx in dt._idx_leaves:
        sp = leaf_species_dt.get(leaf_idx)
        mask = species_to_bits.get(sp, 0) if sp is not None else 0
        configs_at[leaf_idx] = {mask: 0.0}

    children_idx = dt._idx_children
    edge_lengths_idx = dt._idx_edge_lengths
    apply_branch = _apply_branch_coalescent_int
    combine = _combine_configs_int
    for i in range(n):
        if configs_at[i] is not None:
            continue
        kids = children_idx[i]
        if not kids:
            configs_at[i] = {0: 0.0}
            continue
        lengths = edge_lengths_idx[i]
        child_top = [
            apply_branch(configs_at[kids[j]], lengths[j], gti, engine)
            for j in range(len(kids))
        ]
        merged = child_top[0]
        for nxt in child_top[1:]:
            merged = combine(merged, nxt)
        configs_at[i] = merged

    # Apply the infinite root-edge to force collapse to one lineage.
    root_idx = dt._idx_root
    root_config = configs_at[root_idx] if root_idx >= 0 else {}
    top_config = apply_branch(root_config or {}, None, gti, engine)
    target = 1 << gti.root_bit
    best = top_config.get(target)
    if best is not None:
        return best
    # Fall back: any single-active-bit config containing the gene-tree root.
    acc: list[float] = []
    for cfg, lp in top_config.items():
        if cfg == target:
            acc.append(lp)
        elif _popcount(cfg) == 1 and (cfg >> gti.root_bit) & 1:
            acc.append(lp)
    if not acc:
        return _LOG_FLOOR
    return _logsumexp(acc)


def _apply_branch_coalescent_int_py(
    config_in: dict[int, float],
    length: "float | None",
    gti: "_GeneTreeIndex",
    engine: "MSCBranchKernel",
) -> dict[int, float]:
    """Pure-Python bitmask version of :func:`_apply_branch_coalescent`.

    Hot inner loop of the MSC DP.  All length-independent quantities
    -- popcounts, ``log(linear_extensions)``, the merge-mask itself
    -- come pre-baked from :meth:`_GeneTreeIndex.coarsenings`; only
    ``log(g_ij)`` and ``log_denom`` (both functions of branch length)
    are looked up each call, against engine-level caches.

    The Cython extension :mod:`phynetpy.cython.gt_msc_cy` provides a
    drop-in replacement bound to :func:`_apply_branch_coalescent_int`
    when available; this Python copy is the fallback (and the
    ground truth used by the unit tests when the extension is
    rebuilt).
    """
    out: dict[int, list[float]] = {}
    coarsen = gti.coarsenings
    log_gij = engine._log_gij
    log_denom = engine._log_denom
    log_floor = _LOG_FLOOR
    for cfg_in, lp_in in config_in.items():
        n = _popcount(cfg_in)
        for cfg_out, merge_mask, m, k, log_le in coarsen(cfg_in):
            log_branch = log_gij(length, n, m)
            if log_branch <= log_floor:
                continue
            if k > 0:
                log_branch += log_le - log_denom(n, k)
            log_total = lp_in + log_branch
            existing = out.get(cfg_out)
            if existing is None:
                out[cfg_out] = [log_total]
            else:
                existing.append(log_total)
    result: dict[int, float] = {}
    for cfg, terms in out.items():
        if len(terms) == 1:
            result[cfg] = terms[0]
        else:
            result[cfg] = _logsumexp(terms)
    return result


def _combine_configs_int_py(
    left: dict[int, float],
    right: dict[int, float],
) -> dict[int, float]:
    """Pure-Python outer-product disjoint-union combine.

    Disjoint-union of bitmasks; entries with overlapping bits are
    skipped (mapping ill-formed) and same-union duplicates merged
    via log-sum-exp.  See :func:`_apply_branch_coalescent_int_py`
    for the Cython-acceleration story.
    """
    out: dict[int, list[float]] = {}
    for cfg_l, lp_l in left.items():
        for cfg_r, lp_r in right.items():
            if cfg_l & cfg_r:
                continue
            union = cfg_l | cfg_r
            existing = out.get(union)
            if existing is None:
                out[union] = [lp_l + lp_r]
            else:
                existing.append(lp_l + lp_r)
    result: dict[int, float] = {}
    for cfg, terms in out.items():
        if len(terms) == 1:
            result[cfg] = terms[0]
        else:
            result[cfg] = _logsumexp(terms)
    return result


# Bind the public names to the Cython extension when it's available;
# otherwise fall back to the pure-Python implementations above.  Both
# paths produce numerically-identical dicts (same logaddexp identity,
# same iteration order); the Cython path just runs the inner loop in
# C with libc ``log1p``/``exp`` and avoids the per-cfg-out list
# allocation entirely (see ``src/cython/gt_msc_cy.pyx``).
if _CYTHON_AVAILABLE:
    def _apply_branch_coalescent_int(
        config_in: dict[int, float],
        length: "float | None",
        gti: "_GeneTreeIndex",
        engine: "MSCBranchKernel",
    ) -> dict[int, float]:
        """Cython-accelerated wrapper for the per-branch coalescent step."""
        return _apply_branch_coalescent_cy(
            config_in,
            gti.coarsenings,
            engine._log_gij,
            engine._log_denom,
            length,
        )

    def _combine_configs_int(
        left: dict[int, float],
        right: dict[int, float],
    ) -> dict[int, float]:
        """Cython-accelerated wrapper for the disjoint-union combine."""
        return _combine_configs_cy(left, right)
else:
    _apply_branch_coalescent_int = _apply_branch_coalescent_int_py
    _combine_configs_int = _combine_configs_int_py


# ----------------------------------------------------------------------
# Full MSNC ancestral-configurations DP on the species network.
# ----------------------------------------------------------------------

def _msnc_log_prob_network_int(
    net_idx: "_NetworkIndex",
    gti: "_GeneTreeIndex",
    engine: "MSCBranchKernel",
) -> float:
    """Full MSNC ancestral-configurations DP on the species network.

    Implements the network-coalescent likelihood of Yu, Degnan &
    Nakhleh (2012; PLoS Genetics 8(4):e1002660) and underlies the
    MCMC inference of Yu & Nakhleh (2014; PNAS 111(46):16448-16453).

    State.  The DP carries a *frontier* dict mapping a sorted tuple
    of ``(species_edge_id, lineage_mask)`` pairs to log-probability.
    Each entry encodes the joint state on every currently-open
    species-network edge (the parent-side, "top-of-edge", lineage
    mask after branch coalescent has been applied).  An empty key
    ``()`` is the initial state (no edges open) at log_prob ``0.0``.

    Per-node operations (executed in bottom-up topological order):

      * **Leaf** (out-deg 0).  No down-edges to pull off the frontier.
        Seed the up-edge bottom with the bitmask of all gene-tree
        lineages mapped to this species, apply branch coalescent on
        the up-edge, append ``(up_edge_id, top_mask)`` to every
        frontier key with the corresponding log-prob accumulated.
      * **Tree internal** (in-deg 1, out-deg >= 1).  Pop the down-
        edges' ``(edge_id, mask)`` pairs from each frontier key
        (their masks are disjoint by construction; OR them) and
        apply branch coalescent on the single up-edge.
      * **Reticulation** (in-deg 2, out-deg 1).  Pop the (single)
        down-edge from each frontier key.  For every subset
        ``S`` of the popped mask ``A`` (``2^|A|`` subsets):
        route ``S`` up parent edge ``e1`` with weight
        ``gamma1^|S|``, and ``A\\S`` up parent edge ``e2`` with
        weight ``(1-gamma1)^|A\\S|``.  Each lineage independently
        chooses a parent (so the labelled count is exactly
        ``gamma^|S| (1-gamma)^|A\\S|``; no binomial coefficient).
        Apply branch coalescent on each parent edge separately.
        The resulting joint state on ``(e1, e2)`` is preserved
        through the frontier tuple and naturally collapses when
        the two parent-paths merge at the LCA of the retic.
      * **Root** (in-deg 0).  Pop all down-edges, OR their masks,
        apply infinite-branch coalescent (forces collapse to a
        single lineage).  The result is the gene-tree root with
        log-probability equal to the full MSNC log-likelihood.

    Args:
        net_idx: Pre-computed :class:`_NetworkIndex` view of the
            species network.
        gti: Pre-computed :class:`_GeneTreeIndex` view of the gene
            tree (provides bit indices, coarsening / linear-extension
            cache).
        engine: Backing :class:`_GTLikelihoodEngine` (provides
            ``_log_gij`` and ``_log_denom`` caches).

    Returns:
        ``log P(gene_tree | net_idx)``; clamped at ``_LOG_FLOOR`` for
        incompatible configurations.
    """
    if gti.root_bit < 0 or not gti.leaves:
        return 0.0
    if net_idx.n_nodes == 0 or net_idx.root < 0:
        return _LOG_FLOOR

    # Species label -> bitmask of all gene-tree lineages mapped there.
    species_to_bits: dict[str, int] = {}
    leaf_species_g = gti.leaf_species_of
    for leaf_bit in gti.leaves:
        sp = leaf_species_g.get(leaf_bit)
        if sp is None:
            continue
        species_to_bits[sp] = species_to_bits.get(sp, 0) | (1 << leaf_bit)

    apply_branch = _apply_branch_coalescent_int
    log_floor = _LOG_FLOOR
    log_gij = engine._log_gij  # noqa: F841 (referenced for caching warm-up)

    # Frontier: empty tuple at log-prob 0.
    frontier: dict[tuple[tuple[int, int], ...], float] = {(): 0.0}

    for v in net_idx.topo_order:
        down_es = net_idx.down_edges[v]
        up_es = net_idx.up_edges[v]
        is_retic_v = net_idx.is_retic[v]

        new_frontier: dict[tuple[tuple[int, int], ...], float] = {}

        for key, lp in frontier.items():
            # Compute mask_at_v and the trimmed key (drop v's down-edges).
            if not down_es:
                # Leaf: deterministic seed from the species map.
                sp = net_idx.leaf_label.get(v)
                mask_at_v = species_to_bits.get(sp, 0) if sp is not None else 0
                new_key_base = key
            else:
                # Internal: pop v's down-edges from key.
                new_key_list = list(key)
                mask_at_v = 0
                ok = True
                for de in down_es:
                    found_idx = -1
                    for idx in range(len(new_key_list)):
                        if new_key_list[idx][0] == de:
                            found_idx = idx
                            break
                    if found_idx < 0:
                        ok = False
                        break
                    mask_at_v |= new_key_list[found_idx][1]
                    new_key_list.pop(found_idx)
                if not ok:
                    # Down-edge missing from key -- shouldn't happen
                    # given the topo order, but be defensive.
                    continue
                new_key_base = tuple(new_key_list)

            # Push v's up-edges (with branch coalescent and, for
            # reticulations, the per-lineage gamma split).
            if not up_es:
                # Root: collapse to the gene-tree root by infinite
                # branch coalescent.  Encode the post-root config under
                # the sentinel edge ID -1.
                out = apply_branch({mask_at_v: 0.0}, None, gti, engine)
                for top_mask, top_lp in out.items():
                    new_key = _frontier_insert(new_key_base, (-1, top_mask))
                    _frontier_acc(new_frontier, new_key, lp + top_lp)
                continue

            if len(up_es) >= 2 and is_retic_v:
                # Reticulation: split each lineage independently among
                # the two parent edges.  Iterate every subset S of the
                # active mask.
                e1, e2 = up_es[0], up_es[1]
                gamma1 = net_idx.edge_gamma(e1)
                gamma2 = net_idx.edge_gamma(e2)
                if gamma1 is None and gamma2 is None:
                    gamma1 = 0.5
                    gamma2 = 0.5
                elif gamma1 is None:
                    gamma1 = max(0.0, 1.0 - float(gamma2))
                elif gamma2 is None:
                    gamma2 = max(0.0, 1.0 - float(gamma1))
                log_g1 = math.log(gamma1) if gamma1 > 0.0 else log_floor
                log_g2 = math.log(gamma2) if gamma2 > 0.0 else log_floor
                length1 = net_idx.edge_length(e1)
                length2 = net_idx.edge_length(e2)
                n_total = _popcount(mask_at_v)

                S = mask_at_v
                while True:
                    k_S = _popcount(S)
                    factor = k_S * log_g1 + (n_total - k_S) * log_g2
                    AS = mask_at_v ^ S
                    out1 = apply_branch({S: 0.0}, length1, gti, engine)
                    out2 = apply_branch({AS: 0.0}, length2, gti, engine)
                    for top1, lp1 in out1.items():
                        ent1 = (e1, top1)
                        for top2, lp2 in out2.items():
                            new_key = _frontier_insert(new_key_base, ent1)
                            new_key = _frontier_insert(new_key, (e2, top2))
                            _frontier_acc(
                                new_frontier,
                                new_key,
                                lp + factor + lp1 + lp2,
                            )
                    if S == 0:
                        break
                    S = (S - 1) & mask_at_v
                continue

            # Tree internal (in-deg 1) or pathological in-deg-1 retic.
            e_up = up_es[0]
            length = net_idx.edge_length(e_up)
            out = apply_branch({mask_at_v: 0.0}, length, gti, engine)
            for top_mask, top_lp in out.items():
                new_key = _frontier_insert(new_key_base, (e_up, top_mask))
                _frontier_acc(new_frontier, new_key, lp + top_lp)

        frontier = new_frontier

    # After processing the root: every key should be a singleton
    # ``((-1, mask),)``.  Sum (log-add) the log-probs of every entry
    # whose mask is the gene-tree root bit (or, defensively, any
    # single-bit mask containing the root bit).
    target = 1 << gti.root_bit
    log_terms: list[float] = []
    for key, lp in frontier.items():
        if len(key) != 1:
            continue
        eid, mask = key[0]
        if eid != -1:
            continue
        if mask == target:
            log_terms.append(lp)
        elif _popcount(mask) == 1 and (mask >> gti.root_bit) & 1:
            log_terms.append(lp)
    if not log_terms:
        return _LOG_FLOOR
    return _logsumexp(log_terms)


def _frontier_insert(
    tup: tuple[tuple[int, int], ...],
    item: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Insert ``item`` into a sorted-tuple frontier key, preserving order."""
    if not tup:
        return (item,)
    eid = item[0]
    out = list(tup)
    pos = 0
    while pos < len(out) and out[pos][0] < eid:
        pos += 1
    out.insert(pos, item)
    return tuple(out)


def _frontier_acc(
    frontier: dict[tuple[tuple[int, int], ...], float],
    key: tuple[tuple[int, int], ...],
    lp: float,
) -> None:
    """Accumulate ``lp`` into ``frontier[key]`` via log-sum-exp."""
    existing = frontier.get(key)
    if existing is None:
        frontier[key] = lp
    elif existing > lp:
        frontier[key] = existing + math.log1p(math.exp(lp - existing))
    else:
        frontier[key] = lp + math.log1p(math.exp(existing - lp))