"""One-off helper to bootstrap src/_msnc_density.py from _mcmc_gt.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lines = (ROOT / "src/_mcmc_gt.py").read_text(encoding="utf-8").splitlines()
chunks = [
    lines[209:211],
    lines[474:876],
    lines[1347:1366],
    lines[1639:2054],
]
header = r'''#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared MSNC density: ancestral-configurations DP (Yu, Degnan and Nakhleh 2012).

Used by :mod:`phynetpy._mcmc_gt` and :mod:`phynetpy._seq_likelihood` / MCMC_SEQ.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional, Sequence

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

'''
kernel = r'''
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


def gene_tree_msnc_log_density(
    gene_tree: Network,
    species_net: Network,
    species_of: dict[str, str],
    *,
    theta: float = 0.02,
    pop_sizes: Optional[dict] = None,
) -> float:
    """Log MSNC density log P(g | Psi) via the joint-edge AC DP."""
    if pop_sizes is not None:
        raise NotImplementedError(
            "per-branch pop_sizes not yet supported in shared MSNC DP"
        )
    kernel = MSCBranchKernel(theta=theta)
    net_idx = _NetworkIndex(species_net)
    gti = _GeneTreeIndex(gene_tree, species_of)
    if net_idx.n_nodes == 0 or net_idx.root < 0:
        return float("-inf")
    score = _msnc_log_prob_network_int(net_idx, gti, kernel)
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

'''
body = "\n".join("\n".join(c) for c in chunks)
body = body.replace('"_GTLikelihoodEngine"', '"MSCBranchKernel"')
body = body.replace(": _GTLikelihoodEngine", ": MSCBranchKernel")
body = body.replace('engine: "_GTLikelihoodEngine"', "engine: MSCBranchKernel")
out = ROOT / "src/_msnc_density.py"
out.write_text(header + kernel + body, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes)")
