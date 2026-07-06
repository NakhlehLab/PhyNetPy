#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

"""
Author : Mark Kessler
First Included in Version : 0.5.0

Shared helpers for the unified inference search flags
(``opt_bl`` / ``fix_st`` / ``max_lvl`` / ``pseudo``).

These flags are wired into the topology-based inference methods ``MPL``,
``MCMC_GT``, and ``InferNetwork_ML``.  This module centralises the two pieces
of cross-method logic so the three search drivers stay in lock-step:

* :func:`resolve_move_types` -- turn ``opt_bl`` / ``fix_st`` into the concrete
  set of move classes the proposal kernel may draw from.
* :func:`make_level_validator` -- compose the standard network-invariant
  validator with a ``GraphUtils.level(net) <= max_lvl`` guard so *every*
  accepted proposal is level-checked regardless of which move produced it.

Design notes
------------
* ``opt_bl`` drops the continuous-parameter moves
  (:class:`ChangeNodeHeight`, :class:`ChangeInheritanceProb`) during the
  topology search; branch lengths and gammas are optimised once at the end via
  :func:`phynetpy._optimize.optimize_network_parameters`.
* ``fix_st`` drops backbone-changing moves (currently just :class:`SPR`) so
  only reticulation add/remove/relocate/endpoint and gamma moves are proposed,
  leaving the starting-tree backbone fixed.  ``SwitchParentage`` is
  deliberately *not* part of ``BACKBONE_MOVES``: it belongs to the
  allopolyploid (``INFER_MP_ALLOP``) search, which is out of scope for these
  flags.
* ``max_lvl`` enforcement is authoritative in the accept path via
  :func:`make_level_validator`.  The per-move bake-ins (see
  :mod:`phynetpy.ModelMove`) are only an efficiency layer that lets a
  level-raising move self-reject before it is scored.

Docs   - [x]
Tests  - [ ]
Design - [x]
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

from .ModelMove import (
    Move,
    SPR,
    ChangeNodeHeight,
    ChangeInheritanceProb,
    AddReticulation,
    RemoveReticulation,
    FlipReticulation,
    ChangeReticSource,
    ChangeReticDest,
    RelocateReticulation,
)
from .ModelGraph import Model
from .State import network_invariants_routine
from .GraphUtils import level as network_level


__all__ = [
    "DEFAULT_MOVE_TYPES",
    "CONTINUOUS_MOVES",
    "BACKBONE_MOVES",
    "LEVEL_RAISING_MOVES",
    "resolve_move_types",
    "make_level_validator",
    "SearchSettings",
    "SEARCH_PRESETS",
    "resolve_search_preset",
]


# ======================================================================
# Search presets -- one-word intent bundles shared by every method
# ======================================================================
# The topology-search methods (MPL, MCMC_GT, InferNetwork_ML) expose a
# family of *mechanism* flags (``optimize_params`` / ``opt_bl`` /
# ``fix_st`` / ``optimize_scope`` / ``final_optimize``).  Asking users to
# remember which combination yields "accurate" or "PhyloNet-compatible"
# behaviour is a usability trap, so each method also accepts a single
# ``preset=`` string that expands to a coherent bundle here.  Any flag the
# caller passes explicitly overrides the preset (see
# :func:`resolve_search_preset`), and each method consumes only the fields
# it supports (e.g. MCMC_GT, which has no per-topology optimisation, reads
# ``opt_bl`` / ``fix_st`` and ignores ``optimize_params``).


@dataclass(frozen=True)
class SearchSettings:
    """Resolved, method-agnostic search behaviour.

    Attributes:
        optimize_params: Per-topology continuous-parameter optimisation
            during the climb (judging each topology near its optimum).
        optimize_scope: Which parameters that per-topology optimisation
            touches -- ``"gamma"`` (cheapest, fixes the r>=1 bias),
            ``"reticulation"`` (gammas + incident branches), or ``"all"``.
        opt_bl: Drop the continuous-parameter moves during the climb and
            optimise branch lengths + gammas once at the end (PhyloNet's
            ``opt_bl``).
        fix_st: Fix the starting-tree backbone (drop ``SPR``).
        final_optimize: Optimise the overall best network's continuous
            parameters once at the end of the search.
    """

    optimize_params: bool
    optimize_scope: str
    opt_bl: bool
    fix_st: bool
    final_optimize: bool


# The four shipped presets.  ``"default"`` is the recommended balance and
# is what every method uses when ``preset`` is left unset: it turns on the
# near-free gamma optimisation so r>=1 inference is accurate out of the box.
SEARCH_PRESETS: dict[str, SearchSettings] = {
    # Recommended: accurate r>=1 results at ~baseline speed.
    "default": SearchSettings(
        optimize_params=True, optimize_scope="gamma",
        opt_bl=False, fix_st=False, final_optimize=True,
    ),
    # Quickest: raw hill-climb, no per-topology optimisation.
    "fast": SearchSettings(
        optimize_params=False, optimize_scope="gamma",
        opt_bl=False, fix_st=False, final_optimize=True,
    ),
    # Most thorough: per-topology optimisation of gammas + incident
    # branches, continuous moves dropped, full final optimisation.
    "accurate": SearchSettings(
        optimize_params=True, optimize_scope="reticulation",
        opt_bl=True, fix_st=False, final_optimize=True,
    ),
    # Reproduce PhyloNet: optimise *every* continuous parameter per
    # examined topology (its ``-o`` behaviour) for cross-checking.
    "phylonet": SearchSettings(
        optimize_params=True, optimize_scope="all",
        opt_bl=True, fix_st=False, final_optimize=True,
    ),
}


def resolve_search_preset(
    preset: str = "default",
    *,
    optimize_params: Optional[bool] = None,
    optimize_scope: Optional[str] = None,
    opt_bl: Optional[bool] = None,
    fix_st: Optional[bool] = None,
    final_optimize: Optional[bool] = None,
) -> SearchSettings:
    """Expand a ``preset`` into concrete :class:`SearchSettings`.

    Any keyword passed as non-``None`` overrides the corresponding preset
    field, so power users keep full control while newcomers get a coherent
    one-word configuration.

    Args:
        preset: One of :data:`SEARCH_PRESETS` (``"default"``, ``"fast"``,
            ``"accurate"``, ``"phylonet"``).
        optimize_params, optimize_scope, opt_bl, fix_st, final_optimize:
            Explicit overrides; ``None`` (the default) defers to the
            preset.

    Returns:
        The resolved :class:`SearchSettings`.

    Raises:
        ValueError: If ``preset`` is not a known preset name.
    """
    if preset not in SEARCH_PRESETS:
        valid = ", ".join(repr(k) for k in SEARCH_PRESETS)
        raise ValueError(
            f"Unknown search preset {preset!r}; choose one of {valid}."
        )
    base = SEARCH_PRESETS[preset]
    overrides = {
        "optimize_params": optimize_params,
        "optimize_scope": optimize_scope,
        "opt_bl": opt_bl,
        "fix_st": fix_st,
        "final_optimize": final_optimize,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **overrides)


# Default proposal-kernel move set, matching ``MPLKernel`` /
# ``MCMCGTKernel`` defaults.
DEFAULT_MOVE_TYPES: tuple[type[Move], ...] = (
    SPR,
    ChangeNodeHeight,
    ChangeInheritanceProb,
    AddReticulation,
    RemoveReticulation,
    FlipReticulation,
    ChangeReticSource,
    ChangeReticDest,
    RelocateReticulation,
)

# Continuous-parameter moves dropped by ``opt_bl``.
CONTINUOUS_MOVES: tuple[type[Move], ...] = (
    ChangeNodeHeight,
    ChangeInheritanceProb,
)

# Backbone-changing moves dropped by ``fix_st``.  ``SwitchParentage`` is
# intentionally excluded (it is MP-Allop's only move; these flags do not
# apply to that method).
BACKBONE_MOVES: tuple[type[Move], ...] = (
    SPR,
)

# Moves that can *raise* a network's level (move a reticulation into a blob
# that already contains another).  The kernel passes ``max_level`` to these so
# they can self-reject early.  ``RemoveReticulation`` only lowers level, so it
# is intentionally absent.
LEVEL_RAISING_MOVES: tuple[type[Move], ...] = (
    AddReticulation,
    RelocateReticulation,
    ChangeReticSource,
    ChangeReticDest,
    FlipReticulation,
)


def resolve_move_types(
    opt_bl: bool = False,
    fix_st: bool = False,
    base: Optional[list[type[Move]]] = None,
) -> list[type[Move]]:
    """Resolve the kernel's move set from the ``opt_bl`` / ``fix_st`` flags.

    Args:
        opt_bl: When ``True`` drop the continuous-parameter moves
            (:data:`CONTINUOUS_MOVES`) -- branch lengths and gammas are
            optimised at the end of the search instead of sampled during it.
        fix_st: When ``True`` drop the backbone-changing moves
            (:data:`BACKBONE_MOVES`) so the starting-tree backbone is fixed
            and only reticulation/gamma moves are proposed.
        base: Optional base move set to filter.  Defaults to
            :data:`DEFAULT_MOVE_TYPES`.

    Returns:
        The filtered list of move classes (a new list; never mutates
        ``base``).
    """
    moves = list(base) if base is not None else list(DEFAULT_MOVE_TYPES)
    if opt_bl:
        moves = [m for m in moves if m not in CONTINUOUS_MOVES]
    if fix_st:
        moves = [m for m in moves if m not in BACKBONE_MOVES]
    return moves


def make_level_validator(
    max_lvl: Optional[int],
    base: Callable[[Model], bool] = network_invariants_routine,
) -> Callable[[Model], bool]:
    """Compose ``base`` with a ``level(net) <= max_lvl`` guard.

    This is the authoritative ``max_lvl`` enforcement: because it runs on the
    *resulting* network in the accept path, it catches every move that can
    raise the level -- including reticulation relocation / endpoint moves that
    keep the reticulation count fixed but slide a hybrid into an already
    occupied blob.

    Args:
        max_lvl: Maximum allowed network level (e.g. ``1`` for level-1).
            ``None`` disables the guard and returns ``base`` unchanged.
        base: Underlying validator to compose with.  Defaults to
            :func:`phynetpy.State.network_invariants_routine`.

    Returns:
        A ``Callable[[Model], bool]`` suitable for :class:`phynetpy.State.State`
        (returns ``True`` iff the model is valid *and* within the level cap).
    """
    if max_lvl is None:
        return base

    def _validator(model: Model) -> bool:
        if not base(model):
            return False
        net = model.network
        if net is None:
            return False
        return network_level(net) <= max_lvl

    return _validator
