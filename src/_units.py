#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Branch-length units and heterogeneous MSC population-rate helpers."""

from __future__ import annotations

from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from .Network import Network

BranchThetaKey = str | tuple[str, str]


class BranchLengthUnit(str, Enum):
    """Units carried by every biologically interpreted network.

    ``SUBSTITUTIONS_PER_SITE`` is required by sequence, marker, and timed-MSC
    calculations. ``COALESCENT_2N`` supports topology/coalescent workflows.
    ``UNSPECIFIED`` is safe for graph operations but rejected by biological
    calculations until the network is explicitly tagged or converted.
    """

    UNSPECIFIED = "unspecified"
    SUBSTITUTIONS_PER_SITE = "substitutions_per_site"
    COALESCENT_2N = "coalescent_2n"


def coerce_branch_length_unit(value: Any) -> BranchLengthUnit:
    """Return ``value`` as a :class:`BranchLengthUnit`."""

    if value is None:
        return BranchLengthUnit.UNSPECIFIED
    if isinstance(value, BranchLengthUnit):
        return value
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "unspecified": BranchLengthUnit.UNSPECIFIED,
        "substitution": BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        "substitutions": BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        "substitutions_per_site": BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        "expected_substitutions_per_site": BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        "coalescent": BranchLengthUnit.COALESCENT_2N,
        "coalescent_2n": BranchLengthUnit.COALESCENT_2N,
        "2n": BranchLengthUnit.COALESCENT_2N,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        known = sorted(aliases)
        raise ValueError(
            f"unknown branch-length unit {value!r}; expected one of {known}."
        ) from exc


def _positive_theta(value: Any, name: str) -> float:
    """Return a finite positive theta or raise a clear error."""

    try:
        theta = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; got {value!r}.") from exc
    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {theta}.")
    return theta


def validate_branch_thetas(
    branch_thetas: Optional[Mapping[BranchThetaKey, float]],
) -> None:
    """Validate fixed per-population ``theta = 4*N*mu`` overrides."""

    for key, value in (branch_thetas or {}).items():
        if not (
            isinstance(key, str)
            or (
                isinstance(key, tuple)
                and len(key) == 2
                and all(isinstance(label, str) for label in key)
            )
        ):
            raise ValueError(
                "branch theta keys must be child labels or "
                f"(parent_label, child_label) pairs; got {key!r}."
            )
        _positive_theta(value, f"branch theta for {key!r}")


def resolve_branch_theta(
    default_theta: float,
    branch_thetas: Optional[Mapping[BranchThetaKey, float]],
    child: Any,
    *,
    parent: Optional[Any] = None,
) -> float:
    """Resolve theta for the population branch above ``child``.

    Stable ``(parent_label, child_label)`` edge keys take precedence over
    child-label keys. These label-based keys remain valid across network copies.
    For the infinite population above the root, use ``("__root__", label)`` or
    ``"__root__"``; the root label is the final fallback.
    """

    theta = _positive_theta(default_theta, "default theta")
    if not branch_thetas:
        return theta

    child_label = child.label if hasattr(child, "label") else str(child)
    if parent is not None:
        parent_label = parent.label if hasattr(parent, "label") else str(parent)
        candidates: tuple[BranchThetaKey, ...] = (
            (parent_label, child_label),
            child_label,
        )
    else:
        candidates = (("__root__", child_label), "__root__", child_label)

    for key in candidates:
        if key in branch_thetas:
            return _positive_theta(
                branch_thetas[key], f"branch theta for {key!r}"
            )
    return theta


def require_branch_length_unit(
    network: "Network",
    expected: BranchLengthUnit,
    *,
    context: str,
) -> None:
    """Reject untagged networks or units incompatible with ``context``."""

    actual = network.get_branch_length_unit()
    if actual is BranchLengthUnit.UNSPECIFIED:
        raise ValueError(
            f"{context} requires explicit branch-length units; tag the network "
            f"with BranchLengthUnit.{expected.name} or convert it first."
        )
    if actual is not expected:
        raise ValueError(
            f"{context} requires {expected.value} branch lengths; got "
            f"{actual.value}. Use convert_network_branch_lengths()."
        )


def convert_network_branch_lengths(
    network: "Network",
    target: BranchLengthUnit | str,
    *,
    theta: float,
    branch_thetas: Optional[Mapping[BranchThetaKey, float]] = None,
) -> "Network":
    """Return a structural copy with branch lengths converted explicitly.

    Conversion follows ``t_coal = 2*t_sub/theta_branch``.  The source unit is
    read from ``network`` and must already be explicit. Node-time annotations
    are copied unchanged; biological calculations derive timing from edges.
    """

    source = network.get_branch_length_unit()
    target_unit = coerce_branch_length_unit(target)
    if source is BranchLengthUnit.UNSPECIFIED:
        raise ValueError("cannot convert a network with unspecified units.")
    if target_unit is BranchLengthUnit.UNSPECIFIED:
        raise ValueError("conversion target must be an explicit unit.")

    copied, _ = network.copy()
    if source is target_unit:
        return copied

    validate_branch_thetas(branch_thetas)
    _positive_theta(theta, "default theta")
    copied.set_branch_length_unit(target_unit)

    def edge_key(edge: Any) -> tuple[Any, ...]:
        return (
            edge.src.label,
            edge.dest.label,
            edge.get_gamma(),
            edge.get_tag(),
            edge.get_length(),
        )

    copied_edges: dict[tuple[Any, ...], list[Any]] = {}
    for edge in copied.E():
        copied_edges.setdefault(edge_key(edge), []).append(edge)
    for edge in network.E():
        matches = copied_edges.get(edge_key(edge))
        if not matches:  # pragma: no cover - structural-copy invariant
            raise RuntimeError("network copy lost an edge during conversion.")
        copied_edge = matches.pop()
        length = edge.get_length()
        if length is None:
            continue
        branch_theta = resolve_branch_theta(
            theta,
            branch_thetas,
            edge.dest,
            parent=edge.src,
        )
        factor = (
            2.0 / branch_theta
            if source is BranchLengthUnit.SUBSTITUTIONS_PER_SITE
            else branch_theta / 2.0
        )
        copied_edge.set_length(float(length) * factor)
    return copied


__all__ = [
    "BranchLengthUnit",
    "BranchThetaKey",
    "coerce_branch_length_unit",
    "convert_network_branch_lengths",
    "require_branch_length_unit",
    "resolve_branch_theta",
    "validate_branch_thetas",
]
