#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##############################################################################

"""
Gene-tree estimates on the data axis.

:class:`GeneTrees` is the data-axis type for gene-tree estimates.  It
extends the general-purpose container in :mod:`phynetpy.GeneTrees` (whose
clustering, consensus, and distance utilities it inherits wholesale) with
the two things the two-verb API needs: a ``has_branch_lengths`` flag so the
registry can honour a criterion's branch-length policy, and ``from_file``
constructors so a run can be written in one expression.

PhyNetPy does not estimate gene trees from alignments.  Summarise an
alignment into gene trees with an external tool (RAxML, IQ-TREE, MrBayes,
ASTRAL...) and supply the result here.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ..GeneTrees import GeneTrees as _GeneTreeCollection
from ..Network import Network
from ._base import Data, DataError

# A Newick branch length is a colon followed by a (possibly signed,
# possibly exponential) number.  Used to decide whether a file carried
# branch lengths *before* the reader back-fills the missing ones with 1.0.
_BRANCH_LENGTH_TOKEN = re.compile(r":\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _has_length_token(text: str) -> bool:
    """Whether Newick *text* carries a branch length.

    Comments are stripped first, since they may legitimately contain ``":"``
    (e.g. the inheritance-probability annotations ``"[&gamma=0.3]"``).
    """
    return bool(_BRANCH_LENGTH_TOKEN.search(re.sub(r"\[[^\]]*\]", "", text)))


def _text_has_branch_lengths(path: Union[str, os.PathLike]) -> bool:
    """Detect branch lengths in a Newick / NEXUS file by inspecting its text.

    This has to happen at parse time.  ``phynetpy.IO`` deliberately
    back-fills a missing branch length with ``1`` (emitting a warning) so
    that downstream code always finds a length on every edge -- which means
    a parsed :class:`~phynetpy.Network.Network` can no longer tell you
    whether the lengths were *observed* or *invented*.

    Args:
        path: Path to the gene-tree file.

    Returns:
        bool: ``True`` when at least one branch-length token is present.
    """
    try:
        with open(path, "r", encoding="utf8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        raise DataError(f"could not read gene-tree file {path!r}: {exc}") from exc

    return _has_length_token(text)


def _networks_have_branch_lengths(trees: Sequence[Network]) -> bool:
    """Whether every edge of every tree carries a length."""
    saw_edge = False
    for tree in trees:
        for edge in tree.E():
            saw_edge = True
            try:
                if edge.get_length() is None:
                    return False
            except Exception:
                return False
    return saw_edge


class GeneTrees(Data, _GeneTreeCollection):
    """Gene-tree estimates: the data axis type for gene trees.

    Accepted by the MDC, likelihood, pseudo-likelihood, and Bayesian
    criteria -- gene trees are the only data type all four are defined on.

    Because this subclasses :class:`phynetpy.GeneTrees.GeneTrees`, every
    inherited utility (``build_majority_rule_consensus_tree``,
    ``most_frequent_gene_tree``, ``cluster_support``, ``rf_distance``,
    ``astral``, ``mp_allop_map``, ...) is available, and the object can be
    handed straight to the numerical engines, which expect that type.

    Attributes:
        trees: Set of gene trees, each a :class:`~phynetpy.Network.Network`.
        taxa_names: Union of all gene-copy (allele) labels.
    """

    def __init__(
        self,
        trees: Optional[List[Network]] = None,
        mapping: Optional[Dict[str, List[str]]] = None,
        *,
        naming_rule: Optional[Callable[..., Any]] = None,
        has_branch_lengths: Optional[bool] = None,
    ) -> None:
        """Wrap a set of gene trees as a data-axis object.

        Args:
            trees: The gene trees.  Each should be a rooted binary
                :class:`~phynetpy.Network.Network` with no reticulations.
            mapping: Species -> list of allele labels.  Omit for
                single-copy data, where each label is its own species.
            naming_rule: ``f : str -> str`` deriving a species key from an
                allele label; an alternative to an explicit *mapping*
                (e.g. :func:`phynetpy.GeneTrees.phynetpy_naming`).
            has_branch_lengths: Override the branch-length flag.  ``None``
                infers it from the trees' edges.

        Raises:
            DataError: If *trees* is empty or contains a reticulate network.
        """
        tree_list = list(trees) if trees is not None else []

        # Validate before delegating: the inherited constructor indexes leaf
        # labels as it goes, so a bad input fails there with an obscure
        # AttributeError rather than a message naming the actual problem.
        if not tree_list:
            raise DataError(
                "GeneTrees requires at least one gene tree; got an empty "
                "collection."
            )
        for tree in tree_list:
            if not isinstance(tree, Network):
                raise DataError(
                    "gene trees must be phynetpy.Network.Network objects; got "
                    f"{type(tree).__name__}. Use GeneTrees.from_file() or "
                    "GeneTrees.from_newick() to parse them."
                )
            if any(tree.in_degree(node) > 1 for node in tree.V()):
                raise DataError(
                    "gene trees must be trees, but one input network carries "
                    "a reticulation. Pass a species network to score() or to "
                    "infer(start=...) instead."
                )

        Data.__init__(self, mapping=mapping)
        _GeneTreeCollection.__init__(
            self,
            gene_tree_list=tree_list,
            naming_rule=naming_rule,
            species_gene_mapping=mapping,
        )

        self._has_branch_lengths = (
            has_branch_lengths
            if has_branch_lengths is not None
            else _networks_have_branch_lengths(tree_list)
        )

    # ── Data axis interface ───────────────────────────────────────────

    @property
    def taxa(self) -> set:
        """The set of gene-copy (allele) labels across all trees."""
        return set(self.taxa_names)

    @property
    def mapping(self) -> Optional[Dict[str, List[str]]]:
        """Explicit species -> allele mapping, or ``None`` if unset.

        Kept in step with the inherited ``species_gene_mapping`` so the
        numerical engines and the two verbs read the same value.
        """
        return self._species_gene_mapping

    @mapping.setter
    def mapping(self, value: Optional[Dict[str, List[str]]]) -> None:
        """Set the explicit species -> allele mapping, keeping it in step with ``species_gene_mapping``."""
        self._mapping = value
        self._species_gene_mapping = value

    def resolved_mapping(self) -> Dict[str, List[str]]:
        """Species -> allele mapping, resolved through the naming rule.

        Defers to the inherited resolution chain: an explicit mapping, then
        the naming rule, then identity.
        """
        return self._resolve_mapping()

    @property
    def has_branch_lengths(self) -> bool:
        """Whether these gene trees carry observed branch lengths.

        ``False`` means the trees are topologies only.  Set at construction
        (from the source file's text, or from the trees' edges) rather than
        inferred later, because ``phynetpy.IO`` back-fills missing lengths
        with ``1``.
        """
        return self._has_branch_lengths

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def from_file(
        cls,
        path: Union[str, os.PathLike],
        mapping: Optional[Dict[str, List[str]]] = None,
        *,
        naming_rule: Optional[Callable[..., Any]] = None,
        format: Optional[str] = None,
    ) -> "GeneTrees":
        """Load gene trees from a NEXUS or Newick file.

        Args:
            path: Path to the file.
            mapping: Species -> list of allele labels.
            naming_rule: ``f : str -> str`` deriving species from labels.
            format: ``"nexus"`` or ``"newick"``.  ``None`` infers from the
                file extension, defaulting to NEXUS.

        Returns:
            GeneTrees: The loaded collection, with ``has_branch_lengths``
            determined from the file's text.

        Raises:
            DataError: If the file cannot be read or holds no trees.
        """
        from .. import IO as io

        path_str = str(path)
        if format is None:
            suffix = os.path.splitext(path_str)[1].lower()
            format = "newick" if suffix in (".nwk", ".newick", ".tre", ".tree") else "nexus"

        fmt = format.lower()
        if fmt == "newick":
            networks = io.read_newick_file(path_str, return_type="networks")
        elif fmt == "nexus":
            networks = io.read_nexus(path_str, return_type="networks")
        else:
            raise DataError(
                f"unknown gene-tree format {format!r}; expected 'nexus' or "
                "'newick'."
            )

        trees = list(networks) if isinstance(networks, (list, tuple)) else [networks]
        if not trees:
            raise DataError(f"no gene trees found in {path_str!r}.")

        return cls(
            trees,
            mapping,
            naming_rule=naming_rule,
            has_branch_lengths=_text_has_branch_lengths(path_str),
        )

    @classmethod
    def from_newick(
        cls,
        strings: Union[str, Sequence[str]],
        mapping: Optional[Dict[str, List[str]]] = None,
        *,
        naming_rule: Optional[Callable[..., Any]] = None,
    ) -> "GeneTrees":
        """Build from one or more Newick strings.

        Args:
            strings: A single Newick string, or a sequence of them.  A
                single string containing several ``;``-terminated trees is
                split.
            mapping: Species -> list of allele labels.
            naming_rule: ``f : str -> str`` deriving species from labels.

        Returns:
            GeneTrees: The parsed collection.
        """
        from .. import IO as io

        if isinstance(strings, str):
            parts = [s.strip() for s in strings.split(";") if s.strip()]
        else:
            parts = [str(s).strip().rstrip(";") for s in strings if str(s).strip()]

        if not parts:
            raise DataError("no Newick strings supplied.")

        trees = [io.read_newick(part if part.endswith(";") else part + ";")
                 for part in parts]
        return cls(
            trees,
            mapping,
            naming_rule=naming_rule,
            has_branch_lengths=all(_has_length_token(p) for p in parts),
        )

    def __repr__(self) -> str:
        lengths = "with" if self._has_branch_lengths else "without"
        return (
            f"GeneTrees({len(self.trees)} trees, {len(self.taxa_names)} labels, "
            f"{lengths} branch lengths)"
        )
