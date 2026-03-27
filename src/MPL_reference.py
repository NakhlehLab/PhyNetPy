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
Last Stable Edit : 3/26/26
First Included in Version : 0.3.2
 
Docs   - [ ]
Tests  - [ ]
Design - [x]
 
Maximum pseudo-likelihood (MPL) for phylogenetic network inference.
 
Implements the scoring function from:
    Yu, Y. & Nakhleh, L. (2015). "A maximum pseudo-likelihood approach
    for phylogenetic networks." BMC Genomics, 16(S10), S10.
 
    log L(Psi, gamma | G)
        = sum over {X,Y,Z} of
          [ rho(XY|Z) * log P(XY|Z | Psi, gamma)
          + rho(XZ|Y) * log P(XZ|Y | Psi, gamma)
          + rho(YZ|X) * log P(YZ|X | Psi, gamma) ]
"""
 
from __future__ import annotations
 
import math
from itertools import combinations
from typing import TYPE_CHECKING, Optional
 
from .Network import Network, Node, Edge
from .GeneTrees import GeneTrees
from .GraphUtils import subnet_given_leaves, _displayed_trees_with_probs
from . import IO as io
 
if TYPE_CHECKING:
    pass
 
_LOG_FLOOR = math.log(1e-200)
 
 
# ═══════════════════════════════════════════════════════════════════════
# TRIPLE TOPOLOGY HELPERS
# ═══════════════════════════════════════════════════════════════════════
 
def _induced_triple(tree: Network, x: str, y: str, z: str) -> str:
    """Topology of the triple {x, y, z} in a gene tree.
    
    Returns one of ``"xy|z"``, ``"xz|y"``, ``"yz|x"``, ``"star"``.
    """
    mrca_xy = tree.mrca({x, y})
    mrca_xz = tree.mrca({x, z})
    mrca_yz = tree.mrca({y, z})
    mrca_xyz = tree.mrca({x, y, z})
 
    if mrca_xy != mrca_xyz:
        return "xy|z"
    if mrca_xz != mrca_xyz:
        return "xz|y"
    if mrca_yz != mrca_xyz:
        return "yz|x"
    return "star"
 
 
def _coalescent_triple_probs(tree: Network, X: str, Y: str, Z: str) -> tuple[float, float, float]:
    """Coalescent probabilities for 3-taxon displayed tree.
    
    For a resolved tree ``((A,B):tau, C)``::
    
        P(AB|C) = 1 - (2/3) exp(-tau)
        P(AC|B) = P(BC|A) = (1/3) exp(-tau)
    
    Args:
        tree: A 3-taxon tree (no reticulations).
        X, Y, Z: Leaf labels in canonical sorted order.
        
    Returns:
        ``(P(XY|Z), P(XZ|Y), P(YZ|X))``
    """
    root = tree.root()
    children = tree.get_children(root)
    
    # Find which pair of taxa are sisters (share an internal MRCA != root)
    sister_pair: Optional[set[str]] = None
    tau = 0.0
    
    for child in children:
        descs = {n.label for n in tree.leaf_descendants(child)}
        if len(descs) == 2:
            sister_pair = descs
            # Branch length of the edge into this cherry node
            in_edges = list(tree.in_edges(child))
            if in_edges:
                length = in_edges[0].get_length()
                if length is not None and length > 0:
                    tau = length
            break
    
    # Star topology — no resolved cherry
    if sister_pair is None:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    
    exp_neg_tau = math.exp(-tau)
    p_match = 1.0 - (2.0 / 3.0) * exp_neg_tau
    p_mismatch = (1.0 / 3.0) * exp_neg_tau
 
    if sister_pair == {X, Y}:
        return (p_match, p_mismatch, p_mismatch)
    if sister_pair == {X, Z}:
        return (p_mismatch, p_match, p_mismatch)
    if sister_pair == {Y, Z}:
        return (p_mismatch, p_mismatch, p_match)
    
    return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
 
 
# ═══════════════════════════════════════════════════════════════════════
# RHO COMPUTATION (Eq. 1 of Yu & Nakhleh 2015)
# ═══════════════════════════════════════════════════════════════════════
 
def _compute_rho(
    X: str, Y: str, Z: str,
    gene_trees: GeneTrees,
    mapping: dict[str, list[str]],
) -> tuple[float, float, float]:
    """Accumulated triple frequencies across all gene trees.
    
    Handles multi-allele sampling per species. Star topologies in 
    gene trees contribute 1/3 to each resolution.
    
    Args:
        X, Y, Z: Species labels.
        gene_trees: The gene tree collection.
        mapping: Species name -> allele leaf labels.
        
    Returns:
        ``(rho_xy_z, rho_xz_y, rho_yz_x)``
    """
    ax_all = mapping.get(X, [])
    ay_all = mapping.get(Y, [])
    az_all = mapping.get(Z, [])
 
    rho_xy, rho_xz, rho_yz = 0.0, 0.0, 0.0
 
    for tree in gene_trees.trees:
        leaves = {leaf.label for leaf in tree.get_leaves()}
        ax = [a for a in ax_all if a in leaves]
        ay = [a for a in ay_all if a in leaves]
        az = [a for a in az_all if a in leaves]
 
        denom = len(ax) * len(ay) * len(az)
        if denom == 0:
            continue
 
        cnt_xy, cnt_xz, cnt_yz = 0.0, 0.0, 0.0
        for xi in ax:
            for yj in ay:
                for zk in az:
                    topo = _induced_triple(tree, xi, yj, zk)
                    if topo == "xy|z":
                        cnt_xy += 1.0
                    elif topo == "xz|y":
                        cnt_xz += 1.0
                    elif topo == "yz|x":
                        cnt_yz += 1.0
                    else:  # star
                        cnt_xy += 1.0 / 3.0
                        cnt_xz += 1.0 / 3.0
                        cnt_yz += 1.0 / 3.0
 
        rho_xy += cnt_xy / denom
        rho_xz += cnt_xz / denom
        rho_yz += cnt_yz / denom
 
    return (rho_xy, rho_xz, rho_yz)
 
 
# ═══════════════════════════════════════════════════════════════════════
# SUBNETWORK TRIPLE PROBABILITIES
# ═══════════════════════════════════════════════════════════════════════
 
def _subnet_triple_probs(subnet: Network) -> tuple[float, float, float]:
    """P(XY|Z), P(XZ|Y), P(YZ|X) on a restricted 3-taxon subnetwork.
    
    Decomposes the subnetwork into its displayed trees (weighted by 
    inheritance probabilities), then applies the closed-form coalescent 
    formula on each.
    
    Args:
        subnet: Restricted subnetwork with exactly 3 leaves.
        
    Returns:
        ``(p_xy_z, p_xz_y, p_yz_x)``
    """
    leaves = sorted(subnet.get_leaves(), key=lambda n: n.label)
    assert len(leaves) == 3, f"Expected 3 leaves, got {len(leaves)}"
    X, Y, Z = (l.label for l in leaves)
    
    displayed = _displayed_trees_with_probs(subnet)
    
    p_xy, p_xz, p_yz = 0.0, 0.0, 0.0
    for tree, weight in displayed:
        tp = _coalescent_triple_probs(tree, X, Y, Z)
        p_xy += weight * tp[0]
        p_xz += weight * tp[1]
        p_yz += weight * tp[2]
 
    return (p_xy, p_xz, p_yz)
 
 
# ═══════════════════════════════════════════════════════════════════════
# MPL SCORER
# ═══════════════════════════════════════════════════════════════════════
 
class MPL:
    """Maximum pseudo-likelihood scorer for a species network.
    
    Precomputes rho (triple frequencies from gene trees) once at init.
    Call :meth:`score` to evaluate the current species network; only the
    network-side probabilities are recomputed.
    
    Example::
    
        >>> mpl = MPL(species_net, gene_trees, mapping)
        >>> log_pl = mpl.score()
    """
    
    def __init__(
        self,
        species_net: Network,
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
    ) -> None:
        self.net = species_net
        self.gene_trees = gene_trees
        self.mapping = mapping
        
        # Enumerate triplets and precompute rho (constant for fixed gene trees)
        self._triplets = list(combinations(
            sorted(n.label for n in self.net.get_leaves()), 3
        ))
        
        self._rho: dict[frozenset[str], tuple[float, float, float]] = {}
        for t in self._triplets:
            self._rho[frozenset(t)] = _compute_rho(
                t[0], t[1], t[2], self.gene_trees, self.mapping
            )
    
    @classmethod
    def from_nexus(cls, gt_file : str, st_file : str, mapping : dict[str, list[str]]):
        """
        Instantiate instead from nexus file paths.

        Args:
            gt_file (str): Path to the gene tree file.
            st_file (str): Path to the species tree file.
            mapping (dict[str, list[str]]): Mapping of genes to allele labels.
        Returns:
            MPL: A MPL object.
        """
        st : Network = io.read_nexus(st_file, return_type="networks")
        gts : GeneTrees = io.read_nexus(gt_file, return_type="genetrees")
        gts.species_gene_mapping(mapping)
        return cls(st, gts, mapping)

    # ── Scoring ───────────────────────────────────────────────────
    
    def score(self) -> float:
        """Log pseudo-likelihood of the species network given gene trees.
        
        Returns:
            The log pseudo-likelihood (float, typically negative).
        """
        total = 0.0
        for triplet in self._triplets:
            total += self._score_triplet(triplet)
        return total
    
    def _score_triplet(self, triplet: tuple[str, str, str]) -> float:
        """Log-PL contribution of one species triplet."""
        key = frozenset(triplet)
        subnet = self._restricted_subnet(triplet)
        probs = _subnet_triple_probs(subnet)
        rho = self._rho[key]
        
        contribution = 0.0
        for rho_i, p_i in zip(rho, probs):
            if rho_i > 0.0:
                contribution += rho_i * (math.log(p_i) if p_i > 0.0 else _LOG_FLOOR)
        return contribution
    
    def _restricted_subnet(self, triplet: tuple[str, str, str]) -> Network:
        """Build the restricted subnetwork for a species triplet."""
        leaf_nodes = []
        for label in triplet:
            node = self.net.has_node_named(label)
            if node is None:
                raise ValueError(f"Species leaf '{label}' not in network")
            leaf_nodes.append(node)
        return subnet_given_leaves(self.net, leaf_nodes)
