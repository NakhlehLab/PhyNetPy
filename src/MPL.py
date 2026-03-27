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
Last Stable Edit : 3/24/26
First Included in Version : 0.3.2

Docs   - [ ]
Tests  - [ ]
Design - [ ]

This file implements maximum pseudolikelihood calculations using the Strategy/Visitor/Executor model.
"""


import warnings

from .GraphUtils import subnet_given_leaves
from .Strategy import Strategy
from .Visitor import Visitor
from .GeneTrees import GeneTrees
from .ModelGraph import *
from . import IO as io
from .Network import *


# ── Triple topology helper ─────────────────────────────────────────────

def _induced_triple(tree: Network, x: str, y: str, z: str) -> str:
    """
    Determine the topology of the triple induced by allele leaves
    *x*, *y*, *z* in a gene tree.

    Args:
        tree (Network): A gene tree .
        x (str): Leaf label present in the tree.
        y (str): Leaf label present in the tree.
        z (str): Leaf label present in the tree.

    Returns:
        str: One of ``"xy|z"``, ``"xz|y"``, ``"yz|x"``, or ``"star"``.
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


# ── Rho computation (Eq. 1 of Yu & Nakhleh 2015) ──────────────────────

def _compute_rho_for_triplet(species_X: str,
                             species_Y: str,
                             species_Z: str,
                             gene_trees: GeneTrees,
                             species_gene_mapping: dict[str, list[str]]) -> tuple[float, float, float]:
    """
    Compute rho(XY|Z, G), rho(XZ|Y, G), rho(YZ|X, G) for species
    triplet {X, Y, Z} across all gene trees G.

    Handles multi-allele sampling per species and non-binary (star)
    triples in gene trees (each star contributes 1/3 to every resolution).

    The returned values are *accumulated* counts (summed across gene
    trees), not normalised frequencies — the normalisation per gene tree
    is already applied via the denominator in Eq. 1.

    Args:
        species_X, species_Y, species_Z: Species leaf labels (sorted).
        gene_trees: The GeneTrees collection.
        species_gene_mapping: Species name -> list of allele labels.

    Returns:
        ``(rho_xy_z, rho_xz_y, rho_yz_x)``
    """
    all_alleles_X = species_gene_mapping.get(species_X, [])
    all_alleles_Y = species_gene_mapping.get(species_Y, [])
    all_alleles_Z = species_gene_mapping.get(species_Z, [])

    rho_xy_z = 0.0
    rho_xz_y = 0.0
    rho_yz_x = 0.0

    for tree in gene_trees.trees:
        tree_leaves = {leaf.label for leaf in tree.get_leaves()}

        ax = [a for a in all_alleles_X if a in tree_leaves]
        ay = [a for a in all_alleles_Y if a in tree_leaves]
        az = [a for a in all_alleles_Z if a in tree_leaves]

        denom = len(ax) * len(ay) * len(az)
        if denom == 0:
            continue

        cnt_xy = 0.0
        cnt_xz = 0.0
        cnt_yz = 0.0

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
                    else:
                        cnt_xy += 1.0 / 3.0
                        cnt_xz += 1.0 / 3.0
                        cnt_yz += 1.0 / 3.0

        rho_xy_z += cnt_xy / denom
        rho_xz_y += cnt_xz / denom
        rho_yz_x += cnt_yz / denom

    return (rho_xy_z, rho_xz_y, rho_yz_x)


## Restricted Subnetwork Visitor/Strategy
class MPLVisitor(Visitor):

    def __init__(self, net : Network) -> None:
        self.triplet_cache = None

    def visit_leaf(self, n : LeafNode) -> None:
        """

        """
        pass

class MPLStrategy(Strategy):
    pass


class MPLOrchestrator:

    def __init__(self, 
                 species_net_filename : str, 
                 gene_trees_filename : str, 
                 gene_mapping : dict[str, list[str]]) -> None:
        
        self.gene_trees : GeneTrees = io.read_nexus(gene_trees_filename, return_type="genetrees")
        self.gene_trees.species_gene_mapping(gene_mapping)

        species_networks = io.read_nexus(species_net_filename)
        if len(species_networks) > 1:
            warnings.warn("You have provided more than one species network to analyze. We will use the first one given")
        elif len(species_networks) == 0:
            raise Exception("No species networks found in input file")
        
        #Species Network
        self.S : Network = species_networks[0]
        

    def _enumerate_triplets(self)-> set[list[Node]]:
        """
        Get all the triplets for the species network.

        IE: for n=10 leaves, there are C(10,3) sets of triplets.
        """
        pass

    def _restricted_subnets(self) -> list[Network]:
        """
        Get the restricted_subnetwork for each of the triplets.

        Args:
            N/A
        Returns:
            list[Network]: subnetwork copies based on the triplets.
        """
        return [subnet_given_leaves(self.S, triplet) for triplet in self._enumerate_triplets()]
        
        
    def _map_gene_trees(self)-> dict[Network, Network]:
        """
        Map each gene tree to the triplet whose topology is contained in the gene tree.

        Args:
            N/A
        Returns:
            dict[Network, Network]: Mapping from gene trees to their corresponding triplets. Values will not be unique.
        """
        pass

    def _score(self, restricted_subnetwork : Network):
        """
        Calculate the log likelihood of one restricted subnetwork given the gene trees.


        """
        pass

    def likelihood(self) -> float:
        """
        Accumulate the log probabilities for each restricted subnetwork and return the sum/final Pseudo Likelihood
        
        Args:
            N/A
        Returns:
            float: The pseudo likelihood of the species network given the gene trees.
        """
        for subnet in self._restricted_subnets():
            self._score(subnet)
