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
Module that contains classes and functions that assist developers while using
PhyNetPy.

Release Version: 2.0.0

Author: Mark Kessler
"""


import numpy as np
import random
from collections import defaultdict
from PhyNetPy.Network import Network, Node, Edge
from PhyNetPy.GeneTrees import GeneTrees
from PhyNetPy.MetropolisHastings import ProposalKernel, MetropolisHastings
from PhyNetPy.ModelMove import *
from PhyNetPy.BirthDeath import *
from PhyNetPy.ModelFactory import *

def generate_initial_gene_tree(species_tree):
    """Generate an initial gene tree given a species tree.
    This function currently just copies the species tree structure,
    but in a real implementation, it should generate a gene tree sampled from the coalescent model."""
    gene_tree = Network()
    for node in species_tree.V():
        gene_tree.add_nodes(Node(node.label))
    for edge in species_tree.E():
        gene_tree.add_edges(Edge(gene_tree.has_node_named(edge.src.label), gene_tree.has_node_named(edge.dest.label)))
    return gene_tree

def propose_new_gene_tree(current_tree):
    """Propose a new gene tree by modifying the current tree.
    This is done by randomly swapping two nodes in the tree to explore new topologies."""
    new_tree, node_map = current_tree.copy()
    node1, node2 = random.sample(list(new_tree.V()), 2)
    node1.label, node2.label = node2.label, node1.label
    return new_tree

def map_gene_to_species(gene_tree, species_mapping):
    """Map gene tree leaves to their corresponding species.
    This helps track which gene sequences belong to which species."""
    species_to_genes = defaultdict(set)
    for gene, species in species_mapping.items():
        species_to_genes[species].add(gene)
    return species_to_genes

def count_gene_lineages_recursive(species, species_tree, lineage_counts):
    """Recursively count gene lineages at each species tree branch."""
    if species not in species_tree.V():  # Base case: leaf node
        return lineage_counts.get(species, 1)
    
    parent = species_tree.get_parents(species_tree.has_node_named(species))[0]
    child_lineages = count_gene_lineages_recursive(parent.label, species_tree, lineage_counts)
    lineage_counts[species] += child_lineages
    return lineage_counts[species]

def waiting_time(k, Ne):
    """Compute the waiting time for coalescence of k lineages in a population of effective size Ne.
    This follows an exponential distribution with rate lambda_k = k(k-1) / (2Ne)."""
    if k < 2:
        return float('inf')  # No coalescence if fewer than 2 lineages
    lambda_k = (k * (k - 1)) / (2 * Ne)
    return np.random.exponential(1 / lambda_k)  # Sample from exponential distribution

def coalescent_probability_recursive(species, species_tree, lineage_counts, Ne=1e6):
    """Recursively compute the likelihood of a gene tree given a species tree using the coalescent model."""
    if species not in species_tree.V():
        return 1.0  # Base case: leaf node, probability is neutral
    
    parent = species_tree.get_parents(species_tree.has_node_named(species))[0]
    t = species_tree.get_edge(species_tree.has_node_named(species), parent).get_length()
    k = lineage_counts.get(species, 1)
    
    if k > 1:
        tau = waiting_time(k, Ne)
        if tau > t:
            branch_likelihood = np.exp(-(k * (k - 1) * t) / (2 * Ne))
        else:
            branch_likelihood = (1 - np.exp(-(k * (k - 1) * t) / (2 * Ne)))
        
        return branch_likelihood * coalescent_probability_recursive(parent.label, species_tree, lineage_counts, Ne)
    
    return coalescent_probability_recursive(parent.label, species_tree, lineage_counts, Ne)

def mcmc_gene_tree(species_tree, species_mapping, iterations=10000, burn_in=1000):
    """MCMC sampling for gene trees given a species tree.
    This function implements a Markov Chain Monte Carlo (MCMC) process to explore gene tree topologies.
    It proposes new gene trees, evaluates their likelihoods, and accepts or rejects them based on the acceptance ratio."""
    
    current_tree = generate_initial_gene_tree(species_tree)
    lineage_counts = {species: len(genes) for species, genes in map_gene_to_species(current_tree, species_mapping).items()}
    
    current_likelihood = coalescent_probability_recursive("D", species_tree, lineage_counts)
    sampled_trees = []
    
    for i in range(iterations):
        proposed_tree = propose_new_gene_tree(current_tree)
        proposed_lineage_counts = {species: len(genes) for species, genes in map_gene_to_species(proposed_tree, species_mapping).items()}
        proposed_likelihood = coalescent_probability_recursive("D", species_tree, proposed_lineage_counts)
        
        acceptance_ratio = min(1, proposed_likelihood / current_likelihood)
        
        if random.random() < acceptance_ratio:
            current_tree = proposed_tree
            current_likelihood = proposed_likelihood
        
        if i >= burn_in:
            sampled_trees.append(current_tree)
    
    return sampled_trees

#################
## Mark's Code ##
#################

class GTPartials:
    """
    Class that stores partial likelihoods.
    
    For each node in a network, this stores the partial likelihood for each 
    individual gene tree
    """
    
    pass

class GTKernel(ProposalKernel):
    
    def __init__(self) -> None:
        """
        Initialize the MCMC proposal kernel for gene trees.
        
        Args:
            N/A
        Returns:
            N/A
        """
        pass

    def generate(self):
        
        """
        This kernel prioritizes in the following order:
        1. Adjusting the topology of the network within the space of networks 
           that have the same number of reticulations.
        2. Moving the hybrid edges around in the network
        3. Adjusting the number of reticulations in the network.

        Returns:
            Move: A move object that represents the proposed change to the network.
        """
        num = random.random()
        if num < 0.05:
            return AddReticulation()
        elif num < 0.1:
            return RemoveReticulation()
        elif num < 0.2:
            return RelocateReticulation()
        else:
            return SPR()


def generate_starting_network(taxa_labels : list[str]) -> Network:
    """
    Generate a starting network for MCMC sampling.

    Args:
        taxa_labels (list[str]): A list of taxa labels.

    Returns:
        Network: A starting network.
    """
    pass

    
def MCMC_GT(gene_trees : GeneTrees,
            taxon_genemap : dict[str, list[str]],
            iter_count : int = 1000) -> dict[Network, float]:
    """
    Perform MCMC sampling of gene trees.
    
    Args:
        gene_trees (GeneTrees): A GeneTrees object containing gene trees.
        taxon_genemap (dict[str, list[str]]): A mapping of taxa names to 
                                              genes/gene copies.
        iter_count (int, optional): The number of iterations to run. 
                                    Defaults to 1000.
    Returns:
        dict[Network, float]: A dictionary mapping inferred networks to their 
                              likelihoods.
    """            
    
    start_net : Network = generate_starting_network(taxon_genemap.keys())
    kernel = GTKernel()
    components = []
    components.append(NetworkComponent({}, start_net))
    components.append(ParameterComponent({NetworkComponent}, "Effective Population Size", 1e6), [ANetworkNode])
    components.append(AccumulatorComponent({NetworkComponent}, "Partial Likelihoods", GTPartials()), [ANetworkNode])
    
    model : Model = ModelFactory().build()
    mcmc = MetropolisHastings(start_net, kernel)
    
    

       
