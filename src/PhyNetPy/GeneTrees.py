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
Last Edit : 3/11/25
First Included in Version : 2.0.0

Docs   - [x]
Tests  - [ ]
Design - [x]
"""

from typing import Any, Callable
from Network import Network

#########################
#### EXCEPTION CLASS #### 
#########################   
    
class GeneTreeError(Exception):
    def __init__(self, message : str = "Gene Tree Module Error") -> None:
        super().__init__(message)
        self.message = message
        
##########################
#### HELPER FUNCTIONS #### 
##########################
  
def phynetpy_naming(taxa_name : str) -> str:
    """
    The default method for sorting taxa labels into groups

    Args:
        taxa_name (str): a taxa label from a nexus file
    Raises:
        GeneTreeError: if there is a problem applying the naming rule

    Returns:
        str: a string that is the key for this label
    """
    if not taxa_name[0:2].isnumeric():
        raise GeneTreeError("Error Applying PhyNetPy Naming Rule: \
                             first 2 digits is not numerical")
    
    if taxa_name[2].isalpha():
        return taxa_name[2].upper()
    else:
        raise GeneTreeError("Error Applying PhyNetPy Naming Rule: \
                             3rd position is not an a-z character")

####################
#### GENE TREES ####
####################

class GeneTrees:
    """
    A container for a set of networks that are binary and represent a 
    gene tree.
    """
    
    def __init__(self, 
                 gene_tree_list : list[Network] | None = None, 
                 naming_rule : Callable[..., Any] = phynetpy_naming) -> None:
        """
        Wrapper class for a set of networks that represent gene trees

        Args:
            gene_tree_list (list[Network], optional): A list of networks, 
                                                      of the binary tree 
                                                      variety. Defaults to None.
            naming_rule (Callable[..., Any], optional): A function 
                                                        f : str -> str. 
                                                        Defaults to 
                                                        phynetpy_naming.
        """
        
        self.trees : set[Network] = set()
        self.taxa_names : set[str] = set()
        self.naming_rule : Callable[..., Any] = naming_rule
        
        if gene_tree_list is not None:
            for tree in gene_tree_list:
                self.add(tree)
        
    def add(self, tree : Network) -> None:
        """
        Add a gene tree to the collection. Any new gene labels that belong to
        this tree will also be added to the collection of all 
        gene tree leaf labels.

        Args:
            tree (Network): A network that is a tree, must be binary.
        """

        self.trees.add(tree)
        
        for leaf in tree.get_leaves():
            self.taxa_names.add(leaf.label)
        
    def mp_allop_map(self) -> dict[str, list[str]]:
        """
        Create a subgenome mapping from the stored set of gene trees

        Returns:
            dict[str, list[str]]: subgenome mapping
        """
        subgenome_map : dict[str, list[str]] = {}
        if len(self.taxa_names) != 0:
            for taxa_name in self.taxa_names:
                key = self.naming_rule(taxa_name)
                if key in subgenome_map.keys(): 
                    subgenome_map[key].append(taxa_name)
                else:
                    subgenome_map[key] = [taxa_name]
        return subgenome_map