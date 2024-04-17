""" 
Author : Mark Kessler
Last Edit : 3/28/24
First Included in Version : 1.0.0
Approved for Release: Yes, post test-suite final inspection.
"""

from typing import Callable
from Network import Network

#########################
#### EXCEPTION CLASS #### 
#########################   
    
class GeneTreeError(Exception):
    def __init__(self, message = "Gene Tree Module Error") -> None:
        super().__init__(message = message)
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
    
    def __init__(self, gene_tree_list : list[Network] = None, 
                 naming_rule : Callable = phynetpy_naming) -> None:
        """
        Wrapper class for a set of networks that represent gene trees

        Args:
            gene_tree_list (list[Network], optional): A list of networks, 
                                                      of the binary tree 
                                                      variety. Defaults to None.
            naming_rule (Callable, optional): a function f : str -> str. 
                                              Defaults to phynetpy_naming.
        """
        
        self.trees : set[Network] = set()
        self.taxa_names : set[str]= set()
        self.naming_rule : Callable = naming_rule
        
        if gene_tree_list is not None:
            for tree in gene_tree_list:
                self.add(tree)
        
    def add(self, tree : Network):
        """
        Add a gene tree to the collection. Any new gene labels that belong to
        this tree will also be added to the collection of all 
        gene tree leaf labels.

        Args:
            tree (Network): A network that is a tree, must be binary.
        """

        self.trees.add(tree)
        
        for leaf in tree.get_leaves():
            self.taxa_names.add(leaf.get_name())
        
    def mp_sugar_map(self) -> dict[str, list[str]]:
        """
        Create a subgenome mapping from the stored set of gene trees

        Returns:
            dict[str, list[str]]: subgenome mapping
        """
        subgenome_map = {}
        if self.naming_rule is not None and len(self.taxa_names) != 0:
            for taxa_name in self.taxa_names:
                key = self.naming_rule(taxa_name)
                if key in subgenome_map.keys(): 
                    subgenome_map[key].append(taxa_name)
                else:
                    subgenome_map[key] = [taxa_name]
        return subgenome_map