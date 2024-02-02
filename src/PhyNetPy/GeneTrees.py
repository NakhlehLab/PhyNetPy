""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0
Approved to Release Date : N/A
"""

from typing import Callable
from Graph import DAG



def phynetpy_naming(taxa_name : str) -> str:
    if taxa_name[0:2].isnumeric():
        sample_no = int(taxa_name[0:2])
    else:
        raise GeneTreeError("Error Applying PhyNetPy Naming Rule: first 2 digits is not numerical")
    
    if taxa_name[2].isalpha():
        return taxa_name[2].upper()
    else:
        raise GeneTreeError("Error Applying PhyNetPy Naming Rule: 3rd position is not an a-z character")
    
    
class GeneTreeError(Exception):
    def __init__(self, message = "Gene Tree Module Error") -> None:
        super().__init__(message = message)
        self.message = message

class GeneTrees:
    #TODO: Alter to use the GeneTree wrapper class
    
    def __init__(self, gene_tree_list : list[DAG] = None, naming_rule : Callable = phynetpy_naming) -> None:
        """
        Wrapper class for a set of DAGs that represent gene trees

        Args:
            gene_tree_list (list[DAG], optional): A list of DAGs, should be trees (NOT networks). Defaults to None.
            naming_rule (Callable, optional): a function f : str -> str. Defaults to phynetpy_naming.
        """
        
        self.trees : set[DAG] = set()
        self.taxa_names : set[str]= set()
        self.naming_rule : Callable = naming_rule
        
        if gene_tree_list is not None:
            for tree in gene_tree_list:
                self.add(tree)
        
    def add(self, tree : DAG):
        """
        Add a gene tree to the collection. Any new gene labels that are apart of this tree will 
        also be added to the collection of all gene tree leaf labels

        Args:
            tree (DAG): A DAG that is a tree, must not be a network.
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
    



class GeneTree(DAG):
    """
    Wrapper class for a DAG that verifies tree status and contains operators that are specifically designed for gene trees.
    """
    
    def __init__(self, edges=None, nodes=None, weights=None) -> None:
        super().__init__(edges, nodes, weights)
        #TODO: verify tree status somehow
    
    def map_to_species_network(self):
        pass
    