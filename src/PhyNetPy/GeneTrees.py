

from typing import Callable
from Graph import DAG
from NetworkParser import NetworkBuilder2 as nb

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
    
    def __init__(self, naming_rule : Callable = phynetpy_naming) -> None:
        self.trees = set()
        self.taxa_names = set()
        self.naming_rule = naming_rule
        
    def add(self, tree : DAG):
        self.trees.add(tree)
        for leaf in tree.get_leaves():
            self.taxa_names.add(leaf.get_name())
        
    def mp_allop_map(self) -> dict[str, list[str]]:
        subgenome_map = {}
        if self.naming_rule is not None and len(self.taxa_names) != 0:
            for taxa_name in self.taxa_names:
                key = self.naming_rule(taxa_name)
                if key in subgenome_map.keys(): 
                    subgenome_map[key].append(taxa_name)
                else:
                    subgenome_map[key] = [taxa_name]
        return subgenome_map
    
    
            
            
nets = nb("/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/J_nex_n1.nex").get_all_networks()
gt = GeneTrees()
for gene_tree in nets:
    gt.add(gene_tree)

print(gt.mp_allop_map())     


