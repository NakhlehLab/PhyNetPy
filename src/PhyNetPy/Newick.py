""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0

"""

import os
from pathlib import Path


def get_labels(newick_str : str):
    
    label_set = set()
    
    pos = 0
    cur_label = ""
    while pos < len(newick_str):
        if newick_str[pos] in set([")", "(", ","]):
            if len(cur_label) > 0:
                label_set.add(cur_label.split(":")[0].split("[")[0].strip())
            cur_label = ""
        else:
            cur_label += newick_str[pos]
        
        pos += 1
    
    
    return label_set
            
    
    


class NewickParserError(Exception):
    def __init__(self, message = "Error Parsing File Containing Newick Networks") -> None:
         super().__init__(message = message)
         self.message = message
    
class NewickParser:
    
    def __init__(self, filename : str) -> None:
        self.filename = filename
        
    
    def parse_networks(self, new_filename, file_loc = None):
        if file_loc is None:
            par_folder = Path(self.filename).parent
        else:
            par_folder = Path(file_loc)
            
        nf = NexusTemplate()
        with open(self.filename, "r") as file:
            strings : list[str] = [line for line in file.readlines() if line != "\n"]
            
        
        for newick in strings:
            nf.add(newick_str=newick)
        
        nf.generate(par_folder, new_filename)



class NexusTemplate:
    
    def __init__(self) -> None:
        self.trees = list()
        self.tax = set()
        self.tree_index = 0
    
    def add(self, newick_str : str):
        print(newick_str)
        
        labels = get_labels(newick_str)
        print(labels)
        
        self.trees.append(f"Tree g{self.tree_index} = {newick_str}\n")
        self.tree_index += 1
        self.tax = self.tax.union(labels)
        
    def generate(self, loc : Path, end_name : str):
        
        start_block= ["#NEXUS\n", "\n", "BEGIN TAXA;\n", 
                      f"DIMENSIONS NTAX={len(self.tax)};\n", 
                      "TAXALABELS\n"]
        
        middle_block = [";\n", "END;\n", "BEGIN TREES;\n"]
        
    
        end = "END;\n"
        
        #CHECK VALIDITY OF ENDNAME
        new_file_path = loc.absolute() / end_name 
        if os.path.exists(new_file_path):
            raise NewickParserError("File already exists in this location")
        else:
            # create a file
            with open(new_file_path, 'w') as fp:
                # uncomment if you want empty file
                for line in start_block:
                    fp.write(line)
                
                for taxa in self.tax:
                    fp.write(f"{taxa}\n")
                    
                for line in middle_block:
                    fp.write(line)
                
                for tree in self.trees:
                    fp.write(tree)
                
                fp.write(end)
        
        
#NewickParser("/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/J_pruned_v2.newick").parse_networks("J_pruned_v2.nex")        

 
        