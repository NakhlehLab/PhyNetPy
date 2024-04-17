""" 
Author : Mark Kessler
Last Stable Edit : 4/9/24
First Included in Version : 1.0.0
Approved for Release Date : Yes, unless design changes are wanted.
"""

import os
from pathlib import Path

#########################
#### EXCEPTION CLASS ####
#########################

class NewickParserError(Exception):
    def __init__(self, message = "Error parsing a newick string") -> None:
        super().__init__(message = message)
        self.message = message

##########################
#### HELPER FUNCTIONS ####
##########################

def get_labels(newick_str : str) -> set[str]:
    """
    Given a newick string, gather a list of all unique taxa labels present in
    the string.

    Args:
        newick_str (str): a newick string.

    Returns:
        set[str]: a set of unique taxa labels
    """
    label_set : set[str] = set()
    
    pos : int = 0
    cur_label = ""
    
    # Search through the string and separate until you have only a taxa label.
    while pos < len(newick_str):
        if newick_str[pos] in {")", "(", ","}:
            if len(cur_label) > 0:
                label_set.add(cur_label.split(":")[0].split("[")[0].strip())
            cur_label = ""
        else:
            cur_label += newick_str[pos]
        
        pos += 1
    
    return label_set
            
###########################################
#### NEWICK PARSER AND NEXUS GENERATOR ####
###########################################

class NewickParser:
    """
    Class that handles the parsing of newick strings
    """
    
    def __init__(self, filename : str) -> None:
        """
        Initialize a parser based on a filename that points to a commonly
        accepted file type that contains newick strings.

        Args:
            filename (str): path to a file containing newick strings
        """
        self.filename = filename
        
    
    def parse_networks(self, new_filename : str, file_loc : str = None) -> None:
        """
        Assuming the newick strings are separated by newline chars, generate a
        nexus file that gives each newick string a label and value
        in the "TREES" block.

        Args:
            new_filename (str): A name for the nexus file that will be created
            file_loc (str, optional): The save folder of the newly created 
                                      nexus file. Defaults to None, in which
                                      case the nexus file will be saved to 
                                      the same folder that the original file
                                      that contained the trees came from.
        """
        if file_loc is None:
            par_folder = Path(self.filename).parent
        else:
            par_folder = Path(file_loc)
            
        nf = NexusTemplate()
        with open(self.filename, "r") as file:
            # Get a list of newick strings, assuming the newick strings
            # are separated by newline chars
            strings : list[str] = [line for line in file.readlines() 
                                   if line != "\n"]
            
        #place each newick string into the nexus template object
        for newick in strings:
            nf.add(newick)
        
        #Generate a nexus file with a certain filename and folder destination
        nf.generate(par_folder, new_filename)

class NexusTemplate:
    """
    Class that generates a nexus file given tree and biological data.
    """
    
    def __init__(self) -> None:
        """
        Initialize a blank nexus template
        """
        # A list of strings that represent one singular line in a "TREE" block
        # ie: "tree t1 = (A, B)C;"
        self.trees : list[str] = list()
        
        # Set of taxa labels that are present across all trees 
        # in the "TREE" block
        self.tax : set[str] = set()
        
        #Counter for tree indeces
        self.tree_index : int = 0
    
    def add(self, newick_str : str) -> None:
        """
        Create a new line in the "TREES" block.

        Args:
            newick_str (str): The next network to be added, in newick format.
        """
        labels : set[str] = get_labels(newick_str)
        
        # Make a new line for the "TREE" block
        self.trees.append(f"Tree t{self.tree_index} = {newick_str}\n")
        
        # Increment counter
        self.tree_index += 1
        
        # Add all labels in this tree to the set of all taxa labels
        self.tax = self.tax.union(labels)
        
    def generate(self, loc : Path, end_name : str) -> None:
        """
        Create a nexus file at "<loc>/<end_name>", end_name should include .nex
        extension.

        Args:
            loc (Path): Folder location to save the file to.
            end_name (str): The new file name. Must include .nex extension.

        Raises:
            NewickParserError: If the file location already exists or cannot be
                               found.
        """
        
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
        

# test_path = "/Users/mak17/Documents/PhyNetPy/src/J_pruned_v3.newick"
# NewickParser(test_path).parse_networks("J_pruned_v3.nex")        

 
        