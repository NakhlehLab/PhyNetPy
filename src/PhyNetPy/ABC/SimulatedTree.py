import random
import elfi
from ete3 import Tree
import random
import numpy
import statistics
import math



class SimulatedTree:
    
    def __init__(self, tree : Tree, seq_dict: dict, lineage_dict: dict, sum_rates: float, cur_lineages: int) -> None:
        self.tree : Tree = tree
        self.cur_lineages = cur_lineages
        self.sum_rates = sum_rates
        self.lineage_dict = lineage_dict
        self.seq_dict = seq_dict
        
        
    def accept(self):
        pass
    
    def reject(self):
        pass
    
    def getNewick(self, t):
        """
        Returns a tree in Newick tree format.
        """
        if (t != None):
            return t.write()
        return ";"

    def outputNewick(self, t, name):
        """
        Writes a tree, 't', in Newick tree format into a file. 'name' specifies the 
        file's name in which the tree is written into. If the tree is empty (i.e. if 
        't' is 'None') no output file is created.
        """
        if (t != None):
            t.write(outfile = name + ".nw")
        else:
            print("Empty tree, no output file created.")
    
    def get_seq(self, seq_num = None):
        """
        Returns a genetic sequence for a cell in the simulated phylogeny (simulated using 'gen_tree'). 
        If 'seq_num' is None (which is the default), the whole 'sequence number : sequence' dictionary
        is returned. Otherwise, 'seq_num' can be initialized to an integer argument for the specific
        sequence corresponding to that sequence number to be returned.
        """
        if(seq_num == None): # 'seq_num' is not initialized so return whole dictionary
            return self.seq_dict
        else:
            key = "SEQUENCE_" + str(seq_num) # find the specific key corresponding to 'seq_num'
            value = self.seq_dict.get(key) # find the sequence in the dictionary corresponding to the key
            if(value == None): # no sequence exists corresponding to 'key'
                print(key, "does not exist.")
                return None
            return value # return this specific sequence
    
    def print_seq(self, seq_num = None):
        """
        Prints a genetic sequence for a cell in the simulated phylogeny (simulated using 'gen_tree'). 
        If 'seq_num' is None (which is the default), the whole 'sequence number : sequence' dictionary
        is printed in FASTA format. Otherwise, 'seq_num' can be initialized to an integer argument 
        for the specific sequence corresponding to that sequence number to be printed in FASTA format
        (i.e. the single 'sequence number : sequence' pair is printed).
        """
        if(seq_num == None): # 'seq_num' is not initialized so print whole dictionary in FASTA format
            for key, value in self.seq_dict.items():
                print('>', key, '\n', value)
        else:
            key = "SEQUENCE_" + str(seq_num) # find the specific key corresponding to 'seq_num'
            value = self.seq_dict.get(key) # find the sequence in the dictionary corresponding to the key
            if(value == None): # no sequence exists corresponding to 'key'
                print(key, "does not exist.")
            else:
                print('>', key, '\n', value) # print this 'key' : 'value' pair in FASTA format
    
    def get_tree(self)-> Tree:
        return self.tree
    
    
        