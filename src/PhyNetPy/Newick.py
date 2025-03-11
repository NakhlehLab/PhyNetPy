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
Last Stable Edit : 3/11/25
First Included in Version : 1.0.0
Approved for Release Date : Yes.
"""

import os
from pathlib import Path
from typing import Union

#########################
#### EXCEPTION CLASS ####
#########################

class NewickParserError(Exception):
    """
    Error class for any exceptions relating to failing to parse a newick string
    into a Network/Graph object
    """
    def __init__(self, message : str = "Error parsing a newick string") -> None:
        """
        Initialize a new error message

        Args:
            message (str, optional): The error message. Defaults to "Error
                                     parsing a newick string".
        Returns:
            N/A
        """
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
                label_set.add(cur_label.split(":", maxsplit = 1)[0].split("[")[0].strip())
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
        Returns:
            N/A
        """
        self.filename = filename
        
    
    def parse_networks(self, 
                       new_filename : str, 
                       file_loc : str = None,
                       phylonet_cmd : str = None) -> None:
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
            phylonet_cmd (str, optional): A phylonet command to be added to the
                                          nexus file. Defaults to None.
        Returns:
            N/A
        """
        if file_loc is None:
            file_location = Path(self.filename).parent
        else:
            file_location = Path(file_loc)
            
        
        nf = NexusTemplate()
        
        with open(self.filename, "r", encoding = "utf8") as file:
            # Get a list of newick strings, assuming the newick strings
            # are separated by newline chars
            strings : list[str] = [line for line in file.readlines() \
                                   if line != "\n"]
            
        #place each newick string into the nexus template object
        for newick in strings:
            nf.add(newick)
        
        if phylonet_cmd is not None:
            nf.add_phylonet_cmd(phylonet_cmd)
        
        #Generate a nexus file with a certain filename and folder destination
        nf.generate(file_location, new_filename)

class NexusTemplate:
    """
    Class that generates a nexus file given tree and biological data.
    """
    
    def __init__(self) -> None:
        """
        Initialize a blank nexus template
        
        Args:
            N/A
        Returns:
            N/A
        """
        # A list of strings that represent one singular line in a "TREE" block
        # ie: "tree t1 = (A, B)C;"
        self.networks : list[str] = list()
        
        # Set of taxa labels that are present across all trees 
        # in the "TREE" block
        self.tax : set[str] = set()
        
        #Counter for tree indeces
        self.net_index : int = 1
        
        self.phylonet_cmds : list[str] = list()
    
    def add(self, newick_str : str) -> None:
        """
        Create a new line in the "TREES" block.

        Args:
            newick_str (str): The next network to be added, in newick format.
        Returns:
            N/A
        """
        labels : set[str] = get_labels(newick_str)
        
        # Make a new line for the "TREE" block
        self.networks.append(f"Tree net{self.net_index} = {newick_str}\n")
        
        # Increment counter
        self.net_index += 1
        
        # Add all labels in this tree to the set of all taxa labels
        self.tax = self.tax.union(labels)
        
    def add_phylonet_cmd(self, cmd : str) -> None:
        """
        Add a phylonet command to the nexus file.

        Args:
            cmd (str): A phylonet command.
        Returns:
            N/A
        """
        self.phylonet_cmds.append(cmd)
        
    def generate(self, loc : Path | str, end_name : str) -> None:
        """
        Create a nexus file at "<loc>/<end_name>", end_name should include .nex
        extension.

        Raises:
            NewickParserError: If the file location already exists or cannot be
                               found.
        Args:
            loc (Path | str): Directory location to save the file to. Can either 
                              be a python Path obj or simply a string.
            end_name (str): The new file name. Must include .nex extension.
        Returns:    
            N/A
        """
        
        start_block= ["#NEXUS\n", "\n", "BEGIN TAXA;\n", 
                      f"DIMENSIONS NTAX={len(self.tax)};\n", 
                      "TAXALABELS\n"]
        
        middle_block = [";\n", "END;\n", "BEGIN TREES;\n"]
        
        phylonet_block = ["BEGIN PHYLONET; \n", "END;\n"]
    
        end = "END;\n"
        
        #CHECK VALIDITY OF ENDNAME
        if type(loc) is Path:
            new_file_path = loc.absolute() / end_name 
        else:
            new_file_path = Path(loc).absolute() / end_name
            
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
                
                for net in self.networks:
                    fp.write(net)
                
                fp.write(end)
                
                if self.phylonet_cmds != []:
                    fp.write(phylonet_block[0])
                    for cmd in self.phylonet_cmds:
                        fp.write(cmd + "\n")
                    fp.write(phylonet_block[1])
                
                fp.write(end)
                fp.close()
        


# test_path = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/Data/DEFJ/100Genes/withOG/J/g100/n1/t20/r1/J2GTg100n1t20r1-g_trees.newick"
# NewickParser(test_path).parse_networks("J_100.nex", "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/")


