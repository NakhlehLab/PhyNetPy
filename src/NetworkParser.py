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
Approved for Release : Most Likely. Further Testing Needed. Fully Documented
"""

import traceback
from networkx.readwrite.json_graph import adjacency
from nexus import NexusReader
import newick as nw
from Bio import Phylo
from io import StringIO
from .Network import Node, Network, Edge
from warnings import warn
from typing import Any
from .Validation import NexusValidator, ValidationSummary


#####################
#### Error Class ####
#####################

class NetworkParserError(Exception):
    """
    Error that is raised whenever an input file or newick string contains
    issues that disallow a proper parse of a network.
    """
    def __init__(self, message : str = "Something went wrong \
                                        parsing a network") -> None:
        """
        Initialize the error with a message.

        Args:
            message (str, optional): Custom error message. Defaults to 
                                     "Something went wrong parsing a network".
        """
        self.message = message
        super().__init__(self.message)

#########################
#### Helper Function ####
#########################

def merge_attributes(inheritances : dict[str],
                     parsed_node : Node, 
                     attr1 : dict, 
                     attr2 : dict) -> dict:
    """
    Given two attribute dictionaries, combine them into one, taking the
    union of the two. 
    
    Args:
        inheritances (dict[str, dict]): a mapping from node names to their 
                                        gamma entries. 
        parsed_node (Node): The node for which to combine attributes parsed for 
                            different parents. Should be a reticulation node.
        attr1 (dict): The attribute dict derived from the first parent
        attr2 (dict): The attribute dict derived from the second parent

    Returns:
        dict: a combined attribute dictionary.
    """
    final_attr = dict[Any, Any]()
    if "eventType" in attr1:
        final_attr["eventType"] = attr1["eventType"]
    if "index" in attr1:
        final_attr["index"] = attr1["index"]
        
    comment_set = set[Any]()
    if "comment" in attr1:
        comment_set.add(attr1["comment"])
    if "comment" in attr2:
        comment_set.add(attr2["comment"])
    
    # Sort out inheritance probabilities and branches
    if parsed_node.label in inheritances:
        final_attr["gamma"] = inheritances[parsed_node.label]
    return final_attr
    
                    
###########################
#### Nexus File Parser ####
###########################
    
class NetworkParser:
    """
    Class that parses networks that are from nexus files.
    Now includes pre-parsing validation for better error reporting.
    """
    
    def __init__(self, 
                 filename : str, 
                 validate_input : bool = True, 
                 print_validation_summary : bool = True) -> None:
        """
        Initialize the parser with a nexus file.

        Raises:
            NetworkParserError: If the NexusReader library cannot parse the file.
        Args:
            filename (str): the path to the nexus file to be parsed.
            validate_input (bool): whether to validate the input file before parsing
            print_validation_summary (bool): whether to print validation summary
        Returns:
            N/A
        """
        self.filename = filename
        self.validation_summary : ValidationSummary = None
        
        # Validate input file if requested
        if validate_input:
            validator = NexusValidator()
            self.validation_summary = validator.validate(filename)
            
            if print_validation_summary:
                print(self.validation_summary)
                
            # Check if validation found critical errors
            if not self.validation_summary.is_valid:
                critical_errors = [error for error in self.validation_summary.errors 
                                 if "required" in error.lower() or "not found" in error.lower()]
                if critical_errors:
                    raise NetworkParserError(f"Critical validation errors found: {'; '.join(critical_errors)}")
        
        try:
            self.reader = NexusReader.from_file(filename)
        except Exception as err:
            traceback.print_exc()
            raise NetworkParserError("NexusReader library could not find \
                                      or parse this file.")
        
        #List of all parsed networks
        self.networks : list[Network] = []
        
        self.internal_count : int = 0
        
        # Map from network names (the label that appears before the newick 
        # string in a nexus file) to the parsed networks.
        self.name_2_net : dict[str, Network] = {}
        # Map from node names to their inheritance probabilities
        self.inheritance : dict[str, float] = {} 
        
        # Finally, parse the file.
        self.parse()

    def parse(self) -> None:
        """
        Using the reader object, iterate through each of the trees 
        defined in the file and store them as Network objects into the 
        networks array.
        
        Args:
            N/A
        Returns:
            N/A
        """

        if self.reader.trees is None:
            raise NetworkParserError("There are no trees listed in the file")

        for t in self.reader.trees:
            # grab the right hand side of the tree definition for 
            # the tree, and the left for the name
            
            name : str = str(t).split("=")[0].split(" ")[1]
            handle : StringIO = StringIO("=".join(str(t).split("=")[1:]))

            # parse the string handle
            tree = Phylo.read(handle, "newick")
            new_network = self.parse_tree_block(tree)
            
            # build the graph of the network
            self.networks.append(new_network)
            self.name_2_net[new_network] = name
            #reset the inheritance map for the next network
            self.inheritance = {}
            
    def parse_tree_block(self, tree : Any) -> Network:
        """
        Given a biopython Tree object (with nested clade objects)
        walk through the tree and build a network/ultrametric network
        from the nodes

        Args:
            tree (Any): the biopython library tree data structure

        Returns:
            Network : A phynetpy Network obj that has the same topology 
                      and names as the input network.
        """
        
        # Build a parent dictionary from the biopython tree obj
        parents = {}
        for clade in tree.find_clades(order = "level"):
            for child in clade:
                parents[child] = clade

        # create new empty directed acyclic graph
        net = Network()

        # populate said graph with nodes and their attributes
        for node, par in parents.items():
            # print(node.name)
            # print(par.name)
            parent_node = self.parse_parent(par, net)
            child_node = self.parse_child(node, net, parent_node)
        
        return net

    def parse_attributes(self, attr_str : str) -> list:
        """
        Takes the formatting string from the extended newick grammar and parses
        it into the event type and index.

        IE: H1 returns "Hybridization", 1
        IE: LGT21 returns "Lateral Gene Transfer", 21

        Raises:
            NodeError: if the node label does not match any sort of 
                       extended newick rules
        Args:
            attr_str (str): A node name, but one that carries 
                           information about the type of node.
        Returns:
            list: a list of two items, the first being a string describing the 
                  node type, the second an integer that gives an index
        """
        
        if len(attr_str) < 2:
            raise NodeError("reticulation event label formatting incorrect")

        indexLookup = 0

        # decipher event type
        if attr_str[0] == "R":
            event = "Recombination"
            indexLookup = 1
        elif attr_str[0] == "H":
            event = "Hybridization"
            indexLookup = 1
        elif attr_str[0] == "L":
            try:
                if attr_str[1] == "G" and attr_str[2] == "T":
                    event = "Lateral Gene Transfer"
                    indexLookup = 3
            except:
                raise NodeError("Invalid label format string (event error)")
        else:
            raise NodeError("Invalid label format string (event error) ")

        # parse node index
        try:
            strnum = attr_str[indexLookup:]
            num = int(strnum)
            return event, num
        except:
            raise NodeError("Invalid label format string (number error)")

    def parse_child(self, node : Any, network : Network, parent : Node) -> Node:
        """
        Process a child found in the Biopython tree.
        
        Args:
            node (Any): a node from the biopython tree
            network (Network): The partially built PhyNetPy Network
            parent (Node): the parent node of node, which should already exist  
                           as a PhyNetPy object.
        Returns:
            Node : 'node' translated to a PhyNetPy Node object. 
        """
        parsed_node : Node = network.has_node_named(node.name)
        
        if parsed_node is not None:
            more_attr = self.parse_comment(node, parent)
            
            parsed_node.set_attributes(merge_attributes(self.inheritance, 
                                                      parsed_node,
                                                      more_attr, 
                                                      parsed_node.get_attributes())) 
        else:
            if node.name is None:
                new_internal = "Internal" + str(self.internal_count)
                self.internal_count += 1
                node.name = new_internal
                
            parsed_node = Node(name = node.name)
            
            if node.name[0] == "#":
                parsed_node.set_is_reticulation(True)
                
            if node.branch_length is not None:
                parsed_node.set_time(parent.get_time() + node.branch_length)
            else:
                warn("No branch length has been provided for this node.\
                        Setting the branch length to 1.")
                parsed_node.set_time(parent.get_time() + 1)
                
            # Get additional information
            parsed_node.set_attributes(self.parse_comment(node, parent))
            
            # Add to network
            network.add_nodes(parsed_node) 
            
        # Add edge from the parent to child
        new_edge : Edge = Edge(parent, parsed_node)
        
        # Add branch length
        if node.branch_length is not None:
            new_edge.set_length(node.branch_length)
        else:
            new_edge.set_length(1)
        
        # Add inheritance probability (gamma) if applicable
        inheritance_prob : dict = parsed_node.attribute_value("gamma")
        
        if inheritance_prob is not None:
            #print(inheritance_prob)
            gamma_value = inheritance_prob[parent.label][0]
            new_edge.set_gamma(gamma_value)
            
        # Add edge to network
        network.add_edges(new_edge)
            
        return parsed_node
    
    def parse_parent(self, 
                     node : Any, 
                     network : Network, 
                     parent : Node = None) -> Node:
        """
        Process a node that is the parent of the other node due to be processed
        next.
        
        Args:
            node (Any): a biopython node
            network (Network): the partially built Network obj.
            parent (Node, optional): The parent of the biopython node, if
                                     available. Defaults to None.
        Returns:
            Node: The PhyNetPy node version of the biopython node
        """
        #Check to see if the node has already been processed.
        parsed_node : Node = network.has_node_named(node.name)
        
        if parsed_node is not None:
            return parsed_node
        else:
            
            # Give a name if the node doesn't have one (ie, an internal node)
            if node.name is None:
                newInternal = "Internal" + str(self.internal_count)
                self.internal_count += 1
                node.name = newInternal
            
            # Make a Node obj
            parsed_node : Node = Node(name = node.name)
            
            # Set the time of the Node
            if parent is None:
                #then we are processing the root, at t = 0
                parsed_node.set_time(0)
            else:
                par_time : float = parent.get_time()
                # The time of a node is the time of the parent plus the branch 
                # length.
                if node.branch_length is not None:
                    parsed_node.set_time(par_time + node.branch_length)
                else:
                    warn("No branch length has been provided for this node.\
                          Setting it to 1.")
                    parsed_node.set_time(par_time + 1)
            
            # Mark as a reticulation if necessary
            if node.name[0] == "#":
                parsed_node.set_is_reticulation(True)
                
            # Add any relevant attributes
            if parent is not None:
                #Not sure that this ever happens...
                parsed_node.set_attributes(self.parse_comment(node, parent))
            
            # Add to the network
            network.add_nodes(parsed_node)
            
        return parsed_node

    def parse_comment(self, node : Any, parent : Node) -> dict:
        """
        Each time a node block is processed, given its parent node (that should 
        have already been processed) look at the comment block if one exists 
        and bookkeep any relevant information.
        
        In particular, hybrid nodes with a [&gamma = (0,1)] comment need to be 
        recorded to keep track of inheritance probabilities.
        
        gamma attribute entries take the form of:
        {parent1 : [gamma1 , branch_length1],   
        parent2 : [gamma2, branch_length2]}
        
        where gamma1 and gamma2 sum to 1.
        
        Raises:
            NetworkParserError: if two gamma values for a node do not sum to 1.
        Args:
            node (Any): a node block from the nexus library
            parent (Node): a phynetpy Node obj that is the node block's 
                           parent in the network
        Returns:
           dict: an attribute dictionary for the newly processed node block.
        """
        attr = {}
        if node.name[0] == "#":
            event, num = self.parse_attributes(node.name.split("#")[1])
            attr["eventType"] = event
            attr["index"] = num
            
            if node.comment is not None:
                #CASE WHERE A COMMENT IS FILLED OUT
                if node.comment.split("=")[0] == "&gamma": 
                    # Parse out value
                    gamma = float(node.comment.split("=")[1])
                    
                    # Set attribute key and value.
                    attr["gamma"] = {parent.label : 
                                    [gamma, node.branch_length]}
                    
                    if node.name in self.inheritance:
                        for par, info in self.inheritance[node.name].items():
                            if info[0] == -1:
                                old_info = [1 - gamma, info[1]]
                                
                                par_2 = parent.label
                                info_2 = [gamma, node.branch_length]
                                self.inheritance[node.name] = {par: old_info, 
                                                               par_2 : info_2}
                            else:
                                if info[0] + gamma != 1:
                                    raise NetworkParserError("Gamma values\
                                                              provided in \
                                                              newick string do \
                                                              not add to 1")
                        
                                info_2 = [gamma, node.branch_length]
                                par_2 = parent.label
                                self.inheritance[node.name] = {par: info, 
                                                                par_2 : info_2}
                            break
                    else:
                        par = parent.label
                        info = [gamma, node.branch_length]
                        self.inheritance[node.name] = {par : info}
                else:
                    attr["comment"] = node.comment
            else:
                #CASE WHERE A COMMENT IS NULL
                
                if node.name in self.inheritance:
                    #should only be one, just using a for loop to access it ez
                    for par, info in self.inheritance[node.name].items(): 
                        if info[0] == -1:
                            #No info available, set to .5 and .5 
                            branch_length = node.branch_length #may be None
                            
                            gammas = {par : [.5, info[1]],
                                     parent.label : [.5, branch_length]}
                            
                            self.inheritance[node.name] = gammas
                        else:
                            new_par = parent.label
                            new_info = [1 - info[0], node.branch_length]
                            
                            self.inheritance[node.name] = {par : info,
                                                           new_par : new_info}
                        break
                else:
                    #No entry yet, and no information found.
                    # Initially set to -1 to indicate non-existence.
                    
                    par = parent.label
                    info = [-1, node.branch_length]
                    self.inheritance[node.name] = {par : info}  
        else:
            if node.comment is not None:
                attr["comment"] = node.comment
        
        return attr
            
    def get_network(self, index : int) -> Network:
        """
        Retrieves the network at index 'index' in the networks field

        Args:
            index (int): index

        Returns:
            Network: a parsed Network
        """
        return self.networks[index]

    def get_all_networks(self) -> list[Network]:
        """
        Retrieves the network array field

        Args:
            N/A
        Returns:
            list[Network] : the set of parsed networks
        """
        return self.networks

    def name_of_network(self, network : Network) -> str:
        """
        Given a parsed network, get the label for it.

        Args:
            network (Network): a network parsed from this NetworkParser
        Returns:
            str: Network label, as appears in the nexus file.
        """
        return self.name_2_net[network]
    
    def get_validation_summary(self) -> ValidationSummary:
        """
        Get the validation summary from the input file validation.
        
        Returns:
            ValidationSummary: The validation results, or None if validation was skipped
        """
        return self.validation_summary



class NewickStringParser:
    
    def __init__(self, newick_str : str) -> None:
        self.newick_str = newick_str
        self.intermediate = nw.loads(self.newick_str)

    def parse(self) -> None:
        """
        Using the reader object, iterate through each of the trees 
        defined in the file and store them as Network objects into the 
        networks array.
        
        Args:
            N/A
        Returns:
            N/A
        """

        edges = EdgeSet()
        nodes = NodeSet()
        adjacency = {} # dict from nodes to their children
        
        cur_node = self.intermediate[0]
        
        while len(cur_node.descendants) != 0:
            adjacency[cur_node.name] = "hi"
            
        
        return Network(edges, nodes)


