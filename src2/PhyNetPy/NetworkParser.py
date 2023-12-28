""" 
Author : Mark Kessler
Last Stable Edit : 12/5/23
First Included in Version : 0.1.0
Approved to Release Date : N/A
"""

import traceback
from nexus import NexusReader
from Bio import Phylo
from io import StringIO
from Graph import DAG
from Node import Node, NodeError


class NetworkParserError(Exception):
    def __init__(self, message = "Something went wrong with building the network") -> None:
        self.message = message
        super().__init__(self.message)


def merge_attributes(inheritances : dict, parsed_node : Node, attr1 : dict, attr2 : dict) -> dict:
    """

    Args:
        inheritances (dict):
        parsed_node (Node): _description_
        attr1 (dict): _description_
        attr2 (dict): _description_

    Returns:
        dict: _description_
    """
    final_attr = dict()
    if "eventType" in attr1:
        final_attr["eventType"] = attr1["eventType"]
    if "index" in attr1:
        final_attr["index"] = attr1["index"]
        
    comment_set = set()
    if "comment" in attr1:
        comment_set.add(attr1["comment"])
    if "comment" in attr2:
        comment_set.add(attr2["comment"])
    
    #Sort out inheritance probabilities and branches
    
    final_attr["gamma"] = inheritances[parsed_node.get_name()]
    return final_attr
    
                
        
    
class NetworkParser:

    def __init__(self, filename):
        try:
            self.reader = NexusReader.from_file(filename)
        except Exception as err:
            traceback.print_exc()
            raise NetworkParserError("NexusReader library could not find or parse this file.")
        
        self.networks = []
        self.internal_count = 0
        self.name_2_net = {}
        self.inheritance = {} #Map from node names to their inheritance probabilities
        self.parse()

    def parse(self):
        """
        Using the reader object, iterate through each of the trees 
        defined in the file and store them as Network objects into the 
        networks array
        """

    
        if self.reader.trees is None:
            raise NetworkParserError("There are no trees listed in the file")

        for t in self.reader.trees:
            # grab the right hand side of the tree definition for the tree, and the left for the name
            name = str(t).split("=")[0].split(" ")[1]
            handle = StringIO("=".join(str(t).split("=")[1:]))

            # parse the string handle
            tree = Phylo.read(handle, "newick")
            new_network = self.parse_tree_block(tree)
            # build the graph of the network
            self.networks.append(new_network)
            self.name_2_net[new_network] = name
            
    def parse_tree_block(self, tree)-> DAG:
        """
        Given a biopython Tree object (with nested clade objects)
        walk through the tree and build a network/ultrametric network
        from the nodes

        Args:
            tree (nexusreader tree): the nexus reader library tree data structure

        Returns:
            DAG: A phynetpy network obj that has the same topology and names as the input network.
        """
        
        # Build a parent dictionary from the biopython tree obj
        parents = {}
        for clade in tree.find_clades(order="level"):
            for child in clade:
                parents[child] = clade

        # create new empty directed acyclic graph
        net = DAG()

        # populate said graph with nodes and their attributes
        edges = []
        
        for node, par in parents.items():
            parent_node = self.parse_parent(par, net)
            child_node = self.parse_child(node, net, parent_node)
            child_node.add_parent(parent_node)
            edges.append([parent_node, child_node])
        

        net.add_edges(edges, as_list=True)
        
        return net

    def parse_attributes(self, attrStr : str) -> list:
        """
        Takes the formatting string from the extended newick grammar and parses
        it into the event type and index.

        IE: H1 returns "Hybridization", 1
        IE: LGT21 returns "Lateral Gene Transfer", 21

        Args:
            attrStr (str): A node name, but one that carries information about the type of node.

        Raises:
            NodeError: if the node label does not match any sort of extended newick rules

        Returns:
            list: a list of two items, the first being a string describing the node type, the second an integer that gives an index
        """
        
        if len(attrStr) < 2:
            raise NodeError("reticulation event label formatting incorrect")

        indexLookup = 0

        # decipher event type
        if attrStr[0] == "R":
            event = "Recombination"
            indexLookup = 1
        elif attrStr[0] == "H":
            event = "Hybridization"
            indexLookup = 1
        elif attrStr[0] == "L":
            try:
                if attrStr[1] == "G" and attrStr[2] == "T":
                    event = "Lateral Gene Transfer"
                    indexLookup = 3
            except:
                raise NodeError("Invalid label format string (event error)")
        else:
            raise NodeError("Invalid label format string (event error) ")

        # parse node index
        try:
            strnum = attrStr[indexLookup:]
            num = int(strnum)
            return event, num
        except:
            raise NodeError("Invalid label format string (number error)")

    def parse_child(self, node, network: DAG, parent : Node):
        """
        

        Args:
            node (_type_): _description_
            network (DAG): _description_
            parent (Node): _description_

        Returns:
            _type_: _description_
        """
        parsed_node : Node = network.has_node_named(node.name)
        if parsed_node is not None:
            if node.branch_length is not None:
                parsed_node.add_length(node.branch_length, parent)
            more_attr = self.parse_comment(node, parent)
            
            parsed_node.attributes = merge_attributes(self.inheritance, parsed_node, more_attr, parsed_node.attributes)    
        else:
            if node.name is None:
                new_internal = "Internal" + str(self.internal_count)
                self.internal_count += 1
                node.name = new_internal
                
            parsed_node = Node(name=node.name)
            if node.name[0] == "#":
                parsed_node.set_is_reticulation(True)
                
            parsed_node.add_length(node.branch_length, parent)
            parsed_node.attributes = self.parse_comment(node, parent)
            
        network.add_nodes(parsed_node) 
        return parsed_node
    
    def parse_parent(self, node, network: DAG, parent : Node = None) -> Node:
        
        parsed_node : Node = network.has_node_named(node.name)
        if parsed_node is not None:
            return parsed_node
        else:
            if node.name is None:
                newInternal = "Internal" + str(self.internal_count)
                self.internal_count += 1
                node.name = newInternal
                
            parsed_node = Node(name=node.name)
            if node.name[0] == "#":
                parsed_node.set_is_reticulation(True)
              
            if not(parent is None or node.branch_length is None):  
                parsed_node.add_length(node.branch_length, parent)
                
            parsed_node.attributes = self.parse_comment(node, parent)
            
        network.add_nodes(parsed_node)
        return parsed_node

    def parse_comment(self, node, parent : Node):
        """
        Each time a node block is processed, given its parent node (that should have already been processed)
        look at the comment block if one exists and bookkeep any relevant information.
        
        In particular, hybrid nodes with a [&gamma = (0,1)] comment need to be recorded to keep track of inheritance probabilities
        

        Args:
            node (nexus.node): a node block from the nexus library
            parent (Node): a phynetpy Node obj that is the node block's parent in the network

        Raises:
            NetworkParserError: if two gamma values for a node do not sum to 1.

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
                    gamma = float(node.comment.split("=")[1])
                    attr["gamma"] = {parent.get_name(): [gamma, node.branch_length]}
                    if node.name in self.inheritance:
                        for par, info in self.inheritance[node.name].items():
                            if info[0] == 0:
                                self.inheritance[node.name] = {par: [1-gamma, info[1]], parent.get_name():[gamma, node.branch_length]}
                            else:
                                if info[0] + gamma == 1:
                                    self.inheritance[node.name] = {par: info, parent.get_name():[gamma, node.branch_length]}
                                else:
                                    raise NetworkParserError("Gamma values provided in newick string do not add to 1")
                            break
                    else:
                        self.inheritance[node.name] = {parent.get_name():[gamma, node.branch_length]}
                else:
                    attr["comment"] = node.comment
            else:
                #CASE WHERE A COMMENT IS NULL
                
                if node.name in self.inheritance:
                    for par, info in self.inheritance[node.name].items(): #should only be one, just using a for loop to access it ez
                        if info[0] == 0:
                            self.inheritance[node.name] = {par: [.5, info[1]], parent.get_name():[.5, node.branch_length]}
                        else:
                            self.inheritance[node.name] = {par: info, parent.get_name():[1 - info[0], node.branch_length]}
                        break
        
                else:
                    self.inheritance[node.name] = {parent.get_name() : [0, node.branch_length]}  
        else:
            if node.comment is not None:
                attr["comment"] = node.comment
        
        return attr
            
        
    def getNetwork(self, i):
        return self.networks[i]

    def get_all_networks(self):
        return self.networks

    def name_of_network(self, network):
        return self.name_2_net[network]
