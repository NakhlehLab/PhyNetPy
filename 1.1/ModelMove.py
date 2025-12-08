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
"""

from __future__ import annotations
from collections import deque
import copy
import random
from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING
from PhyNetPy.Network import Network, Edge, Node
from PhyNetPy.Logger import *
if TYPE_CHECKING:
    from PhyNetPy.ModelGraph import *


#########################
#### EXCEPTION CLASS ####
#########################

class MoveError(Exception):
    """
    This exception is raised whenever there is a fatal error in executing a 
    network move.
    """
    def __init__(self, message : str = "Error executing a move") -> None:
        """
        Initializes a MoveError object.

        Args:
            message (str, optional): Custom error message. Defaults to
                                     "Error executing a move".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)

##########################
#### HELPER FUNCTIONS ####
##########################

def insert_node_in_edge(edge : Edge, node : Node, net : Network) -> None:
    """
    Given an edge, a -> b, place a node c, such that a -> c -> b.
    This requires the deletion of edge a -> b, then the addition of edges
    a -> c and c -> b.

    Args:
        edge (Edge): An edge, a -> b
        node (Node): A node, c.
        net (Network): The network that contains nodes a, b, and c
    Returns:
        N/A
    """
    a : Node = edge.src 
    b : Node = edge.dest
    
    #Rewire the edges
    net.remove_edge(edge)
    net.add_edges(Edge(a, node))
    net.add_edges(Edge(node, b))
    
def connect_nodes(src : Node, dest : Node, net : Network) -> None:
    """
    Given two nodes in a network, connect them and check whether or not a 
    reticulation is created.

    Args:
        src (Node): The parent of the new edge
        dest (Node): The child of the new edge
        net (Network): Network for which to add the edge
    Returns:
        N/A
    """
    #Add the edge to the network
    net.add_edges(Edge(src, dest))
  
    #Check if dest is now a reticulation
    if net.in_degree(dest) > 1:
        dest.set_is_reticulation(True)

###########################
#### MOVE PARENT CLASS ####
###########################

class Move(ABC):
    """
    Abstract superclass for all model move types.

    A move can be executed on a model that is passed in, and edits an aspect 
    of the model.
    """

    def __init__(self):
        """
        Moves in general do not require any parameters
        
        Args:
            N/A
        Returns:
            N/A
        """
        self.model = None
        self.undo_info = None
        self.same_move_info = None

    @abstractmethod
    def execute(self, model : Model) -> Model:
        """
        Args: 
            model (Model): A Model obj
        Returns: 
            Model: A new Model obj that is the result of this operation on
                     model
        """
        pass

    @abstractmethod
    def undo(self, model : Model) -> None:
        """
        A function that will undo what "execute" did.
        
        Ie:
        1) model1 executes this move
        2) model1 is rejected
        3) model1 now calls "undo" to revert to original topology.

        Args:
            model (Model): A phylogenetic network model object.
        Returns:
            N/A
        """
        pass

    @abstractmethod
    def same_move(self, model : Model) -> None:
        """
        Applies the exact move as execute, on a different but identical (with 
        respect to topology) Model object to a model that has had "execute" 
        called on it.
        
        Ie:
        1) model1 executes this move
        2) model1 is accepted
        3) model2 is identical to what model1 was pre-move
        4) "same_move" is applied to model2 to catch it up to model1
        

        Args:
            model (Model): A phylogenetic network model obj.
        Returns:
            N/A
        """
        pass
    
    @abstractmethod
    def hastings_ratio(self) -> float:
        """
        Returns the hastings-ratio for a move-- that is the ratio of valid 
        states to return to post-move, to the number of valid states to 
        transition to pre-move.

        Args:
            N/A
        Returns:
            float: Hastings Ratio. For symmetric moves, this is 1.0.
        """
        pass

# class UniformBranchMove(Move):

#     def execute(self, model : Model) -> Model:
#         """
#         Changes either the node height or branch length of a randomly 
#         selected node that is not the root.

#         Inputs: a Model obj, model
#         Outputs: new Model obj that is the result of changing one branch
#         """
#         # Make a copy of the model

#         proposedModel = model

#         # Select random internal node
#         selected = random.randint(0, len(proposedModel.netnodes_sans_root) - 1)
#         selected_node = proposedModel.netnodes_sans_root[selected]

#         # Change the branch to a value chosen uniformly from the allowable bounds
#         bounds = selected_node.node_move_bounds()
#         new_node_height = np.random.uniform(bounds[0], bounds[1])  # Assumes time starts at root and leafs are at max time

#         self.undo_info = [selected_node, selected_node.get_branches()[0].get()]
#         self.same_move_info = [selected_node.get_branches()[0].get_index(), new_node_height]
#         # Update the branch in the model
#         proposedModel.change_branch(selected_node.get_branches()[0].get_index(), new_node_height)

#         return proposedModel

#     def undo(self, model: Model)-> None:
#         model.change_branch(self.undo_info[0].get_branches()[0].get_index(), self.undo_info[1])

#     def same_move(self, model: Model) -> None:
#         model.change_branch(self.same_move_info[0], self.same_move_info[1])
    
#     def hastings_ratio(self) -> float:
#         return 1.0

# class RootBranchMove(Move, ABC):
    
#     def __init__(self):
#         super().__init__()
#         self.exp_param = 1
#         self.old_root_height = None
#         self.new_root_height = None

#     def execute(self, model: Model) -> Model:
#         """
#         Change the age of the tree by changing the height of the root node.

#         Inputs: a Model obj, model
#         Outputs: new Model obj that is the result of changing the root age

#         """
#         # Make a copy of model that is identical
#         proposedModel = model

#         # get the root and its height
#         #TODO: be flexible for snp root
#         speciesTreeRoot = proposedModel.felsenstein_root

#         currentRootHeight = speciesTreeRoot.get_branches()[0].get()

#         children = speciesTreeRoot.get_children()
#         if len(children) != 2:
#             raise MoveError("NOT A TREE, There are either too many or not enough children for the root")

#         # Calculate height that is the closest to the root
#         leftChildHeight = children[0].get_branches()[0].get()
#         rightChildHeight = children[1].get_branches()[0].get()

#         # the youngest age the species tree root node can be(preserving topologies)
#         # The lowest number that can be drawn from exp dist is 0, we guarantee that the root doesn't encroach on child
#         # heights.
#         uniformShift = np.random.exponential(self.exp_param) - min([currentRootHeight - leftChildHeight, currentRootHeight - rightChildHeight])
        
        
#         self.undo_info = [speciesTreeRoot, speciesTreeRoot.get_branches()[0].get()]
#         self.same_move_info = [speciesTreeRoot.get_branches()[0].get_index(), currentRootHeight + uniformShift]
#         self.new_root_height = currentRootHeight + uniformShift
#         self.old_root_height = currentRootHeight
#         # Change the node height of the root in the new model
#         proposedModel.change_branch(speciesTreeRoot.get_branches()[0].get_index(), currentRootHeight + uniformShift)

#         # return the slightly modified model
#         return proposedModel

#     def undo(self, model: Model) -> None:
#         model.change_branch(self.undo_info[0].get_branches()[0].get_index(), self.undo_info[1])

#     def same_move(self, model: Model) -> None:
#         model.change_branch(self.same_move_info[0], self.same_move_info[1])
    
#     def hastings_ratio(self) -> float:
#         return -1 * self.exp_param * (self.old_root_height - self.new_root_height)

# class TopologyMove(Move):
#     """
#     NNI Topology Move
#     """
    
#     def __init__(self):
#         super().__init__()
#         self.legal_forward_moves = None
#         self.legal_backwards_moves = None

#     def execute(self, model : Model) -> Model:
#         proposedModel = model
#         net : Network = proposedModel.network

#         valid_focals = {}

#         leaves = net.get_leaves()
#         roots = net.get_roots()
        
#         for n in net.V():
#             if n not in leaves and n not in roots and not n.is_reticulation():
#                 par = net.get_parents(n)[0]
#                 children = net.get_children(par)
#                 if children[1] == n:
#                     s = children[0]
#                 else:
#                     s = children[1]

#                 if s.get_time() < n.get_time():
#                     chosen_child = net.get_children(n)[random.randint(0, 1)]
#                     valid_focals[n] = [s, par, chosen_child]

#         if len(list(valid_focals.keys())) != 0:
#             self.legal_forward_moves = len(list(valid_focals.keys()))
#             choice = random.choice(list(valid_focals.keys()))

#             relatives : list[NetworkNode] = valid_focals[choice]
#             self.undo_info = [choice, relatives]
#             relatives[2].unjoin(choice)  # disconnect c1 from n
#             relatives[0].unjoin(relatives[1])  # disconnect s from par
#             relatives[2].join(relatives[1])  # connect c1 to par
#             relatives[0].join(choice)  # connect s to n

#             # No need to change branches, the right branches are already pointed
#             # at the right nodes

#             # mark each of c1 and choice as needing updating
#             relatives[0].upstream()
#             relatives[2].upstream()
#         else:
#             raise MoveError("DID NOT FIND A LEGAL TOPOLOGY MOVE")
        
        
#         # Calculate legal backwards moves for hastings ratio
#         valid_focals2 = {}
#         for n in proposedModel.internal:
#             par = n.get_parent()
#             children = par.get_children()
#             if children[1] == n:
#                 s = children[0]
#             else:
#                 s = children[1]

#             if s.get_branches()[0].get() < n.get_branches()[0].get():
#                 chosen_child = n.get_children()[random.randint(0, 1)]
#                 valid_focals2[n] = [s, par, chosen_child]

#         self.legal_backwards_moves = len(list(valid_focals2.keys()))
#         if self.legal_backwards_moves == 0:
#             raise MoveError("ENTERED INTO STATE WHERE THERE ARE NO MORE LEGAL TOPOLOGY MOVES")
        
#         return proposedModel

#     def undo(self, model: Model) -> None:
#         if self.undo_info is not None:
#             relatives = self.undo_info[1]
#             choice = self.undo_info[0]

#             # Do the reverse operations
#             relatives[2].join(choice)  # connect c1 back to n
#             relatives[0].join(relatives[1])  # connect s back to par
#             relatives[2].unjoin(relatives[1])  # disconnect c1 from par
#             relatives[0].unjoin(choice)  # disconnect s from n

#             # mark each of c1 and choice as needing updating
#             relatives[0].upstream()
#             relatives[2].upstream()

#     def same_move(self, model: Model) -> None:
#         if self.undo_info is not None:
#             relatives = self.undo_info[1]
#             choice = self.undo_info[0]

#             relatives_model = [None, None, None]
#             node_names = {node.label: node for node in relatives}
#             choice_model = None
#             choice_name = choice.label

#             # Use names to map this model instances nodes to the proposed_model nodes
#             netnodes = model.netnodes_sans_root
            
#             #TODO: be flexible for SNP stuff
#             netnodes.append(model.felsenstein_root)  # include root???????
#             for node in netnodes:
#                 if node.label in node_names.keys():
#                     index = relatives.index(node_names[node.label])
#                     relatives_model[index] = node
#                 if node.label == choice_name:
#                     choice_model = node

#             # Do the operations
#             relatives_model[2].unjoin(choice_model)  # disconnect c1 from n
#             relatives_model[0].unjoin(relatives_model[1])  # disconnect s from par
#             relatives_model[2].join(relatives_model[1])  # connect c1 to par
#             relatives_model[0].join(choice_model)

#             # mark each of c1 and choice as needing updating
#             relatives_model[0].upstream()
#             relatives_model[2].upstream()
    
#     def hastings_ratio(self) -> float:
#         return self.legal_forward_moves / self.legal_backwards_moves
 
    
####GRAPH MOVES####
"""
ALL OF THE FOLLOWING NETWORK MOVES HAVE VARIABLE NAMES THAT ARE BASED OFF OF THIS BASIC NETWORK STRUCTURE:

                  a
                    \
                     \  
                      \
            x          z  
           / \        / \
          /   \      /   \
         /     \    /     \
        /        c         \
       /         |          \
      /          |           \
                 y            b

"""


class AddReticulation(Move):
    """
    A move that adds a reticulation to a network.
    """
    
    def __init__(self) -> None:
        """
        Initializes a move that adds a reticulation to a network.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()

    def execute(self, model : Model) -> Model:
        """
        Adds a reticulation to a network.

        Args:
            model (Model): The model object containing the network.

        Returns:
            Model: The modified model with the added reticulation.
        """
        net : Network = model.network
        
        # Select random two edges
        src_e = random.choice(net.E())
        avoid_these_edges = net.edges_upstream_of_node(src_e.src)
        dest_e = random.choice([e for e in net.E() if e not in avoid_these_edges])
        
        a : Node = src_e.src
        b : Node = src_e.dest
        x : Node = dest_e.src
        y : Node = dest_e.dest
        z : Node = net.add_uid_node() # in branch a->b
        c : Node = net.add_uid_node() # in branch x->y
        c.set_is_reticulation(True)
        
        if a == x and b == y: 
            # Bubble
            insert_node_in_edge(net.get_edge(a, b), z, net)
            insert_node_in_edge(net.get_edge(z, b), c, net)
            connect_nodes(z, c, net)  
        else: 
            # Not a bubble
            insert_node_in_edge(net.get_edge(x, y), c, net)
            insert_node_in_edge(net.get_edge(a, b), z, net)
            connect_nodes(z, c, net)
            
        self.undo_info = [a, b, x, y, c, z]
        self.same_move_info = [node.label for node in self.undo_info]
        
        model.update_network()
        return model

    def undo(self, model : Model) -> None:
        """
        Undoes the addition of a reticulation to a network.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net : Network = model.network
        
        if self.undo_info is not None:
            a : Node = self.undo_info[0]
            b : Node = self.undo_info[1]
            x : Node = self.undo_info[2]
            y : Node = self.undo_info[3]
            c : Node = self.undo_info[4]
            z : Node = self.undo_info[5]
            
            net.remove_nodes(c)
            net.remove_nodes(z)
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)

        model.update_network()

    def same_move(self, model : Model) -> None:
        """
        Applies the same addition of a reticulation to another model.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net : Network = model.network
        
        if self.same_move_info is not None:
            nodes : list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            
            a : Node = nodes[0]
            b : Node = nodes[1]
            x : Node = nodes[2]
            y : Node = nodes[3]
            c : Node = Node(name=self.same_move_info[4], is_reticulation=True)
            z : Node = Node(name=self.same_move_info[5])
            
            net.add_nodes(z)
            net.add_nodes(c)
            
            insert_node_in_edge([a, b], z, net)
            insert_node_in_edge([x, y], c, net)
            connect_nodes(z, c, net)
        
        model.update_network()
    
    def hastings_ratio(self) -> float:
        """
        Returns the Hastings ratio for the addition move.
        
        Args:
            N/A 
        Returns:
            float: _description_
        """
        return 1.0

class RemoveReticulation(Move):
    """
    A move that removes a reticulation from a network.
    """
    def __init__(self) -> None:
        """
        Initializes a move that removes a reticulation from a network.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
    
    def execute(self, model : Model) -> Model:
        net : Network = model.network
        
        # Select a random reticulation edge to remove
        retic_edge : Edge = random.choice([e for e in net.E() if e.dest.is_reticulation()])
        
        c : Node = retic_edge.dest
        z : Node = retic_edge.src
        
        a : Node = net.get_children(c)[0]
        b : Node = [node for node in c.get_parents() if node != z][0]
        x : Node = net.get_children(z)[0]
        y : Node = z.get_parent()
        
        if a != x or b != y: # Not a bubble
            a.remove_parent(c)
            x.remove_parent(z)
            
            net.remove_edge(retic_edge)
            net.remove_nodes(c, True)
            net.remove_nodes(z, True)
            
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
            
            self.undo_info = [c, z, a, b, x, y]
            self.same_move_info = [node.label for node in self.undo_info]
        
        model.update_network()
        return model

    def undo(self, model : Model) -> None:
        net : Network = model.network
        if self.undo_info is not None:
            c : Node = self.undo_info[0]
            z : Node = self.undo_info[1]
            a : Node = self.undo_info[2]
            b : Node = self.undo_info[3]
            x : Node = self.undo_info[4]
            y : Node = self.undo_info[5]
            
            net.remove_edge([z, c])
            net.remove_nodes(c, True)
            net.remove_nodes(z, True)
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
        
        model.update_network()
    
    def same_move(self, model : Model) -> None:
        net : Network = model.network
        if self.same_move_info is not None:
            nodes : list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            c : Node = nodes[0]
            z : Node = nodes[1]
            a : Node = nodes[2]
            b : Node = nodes[3]
            x : Node = nodes[4]
            y : Node = nodes[5]
            
            net.remove_edge([z, c])
            net.remove_nodes(c, True)
            net.remove_nodes(z, True)
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
        
        model.update_network()
    
    def hastings_ratio(self) -> float:
        return 1.0
    
class RelocateReticulationSource(Move):
    """
    A move that relocates the source of a reticulation edge
    """
    def __init__(self) -> None:
        """
        Initializes a move that relocates the source of a reticulation edge.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
    
    def execute(self, model : Model) -> Model:
        """
        Relocates the source of a reticulation edge to a new edge in the network.

        Args:
            model (Model): The model object containing the network.

        Returns:
            Model: The modified model with the relocated reticulation source.
        """
        net : Network = model.network
        
        # Select a random reticulation edge to relocate
        retic_edge : Edge = random.choice([e for e in net.E() if e.dest.is_reticulation()])
        
        c : Node = retic_edge.dest
        z : Node = retic_edge.src
        x : Node = [node for node in net.get_children(z) if node != c][0]
        y : Node = z.get_parent()
       
        # Remove edge destination
        net.remove_nodes(z, True)
        x.remove_parent(z)
        connect_nodes(x, y, net)
        
        # Select a new edge
        new_edge : Edge = random.choice([e for e in net.E() if not e.dest.is_reticulation()])
        
        # Add new destination and reconnect c and z
        net.add_nodes(z)
        insert_node_in_edge(new_edge, z, net)
        connect_nodes(c, z, net)
        
        self.undo_info = [c, z, x, y, new_edge.dest, new_edge.src]
        self.same_move_info = [node.label for node in self.undo_info]
        
        model.update_network()
        return model

    def undo(self, model : Model) -> None:
        """
        Undoes the relocation of the reticulation source.

        Args:
            model (Model): The model object containing the network.

        Returns:
            N/A
        """
        net : Network = model.network
        if self.undo_info is not None:
            c : Node = self.undo_info[0]
            z : Node = self.undo_info[1]
            x : Node = self.undo_info[2]
            y : Node = self.undo_info[3]
            a : Node = self.undo_info[4]
            b : Node = self.undo_info[5]
            
            net.remove_nodes(z, True)
            a.remove_parent(z)
            connect_nodes(a, b, net)
            net.add_nodes(z)
            connect_nodes(c, z, net)
            z.set_parent([y])
            insert_node_in_edge([y, x], z, net)
        
        model.update_network()

    def same_move(self, model : Model) -> None:
        """
        Applies the same relocation of the reticulation source to another model.

        Args:
            model (Model): The model object containing the network.

        Returns:
            N/A
        """
        net : Network = model.network
        if self.same_move_info is not None:
            nodes : list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            c : Node = nodes[0]
            z : Node = nodes[1]
            a : Node = nodes[2]
            b : Node = nodes[3]
            x : Node = nodes[4]
            y : Node = nodes[5]
            
            net.remove_nodes(z, True)
            connect_nodes(x, y, net)
            insert_node_in_edge([b, a], z, net)
            connect_nodes(c, z, net)
    
    def hastings_ratio(self) -> float:
        """
        Returns the Hastings ratio for the relocation move.

        Args:
            N/A
        Returns:
            float: The Hastings ratio.
        """
        return 1.0
    
class RelocateReticulationDestination(Move):
    """
    A move that relocates the destination of a reticulation edge
    """
    
    def __init__(self) -> None:
        """
        Initializes a move that relocates the destination of a reticulation edge.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
    

    def execute(self, model : Model) -> Model:
        """
        Removes a reticulation edge from the network
        
        Args:
            model (Model): The model object containing the network.
        Returns:
            Model: The modified model with the relocated reticulation 
                   destination.
        """
        net : Network = model.network
        #net.print_graph()
        #Select a random reticulation edge to relocate
        retic_edge : tuple[Node] = random.choice([edge for edge in net.edges \
                                   if edge[1].is_reticulation()])
        
        c : Node = retic_edge[1]
        z : Node = retic_edge[0]
        
        #In all 4 cases, c/z both have exactly one parent and one child after removal of the retic edge
        a : Node = net.get_children(c)[0]
        b : Node = [node for node in c.get_parent(return_all=True) if node != z][0]
       
        
        #Remove edge src
        c.remove_parent(b)
        a.remove_parent(c)
        net.remove_nodes(c, True) 
        connect_nodes(a, b, net)
        
        #Select a new edge
        edge_set : list[tuple[Node]] = [edge for edge in net.edges \
                                        if edge[1].is_reticulation() == False]
        new_edge : tuple[Node] = random.choice(edge_set)
        
        #Add new destination and reconnect c and z
        net.add_nodes(c)
        insert_node_in_edge(new_edge, c, net)
        connect_nodes(c, z, net)
        
        self.undo_info = [c, z, a, b, new_edge[1], new_edge[0]]
        self.same_move_info = [node.label for node in self.undo_info]
        
        #Not handling branch lengths/bubbles at this time
        #net.print_graph()
        model.update_network()
        return model

    def undo(self, model : Model) -> None:
        """
        Undoes the relocation of the reticulation destination

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net : Network = model.network
        if self.undo_info is not None:
            c : Node = self.undo_info[0]
            z : Node = self.undo_info[1]
            a : Node = self.undo_info[2]
            b : Node = self.undo_info[3]
            x : Node = self.undo_info[4]
            y : Node = self.undo_info[5]
        
            # y : Node = [node for node in c.get_parent(return_all=True) if node.label != z.label][0]
            # x : Node = net.get_children(c)[0]
            
            #restore current edge
            net.remove_nodes(c, True)
            x.remove_parent(c)
            connect_nodes(x, y, net)
            net.add_nodes(c)
            
            #restore old edge
            net.add_edges([z, c])
            c.set_parent([b, z])
            insert_node_in_edge([b, a], c, net)
        model.update_network()
        net.print_graph()
            

    def same_move(self, model : Model) -> None:
        """
        Applies the same relocation of the reticulation destination to 
        another model.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net : Network = model.network
        if self.same_move_info is not None:
            nodes : list[Node] = [net.has_node_named(nodename) \
                                  for nodename in self.same_move_info]
            c : Node = nodes[0]
            z : Node = nodes[1]
            a : Node = nodes[2]
            b : Node = nodes[3]
            x : Node = nodes[4]
            y : Node = nodes[5]
            
            #Remove edge src
            net.remove_nodes(c, True) 
            a.remove_parent(c)
            connect_nodes(a, b, net)
            
            #Add new destination and reconnect c and z
            net.add_nodes(c)
            insert_node_in_edge([y, x], c, net)
            connect_nodes(c, z, net)
        
        model.update_network() 
        
    def hastings_ratio(self) -> float:
        """
        Returns the Hastings ratio for the relocation move.

        Args:
            N/A
        Returns:
            float: The Hastings ratio.
        """
        return 1.0
     
class RelocateReticulation(Move):
    """
    A move that relocates a reticulation edge
    """
    def __init__(self) -> None:
        """
        Initializes a move that relocates a reticulation edge.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
    
    def execute(self, model : Model) -> Model:
        """
        Removes a reticulation edge from the network
        
        Args:
            model (Model): The model object containing the network.
        Returns:
            Model: The modified model with the relocated reticulation
        """
        net : Network = model.network
    
        #Select a random reticulation edge to remove
        retic_edge : tuple[Node] = random.choice([edge for edge in net.edges \
                                                  if edge[1].is_reticulation()])
        
        c : Node = retic_edge[1]
        z : Node = retic_edge[0]
        a : Node = net.get_children(c)[0] 
        b : Node = [node for node in c.get_parent(return_all=True) if node != z][0]
        x : Node = [node for node in net.get_children(z) if node != c][0]
        y : Node = z.get_parent()

       
        if a!=x or b!=y: #Not a bubble
            net.remove_nodes(c, True)
            net.remove_nodes(z, True)
            x.remove_parent(z)
            a.remove_parent(c)
            c.set_parent([])
            z.set_parent([])
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
            
            E_set = [item for item in net.edges \
                     if item != [z, c] and item != [b, c]] 
            
            # Select 2 perhaps non-distinct edges to connect with a 
            # reticulation edge-- we allow bubbles
            random_edges = [random.choice(E_set), random.choice(E_set)]
            
            l : Node = random_edges[0][0]
            m : Node = random_edges[0][1]
            n : Node = random_edges[1][0]
            o : Node = random_edges[1][1]
            
            insert_node_in_edge([l, m], c, net)
            insert_node_in_edge([n, o], z, net)
            connect_nodes(c, z, net)
            
            #add the nodes back
            net.add_nodes(c)
            net.add_nodes(z)
            
            self.undo_info = [c, z, a, b, x, y, l, m, n, o]
            self.same_move_info = [node.label for node in self.undo_info]
        
        #Not handling branch lengths/bubbles at this time
        model.update_network()
        return model

    def undo(self, model : Model) -> None:
        """
        Undoes the relocation of the reticulation edge.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net : Network = model.network
        if self.undo_info is not None:
            c : Node = self.undo_info[0]
            z : Node = self.undo_info[1]
            a : Node = self.undo_info[2]
            b : Node = self.undo_info[3]
            x : Node = self.undo_info[4]
            y : Node = self.undo_info[5]
            l : Node = self.undo_info[6]
            m : Node = self.undo_info[7]
            n : Node = self.undo_info[8]
            o : Node = self.undo_info[9]
            
            net.remove_nodes(c, True)
            net.remove_nodes(z, True)
            c.set_parent([z, b])
            z.set_parent([y])
            m.remove_parent(c)
            o.remove_parent(z)
            connect_nodes(m, l, net)
            connect_nodes(o, n, net)
            
            insert_node_in_edge([b, a], c, net)
            insert_node_in_edge([y, x], z, net)
            
            net.add_nodes(c)
            net.add_nodes(z)
        
        net.print_graph()
        model.update_network()
            
            
    def same_move(self, model : Model) -> None:
        """
        Executes the same move on the other Model object.
        
        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net : Network = model.network
        if self.same_move_info is not None:
            nodes : list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            c : Node = nodes[0]
            z : Node = nodes[1]
            a : Node = nodes[2]
            b : Node = nodes[3]
            x : Node = nodes[4]
            y : Node = nodes[5]
            l : Node = nodes[6]
            m : Node = nodes[7]
            n : Node = nodes[8]
            o : Node = nodes[9]
            
            net.remove_nodes(c, True)
            net.remove_nodes(z, True)
            
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
            
            insert_node_in_edge((l, m), c, net)
            insert_node_in_edge((n, o), z, net)
            
            net.add_nodes(c)
            net.add_nodes(z)
        model.update_network()
        
    def hastings_ratio(self) -> float:
        """
        Return the hastings ratio of this move.
        
        Args:
            N/A
        Returns:
            float: The hastings ratio for the relocation move.
        """
        return 1.0
        
class FlipReticulation(Move):
    """
    A move that flips the direction of a reticulation edge
    """
    
    def __init__(self) -> None:
        """
        Initializes a move that flips the direction of a reticulation edge.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
    
    def execute(self, model : Model) -> Model:
        """
        Removes a reticulation edge from the network
        
        Args:
            model (Model): The model object containing the network.
        Returns:
            Model: The modified model with the flipped reticulation
        """
        net : Network = model.network
        #net.print_graph()
        #Select a random reticulation edge to remove
        retic_edge : tuple[Node] = random.choice([edge for edge in net.edges \
                                                  if edge[1].is_reticulation()])
        
        c : Node = retic_edge[1]
        z : Node = retic_edge[0]
       
        net.remove_edge([z, c])
        net.add_edges([c, z])
        
        c.remove_parent(z)
        z.add_parent(c)
        c.set_is_reticulation(False)
        z.set_is_reticulation(True)
        
        self.undo_info = [c, z]
        self.same_move_info = [c.label, z.label]
        
        #Not handling branch lengths/bubbles at this time
        #net.print_graph()
        model.update_network()
        return model

    def undo(self, model : Model) -> None:
        """
        Undoes the flipping of the reticulation edge.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net : Network = model.network
        if self.undo_info is not None:
            
            c : Node = self.undo_info[0]
            z : Node = self.undo_info[1]
            
            net.remove_edge([c, z])
            net.add_edges([z, c])
            
            z.remove_parent(c)
            c.add_parent(z)
            
            c.set_is_reticulation(True)
            z.set_is_reticulation(False)
        
        model.update_network()
        net.print_graph()

    def same_move(self, model : Model) -> None:
        """
        Applies the same flipping of the reticulation edge to another model.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net : Network = model.network
        if self.same_move_info is not None:
            nodes : list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            c : Node = nodes[0]
            z : Node = nodes[1]
            
            net.remove_edge([z, c])
            net.add_edges([c, z])
            
            c.remove_parent(z)
            z.add_parent(c)
            c.set_is_reticulation(False)
            z.set_is_reticulation(True)
        model.update_network()
            
    def hastings_ratio(self) -> float:
        """
        Return the hastings ratio for this move
        
        Args:
            N/A
        Returns:
            float: The hastings ratio for the flip reticulation move
        """
        return 1.0
    
class SwitchParentage(Move):
    """
    For use in Infer_MP_Allop, this move alters the genetic parentage of an 
    entire subnetwork, while maintaining the same ploidy values for each leaf.
    """
    def __init__(self, debug_id : int) -> None:
        """
        Initializes a move that switches the parentage of a subnetwork.
        
        Args:
            debug_id (int): The debug id for the move.
        Returns:
            N/A
        """
        super().__init__()
        
        #STEP 0: Set up edge/node tracking
        self.added_edges : list[list[Node]] = list()
        self.valid_attachment_edges : list[list[Node]] = list()
        self.added_nodes : set[Node] = set()
        self.removed_nodes : set[Node] = set()
        self.removed_edges : set[list[Node]] = set()
        self.logger = Logger(debug_id)
        self.print_net = False
        
    def random_object(self, mylist : list, rng : np.random.Generator) -> object:
        """
        Selects a random object from a list.

        Args:
            mylist (list): The list of objects to select from.
            rng (np.random.Generator): The random number generator.

        Returns:
            object: The randomly selected object.
        """
        rand_index = rng.integers(0, len(mylist))
        return mylist[rand_index]

    
    def execute(self, model : Model) -> Model:
        """
        Executes the Swap-Parentage Move, described in detail at
        https://phylogenomics.rice.edu/tutorials.html.

        Raises:
            MoveError : Aborts move if something irrecoverably wrong happens 
                        trying to make the move. Errors should not 
                        happen and these will be removed in production 
                        after things are fully tested.
        Args:
            model (Model): A model object, for which there must be a populated 
                           network field 
        Returns:
            Model: A modified model, with a newly proposed network topology
            
        """
        
        net : Network = model.network
        self.undo_info = copy.deepcopy(net) #TODO: get rid of this deep copy
                
        #STEP 1: Select random non-root node
        node_2_change : Node = self.random_object([node for node in net.V() if node != net.root()], model.rng)
    
        # STEP 1b: Disallow pointless changes 
        node_pars = net.get_parents(node_2_change)
        
        if len(node_pars) == 1:
            root_node = net.root()
            if node_pars[0] == root_node:
                root_kids = net.get_children(root_node)
                other_kid = [node for node in root_kids if node != node_2_change][0]
                if net.out_degree(other_kid) == 0:
                    return model
        
       
        if net.in_degree(node_2_change) == 2:    
            changing_retic = True    
        else:
            changing_retic = False
            
        #STEP 2: Get target subgenome count
        
        target : int = net.subgenome_count(node_2_change)
        
        #STEP 3: Remove a parent edge
        edge_2_remove : Edge = self.random_object(net.in_edges(node_2_change), model.rng)
        self.delete_edge(net, edge_2_remove)
        
        is_first_iter = True
        
        #STEP 4: Create new edges/parents
        if net.in_degree(node_2_change) == 1:
            cur_ct = net.subgenome_count(node_2_change)
        else:
            #We severed the only path out of the node
            cur_ct = 0
       
       
        iter_no = 0
        while cur_ct != target:

            # 4.0: Select the next edge to branch from
            if not is_first_iter:
                branch : Edge = self.random_object(list(self.valid_attachment_edges), model.rng)
                node_2_change = net.add_uid_node()
                net.remove_edge(branch)
                self.valid_attachment_edges.remove(branch)
                net.add_edges([Edge(branch.src, node_2_change),
                               Edge(node_2_change, branch.dest)])
                self.valid_attachment_edges.append(Edge(branch.src, node_2_change))
                self.valid_attachment_edges.append(Edge(node_2_change, branch.dest))
                downstream_node : Node =  node_2_change #branch[1]
            else:
                downstream_node = node_2_change
                
            # 4.1 : Select an edge with a key of <= cur_ct and that wont create a cycle (ensured by edges_2_subgct)
            bfs_starts = [node for node in net.V() if net.in_degree(node) == 0 and net.out_degree(node) != 0]
            
            if len(bfs_starts) > 1:
                if node_2_change in bfs_starts:
                    bfs_starts.remove(node_2_change)
                else:
                    raise MoveError("hmmm idk man")
            
            if len(bfs_starts) == 0:
                net.print_graph()
                model.update_network()
                return model
            else:
                bfs_start = bfs_starts[0]
                
            edges_to_ct = net.edges_to_subgenome_count(downstream_node, 
                                                       target - cur_ct, 
                                                       bfs_start)
        
            random_key = self.random_object([key for key in edges_to_ct.keys()],
                                            model.rng)
            
            try:
                new_edge : Edge = self.random_object(list(edges_to_ct[random_key]), model.rng)
                if new_edge.dest == downstream_node:
                    self.print_net = True
            except:
                raise MoveError("No edges with a sufficiently low/exact amount")
            
            
            # 4.2 : Connect the unconnected node to the new branch selected in 4.1
            connector_node = net.add_uid_node()
            new_edge_list = [Edge(connector_node, new_edge.dest), 
                             Edge(new_edge.src, connector_node), 
                             Edge(connector_node, node_2_change)]
            net.add_edges(new_edge_list)
            self.valid_attachment_edges.append(new_edge_list[2])
            net.remove_edge(new_edge)
            
            cur_ct = net.subgenome_count(node_2_change)
        
            is_first_iter = False
            iter_no += 1
            
            
        #STEP 5: Remove excess nodes created by initial edge removal if they exist
        
        net.clean()
        
        if changing_retic:
            changing_retic = False
       
        model.update_network()
        
        self.same_move_info = copy.deepcopy(net)
    
        return model

    def undo(self, model : Model) -> None:
        """
        Undoes the Swap-Parentage Move

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        
        if self.undo_info is not None:
            
            model.network = self.undo_info
    
        model.update_network()

    def same_move(self, model : Model) -> None:
        """
        Executes the same topology change on another model

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        if self.same_move_info is not None:
            model.network = self.same_move_info
            
        model.update_network()
    
    def hastings_ratio(self) -> float:
        """
        Returns the Hastings ratio for the Swap-Parentage Move.

        Args:
            N/A
        Returns:
            float: The Hastings ratio.
        """
        return 1.0
         
    def delete_edge(self, net : Network, edge : Edge) -> None:
        """
        Deletes an edge from the network, and any edges/nodes up until the first
        tree node has been discovered.

        Args:
            net (Network): The network object.
            edge (Edge): The edge to delete.
        Returns:
            N/A
        """
        root = edge.dest 

        q = deque()
        q.appendleft(root)

        if net.in_degree(root) == 2:
            bypass = True
        else:
            bypass = False 
            
        while len(q) != 0:
            cur = q.pop() 
            
            neighbors = copy.copy(net.get_parents(cur)) 
            
            if len(neighbors) == 2 and neighbors[0].label == neighbors[1].label and bypass:
                #Bubble
                b = copy.copy(net.get_parents(neighbors[1]))
                
                if len(b) != 0:
                    net.remove_edge([b[0], neighbors[0]])
                    net.remove_edge([neighbors[0], cur])
                    net.remove_edge([neighbors[1], cur])
                    net.add_edges(Edge(b[0], cur))
                else:
                    net.remove_edge([neighbors[0], cur])
                
                return 
            else: 
                i = 1
                b = copy.copy(net.get_parents(neighbors[0])) 
                for neighbor in neighbors:
                    
                    if not bypass or neighbor == edge.src: 
                        net.remove_edge([neighbor, cur]) 
                        
                        if net.in_degree(neighbor) == 2:
                            q.append(neighbor)
                        else:
                            try:
                                a : Node = [node for node in net.get_children(neighbor) if node != cur][0] 
                                
                                if net.in_degree(neighbor) != 0:
                                    b : Node = net.get_parents(neighbor)[0] 
                                    
                                    redundant_tree_edge1 = [b, neighbor] 
                                    redundant_tree_edge2 = [neighbor, a] 

                                    net.remove_edge(redundant_tree_edge1)
                                    net.remove_edge(redundant_tree_edge2)

                                    net.add_edges(Edge(b, a))
                            except:
                                if i == 2:
                                    if len(b) != 0:
                                        net.remove_edge([b[0], neighbor])
                    i += 1
                                    
                    
            bypass = False
            
# class SwitchParentage2(Move):
#     def __init__(self):
#         super().__init__()
        
#         #STEP 0: Set up edge/node tracking
#         self.added_edges : list[list[Node]] = list()
#         self.valid_attachment_edges : list[list[Node]] = list()
#         self.added_nodes : set[Node] = set()
#         self.removed_nodes : set[Node] = set()
#         self.removed_edges : set[list[Node]] = set()
        
#         self.print_net = False
#         self.ready = False
        
#     def random_object(self, mylist : list, rng: np.random.Generator) -> Any:
#         """
#         Randomly selects an object from a list.
        
#         Raises:
#             MoveError: If the list is empty, this error is raised.
#         Args:
#             mylist (list): A list of objects to randomly select from
#             rng (np.random.Generator): A numpy random number generator object
#         Returns:
#             Any: A randomly selected object from the list
#         """
#         if not self.ready:
#             raise NotImplementedError("This function is not ready for use.")
        
#         if len(mylist) == 0:
#             raise MoveError("Tried to randomly access elements of an empty list.")
#         rand_index = rng.integers(0, len(mylist))
#         return mylist[rand_index]

#     def execute(self, model : Model) -> Model:
#         """
#         Executes the Swap-Parentage Move, described in detail at 
#         https://phylogenomics.rice.edu/tutorials.html
#         This is the streamlined version.
#         --NOT READY FOR USE YET--
        
#         Raises:
#             MoveError : Aborts move if something irrecoverably wrong happens 
#                         trying to make the move. Errors should not 
#                         happen and these will be removed in production after 
#                         things are fully tested.
#         Args:
#             model (Model): A model object, for which there must be a populated 
#                            network field 
#         Returns:
#             Model: A modified model, with a newly proposed network topology
            
#         """
#         if not self.ready: 
#             raise NotImplementedError("This function is not ready for use.")
        
#         #STEP 0: Setup
#         net : Network = model.network
#         self.undo_info = copy.deepcopy(net)
                
#         #STEP 1: Select random non-root node, n.
#         n : Node = self.random_object([node for node in net.nodes if node != net.root()], model.rng)
            
#         #STEP 2: Get target subgenome count
#         target : int = net.subgenome_count(n)
        
#         #STEP 3: Disconnect Psi_n and Recalculate subgenome counts
#         e : list[Node] = self.random_object(net.in_edges(n), model.rng)
#         self.disconnect_mswle(net, e)
        
#         #STEP 4: Calculate current subgenome counts
#         cur_ct = net.subgenome_count(n)
        
#         #STEP 5: Iteratively Reattach
#         while cur_ct != target:
#             self.reattach()
            
#         #STEP 6: book keeping
        
#         net.clean()
        
#         model.update_network()
#         self.same_move_info = copy.deepcopy(net)
        
#         return model

#     def undo(self, model : Model) -> None:
#         """
#         Undoes the last move made by the model.

#         Raises:
#             NotImplementedError: If the function is not ready for use, 
#                                  this error is raised.
        
#         Args:
#             model (Model): The model object to undo the last move on.
#         Returns:
#             N/A
#         """
#         if not self.ready:
#             raise NotImplementedError("This function is not ready for use.")
        
#         if self.undo_info is not None:
            
#             model.network = self.undo_info
    
#         model.update_network()

#     def same_move(self, model : Model) -> None:
#         """
#         Repeats the last move made by the model.

#         Raises:
#             NotImplementedError: If the function is not ready for use,
#                                  this error is raised
                                 
#         Args:
#             model (Model): The model object to repeat the last move on.
#         Returns:
#             N/A
#         """
#         if not self.ready:
#             raise NotImplementedError("This function is not ready for use.")
        
#         if self.same_move_info is not None:
#             model.network = self.same_move_info
            
#         model.update_network()
    
#     def hastings_ratio(self) -> float:
#         """
#         Calculates the Hastings ratio for the last move made.

#         Raises:
#             NotImplementedError: If the function is not ready for use,
#                                  this error is raised.
#         Args:
#             N/A
#         Returns:
#             float: The Hastings ratio for the last move made.
        
#         """
#         if not self.ready:
#             raise NotImplementedError("This function is not ready for use.")
#         return 1.0
    
            
#     def disconnect_mswle(self, net : Network, edge : list[Node]) -> None:
#         """
#         Algorithm 2, DisconnectMSWLE, from the new paper.

#         Args:
#             net (Network): The network object to disconnect the subnetwork from
#             edge (list[Node]): The edge to disconnect the subnetwork from.
#         Returns:
#             N/A
#         """
        
#         if not self.ready:
#             raise NotImplementedError("This function is not ready for use.")
        
#         n = edge[1]
#         q = deque()
#         q.appendleft(edge[0])
#         prev = n
        
#         ld : dict[Node, set[Node]] = net.leaf_descendants_all()
        
#         while len(q) != 0:
#             cur = q.pop()
#             net.remove_edge([cur, prev])
            
#             if ld[n] != ld[cur]:
#                 continue
            
#             for par in net.get_parents(cur):
#                 q.appendleft(par)
#             prev = cur
#             net.remove_nodes(cur)
    
#     def reattach(self, 
#                  net : Network, 
#                  cur : int, 
#                  target : int, 
#                  rng : np.random.Generator) -> None:
#         """
#         Algorithm 3, Reattach, from the new paper.
        
#         Raises:
#             NotImplementedError: If the function is not ready for use,
#                                  this error is raised.
#         Args:
#             net (Network): The network object to reattach the subnetwork to
#             cur (int): The current subgenome count of the node
#             target (int): The target subgenome count of the node
#             rng (np.random.Generator): A numpy random number generator object
#         Returns:
#             N/A
#         """
        
#         if not self.ready:
#             raise NotImplementedError("This function is not ready for use.")
#         ##STILL NEED TO DO THIS....
#         """
#         if not is_first_iter:
#             branch : list[Node] = self.random_object(list(self.valid_attachment_edges), rng)
        
#             node_2_change = net.add_uid_node()
#             net.remove_edge(branch)
#             self.valid_attachment_edges.remove(branch)
#             net.add_edges([[branch[0], node_2_change], [node_2_change, branch[1]]])
#             self.valid_attachment_edges.append([branch[0], node_2_change])
#             self.valid_attachment_edges.append([node_2_change, branch[1]])
#             downstream_node : Node =  node_2_change #branch[1]
#         else:
#             downstream_node = node_2_change
            
#         # 4.1 : Select an edge with a key of <= cur_ct and that wont create a cycle (ensured by edges_2_subgct)
#         bfs_starts = [node for node in net.nodes if net.in_degree(node) == 0 and net.out_degree(node) != 0]
        
#         if len(bfs_starts)>1:
#             if node_2_change in bfs_starts:
#                 bfs_starts.remove(node_2_change)
#             else:
#                 raise MoveError("hmmm idk man")
        
#         if len(bfs_starts) == 0:
#             print(f"SOMETHING FUNKY : {node_2_change.label}")
#             net.print_graph()
#             model.update_network()
#             return model
#         else:
#             bfs_start = bfs_starts[0]
            
#         edges_to_ct : dict[int, set] = net.edges_to_subgenome_count(downstream_node, target - cur_ct, bfs_start)
        

    
#         random_key = self.random_object([key for key in edges_to_ct.keys()], model.rng)
        
#         try:
#             new_edge : list[Node] = self.random_object(list(edges_to_ct[random_key]), model.rng)
#             if new_edge[1] == downstream_node:
#                 #print("MAKING A BUBBLE-- THEORETICALLY")
#                 self.print_net = True
#         except:
#             raise MoveError("No edges with a sufficiently low/exact amount")
        
        
#         # 4.2 : Connect the unconnected node to the new branch selected in 4.1
       
#         connector_node = net.add_uid_node()
#         new_edge_list = [[connector_node, new_edge[1]], [new_edge[0], connector_node], [connector_node, node_2_change]]
#         net.add_edges(new_edge_list)
#         self.valid_attachment_edges.append(new_edge_list[2])
#         net.remove_edge(new_edge)
        
        
#         if len(net.root()) > 1:
#             raise MoveError("OOPS, more than one root")
        
#         cur_ct = net.subgenome_count(node_2_change)
    
#         is_first_iter = False
#         """
#         pass

class SPR(Move):
    """
    A move that performs a Subtree Prune and Regraft operation on a network.
    """
    def __init__(self, debug_id : int) -> None:
        """
        Initializes a move that performs a Subtree Prune and Regraft operation.

        Args:
            debug_id (int): The debug id for the move.
        Returns:
            N/A
        """
        super().__init__()
        self.logger = Logger(debug_id)
        self.undo_info = None
        self.same_move_info = None

    def random_object(self, mylist : list, rng : np.random.Generator) -> object:    
        """
        Selects a random object from a list.

        Args:
            mylist (list): The list of objects to select from.
            rng (np.random.Generator): The random number generator.

        Returns:
            object: The randomly selected object.
        """
        rand_index = rng.integers(0, len(mylist))
        return mylist[rand_index]

    def execute(self, model: Model) -> Model:
        """
        Executes the Subtree Prune and Regraft move.

        Args:
            model (Model): The model object containing the network.
        Returns:
            Model: The modified model with the Subtree Prune and Regraft 
                   move executed.
        """
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        # Select a random edge to cut
        edge_to_cut: Edge = self.random_object(net.E(), model.rng)
        src, dest = edge_to_cut.src, edge_to_cut.dest

        # Remove the selected edge
        net.remove_edge(edge_to_cut)

        # Collect the subtree rooted at dest
        subtree_nodes = net.get_subtree_at(dest)
        subtree_edges = net.edges_downstream_of_node(dest)

        # Remove the subtree from the network
        for edge in subtree_edges:
            net.remove_edge(edge)
        for node in subtree_nodes:
            net.remove_nodes(node)

        # Select a random edge to reattach the subtree
        reattachment_edge: Edge = self.random_object(net.E(), model.rng)
        reattachment_src, reattachment_dest = reattachment_edge.src, reattachment_edge.dest

        # Insert the subtree back into the network
        net.add_edges(Edge(reattachment_src, dest))
        for edge in subtree_edges:
            net.add_edges(edge)

        model.update_network()
        self.same_move_info = copy.deepcopy(net)
        return model

    def undo(self, model : Model) -> None: 
        """
        Undoes the Subtree Prune and Regraft move.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model : Model) -> None:
        """
        Perform the same Subtree Prune and Regraft move on another model.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        if self.same_move_info is not None:
            model.network = self.same_move_info
        model.update_network()

    def hastings_ratio(self) -> float:
        """
        Returns the Hastings ratio for the Subtree Prune and Regraft move.

        Args:
            N/A
        Returns:
            float: The Hastings ratio.
        """
        return 1.0



