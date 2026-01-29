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

V1 Architecture - Model Move operations for MCMC.
"""

from __future__ import annotations
from collections import deque
import copy
import random
from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING, Any

# Relative imports
from .Network import Network, Edge, Node
from .Logger import Logger

if TYPE_CHECKING:
    from .ModelGraph import Model


#########################
#### EXCEPTION CLASS ####
#########################

class MoveError(Exception):
    """
    This exception is raised whenever there is a fatal error in executing a 
    network move.
    """
    def __init__(self, message: str = "Error executing a move") -> None:
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

def insert_node_in_edge(edge: Edge, node: Node, net: Network) -> None:
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
    a: Node = edge.src 
    b: Node = edge.dest
    
    # Rewire the edges
    net.remove_edge(edge)
    net.add_edges(Edge(a, node))
    net.add_edges(Edge(node, b))
    
def connect_nodes(src: Node, dest: Node, net: Network) -> None:
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
    # Add the edge to the network
    net.add_edges(Edge(src, dest))
  
    # Check if dest is now a reticulation
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
    def execute(self, model: Model) -> Model:
        """
        Args: 
            model (Model): A Model obj
        Returns: 
            Model: A new Model obj that is the result of this operation on
                     model
        """
        pass

    @abstractmethod
    def undo(self, model: Model) -> None:
        """
        A function that will undo what "execute" did.
        
        Args:
            model (Model): A phylogenetic network model object.
        Returns:
            N/A
        """
        pass

    @abstractmethod
    def same_move(self, model: Model) -> None:
        """
        Applies the exact move as execute, on a different but identical (with 
        respect to topology) Model object to a model that has had "execute" 
        called on it.

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


####GRAPH MOVES####
r"""
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

    def execute(self, model: Model) -> Model:
        """
        Adds a reticulation to a network.

        Args:
            model (Model): The model object containing the network.

        Returns:
            Model: The modified model with the added reticulation.
        """
        net: Network = model.network
        
        # Select random two edges
        src_e = random.choice(net.E())
        avoid_these_edges = net.edges_upstream_of_node(src_e.src)
        dest_e = random.choice([e for e in net.E() if e not in avoid_these_edges])
        
        a: Node = src_e.src
        b: Node = src_e.dest
        x: Node = dest_e.src
        y: Node = dest_e.dest
        z: Node = net.add_uid_node()  # in branch a->b
        c: Node = net.add_uid_node()  # in branch x->y
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

    def undo(self, model: Model) -> None:
        """
        Undoes the addition of a reticulation to a network.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net: Network = model.network
        
        if self.undo_info is not None:
            a: Node = self.undo_info[0]
            b: Node = self.undo_info[1]
            x: Node = self.undo_info[2]
            y: Node = self.undo_info[3]
            c: Node = self.undo_info[4]
            z: Node = self.undo_info[5]
            
            net.remove_nodes(c)
            net.remove_nodes(z)
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)

        model.update_network()

    def same_move(self, model: Model) -> None:
        """
        Applies the same addition of a reticulation to another model.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net: Network = model.network
        
        if self.same_move_info is not None:
            nodes: list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            
            a: Node = nodes[0]
            b: Node = nodes[1]
            x: Node = nodes[2]
            y: Node = nodes[3]
            c: Node = Node(name=self.same_move_info[4], is_reticulation=True)
            z: Node = Node(name=self.same_move_info[5])
            
            net.add_nodes(z)
            net.add_nodes(c)
            
            insert_node_in_edge(Edge(a, b), z, net)
            insert_node_in_edge(Edge(x, y), c, net)
            connect_nodes(z, c, net)
        
        model.update_network()
    
    def hastings_ratio(self) -> float:
        """
        Returns the Hastings ratio for the addition move.
        
        Args:
            N/A 
        Returns:
            float: The Hastings ratio.
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
    
    def execute(self, model: Model) -> Model:
        net: Network = model.network
        
        # Select a random reticulation edge to remove
        retic_edges = [e for e in net.E() if e.dest.is_reticulation()]
        if not retic_edges:
            return model
            
        retic_edge: Edge = random.choice(retic_edges)
        
        c: Node = retic_edge.dest
        z: Node = retic_edge.src
        
        c_children = net.get_children(c)
        c_parents = c.get_parents() if hasattr(c, 'get_parents') else net.get_parents(c)
        z_children = net.get_children(z)
        z_parent = z.get_parent() if hasattr(z, 'get_parent') else None
        
        if not c_children or not z_children:
            return model
            
        a: Node = c_children[0]
        b: Node = [node for node in c_parents if node != z][0] if len(c_parents) > 1 else None
        x: Node = [node for node in z_children if node != c][0] if len(z_children) > 1 else None
        y: Node = z_parent
        
        if b is None or x is None or y is None:
            return model
        
        if a != x or b != y:  # Not a bubble
            net.remove_edge(retic_edge)
            net.remove_nodes(c, True)
            net.remove_nodes(z, True)
            
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
            
            self.undo_info = [c, z, a, b, x, y]
            self.same_move_info = [node.label for node in self.undo_info]
        
        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        net: Network = model.network
        if self.undo_info is not None:
            c: Node = self.undo_info[0]
            z: Node = self.undo_info[1]
            a: Node = self.undo_info[2]
            b: Node = self.undo_info[3]
            x: Node = self.undo_info[4]
            y: Node = self.undo_info[5]
            
            net.add_nodes(c)
            net.add_nodes(z)
            net.add_edges(Edge(z, c))
            insert_node_in_edge(Edge(b, a), c, net)
            insert_node_in_edge(Edge(y, x), z, net)
        
        model.update_network()
    
    def same_move(self, model: Model) -> None:
        net: Network = model.network
        if self.same_move_info is not None:
            nodes: list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            c: Node = nodes[0]
            z: Node = nodes[1]
            a: Node = nodes[2]
            b: Node = nodes[3]
            x: Node = nodes[4]
            y: Node = nodes[5]
            
            net.remove_edge(Edge(z, c))
            net.remove_nodes(c, True)
            net.remove_nodes(z, True)
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
        
        model.update_network()
    
    def hastings_ratio(self) -> float:
        return 1.0


class FlipReticulation(Move):
    """
    A move that flips the direction of a reticulation edge.
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
    
    def execute(self, model: Model) -> Model:
        """
        Removes a reticulation edge from the network
        
        Args:
            model (Model): The model object containing the network.
        Returns:
            Model: The modified model with the flipped reticulation
        """
        net: Network = model.network
        
        # Select a random reticulation edge to remove
        retic_edges = [e for e in net.E() if e.dest.is_reticulation()]
        if not retic_edges:
            return model
            
        retic_edge: Edge = random.choice(retic_edges)
        
        c: Node = retic_edge.dest
        z: Node = retic_edge.src
       
        net.remove_edge(retic_edge)
        net.add_edges(Edge(c, z))
        
        if hasattr(c, 'remove_parent'):
            c.remove_parent(z)
        if hasattr(z, 'add_parent'):
            z.add_parent(c)
        c.set_is_reticulation(False)
        z.set_is_reticulation(True)
        
        self.undo_info = [c, z]
        self.same_move_info = [c.label, z.label]
        
        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        """
        Undoes the flipping of the reticulation edge.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net: Network = model.network
        if self.undo_info is not None:
            c: Node = self.undo_info[0]
            z: Node = self.undo_info[1]
            
            net.remove_edge(Edge(c, z))
            net.add_edges(Edge(z, c))
            
            if hasattr(z, 'remove_parent'):
                z.remove_parent(c)
            if hasattr(c, 'add_parent'):
                c.add_parent(z)
            
            c.set_is_reticulation(True)
            z.set_is_reticulation(False)
        
        model.update_network()

    def same_move(self, model: Model) -> None:
        """
        Applies the same flipping of the reticulation edge to another model.

        Args:
            model (Model): The model object containing the network.
        Returns:
            N/A
        """
        net: Network = model.network
        if self.same_move_info is not None:
            nodes: list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            c: Node = nodes[0]
            z: Node = nodes[1]
            
            net.remove_edge(Edge(z, c))
            net.add_edges(Edge(c, z))
            
            if hasattr(c, 'remove_parent'):
                c.remove_parent(z)
            if hasattr(z, 'add_parent'):
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
    def __init__(self, debug_id: int = 0) -> None:
        """
        Initializes a move that switches the parentage of a subnetwork.
        
        Args:
            debug_id (int): The debug id for the move.
        Returns:
            N/A
        """
        super().__init__()
        
        self.added_edges: list[Edge] = list()
        self.valid_attachment_edges: list[Edge] = list()
        self.added_nodes: set[Node] = set()
        self.removed_nodes: set[Node] = set()
        self.removed_edges: set[Edge] = set()
        self.logger = Logger(str(debug_id))
        self.print_net = False
        
    def random_object(self, mylist: list, rng: np.random.Generator) -> object:
        """
        Selects a random object from a list.

        Args:
            mylist (list): The list of objects to select from.
            rng (np.random.Generator): The random number generator.

        Returns:
            object: The randomly selected object.
        """
        if len(mylist) == 0:
            return None
        rand_index = rng.integers(0, len(mylist))
        return mylist[rand_index]
    
    def execute(self, model: Model) -> Model:
        """
        Executes the Swap-Parentage Move.

        Args:
            model (Model): A model object, for which there must be a populated 
                           network field 
        Returns:
            Model: A modified model, with a newly proposed network topology
        """
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)
        
        # STEP 1: Select random non-root node
        non_root_nodes = [node for node in net.V() if node != net.root()]
        if not non_root_nodes:
            return model
            
        node_2_change: Node = self.random_object(non_root_nodes, model.rng)
        if node_2_change is None:
            return model
    
        # STEP 1b: Disallow pointless changes 
        node_pars = net.get_parents(node_2_change)
        
        if len(node_pars) == 1:
            root_node = net.root()
            if node_pars[0] == root_node:
                root_kids = net.get_children(root_node)
                other_kids = [node for node in root_kids if node != node_2_change]
                if other_kids:
                    other_kid = other_kids[0]
                    if net.out_degree(other_kid) == 0:
                        return model
        
        changing_retic = net.in_degree(node_2_change) == 2
            
        # STEP 2: Get target subgenome count
        target: int = net.subgenome_count(node_2_change)
        
        # STEP 3: Remove a parent edge
        in_edges = net.in_edges(node_2_change)
        if not in_edges:
            return model
            
        edge_2_remove: Edge = self.random_object(in_edges, model.rng)
        if edge_2_remove is None:
            return model
            
        self.delete_edge(net, edge_2_remove)
        
        is_first_iter = True
        
        # STEP 4: Create new edges/parents
        if net.in_degree(node_2_change) == 1:
            cur_ct = net.subgenome_count(node_2_change)
        else:
            cur_ct = 0
       
        iter_no = 0
        max_iter = 100  # Safety limit
        
        while cur_ct != target and iter_no < max_iter:
            # 4.0: Select the next edge to branch from
            if not is_first_iter:
                if not self.valid_attachment_edges:
                    break
                branch: Edge = self.random_object(list(self.valid_attachment_edges), model.rng)
                if branch is None:
                    break
                    
                node_2_change = net.add_uid_node()
                net.remove_edge(branch)
                self.valid_attachment_edges.remove(branch)
                net.add_edges(Edge(branch.src, node_2_change))
                net.add_edges(Edge(node_2_change, branch.dest))
                self.valid_attachment_edges.append(Edge(branch.src, node_2_change))
                self.valid_attachment_edges.append(Edge(node_2_change, branch.dest))
                downstream_node: Node = node_2_change
            else:
                downstream_node = node_2_change
                
            # 4.1: Select an edge with appropriate subgenome count
            bfs_starts = [node for node in net.V() if net.in_degree(node) == 0 and net.out_degree(node) != 0]
            
            if len(bfs_starts) > 1:
                if node_2_change in bfs_starts:
                    bfs_starts.remove(node_2_change)
            
            if len(bfs_starts) == 0:
                model.update_network()
                return model
            else:
                bfs_start = bfs_starts[0]
                
            edges_to_ct = net.edges_to_subgenome_count(downstream_node, 
                                                       target - cur_ct, 
                                                       bfs_start)
        
            if not edges_to_ct:
                break
                
            random_key = self.random_object([key for key in edges_to_ct.keys()], model.rng)
            if random_key is None:
                break
            
            edge_list = list(edges_to_ct[random_key])
            if not edge_list:
                break
                
            new_edge: Edge = self.random_object(edge_list, model.rng)
            if new_edge is None:
                break
            
            # 4.2: Connect the unconnected node to the new branch
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
            
        # STEP 5: Remove excess nodes
        net.clean()
       
        model.update_network()
        self.same_move_info = copy.deepcopy(net)
    
        return model

    def undo(self, model: Model) -> None:
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

    def same_move(self, model: Model) -> None:
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
         
    def delete_edge(self, net: Network, edge: Edge) -> None:
        """
        Deletes an edge from the network.

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
                # Bubble
                b = copy.copy(net.get_parents(neighbors[1]))
                
                if len(b) != 0:
                    net.remove_edge(Edge(b[0], neighbors[0]))
                    net.remove_edge(Edge(neighbors[0], cur))
                    net.remove_edge(Edge(neighbors[1], cur))
                    net.add_edges(Edge(b[0], cur))
                else:
                    net.remove_edge(Edge(neighbors[0], cur))
                
                return 
            else: 
                i = 1
                b = copy.copy(net.get_parents(neighbors[0])) if neighbors else []
                for neighbor in neighbors:
                    if not bypass or neighbor == edge.src: 
                        net.remove_edge(Edge(neighbor, cur)) 
                        
                        if net.in_degree(neighbor) == 2:
                            q.append(neighbor)
                        else:
                            try:
                                other_children = [node for node in net.get_children(neighbor) if node != cur]
                                if other_children:
                                    a: Node = other_children[0]
                                    
                                    if net.in_degree(neighbor) != 0:
                                        b_nodes: list = net.get_parents(neighbor)
                                        if b_nodes:
                                            b_node: Node = b_nodes[0]
                                            
                                            net.remove_edge(Edge(b_node, neighbor))
                                            net.remove_edge(Edge(neighbor, a))
                                            net.add_edges(Edge(b_node, a))
                            except Exception:
                                if i == 2 and len(b) != 0:
                                    net.remove_edge(Edge(b[0], neighbor))
                    i += 1
                                    
            bypass = False


class SPR(Move):
    """
    A move that performs a Subtree Prune and Regraft operation on a network.
    """
    def __init__(self, debug_id: int = 0) -> None:
        """
        Initializes a move that performs a Subtree Prune and Regraft operation.

        Args:
            debug_id (int): The debug id for the move.
        Returns:
            N/A
        """
        super().__init__()
        self.logger = Logger(str(debug_id))
        self.undo_info = None
        self.same_move_info = None

    def random_object(self, mylist: list, rng: np.random.Generator) -> object:    
        """
        Selects a random object from a list.

        Args:
            mylist (list): The list of objects to select from.
            rng (np.random.Generator): The random number generator.

        Returns:
            object: The randomly selected object.
        """
        if len(mylist) == 0:
            return None
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

        edges = net.E()
        if not edges:
            return model
            
        # Select a random edge to cut
        edge_to_cut: Edge = self.random_object(edges, model.rng)
        if edge_to_cut is None:
            return model
            
        src, dest = edge_to_cut.src, edge_to_cut.dest

        # Remove the selected edge
        net.remove_edge(edge_to_cut)

        # Collect the subtree rooted at dest
        subtree_nodes = net.get_subtree_at(dest) if hasattr(net, 'get_subtree_at') else [dest]
        subtree_edges = net.edges_downstream_of_node(dest) if hasattr(net, 'edges_downstream_of_node') else []

        # Remove the subtree from the network
        for edge in subtree_edges:
            net.remove_edge(edge)
        for node in subtree_nodes:
            net.remove_nodes(node)

        # Select a random edge to reattach the subtree
        remaining_edges = net.E()
        if not remaining_edges:
            model.update_network()
            return model
            
        reattachment_edge: Edge = self.random_object(remaining_edges, model.rng)
        if reattachment_edge is None:
            model.update_network()
            return model
            
        reattachment_src, reattachment_dest = reattachment_edge.src, reattachment_edge.dest

        # Insert the subtree back into the network
        net.add_edges(Edge(reattachment_src, dest))
        for edge in subtree_edges:
            net.add_edges(edge)

        model.update_network()
        self.same_move_info = copy.deepcopy(net)
        return model

    def undo(self, model: Model) -> None: 
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

    def same_move(self, model: Model) -> None:
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

