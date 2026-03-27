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
    PSPP (Parentage-Switching for Polyploid Phylogenetics) move for 
    Infer_MP_Allop. Alters the genetic parentage of a subnetwork while 
    maintaining the same ploidy values for each leaf.
    
    Uses deep-copy for undo/same_move to guarantee correctness.
    """
    def __init__(self, debug_id: int = 0) -> None:
        super().__init__()
        self.valid_attachment_edges: list[Edge] = []
        self.logger = Logger(str(debug_id))
        
    def _random_choice(self, mylist: list, rng: np.random.Generator) -> Any:
        """Select a random element from a list, or None if empty."""
        if not mylist:
            return None
        return mylist[int(rng.integers(0, len(mylist)))]
    
    def execute(self, model: Model) -> Model:
        """
        Executes the PSPP (Switch-Parentage) move.

        Args:
            model (Model): A model object with a populated network field.
        Returns:
            Model: The modified model with a newly proposed network topology.
        """
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)
        self.valid_attachment_edges = []
        
        # STEP 1: Select random non-root node
        non_root_nodes = [node for node in net.V() if node != net.root()]
        if not non_root_nodes:
            model.update_network()
            return model
            
        node_2_change: Node = self._random_choice(non_root_nodes, model.rng)
        if node_2_change is None:
            model.update_network()
            return model
    
        # Skip pointless changes: a child of root whose sibling is a leaf
        node_pars = net.get_parents(node_2_change)
        if len(node_pars) == 1:
            root_node = net.root()
            if node_pars[0] == root_node:
                root_kids = net.get_children(root_node)
                other_kids = [n for n in root_kids if n != node_2_change]
                if other_kids and net.out_degree(other_kids[0]) == 0:
                    model.update_network()
                    return model
            
        # STEP 2: Record target subgenome count before modification
        target: int = net.subgenome_count(node_2_change)
        
        # STEP 3: Remove a parent edge
        in_edges = list(net.in_edges(node_2_change))
        if not in_edges:
            model.update_network()
            return model
            
        edge_2_remove: Edge = self._random_choice(in_edges, model.rng)
        if edge_2_remove is None:
            model.update_network()
            return model
            
        self._delete_parent_edge(net, edge_2_remove)
        
        # STEP 4: Reconnect to achieve target subgenome count
        original_node: Node = node_2_change
        cur_ct = net.subgenome_count(original_node) if net.in_degree(original_node) >= 1 else 0
        is_first_iter = True
        
        for _ in range(100):
            if cur_ct == target:
                break
                
            if not is_first_iter:
                if not self.valid_attachment_edges:
                    break
                branch = self._random_choice(self.valid_attachment_edges, model.rng)
                if branch is None:
                    break
                    
                intermediate_node = net.add_uid_node()
                net.remove_edge(branch)
                self.valid_attachment_edges.remove(branch)
                
                e1 = Edge(branch.src, intermediate_node)
                e2 = Edge(intermediate_node, branch.dest)
                net.add_edges(e1)
                net.add_edges(e2)
                self.valid_attachment_edges.extend([e1, e2])
                downstream_node = intermediate_node
            else:
                downstream_node = original_node
                
            # Find a root for BFS (handle disconnected components)
            bfs_starts = [n for n in net.V()
                          if net.in_degree(n) == 0 and net.out_degree(n) != 0]
            if len(bfs_starts) > 1 and original_node in bfs_starts:
                bfs_starts.remove(original_node)
            if not bfs_starts:
                break
                
            edges_to_ct = net.edges_to_subgenome_count(
                downstream_node, target - cur_ct, bfs_starts[0])
        
            if not edges_to_ct:
                break
                
            random_key = self._random_choice(list(edges_to_ct.keys()), model.rng)
            if random_key is None:
                break
            
            edge_list = list(edges_to_ct[random_key])
            if not edge_list:
                break
                
            new_edge = self._random_choice(edge_list, model.rng)
            if new_edge is None:
                break
            
            # Insert connector node and attach
            connector_node = net.add_uid_node()
            attach_target = downstream_node
            e_to_child = Edge(connector_node, new_edge.dest)
            e_from_parent = Edge(new_edge.src, connector_node)
            e_to_target = Edge(connector_node, attach_target)
            
            net.add_edges([e_to_child, e_from_parent, e_to_target])
            self.valid_attachment_edges.append(e_to_target)
            net.remove_edge(new_edge)
            
            if net.in_degree(attach_target) > 1:
                attach_target.set_is_reticulation(True)
            
            cur_ct = net.subgenome_count(original_node)
            is_first_iter = False
            
        net.clean()
        self._reconcile_reticulation_flags(net)
        model.update_network()
        self.same_move_info = copy.deepcopy(net)
        return model

    def undo(self, model: Model) -> None:
        """Restores the network to its pre-move state."""
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        """Replays the same topology change on a different model instance."""
        if self.same_move_info is not None:
            model.network = copy.deepcopy(self.same_move_info)
        model.update_network()
    
    def hastings_ratio(self) -> float:
        return 1.0
         
    def _reconcile_reticulation_flags(self, net: Network) -> None:
        """
        Set each node's reticulation flag to match its actual in-degree.
        This corrects any stale flags left after topology edits or clean().
        """
        for node in net.V():
            node.set_is_reticulation(net.in_degree(node) > 1)

    def _delete_parent_edge(self, net: Network, edge: Edge) -> None:
        """
        Removes a parent edge from the network and cleans up cascading
        degree-1 nodes and reticulation chains.

        Args:
            net (Network): The network object.
            edge (Edge): The edge to delete (must be an actual edge in net).
        """
        target_node = edge.dest
        parent_node = edge.src

        net.remove_edge(edge)
        
        if target_node.is_reticulation() and net.in_degree(target_node) == 1:
            target_node.set_is_reticulation(False)

        # Clean up parent_node if it became degree-1 internal
        self._cleanup_node(net, parent_node)
    
    def _cleanup_node(self, net: Network, node: Node) -> None:
        """
        Remove or bypass a node that has become topologically degenerate
        after an edge deletion.  Handles:

        - passthrough  (in=1, out=1)  → bypass
        - orphan root  (in=0, out=1)  → remove, disconnect child
        - isolated      (in=0, out=0)  → remove
        - dead-end      (in>=1, out=0, non-leaf)  → remove, recurse parents

        The dead-end case covers reticulation nodes whose only child edge
        was deleted, leaving them with parents but no children.
        """
        in_deg = net.in_degree(node)
        out_deg = net.out_degree(node)
        
        if node == net.root():
            return
        
        if in_deg == 1 and out_deg == 1:
            parent = net.get_parents(node)[0]
            child = net.get_children(node)[0]
            
            parent_edge = net.get_edge(parent, node)
            child_edge = net.get_edge(node, child)
            
            net.remove_edge(parent_edge)
            net.remove_edge(child_edge)
            net.remove_nodes(node)
            net.add_edges(Edge(parent, child))
            
            if parent.is_reticulation() and net.in_degree(parent) == 1:
                parent.set_is_reticulation(False)
                self._cleanup_node(net, parent)
        
        elif in_deg == 0 and out_deg == 1:
            child = net.get_children(node)[0]
            child_edge = net.get_edge(node, child)
            net.remove_edge(child_edge)
            net.remove_nodes(node)
        
        elif in_deg == 0 and out_deg == 0:
            net.remove_nodes(node)

        elif in_deg >= 1 and out_deg == 0:
            # Dead-end internal node: has parent(s) but no children.
            # _cleanup_node only recurses upward through parents, so it
            # never encounters a genuine leaf taxon here.
            parents = list(net.get_parents(node))
            for p in parents:
                net.remove_edge(net.get_edge(p, node))
            node.set_is_reticulation(False)
            net.remove_nodes(node)
            for p in parents:
                if p.is_reticulation() and net.in_degree(p) == 1:
                    p.set_is_reticulation(False)
                self._cleanup_node(net, p)


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
            model.network = copy.deepcopy(self.same_move_info)
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

