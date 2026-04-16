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

    Host edges are split at a random point so their original branch
    length is preserved across the two halves.  The new reticulation
    edge is given a short length drawn from Exp(mean=0.1*min_host_len)
    and both parent edges of the new reticulation node receive
    gamma = 0.5.

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    def __init__(self, max_reticulations: int | None = None) -> None:
        super().__init__()
        self._max_retics = max_reticulations

    def execute(self, model: Model) -> Model:
        net: Network = model.network

        if self._max_retics is not None:
            cur_retics = sum(1 for n in net.V() if n.is_reticulation())
            if cur_retics >= self._max_retics:
                return model

        self.undo_info = copy.deepcopy(net)

        src_e = random.choice(net.E())
        avoid_these_edges = net.edges_upstream_of_node(src_e.src)
        eligible = [e for e in net.E() if e not in avoid_these_edges]
        if not eligible:
            model.update_network()
            return model
        dest_e = random.choice(eligible)

        a: Node = src_e.src
        b: Node = src_e.dest
        x: Node = dest_e.src
        y: Node = dest_e.dest
        len_ab: float = src_e.get_length() or 1.0
        len_xy: float = dest_e.get_length() or 1.0

        z: Node = net.add_uid_node()
        c: Node = net.add_uid_node()
        c.set_is_reticulation(True)

        split_ab = random.random()
        split_xy = random.random()
        retic_len = random.expovariate(1.0 / max(0.1 * min(len_ab, len_xy), 1e-6))

        if a == x and b == y:
            # Bubble
            insert_node_in_edge(net.get_edge(a, b), z, net)
            net.get_edge(a, z).set_length(len_ab * split_ab)
            remaining = len_ab * (1.0 - split_ab)
            net.get_edge(z, b).set_length(remaining)

            insert_node_in_edge(net.get_edge(z, b), c, net)
            tree_zc = net.get_edge(z, c)
            tree_zc.set_length(remaining * split_xy)
            tree_zc.set_gamma(0.5)
            net.get_edge(c, b).set_length(remaining * (1.0 - split_xy))

            net.add_edges(Edge(z, c, length=retic_len, gamma=0.5))
            c.set_is_reticulation(True)
        else:
            # Standard case
            insert_node_in_edge(net.get_edge(x, y), c, net)
            net.get_edge(x, c).set_length(len_xy * split_xy)
            net.get_edge(c, y).set_length(len_xy * (1.0 - split_xy))
            net.get_edge(x, c).set_gamma(0.5)

            insert_node_in_edge(net.get_edge(a, b), z, net)
            net.get_edge(a, z).set_length(len_ab * split_ab)
            net.get_edge(z, b).set_length(len_ab * (1.0 - split_ab))

            retic_edge = Edge(z, c, length=retic_len, gamma=0.5)
            net.add_edges(retic_edge)
            c.set_is_reticulation(True)

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0


class RemoveReticulation(Move):
    """Remove a reticulation node from the network.

    Algorithm:
      1. Pick a random reticulation node c (in=2, out=1).
      2. Randomly choose one of its two parent edges to delete.
      3. Remove that parent edge.  c now has (in=1, out=1) -- suppress c.
      4. The source of the deleted edge may now be degree-2 -- suppress it.

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    def __init__(self) -> None:
        super().__init__()

    def execute(self, model: Model) -> Model:
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        retic_nodes = [n for n in net.V()
                       if n.is_reticulation() and net.in_degree(n) == 2]
        if not retic_nodes:
            return model

        c: Node = random.choice(retic_nodes)
        parents = net.get_parents(c)
        if len(parents) != 2:
            return model

        drop_parent: Node = random.choice(parents)
        drop_edge = net.get_edge(drop_parent, c)

        net.remove_edge(drop_edge)

        c.set_is_reticulation(False)
        _suppress_deg2(net, c)

        _suppress_deg2(net, drop_parent)

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0


class FlipReticulation(Move):
    """Flip the direction of one reticulation edge, keeping degree constraints.

    Picks a random reticulation edge (z -> c) and reverses it to (c -> z),
    provided the result is still a valid DAG and doesn't violate degree
    constraints.  The source ``z`` becomes a reticulation (in-degree 2)
    and ``c`` may lose its reticulation status.

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    def __init__(self) -> None:
        super().__init__()

    def execute(self, model: Model) -> Model:
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        root = net.root()
        retic_edges = [e for e in net.E()
                       if e.dest.is_reticulation() and e.src != root]
        if not retic_edges:
            return model

        retic_edge: Edge = random.choice(retic_edges)
        z: Node = retic_edge.src
        c: Node = retic_edge.dest
        saved_gamma = retic_edge.get_gamma()
        saved_length = retic_edge.get_length()

        net.remove_edge(retic_edge)
        net.add_edges(Edge(c, z, length=saved_length, gamma=saved_gamma))

        if net.in_degree(c) <= 1:
            c.set_is_reticulation(False)
        if net.in_degree(z) > 1:
            z.set_is_reticulation(True)

        if not net.is_acyclic():
            model.network = self.undo_info
            self.undo_info = None
            model.update_network()
            return model

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
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
    """Subtree Prune and Regraft on a phylogenetic network.

    Network-safe algorithm:
      1. Select a random edge (u -> v) to prune, subject to:
         - v is not a leaf
         - v is not a reticulation (we don't detach reticulation children)
         - u is not a reticulation with out-degree 1 (would leave it (2,0))
         - u is not root with out-degree 2 (would leave root with 1 child)
      2. Remove the edge (u -> v).
      3. Repair u: if u becomes (1,1), suppress u (merge incident edges).
      4. Collect subtree nodes rooted at v (used to prevent cycles).
      5. Select a target edge (a -> b) where neither endpoint is in the
         subtree (prevents cycles).  Edges are weighted by inverse
         topological distance from the prune point so that nearby
         regraft locations are strongly preferred.
      6. Subdivide (a -> b) by inserting a new node w:
         a -> w -> b, with split branch lengths.
      7. Attach: w -> v.  w is now (1,2) -- a valid internal node.

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    _DISTANCE_DECAY = 2.0

    def __init__(self, debug_id: int = 0) -> None:
        super().__init__()
        self.undo_info = None
        self.same_move_info = None

    @staticmethod
    def _hop_distances(net: Network, origin: Node,
                       forbidden: set[Node]) -> dict[Node, int]:
        """BFS hop-distance from *origin* ignoring forbidden nodes."""
        dist: dict[Node, int] = {origin: 0}
        q: deque[Node] = deque([origin])
        while q:
            cur = q.popleft()
            d = dist[cur] + 1
            for nbr in list(net.get_children(cur)) + list(net.get_parents(cur)):
                if nbr in forbidden or nbr in dist:
                    continue
                dist[nbr] = d
                q.append(nbr)
        return dist

    def execute(self, model: Model) -> Model:
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        root = net.root()
        prunable = []
        for e in net.E():
            u, v = e.src, e.dest
            if v in net.get_leaves():
                continue
            if v.is_reticulation():
                continue
            if u == root and net.out_degree(root) <= 2:
                continue
            if u.is_reticulation() and net.out_degree(u) <= 1:
                continue
            prunable.append(e)

        if not prunable:
            return model

        prune_edge: Edge = random.choice(prunable)
        prune_parent: Node = prune_edge.src
        subtree_root: Node = prune_edge.dest
        prune_len: float = prune_edge.get_length() or 1.0

        subtree_nodes = net.get_subtree_at(subtree_root)

        net.remove_edge(prune_edge)

        _suppress_deg2(net, prune_parent)

        forbidden = subtree_nodes
        eligible = [e for e in net.E()
                    if e.src not in forbidden and e.dest not in forbidden]
        if not eligible:
            model.network = self.undo_info
            self.undo_info = None
            model.update_network()
            return model

        hop = self._hop_distances(net, prune_parent, forbidden)
        weights: list[float] = []
        for e in eligible:
            d_src = hop.get(e.src, len(hop))
            d_dst = hop.get(e.dest, len(hop))
            d = min(d_src, d_dst) + 1
            weights.append(1.0 / (d ** self._DISTANCE_DECAY))

        target_edge: Edge = random.choices(eligible, weights=weights, k=1)[0]
        a: Node = target_edge.src
        b: Node = target_edge.dest
        target_len: float = target_edge.get_length() or 1.0

        new_node: Node = net.add_uid_node()
        split = random.random()
        insert_node_in_edge(target_edge, new_node, net)
        net.get_edge(a, new_node).set_length(target_len * split)
        net.get_edge(new_node, b).set_length(target_len * (1.0 - split))

        net.add_edges(Edge(new_node, subtree_root, length=prune_len))

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0


################################
#### PARAMETER-LEVEL MOVES #####
################################

def _suppress_deg2(net: Network, node: Node) -> None:
    """Suppress a node that has become a degree-2 passthrough (in=1, out=1).

    The two incident edges are merged into one whose branch length is the
    sum of the originals. Gamma is inherited from the child-side edge.
    """
    if net.in_degree(node) != 1 or net.out_degree(node) != 1:
        return
    parent = net.get_parents(node)[0]
    child = net.get_children(node)[0]
    e_up = net.get_edge(parent, node)
    e_dn = net.get_edge(node, child)
    len_up = e_up.get_length() or 0.0
    len_dn = e_dn.get_length() or 0.0
    new_gamma = e_dn.get_gamma() if e_dn.get_gamma() else e_up.get_gamma()
    net.remove_edge(e_up)
    net.remove_edge(e_dn)
    net.remove_nodes(node)
    net.add_edges(Edge(parent, child, length=len_up + len_dn, gamma=new_gamma))


class ChangeNodeHeight(Move):
    """Slide an internal node up or down by adjusting incident branch lengths.

    A random internal (non-root, non-leaf) node is chosen and shifted by
    a uniform delta.  The valid range is bounded so that no incident
    branch length drops below a small epsilon.
    """

    _EPSILON = 1e-6

    def __init__(self) -> None:
        super().__init__()

    def execute(self, model: Model) -> Model:
        net: Network = model.network

        candidates = [
            n for n in net.V()
            if net.out_degree(n) > 0 and net.in_degree(n) > 0
        ]
        if not candidates:
            model.update_network()
            return model

        node = random.choice(candidates)
        parent_edges = list(net.in_edges(node))
        child_edges = list(net.out_edges(node))

        if not parent_edges or not child_edges:
            model.update_network()
            return model

        min_parent_len = min(e.get_length() for e in parent_edges)
        min_child_len = min(e.get_length() for e in child_edges)

        lower = -(min_parent_len - self._EPSILON)
        upper = min_child_len - self._EPSILON

        if lower >= upper:
            model.update_network()
            return model

        delta = random.uniform(lower, upper)

        edge_changes: list[tuple[str, str, float, float]] = []
        for e in parent_edges:
            old_len = e.get_length()
            new_len = old_len + delta
            edge_changes.append((e.src.label, e.dest.label, old_len, new_len))
            e.set_length(new_len)
        for e in child_edges:
            old_len = e.get_length()
            new_len = old_len - delta
            edge_changes.append((e.src.label, e.dest.label, old_len, new_len))
            e.set_length(new_len)

        self.undo_info = edge_changes
        self.same_move_info = edge_changes
        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        net: Network = model.network
        if self.undo_info is not None:
            for src_lbl, dest_lbl, old_len, _new_len in self.undo_info:
                src = net.has_node_named(src_lbl)
                dest = net.has_node_named(dest_lbl)
                if src is not None and dest is not None:
                    net.get_edge(src, dest).set_length(old_len)
        model.update_network()

    def same_move(self, model: Model) -> None:
        net: Network = model.network
        if self.same_move_info is not None:
            for src_lbl, dest_lbl, _old_len, new_len in self.same_move_info:
                src = net.has_node_named(src_lbl)
                dest = net.has_node_named(dest_lbl)
                if src is not None and dest is not None:
                    net.get_edge(src, dest).set_length(new_len)
        model.update_network()

    def hastings_ratio(self) -> float:
        return 1.0


class ChangeInheritanceProb(Move):
    """Propose a new inheritance probability for a random reticulation node.

    The two parent edges of a reticulation always carry complementary
    gammas (gamma and 1-gamma).  The new value is drawn from a
    Gaussian centered on the current gamma, truncated to (epsilon,
    1-epsilon), so proposals stay close to the current value and
    accept more often than a flat Uniform(0, 1) draw.
    """

    _SIGMA = 0.1
    _EPS = 0.01

    def __init__(self, sigma: float | None = None) -> None:
        super().__init__()
        if sigma is not None:
            self._sigma = sigma
        else:
            self._sigma = self._SIGMA

    def execute(self, model: Model) -> Model:
        net: Network = model.network

        retic_nodes = [
            n for n in net.V() if net.in_degree(n) == 2 and n.is_reticulation()
        ]
        if not retic_nodes:
            model.update_network()
            return model

        node = random.choice(retic_nodes)
        parent_edges = list(net.in_edges(node))
        if len(parent_edges) != 2:
            model.update_network()
            return model

        e1, e2 = parent_edges
        old_g1 = e1.get_gamma()
        old_g2 = e2.get_gamma()

        current = old_g1 if old_g1 is not None else 0.5
        new_g1 = random.gauss(current, self._sigma)
        new_g1 = max(self._EPS, min(1.0 - self._EPS, new_g1))
        e1.set_gamma(new_g1)
        e2.set_gamma(1.0 - new_g1)

        self.undo_info = (e1.src.label, e1.dest.label, old_g1,
                          e2.src.label, e2.dest.label, old_g2)
        self.same_move_info = (e1.src.label, e1.dest.label, new_g1,
                               e2.src.label, e2.dest.label, 1.0 - new_g1)
        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        net: Network = model.network
        if self.undo_info is not None:
            s1, d1, g1, s2, d2, g2 = self.undo_info
            n_s1, n_d1 = net.has_node_named(s1), net.has_node_named(d1)
            n_s2, n_d2 = net.has_node_named(s2), net.has_node_named(d2)
            if n_s1 is not None and n_d1 is not None:
                net.get_edge(n_s1, n_d1).set_gamma(g1)
            if n_s2 is not None and n_d2 is not None:
                net.get_edge(n_s2, n_d2).set_gamma(g2)
        model.update_network()

    def same_move(self, model: Model) -> None:
        net: Network = model.network
        if self.same_move_info is not None:
            s1, d1, g1, s2, d2, g2 = self.same_move_info
            n_s1, n_d1 = net.has_node_named(s1), net.has_node_named(d1)
            n_s2, n_d2 = net.has_node_named(s2), net.has_node_named(d2)
            if n_s1 is not None and n_d1 is not None:
                net.get_edge(n_s1, n_d1).set_gamma(g1)
            if n_s2 is not None and n_d2 is not None:
                net.get_edge(n_s2, n_d2).set_gamma(g2)
        model.update_network()

    def hastings_ratio(self) -> float:
        return 1.0


class ChangeReticSource(Move):
    """Move the source (tail / parent) of a reticulation edge to a new location.

    Picks a random reticulation edge z->c, detaches the source end,
    suppresses z if it becomes degree-2, then reattaches from a
    freshly-inserted node on a randomly chosen edge (avoiding edges
    downstream of c to prevent cycles).

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    def __init__(self) -> None:
        super().__init__()

    def execute(self, model: Model) -> Model:
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        retic_edges = [e for e in net.E() if e.dest.is_reticulation()]
        if not retic_edges:
            model.update_network()
            return model

        retic_edge = random.choice(retic_edges)
        z: Node = retic_edge.src
        c: Node = retic_edge.dest
        saved_gamma = retic_edge.get_gamma()
        saved_length = retic_edge.get_length()

        net.remove_edge(retic_edge)

        if net.in_degree(z) == 1 and net.out_degree(z) == 1:
            _suppress_deg2(net, z)

        forbidden = set(net.edges_downstream_of_node(c))
        forbidden.update(net.in_edges(c))
        forbidden.update(net.out_edges(c))
        eligible = [e for e in net.E() if e not in forbidden]

        if not eligible:
            model.network = self.undo_info
            model.update_network()
            self.undo_info = None
            return model

        host = random.choice(eligible)
        a, b = host.src, host.dest
        z_new = net.add_uid_node()
        host_len = host.get_length()
        split = random.random()

        net.remove_edge(host)
        net.add_edges(Edge(a, z_new, length=host_len * split))
        net.add_edges(Edge(z_new, b, length=host_len * (1.0 - split)))
        net.add_edges(Edge(z_new, c, length=saved_length, gamma=saved_gamma))

        if net.in_degree(c) > 1:
            c.set_is_reticulation(True)

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0


class RelocateReticulation(Move):
    """Simultaneously move both parent edges of a reticulation node.

    Picks a random reticulation node, detaches both incoming edges
    (suppressing any degree-2 passthrough nodes left behind), then
    reattaches them from two randomly chosen edges in the resulting
    network.  This is equivalent to ChangeReticSource + ChangeReticDest
    in a single atomic step, avoiding the deep likelihood valley that
    arises from the intermediate partially-detached state.

    Uses deep-copy for undo to guarantee correctness.
    """

    def __init__(self) -> None:
        super().__init__()

    def execute(self, model: Model) -> Model:
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        retic_nodes = [n for n in net.V() if n.is_reticulation()]
        if not retic_nodes:
            model.update_network()
            return model

        retic = random.choice(retic_nodes)
        in_edges = list(net.in_edges(retic))
        if len(in_edges) != 2:
            model.update_network()
            return model

        children = net.get_children(retic)
        if not children:
            model.update_network()
            return model
        child = children[0]

        e_child = net.get_edge(retic, child)
        saved_child_len = e_child.get_length() if e_child else None

        saved_gammas = [e.get_gamma() for e in in_edges]
        saved_lengths = [e.get_length() for e in in_edges]
        old_parents = [e.src for e in in_edges]

        net.remove_edge(e_child)
        for e in in_edges:
            net.remove_edge(e)

        retic.set_is_reticulation(False)
        net.remove_nodes(retic)

        for p in old_parents:
            if net.in_degree(p) == 1 and net.out_degree(p) == 1:
                _suppress_deg2(net, p)

        if net.in_degree(child) == 1 and net.out_degree(child) == 1:
            if not child.is_reticulation():
                _suppress_deg2(net, child)

        all_edges = list(net.E())
        if len(all_edges) < 2:
            model.network = self.undo_info
            model.update_network()
            self.undo_info = None
            return model

        host1 = random.choice(all_edges)
        a1, b1 = host1.src, host1.dest

        downstream_of_b1 = set(net.edges_downstream_of_node(b1))
        downstream_of_b1.add(host1)
        eligible2 = [e for e in all_edges if e not in downstream_of_b1
                      and e is not host1]

        if not eligible2:
            model.network = self.undo_info
            model.update_network()
            self.undo_info = None
            return model

        host2 = random.choice(eligible2)
        a2, b2 = host2.src, host2.dest

        new_retic = net.add_uid_node()
        new_retic.set_is_reticulation(True)

        z1 = net.add_uid_node()
        h1_len = host1.get_length()
        split1 = random.random()
        net.remove_edge(host1)
        net.add_edges(Edge(a1, z1, length=(h1_len or 0) * split1))
        net.add_edges(Edge(z1, b1, length=(h1_len or 0) * (1 - split1)))

        all_edges_now = list(net.E())
        if host2 not in all_edges_now:
            z2 = net.add_uid_node()
            h2_len = host2.get_length()
            split2 = random.random()
            try:
                net.remove_edge(host2)
            except Exception:
                model.network = self.undo_info
                model.update_network()
                self.undo_info = None
                return model
            net.add_edges(Edge(a2, z2, length=(h2_len or 0) * split2))
            net.add_edges(Edge(z2, b2, length=(h2_len or 0) * (1 - split2)))
        else:
            z2 = net.add_uid_node()
            h2_len = host2.get_length()
            split2 = random.random()
            net.remove_edge(host2)
            net.add_edges(Edge(a2, z2, length=(h2_len or 0) * split2))
            net.add_edges(Edge(z2, b2, length=(h2_len or 0) * (1 - split2)))

        net.add_edges(Edge(z1, new_retic,
                           length=saved_lengths[0], gamma=saved_gammas[0]))
        net.add_edges(Edge(z2, new_retic,
                           length=saved_lengths[1], gamma=saved_gammas[1]))

        net.add_edges(Edge(new_retic, child, length=saved_child_len))

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0


class ChangeReticDest(Move):
    """Move the destination (head) of a reticulation edge to a new location.

    Picks a random reticulation edge z->c, detaches the destination end,
    suppresses c if it becomes degree-2 (it loses its reticulation status),
    then reattaches to a freshly-inserted reticulation node on a randomly
    chosen edge (avoiding edges upstream of z to prevent cycles).

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    def __init__(self) -> None:
        super().__init__()

    def execute(self, model: Model) -> Model:
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        retic_edges = [e for e in net.E() if e.dest.is_reticulation()]
        if not retic_edges:
            model.update_network()
            return model

        retic_edge = random.choice(retic_edges)
        z: Node = retic_edge.src
        c: Node = retic_edge.dest
        saved_gamma = retic_edge.get_gamma()
        saved_length = retic_edge.get_length()

        net.remove_edge(retic_edge)

        if net.in_degree(c) <= 1:
            c.set_is_reticulation(False)
        if net.in_degree(c) == 1 and net.out_degree(c) == 1:
            _suppress_deg2(net, c)

        forbidden = set(net.edges_upstream_of_node(z))
        forbidden.update(net.in_edges(z))
        forbidden.update(net.out_edges(z))
        eligible = [e for e in net.E() if e not in forbidden]

        if not eligible:
            model.network = self.undo_info
            model.update_network()
            self.undo_info = None
            return model

        host = random.choice(eligible)
        a, b = host.src, host.dest
        c_new = net.add_uid_node()
        c_new.set_is_reticulation(True)
        host_len = host.get_length()
        split = random.random()

        net.remove_edge(host)
        net.add_edges(Edge(a, c_new, length=host_len * split))
        net.add_edges(Edge(c_new, b, length=host_len * (1.0 - split)))
        net.add_edges(Edge(z, c_new, length=saved_length, gamma=saved_gamma))

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0

