""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0
Approved to Release Date : N/A
"""

from __future__ import annotations
from collections import deque
import copy
import random
from abc import ABC, abstractmethod
import numpy as np
from Node import Node
from typing import TYPE_CHECKING
from Graph import DAG
if TYPE_CHECKING:
    from ModelGraph import *


class MoveError(Exception):
    def __init__(self, message="Error making a move"):
        self.message = message
        super().__init__(self.message)


#HELPER FUNCTIONS#

def insert_node_in_edge(edge : list[Node], node : Node, net : DAG) -> None:
    y : Node = edge[0] #parent
    x : Node = edge[1] #child
    
    #Rewire the edges
    ## print(f"EDGE: <{edge[0].get_name()}, {edge[1].get_name()}>")
    net.remove_edge([y , x])
    net.add_edges([y, node])
    net.add_edges([node, x])

    #update parent attributes
    x.remove_parent(y)
    x.add_parent(node)
    node.add_parent(y)
    
def connect_nodes(n1 : Node, n2 : Node, net : DAG) -> None:
    #Add the edge to the network
    net.add_edges([n2, n1])
    
    #update parents
    # # print(n1)
    # print(n2)
    n1.add_parent(n2)
    
    #Check if n1 is now a reticulation
    if len(n1.get_parent(return_all=True))>1:
        n1.set_is_reticulation(True)

#END HELPER FUNCTIONS#



class Move(ABC):
    """
    Abstract superclass for all model move types.

    A move can be executed on a model that is passed in, and makes a reversible, equally likely edit to one
    aspect of the model.
    """

    def __init__(self):
        self.model = None
        self.undo_info = None
        # Same move info needs to be information that is decoupled from the model objects itself
        self.same_move_info = None

    @abstractmethod
    def execute(self, model: Model) -> Model:
        """
        Input: model, a Model obj
        Output: a new Model obj that is the result of this operation on model

        """
        pass

    @abstractmethod
    def undo(self, model: Model):
        pass

    @abstractmethod
    def same_move(self, model: Model):
        pass
    
    @abstractmethod
    def hastings_ratio(self) -> float:
        pass

class UniformBranchMove(Move, ABC):

    def execute(self, model: Model) -> Model:
        """
        Changes either the node height or branch length of a randomly selected node that is not the root.

        Inputs: a Model obj, model
        Outputs: new Model obj that is the result of changing one branch
        """
        # Make a copy of the model

        proposedModel = model

        # Select random internal node
        selected = random.randint(0, len(proposedModel.netnodes_sans_root) - 1)
        selected_node = proposedModel.netnodes_sans_root[selected]

        # Change the branch to a value chosen uniformly from the allowable bounds
        bounds = selected_node.node_move_bounds()
        new_node_height = np.random.uniform(bounds[0], bounds[1])  # Assumes time starts at root and leafs are at max time

        self.undo_info = [selected_node, selected_node.get_branches()[0].get()]
        self.same_move_info = [selected_node.get_branches()[0].get_index(), new_node_height]
        # Update the branch in the model
        proposedModel.change_branch(selected_node.get_branches()[0].get_index(), new_node_height)

        return proposedModel

    def undo(self, model: Model)-> None:
        model.change_branch(self.undo_info[0].get_branches()[0].get_index(), self.undo_info[1])

    def same_move(self, model: Model) -> None:
        model.change_branch(self.same_move_info[0], self.same_move_info[1])
    
    def hastings_ratio(self) -> float:
        return 1.0

class RootBranchMove(Move, ABC):
    
    def __init__(self):
        super().__init__()
        self.exp_param = 1
        self.old_root_height = None
        self.new_root_height = None

    def execute(self, model: Model) -> Model:
        """
        Change the age of the tree by changing the height of the root node.

        Inputs: a Model obj, model
        Outputs: new Model obj that is the result of changing the root age

        """
        # Make a copy of model that is identical
        proposedModel = model

        # get the root and its height
        #TODO: be flexible for snp root
        speciesTreeRoot : FelsensteinInternalNode = proposedModel.felsenstein_root

        currentRootHeight = speciesTreeRoot.get_branches()[0].get()

        children = speciesTreeRoot.get_children()
        if len(children) != 2:
            raise MoveError("NOT A TREE, There are either too many or not enough children for the root")

        # Calculate height that is the closest to the root
        leftChildHeight = children[0].get_branches()[0].get()
        rightChildHeight = children[1].get_branches()[0].get()

        # the youngest age the species tree root node can be(preserving topologies)
        # The lowest number that can be drawn from exp dist is 0, we guarantee that the root doesn't encroach on child
        # heights.
        uniformShift = np.random.exponential(self.exp_param) - min([currentRootHeight - leftChildHeight, currentRootHeight - rightChildHeight])
        
        
        self.undo_info = [speciesTreeRoot, speciesTreeRoot.get_branches()[0].get()]
        self.same_move_info = [speciesTreeRoot.get_branches()[0].get_index(), currentRootHeight + uniformShift]
        self.new_root_height = currentRootHeight + uniformShift
        self.old_root_height = currentRootHeight
        # Change the node height of the root in the new model
        proposedModel.change_branch(speciesTreeRoot.get_branches()[0].get_index(), currentRootHeight + uniformShift)

        # return the slightly modified model
        return proposedModel

    def undo(self, model: Model) -> None:
        model.change_branch(self.undo_info[0].get_branches()[0].get_index(), self.undo_info[1])

    def same_move(self, model: Model) -> None:
        model.change_branch(self.same_move_info[0], self.same_move_info[1])
    
    def hastings_ratio(self) -> float:
        return -1 * self.exp_param * (self.old_root_height - self.new_root_height)

class TaxaSwapMove(Move, ABC):
    #TODO: Figure out why I even need this

    def execute(self, model: Model) -> Model:
        """

        Args:
            model (Model): A model

        Raises:
            MoveError: if there aren't enough taxa to warrant a swap

        Returns:
            Model: An altered model that is the result of swapping around taxa sequences
        """
        # Make a copy of the model
        proposedModel = model

        # Select two random leaf nodes
        net_leaves = proposedModel.get_network_leaves()

        if len(net_leaves) < 3:
            raise MoveError("TAXA SWAP: NOT ENOUGH TAXA")

        indeces = np.random.choice(len(net_leaves), 2, replace=False)
        first = net_leaves[indeces[0]]
        second = net_leaves[indeces[1]]

        # Grab ExtantTaxa nodes
        first_taxa = first.get_model_parents()[0]
        sec_taxa = second.get_model_parents()[0]

        # Swap names and sequences
        first_seq = first_taxa.get_seq()
        sec_seq = sec_taxa.get_seq()
        first_name = first_taxa.get_name()
        sec_name = sec_taxa.get_name()

        self.undo_info = [first_taxa, sec_taxa]
        self.same_move_info = indeces

        # Update the data
        first_taxa.update(sec_seq, sec_name)
        sec_taxa.update(first_seq, first_name)

        return proposedModel

    def undo(self, model: Model) -> None:
        """
        Literally just swap them back
        """
        first_taxa = self.undo_info[0]
        sec_taxa = self.undo_info[1]
        # Swap names and sequences
        first_seq = first_taxa.get_seq()
        sec_seq = sec_taxa.get_seq()
        first_name = first_taxa.get_name()
        sec_name = sec_taxa.get_name()

        # Update the data
        first_taxa.update(sec_seq, sec_name)
        sec_taxa.update(first_seq, first_name)

    def same_move(self, model: Model) -> None:
        net_leaves = model.get_network_leaves()

        indeces = self.same_move_info
        first = net_leaves[indeces[0]]
        second = net_leaves[indeces[1]]

        # Grab ExtantTaxa nodes
        first_taxa = first.get_model_parents()[0]
        sec_taxa = second.get_model_parents()[0]

        # Swap names and sequences
        first_seq = first_taxa.get_seq()
        sec_seq = sec_taxa.get_seq()
        first_name = first_taxa.get_name()
        sec_name = sec_taxa.get_name()

        # Update the data
        first_taxa.update(sec_seq, sec_name)
        sec_taxa.update(first_seq, first_name)

    def hastings_ratio(self) -> None:
        return 1.0

#THIS IS NNI
class TopologyMove(Move):
    
    def __init__(self):
        super().__init__()
        self.legal_forward_moves = None
        self.legal_backwards_moves = None

    def execute(self, model: Model) -> Model:
        proposedModel = model

        valid_focals = {}

        for n in proposedModel.internal:
            par = n.get_parent()
            children = par.get_children()
            if children[1] == n:
                s = children[0]
            else:
                s = children[1]

            if s.get_branches()[0].get() < n.get_branches()[0].get():
                chosen_child = n.get_children()[random.randint(0, 1)]
                valid_focals[n] = [s, par, chosen_child]

        if len(list(valid_focals.keys())) != 0:
            self.legal_forward_moves = len(list(valid_focals.keys()))
            choice = random.choice(list(valid_focals.keys()))

            relatives = valid_focals[choice]
            self.undo_info = [choice, relatives]
            relatives[2].unjoin(choice)  # disconnect c1 from n
            relatives[0].unjoin(relatives[1])  # disconnect s from par
            relatives[2].join(relatives[1])  # connect c1 to par
            relatives[0].join(choice)  # connect s to n

            # No need to change branches, the right branches are already pointed
            # at the right nodes

            # mark each of c1 and choice as needing updating
            relatives[0].upstream()
            relatives[2].upstream()
        else:
            raise MoveError("DID NOT FIND A LEGAL TOPOLOGY MOVE")
        
        
        # Calculate legal backwards moves for hastings ratio
        valid_focals2 = {}
        for n in proposedModel.internal:
            par = n.get_parent()
            children = par.get_children()
            if children[1] == n:
                s = children[0]
            else:
                s = children[1]

            if s.get_branches()[0].get() < n.get_branches()[0].get():
                chosen_child = n.get_children()[random.randint(0, 1)]
                valid_focals2[n] = [s, par, chosen_child]

        self.legal_backwards_moves = len(list(valid_focals2.keys()))
        if self.legal_backwards_moves == 0:
            raise MoveError("ENTERED INTO STATE WHERE THERE ARE NO MORE LEGAL TOPOLOGY MOVES")
        
        return proposedModel

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            relatives = self.undo_info[1]
            choice = self.undo_info[0]

            # Do the reverse operations
            relatives[2].join(choice)  # connect c1 back to n
            relatives[0].join(relatives[1])  # connect s back to par
            relatives[2].unjoin(relatives[1])  # disconnect c1 from par
            relatives[0].unjoin(choice)  # disconnect s from n

            # mark each of c1 and choice as needing updating
            relatives[0].upstream()
            relatives[2].upstream()

    def same_move(self, model: Model) -> None:
        if self.undo_info is not None:
            relatives = self.undo_info[1]
            choice = self.undo_info[0]

            relatives_model = [None, None, None]
            node_names = {node.get_name(): node for node in relatives}
            choice_model = None
            choice_name = choice.get_name()

            # Use names to map this model instances nodes to the proposed_model nodes
            netnodes = model.netnodes_sans_root
            
            #TODO: be flexible for SNP stuff
            netnodes.append(model.felsenstein_root)  # include root???????
            for node in netnodes:
                if node.get_name() in node_names.keys():
                    index = relatives.index(node_names[node.get_name()])
                    relatives_model[index] = node
                if node.get_name() == choice_name:
                    choice_model = node

            # Do the operations
            relatives_model[2].unjoin(choice_model)  # disconnect c1 from n
            relatives_model[0].unjoin(relatives_model[1])  # disconnect s from par
            relatives_model[2].join(relatives_model[1])  # connect c1 to par
            relatives_model[0].join(choice_model)

            # mark each of c1 and choice as needing updating
            relatives_model[0].upstream()
            relatives_model[2].upstream()
    
    def hastings_ratio(self) -> float:
        return self.legal_forward_moves / self.legal_backwards_moves
 
    
####GRAPH MOVES####
"""
ALL OF THE FOLLOWING NETWORK MOVES HAVE VARIABLE NAMES THAT ARE BASED OFF OF THIS BASIC NETWORK STRUCTURE:

                  y
                    \
                     \  
                      \
            b          z  
           / \        / \
          /   \      /   \
         /     \    /     \
        /        c         \
       /         |          \
      /          |           \
                 a            x

"""


class AddReticulation(Move):
    def __init__(self):
        super().__init__()

    
    def execute(self, model : Model) -> Model:
        """
        Adds a reticulation edge to the network
        """
        net : DAG = model.network
        print("-----BEFORE MOVE-----")
        net.pretty_print_edges()
        E_set = [item for item in net.edges] #.append((net.root()[0], None)) # add (root, null) to edge set #TODO: ERROR
        # Select 2 perhaps non-distinct edges to connect with a reticulation edge-- we allow bubbles
        random_edges = [random.choice(E_set), random.choice(E_set)]
        
        a : Node = random_edges[0][1]
        b : Node = random_edges[0][0]
        y : Node = random_edges[1][0]
        x : Node = random_edges[1][1]
        
        # print(f"Edge 1: <{b.get_name()}, {a.get_name()}")
        # print(f"Edge 2: <{y.get_name()}, {x.get_name()}")
        
        
        z : Node = Node() #, parent_nodes=[b], branch_len={b:[top_bl_1]})
        c : Node = Node(is_reticulation=True) #, parent_nodes=[retic_bot_parent, z], branch_len={retic_bot_parent:[top_bl_2], retic_top:[]})
        
        
        if a!=x or b!=y: #Not a bubble
            net.add_uid_node(z)
            net.add_uid_node(c)
            
            insert_node_in_edge([b, a], c, net)
            # print("DOING NEXT INSERT")
            insert_node_in_edge([y, x], z, net)
            # print("NOW CONNECTING NODES")
            connect_nodes(c, z, net)
            
            
            self.undo_info = [c, z, a, b, x, y]
            self.same_move_info = [node.get_name() for node in self.undo_info]
        
        print("-----AFTER MOVE-----")
        net.pretty_print_edges()
        #net.print_graph()
        model.update_network()
        #Not handling branch lengths/bubbles at this time
        return model

    def undo(self, model : Model)-> None:
        net : DAG = model.network
        print("--------UNDOING MOVE------")
        if self.undo_info is not None:
            
            #undo parent settings
            self.undo_info[2].remove_parent(self.undo_info[0]) # c is not a's parent anymore
            self.undo_info[4].remove_parent(self.undo_info[1]) # z is not x's parent anymore
            
            #essentially RemoveRetic
            edges_2_remove = [[self.undo_info[1], self.undo_info[0]],
                            [self.undo_info[3], self.undo_info[0]],
                            [self.undo_info[0], self.undo_info[2]],
                            [self.undo_info[5], self.undo_info[1]],
                            [self.undo_info[1], self.undo_info[4]]]
            
            for edge in edges_2_remove:
                net.remove_edge(edge)
            
            net.remove_node(self.undo_info[0])
            net.remove_node(self.undo_info[1])
            connect_nodes(self.undo_info[2], self.undo_info[3], net)
            connect_nodes(self.undo_info[4], self.undo_info[5], net)
            
        net.pretty_print_edges()
        model.update_network()

    def same_move(self, model : Model) -> None:
        if type(model.network) is DAG:
            net : DAG = model.network
        else:
            net : DAG = model.network.get()
            
        if self.same_move_info is not None:
            nodes : list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            
            #C and Z do not exist in the other network, just copy the names into new nodes
            a : Node = nodes[2]
            b : Node = nodes[3]
            x : Node = nodes[4]
            y : Node = nodes[5]
            c : Node = Node(name=self.same_move_info[0])
            z : Node = Node(name=self.same_move_info[1])
            
            insert_node_in_edge([b, a], c, net)
            insert_node_in_edge([y, x], z, net)
            connect_nodes(c, z, net)
            net.add_nodes(z)
            net.add_nodes(c)
        
        model.update_network()
    
    def hastings_ratio(self) -> float:
        return 1.0
    
class RemoveReticulation(Move):
    def __init__(self):
        super().__init__()
    
    
    def execute(self, model : Model) -> DAG:
        """
        Removes a reticulation edge from the network
        """
        net : DAG = model.network
        # net.print_graph()
        net.pretty_print_edges()
        #Select a random reticulation edge to remove
        retic_edge : list[Node] = random.choice([edge for edge in net.edges if edge[1].is_reticulation()])
        
        c : Node = retic_edge[1]
        z : Node = retic_edge[0]
        
        
        #In all 4 cases, c/z both have exactly one parent and one child after removal of the retic edge
        a : Node = net.get_children(c)[0] 
        b : Node = [node for node in c.get_parent(return_all=True) if node != z][0]
        x : Node = net.get_children(z)[0] 
        y : Node = z.get_parent()
        
        if a!=x or b!=y: #Not a bubble
            a.remove_parent(c)
            x.remove_parent(z)
            
            net.remove_edge([z, c])
            net.remove_node(c, True)
            net.remove_node(z, True)
            
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
            
            self.undo_info = [c, z, a, b, x, y]
            self.same_move_info = [node.get_name() for node in self.undo_info]
        
        #Not handling branch lengths/bubbles at this time
        net.pretty_print_edges()
        #net.print_graph()
        model.update_network()
        return model

    def undo(self, model : Model)-> None:
        net : DAG = model.network
        if self.undo_info is not None:
            net.add_nodes(self.undo_info[0])
            net.add_nodes(self.undo_info[1])
            insert_node_in_edge([self.undo_info[3], self.undo_info[2]], self.undo_info[0], net) # c into (a,b)
            insert_node_in_edge([self.undo_info[5], self.undo_info[4]], self.undo_info[1], net) # z into (x,y)
            connect_nodes(self.undo_info[0], self.undo_info[1], net) # connect c to z
        
        #net.print_graph()
        net.pretty_print_edges()
        model.update_network()

    def same_move(self, model : Model) -> None:
        net : DAG = model.network
        if self.same_move_info is not None:
            nodes : list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            c : Node = nodes[0]
            z : Node = nodes[1]
            a : Node = nodes[2]
            b : Node = nodes[3]
            x : Node = nodes[4]
            y : Node = nodes[5]
            
            net.remove_edge([z, c])
            net.remove_node(c, True)
            net.remove_node(z, True)
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
        
        model.update_network()
    
    def hastings_ratio(self) -> float:
        return 1.0
    
class RelocateReticulationSource(Move):
    def __init__(self):
        super().__init__()
    
    
    def execute(self, model : Model) -> DAG:
        """
        Removes a reticulation edge from the network
        """
        net : DAG = model.network
        #net.print_graph()
        #net.pretty_print_edges()
        
        #Select a random reticulation edge to relocate
        retic_edge : list[Node] = random.choice([edge for edge in net.edges if edge[1].is_reticulation()])
        
    
        c : Node = retic_edge[1]
        z : Node = retic_edge[0]
        x : Node = [node for node in net.get_children(z) if node != c][0]
        y : Node = z.get_parent()
       
        #Remove edge destination
        net.remove_node(z, True) 
        x.remove_parent(z)
        connect_nodes(x, y, net)
        
        #Select a new edge
        edge_set : list[list[Node]] = [edge for edge in net.edges if edge[1].is_reticulation() == False]
        new_edge : list[Node] = random.choice(edge_set)
        #print(f"NEW EDGE: {[node.get_name() for node in new_edge]}")
        #Add new destination and reconnect c and z
        net.add_nodes(z)
        insert_node_in_edge(new_edge, z, net)
        connect_nodes(c, z, net)
        
        self.undo_info = [c, z, x, y, new_edge[1], new_edge[0]]
        self.same_move_info = [node.get_name() for node in self.undo_info]
        
        #Not handling branch lengths/bubbles at this time
        #net.pretty_print_edges()
        #net.print_graph()
        model.update_network()
        return model

    def undo(self, model : Model)-> None:
        net : DAG = model.network
        if self.undo_info is not None:
            c : Node = self.undo_info[0]
            z : Node = self.undo_info[1]
            x : Node = self.undo_info[2]
            y : Node = self.undo_info[3]
            a : Node = self.undo_info[4]
            b : Node = self.undo_info[5]
            
            #restore old edge
            net.remove_node(z, True)
            a.remove_parent(z)
            connect_nodes(a, b, net)
            net.add_nodes(z)
            connect_nodes(c, z, net)
            
            #set z's parent back to y
            z.set_parent([y])
            
            #insert z back into the x, y edge
            insert_node_in_edge([y, x], z, net)
        
        model.update_network()
        net.print_graph()
        
        

    def same_move(self, model : Model) -> None:
        net : DAG = model.network
        if self.same_move_info is not None:
            nodes : list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            c : Node = nodes[0]
            z : Node = nodes[1]
            a : Node = nodes[2]
            b : Node = nodes[3]
            x : Node = nodes[4]
            y : Node = nodes[5]
            
            #Remove edge destination
            net.remove_node(z, True) 
            connect_nodes(x, y, net)
            
            #Add new destination and reconnect c and z
            insert_node_in_edge([b, a], z, net)
            connect_nodes(c, z, net)
    
    def hastings_ratio(self) -> float:
        return 1.0
    
class RelocateReticulationDestination(Move):
    def __init__(self):
        super().__init__()
    

    def execute(self, model : Model) -> DAG:
        """
        Removes a reticulation edge from the network
        """
        net : DAG = model.network
        #net.print_graph()
        #Select a random reticulation edge to relocate
        retic_edge : tuple[Node] = random.choice([edge for edge in net.edges if edge[1].is_reticulation()])
        
        c : Node = retic_edge[1]
        z : Node = retic_edge[0]
        
        #In all 4 cases, c/z both have exactly one parent and one child after removal of the retic edge
        a : Node = net.get_children(c)[0]
        b : Node = [node for node in c.get_parent(return_all=True) if node != z][0]
       
        
        #Remove edge src
        c.remove_parent(b)
        a.remove_parent(c)
        net.remove_node(c, True) 
        connect_nodes(a, b, net)
        
        #Select a new edge
        edge_set : list[tuple[Node]] = [edge for edge in net.edges if edge[1].is_reticulation() == False]
        new_edge : tuple[Node] = random.choice(edge_set)
        
        #Add new destination and reconnect c and z
        net.add_nodes(c)
        insert_node_in_edge(new_edge, c, net)
        connect_nodes(c, z, net)
        
        self.undo_info = [c, z, a, b, new_edge[1], new_edge[0]]
        self.same_move_info = [node.get_name() for node in self.undo_info]
        
        #Not handling branch lengths/bubbles at this time
        #net.print_graph()
        model.update_network()
        return model

    def undo(self, model : Model)-> None:
        net : DAG = model.network
        if self.undo_info is not None:
            c : Node = self.undo_info[0]
            z : Node = self.undo_info[1]
            a : Node = self.undo_info[2]
            b : Node = self.undo_info[3]
            x : Node = self.undo_info[4]
            y : Node = self.undo_info[5]
        
            # y : Node = [node for node in c.get_parent(return_all=True) if node.get_name() != z.get_name()][0]
            # x : Node = net.get_children(c)[0]
            
            #restore current edge
            net.remove_node(c, True)
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
        net : DAG = model.network
        if self.same_move_info is not None:
            nodes : list[Node] = [net.has_node_named(nodename) for nodename in self.same_move_info]
            c : Node = nodes[0]
            z : Node = nodes[1]
            a : Node = nodes[2]
            b : Node = nodes[3]
            x : Node = nodes[4]
            y : Node = nodes[5]
            
            #Remove edge src
            net.remove_node(c, True) 
            a.remove_parent(c)
            connect_nodes(a, b, net)
            
            #Add new destination and reconnect c and z
            net.add_nodes(c)
            insert_node_in_edge([y, x], c, net)
            connect_nodes(c, z, net)
        
        model.update_network() 
        
    def hastings_ratio(self) -> float:
        return 1.0
     
class RelocateReticulation(Move):
    def __init__(self):
        super().__init__()
    

    
    def execute(self, model : Model) -> DAG:
        """
        Removes a reticulation edge from the network
        """
        net : DAG = model.network
    
        #Select a random reticulation edge to remove
        retic_edge : tuple[Node] = random.choice([edge for edge in net.edges if edge[1].is_reticulation()])
        
        c : Node = retic_edge[1]
        z : Node = retic_edge[0]
        a : Node = net.get_children(c)[0] 
        b : Node = [node for node in c.get_parent(return_all=True) if node != z][0]
        x : Node = [node for node in net.get_children(z) if node != c][0]
        y : Node = z.get_parent()

       
        if a!=x or b!=y: #Not a bubble
            net.remove_node(c, True)
            net.remove_node(z, True)
            x.remove_parent(z)
            a.remove_parent(c)
            c.set_parent([])
            z.set_parent([])
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
            
            E_set = [item for item in net.edges if item != [z, c] and item != [b, c]] #.append((net.root()[0], None)) # add (root, null) to edge set
            # Select 2 perhaps non-distinct edges to connect with a reticulation edge-- we allow bubbles
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
            self.same_move_info = [node.get_name() for node in self.undo_info]
        
        #Not handling branch lengths/bubbles at this time
        model.update_network()
        return model

    def undo(self, model : Model)-> None:
        net : DAG = model.network
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
            
            net.remove_node(c, True)
            net.remove_node(z, True)
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
        net : DAG = model.network
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
            
            net.remove_node(c, True)
            net.remove_node(z, True)
            
            connect_nodes(a, b, net)
            connect_nodes(x, y, net)
            
            insert_node_in_edge((l, m), c, net)
            insert_node_in_edge((n, o), z, net)
            
            net.add_nodes(c)
            net.add_nodes(z)
        model.update_network()
        
    def hastings_ratio(self) -> float:
        return 1.0
        
class FlipReticulation(Move):
    def __init__(self):
        super().__init__()
    
    
    def execute(self, model : Model) -> DAG:
        """
        Removes a reticulation edge from the network
        """
        net : DAG = model.network
        #net.print_graph()
        #Select a random reticulation edge to remove
        retic_edge : tuple[Node] = random.choice([edge for edge in net.edges if edge[1].is_reticulation()])
        
        c : Node = retic_edge[1]
        z : Node = retic_edge[0]
       
        net.remove_edge([z, c])
        net.add_edges([c, z])
        
        c.remove_parent(z)
        z.add_parent(c)
        c.set_is_reticulation(False)
        z.set_is_reticulation(True)
        
        self.undo_info = [c, z]
        self.same_move_info = [c.get_name(), z.get_name()]
        
        #Not handling branch lengths/bubbles at this time
        #net.print_graph()
        model.update_network()
        return model

    def undo(self, model : Model)-> None:
        net : DAG = model.network
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
        net : DAG = model.network
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
        return 1.0
    


class SwitchParentage(Move):
    def __init__(self):
        super().__init__()
        
        #STEP 0: Set up edge/node tracking
        self.added_edges : list[list[Node]] = list()
        self.valid_attachment_edges : list[list[Node]] = list()
        self.added_nodes : set[Node] = set()
        self.removed_nodes : set[Node] = set()
        self.removed_edges : set[list[Node]] = set()
        
        self.print_net = False
        
    def random_object(self, mylist, rng):
        if len(mylist) == 0:
            raise MoveError("sigh")
        rand_index = rng.integers(0, len(mylist))
        return mylist[rand_index]

    
    def execute(self, model : Model) -> Model:
        """
        Executes the Swap-Parentage Move, described in detail at https://phylogenomics.rice.edu/tutorials.html

        Args:
            model (Model): A model object, for which there must be a populated network field 

        Raises:
            MoveError : Aborts move if something irrecoverably wrong happens trying to make the move. Errors should not 
            happen and these will be removed in production after things are fully tested.

        Returns:
            Model: A modified model, with a newly proposed network topology
            
        """
        net : DAG = model.network
        self.undo_info = copy.deepcopy(net)
                
        #STEP 1: Select random non-root node
        node_2_change : Node = self.random_object([node for node in net.nodes if node != net.root()[0]], model.rng)
    
        # STEP 1b: Disallow pointless changes 
        node_pars = net.get_parents(node_2_change)
        
        if len(node_pars) == 1:
            root_node = net.root()[0]
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
        edge_2_remove : list[Node] = self.random_object(net.in_edges(node_2_change), model.rng)
        self.delete_edge(net, edge_2_remove)
        
        is_first_iter = True
        
        #STEP 4: Create new edges/parents
        if net.in_degree(node_2_change) == 1:
            cur_ct = net.subgenome_count(node_2_change)
        else:
            #We severed the only path out of the node
            cur_ct = 0
       
       
        while cur_ct != target:
            
            # 4.0: Select the next edge to branch from
            if not is_first_iter:
                branch : list[Node] = self.random_object(list(self.valid_attachment_edges), model.rng)
                ## print(branch)
                node_2_change = Node()
                net.add_uid_node(node_2_change)
                net.remove_edge(branch)
                self.valid_attachment_edges.remove(branch)
                net.add_edges([[branch[0], node_2_change], [node_2_change, branch[1]]], as_list=True)
                self.valid_attachment_edges.append([branch[0], node_2_change])
                self.valid_attachment_edges.append([node_2_change, branch[1]])
                downstream_node : Node =  node_2_change #branch[1]
            else:
                downstream_node = node_2_change
                
            # 4.1 : Select an edge with a key of <= cur_ct and that wont create a cycle (ensured by edges_2_subgct)
            bfs_starts = [node for node in net.nodes if net.in_degree(node) == 0 and net.out_degree(node) != 0]
            
            if len(bfs_starts)>1:
                if node_2_change in bfs_starts:
                    bfs_starts.remove(node_2_change)
                else:
                    raise MoveError("hmmm idk man")
            
            if len(bfs_starts) == 0:
                print(f"SOMETHING FUNKY : {node_2_change.get_name()}")
                net.print_graph()
                model.update_network()
                return model
            else:
                bfs_start = bfs_starts[0]
                
            edges_to_ct : dict[int, set] = net.edges_to_subgenome_count(downstream_node, target - cur_ct, bfs_start)
            

        
            random_key = self.random_object([key for key in edges_to_ct.keys()], model.rng)
            
            try:
                new_edge : list[Node] = self.random_object(list(edges_to_ct[random_key]), model.rng)
                if new_edge[1] == downstream_node:
                    #print("MAKING A BUBBLE-- THEORETICALLY")
                    self.print_net = True
            except:
                raise MoveError("No edges with a sufficiently low/exact amount")
            
            
            # 4.2 : Connect the unconnected node to the new branch selected in 4.1
            connector_node = Node()
            net.add_uid_node(connector_node)
            new_edge_list = [[connector_node, new_edge[1]], [new_edge[0], connector_node], [connector_node, node_2_change]]
            net.add_edges(new_edge_list, as_list = True)
            self.valid_attachment_edges.append(new_edge_list[2])
            net.remove_edge(new_edge)
            
            
            if len(net.root()) > 1:
                raise MoveError("OOPS, more than one root")
            
            cur_ct = net.subgenome_count(node_2_change)
        
            is_first_iter = False
            
        #STEP 5: Remove excess nodes created by initial edge removal if they exist
        net.remove_excess_branch()
        net.remove_floaters()
        net.prune_excess_nodes()
        
        
        if changing_retic:
            # print(net.newick())
            # net.pretty_print_edges()
            # net.print_graph()
            changing_retic = False
       
        model.update_network()
        
        self.same_move_info = copy.deepcopy(net)
        
        return model

    def undo(self, model : Model)-> None:
        
        if self.undo_info is not None:
            
            model.network = self.undo_info
    
        model.update_network()

    def same_move(self, model : Model) -> None:
            
        if self.same_move_info is not None:
            model.network = self.same_move_info
            
        model.update_network()
    
    def hastings_ratio(self) -> float:
        return 1.0
    
            
    def delete_edge(self, net:DAG, edge:list[Node]):
        root = edge[1]

        
        q = deque()
        q.appendleft(root)

        if net.in_degree(root) == 2:
            bypass = True
        else:
            bypass = False
            
        while len(q) != 0:
            cur = q.pop()
            
            neighbors = copy.copy(net.get_parents(cur))
            
            if len(neighbors) == 2 and neighbors[0].get_name() == neighbors[1].get_name() and bypass:
                #Bubble
                b = copy.copy(net.get_parents(neighbors[1]))
                
                if len(b) != 0:
                    net.remove_edge([b[0], neighbors[0]])
                    net.remove_edge([neighbors[0], cur])
                    net.remove_edge([neighbors[1], cur])
                    net.add_edges([b[0], cur])
                else:
                    net.remove_edge([neighbors[0], cur])
                
                return 
            else:
                i = 1
                b = copy.copy(net.get_parents(neighbors[0]))
                for neighbor in neighbors:
                    
                    if not bypass or neighbor == edge[0]:
                        net.remove_edge([neighbor, cur])
                        
                        if net.in_degrees[neighbor] == 2:
                            q.append(neighbor)
                        else:
                            try:
                                a : Node = [node for node in net.get_children(neighbor) if node != cur][0]
                                
                                if net.in_degree(neighbor) != 0:
                                    b : Node = net.get_parents(neighbor)[0] #tree node will only have 1
                                    
                                    redundant_tree_edge1 = [b, neighbor]
                                    redundant_tree_edge2 = [neighbor, a]

                                    net.remove_edge(redundant_tree_edge1)
                                    net.remove_edge(redundant_tree_edge2)

                                    net.add_edges([b, a])
                            except:
                                if i == 2:
                                    if len(b) != 0:
                                        net.remove_edge([b[0], neighbor])
                    i += 1
                                    
                    
            bypass = False
            
        
    