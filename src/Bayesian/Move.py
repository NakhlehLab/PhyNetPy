from __future__ import annotations
import random
from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ModelGraph import *


class MoveError(Exception):
    def __init__(self, message="Error making a move"):
        self.message = message
        super().__init__(self.message)


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
        first_taxa = first.get_predecessors()[0]
        sec_taxa = second.get_predecessors()[0]

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
        first_taxa = first.get_predecessors()[0]
        sec_taxa = second.get_predecessors()[0]

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


class NetworkTopologyMove(Move):
    
    def execute(self, model):
        return super().execute(model)
    
    def same_move(self, model):
        return super().same_move(model)
    
    def undo(self, model):
        return super().undo(model)
    
    def hastings_ratio(self):
        return super().hastings_ratio()
    
    

