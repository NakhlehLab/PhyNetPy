import copy
import random
from abc import ABC, abstractmethod
import numpy as np

from ModelGraph import ModelError


class Move(ABC):
    """
    Abstract superclass for all model move types.

    A move can be executed on a model that is passed in, and makes a reversible, equally likely edit to one
    aspect of the model.
    """

    def __init__(self):
        self.model = None

    @abstractmethod
    def execute(self, model):
        """
        Input: model, a Model obj
        Output: a new Model obj that is the result of this operation on model

        """
        pass


class UniformBranchMove(Move, ABC):

    def execute(self, model):
        """
        Changes either the node height or branch length of a randomly selected node that is not the root.

        Inputs: a Model obj, model
        Outputs: new Model obj that is the result of changing one branch
        """
        # Make a copy of the model
        proposedModel = copy.deepcopy(model)

        # Select random internal node
        selected = random.randint(0, len(proposedModel.netnodes_sans_root) - 1)
        selected_node = proposedModel.netnodes_sans_root[selected]

        # Change the branch to a value chosen uniformly from the allowable bounds
        bounds = selected_node.node_move_bounds()
        new_node_height = np.random.uniform(bounds[0],
                                            bounds[1])  # Assumes time starts at root and leafs are at max time

        # Update the branch in the model
        proposedModel.change_branch(selected_node.get_branch().get_index(), new_node_height)

        return proposedModel


class RootBranchMove(Move, ABC):

    def execute(self, model):
        """
        Change the age of the tree by changing the height of the root node.

        Inputs: a Model obj, model
        Outputs: new Model obj that is the result of changing the root age

        """
        # Make a copy of model that is identical
        proposedModel = copy.deepcopy(model)

        # get the root and its height
        speciesTreeRoot = proposedModel.felsenstein_root
        currentRootHeight = speciesTreeRoot.get_branch().get()

        children = speciesTreeRoot.network_children
        if len(children) != 2:
            raise ModelError("NOT A TREE, There are either too many or not enough children for the root")

        # Calculate height that is the closest to the root
        leftChildHeight = children[0].get_branch().get()
        rightChildHeight = children[1].get_branch().get()

        # the youngest age the species tree root node can be(preserving topologies)
        # Lowest number that can be drawn from exp dist is 0, so we guarantee that the root doesn't encroach on child
        # heights.
        uniformShift = np.random.exponential(10) - min([currentRootHeight - leftChildHeight, currentRootHeight - rightChildHeight])

        # Change the node height of the root in the new model
        proposedModel.change_branch(speciesTreeRoot.get_branch().get_index(), currentRootHeight + uniformShift)

        # return the slightly modified model
        return proposedModel


class TaxaSwapMove(Move, ABC):

    def execute(self, model):
        """
        
        """
        # Make a copy of the model
        proposedModel = copy.deepcopy(model)

        # Select two random leaf nodes
        net_leaves = proposedModel.get_network_leaves()

        if len(net_leaves) < 3:
            raise ModelError("NOT ENOUGH TAXA")

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

        # Update the data
        first_taxa.update(sec_seq, sec_name)
        sec_taxa.update(first_seq, first_name)

        return proposedModel


class TopologyMove(Move):

    def execute(self, model):
        proposedModel = copy.deepcopy(model)

        valid_focals = {}

        for n in proposedModel.internal:
            par = n.get_parent()
            children = par.get_children()
            if children[1] == n:
                s = children[0]
            else:
                s = children[1]

            if proposedModel.as_length or s.get_branch().get() < n.get_branch().get():
                chosen_child = n.get_children()[random.randint(0, 1)]
                valid_focals[n] = [s, par, chosen_child]

        choice = random.choice(list(valid_focals.keys()))

        relatives = valid_focals[choice]
        relatives[2].unjoin(choice)  # disconnect c1 from n
        relatives[0].unjoin(relatives[1])  # disconnect s from par
        relatives[2].join(relatives[1])  # connect c1 to par
        relatives[0].join(choice)  # connect s to n

        # No need to change branches, the right branches are already pointed
        # at the right nodes

        # mark each of c1 and choice as needing updating
        relatives[0].upstream()
        relatives[2].upstream()

        return proposedModel
