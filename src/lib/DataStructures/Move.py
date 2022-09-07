import copy
import random
from abc import ABC, abstractmethod
import numpy as np

from src.lib.DataStructures.ModelGraph import ModelError


class Move(ABC):

    def __init__(self):
        self.model = None

    @abstractmethod
    def execute(self, model):
        """
        Input: model, a Model obj
        Output: a new Model obj that is the result of this operation on model

        """
        pass


class BranchMove(Move, ABC):

    def execute(self, model):

        #Make a copy of the model
        proposedModel = copy.deepcopy(model)

        vector_of_heights = proposedModel.tree_heights.heights()

        selected = random.randint(0, len(proposedModel.network_internals)-1)
        selected_node = proposedModel.network_internals[selected]

        upper_limit = selected_node.get_netparent().get_branch().get()
        children = selected_node.get_children()
        lower_limit = min([children[0].get_branch().get(), children[1].get_branch().get()])





class RootBranchMove(Move, ABC):

    def execute(self, model):
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

        tipwardFreedom = [0, 0]
        tipwardFreedom[0] = (currentRootHeight - leftChildHeight)
        tipwardFreedom[1] = (currentRootHeight - rightChildHeight)

        # the youngest age the species tree root node can be(preserving topologies)
        # Lowest number that can be drawn from exp dist is 0, so we guarantee that the root doesn't encroach on child
        # heights. TODO: test choice of rate param
        uniformShift = np.random.exponential(10) - min(tipwardFreedom)

        # Change the node height of the root in the new model
        proposedModel.change_branch(speciesTreeRoot.get_branch().get_index(), currentRootHeight + uniformShift)

        # TODO: Need this?
        # the log ratio of the density of the proposed over the current species tree root heights
        # fLogHastingsRatio = .1 * uniformShift
        # return fLogHastingsRatio

        # return the slightly modified model
        return proposedModel