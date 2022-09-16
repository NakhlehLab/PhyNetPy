from Matrix import Matrix
import GTR
import numpy as np
import math
from Bio import AlignIO
from NetworkBuilder import NetworkBuilder


class ProbabilityError(Exception):
    """
    Class that handles any probability exceptions
    """

    def __init__(self, message="Error"):
        self.message = message
        super().__init__(self.message)


class Probability:
    """
        A class for drawing objects from a distribution, or calculating the 
        probability of an object from within a distribution space (given some data
        and hyperparameters)

        network-- A DAG object that represents a tree/network

        model-- A substitution model, default being Jukes Cantor. May be any subtype
                of the GTR class
        
        data-- A Matrix object that is a compressed MSA matrix containing site data
                for each taxa
    """

    def __init__(self, network, model=GTR.JC(), data=None):
        # init inputs
        self.sub = model
        self.data = data
        self.tree = network

        self.transitions = {}
        self.update_transition()

    def setModel(self, sub_model):
        """
        set the substitution model

        subModel-- the new substitution model of subtype GTR
        """
        self.sub = sub_model
        self.update_transition()

    def setData(self, data_matrix):
        """
                Set the data matrix, must be of class type Matrix

                dataMatrix-- a Matrix object
        """
        if type(data_matrix) != Matrix:
            raise ProbabilityError("Tried to set data to type other than Matrix")
        self.data = data_matrix

    def generate(self):
        """
                A method that generates an object based on the model parameters,
                hyperparameters, and sequence data that are passed in
                """
        return 0

    def felsenstein_likelihood(self):
        """
        Using a recursive subroutine, calculate the log likelihood
        of the tree/network given the data and substitution model.

        Works by calling the subroutine on the root of the network,
        and then treats each column/unique site pattern as an IID.

        We take the base frequencies and dot product them with the column likelihoods,
        and then take the log of each column and ADD THEM, since we are dealing with logs.

        Inputs: nothing
        Outputs: the Log Likelihood of the tree/network given the data.

        """
        # call subroutine on the root
        matrix = self.likelihood_helper(self.tree.findRoot()[0])

        # get the base frequencies from the model
        base_freqs = np.array(self.sub.get_hyperparams()[0])
        base_freqs = base_freqs.reshape((4,))

        # tally up the logs of the dot products
        result = 0
        for row in range(np.shape(matrix)[0]):
            logLikelihoodSite = math.log(np.dot(base_freqs, matrix[row]))
            result += logLikelihoodSite

        return result

    def likelihood_helper(self, start_node):
        """
        The recursive helper function for walking the tree
        Returns a matrix with size (siteCount, 4), where each row is the partial likelihood
        for a certain site.

        Input: start_node, the node for which to calculate the matrix of partial likelihoods
        Output: a (siteCount, 4) sized np.array

        """

        # if the node is a leaf node, simply grab the state from the data matrix and
        # transform into array ie. [0,0,1,0] = G
        if self.tree.outDegree(start_node) == 0:

            likelihoods = np.zeros((self.data.siteCount(), 4))

            # Get the sequence that corresponds with the node
            seq = self.data.getSeq(start_node.get_name())

            # Map each character in the sequence to an array of length 4
            for col in range(self.data.siteCount()):
                if seq[col] == "A":
                    row = np.array([1, 0, 0, 0])
                elif seq[col] == "C":
                    row = np.array([0, 1, 0, 0])
                elif seq[col] == "G":
                    row = np.array([0, 0, 1, 0])
                elif seq[col] == "T":
                    row = np.array([0, 0, 0, 1])

                # append row to matrix
                likelihoods[col, :] = row

            # for debugging
            # print("----THIS IS A LEAF----")
            # print(likelihoods)
            # print("-----------------------")
            return likelihoods

        # if not a leaf, aka an internal node. We must combine the child matrices
        children = self.tree.findDirectSuccessors(start_node)

        if len(children) == 1:
            # Is this case necessary/correct?
            return self.likelihood_helper(children[0])
        else:
            # Use the combine function to merge all child partial likelihoods
            return self.combine_child_likelihoods(children)

    def combine_child_likelihoods(self, children):
        """
        Takes a list of child nodes and combines the partial likelihoods via Felsenstein's algo

        Input: List of Node objs
        Output: a (siteCount, 4) sized np.array
        """

        matrices = []
        for child in children:
            # recursively compute the child partial likelihood. Could be another internal node, but could be a leaf
            matrix = self.likelihood_helper(child)

            # compute matrix * Pij transpose. Math explained in doc string
            step1 = np.matmul(matrix, self.transitions[child].transpose())

            # add to list of child matrices
            matrices.append(step1)

        # Element-wise multiply each matrix in the list
        result = np.ones(np.shape(matrices[0]))
        for matrix in matrices:
            result = np.multiply(result, matrix)

        return result

    def update_transition(self, node=None):
        if node is None:
            for uncalculated_node in self.tree.nodes:
                if uncalculated_node.length() is not None:
                    self.transitions[uncalculated_node] = self.sub.expt(uncalculated_node.length())
        else:
            self.transitions[node] = self.sub.expt(node.length())



## TESTS ##

# test network

# n2 = NetworkBuilder(
#     "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\MetroHastingsTests\\raxml.nex")
# n3 = NetworkBuilder(
#     "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\MetroHastingsTests\\truePhylogeny.nex")
#
# test2 = n2.getNetwork(0)
# test3 = n3.getNetwork(0)
#
# msa2 = AlignIO.read(
#     "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\MetroHastingsTests\\raxml.nex",
#     "nexus")
# msa3 = AlignIO.read(
#     "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\MetroHastingsTests\\truePhylogeny.nex", "nexus")
#
#
# data2 = Matrix(msa2)  # default is to use the DNA alphabet
# data3 = Matrix(msa3)
#
# prob2 = Probability(test2, data=data2)
# prob3 = Probability(test3, data=data3)
#
#
# print(prob2.felsenstein_likelihood())
# print(prob3.felsenstein_likelihood())


