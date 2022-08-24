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
        self.cache = {}

    def setModel(self, sub_model):
        """
        set the substitution model

        subModel-- the new substitution model of subtype GTR
        """
        self.sub = sub_model

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
            step1 = np.matmul(matrix, self.sub.expt(child.__len__()).transpose())

            # add to list of child matrices
            matrices.append(step1)

        # Element-wise multiply each matrix in the list
        result = np.ones(np.shape(matrices[0]))
        for matrix in matrices:
            result = np.multiply(result, matrix)

        return result

    # def likelihood(self):
    #     """
    #             computes the likelihood of a data structure given the data and
    #             a substitution model
    #
    #             For now, this is felsensteins algorithm, which only works on DNA
    #             and trees that lack reticulations/hybridizations.
    #
    #             Outputs: The log likelihood probability P(Tree | Data & Substitution model)
    #
    #             """
    #
    #     # need data
    #     if self.data == None:
    #         raise ProbabilityError("Attempted to run likelihood computation on empty data source")
    #
    #     # compute the likelihood for each column in the data matrix
    #     likelihoods = np.zeros((self.data.siteCount(), 4))
    #     for colIndex in range(self.data.siteCount()):
    #         likelihoods[colIndex] = self.computeColumnLikelihood(colIndex, self.tree.findRoot()[0]).reshape((4,))
    #
    #     # multiply the likelihoods together, via the assumption that each site
    #     # is an independent event
    #     result = 0
    #     for likelihood in likelihoods:
    #         # dot product of base frequencies and the likelihood array
    #         compressed = np.dot(likelihood,
    #                             self.sub.getHyperParams()[0])  # dot product of likelihoods and the base frequencies
    #         # use the log likelihood
    #         print(compressed)
    #         result += math.log(compressed)
    #
    #     return result
    #
    # def computeColumnLikelihood(self, i, startNode):
    #
    #     """
    #             Computes the likelihood for one unique site(column) in the data matrix
    #
    #             If this function is called on a leaf, the likelihood will be simply a unit vector of length
    #             4 that describes the state A C G or T based on the actual data matrix.
    #
    #             If this function is called on an internal node, then Felsenstein's algo will be invoked
    #             on its children nodes to compute its likelihood.
    #
    #             i-- the column index into the data matrix
    #
    #             startNode-- the node for which you would like to compute the Felsenstein likelihood for
    #
    #             Output: an array of length 4. arr[j] is the probability that state j is seen in this node
    #                     given branch lengths and child probabilities
    #             """
    #
    #     # only works for DNA right now
    #     genes = ["A", "C", "G", "T"]
    #     likelihoods = np.zeros((4, 1))
    #
    #     # if the node is a leaf node, simply grab the state from the data matrix and
    #     # transform into array ie. [0,0,1,0] = G
    #     if self.tree.outDegree(startNode) == 0:
    #         letter = genes.index(self.data.getIJ_char(self.data.rowGivenName(startNode.getName()), i))
    #         likelihoods[letter] = 1
    #         return likelihoods
    #     else:
    #         # internal node. use felsensteins algo to merge likelihoods from all children
    #         children = self.tree.findDirectSuccessors(startNode)
    #         childLikelihoods = {child: self.computeColumnLikelihood(i, child) for child in children}
    #         childCount = len(children)
    #
    #         # iterate over each combo of ({a, c, g, t}, {a, c, g, t})
    #         for m in range(4):
    #             tempValues = np.zeros((childCount, 1))
    #             for n in range(4):
    #
    #                 # for each child, add up the likelihoods
    #                 for k in range(childCount):
    #                     childLikelihood = childLikelihoods[children[k]][n]  # index into child's likelihood array
    #                     # multiply child likelihood by the P_ij value for the branch length
    #                     tempValues[k] += childLikelihood * self.pij(m, n, children[k].branchLen())
    #
    #             # multiply likelihoods
    #             product = 1.0
    #             for l in range(childCount):
    #                 product *= tempValues[l]
    #
    #             # set parent likelihood array
    #             likelihoods[m] = product
    #
    #         return likelihoods
    #
    # def pij(self, i, j, branchLen):
    #     """
    #             Calculate Q^branchLen and select the ijth entry. Only calculate
    #             Q^branchLen if it has not been calculated before.
    #
    #             i-- row index
    #             j-- column index
    #             branchLen-- time, or the power to raise the transition matrix to
    #
    #             Output: P_ij
    #             """
    #
    #     if branchLen in self.cache.keys():
    #         # simply grab it from the cache
    #         return self.cache[branchLen][i][j]
    #     else:
    #         # store in cache
    #         self.cache[branchLen] = self.sub.expt(branchLen)
    #         return self.cache[branchLen][i][j]


## TESTS ##

# test network

n = NetworkBuilder(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\optimal4Taxa.nex")
# n.printNetworks()
n2 = NetworkBuilder(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxaMultipleSites.nex")
n3 = NetworkBuilder(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxa1Site.nex")

test = n.getNetwork(0)
test2 = n2.getNetwork(0)
test3 = n3.getNetwork(0)

msa = AlignIO.read(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\optimal4Taxa.nex", "nexus")
msa2 = AlignIO.read(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxaMultipleSites.nex",
    "nexus")
msa3 = AlignIO.read(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxa1Site.nex", "nexus")

# data = Matrix(msa)
data2 = Matrix(msa2)  # default is to use the DNA alphabet
data3 = Matrix(msa3)

# prob = Probability(test, data=data)
prob2 = Probability(test2, data=data2)
prob3 = Probability(test3, data=data3)

# print(prob.computeColumnLikelihood(0, test.findRoot()[0]))
# print(prob.likelihoodHelper(test.findRoot()[0]))


# print(prob.likelihood())
# print(prob2.likelihoodHelper(test2.findRoot()[0]))
# print(prob.likelihood1())
# print(prob2.likelihood())
# print(prob2.likelihoodHelper(test2.findRoot()[0]))
# print(prob2.likelihood1())

print(prob3.felsenstein_likelihood())
# print(prob3.likelihood())


print("=======================")

# left = np.array([0.00097386, 0.0282851, 0.0282851, 0.00097386])
# right = np.array([0.00097386, 0.82152469, 0.00097386, 0.00097386])
#
# pij = np.array([[0.90637999, 0.03120667, 0.03120667, 0.03120667], [0.03120667, 0.90637999, 0.03120667, 0.03120667],
#                 [0.03120667, 0.03120667, 0.90637999, 0.03120667], [0.03120667, 0.03120667, 0.03120667, 0.90637999]])

left = np.array([0.00097, 0.0282, 0.0282, 0.00097])
right = np.array([0.00097, 0.821, 0.00097, 0.00097])

pij = np.array([[0.90638, 0.031206, 0.031206, 0.031206], [0.031206, 0.90638, 0.031206, 0.031206],
                [0.031206, 0.031206, 0.90638, 0.031206], [0.031206, 0.031206, 0.031206, 0.90638]])

leftMult = np.matmul(left, pij)
rightMult = np.matmul(right, pij)

print(leftMult)
print(rightMult)

combined = np.multiply(leftMult, rightMult)

dot = np.dot(combined, np.array([.25, .25, .25, .25]))

print(math.log(dot))
