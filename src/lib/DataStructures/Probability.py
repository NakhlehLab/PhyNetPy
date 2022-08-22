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
        # if branch length ever changes, update this
        self.tensor = {}
        for node in self.tree.nodes:
            self.tensor[node] = self.sub.expt(node.branchLen())

    def setModel(self, subModel):
        """
                set the substitution model

                subModel-- the new substitution model of subtype GTR
                """
        self.sub = subModel

    def setData(self, dataMatrix):
        """
                set the data matrix, must be of class type Matrix

                dataMatrix-- a Matrix object
                """
        if type(dataMatrix) != Matrix:
            raise ProbabilityError("Tried to set data to type other than Matrix")
        self.data = dataMatrix

    def generate(self):
        """
                A method that generates an object based on the model parameters,
                hyperparameters, and sequence data that are passed in
                """
        return 0

    def likelihood1(self):
        matrix = self.likelihoodHelper(self.tree.findRoot()[0])
        baseFreqs = np.array(self.sub.getHyperParams())

        result = 0
        for row in range(np.shape(matrix)[0]):
            logLikelihoodSite = math.log(np.dot(baseFreqs, matrix[row]))
            result += logLikelihoodSite

        return result

    def likelihoodHelper(self, startNode):

        # if the node is a leaf node, simply grab the state from the data matrix and
        # transform into array ie. [0,0,1,0] = G
        print(startNode.getName() + " HAS OUT DEGREE " + str(len(self.tree.findDirectSuccessors(startNode))))
        if self.tree.outDegree(startNode) == 0:

            likelihoods = np.zeros((self.data.siteCount(), 4))
            seq = self.data.getSeq(startNode.getName())

            for col in range(self.data.siteCount()):
                if seq[col] == "A":
                    row = np.array([1, 0, 0, 0])
                elif seq[col] == "C":
                    row = np.array([0, 1, 0, 0])
                elif seq[col] == "G":
                    row = np.array([0, 0, 1, 0])
                elif seq[col] == "T":
                    row = np.array([0, 0, 0, 1])
                likelihoods[col, :] = row

            print("----THIS IS A LEAF----")
            print(likelihoods)
            print("-----------------------")
            return likelihoods

        # if not a leaf
        children = self.tree.findDirectSuccessors(startNode)
        if len(children) > 2 or len(children) <= 0:
            print(children)
            raise ProbabilityError("Malformed tree, node has an incorrect number of child nodes")
        if len(children) == 1:
            return self.likelihoodHelper(children[0])
        else:
            return self.combineChildLikelihoods(children[0], children[1])

    def combineChildLikelihoods(self, child1, child2):

        child1Matrix = self.likelihoodHelper(child1)
        child2Matrix = self.likelihoodHelper(child2)

        child1Step1 = np.matmul(child1Matrix, self.sub.expt(child1.branchLen()).transpose())
        child2Step1 = np.matmul(child2Matrix, self.sub.expt(child2.branchLen()).transpose())

        return np.multiply(child1Step1, child2Step1)

    def likelihood(self):
        """
                computes the likelihood of a data structure given the data and 
                a substitution model

                For now, this is felsensteins algorithm, which only works on DNA
                and trees that lack reticulations/hybridizations.

                Outputs: The log likelihood probability P(Tree | Data & Substitution model) 

                """

        # need data
        if self.data == None:
            raise ProbabilityError("Attempted to run likelihood computation on empty data source")

        # compute the likelihood for each column in the data matrix
        likelihoods = np.zeros((self.data.siteCount(), 1))
        for colIndex in range(self.data.siteCount()):
            likelihoods[colIndex] = self.computeColumnLikelihood(colIndex, self.tree.findRoot()[0])

        # multiply the likelihoods together, via the assumption that each site
        # is an independent event
        result = 0
        for likelihood in likelihoods:
            # dot product of base frequencies and the likelihood array
            compressed = np.dot(likelihood,
                                self.sub.getHyperParams()[0])  # dot product of likelihoods and the base frequencies
            # use the log likelihood
            result += math.log(compressed)

        return result

    def computeColumnLikelihood(self, i, startNode):

        """
                Computes the likelihood for one unique site(column) in the data matrix

                If this function is called on a leaf, the likelihood will be simply a unit vector of length
                4 that describes the state A C G or T based on the actual data matrix.

                If this function is called on an internal node, then Felsenstein's algo will be invoked
                on its children nodes to compute its likelihood.

                i-- the column index into the data matrix

                startNode-- the node for which you would like to compute the Felsenstein likelihood for

                Output: an array of length 4. arr[j] is the probability that state j is seen in this node
                        given branch lengths and child probabilities
                """

        # only works for DNA right now
        genes = ["A", "C", "G", "T"]
        likelihoods = np.zeros((4, 1))

        # if the node is a leaf node, simply grab the state from the data matrix and
        # transform into array ie. [0,0,1,0] = G
        if self.tree.outDegree(startNode) == 0:
            letter = genes.index(self.data.getIJ_char(self.data.rowGivenName(startNode.getName()), i))
            likelihoods[letter] = 1
            return likelihoods
        else:
            # internal node. use felsensteins algo to merge likelihoods from all children
            children = self.tree.findDirectSuccessors(startNode)
            childLikelihoods = {child: self.computeColumnLikelihood(i, child) for child in children}
            childCount = len(children)

            # iterate over each combo of ({a, c, g, t}, {a, c, g, t})
            for m in range(4):
                tempValues = np.zeros((childCount, 1))
                for n in range(4):

                    # for each child, add up the likelihoods
                    for k in range(childCount):
                        childLikelihood = childLikelihoods[children[k]][n]  # index into child's likelihood array
                        # multiply child likelihood by the P_ij value for the branch length
                        tempValues[k] += childLikelihood * self.pij(m, n, children[k].branchLen())

                # multiply likelihoods
                product = 1.0
                for l in range(childCount):
                    product *= tempValues[l]

                # set parent likelihood array
                likelihoods[m] = product

            return likelihoods

    def pij(self, i, j, branchLen):
        """
                Calculate Q^branchLen and select the ijth entry. Only calculate
                Q^branchLen if it has not been calculated before. 

                i-- row index
                j-- column index
                branchLen-- time, or the power to raise the transition matrix to

                Output: P_ij
                """

        if branchLen in self.cache.keys():
            # simply grab it from the cache
            return self.cache[branchLen][i][j]
        else:
            # store in cache
            self.cache[branchLen] = self.sub.expt(branchLen)
            return self.cache[branchLen][i][j]


## TESTS ##

# test network

n = NetworkBuilder(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxa1Site.nex")
# n.printNetworks()
n2 = NetworkBuilder("C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxaMultipleSites.nex")

test = n.getNetwork(0)
test2 = n2.getNetwork(0)
msa = AlignIO.read(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxa1Site.nex", "nexus")
msa2 = AlignIO.read(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxaMultipleSites.nex", "nexus")

data = Matrix(msa)
data2 = Matrix(msa2)# default is to use the DNA alphabet

prob = Probability(test, data=data)
prob2 = Probability(test2, data=data2)

print(prob.computeColumnLikelihood(0, test.findRoot()[0]))
print(prob.likelihoodHelper(test.findRoot()[0]))

for i in range(4):
        print("=============" + str(i) + "==============")
        print(prob2.computeColumnLikelihood(i, test2.findRoot()[0]))
        print("===============================")

print(prob2.likelihoodHelper(test2.findRoot()[0]))

