


from Matrix import Matrix
import GTR
import numpy as np
import math

class ProbabilityError(Exception):
        def __init__(self, message = "Error"):
                self.message = message
                super().__init__(self.message)


class Probability:
        """
        A class for drawing objects from a distribution, or calculating the 
        probability of an object from within a distribution space (given some data
        and hyperparameters)
        """

        def __init__(self, network, model = GTR.JC(), data = None):
                self.sub = model
                self.data = data
                self.tree = network
                self.cache = {}

        def setModel(self, subModel):
                """
                set the substitution model
                """
                self.sub = subModel

        def setData(self, dataMatrix):
                """
                set the data matrix, must be of class type Matrix
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

        def likelihood(self):
                """
                computes the likelihood of a data structure given the data and 
                a substitution model
                """

                #need data
                if self.data == None:
                        raise ProbabilityError("Attempted to run likelihood computation on empty data source")

                #compute the likelihood for each column in the data matrix
                likelihoods = np.zeros((self.data.siteCount(), 1))
                for colIndex in range(self.data.siteCount()):
                        likelihoods[colIndex] = self.computeColumnLikelihood(colIndex, self.tree.findRoot()[0])
                
                #multiply the likelihoods together, via the assumption that each site
                #is an independent event
                result = 1
                for likelihood in likelihoods:
                        #dot product of base frequencies and the likelihood array
                        compressed = np.dot(likelihood, self.sub.getHyperParams()[0]) #dot product of likelihoods and the base frequencies
                        #use the log likelihood
                        result *= math.log(compressed)
                
                return result
                
                
        def computeColumnLikelihood(self, i, startNode):

                #only works for DNA right now
                genes = ["A", "C", "G", "T"]
                likelihoods = np.zeros((4,1))
                
                #if the node is a leaf node, simply grab the state from the data matrix and
                #transform into array ie. [0,0,1,0] = G
                if self.tree.outDegree(startNode) == 0:
                        letter = genes.index(self.data.getIJ(self.data.rowGivenName(startNode.getName()), i))
                        likelihoods[letter]=1
                        return likelihoods
                else:
                        #internal node. use felsensteins algo to merge likelihoods from all children
                        children = self.tree.findDirectSuccessors(startNode)
                        childLikelihoods = {child: self.computeColumnLikelihood(i, child) for child in children }
                        childCount = len(children)

                        #iterate over each combo of ({a, c, g, t}, {a, c, g, t})
                        for m in range(4):
                                tempValues = np.zeros((childCount, 1))
                                for n in range(4):

                                        #for each child, add up the likelihoods 
                                        for k in range(childCount):
                                                childLikelihood = childLikelihoods[children[k]][n] #index into child's likelihood array
                                                #multiply child likelihood by the P_ij value for the branch length
                                                tempValues[k] += childLikelihood * self.pij(genes[m], genes[n], children[k].branchLen())
                                
                                #multiply likelihoods
                                product = 1.0
                                for l in range(childCount):
                                        product *= tempValues[k]
                                
                                #set parent likelihood array
                                likelihoods[m] = product
                        
                        return likelihoods
                        



        def pij(self, i, j, branchLen):
                """
                Calculate Q^branchLen and select the ijth entry. Only calculate
                Q^branchLen if it has not been calculated before. In 
                """

                if branchLen in self.cache.keys():
                        #simply grab it from the cache
                        return self.cache[branchLen][i][j]
                else:
                        #store in cache
                        self.cache[branchLen] = self.sub.expt(branchLen)
                        return self.cache[branchLen][i][j]
                

