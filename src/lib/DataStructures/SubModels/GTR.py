import numpy as np
from numpy import linalg as lg
import random 
import math
from numba import jit, cuda
from timeit import default_timer as timer  
import copy

class SubstitutionModelError(Exception):


        def __init__(self, message="Unknown substitution model error"):
                self.message = message
                super().__init__(self.message)



class GTR:
        """
        General superclass for substitution models. Implements 
        Eigenvalue decomposition for computing P(t).
        Special case subclasses attempt to improve on the time 
        complexity of that operation
        """

        def __init__(self, baseFreqs, transitions, states=4):

                #Check for malformed inputs
                if len(baseFreqs)!=states or sum(baseFreqs)!= 1:
                        raise SubstitutionModelError("Base frequency list either does not sum to 1 or is not of correct length")

                if len(transitions) != ((states-1)*states) / 2:
                        raise SubstitutionModelError("incorrect number of transition rates")


                self.states = states
                self.freqs = baseFreqs
                self.trans = transitions

                #compute Q, the instantaneous probability matrix
                self.Q = self.buildQ()

                self.qIsUpdated = True
                self.Qt = None
                

        def getHyperParams(self):
                return self.freqs, self.trans

        def getStates(self):
                return self.states

        
        def buildQ(self):
                """
                Populate the Q matrix with the correct values. 
                Based on https://en.wikipedia.org/wiki/Substitution_model
                """
                self.Q = np.zeros((self.states, self.states), dtype = np.double)

                for i in range(self.states):
                        for j in range(self.states):
                                if j>i:
                                        self.Q[i][j] = self.trans[i+j]
                                elif i>j:
                                        self.Q[i][j] = self.freqs[j]*self.trans[j+i-1]/self.freqs[i]

                for i in range(self.states):
                        self.Q[i][i] = -1 * sum(self.Q[i, :])


                
                #normalize such that -1 * SUM Q_ii * pi_i = 1
                normFactor = 0
                for i in range(self.states):
                        normFactor += (self.Q[i][i] * self.freqs[i])
                
                normFactor = -1/normFactor

                #scale the matrix
                self.Q = self.Q * normFactor

                
        
        def updateQ(self):
                """
                If any parameters to the model are changed, repopulate the Q matrix
                """
                self.qIsUpdated = True
                #do updates

        @jit(target_backend="cuda")
        def exptCUDA(self, t):
                """
                DOES NOT WORK RIGHT NOW

                Compute the matrix exponential Q^t and store the result.
                If the solution has been computed already but the Q matrix has not 
                changed, simply return the value
                """
                if self.qIsUpdated:
                        eigenvals, eigenvecs = lg.eigh(self.Q)
                        self.q = eigenvecs
                        self.qinv = np.transpose(self.q)
                        self.diag = np.diag(eigenvals)

                        self.Qt = np.real(np.matmul(np.matmul(self.q, lg.matrix_power(self.diag, t)), self.qinv))
                        self.qIsUpdated = False
   
                return self.Qt
        
        
        def expt(self, t):
                """
                Compute the matrix exponential Q^t and store the result.
                If the solution has been computed already but the Q matrix has not 
                changed, simply return the value
                """
                if self.qIsUpdated:
                        eigenvals, eigenvecs = lg.eigh(self.Q)
                        self.q = eigenvecs
                        self.qinv = np.transpose(self.q)
                        self.diag = np.diag(eigenvals)

                        self.Qt = np.real(np.matmul(np.matmul(self.q, lg.matrix_power(self.diag, t)), self.qinv))
                        #self.qIsUpdated = False   TODO:make sure laziness works here
   
                return self.Qt


class K2P(GTR):

        def __init__(self, alpha, beta):
                if alpha + beta != 1:
                        raise SubstitutionModelError("K2P Transversion + Transition params do not add to 1")

                bases = [.25, .25, .25, .25]
                trans = np.ones((6 , 1)) * alpha
                trans[1] = beta
                trans[4] = beta
                self.beta = beta
                self.alpha = alpha
                super().__init__(bases, trans, 4)


        def expt(self, t):

                self.Qt = np.zeros((self.states, self.states), dtype = np.double)

                for i in range(self.states):
                        for j in range(self.states):
                                self.Qt[i][j] = .25 * (1 - 2*math.exp(-4*t*(self.beta+self.alpha)) + math.exp(-8*self.beta*t))
                
                return self.Qt

        
class F81(GTR):

        def __init__(self, bases, states = 4):
                trans = np.ones(((states * (states - 1)) / 2 , 1))
                super().__init__(bases, trans, states)


        def expt(self, t):

                self.Qt = np.zeros((self.states, self.states), dtype = np.double)

                for i in range(self.states):
                        for j in range(self.states):
                                if i==j:
                                        delta = 1
                                else:
                                        delta = 0
                                self.Qt[i][j] = math.exp(-1)*delta + (1- math.exp(-1))*self.freqs[j]
                
                return self.Qt



class JC(F81):
        def __init__(self, states = 4):
                bases = np.ones((states, 1)) * (1/states)
                super().__init__(bases, states)


class HKY(GTR):

        def __init__(self, baseFreqs, transitions):
                if len(baseFreqs) != 4 and len(transitions)!= 6:
                        raise SubstitutionModelError("Incorrect parameter input length")

                if transitions[0] != transitions[2] or transitions[2] != transitions[3] or transitions[3]!=transitions[5]:
                        raise SubstitutionModelError("Error in HKY Transversions. Not all equal") 
                
                if transitions[1] != transitions[4]:
                        raise SubstitutionModelError("Error in HKY Transitions. Not all equal") 

                super().__init__(baseFreqs, transitions, 4)


class K3ST(GTR):

        ##POTENTIAL SPEEDUP???

        def __init__(self, transitions):
                if len(transitions)!= 6:
                        raise SubstitutionModelError("Incorrect parameter input length")

                if transitions[0] != transitions[5] or transitions[1] != transitions[4] or transitions[2]!=transitions[3]:
                        raise SubstitutionModelError("K3ST parameter mismatch") 
                
                baseFreqs = [.25, .25, .25, .25]

                super().__init__(baseFreqs, transitions, 4)

class SYM(GTR):

        def __init__(self, transitions, states=4):
                baseFreqs = np.ones((states, 1)) * (1/states)
                super().__init__(baseFreqs, transitions, states)



class TN93(GTR):

        def __init__(self, baseFreqs, transitions):

                if len(baseFreqs) != 4 and len(transitions)!= 6:
                        raise SubstitutionModelError("Incorrect parameter input length")

                if transitions[0] != transitions[2] or transitions[2] != transitions[3] or transitions[3]!=transitions[5]:
                        raise SubstitutionModelError("Error in TN93 Transversions. Not all equal") 

                super().__init__(baseFreqs, transitions, 4)








model = GTR([.25, .25, .25, .25], [1,1,1,1,1,1])
model2 = copy.deepcopy(GTR([.25, .25, .25, .25], [1,1,1,1,1,1]))

start = timer()
model.exptCUDA(50)
print("with GPU:", timer()-start)    

start = timer()
model2.expt(50)
print("without GPU:", timer()-start)




