import numpy as np
from numpy import linalg as lg
import random 
import math


class SubstitutionModelError(Exception):


        def __init__(self, message="Unknown substitution model error"):
                self.message = message
                super().__init__(self.message)





class GTR:

        def __init__(self, baseFreqs, transitions, states=4):

                if len(baseFreqs)!=states or sum(baseFreqs)!= 1:
                        raise SubstitutionModelError("Base frequency list either does not sum to 1 or is not of correct length")

                if len(transitions) != ((states-1)*states) / 2:
                        raise SubstitutionModelError("incorrect number of transition rates")
                self.states = states
                self.freqs = baseFreqs
                self.trans = transitions
                self.Q = self.buildQ()
                self.qIsUpdated = True
                self.Qt = None
                
                
        
        def populateTest(self):
                for i in range(self.states):
                        for j in range(self.states):
                                self.Q[i][j] = random.random()
        
        def buildQ(self):
                self.Q = np.zeros((self.states, self.states), dtype = np.double)

                for i in range(self.states):
                        for j in range(self.states):
                                if j>i:
                                        self.Q[i][j] = self.trans[i+j]
                                elif i>j:
                                        self.Q[i][j] = self.freqs[j]*self.trans[j+i-1]/self.freqs[i]

                for i in range(self.states):
                        self.Q[i][i] = -1 * sum(self.Q[i, :])


                
                #normalize
                normFactor = 0
                for i in range(self.states):
                        normFactor += (self.Q[i][i] * self.freqs[i])
                
                normFactor = -1/normFactor
                self.Q = self.Q * normFactor

                print(self.Q)
                
        
        def updateQ(self):
                self.qIsUpdated = True
                #do updates

        def expt(self, t):
        
                if self.qIsUpdated:
                        eigenvals, eigenvecs = lg.eigh(self.Q)
                        self.q = eigenvecs
                        self.qinv = np.transpose(self.q)
                        self.diag = np.diag(eigenvals)

                        self.Qt = np.real(np.matmul(np.matmul(self.q, lg.matrix_power(self.diag, t)), self.qinv))
                        self.qIsUpdated = False
   
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