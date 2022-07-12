import numpy as np
from numpy import linalg as lg
import random 


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

                self.freqs = baseFreqs
                self.trans = transitions
                self.Q = self.buildQ()
                self.qIsUpdated = True
                self.Qt = None
                self.states = states
                
        
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
                                else:

                                        self.Q[i][j] = self.freqs[j]*self.trans[j+i]/self.freqs[i]

                for i in range(self.states):
                        self.Q[i][i] = -1 * sum(self.Q[i, :])

                #normalize
                normFactor = 0
                for i in range(self.states):
                        normFactor += (self.Q[i][i] * self.freqs[i])
                
                normFactor = -1/normFactor
                self.Q = self.Q * normFactor
                
        
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


        

Q = GTR()
print(Q.Q)


print(Q.expt(2))