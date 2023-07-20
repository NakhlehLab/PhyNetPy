""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0

"""


import numpy as np
from numpy import linalg as lg
import math


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

    def __init__(self, base_freqs, transitions, states=4):

        # Check for malformed inputs
        if len(base_freqs) != states or sum(base_freqs) != 1:
            raise SubstitutionModelError("Base frequency list either does not sum to 1 or is not of correct length")

        if len(transitions) != ((states - 1) * states) / 2:
            raise SubstitutionModelError("incorrect number of transition rates")

        self.states = states
        self.freqs = base_freqs
        self.trans = transitions

        # compute Q, the instantaneous probability matrix
        self.Q = self.buildQ()
        self.Qt = None

    def getQ(self) -> np.ndarray:
        """
        Get Q matrix

        Returns: np array obj
        """
        return self.Q

    def get_hyperparams(self) -> list:
        return self.freqs, self.trans

    def state_count(self) -> int:
        return self.states

    def buildQ(self) -> np.ndarray:
        """
        Populate the Q matrix with the correct values.
        Based on https://en.wikipedia.org/wiki/Substitution_model
        """
        self.Q = np.zeros((self.states, self.states), dtype=np.double)

        for i in range(self.states):
            for j in range(self.states):
                if j > i:
                    self.Q[i][j] = self.trans[i + j]
                elif i > j:
                    self.Q[i][j] = self.freqs[j] * self.trans[j + i - 1] / self.freqs[i]

        for i in range(self.states):
            self.Q[i][i] = -1 * sum(self.Q[i, :])

        # normalize such that -1 * SUM Q_ii * pi_i = 1
        normFactor = 0
        for i in range(self.states):
            normFactor += (self.Q[i][i] * self.freqs[i])

        normFactor = -1 / normFactor

        # scale the matrix
        self.Q = self.Q * normFactor
        return self.Q

    def expt(self, t:float) -> np.ndarray:
        """
        Compute the matrix exponential Q^t and store the result.
        If the solution has been computed already but the Q matrix has not
        changed, simply return the value
        """

        eigenvals, eigenvecs = lg.eigh(self.Q)
        q = eigenvecs
        qinv = lg.inv(q)
        diag = np.diag(eigenvals)
        diagt = np.zeros(np.shape(diag))
        for i in range(np.shape(diag)[0]):
            diagt[i][i] = math.exp(diag[i][i] * t)

        self.Qt = np.real(np.matmul(np.matmul(q, diagt), qinv))

        return self.Qt


class K2P(GTR):

    def __init__(self, alpha:float, beta:float):
        if alpha + beta != 1:
            raise SubstitutionModelError("K2P Transversion + Transition params do not add to 1")

        bases = [.25, .25, .25, .25]
        trans = np.ones((6, 1)) * alpha
        trans[1] = beta
        trans[4] = beta
        self.beta = beta
        self.alpha = alpha
        super().__init__(bases, trans, 4)

    def expt(self, t:float) -> np.ndarray:

        self.Qt = np.zeros((self.states, self.states), dtype=np.double)

        for i in range(self.states):
            for j in range(self.states):
                self.Qt[i][j] = .25 * (
                        1 - 2 * math.exp(-4 * t * (self.beta + self.alpha)) + math.exp(-8 * self.beta * t))

        return self.Qt


class F81(GTR):

    def __init__(self, bases:list, states:int=4):
        trans = np.ones((int((states * (states - 1)) / 2), 1))
        super().__init__(bases, trans, states)

    # def expt(self, t:float) -> np.ndarray:

    #     self.Qt = np.zeros((self.states, self.states))

    #     for i in range(self.states):
    #         for j in range(self.states):
    #             if i == j:
    #                 self.Qt[i][j] = .25 + (.75 * math.exp(-1.333333333 * t))
    #             else:
    #                 self.Qt[i][j] = .25 - (.25 * math.exp(-1.333333333 * t))

    #     return self.Qt


class JC(F81):
    def __init__(self, states=4):
        bases = np.ones((states, 1)) * (1 / states)
        super().__init__(bases, states)


class HKY(GTR):

    def __init__(self, base_freqs:list, transitions:list):
        if len(base_freqs) != 4 and len(transitions) != 6:
            raise SubstitutionModelError("Incorrect parameter input length")

        if transitions[0] != transitions[2] or transitions[2] != transitions[3] or transitions[3] != transitions[5]:
            raise SubstitutionModelError("Error in HKY Transversions. Not all equal")

        if transitions[1] != transitions[4]:
            raise SubstitutionModelError("Error in HKY Transitions. Not all equal")

        super().__init__(base_freqs, transitions, 4)


class K3ST(GTR):

    def __init__(self, transitions:list):
        if len(transitions) != 6:
            raise SubstitutionModelError("Incorrect parameter input length")

        if transitions[0] != transitions[5] or transitions[1] != transitions[4] or transitions[2] != transitions[3]:
            raise SubstitutionModelError("K3ST parameter mismatch")

        base_freqs = [.25, .25, .25, .25]

        super().__init__(base_freqs, transitions, 4)


class SYM(GTR):

    def __init__(self, transitions:list, states=4):
        base_freqs = np.ones((states, 1)) * (1 / states)
        super().__init__(base_freqs, transitions, states)


class TN93(GTR):

    def __init__(self, base_freqs:list, transitions:list):

        if len(base_freqs) != 4 and len(transitions) != 6:
            raise SubstitutionModelError("Incorrect parameter input length")

        if transitions[0] != transitions[2] or transitions[2] != transitions[3] or transitions[3] != transitions[5]:
            raise SubstitutionModelError("Error in TN93 Transversions. Not all equal")

        super().__init__(base_freqs, transitions, 4)


