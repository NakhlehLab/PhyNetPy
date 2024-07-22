""" 
Author : Mark Kessler
Last Stable Edit : 4/9/24
First Included in Version : 1.0.0
Approved to Release Date : No
"""

from GTR import *
import numpy as np


class SeqSim:
        """
        Class that simulates the evolution of DNA sequences
        """
        
        def __init__(self, submodel = JC()) -> None:
                self.sub = submodel

        def modify_seq(self, seq:list):
                func = np.vectorize(self.dna_evolve)
                return func(seq)

        def dna_evolve(self, letter):
                alphabet = ['A', 'C', 'G', 'T']
                probs = self.transition[alphabet.index(letter)]
                new_letter = np.random.choice(alphabet, 1, p = probs)
                return new_letter[0]

        def change_transition(self, t):
                self.transition = self.sub.expt(t)
