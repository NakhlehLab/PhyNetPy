""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0
Approved to Release Date : N/A
"""

from GTR import *
import numpy as np


class SeqSim:
        
        def __init__(self, submodel=JC()) -> None:
                self.sub = submodel
        
        def modify_seq(self, seq:list):
                func = np.vectorize(self.dna_evolve)
                return func(seq)

        def dna_evolve(self, letter):
                alphabet = ['A', 'C', 'G', 'T']
                new_letter = np.random.choice(alphabet, 1, p = self.transition[alphabet.index(letter)])
                return new_letter[0]
        
        def change_transition(self, t):
                self.transition = self.sub.expt(t)
