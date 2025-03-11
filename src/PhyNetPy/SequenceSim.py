#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --                                                              
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

""" 
Author : Mark Kessler
Last Stable Edit : 3/11/25
First Included in Version : 1.0.0
Approved to Release Date : No
"""

from GTR import *
import numpy as np


class SeqSim:
        """
        Class that simulates the evolution of DNA sequences
        """
        
        def __init__(self, submodel : GTR = JC()) -> None:
                """
                Initialize the simulator with a substitution model

                Args:
                        submodel (GTR, optional): A substitution model. 
                                                  Defaults to JC().
                """
                self.sub = submodel

        def modify_seq(self, seq : list) -> list:
                """
                Modify a sequence of DNA letters according to the substitution 
                model

                Args:
                    seq (list): _description_

                Returns:
                    list: _description_
                """
                func = np.vectorize(self.dna_evolve)
                return func(seq)

        def dna_evolve(self, letter : str) -> str:
                """
                Simulate the evolution of a DNA sequence 

                Args:
                    letter (str): A DNA letter

                Returns:
                    str: A new DNA letter
                """
                alphabet = ['A', 'C', 'G', 'T']
                probs = self.transition[alphabet.index(letter)]
                new_letter = np.random.choice(alphabet, 1, p = probs)
                return new_letter[0]

        def change_transition(self, t : float) -> None:
                """
                Change the transition matrix to simulate evolution at a different
                rate

                Args:
                    t (float): the time at which to simulate evolution
                Returns:
                    N/A
                """
                self.transition = self.sub.expt(t)
