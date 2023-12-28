

from math import sqrt, comb, pow
import numpy as np
import scipy
from scipy.linalg import expm
from MSA import MSA
from BirthDeath import CBDP
from NetworkParser import NetworkParser
from Alphabet import Alphabet
from Matrix import Matrix
from ModelGraph import Model
from ModelFactory import *
from Graph import DAG



## Step 1a: Implement Likelihood Function


## Step 1b: Implement NetworkNode class ##

class CoalNetworkNode(ANetworkNode):
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def calc(self):
        if self.get_model_parents() is not None:
            self.cached = self.likelihood([child.get() for child in self.get_model_parents()])
        else:
            self.cached = self.likelihood()
            
        self.updated = False
        
        return self.cached
        





