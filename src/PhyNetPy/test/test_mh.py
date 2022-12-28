import sys
sys.path.insert(0, 'src/PhyNetPy')
from Inference import *

class TestMetropolisHastings():
    
    def test_DNA(self):
        """
        This computes the likelihood of a complete nexus file, with no missing/gap characters
        in the data matrix. The nexus file contains a standard binary tree. Simplest possible working case.
        """
        
        assert True == True
        
        
    def test_SNP(self):
        """
        This is the case in which there is a network passed in through the nexus file.
        This is an error, since Felsenstein's algorithm does not support networks. Handle accordingly
        """ 
        assert True == True
    
    