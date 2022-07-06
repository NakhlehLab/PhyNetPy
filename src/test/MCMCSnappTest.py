
from TestDriver import runPhyloNet
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np


def runPhyloPy():
        """
        Uses the PhyloPy methods for MCMC Snapp 
        """
        #SeqIO.parse("src/test/testNex.nexus", "nexus")
        return


class MCMCSnappUnitTests:

        numTests = 3

        def __init__(self, tests):
                self.results = {}
                if(tests == "all"):
                        for i in range(self.numTests):
                                self.results[i] = [False]
                else:
                        for test in tests:
                                self.results[test] = [False]

                
        def test0(self):
                """
                1st test

                Compares the text output of PhyloNet to the text output of PhyloPy
                """
                #runPhyloNet()
                #test phylopy
                #compare .out files
                isPassed = True
                self.results[0][0] = isPassed
                self.results[0].append("complex")
                #How to test when output may be different but correct?

        def test1(self):
                #runPhyloNet()
                isPassed = True
                self.results[1][0] = isPassed
                self.results[1].append("simple")

        def test2(self):
                #runPhyloNet()
                isPassed = False
                self.results[2][0] = isPassed
                self.results[2].append("complex")

        def runTests(self):
                self.test0()
                self.test1()
                self.test2()
        
        def showResults(self):
                correct = 0
                wrong = 0
                for res, value in self.results.items():
                        if(value[0] == True):
                                correct += 1
                        else:
                                wrong += 1
                
                plt.pie(np.array([correct,wrong]), labels = ["Passed", "Failed"])
                plt.show()
        
                



