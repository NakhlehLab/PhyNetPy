"""
This class is the superclass for all types of Monte Carlo Markov Chain inference.
Any subclass is a type of MCMC algorithm.
"""

class MCMC:
        def __init__(self, locusList=[], chainLen=10000000, burnInLen=2000000, sampleFreq=5000, seed=12345678, threads=1, path="/home/dir"):
                self.locus = locusList
                self.chain = chainLen
                self.burn = burnInLen
                self.freq = sampleFreq
                self.seed = seed
                self.threads = threads
                self.outDir = path



