class ProposalKernel:

        def __init__(self):
                pass

        def accept(self):
                pass

        def reject(self):
                pass

        def propose(self):
                pass



class HillClimbing:
        """
        If the likelihood is better we take it. Simple Proposal Kernel
        """
        def __init__(self):
                pass



class MetropolisHastings:
        """
        A special case of Hill Climbing, with a special proposal kernel
        """

        def __init__(self, kernel, hc):
                self.kernel = kernel
                self.hill_climb = hc
        

        def run(self):
                print("RUNNING METROPOLIS HASTINGS")
                return 0
        
