from State import State
from src.lib.DataStructures.MSA import MSA
from src.lib.DataStructures.Matrix import Matrix

import cProfile

from Move import *
from GTR import *
import random


class ProposalKernel:

    def __init__(self):
        self.taxa_move_count = 0

    def generate(self):
        """
            Given a State obj, make changes to generate a new State that
            is "close" to the prior state.

            Input: the state to be manipulated
        """
        if self.taxa_move_count < 30:
            self.taxa_move_count += 1
            return TaxaSwapMove()
        else:
            random_num = random.random()

            if random_num < .1:
                return RootBranchMove()
            elif random_num < .8:
                return UniformBranchMove()
            else:
                return UniformBranchMove()


class HillClimbing:
    """
        If the likelihood is better we take it. Simple Proposal Kernel
    """

    def __init__(self, pkernel, submodel, data, num_iter):
        self.current_state = State()
        self.current_state.bootstrap(data, submodel)
        self.data = data
        self.submodel = submodel
        self.kernel = pkernel
        self.num_iter = num_iter

    def run(self):
        """
        Run the hill climbing algorithm on a bootstrapped starting state.

        Inputs: none
        Outputs: The end state (a State obj) that locally minimizes the score
        """

        self.current_state.write_line_to_summary("--------------------------")
        self.current_state.write_line_to_summary("------Begin Hillclimb-----")

        # run a maximum of 10000 iterations
        iter_no = 0

        while iter_no < self.num_iter:

            # propose a new state
            next_move = self.kernel.generate()
            self.current_state.generate_next(next_move)

            # calculate the difference in score between the proposed state and the current state
            delta = self.current_state.likelihood() - self.current_state.proposed().likelihood()

            if delta <= 0:
                # the new state is more likely. Take it
                self.current_state.commit(next_move)
            else:
                self.current_state.revert(next_move)

            self.current_state.write_line_to_summary(
                "ITER #" + str(iter_no) + " LIKELIHOOD = " + str(self.current_state.likelihood()))
            iter_no += 1

        self.current_state.write_line_to_summary("DONE. EXITED WITH 0 ERRORS")
        self.current_state.write_line_to_summary("--------------------------")

        return self.current_state

    def runMany(self, num_iter):
        """
            Runs the hill climbing algorithm num_iter times, with different starting states.

            Inputs: num_iter, the number of times to run the algo
            Outputs: Prints out the statistics for the scores on each run. Returns a list of numbers
             [mean, median, max, min]

        """
        all_end_states = []
        for dummy in range(num_iter):
            self.current_state = State()
            self.current_state.bootstrap(self.data, self.submodel)
            end_state = self.run()
            all_end_states.append(end_state.likelihood())

        all_end_states.sort()
        length = len(all_end_states)

        if length % 2 == 0:
            median = .5 * (all_end_states[int(length / 2)] + all_end_states[int(length / 2) - 1])
        else:
            median = all_end_states[int((length + 1) * .5) - 1]

        mean = sum(all_end_states) / length
        max_val = all_end_states[-1]
        min_val = all_end_states[0]

        print("===============================================")
        print("Hill Climbing ran " + str(num_iter) + " times...")
        print("===============================================")
        print("Mean score: " + str(mean) + "\nMedian score: " + str(median) + "\nMaximum score: " + str(max_val) +
              "\nMinimum score: " + str(min_val))
        print("===============================================")

        return [mean, median, max_val, min_val]


class MetropolisHastings:
    """
        A special case of Hill Climbing, with a special proposal kernel
    """

    def __init__(self, kernel, hc):
        self.kernel = kernel
        self.hill_climb = hc


def test():
    pr = cProfile.Profile()

    # n = NetworkBuilder(
    # "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\MetroHastingsTests\\truePhylogeny.nex")

    # testnet = n.getNetwork(0)

    msa = MSA(
        "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\MetroHastingsTests\\truePhylogeny.nex")

    data = Matrix(msa)  # default is to use the DNA alphabet

    # goalprob = Probability(testnet, data=data).felsenstein_likelihood()

    # print(goalprob)

    pr.enable()
    hill = HillClimbing(ProposalKernel(), JC(), data, 1000)
    pr.disable()
    # pr.enable()
    final_state = hill.run()
    print(final_state)
    print(final_state.current_model)
    final_state.current_model.summary(
        "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\lib\\DataStructures\\finalTree.txt",
        "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\lib\\DataStructures\\summary.txt")
    # pr.disable()
    pr.print_stats(sort="tottime")
    print("----------------------")


test()
