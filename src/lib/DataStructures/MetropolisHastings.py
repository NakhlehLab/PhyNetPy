from State import State


class ProposalKernel:

    def __init__(self):
        pass

    def generate(self, state):
        """
            Given a State obj, make changes to generate a new State that
            is "close" to the prior state.

            Input: the state to be manipulated
        """
        return state


class HillClimbing:
    """
        If the likelihood is better we take it. Simple Proposal Kernel
    """

    def __init__(self):
        self.current_state = State().bootstrap()

    def run(self):
        """
        Run the hill climbing algorithm on a bootstrapped starting state.

        Inputs: none
        Outputs: The end state (a State obj) that locally minimizes the score
        """

        # run a maximum of 10000 iterations
        iter_no = 0
        iter_with_small_delta = 0
        iter_rejections = 0
        while iter_no < 10000:

            # propose a new state
            self.current_state.propose()

            # undo the state if the current state is mathematically impossible
            if self.current_state.is_illegal():
                self.current_state.undo()

            # calculate the difference in score between the proposed state and the current state
            delta = self.current_state.get_score() > self.current_state.previous().get_score()

            if delta > 0:
                # the new state is more likely. Take it
                self.current_state.accept()

                # reset the rejection counter, we accepted a new state
                iter_rejections = 0

                # return state if we seem to be in a state that locally maximizes the likelihood
                if delta < .001 and iter_with_small_delta >= 100:
                    return self.current_state
                elif delta < .001:
                    iter_with_small_delta += 1
                else:
                    iter_with_small_delta = 0

            else:
                iter_rejections += 1
                self.current_state.reject()

                # if too many rejections in a row, then a local max has been found.
                if iter_rejections >= 100:
                    return self.current_state

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
            self.current_state = State("hi").bootstrap()
            end_state = self.run()
            all_end_states.append(end_state.get_score())

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

    def run(self):
        print("RUNNING METROPOLIS HASTINGS")
        return 0
