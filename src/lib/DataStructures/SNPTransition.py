import numpy as np
import scipy.linalg
from scipy.linalg import fractional_matrix_power


def map_nr_to_index(n, r):
    starts = int(.5 * (n - 1) * (n + 2))
    return starts + r


class SNPTransition:
    """
    Class that encodes the probabilities of transitioning from one (n,r) pair to another under a Biallelic model

    Includes methods for efficiently computing Q^t

    Inputs:
    1) n-- the total number of samples in the species tree
    2) u-- the probability of going from the red allele to the green one
    3) v-- the probability of going from the green allele to the red one
    4) coal-- the coalescent rate constant, theta

    Assumption: Matrix indexes start with n=1, r=0, so Q[0][0] is Q(1,0);(1,0)
    """

    def __init__(self, n: int, u: float, v: float, coal: float):

        # Build Q matrix

        rows = int(.5 * n * (n + 3))
        self.Q = np.zeros((rows, rows))
        for n_prime in range(1, n + 1):  # n ranges from 1 to individuals sampled (both inclusive)
            for r_prime in range(n_prime + 1):  # r ranges from 0 to n (both inclusive)
                index = map_nr_to_index(n_prime, r_prime)

                self.Q[index][index] = -(n_prime * (n_prime - 1) / coal) - (v * (n_prime - r_prime)) - (r_prime * u)

                if n_prime > 0 and r_prime > 0:
                    self.Q[index][map_nr_to_index(n_prime - 1, r_prime - 1)] = (r_prime - 1) * n_prime / coal
                elif r_prime > 0:
                    self.Q[index][map_nr_to_index(n_prime, r_prime - 1)] = (n_prime - r_prime + 1) * v
                elif n_prime > 0 and n_prime > r_prime:
                    self.Q[index][map_nr_to_index(n_prime - 1, r_prime)] = (n_prime - 1 - r_prime) * n_prime / coal

                if r_prime < n_prime:
                    self.Q[index][map_nr_to_index(n_prime, r_prime + 1)] = (r_prime + 1) * u

    def expt(self, t):
        return np.real(fractional_matrix_power(self.Q, t))
