import numpy as np
from scipy.linalg import expm
from SNPModule import map_nr_to_index


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

    Q Matrix is given by Equation 15 from:

    David Bryant, Remco Bouckaert, Joseph Felsenstein, Noah A. Rosenberg, Arindam RoyChoudhury, Inferring Species Trees
    Directly from Biallelic Genetic Markers: Bypassing Gene Trees in a Full Coalescent Analysis, Molecular Biology and
    Evolution, Volume 29, Issue 8, August 2012, Pages 1917–1932, https://doi.org/10.1093/molbev/mss086
    """

    def __init__(self, n: int, u: float, v: float, coal: float):

        # Build Q matrix
        self.n = n
        self.u = u
        self.v = v
        self.coal = coal

        rows = int(.5 * n * (n + 3))
        self.Q : np.ndarray = np.zeros((rows, rows))
        for n_prime in range(1, n + 1):  # n ranges from 1 to individuals sampled (both inclusive)
            for r_prime in range(n_prime + 1):  # r ranges from 0 to n (both inclusive)
                index = map_nr_to_index(n_prime, r_prime)  # get index from n,r pair

                #### EQ 15 ####
                
                # THE DIAGONAL. always calculated
                self.Q[index][index] = -(n_prime * (n_prime - 1) / coal) - (v * (n_prime - r_prime)) - (r_prime * u)

                # These equations only make sense if r isn't 0 (and the second, if n isn't 1).
                if 0 < r_prime <= n_prime:
                    if n_prime > 1:
                        self.Q[index][map_nr_to_index(n_prime - 1, r_prime - 1)] = (r_prime - 1) * n_prime / coal
                    self.Q[index][map_nr_to_index(n_prime, r_prime - 1)] = (n_prime - r_prime + 1) * v

                # These equations only make sense if r is strictly less than n (and the second, if n is not 1).
                if 0 <= r_prime < n_prime:
                    if n_prime > 1:
                        self.Q[index][map_nr_to_index(n_prime - 1, r_prime)] = (n_prime - 1 - r_prime) * n_prime / coal
                    self.Q[index][map_nr_to_index(n_prime, r_prime + 1)] = (r_prime + 1) * u

    def expt(self, t:float) -> np.ndarray:
        """
        Compute exp(Qt) efficiently
        """
        return expm(self.Q * t)

    def cols(self) -> int:
        """
        return the dimension of the Q matrix
        """
        return self.Q.shape[1]

    

