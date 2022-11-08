import numpy as np
import scipy.linalg
from scipy.linalg import fractional_matrix_power
from scipy.linalg import expm


def map_nr_to_index(n, r):
    """
    Takes an (n,r) pair and maps it to a 1d vector index

    (1,0) -> 0
    (1,1) -> 1
    (2,0) -> 2
    ...
    """
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
        self.Q = np.zeros((rows, rows))
        for n_prime in range(1, n + 1):  # n ranges from 1 to individuals sampled (both inclusive)
            for r_prime in range(n_prime + 1):  # r ranges from 0 to n (both inclusive)
                index = map_nr_to_index(n_prime, r_prime)  # get index from n,r pair

                # EQ 15

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

    def expt(self, t):
        """
        Compute exp(Q^t) efficiently using scipy fractional matrix power
        WARNING: USING VALUES LIKE 1.5 IS DANGEROUS. IF THERE DOES NOT EXIST MATRIX A such that A*A = Q, THERE WILL BE
        NO SOLUTION, AND THAT IS NOT GUARANTEED.

        TODO: What happens when branch lengths are 1.5????
        """
        return expm(self.Q * t)

    def cols(self):
        """
        return the dimension of the Q matrix
        """
        return self.Q.shape[1]

    def solveCentralBlockTransposed(self, _n, y, offset):

        x = np.zeros(_n + 1)
        K = (-(self.coal * (_n * (_n - 1.0))) / 2.0) - ((_n * self.v) + offset)

        if self.u == 0.0 and self.v == 0.0:
            for r in range(0, _n + 1):
                x[r] = y[r] / K
        elif self.u == 0.0:
            Mrr = K
            x[0] = y[0] / Mrr
            for r in range(1, _n + 1):
                Mrr = K + r * (self.v - self.u)
                x[r] = (y[r] - ((_n - r + 1.0) * self.v) * x[r - 1]) / Mrr
        elif self.v == 0.0:
            Mrr = K + _n * (self.v - self.u)
            x[_n] = y[_n] / Mrr
            for r in range(0, _n):
                r = _n - r - 1
                Mrr = (K + r * (self.v - self.u))
                x[r] = (y[r] - ((r + 1.0) * self.u) * x[r + 1]) / Mrr
        else:
            d = np.zeros(_n + 1)
            e = np.zeros(_n + 1)
            d[0] = K
            e[0] = y[0]
            for r in range(1, _n + 1):
                m = ((_n - r + 1.0) * self.v) / d[r - 1]
                d[r] = K + r * (self.v - self.u) - m * r * self.u
                e[r] = y[r] - m * e[r - 1]

            x[_n] = e[_n] / d[_n]
            for r in range(0, _n):
                r = _n - r - 1
                x[r] = (e[r] - (r + 1) * self.u * x[r + 1]) / d[r]

        return x

    def findOrthogonalVector(self):

        dim = self.cols()

        x = np.zeros(dim + 1)
        xn = np.zeros(self.n + 1)
        yn = np.zeros(self.n + 1)

        xn[0] = self.u
        xn[1] = self.v
        x[1] = self.u
        x[2] = self.v
        xptr = 3

        for _n in range(2, self.n + 1):
            yn[0] = - ((self.coal * (_n - 1.0) * _n) / 2.0) * xn[0]
            for r in range(1, _n):
                yn[r] = - ((self.coal * (r - 1.0) * _n) / 2.0) * xn[r - 1] - ((self.coal * (_n - 1.0 - r) * _n) / 2.0) * xn[r]

            yn[_n] = - ((self.coal * (_n - 1.0) * _n) / 2.0) * xn[_n - 1]

            xn = self.solveCentralBlockTransposed(_n, yn, 0)

            for i in range(0, len(xn)):
                x[xptr] = xn[i]
                xptr += 1

        return x


Q = SNPTransition(3, 1, 1, .2)
print(Q.Q)

