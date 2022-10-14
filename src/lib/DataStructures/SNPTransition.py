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
        self.n = n
        self.u = u
        self.v = v
        self.coal = coal

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

    def cols(self):
        return self.Q.shape[1]

    def solveCentralBlockTransposed(self, _n, y, offset):

        x = np.zeros(_n + 1)
        K = -(self.coal * (_n * (_n - 1.0))) / (2.0 - _n * self.v + offset)

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
                yn[r] = - ((self.coal * (r - 1.0) * _n) / 2.0) * xn[r - 1] - ((self.coal * (_n - 1.0 - r) * _n) / 2.0) * \
                        xn[r]

            yn[_n] = - ((self.coal * (_n - 1.0) * _n) / 2.0) * xn[_n - 1]

            xn = self.solveCentralBlockTransposed(_n, yn, 0)

            for i in range(0, len(xn)):
                x[xptr] = xn[i]
                xptr += 1

        return x
