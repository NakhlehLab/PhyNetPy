#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --                                                              
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

""" 
Author : Mark Kessler
Last Stable Edit : 8/3/26
First Included in Version : 1.0.0

Docs   - [x]
Tests  - [x]
Design - [x]

Time reversible substitution models for DNA.

All models share the Generalized Time Reversible parameterization: a symmetric
matrix of exchangeabilities scaled by the stationary base frequencies, giving
``Q[i][j] = r_ij * pi_j`` off the diagonal. Q is normalized so the expected
number of substitutions per unit time is 1, which makes the time argument to
:meth:`GTR.expt` a branch length in substitutions per site.

Each model takes exactly the free parameters its literature defines, rather than
a full exchangeability list that would have to be laid out in a particular
pattern:

    JC()                            no free parameters
    K80(kappa)                      transition/transversion ratio
    F81(base_freqs)                 free base frequencies
    HKY(base_freqs, kappa)          both of the above
    TN93(base_freqs, kappa_r, kappa_y)
                                    separate purine and pyrimidine ratios
    K81(alpha, beta, gamma)         three rate classes, uniform frequencies
    SYM(transitions)                six free rates, uniform frequencies
    GTR(base_freqs, transitions)    six free rates, free frequencies

Because Q is normalized, the overall scale of any rate argument is discarded and
only the ratios between rates are identifiable. Every rate argument is therefore
scale free: ``K81(1, 5, 2)`` and ``K81(2, 10, 4)`` are the same model.

Only :class:`SYM` and :class:`GTR` leave all six exchangeabilities free, so only
they take an explicit list. It is indexed in upper-triangular row-major order,
which for the states A, C, G, T is::

    index : 0     1     2     3     4     5
    pair  : A-C   A-G   A-T   C-G   C-T   G-T
    kind  : tv    ti    tv    tv    ti    tv

The two transitions are therefore indices 1 (A<->G) and 4 (C<->T), and the four
transversions are indices 0, 2, 3 and 5. This is the same order PhyloNet uses
for ``-gtr (piA piC piG piT, rAC rAG rAT rCG rCT rGT)``.

Six of the eight models have an exact closed form for ``e^(Q*t)`` and never
touch a matrix exponential:

    * JC   -- ``P_ii = 1/4 + 3/4 e^(-4t/3)``
    * F81  -- ``P = e^(-ut) I + (1 - e^(-ut)) 1 pi^T``
    * K80, HKY, TN93 -- the Tamura-Nei (1993) equations, of which K80 and HKY
      are special cases (see :func:`_tn93_p_matrix`)
    * K81  -- diagonalized by the characters of the Klein four-group
      (see :func:`_k3st_p_matrix`)

SYM and GTR have all six exchangeabilities free, so they fall back to a cached
eigendecomposition of Q.
"""

from typing import Any
import numpy as np
from numpy import linalg as lg
import math

"""
SOURCES:

1) Kimura 1980 (K80)

2) Felsenstein 1981 (F81)

3) Hasegawa et al. 1985 (HKY85)

4) Tamura and Nei, 1993 (TN93)

5) Kimura 1981 (K81)

6) Zharkikh 1994 (SYM)

7) Tavaré 1986 (GTR)

8) Jukes and Cantor 1969 (JC)
"""

#########################
#### EXCEPTION CLASS ####
#########################

class SubstitutionModelError(Exception):
    """
    Class of exception that gets raised when there is an error in the 
    formulation of a substitution model, whether it be inputs that don't 
    adhere to requirements or there is an issue in computation.
    """
    def __init__(self, 
                 message : str = "Unknown substitution model error") -> None:
        """
        Create a custom SubstitutionModelError with a custom message. To 
        be used in situations where substitution model calculations are 
        irrecoverably in err.

        Args:
            message (str, optional): Custom error message. Defaults to 
                                     "Unknown substitution model error".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)

#################################
#### NUCLEOTIDE CONVENTIONS  ####
#################################

# States are ordered A, C, G, T throughout. Purines (R) are {A, G} and
# pyrimidines (Y) are {C, T}, so the two *transitions* are A<->G and C<->T and
# the remaining four off-diagonal pairs are *transversions*.
#
# Exchangeability lists are indexed in upper-triangular row-major order:
#
#   index : 0     1     2     3     4     5
#   pair  : A-C   A-G   A-T   C-G   C-T   G-T
#   kind  : tv    ti    tv    tv    ti    tv
#
# Hence transitions live at indices 1 and 4, transversions at 0, 2, 3 and 5.

# Tolerance for checking that a base frequency vector sums to 1. Exact equality
# is unusable here: [0.4, 0.3, 0.2, 0.1] sums to 0.9999999999999999 in IEEE-754
# double precision.
_SUM_TOL = 1e-9


def _as_float_list(values : Any) -> list[float]:
    """
    Coerce any array-like of numbers into a flat list of Python floats.
    
    Callers routinely pass column vectors such as ``np.ones((6, 1))``, which
    iterate as length-1 arrays rather than scalars and silently poison every
    downstream calculation. Flattening here removes that whole failure mode.

    Args:
        values (Any): Any array-like (list, tuple, or numpy array) of numbers.
    Returns:
        list[float]: A flat list of Python floats.
    """
    return [float(v) for v in np.asarray(values, dtype = np.double).reshape(-1)]


def _positive(name : str, value : Any) -> float:
    """
    Validate that a rate or rate ratio is a positive, finite number.
    
    Raises:
        SubstitutionModelError: If the value is not positive and finite, or is
                                not a number at all.
    Args:
        name (str): Parameter name, used in the error message.
        value (Any): The value to check.
    Returns:
        float: The value as a Python float.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise SubstitutionModelError(f"{name} must be a number. Got \
                                       {value!r}.") from None

    if not math.isfinite(result) or result <= 0.0:
        raise SubstitutionModelError(f"{name} must be positive and finite. Got \
                                       {value!r}.")
    return result


def _rate_index(i : int, j : int, states : int) -> int:
    """
    Map an off-diagonal cell (i, j) to its exchangeability parameter index.
    
    Uses upper-triangular row-major order. Since the exchangeability matrix is
    symmetric, (i, j) and (j, i) map to the same index -- which is exactly what
    makes the resulting rate matrix time reversible.

    Args:
        i (int): Row (from-state) index.
        j (int): Column (to-state) index. Must differ from i.
        states (int): Total number of states.
    Returns:
        int: Index into the transitions list for the pair {i, j}.
    """
    if i > j:
        i, j = j, i
    return i * (2 * states - i - 1) // 2 + (j - i - 1)


def _tn93_p_matrix(freqs : list[float],
                   kappa_r : float,
                   kappa_y : float,
                   t : float) -> np.ndarray:
    """
    Closed-form transition matrix for the Tamura-Nei (1993) family.
    
    This single formula covers three of this module's models exactly, so none of
    them need a matrix exponential:
    
    * TN93 -- purine and pyrimidine ratios differ (kappa_r != kappa_y)
    * HKY  -- one shared ratio (kappa_r == kappa_y)
    * K80  -- HKY with uniform base frequencies
    
    The transversion rate is taken as 1, so the two kappas are ratios relative
    to it. Rates are normalized internally so the expected number of
    substitutions per unit time is 1, matching :meth:`GTR.buildQ`.
    
    Args:
        freqs (list[float]): Base frequencies in A, C, G, T order.
        kappa_r (float): Purine transition/transversion ratio (A<->G).
        kappa_y (float): Pyrimidine transition/transversion ratio (C<->T).
        t (float): Elapsed time, in expected substitutions per site.
    Returns:
        np.ndarray: A 4x4 row-stochastic transition probability matrix.
    """
    pi_a, pi_c, pi_g, pi_t = (float(x) for x in freqs)
    pi_r = pi_a + pi_g
    pi_y = pi_c + pi_t

    # Scale so that -sum_i pi_i Q_ii == 1.
    mean_rate = 2.0 * (pi_a * pi_g * kappa_r
                       + pi_c * pi_t * kappa_y
                       + pi_r * pi_y)
    alpha_r = kappa_r / mean_rate
    alpha_y = kappa_y / mean_rate
    beta = 1.0 / mean_rate

    # Two decay modes: the transversion eigenvalue (-beta), shared by both
    # groups, and the within-group eigenvalue, which differs between purines and
    # pyrimidines.
    decay_tv = math.exp(-beta * t)
    decay_r = math.exp(-(pi_r * alpha_r + (1.0 - pi_r) * beta) * t)
    decay_y = math.exp(-(pi_y * alpha_y + (1.0 - pi_y) * beta) * t)

    # Every same-group entry is pi_j times a per-group coefficient, and the
    # diagonal is that same product plus the group's decay term (the two
    # expressions differ by exactly one factor of e^(lambda t)). Every
    # cross-group entry is pi_j * (1 - decay_tv).
    coef_r = 1.0 + (1.0 / pi_r - 1.0) * decay_tv - decay_r / pi_r
    coef_y = 1.0 + (1.0 / pi_y - 1.0) * decay_tv - decay_y / pi_y
    cross = 1.0 - decay_tv

    # Cross-group (transversion) entries, per destination state.
    xa, xc, xg, xt = pi_a * cross, pi_c * cross, pi_g * cross, pi_t * cross
    # Same-group (transition) entries: A and G columns on the purine rows, C
    # and T columns on the pyrimidine rows.
    ra, rg = pi_a * coef_r, pi_g * coef_r
    yc, yt = pi_c * coef_y, pi_t * coef_y

    #                    A            C            G            T
    return np.array([[ra + decay_r,   xc,          rg,           xt],
                     [xa,             yc + decay_y, xg,          yt],
                     [ra,             xc,          rg + decay_r, xt],
                     [xa,             yc,          xg,           yt + decay_y]],
                    dtype = np.double)


def _k3st_p_matrix(alpha : float,
                   beta : float,
                   gamma : float,
                   t : float) -> np.ndarray:
    """
    Closed-form transition matrix for Kimura's 3-substitution-type model (K81).
    
    K81 assumes uniform base frequencies and three rate classes: transitions
    (A<->G, C<->T), and two transversion classes (A<->C / G<->T and
    A<->T / C<->G). Identifying the four nucleotides with the Klein four-group
    Z2 x Z2 makes Q a group-circulant matrix, which the group's characters
    diagonalize; the resulting eigenvalues are 0 and -2 times each pairwise sum
    of the three normalized rates.
    
    Rates are normalized internally so the expected substitution rate is 1.
    
    Args:
        alpha (float): Transition rate (A<->G, C<->T).
        beta (float): Rate for the A<->C and G<->T transversion class.
        gamma (float): Rate for the A<->T and C<->G transversion class.
        t (float): Elapsed time, in expected substitutions per site.
    Returns:
        np.ndarray: A 4x4 row-stochastic transition probability matrix.
    """
    # With uniform frequencies the normalization is just the rate sum, leaving
    # the three scaled rates summing to 1.
    total = alpha + beta + gamma
    a, b, g = alpha / total, beta / total, gamma / total

    e_ti = math.exp(-2.0 * (a + g) * t)
    e_ac = math.exp(-2.0 * (b + g) * t)
    e_at = math.exp(-2.0 * (a + b) * t)

    same = 0.25 * (1.0 + e_ti + e_ac + e_at)
    p_ti = 0.25 * (1.0 - e_ti + e_ac - e_at)
    p_ac = 0.25 * (1.0 + e_ti - e_ac - e_at)
    p_at = 0.25 * (1.0 - e_ti - e_ac + e_at)

    #        A     C     G     T
    return np.array([[same, p_ac, p_ti, p_at],
                     [p_ac, same, p_at, p_ti],
                     [p_ti, p_at, same, p_ac],
                     [p_at, p_ti, p_ac, same]], dtype = np.double)

#############################
#### SUBSTITUTION MODELS ####
#############################

class GTR:
    """
    General superclass for time reversable substitution models. Implements 
    Eigenvalue decomposition for computing e^(Q*t).
    
    Special case subclasses attempt to improve on the time 
    complexity of the matrix exponential operation.
    
    This is the Generalized Time Reversible (GTR) model.
    """

    #: Hyperparameter names accepted by :meth:`set_hyperparams`. Each subclass
    #: narrows this to the parameters it actually has, so that an unsupported
    #: name is reported rather than silently ignored (which would leave Q and
    #: the closed-form ``expt`` disagreeing).
    HYPERPARAMS : tuple[str, ...] = ("states",
                                     "base frequencies",
                                     "transitions")

    def __init__(self, 
                 base_freqs : list[float], 
                 transitions : list[float], 
                 states : int = 4) -> None:
        """
        Create a GTR substitution model object with the required/needed 
        parameters. 
        
        Raises:
            SubstitutionModelError: If the base frequency or transition arrays
                                    are malformed.
        Args:
            base_freqs (list[float]): An array of floats of 'states' length. 
                                      Must sum to 1.
            transitions (list[float]): An array of floats that is 
                                       ('states'^2 - 'states') / 2 long.
            states (int, optional): Number of possible data states.  
                                    Defaults to 4 (For DNA, {A, C, G, T}).
        Returns:
            N/A
        """

        self.states : int = states
        self.freqs : list[float] = _as_float_list(base_freqs)
        self.trans : list[float] = _as_float_list(transitions)
        
        self._is_valid(self.trans, self.freqs, self.states)

        # Cache of the eigendecomposition of Q, invalidated by buildQ().
        self._decomp : tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        # compute Q, the instantaneous probability matrix
        self.Q = self.buildQ()
        self.Qt = None

    def getQ(self) -> np.ndarray:
        """
        Get the Q matrix.

        Args:
            N/A
        Returns: 
            np.ndarray: numpy array object that represents the Q matrix
        """
        return self.Q

    def set_hyperparams(self, params : dict[str, Any]) -> None:
        """
        Change this model's free parameters and recompute the Q matrix.
        
        Accepted names are listed in this class's ``HYPERPARAMS``; anything else
        raises, since a silently ignored name would leave the caller believing
        a change took effect.
        
        Raises:
            SubstitutionModelError: If a parameter name is not one this model
                                    accepts, or if the resulting values are
                                    malformed/invalid.
        Args:
            params (dict[str, Any]): A mapping from parameter names to their
                                     values. For the GTR superclass the names
                                     are "states" (int), "base frequencies"
                                     (list[float]) and "transitions"
                                     (list[float]).
        Returns: 
            N/A
        """
        unknown = sorted(set(params) - set(self.HYPERPARAMS))
        if unknown:
            accepted = ", ".join(self.HYPERPARAMS) or "(none)"
            raise SubstitutionModelError(f"{type(self).__name__} does not have \
                                           hyperparameter(s) {unknown}. \
                                           Accepted: {accepted}.")

        self._apply_hyperparams(params)
        self._is_valid(self.trans, self.freqs, self.states)
        self.buildQ()

    def _apply_hyperparams(self, params : dict[str, Any]) -> None:
        """
        Write validated hyperparameter values onto this model.
        
        Subclasses override this to consume their own parameter names and
        regenerate ``self.trans`` accordingly. Validation of the resulting
        frequency and rate lists is handled by the caller.

        Args:
            params (dict[str, Any]): A mapping from parameter names to values,
                                     already checked against HYPERPARAMS.
        Returns:
            N/A
        """
        if "states" in params:
            self.states = int(params["states"])
        if "transitions" in params:
            self.trans = _as_float_list(params["transitions"])
        if "base frequencies" in params:
            self.freqs = _as_float_list(params["base frequencies"])

    def get_hyperparams(self) -> tuple[list[float], list[float]]:
        """
        Gets the base frequency and transition arrays.
        Args:
            N/A
        Returns:
            tuple[list[float], list[float]]: List that contains the base 
                                             frequencies in the first element, 
                                             and the transitions in the 
                                             second.
        """
        return self.freqs, self.trans

    def state_count(self) -> int:
        """
        Get the number of states for this substitution model.

        Args:
            N/A
        Returns:
            int: Number of states.
        """
        return self.states

    def buildQ(self) -> np.ndarray:
        """
        Populate the normalized Q matrix with the correct values.
        Based on (7), Tavaré's GTR parameterization.
        
        Off-diagonal entries are ``Q[i][j] = r_ij * pi_j``, where ``r_ij`` is
        the symmetric exchangeability for the pair {i, j}. Multiplying by
        the *destination* frequency is what makes Q time reversible, i.e.
        ``pi_i * Q[i][j] == pi_j * Q[j][i]``, with pi as its stationary
        distribution. Diagonal entries are set so that rows sum to 0, and the
        whole matrix is then scaled so the expected number of substitutions per
        unit time is 1 (``-sum_i pi_i Q_ii == 1``), which makes t directly
        interpretable as branch length in substitutions per site.

        Raises:
            SubstitutionModelError: If the resulting matrix is degenerate (a
                                    non-positive overall substitution rate).
        Args:
            N/A
        Returns:
            np.ndarray: A numpy ndarray that represents the just built Q matrix.
        """
        freqs = np.asarray(self.freqs, dtype = np.double)
        rates = np.asarray(self.trans, dtype = np.double)

        Q = np.zeros((self.states, self.states), dtype = np.double)

        for i in range(self.states):
            for j in range(self.states):
                if i != j:
                    Q[i][j] = rates[_rate_index(i, j, self.states)] * freqs[j]

        np.fill_diagonal(Q, -Q.sum(axis = 1))

        # normalize such that -1 * SUM Q_ii * pi_i = 1
        mean_rate = -float(freqs @ np.diag(Q))
        if mean_rate <= 0.0:
            raise SubstitutionModelError("Degenerate substitution model: the \
                                          overall substitution rate is not \
                                          positive.")

        self.Q = Q / mean_rate
        self._decomp = None
        return self.Q

    def _decompose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Eigendecompose Q, caching the result until Q next changes.
        
        Q is reversible but not symmetric, so it cannot be handed to a symmetric
        eigensolver directly. It is however *similar* to the symmetric matrix
        ``B = P^(1/2) Q P^(-1/2)`` (P = diag(pi)), so B is decomposed instead
        and the eigenvectors of Q recovered from it. This keeps the speed and
        guaranteed-real spectrum of a symmetric solver while staying correct for
        non-uniform base frequencies.

        Args:
            N/A
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Eigenvalues, the matrix
                                                       of right eigenvectors,
                                                       and its inverse.
        """
        if self._decomp is not None:
            return self._decomp

        freqs = np.asarray(self.freqs, dtype = np.double)

        if np.all(freqs > 0.0):
            sqrt_pi = np.sqrt(freqs)
            B = (self.Q * sqrt_pi[:, None]) / sqrt_pi[None, :]
            B = 0.5 * (B + B.T)          # symmetrize away round-off
            evals, evecs = lg.eigh(B)
            U = evecs / sqrt_pi[:, None]
            U_inv = evecs.T * sqrt_pi[None, :]
        else:
            # A zero base frequency breaks the similarity transform; fall back
            # to the general (slower) solver.
            evals, U = lg.eig(self.Q)
            evals = np.real(evals)
            U = np.real(U)
            U_inv = lg.inv(U)

        self._decomp = (evals, U, U_inv)
        return self._decomp

    def expt(self, t : float) -> np.ndarray:
        """
        Compute the matrix exponential e^(Q*t) and store the result.
        The eigendecomposition of Q is cached, so repeated calls with different
        values of t only cost a rescaling and one matrix product.
        
        Args: 
            t (float): Generally going to be a positive number for phylogenetic
                       applications. Represents time, in coalescent units 
                       or any other unit. Negative values are clamped to 0.
        Returns:
            np.ndarray: A numpy ndarray that is the result of the matrix 
                        exponential with respect to Q and time t. Rows are a
                        probability distribution over the destination state, so
                        each sums to 1.
        """
        if t < 0.0:
            t = 0.0

        evals, U, U_inv = self._decompose()
        self.Qt = (U * np.exp(evals * t)[None, :]) @ U_inv

        return self.Qt
    
    def _is_valid(self, 
                  transitions: list[float], 
                  freqs : list[float], 
                  states : int) -> None:
        """
        Ensure frequencies and transitions are well formed.
        
        Raises:
            SubstitutionModelError: If transitions or frequencies are malformed.
        Args:
            transitions (list[float]): Transition list.
            freqs (list[float]): Base frequency list. Must sum to 1.
            states (int): Number of states.
        Returns:
            N/A
        """
        
        # Check for malformed inputs
        if len(freqs) != states \
            or not math.isclose(sum(freqs), 1.0, abs_tol = _SUM_TOL):
            raise SubstitutionModelError("Base frequency list either does not \
                                          sum to 1 or is not of correct length")

        proper_len = ((states - 1) * states) // 2
        if len(transitions) != proper_len:
            raise SubstitutionModelError(f"Incorrect number of transition \
                                          rates. Got {len(transitions)}. \
                                          Expected {proper_len}!")

class JC(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    The Jukes Cantor model is the simplest of all time reversible models,
    in which all parameters (transitions, base frequencies) are assumed to be 
    equal.
    
    JC has no free parameters, so :meth:`set_hyperparams` rejects everything.
    
    A closed form for the matrix exponential, e^(Q*t), exists.
    """

    HYPERPARAMS : tuple[str, ...] = ()

    def __init__(self) -> None:
        """
        No arguments need to be provided, as the JC Q matrix is fixed.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__([0.25] * 4, [1.0] * 6)

    def expt(self, t : float) -> np.ndarray:
        """
        Compute the matrix exponential e^(Q*t) and store the result.
        
        For JC, a closed form solution for e^(Q*t) exists and we do not need to
        perform any exponentiation. With all rates and frequencies equal, the
        normalized Q has off-diagonals 1/3 and diagonal -1, giving
        
            P(no change) = 1/4 + 3/4 e^(-4t/3)
            P(change)    = 1/4 - 1/4 e^(-4t/3)
        
        Args:
            t (float): Generally going to be a positive number for phylogenetic
                       applications. Represents time, in coalescent units or any
                       other unit. Negative values are clamped to 0.
        Returns:
            np.ndarray: A numpy ndarray that is the result of the matrix
                        exponential with respect to Q and time t.
        """
        if t < 0.0:
            t = 0.0

        decay = math.exp(-4.0 * t / 3.0)
        self.Qt = np.full((4, 4), 0.25 - 0.25 * decay, dtype = np.double)
        np.fill_diagonal(self.Qt, 0.25 + 0.75 * decay)
        return self.Qt

class K80(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Kimura 2 parameter model from (1). Also known as K80 or K2P.
    
    Parameterized by ``kappa``, the transition/transversion rate ratio. This is
    the standard parameterization: because Q is normalized to unit mean rate,
    only the *ratio* of the two rates is identifiable, so kappa is the model's
    single free parameter. It is also how BEAST, MrBayes, RAxML, PAML and
    PhyloNet expose K80/HKY, and how this library's own sequence-likelihood
    engine takes HKY85. ``kappa == 1`` gives Jukes-Cantor, and empirical DNA
    estimates usually fall in the 2-10 range.
    
    Kimura's original notation writes the rate matrix with a transition rate
    alpha and a transversion rate beta, where ``kappa = alpha / beta``. Both are
    available as read-only properties holding the normalized rates, which
    satisfy ``alpha + 2 * beta == 1`` and are exactly the corresponding entries
    of Q.
    
    Base frequencies are fixed at .25.
    """

    HYPERPARAMS : tuple[str, ...] = ("kappa",)

    def __init__(self, kappa : float) -> None:
        """
        Initialize a K80 model from a transition/transversion rate ratio.
        
        Raises:
            SubstitutionModelError: If kappa is not positive and finite.
        Args:
            kappa (float): Transition/transversion rate ratio. ``kappa == 1``
                           is Jukes-Cantor. If you have separate transition and
                           transversion rates alpha and beta, pass
                           ``alpha / beta``.
        Returns:
            N/A
        """
        self.kappa = _positive("kappa", kappa)
        super().__init__([.25] * 4, self._rates())

    @property
    def alpha(self) -> float:
        """
        Normalized transition rate (A<->G, C<->T), in Kimura's notation.
        
        Equals ``Q[A][G]``, and satisfies ``alpha / beta == kappa`` and
        ``alpha + 2 * beta == 1``.

        Args:
            N/A
        Returns:
            float: kappa / (kappa + 2)
        """
        return self.kappa / (self.kappa + 2.0)

    @property
    def beta(self) -> float:
        """
        Normalized transversion rate, in Kimura's notation.
        
        Equals ``Q[A][C]``, and satisfies ``alpha + 2 * beta == 1``.

        Args:
            N/A
        Returns:
            float: 1 / (kappa + 2)
        """
        return 1.0 / (self.kappa + 2.0)

    def _rates(self) -> list[float]:
        """
        Build the exchangeability list from kappa.
        
        Only the ratio matters, so transversions take 1 and transitions kappa.

        Args:
            N/A
        Returns:
            list[float]: [1, kappa, 1, 1, kappa, 1] in AC, AG, AT, CG, CT, GT
                         order, placing kappa on A<->G and C<->T.
        """
        k = self.kappa
        return [1.0, k, 1.0, 1.0, k, 1.0]

    def _apply_hyperparams(self, params : dict[str, Any]) -> None:
        """
        Update kappa and regenerate the exchangeability list.

        Raises:
            SubstitutionModelError: If kappa is not positive and finite.
        Args:
            params (dict[str, Any]): May contain "kappa".
        Returns:
            N/A
        """
        if "kappa" in params:
            self.kappa = _positive("kappa", params["kappa"])
            self.trans = self._rates()

    def expt(self, t : float) -> np.ndarray:
        """
        Compute the matrix exponential e^(Q*t) and store the result.
        
        For K2P, a closed form solution for e^(Q*t) exists and we do not need to
        perform any exponentiation. K80 is HKY with uniform base frequencies, so
        this defers to the shared Tamura-Nei kernel with a single transition
        rate. In terms of the normalized rates alpha (transition) and beta
        (transversion), where alpha + 2 beta = 1, that kernel reduces to the
        familiar Kimura 2-parameter equations:
        
            P(no change)    = 1/4 + 1/4 e^(-4 beta t)
                                  + 1/2 e^(-2 (alpha + beta) t)
            P(transition)   = 1/4 + 1/4 e^(-4 beta t)
                                  - 1/2 e^(-2 (alpha + beta) t)
            P(transversion) = 1/4 - 1/4 e^(-4 beta t)
        
        Args: 
            t (float): Generally going to be a positive number for phylogenetic
                       applications. Represents time, in coalescent units 
                       or any other unit. Negative values are clamped to 0.
        Returns:
            np.ndarray: A numpy ndarray that is the result of the matrix 
                        exponential with respect to Q and time t.
        """
        if t < 0.0:
            t = 0.0

        self.Qt = _tn93_p_matrix(self.freqs, self.kappa, self.kappa, t)
        return self.Qt

class F81(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Formulated by Felsenstein in 1981, this substitution model assumes that 
    all base frequencies are free, but all transition probabilities are equal.
    
    A closed form for the matrix (Q) exponential exists.
    """

    HYPERPARAMS : tuple[str, ...] = ("base frequencies",)

    def __init__(self, bases : list[float]) -> None:
        """
        Initialize the F81 model with a list of base frequencies of length 4.
        Transition probabilities will all be the same.

        Raises:
            SubstitutionModelError: If the base frequencies given do not sum to 
                                    1 or if the list does not have exactly 4 
                                    elements.
        Args:
            bases (list[float]): a list of 4 base frequency values.
        Returns:
            N/A
        """
        super().__init__(bases, [1.0] * 6)

    def expt(self, t : float) -> np.ndarray:
        """
        Compute the matrix exponential e^(Q*t) and store the result.
        
        For F81, a closed form solution for e^(Q*t) exists and we do not need
        to perform any exponentiation. Because every exchangeability is equal,
        Q collapses to ``u * (1 pi^T - I)``, whose exponential is
        
            P(t) = e^(-u t) I + (1 - e^(-u t)) 1 pi^T
        
        i.e. ``P[i][j] = pi_j + (delta_ij - pi_j) e^(-u t)``, with
        ``u = 1 / (1 - sum_i pi_i^2)`` fixing the expected substitution rate at
        1. Setting pi = 1/4 recovers the Jukes-Cantor equations with u = 4/3.
        
        Args: 
            t (float): Generally going to be a positive number for
                       phylogenetic applications. Represents time, in 
                       coalescent units or any other unit. Negative values are
                       clamped to 0.
        Returns:
            np.ndarray: A numpy ndarray that is the result of the matrix 
                        exponential with respect to Q and time t.
        """
        if t < 0.0:
            t = 0.0

        pi_a, pi_c, pi_g, pi_t = self.freqs
        u = 1.0 / (1.0 - (pi_a * pi_a + pi_c * pi_c
                          + pi_g * pi_g + pi_t * pi_t))
        decay = math.exp(-u * t)

        spread = 1.0 - decay
        a, c, g, tt = (pi_a * spread, pi_c * spread,
                       pi_g * spread, pi_t * spread)

        self.Qt = np.array([[a + decay, c,         g,         tt        ],
                            [a,         c + decay, g,         tt        ],
                            [a,         c,         g + decay, tt        ],
                            [a,         c,         g,         tt + decay]],
                           dtype = np.double)
        return self.Qt

class HKY(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Developed by Hasegawa et al. (3). Free base frequencies plus a single
    transition/transversion rate ratio, ``kappa``: all four transversions share
    one rate and both transitions share another.
    
    HKY sits between F81 (``kappa == 1``) and TN93 (which frees the purine and
    pyrimidine ratios from each other), and reduces to K80 when the base
    frequencies are uniform.
    """

    HYPERPARAMS : tuple[str, ...] = ("kappa", "base frequencies")

    def __init__(self, base_freqs : list[float], kappa : float) -> None:
        """
        Initialize the HKY model with 4 base frequencies that sum to 1 and a
        transition/transversion rate ratio.

        Raises:
            SubstitutionModelError: If the base frequencies are malformed or
                                    kappa is not positive and finite.
        Args:
            base_freqs (list[float]): Array of 4 values that sum to 1, in
                                      A, C, G, T order.
            kappa (float): Transition/transversion rate ratio. ``kappa == 1``
                           gives F81.
        Returns:
            N/A
        """
        self.kappa = _positive("kappa", kappa)
        super().__init__(base_freqs, self._rates())

    def _rates(self) -> list[float]:
        """
        Build the exchangeability list from kappa.

        Args:
            N/A
        Returns:
            list[float]: [1, kappa, 1, 1, kappa, 1] in AC, AG, AT, CG, CT, GT
                         order, placing kappa on A<->G and C<->T.
        """
        k = self.kappa
        return [1.0, k, 1.0, 1.0, k, 1.0]

    def _apply_hyperparams(self, params : dict[str, Any]) -> None:
        """
        Update kappa and/or the base frequencies.

        Raises:
            SubstitutionModelError: If kappa is not positive and finite.
        Args:
            params (dict[str, Any]): May contain "kappa" and
                                     "base frequencies".
        Returns:
            N/A
        """
        if "kappa" in params:
            self.kappa = _positive("kappa", params["kappa"])
            self.trans = self._rates()
        if "base frequencies" in params:
            self.freqs = _as_float_list(params["base frequencies"])

    def expt(self, t : float) -> np.ndarray:
        """
        Compute the matrix exponential e^(Q*t) and store the result.
        
        HKY is the special case of TN93 in which the purine and pyrimidine
        transition rates are equal, so the Tamura-Nei closed form applies and no
        exponentiation is needed.
        
        Args:
            t (float): Generally going to be a positive number for phylogenetic
                       applications. Represents time, in coalescent units or any
                       other unit. Negative values are clamped to 0.
        Returns:
            np.ndarray: A numpy ndarray that is the result of the matrix
                        exponential with respect to Q and time t.
        """
        if t < 0.0:
            t = 0.0

        self.Qt = _tn93_p_matrix(self.freqs, self.kappa, self.kappa, t)
        return self.Qt

class TN93(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Developed by Tamura and Nei in 1993 (4). Like HKY, but the purine
    transition A<->G and the pyrimidine transition C<->T get separate
    transition/transversion ratios instead of sharing one. Base frequencies are
    free.
    
    This is the most general model in the module that still has a closed-form
    matrix exponential; HKY (``kappa_r == kappa_y``), K80 and F81 are all
    special cases of it.
    """

    HYPERPARAMS : tuple[str, ...] = ("kappa_r", "kappa_y", "base frequencies")

    def __init__(self, 
                 base_freqs : list[float],
                 kappa_r : float,
                 kappa_y : float) -> None:
        """
        Initialize with 4 base frequencies that sum to 1 and the two
        transition/transversion rate ratios.
        
        Raises:
            SubstitutionModelError: If the base frequencies are malformed or
                                    either ratio is not positive and finite.
        Args:
            base_freqs (list[float]): A list of 4 base frequencies, in
                                      A, C, G, T order.
            kappa_r (float): Purine (R = A/G) transition/transversion ratio,
                             governing A<->G.
            kappa_y (float): Pyrimidine (Y = C/T) transition/transversion ratio,
                             governing C<->T.
        Returns:
            N/A
        """
        self.kappa_r = _positive("kappa_r", kappa_r)
        self.kappa_y = _positive("kappa_y", kappa_y)
        super().__init__(base_freqs, self._rates())

    def _rates(self) -> list[float]:
        """
        Build the exchangeability list from the two ratios.

        Args:
            N/A
        Returns:
            list[float]: [1, kappa_r, 1, 1, kappa_y, 1] in AC, AG, AT, CG, CT,
                         GT order, placing kappa_r on A<->G and kappa_y on
                         C<->T.
        """
        return [1.0, self.kappa_r, 1.0, 1.0, self.kappa_y, 1.0]

    def _apply_hyperparams(self, params : dict[str, Any]) -> None:
        """
        Update either ratio and/or the base frequencies.

        Raises:
            SubstitutionModelError: If a ratio is not positive and finite.
        Args:
            params (dict[str, Any]): May contain "kappa_r", "kappa_y" and
                                     "base frequencies".
        Returns:
            N/A
        """
        if "kappa_r" in params:
            self.kappa_r = _positive("kappa_r", params["kappa_r"])
        if "kappa_y" in params:
            self.kappa_y = _positive("kappa_y", params["kappa_y"])
        if "kappa_r" in params or "kappa_y" in params:
            self.trans = self._rates()
        if "base frequencies" in params:
            self.freqs = _as_float_list(params["base frequencies"])

    def expt(self, t : float) -> np.ndarray:
        """
        Compute the matrix exponential e^(Q*t) and store the result.
        
        TN93 admits a closed form (see :func:`_tn93_p_matrix`), so no
        exponentiation is needed.
        
        Args:
            t (float): Generally going to be a positive number for phylogenetic
                       applications. Represents time, in coalescent units or any
                       other unit. Negative values are clamped to 0.
        Returns:
            np.ndarray: A numpy ndarray that is the result of the matrix
                        exponential with respect to Q and time t.
        """
        if t < 0.0:
            t = 0.0

        self.Qt = _tn93_p_matrix(self.freqs, self.kappa_r, self.kappa_y, t)
        return self.Qt

class K81(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Developed by Kimura in 1981 (5). Also known as K3ST or K3P. Base
    frequencies are equal, and the six substitution types collapse into three
    rate classes:
    
        alpha -- transitions,          A<->G and C<->T
        beta  -- transversions,        A<->C and G<->T
        gamma -- transversions,        A<->T and C<->G
    
    K80 is the special case ``beta == gamma``. The rates are scale free, since
    normalizing Q discards their overall magnitude and keeps only their ratios;
    they are stored exactly as given, so ``alpha``, ``beta`` and ``gamma`` read
    back the values passed in. Use :meth:`getQ` for the normalized rates.
    """

    HYPERPARAMS : tuple[str, ...] = ("alpha", "beta", "gamma")

    def __init__(self, alpha : float, beta : float, gamma : float) -> None:
        """
        Initialize with the three K3ST rate classes.
        
        Only the ratios between the three matter, so ``K81(1, 5, 2)`` and
        ``K81(2, 10, 4)`` are the same model.

        Raises:
            SubstitutionModelError: If any rate is not positive and finite.
        Args:
            alpha (float): Transition rate, shared by A<->G and C<->T.
            beta (float): Transversion rate, shared by A<->C and G<->T.
            gamma (float): Transversion rate, shared by A<->T and C<->G.
        Returns:
            N/A
        """
        self._store_rates(alpha, beta, gamma)
        super().__init__([.25] * 4, self._rates())

    def _store_rates(self, alpha : float, beta : float,
                     gamma : float) -> None:
        """
        Validate the three rate classes and store them as given.
        
        Storing the raw values rather than normalized ones keeps a partial
        update meaningful: ``set_hyperparams({"alpha": 9})`` on ``K81(5, 1, 2)``
        gives the ratios 9:1:2, which it could not if beta and gamma had already
        been rescaled.
        
        Raises:
            SubstitutionModelError: If any rate is not positive and finite.
        Args:
            alpha (float): Transition rate.
            beta (float): A<->C / G<->T transversion rate.
            gamma (float): A<->T / C<->G transversion rate.
        Returns:
            N/A
        """
        self.alpha = _positive("alpha", alpha)
        self.beta = _positive("beta", beta)
        self.gamma = _positive("gamma", gamma)

    def _rates(self) -> list[float]:
        """
        Build the exchangeability list from the three rate classes.

        Args:
            N/A
        Returns:
            list[float]: [beta, alpha, gamma, gamma, alpha, beta] in
                         AC, AG, AT, CG, CT, GT order.
        """
        return [self.beta, self.alpha, self.gamma,
                self.gamma, self.alpha, self.beta]

    def _apply_hyperparams(self, params : dict[str, Any]) -> None:
        """
        Update any of the three rate classes.
        
        Values not named in ``params`` keep their current value, so a partial
        update is interpreted relative to the rates already in place.

        Raises:
            SubstitutionModelError: If any rate is not positive and finite.
        Args:
            params (dict[str, Any]): May contain "alpha", "beta" and "gamma".
        Returns:
            N/A
        """
        self._store_rates(params.get("alpha", self.alpha),
                          params.get("beta", self.beta),
                          params.get("gamma", self.gamma))
        self.trans = self._rates()

    def expt(self, t : float) -> np.ndarray:
        """
        Compute the matrix exponential e^(Q*t) and store the result.
        
        K81 has three rate classes over uniform base frequencies, which the
        characters of the Klein four-group diagonalize exactly, so a closed form
        exists and no exponentiation is needed. See :func:`_k3st_p_matrix`.
        
        Args:
            t (float): Generally going to be a positive number for phylogenetic
                       applications. Represents time, in coalescent units or any
                       other unit. Negative values are clamped to 0.
        Returns:
            np.ndarray: A numpy ndarray that is the result of the matrix
                        exponential with respect to Q and time t.
        """
        if t < 0.0:
            t = 0.0

        self.Qt = _k3st_p_matrix(self.alpha, self.beta, self.gamma, t)
        return self.Qt

class SYM(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Developed by Zharkikh in 1994 (6), this model assumes that all base 
    frequencies are equal, and all six exchangeabilities are free.
    
    Because nothing constrains the six rates, SYM has no closed form for
    ``e^(Q*t)`` and inherits :meth:`GTR.expt`. Uniform base frequencies do make
    Q symmetric, so that cached eigendecomposition is exact and cheap.
    """

    HYPERPARAMS : tuple[str, ...] = ("transitions",)

    def __init__(self, transitions : list[float]) -> None:
        """
        Initialize with a list of 6 free transition probabilities. Base 
        frequencies are all equal.

        Raises:
            SubstitutionModelError: if the transitions array is not of length 6.
            
        Args:
            transitions (list[float]): A list of 6 transition rates, in
                                       AC, AG, AT, CG, CT, GT order.
        Returns:
            N/A
        """
        super().__init__([.25] * 4, transitions)

    def _apply_hyperparams(self, params : dict[str, Any]) -> None:
        """
        Update the exchangeability list.

        Args:
            params (dict[str, Any]): May contain "transitions".
        Returns:
            N/A
        """
        if "transitions" in params:
            self.trans = _as_float_list(params["transitions"])
