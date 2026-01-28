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
Last Stable Edit : 3/11/25
First Included in Version : 1.0.0

Docs   - [x]
Tests  - [ ]
Design - [ ]
"""

from typing import Any, Union
import warnings
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
        
def _disable_for_subclass(method):
    """
    A decorator to disable a method for subclasses by raising a 
    NotImplementedError.
    
    Args:
        method (function): The method to be disabled.
    Returns:
        function: The wrapper function that raises the error
    """
    def wrapper(self, *args, **kwargs):
        # Check if the method is being called from an instance of a subclass
        if type(self) is not method.__qualname__.split('.')[0]:
            raise NotImplementedError(
                f"The method '{method.__name__}' is disabled \
                  for the subclass '{self.__class__.__name__}'."
            )
        return method(self, *args, **kwargs)

    return wrapper

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
        self.freqs : list[float] = base_freqs
        self.trans : list[float] = transitions
        
        self._is_valid(self.trans, self.freqs, self.states)

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
        Change any of the base frequencies/states/transitions parameters, and 
        recompute the Q matrix accordingly.
        
        Raises:
            SubstitutionModelError: If parameters are malformed/invalid.
        Args:
            params (dict[str, Any]): A mapping from gtr parameter names to 
                                        their values. For the GTR superclass,
                                        names must be limited to ["states", 
                                        "base frequencies", "transitions"]. 
                                        Parameter value type for "states" is an 
                                        int, parameter value type for 
                                        "base frequencies" and "transitions" is
                                        a list[float].
        Returns: 
            N/A
        """
        
        param_names = params.keys()
        
        if "states" in param_names:
            self.states = params["states"]
        if "transitions" in param_names:
            self.trans = params["transitions"]
        if "base frequencies" in param_names:
            self.freqs = params["base frequencies"]
        
        self._is_valid(self.trans, self.freqs, self.states)
        self.buildQ()
           
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
        Based on (1)

        Args:
            N/A
        Returns:
            np.ndarray: A numpy ndarray that represents the just built Q matrix.
        """
        self.Q = np.zeros((self.states, self.states), dtype = np.double)

        for i in range(self.states):
            for j in range(self.states):
                if j > i:
                    self.Q[i][j] = self.trans[i + j]
                elif i > j:
                    numerator = self.freqs[j] * self.trans[j + i - 1]
                    denominator = self.freqs[i]
                    
                    self.Q[i][j] = numerator / denominator

        for i in range(self.states):
            self.Q[i][i] = -1 * sum(self.Q[i, :])

        # normalize such that -1 * SUM Q_ii * pi_i = 1
        normFactor = 0
        for i in range(self.states):
            normFactor += (self.Q[i][i] * self.freqs[i])

        normFactor = -1 / normFactor

        # scale the matrix
        self.Q = self.Q * normFactor
        return self.Q

    def expt(self, t : float) -> np.ndarray:
        """
        Compute the matrix exponential e^(Q*t) and store the result.
        If the solution has been computed already but the Q matrix has not
        changed, simply return the value
        
        Args: 
            t (float): Generally going to be a positive number for phylogenetic
                       applications. Represents time, in coalescent units 
                       or any other unit.
        Returns:
            np.ndarray: A numpy ndarray that is the result of the matrix 
                        exponential with respect to Q and time t.
        """

        eigenvals, eigenvecs = lg.eigh(self.Q)
        q = eigenvecs
        qinv = lg.inv(q)
        diag = np.diag(eigenvals)
        diagt = np.zeros(np.shape(diag))
        
        for i in range(np.shape(diag)[0]):
            diagt[i][i] = math.exp(diag[i][i] * t)

        self.Qt = np.real(np.matmul(np.matmul(q, diagt), qinv))

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
        if len(freqs) != states or sum(freqs) != 1:
            raise SubstitutionModelError("Base frequency list either does not \
                                          sum to 1 or is not of correct length")

        proper_len = ((states - 1) * states) / 2
        if len(transitions) != proper_len:
            raise SubstitutionModelError(f"Incorrect number of transition \
                                          rates. Got {len(transitions)}. \
                                          Expected {proper_len}!")

class K80(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Kimura 2 parameter model from (2). Also known as K80.
    Parameterized by alpha and beta, the transversion and transition parameters.
    
    Base frequencies are assumed to be all equal at .25.
    Transition probabilities are = [alpha, beta, alpha, alpha, beta, alpha]
    """

    def __init__(self, alpha : float, beta : float) -> None:
        """
        Initialize K80 model.
        
        Raises:
            SubstitutionModelError: if alpha and beta do not sum to 1.
        
        Args:
            alpha (float): transversion param
            beta (float): transition param
        Returns:
            N/A
        """
        if alpha + beta != 1:
            raise SubstitutionModelError("K2P Transversion + Transition params \
                                          do not add to 1")

        bases = [.25, .25, .25, .25]
        trans = np.ones((6, 1)) * alpha
        trans[1] = beta
        trans[4] = beta
        self.beta = beta
        self.alpha = alpha
        super().__init__(bases, list(trans))
    
    def set_hyperparams(self, params : dict[str, float]) -> None:
        """
        Change any of the base frequencies/states/transitions parameters, and 
        recompute the Q matrix accordingly.
        
        Raises:
            SubstitutionModelError: If parameters are malformed/invalid.
        Args:
            params (dict[str, float ]): A mapping from gtr parameter names to 
                                        their values. For the K80 class,
                                        names must be limited to ["alpha", 
                                        "beta"].
        Returns:
            N/A
        """
        
        param_names = params.keys()
        
        if "alpha" in param_names:
            self.alpha = params["alpha"]
        if "beta" in param_names:
            self.beta = params["beta"]
            
        if self.alpha + self.beta != 1:
            raise SubstitutionModelError("Error. K80 Alpha and Beta parameters \
                                          do not sum to 1.")

        self.buildQ()

    def expt(self, t : float) -> np.ndarray:
        """
        Compute the matrix exponential e^(Q*t) and store the result.
        If the solution has been computed already but the Q matrix has not
        changed, simply return the value.
        
        For K2P, a closed form solution for e^(Q*t) exists and we do not need to
        perform any exponentiation.
        
        Args: 
            t (float): Generally going to be a positive number for phylogenetic
                       applications. Represents time, in coalescent units 
                       or any other unit.
        Returns:
            np.ndarray: A numpy ndarray that is the result of the matrix 
                        exponential with respect to Q and time t.
        """
        
        self.Qt = np.zeros((self.states, self.states), dtype = np.double)

        for i in range(self.states):
            for j in range(self.states):
                term1 = .25 * (1 - 2 * math.exp(-4 * t))
                term2 = math.exp(-8 * self.beta * t)
                
                self.Qt[i][j] = term1 + term2

        return self.Qt

class F81(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Formulated by Felsenstein in 1981, this substitution model assumes that 
    all base frequencies are free, but all transition probabilities are equal.
    
    A closed form for the matrix (Q) exponential exists.
    """

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
        trans = np.ones((6, 1))
        super().__init__(bases, list(trans))
        if len(self.freqs) != 4 or sum(self.freqs) != 1:
                raise SubstitutionModelError("F81 is only defined for 4 states\
                                             or your frequencies do not \
                                             sum to 1")
        
    
    @_disable_for_subclass
    def set_hyperparams(self, params : dict[str, list[float]]) -> None:
        """
        Change the base frequency parameter, and 
        recompute the Q matrix accordingly.

        Raises:
            SubstitutionModelError: If the base frequencies given do not sum to 
                                    1 or the list is over 4 elements long.
        Args:
            params (dict[str, list[float]]): A mapping from gtr parameter names 
                                             to their values. For the F81
                                             class, names must be limited 
                                             to ["base frequencies"].
        Returns:
            N/A
        """
        
        param_names = params.keys()
        
        if "base frequencies" in param_names:
            self.freqs = params["base frequencies"]
            if len(self.freqs) != 4 or sum(self.freqs) != 1:
                raise SubstitutionModelError("F81 is only defined for 4 states\
                                             or your frequencies do not \
                                             sum to 1")
            
        self.buildQ()

    # def expt(self, t:float) -> np.ndarray:
    #     """
    #     Compute the matrix exponential e^(Q*t) and store the result.
    #     If the solution has been computed already but the Q matrix has not
    #     changed, simply return the value.
        
    #     For F81, a closed form solution for e^(Q*t) exists and we do not need 
    #     to perform any exponentiation.
        
    #     Args: 
    #         t (float): Generally going to be a positive number for
    #                    phylogenetic applications. Represents time, in 
    #                    coalescent units or any other unit.
    #     """
    #     self.Qt = np.zeros((self.states, self.states))

    #     for i in range(self.states):
    #         for j in range(self.states):
    #             if i == j:
    #                 self.Qt[i][j] = .25 + (.75 * math.exp(-1.333333333 * t))
    #             else:
    #                 self.Qt[i][j] = .25 - (.25 * math.exp(-1.333333333 * t))

    #     return self.Qt

class JC(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    The Jukes Cantor model is the simplest of all time reversible models,
    in which all parameters (transitions, base frequencies) are assumed to be 
    equal.
    
    A closed form for the matrix exponential, e^(Q*t), exists.
    """
    
    def __init__(self) -> None:
        """
        No arguments need to be provided, as the JC Q matrix is fixed.
        
        Args:
            N/A
        Returns:
            N/A
        """
        bases = np.ones((4, 1)) * .25 
        super().__init__(list(bases), np.ones((6, 1)))
               
class HKY(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Developed by Hasegawa et al. Transversion parameters are assumed to be equal
    and the transition parameters are assumed to be equal. Base frequency 
    parameters are free.
    """

    def __init__(self, base_freqs : list[float], 
                 transitions : list[float]) -> None:
        """
        Initialize the HKY model with 4 base frequencies that sum to 1, and a
        transition array of length 6 with the equivalency pattern 
        [a, b, a, a, b, a].

        Raises:
            SubstitutionModelError: If inputs are malformed in any way.
        Args:
            base_freqs (list[float]): Array of 4 values that 
                                                         sum to 1.
            transitions (list[float]): Array of length 6 with
                                                          the equivalency 
                                                          pattern 
                                                          [a, b, a, a, b, a].
        Returns:
            N/A
        """
        
        self._is_valid(transitions, base_freqs, 4)

        super().__init__(base_freqs, transitions)
    
    def set_hyperparams(self, params : dict[str, list[float]]) -> None:
        """
        Change any of the base frequencies/states/transitions parameters, and 
        recompute the Q matrix accordingly.

        Raises:
            SubstitutionModelError: If parameters are malformed/invalid.
        Args:
            params (dict[str, list[float]]): A mapping from gtr parameter names 
                                             to their values. For the HKY class, 
                                             names must be limited to 
                                             ["base frequencies", "transitions"]
        Returns:
            N/A
        """
        
        param_names = params.keys()
        
        if "transitions" in param_names:
            self.trans = params["transitions"]
        if "base frequencies" in param_names:
            self.freqs = params["base frequencies"]
        
        self._is_valid(self.trans, self.freqs, 4)
        self.buildQ()
    
    def _is_valid(self, 
                  transitions: list[float], 
                  freqs : list[float],
                  states : int
                  ) -> None:
        """
        Ensure frequencies and transitions are well formed.
       
        Raises:
            SubstitutionModelError: If parameters are malformed/invalid.
        Args:
            transitions (list[float]): Transition list. Must be of length 6 and 
                                       the transitions must all be equal, and 
                                       all transversions must all be equal.
            freqs (list[float]): Base frequency list. Must be of length 4 
                                 and sum to 1.
            states (int): Number of states.
        Returns:
            N/A
        """
        
        if len(freqs) != states:
            raise SubstitutionModelError("Base frequency list must be of \
                                          length 4.")
        if sum(freqs) != 1:
            raise SubstitutionModelError("Base frequency list must sum to 1.")
        
        if transitions[0] != transitions[2] \
            or transitions[2] != transitions[3] \
            or transitions[3] != transitions[5]:
            raise SubstitutionModelError("HKY Transversions not all equal! \
                                          (Indeces 0, 2, 3, and 5)")

        if transitions[1] != transitions[4]:
            raise SubstitutionModelError("HKY Transitions not all equal! \
                                          (Indeces 1 and 4)")
        
class K81(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Developed by Kimura in 1981. Base frequencies are assumed to be equal, and 
    transition probabilities are assumed to be parameterized by
    the pattern [a, b, c, c, b, a].
    """
    
    def __init__(self, transitions : list[float]) -> None:
        """
        Initialize with a list of 6 transition probabilities that follow the 
        pattern [a, b, c, c, b, a]. All base frequencies are assumed to be
        equal.

        Raises:
            SubstitutionModelError: If the transition probabilities are not 
                                    of correct pattern.
        Args:
            transitions (list[float]): A list of floats, 6 long.
        Returns:
            N/A

        """
        base_freqs = [.25, .25, .25, .25]
        
        self._is_valid(transitions, base_freqs, 4)

        super().__init__(base_freqs, transitions)
    
    def set_hyperparams(self, params : dict[str, list[float]]) -> None:
        """
        Change the transitions parameters, and recompute the Q matrix 
        accordingly.

        Raises: 
            SubstitutionModelError: If the parameters are malformed/invalid.
        Args:
            params (dict[str, list[float]]): A mapping from gtr parameter names 
                                             to their values. For the K81 class, 
                                             names must be limited to
                                             ["transitions"].
        Returns:
            N/A
        """
        
        param_names = params.keys()
        
        if "transitions" in param_names:
            self.trans = params["transitions"]
    
        self._is_valid(self.trans, self.freqs, 4)
        self.buildQ()
    
    def _is_valid(self, 
                 transitions: list[float],
                 freqs : list[float],
                 states : int) -> None:
        """
        Ensure frequencies and transitions are well formed.

        Raises: 
            SubstitutionModelError: If the parameters are malformed/invalid.
        Args:
            transitions (list[float]): Transition list. Must be of length 6 and 
                                       the transitions must follow the 
                                       equivalency pattern of
                                       [a, b, c, c, b, a].
            freqs (list[float]): unused for this function.
            states (int): unused for this function.
        Returns:
            N/A
        """
        
        if transitions[0] != transitions[5] \
            or transitions[1] != transitions[4] \
                or transitions[2] != transitions[3]:
            raise SubstitutionModelError("K81 parameter mismatch!")
    
class SYM(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Developed by Zharkikh in 1994, this model assumes that all base frequencies 
    are equal, and all transition probabilities are free.
    """
    def __init__(self, transitions : list[float]) -> None:
        """
        Initialize with a list of 6 free transition probabilities. Base 
        frequencies are all equal.

        Raises:
            SubstitutionModelError: if the transitions array is not of length 6.
            
        Args:
            transitions (list[float]): A list of 6 transition rates.
        
        Returns:
            N/A

        """
        base_freqs = np.ones((4, 1)) * .25
               
        super().__init__(list(base_freqs), transitions)
    
    def set_hyperparams(self, params : dict[str, list[float]]) -> None:
        """
        Change any of the base frequencies/states/transitions parameters, and 
        recompute the Q matrix accordingly.
        
        Raises:
            SubstitutionModelError: if the transitions array is not of length 6
        Args:
            params (dict[str, list[float]]): A mapping from gtr parameter names 
                                             to their values. For the SYM class,
                                             names must be limited to 
                                             ["transitions"].
        Returns:
            N/A          
                    
        """
        
        param_names = params.keys()
        
        if "transitions" in param_names:
            self.trans = params["transitions"]
        
        if len(self.trans) != 6:
            raise SubstitutionModelError("Transition array must contain 6 \
                                          values.")
        self.buildQ()

class TN93(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Developed by Tamura and Nei in 1993. Similar to HKY, but two different 
    transition parameters are used instead of one (0=2=3=5, 1 != 4). Base 
    frequency parameters are free.
    """

    def __init__(self, 
                 base_freqs : list[float],
                 transitions : list[float]) -> None:
        """
        Initialize with a list of 4 free base frequencies, and 6 transitions 
        that follow the pattern [a, b, a, a, c, a].
        
        Raises:
            SubstitutionModelError: If the transitions or base frequency lists
                                    are malformed.

        Args:
            base_freqs (list[float]): A list of 4 base frequencies 
            transitions (list[float]): A list of 6 transitions that follow the
                                       above pattern.
        Returns:
            N/A
        """

        if transitions[0] != transitions[2] \
            or transitions[2] != transitions[3] \
            or transitions[3] != transitions[5]:
            raise SubstitutionModelError("TN93 Transversions not all equal")

        super().__init__(base_freqs, transitions)
        
    def set_hyperparams(self, params : dict[str, list[float]]) -> None:
        """
        Change any of the base frequencies/transitions parameters, and 
        recompute the Q matrix accordingly.

        Raises:
            SubstitutionModelError: If the new parameters are invalid.
        Args:
            params (dict[str, list[float]]): A mapping from gtr parameter names 
                                             to their values. For the TN93 
                                             class, names must be limited to 
                                             ["base frequencies", "transitions"]
        Returns:
            N/A
        """
        
        param_names = params.keys()
        
        if "transitions" in param_names:
            self.trans = params["transitions"]
        if "base frequencies" in param_names:
            self.freqs = params["base frequencies"]
        
        self._is_valid(self.trans, self.freqs, 4)
        self.buildQ()

    def _is_valid(self, 
                 transitions: list[float], 
                 freqs : list[float], 
                 states : int) -> None:
        """
        Ensure frequencies and transitions are well formed.
        
        Raises:
            SubstitutionModelError: If any of the inputs are malformed/invalid.

        Args:
            transitions (list[float]): Transition rate list.
            freqs (list[float]): Base frequency list. Must sum to 1.
            states (int): Number of states. For DNA, 4.
        Returns:
            N/A
        """
        
        # Check for malformed inputs
        if len(freqs) != 4 or sum(freqs) != 1:
            raise SubstitutionModelError("Base frequency list either does not \
                                          sum to 1 or is not of correct length")

        proper_len = ((states - 1) * states) / 2
        if len(transitions) != proper_len:
            raise SubstitutionModelError(f"Incorrect number of transition \
                                          rates. Got {len(transitions)}. \
                                          Expected {proper_len}!")
        
        if transitions[0] != transitions[2] \
            or transitions[2] != transitions[3] \
            or transitions[3] != transitions[5]:
            raise SubstitutionModelError("TN93 Transversions not all equal")

