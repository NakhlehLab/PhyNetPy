""" 
Author : Mark Kessler
Last Stable Edit : 4/12/24
First Included in Version : 1.0.0
Approved for Release : No. Ensure exponential closed form accuracy.
"""

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
    def __init__(self, message = "Unknown substitution model error") -> None:
        self.message = message
        super().__init__(self.message)

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

    def __init__(self, base_freqs : list[float] | np.ndarray, 
                       transitions : list[float] | np.ndarray, 
                       states : int = 4):
        """
        Args:
            base_freqs (list[float] | np.ndarray): an array of floats of 
                                                   'states' length. Must sum
                                                   to 1.
            transitions (list[float] | np.ndarray): an array of floats that is 
                                                    ('states'^2 - 'states) / 2 
                                                    long.
            states (int, optional): Number of possible data states.  
                                    Defaults to 4 (For DNA, {A, C, G, T}).

        Raises:
            SubstitutionModelError: If the base frequency or transition arrays
                                    are malformed.
        """

        self.states : int = states
        self.freqs : list[float] | np.ndarray = base_freqs
        self.trans : list[float] | np.ndarray = transitions
        
        self.is_valid(self.trans, self.freqs, self.states)

        # compute Q, the instantaneous probability matrix
        self.Q = self.buildQ()
        self.Qt = None

    def getQ(self) -> np.ndarray:
        """
        Get the Q matrix

        Returns: np array obj
        """
        return self.Q

    def set_hyperparams(self, params : dict[str, object]) -> None:
        """
        Change any of the base frequencies/states/transitions parameters, and 
        recompute the Q matrix accordingly.

        Args:
            params (dict[str, object]): A mapping from gtr parameter names to 
                                        their values. For the GTR superclass,
                                        names must be limited to ["states", 
                                        "base frequencies", "transitions"]
        """
        
        param_names = params.keys()
        
        if "states" in param_names:
            self.states = params["states"]
        if "transitions" in param_names:
            self.trans = params["transitions"]
        if "base frequencies" in param_names:
            self.freqs = params["base frequencies"]
        
        self.is_valid(self.trans, self.freqs, self.states)
        self.buildQ()
           
    def get_hyperparams(self) -> list[list[float] | np.ndarray]:
        """
        Gets the base frequency and transition arrays.

        Returns:
            list[list[float] | np.ndarray]: List that contains the base 
                                            frequencies in the first element, 
                                            and the transitions in the second.
        """
        return self.freqs, self.trans

    def state_count(self) -> int:
        """
        Get the number of states for this substitution model.

        Returns:
            int: Number of states.
        """
        return self.states

    def buildQ(self) -> np.ndarray:
        """
        Populate the normalized Q matrix with the correct values.
        Based on (1)
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
    
    def is_valid(self, transitions: list[float] | np.ndarray, 
                 freqs : list[float] | np.ndarray, states : int) -> None:
        """
        Ensure frequencies and transitions are well formed.

        Args:
            transitions (list[float] | np.ndarray): Transition list.
            freqs (list[float] | np.ndarray): Base frequency list. 
                                              Must sum to 1.
        """
        
        # Check for malformed inputs
        if len(freqs) != states or sum(freqs) != 1:
            raise SubstitutionModelError("Base frequency list either does not \
                                          sum to 1 or is not of correct length")

        proper_len = ((states - 1) * states) / 2
        if len(transitions) != proper_len:
            raise SubstitutionModelError(f"Incorrect number of transition \
                                          rates. Got {len(transitions)}. 
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
        Args:
            alpha (float): transversion param
            beta (float): transition param

        Raises:
            SubstitutionModelError: if alpha and beta do not sum to 1.
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
        super().__init__(bases, trans)
    
    def set_hyperparams(self, params : dict[str, object]) -> None:
        """
        Change any of the base frequencies/states/transitions parameters, and 
        recompute the Q matrix accordingly.

        Args:
            params (dict[str, object]): A mapping from gtr parameter names to 
                                        their values. For the K80 class,
                                        names must be limited to ["alpha", 
                                        "beta"]
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

    def __init__(self, bases : list[float] | np.ndarray):
        """
        Initialize the F81 model with a list of base frequencies of length 4.
        Transition probabilities will all be the same.

        Args:
            bases (list[float] | np.ndarray): a list of 4 base frequency values.
        """
        trans = np.ones((6, 1))
        if len(self.freqs) != 4 or sum(self.freqs) != 1:
                raise SubstitutionModelError("F81 is only defined for 4 states\
                                             or your frequencies do not \
                                             sum to 1")
        super().__init__(bases, trans)
    
    def set_hyperparams(self, params : dict[str, object]) -> None:
        """
        Change any of the base frequencies/states/transitions parameters, and 
        recompute the Q matrix accordingly.

        Args:
            params (dict[str, object]): A mapping from gtr parameter names to 
                                        their values. For the GTR superclass,
                                        names must be limited to 
                                        ["base frequencies"].
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

class JC(F81):
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
        """
        bases = np.ones((4, 1)) * .25 
        super().__init__(bases)
        
    def expt(self, t : float) -> np.ndarray:
        """
        TODO: Fill in closed form solution

        Args:
            t (float): Time, in coalescent units.

        Returns:
            np.ndarray: e^(Q*t)
        """
        return super().expt(t)
    
    def set_hyperparams(self, params : dict[str, object]) -> None:
        warnings.warn("Attempting to set parameters for the Jukes Cantor model.\
                       No parameters are needed for this model, and whatever\
                       operation was attempted will have no effect.")
        return 
         
class HKY(GTR):
    """
    For DNA only (4 states, 6 transitions).
    
    Developed by Hasegawa et al. Transversion parameters are assumed to be equal
    and the transition parameters are assumed to be equal. Base frequency 
    parameters are free.
    """

    def __init__(self, base_freqs : list | np.ndarray, 
                 transitions : list | np.ndarray) -> None:
        """
        Initialize the HKY model with 4 base frequencies that sum to 1, and a
        transition array of length 6 with the equivalency pattern 
        [a, b, a, a, b, a].

        Args:
            base_freqs (list | np.ndarray): Array of 4 values that sum to 1.
            transitions (list | np.ndarray): Array of length 6 with the 
                                             equivalency pattern 
                                             [a, b, a, a, b, a].
        """
        
        self.is_valid(transitions, base_freqs)

        super().__init__(base_freqs, transitions)
    
    def set_hyperparams(self, params : dict[str, object]) -> None:
        """
        Change any of the base frequencies/states/transitions parameters, and 
        recompute the Q matrix accordingly.

        Args:
            params (dict[str, object]): A mapping from gtr parameter names to 
                                        their values. For the HKY class,
                                        names must be limited to 
                                        ["base frequencies", "transitions"]
        """
        
        param_names = params.keys()
        
        if "transitions" in param_names:
            self.trans = params["transitions"]
        if "base frequencies" in param_names:
            self.freqs = params["base frequencies"]
        
        self.is_valid(self.trans, self.freqs)
        self.buildQ()
    
    def is_valid(self, transitions: list[float] | np.ndarray, 
                 freqs : list[float] | np.ndarray) -> None:
        """
        Ensure frequencies and transitions are well formed.

        Args:
            transitions (list[float] | np.ndarray): Transition list. Must be of 
                                                    length 6 and the transitions 
                                                    must all be equal, and all 
                                                    transversions must all be 
                                                    equal.
            freqs (list[float] | np.ndarray): Base frequency list. Must be of 
                                              length 4 and sum to 1.
        """
        
        if len(freqs) != 4:
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
    
    def __init__(self, transitions : list[float] | np.ndarray) -> None:
        """
        Initialize with a list of 6 transition probabilities that follow the 
        pattern [a, b, c, c, b, a]. All base frequencies are assumed to be
        equal.

        Args:
            transitions (list[float] | np.ndarray): A list of floats, 6 long.

        Raises:
            SubstitutionModelError: if the transition probabilities are not 
                                    of correct pattern.
        """
        
        self.is_valid(transitions)

        base_freqs = [.25, .25, .25, .25]

        super().__init__(base_freqs, transitions)
    
    def set_hyperparams(self, params : dict[str, object]) -> None:
        """
        Change the transitions parameters, and recompute the Q matrix 
        accordingly.

        Args:
            params (dict[str, object]): A mapping from gtr parameter names to 
                                        their values. For the K81 class,
                                        names must be limited to 
                                        ["transitions"]
        """
        
        param_names = params.keys()
        
        if "transitions" in param_names:
            self.trans = params["transitions"]
    
        self.is_valid(self.trans)
        self.buildQ()
    
    def is_valid(self, transitions: list[float] | np.ndarray) -> None:
        """
        Ensure frequencies and transitions are well formed.

        Args:
            transitions (list[float] | np.ndarray): Transition list. Must be of 
                                                    length 6 and the transitions 
                                                    must follow the equivalency 
                                                    pattern of
                                                    [a, b, c, c, b, a]
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
    def __init__(self, transitions : list[float] | np.ndarray):
        """
        Initialize with a list of 6 free transition probabilities. Base 
        frequencies are all equal.

        Args:
            transitions (list[float] | np.ndarray): A list of 6 probabilities

        Raises:
            SubstitutionModelError: if transitions is not of length 6.
        """
        base_freqs = np.ones((4, 1)) * .25
               
        super().__init__(base_freqs, transitions)
    
    def set_hyperparams(self, params : dict[str, object]) -> None:
        """
        Change any of the base frequencies/states/transitions parameters, and 
        recompute the Q matrix accordingly.

        Args:
            params (dict[str, object]): A mapping from gtr parameter names to 
                                        their values. For the HKY class,
                                        names must be limited to 
                                        ["transitions"]
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

    def __init__(self, base_freqs : list[float] | np.ndarray,
                 transitions : list[float] | np.ndarray) -> None:
        """
        Initialize with a list of 4 free base frequencies, and 6 transitions 
        that follow the pattern [a, b, a, a, c, a].
        
        Raises:
            SubstitutionModelError: If the transitions or base frequency lists
            are malformed.
        """

        if transitions[0] != transitions[2] \
            or transitions[2] != transitions[3] \
            or transitions[3] != transitions[5]:
            raise SubstitutionModelError("TN93 Transversions not all equal")

        super().__init__(base_freqs, transitions)
        
    def set_hyperparams(self, params : dict[str, object]) -> None:
        """
        Change any of the base frequencies/transitions parameters, and 
        recompute the Q matrix accordingly.

        Args:
            params (dict[str, object]): A mapping from gtr parameter names to 
                                        their values. For the HKY class,
                                        names must be limited to 
                                        ["base frequencies", "transitions"]
        """
        
        param_names = params.keys()
        
        if "transitions" in param_names:
            self.trans = params["transitions"]
        
        self.is_valid(self.trans, self.freqs)
        self.buildQ()

    def is_valid(self, transitions: list[float] | np.ndarray, 
                 freqs : list[float] | np.ndarray, states : int) -> None:
        """
        Ensure frequencies and transitions are well formed.

        Args:
            transitions (list[float] | np.ndarray): Transition list.
            freqs (list[float] | np.ndarray): Base frequency list. 
                                              Must sum to 1.
        """
        
        # Check for malformed inputs
        if len(freqs) != 4 or sum(freqs) != 1:
            raise SubstitutionModelError("Base frequency list either does not \
                                          sum to 1 or is not of correct length")

        proper_len = ((states - 1) * states) / 2
        if len(transitions) != proper_len:
            raise SubstitutionModelError(f"Incorrect number of transition \
                                          rates. Got {len(transitions)}. 
                                          Expected {proper_len}!")
        
        if transitions[0] != transitions[2] \
            or transitions[2] != transitions[3] \
            or transitions[3] != transitions[5]:
            raise SubstitutionModelError("TN93 Transversions not all equal")

