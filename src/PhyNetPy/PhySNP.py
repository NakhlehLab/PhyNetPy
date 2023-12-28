""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0
Approved to Release Date : N/A
"""

from math import sqrt, comb, pow
import numpy as np
import scipy
from scipy.linalg import expm
from MSA import MSA
import clr

#Import all C# classes
from System.Collections.Generic import *
from System.Collections.Generic import List
from System import String

from BirthDeath import CBDP
from NetworkParser import NetworkParser
from Alphabet import Alphabet
from Matrix import Matrix
from ModelGraph import Model
from ModelFactory import *
from Graph import DAG
global cs

cs = False
#cs = True

### SETUP ###

#TODO : fix absolute path
clr.AddReference('/Users/mak17/Documents/mark-phynetpy/phynetpy_dev/src/DLLS/PhyNetPy_DLLS.dll')

from PhyNetPy_DLLS import SNPEvals

########################
### HELPER FUNCTIONS ###
########################

def partials_index(n:int) -> int:
    """
    Computes the starting index in computing a linear index for an (n,r) pair.
    Returns the index, if r is 0.
    
    i.e n=1 returns 0, since (1,0) is index 0
    i.e n=3 returns 5 since (3,0) is preceded by (1,0), (1,1), (2,0), (2,1), and (2,2)

    Args:
        n (int): an n value (number of lineages) from an (n,r) pair

    Returns:
        int: starting index for that block of n values
    """
    return int(.5 * (n - 1) * (n + 2))

def undo_index(num: int)->list:
    """
    Takes an index from the linear vector and turns it into an (n,r) pair
    
    i.e 7 -> [3,2]

    Args:
        num (int): the index

    Returns:
        list: a 2-tuple (n,r)
    """
    a = 1
    b = 1
    c = -2 - 2 * num
    d = (b ** 2) - (4 * a * c)
    sol = (-b + sqrt(d)) / (2 * a)
    n = int(sol)
    r = num - partials_index(n)

    return [n, r]

def map_nr_to_index(n:int, r:int) -> int:
    """
    Takes an (n,r) pair and maps it to a 1d vector index

    (1,0) -> 0
    (1,1) -> 1
    (2,0) -> 2
    ...
    """
    
    starts = int(.5 * (n - 1) * (n + 2))
    return starts + r
 
def to_array(Fb_map : dict, vector_len : int, site_count : int) -> np.ndarray:
    """
    Takes a vpi/partial likelihood mapping, and translates it into a matrix
    such that the columns denote the site, and the row indeces correspond to (n,r) pairs.
    
    This function serves the purpose of formatting the likelihoods at the root for easy computation

    Args:
        Fb_map (dict): vpi/partial likelihood mapping of a single dimension 
                        (ie, may not be downstream of a reticulation node, without it having been resolved)
        vector_len (int): rows of resulting matrix
        site_count (int): columns of resulting matrix

    Returns:
        np.ndarray: Matrix of dimension vector_len by site_count, containing floats
    """
    
    F_b = np.zeros((vector_len, site_count))  
    
    if not cs:
        for site in range(site_count):
            for nr_pair, prob in Fb_map[site].items():
                #nr_pair should be of the form ((n),(r))
                F_b[int(map_nr_to_index(nr_pair[0][0], nr_pair[1][0]))][site] = prob
    else:
        for site in range(site_count):
            for nr_key in Fb_map[site].Keys:
                n = nr_key.NX[0]
                r = nr_key.RX[0]
                prob = Fb_map[site][nr_key]
                F_b[map_nr_to_index(n, r)][site] = prob
    
    return F_b

def rn_to_rn_minus_dim(set_of_rns : dict, dim : int):
    """
    This is a function defined as
    
    f: Rn;Rn -> Rn-dim;Rn-dim.
    
    set_of_rns should be a mapping in the form of {(nx , rx) -> probability in R} where nx and rx are both vectors in Rn
    
    This function takes set_of_rns and turns it into a mapping {(nx[:-dim] , rx[:-dim]) : set((nx[-dim:] , rx[-dim:], probability))} where the keys
    are vectors in Rn-dim, and their popped last elements and the probability is stored as values.

    Args:
        set_of_rns (dict): a mapping in the form of {(nx , rx) -> probability in R} where nx and rx are both vectors in Rn
        dim (int): the number of dimensions to reduce from Rn.
    """
    
    rn_minus_dim = {}
    
    for vectors, prob in set_of_rns.items():
        nx = vectors[0]
        rx = vectors[1]
        
        #keep track of the remaining elements and the probability
        new_value = (nx[-dim:], rx[-dim:], prob)
        
        #Reduce vectors dimension by grabbing the first n-dim elements
        new_key = (nx[:-dim], rx[:-dim])
        
        #Take care of duplicates
        if new_key in rn_minus_dim.keys():
            rn_minus_dim[new_key].add(new_value)
        else:
            init_value = set()
            init_value.add(new_value)
            rn_minus_dim[new_key] = init_value

    return rn_minus_dim

def qt_2_cs(Qt:np.ndarray):
    """
    Convert a numpy matrix to a C# List<List<float>>

    Args:
        Qt (np.ndarray): SNP transition matrix

    Returns:
        C# List<List<float>>: The QT matrix in C# form
    """
    matrix = List[List[float]]()
    
    for i in range(Qt.shape[0]):
        matrix.Add(List[float]())
        for j in range(Qt.shape[1]):
            matrix[i].Add(float(Qt[i][j]))
    
    return matrix


#########################
### Transition Matrix ###
#########################


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

        rows = int(.5 * self.n * (self.n + 3))
        self.Q : np.ndarray = np.zeros((rows, rows))
        for n_prime in range(1, self.n + 1):  # n ranges from 1 to individuals sampled (both inclusive)
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

########################
### RULE EVALUATIONS ###
########################

def eval_Rule1(F_b : dict, nx : list, n_xtop : int, rx: list, r_xtop: int, Qt: np.ndarray, mx : int) -> dict:
    """
    Given all the information on the left side of the Rule 1 equation, compute the right side probability

    Args:
        F_b (dict): F x, x_bottom vpi map
        nx (list): a vector containing a 1-1 correspondence of n values to population interfaces
        n_xtop (int): number of lineages at x_top
        rx (list): a vector containing a 1-1 correspondence of r values to population interfaces
        r_xtop (int): number of lineages at x_top that are "red"
        Qt (np.ndarray): EXP(Q*t) where Q is the transition matrix, and t is the branch length
        mx (int): number of possible lineages at the node

    Returns:
        dict: A 1 element mapping of left side vectors to their right side probability
    """
    evaluation = 0
    
    for n_b in range(n_xtop, mx + 1):  # n_b always at least 1
        for r_b in range(0, n_b + 1):
            
            index = map_nr_to_index(n_b, r_b)
            exp_val = Qt[index][map_nr_to_index(n_xtop, r_xtop)]  # Q(n,r);(n_t, r_t)
            n_vec = tuple(np.append(nx, n_b))
            r_vec = tuple(np.append(rx, r_b))
            
            try:
                evaluation += F_b[(n_vec, r_vec)] * exp_val
            except KeyError:
                evaluation += 0
    
    return [(tuple(np.append(nx, n_xtop)), tuple(np.append(rx, r_xtop))), evaluation]
    
def eval_Rule2(F_t_x : dict, F_t_y : dict, nx : list, ny : list, n_zbot : int, rx: list, ry : list, r_zbot: int) -> dict:
    """
    Given the left side information for the Rule 2 equation, calculate the right side probability

    Args:
        F_t_x (dict): F x, x_top (vpi map)
        F_t_y (dict): F y, y_top (vpi map)
        nx (list): a vector containing a 1-1 correspondence of n values to population interfaces in the x vector
        ny (list): a vector containing a 1-1 correspondence of n values to population interfaces in the y vector
        n_zbot (int): number of lineages at branch z's bottom
        rx (list): a vector containing a 1-1 correspondence of r values to population interfaces in the x vector
        ry (list): a vector containing a 1-1 correspondence of r values to population interfaces in the y vector
        r_zbot (int): number of lineages from n_zbot that are "red"

    Returns:
        dict: 1 element mapping from the left side vectors to the right side probability
    """
    evaluation = 0
    
    #iterate through valid range of n_xtop and r_xtop values
    for n_xtop in range(0, n_zbot + 1):
        for r_xtop in range(0, r_zbot + 1):
            if r_xtop <= n_xtop and r_zbot - r_xtop <= n_zbot - n_xtop:
    
                #RULE 2 EQUATION
                const = comb(n_xtop, r_xtop) * comb(n_zbot - n_xtop, r_zbot - r_xtop) / comb(n_zbot, r_zbot)
                try:
                    term1 = F_t_x[(tuple(np.append(nx, n_xtop)), tuple(np.append(rx, r_xtop)))]

                    term2 = F_t_y[(tuple(np.append(ny, n_zbot - n_xtop)), tuple(np.append(ry, r_zbot - r_xtop)))]

                    evaluation += term1 * term2 * const
                except KeyError: 
                    evaluation += 0

    return [(tuple(np.append(np.append(nx, ny), n_zbot)), tuple(np.append(np.append(rx, ry), r_zbot))), evaluation]

def eval_Rule3(F_t: dict, nx:list, rx: list, n_ybot:int, n_zbot:int, r_ybot:int, r_zbot:int, gamma_y:float, gamma_z:float) -> dict:
    """
    Given left side information for the Rule 3 equation, calculate the right side probability

    Args:
        F_t (dict): F x, x_top
        nx (list): a vector containing a 1-1 correspondence of n values to population interfaces
        rx (list): a vector containing a 1-1 correspondence of r values to population interfaces
        n_ybot (int): number of lineages that are inherited from the y branch
        n_zbot (int): number of lineages that are inherited from the z branch
        r_ybot (int): number of the y lineages that are "red"
        r_zbot (int): number of the z lineages that are "red"
        gamma_y (float): inheritance probability for branch y
        gamma_z (float): inheritance probability for branch z

    Returns:
        dict: 1 element map from the left side vectors to the right side probability
    """
    #Rule 3 Equation
    try:
        evaluation = F_t[(tuple(np.append(nx, n_ybot + n_zbot)), tuple(np.append(rx, r_ybot + r_zbot)))] * comb(n_ybot + n_zbot, n_ybot) * pow(gamma_y, n_ybot) * pow(gamma_z, n_zbot)
    except KeyError:
        evaluation = 0
        
    return [(tuple(np.append(np.append(nx, n_ybot), n_zbot)), tuple(np.append(np.append(rx, r_ybot), r_zbot))), evaluation]
    
def eval_Rule4(F_t: dict, nz: list, rz:list, n_zbot:int, r_zbot: int)-> dict:
    """
    Given all the information on the left side of the Rule 4 equation, calculate the right side probability.

    Args:
        F_t (dict): F_z_xtop,ytop, a vpi mapping.
        nz (list): a vector containing a 1-1 correspondence of n values to population interfaces
        rz (list): a vector containing a 1-1 correspondence of r values to population interfaces
        n_zbot (int): number of lineages at z
        r_zbot (int): number of the n_zbot lineages that are "red"
    Returns:
        dict: A new entry into the Fz_bot vpi map
    """
    
    evaluation = 0
    
    #Iterate through all possible values of n_xtop and r_xtop
    for n_xtop in range(1, n_zbot + 1):
        for r_xtop in range(0, r_zbot + 1):
            if r_xtop <= n_xtop and r_zbot - r_xtop <= n_zbot - n_xtop:  # Ensure the combinatorics is well defined
                
                #RULE 4 EQUATION
                const = comb(n_xtop, r_xtop) * comb(n_zbot - n_xtop, r_zbot - r_xtop) / comb(n_zbot, r_zbot)
    
                try:
                    term1= F_t[(tuple(np.append(np.append(nz, n_xtop), n_zbot - n_xtop)), tuple(np.append(np.append(rz, r_xtop), r_zbot - r_xtop)))]
                    evaluation += term1 * const
                except KeyError:
                    evaluation += 0
                   
    
    return [(tuple(np.append(nz, n_zbot)), tuple(np.append(rz, r_zbot))), evaluation] 


###########################
### PARTIAL LIKELIHOODS ###
###########################

class PartialLikelihoods:
    
    def __init__(self) -> None:
        
        # A map from a vector of population interfaces (vpi)-- represented as a tuple of strings-- to probability maps
        # defined by rules 0-4.
        self.vpis : dict = {}
        self.ploidy : int = None
        self.evaluator = SNPEvals()
        
    def set_ploidy(self, ploidyness : int) -> None:
        self.ploidy = ploidyness
        
    def Rule0(self, reds: np.ndarray, samples: int, site_count : int, vector_len : int, branch_index : int) -> tuple:
        """
        Given leaf data, compute the initial partial likelihood values for the interface F x_bot

        Args:
            reds (np.ndarray): _description_
            samples (int): _description_
            site_count (int): _description_
            vector_len (int): _description_
            branch_index (int): _description_

        Returns:
            tuple: The vector of population interfaces, for this rule it is simply a 1 element tuple containing x_bot.
        """
        #print("RULE 0 (1)")
        
        if cs:
            red_ct = List[int]()
            for item in reds:
                red_ct.Add(int(item))
            
            F_map = self.evaluator.Rule0(red_ct, site_count, vector_len, samples) 
        else:
            F_map = {}

            for site in range(site_count):
                F_map[site] = {}
                for index in range(vector_len):
                    actual_index = undo_index(index)
                    n = actual_index[0]
                    r = actual_index[1]

                    # Rule 0 formula
                    if reds[site] == r and n == samples:
                        F_map[site][(tuple([n]),tuple([r]))] = 1
                    else:
                        F_map[site][(tuple([n]),tuple([r]))] = 0
                
                
        # #print(red_ct.ToString())    
        vpi_key = tuple(["branch_" + str(branch_index) + ": bottom"])      
        
        
        #print(vpi_key)
        
        self.vpis[vpi_key] = F_map
        #print("RULE 0 (2)")
        return vpi_key

    def Rule1(self, vpi_key:tuple, site_count : int, vector_len : int, m_x : int, Qt : np.ndarray, branch_index : int) -> tuple:
        """
        Given a branch x, and partial likelihoods for the population interface that includes x_bottom,
        we'd like to compute the partial likelihoods for the population interface that includes x_top.
        
        This uses Rule 1 from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932, Rabier et. al.

        Args:
            vpi_key (tuple): the key to the vpi map, the value of which is a mapping containing mappings from 
                             vectors (nx, n_xbot; rx, r_xbot) to probability values for each site
            site_count (int): number of total sites in the multiple sequence alignment
            vector_len (int): number of possible lineages at the root
            m_x (int): number of possible lineages at the branch x
            Qt (np.ndarray): the transition rate matrix exponential

        Returns:
            dict: a mapping in the same format as the parameter F_b, that represents the partial likelihoods at the population interface 
                that now includes the top of this branch, x_top.
        """
        #print("RULE 1 (1)")
        if "branch_" + str(branch_index) + ": bottom" != vpi_key[-1]:
            vpi_key_temp = self.reorder_vpi(vpi_key, site_count, branch_index, False)
            del self.vpis[vpi_key]
            vpi_key = vpi_key_temp
            
            
        F_b = self.vpis[vpi_key]
        
        if not cs:
            F_t = {}
            
            for site in range(site_count):
                #Gather all combinations of nx, rx values 
                nx_rx_map = rn_to_rn_minus_dim(F_b[site], 1)
                
                #initialize the site mapping
                F_t[site] = {}
                
                for vectors in nx_rx_map.keys():
                    nx = list(vectors[0])
                    rx = list(vectors[1])
                    #Iterate over the possible values for n_xtop and r_xtop
                    for ft_index in range(partials_index(m_x + 1)):
                        actual_index = undo_index(ft_index)
                        n_top = actual_index[0]
                        r_top = actual_index[1]
                        
                        #Evaluate the function using Rule1, and insert that value into F_t
                        entry = eval_Rule1(F_b[site], nx, n_top, rx, r_top, Qt, m_x)
                        F_t[site][entry[0]] = entry[1]

                    
                #Handle the (0,0) case. merely copy the probabilities
                for vecs in F_b[site].keys():
                    nx = list(vecs[0])
                    rx = list(vecs[1])
                    if nx[-1] == 0 and rx[-1]==0:
                        F_t[site][vecs] = F_b[site][vecs]
        else:
        
            F_t = self.evaluator.Rule1(F_b, site_count, m_x, qt_2_cs(Qt))
        
        #Replace the instance of x_bot with x_top
        new_vpi_key = list(vpi_key)
        new_vpi_key[vpi_key.index("branch_" + str(branch_index) + ": bottom")] = "branch_" + str(branch_index) + ": top"
        new_vpi_key = tuple(new_vpi_key)
        #print(new_vpi_key)
        
        self.vpis[new_vpi_key] = F_t
        #print(F_t)
        del self.vpis[vpi_key]
        #print("RULE 1 (2)")
        return new_vpi_key
                
                
    def Rule2(self, vpi_key_x : tuple, vpi_key_y :tuple, site_count : int, vector_len : int, branch_index_x:int, branch_index_y:int, branch_index_z:int) -> tuple:
        """
        Given branches x and y that have no leaf descendents in common and a parent branch z, and partial likelihood mappings for the population 
        interfaces that include x_top and y_top, we would like to calculate the partial likelihood mapping for the population interface
        that includes z_bottom.
        
        This uses Rule 2 from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932, Rabier et. al.

        Args:
            vpi_key_x (tuple): The vpi that contains x_top
            vpi_key_y (tuple): The vpi that contains y_top
            site_count (int): number of total sites in the multiple sequence alignment
            vector_len (int): number of possible lineages at the root
            branch_index_x (int): the index of branch x
            branch_index_y (int): the index of branch y
            branch_index_z (int): the index of branch z
        

        Returns:
            tuple: the vpi that is the result of applying rule 2 to vpi_x and vpi_y. Should include z_bot
        """
        
        #F_b = {}
        #print("RULE 2 (1)")
        #Reorder the vpis if necessary
        if "branch_" + str(branch_index_x) + ": top" != vpi_key_x[-1]:
            
            vpi_key_xtemp = self.reorder_vpi(vpi_key_x, site_count, branch_index_x, True)
            del self.vpis[vpi_key_x]
            vpi_key_x = vpi_key_xtemp
        
        if "branch_" + str(branch_index_y) + ": top" != vpi_key_y[-1]:
            
            vpi_key_ytemp = self.reorder_vpi(vpi_key_y, site_count, branch_index_y, True)
            del self.vpis[vpi_key_y]
            vpi_key_y = vpi_key_ytemp
            
        F_t_x = self.vpis[vpi_key_x]
        F_t_y = self.vpis[vpi_key_y]
        
        if not cs:
            F_b = {}
            #Compute F x,y,z_bot
            for site in range(site_count):
                nx_rx_map_y = rn_to_rn_minus_dim(F_t_y[site], 1)
                nx_rx_map_x = rn_to_rn_minus_dim(F_t_x[site], 1)
                F_b[site] = {}
                
                #Compute all combinations of (nx;rx) and (ny;ry)
                for vectors_x in nx_rx_map_x.keys():
                    for vectors_y in nx_rx_map_y.keys():
                        nx = list(vectors_x[0])
                        rx = list(vectors_x[1])
                        ny = list(vectors_y[0])
                        ry = list(vectors_y[1])
                        
                        #Iterate over all possible values of n_zbot, r_zbot
                        for index in range(vector_len):
                            actual_index = undo_index(index)
                            n_bot = actual_index[0]
                            r_bot = actual_index[1]
                            #Evaluate the formula given in rule 2, and insert as an entry in F_b
                            entry = eval_Rule2(F_t_x[site], F_t_y[site], nx, ny, n_bot, rx, ry, r_bot)
                            F_b[site][entry[0]] = entry[1]
        else:
            F_b = self.evaluator.Rule2(F_t_x, F_t_y, site_count, vector_len)
        
        #Combine the vpis
        new_vpi_key_x= list(vpi_key_x)
        new_vpi_key_x.remove("branch_" + str(branch_index_x) + ": top")
        
        new_vpi_key_y= list(vpi_key_y)
        new_vpi_key_y.remove("branch_" + str(branch_index_y) + ": top")
        
        new_vpi_key = tuple(np.append(new_vpi_key_x, np.append(new_vpi_key_y, "branch_" + str(branch_index_z) + ": bottom")))
        
        
        #print(new_vpi_key) 
        
        self.vpis[new_vpi_key] = F_b
        
        #print(F_b)
        del self.vpis[vpi_key_x]
        del self.vpis[vpi_key_y]
        #print("RULE 2 (2)")                  
        return new_vpi_key

    def Rule3(self, vpi_key : tuple, vector_len : int, g_this : float, g_that : float, site_count : int, m: int, branch_index_x:int, branch_index_y:int, branch_index_z:int)->tuple:
        """
        Given a branch x, its partial likelihood mapping at x_top, and parent branches y and z, we would like to compute
        the partial likelihood mapping for the population interface x, y_bottom, z_bottom.

        This uses Rule 3 from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932, Rabier et. al.
        
        Args:
            vpi_key (tuple): the vpi containing x_top
            vector_len (int): the number of (n,r) pairs to iterate through
            g_this (float): gamma inheritance probability for branch y
            g_that (float): gamma inheritance probability for branch z
            site_count (int): number of sites
            m (int): number of possible lineages at x.
            branch_index_x (int): the index of branch x
            branch_index_y (int): the index of branch y
            branch_index_z (int): the index of branch z

        Returns:
            tuple: the vpi that now corresponds to F x, y_bot, z_bot
        """
        
        #print("RULE 3 (1)")
        F_t_x = self.vpis[vpi_key]
        
        if not cs:
            F_b = {}
            for site in range(site_count):
                nx_rx_map = rn_to_rn_minus_dim(F_t_x[site], 1)
                F_b[site] = {}
                #Iterate over the possible (nx;rx) values
                for vector in nx_rx_map.keys():
                    nx = list(vector[0])
                    rx = list(vector[1])
                    #Iterate over the possible values for n_y, n_z, r_y, and r_z
                    for n_y in range(m + 1):
                        for n_z in range(m - n_y + 1):
                            if n_y + n_z >= 1:
                                for r_y in range(n_y + 1):
                                    for r_z in range(n_z + 1):
                                        #Evaluate the formula in rule 3 and add the result to F_b
                                        entry = eval_Rule3(F_t_x[site], nx, rx, n_y, n_z, r_y, r_z, g_this, g_that)
                                        F_b[site][entry[0]] = entry[1]
        else:
            F_b = self.evaluator.Rule3(F_t_x, site_count, vector_len, m, g_this, g_that)
        
        #Create new vpi                       
        new_vpi_key= list(vpi_key)
        new_vpi_key.remove("branch_" + str(branch_index_x) + ": top")
        new_vpi_key.append("branch_" + str(branch_index_y) + ": bottom")
        new_vpi_key.append("branch_" + str(branch_index_z) + ": bottom")
        
        new_vpi_key = tuple(new_vpi_key)
        
        self.vpis[new_vpi_key] = F_b
        #print(F_b)
        del self.vpis[vpi_key]
        #print("RULE 3 (2)")
        return new_vpi_key               
            
    def Rule4(self, vpi_key : tuple, site_count : int, vector_len : int, branch_index_x : int, branch_index_y : int, branch_index_z : int)->tuple:
        """
        Given a branches x and y that share common leaf descendants and that have parent branch z, compute F z, z_bot via 
        Rule 4 described by https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932, Rabier et. al.

        Args:
            vpi_key (tuple): vpi containing x_top and y_top, F z, x_top, y_top.
            site_count (int): number of sites
            vector_len (int): maximum amount of (n,r) pairs
            branch_index_x (int): the index of branch x
            branch_index_y (int): the index of branch y
            branch_index_z (int): the index of branch z

        Returns:
            tuple: vpi for F z,z_bot
        """
        #print("RULE 4 (1)")
        
        
        if "branch_" + str(branch_index_y) + ": top" != vpi_key[-1]:
            vpi_key_temp = self.reorder_vpi(vpi_key, site_count, branch_index_y, True)
            del self.vpis[vpi_key]
            vpi_key = vpi_key_temp
        
        F_t = self.vpis[vpi_key]
        
        if not cs:
            F_b = {}
            for site in range(site_count):
                nx_rx_map = rn_to_rn_minus_dim(F_t[site], 2)
                
                F_b[site] = {}
                
                #Compute all combinations of (nx;rx) and (ny;ry)
                for vectors_x in nx_rx_map.keys():
                
                    nx = list(vectors_x[0])
                    rx = list(vectors_x[1])
                    
                    #Iterate over all possible values of n_zbot, r_zbot
                    for index in range(vector_len):
                        actual_index = undo_index(index)
                        n_bot = actual_index[0]
                        r_bot = actual_index[1]
                        #Evaluate the formula given in rule 2, and insert as an entry in F_b
                        entry = eval_Rule4(F_t[site], nx, rx, n_bot, r_bot)
                        F_b[site][entry[0]] = entry[1]
        else:      
            F_b = self.evaluator.Rule4(F_t, site_count, vector_len)
        
        #Create new vpi
        new_vpi_key = list(vpi_key)
        
        new_vpi_key.remove("branch_" + str(branch_index_x) + ": top")
        new_vpi_key.remove("branch_" + str(branch_index_y) + ": top")
        new_vpi_key.append("branch_" + str(branch_index_z) + ": bottom")
        
        new_vpi_key = tuple(new_vpi_key)
        
        self.vpis[new_vpi_key] = F_b
        #print(F_b)
        del self.vpis[vpi_key]
        #print("RULE 4 (2)")
        return new_vpi_key
    
    def reorder_vpi(self, vpi_key: tuple, site_count:int, branch_index:int, for_top : bool) -> tuple:
        """
        For use when a rule requires a certain ordering of a vpi, and the current vpi does not satisfy it.
        
        I.E, For Rule1, have vpi (branch_1_bottom, branch_2_bottom) but need to calculate for branch 1 top. 
        
        The vpi needs to be reordered to (branch_2_bottom, branch_1_bottom), and the vectors in the partial likelihood 
        mappings need to be reordered to match.

        Args:
            vpi_key (tuple): a vpi tuple
            site_count (int): number of sites
            branch_index (int): branch index of the branch that needs to be in the front
            for_top (bool): bool indicating whether we are looking for branch_index_top or branch_index_bottom in the vpi key

        Returns:
            tuple: the new, reordered vpi.
        """
        #print("REORDER 1")
        if for_top:
            former_index = list(vpi_key).index("branch_" + str(branch_index) + ": top")
        else:
            former_index = list(vpi_key).index("branch_" + str(branch_index) + ": bottom")
            
        new_vpi_key = list(vpi_key)
        new_vpi_key.append(new_vpi_key.pop(former_index))
        
        F_map = self.vpis[vpi_key]
        
        if not cs:
            new_F = {}
            
            #Reorder all the vectors based on the location of the interface within the vpi
            for site in range(site_count):
                new_F[site] = {}
            
                for vectors, prob in F_map[site].items():
                    nx = list(vectors[0])
                    rx = list(vectors[1])
                    new_nx = list(nx)
                    new_rx = list(rx)
                    
                    #pop the element from the list and move to the front
                    new_nx.append(new_nx.pop(former_index))
                    new_rx.append(new_rx.pop(former_index))
            
                    new_F[site][(tuple(new_nx), tuple(new_rx))] = prob
            
            F_map = new_F
            
        else:
            for site in range(site_count):
                for tup in F_map[site].Keys:
                    tup.MoveToEnd(former_index)
        
        self.vpis[tuple(new_vpi_key)] = F_map
        
        ##print("REORDER 2")
        return tuple(new_vpi_key)
    
    def get_key_with(self, branch_index:int)->tuple:
        """
        From the set of vpis, grab the one (should only be one) that contains the branch 
        identified by branch_index

        Args:
            branch_index (int): unique branch identifier

        Returns:
            tuple: the vpi corresponding to branch_index, or None if no such vpi currently exists
        """
    
        for vpi_key in self.vpis:
            top = "branch_" + str(branch_index) + ": top"
            bottom = "branch_" + str(branch_index) + ": bottom"
            
            #Return vpi if there's a match
            if top in vpi_key or bottom in vpi_key:
                return vpi_key
        
        #No vpis were found containing branch_index
        return None
            

######################
### Model Building ###
######################
            
# def SNAPP_Likelihood(filename: str, u :float , v:float, coal:float, grouping:dict=None, auto_detect:bool = False, summary_path:str = None, network_path:str = None) -> float:
#     """
#     Given a filename that represents a path to a nexus file that defines and data, compute the maximum likelihood 
#     """

#     aln = MSA(filename, grouping=grouping, grouping_auto_detect=auto_detect)
#     #Only generates tree starting conditions
#     network = CBDP(1, .5, aln.num_groups()).generateTree()
    

#     snp_params={"samples": len(aln.get_records()), "u": u, "v": v, "coal" : coal, "grouping":True}
#     m = Matrix(aln, Alphabet("SNP"))
#     snp_model = Model(network, m, snp_params=snp_params)
#     m = m.Matrix(aln, a.Alphabet("SNP"))
#     snp_model = mg.Model(network, m, snp_params=snp_params)

#     mh = MetropolisHastings(ProposalKernel(), JC(), m, 800, snp_model) #TODO: Submodel unnecessary for snp. make optional?
#     mh = mh.MetropolisHastings(mh.ProposalKernel(), GTR.JC(), m, 800, snp_model) #TODO: Submodel unnecessary for snp. make optional?
#     result_state = mh.run()

#     result_state.current_model.summary(network_path, summary_path)

class SNP_Likelihood:
    
    def __init__(self, network : DAG, data : MSA ,snp_params : dict) -> None:
        network_comp : NetworkComponent(set(), network)
        tip_data_comp : MSAComponent(set(network_comp), data.grouping)
        self.snp_model = ModelFactory(network_comp,)
        
class VPIComponent(ModelComponent):
    
    def __init__(self, dependencies: set[type]) -> None:
        super().__init__(dependencies)
    
    def build(self, model: Model) -> None:
        model
        


def SNP_Root_Func(Q_matrix, F_b_root, sample_ct : int, site_ct : int):
    """
    The root likelihood function for the SNP likelihood model

    Args:
        Q_matrix (_type_): _description_
        F_b_root (_type_): _description_
        sample_ct (int): _description_
        site_ct (int): _description_

    Returns:
        _type_: _description_
    """
    q_null_space = scipy.linalg.null_space(Q_matrix)
    x = q_null_space / (q_null_space[0] + q_null_space[1]) # normalized so the first two values sum to one

    F_b = to_array(F_b_root, partials_index(sample_ct + 1), site_ct) 
    
    L = np.zeros(site_ct) #self.data.siteCount()
    
    # EQ 20, Root probabilities
    for site in range(site_ct):
        L[site] = np.dot(F_b[:, site], x)
    
    #for non log probabilities, simply print np.sum(L)
    return np.sum(np.log(L))


def SNP_Internal_Func(Q_matrix, F_b_root, sample_ct : int, site_ct : int):
    pass
    
def SNAPP_Likelihood(filename: str, u :float , v:float, coal:float, grouping:dict=None, auto_detect:bool = False, summary_path:str = None, network_path:str = None) -> list[float]:
    aln = MSA(filename, grouping=grouping, grouping_auto_detect = auto_detect)

    #Read and parse the network described 
    networks = NetworkParser(filename).get_all_networks()
 
    likelihoods = []
    for network in networks:
        snp_params={"samples": len(aln.get_records()), "u": u, "v": v, "coal" : coal, "grouping":False}
        #Create model
        snp_model = Model(network, Matrix(aln, Alphabet("SNP")), None, snp_params=snp_params, verbose = True)
        #Compute the likelihood
        likelihoods.append(snp_model.likelihood())

    return likelihoods
              
                
                
            
            
        
        
