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
Last Edit : 3/11/25
First Included in Version : 1.0.0

Docs   - [ ]
Tests  - [ ]
Design - [ ]
"""

from math import sqrt, comb, pow
from typing import Callable
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
from Network import Network, Edge, Node
from MetropolisHastings import *


"""
SOURCES:

(1): 
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932

"""

global cs

cs = False
#cs = True

### SETUP ###
clr.AddReference('PhyNetPy/src/DLLS/PhyNetPy_DLLS.dll')

from PhyNetPy_DLLS import SNPEvals

########################
### HELPER FUNCTIONS ###
########################

def n_to_index(n : int) -> int:
    """
    Computes the starting index in computing a linear index for an (n,r) pair.
    Returns the index, if r is 0.
    
    i.e n=1 returns 0, since (1,0) is index 0
    i.e n=3 returns 5 since (3,0) is preceded by 
        (1,0), (1,1), (2,0), (2,1), and (2,2)

    Args:
        n (int): an n value (number of lineages) from an (n,r) pair
    Returns:
        int: starting index for that block of n values
    """
    return int(.5 * (n - 1) * (n + 2))

def index_to_nr(index : int) -> list[int]:
    """
    Takes an index from the linear vector and turns it into an (n,r) pair
    
    i.e 7 -> [3,2]

    Args:
        index (int): the index

    Returns:
        list[int]: a 2-tuple (n,r)
    """
    a = 1
    b = 1
    c = -2 - 2 * index
    d = (b ** 2) - (4 * a * c)
    sol = (-b + sqrt(d)) / (2 * a)
    n = int(sol)
    r = index - n_to_index(n)

    return [n, r]

def nr_to_index(n : int, r : int) -> int:
    """
    Takes an (n,r) pair and maps it to a 1d vector index

    (1,0) -> 0
    (1,1) -> 1
    (2,0) -> 2
    ...
    
    Args:
        n (int): the number of lineages
        r (int): the number of red lineages (<= n)

    Returns:
        int: the index into the linear vector, that represents by (n, r)
    """
    
    return n_to_index(n) + r
 
def to_array(Fb_map : dict, 
             vector_len : int, 
             site_count : int) -> np.ndarray:
    """
    Takes a vpi/partial likelihood mapping, and translates it into a matrix
    such that the columns denote the site, and the row indeces correspond to 
    (n,r) pairs.
    
    This function serves the purpose of formatting the likelihoods at the root 
    for easy computation.

    Args:
        Fb_map (dict): vpi/partial likelihood mapping of a single dimension 
                       (ie, may not be downstream of a reticulation node, 
                       without it having been resolved)
        vector_len (int): rows of resulting matrix
        site_count (int): columns of resulting matrix

    Returns:
        np.ndarray: Matrix of dimension vector_len by site_count, 
                    containing floats
    """
    
    F_b = np.zeros((vector_len, site_count))  
    
    if not cs:
        for site in range(site_count):
            for nr_pair, prob in Fb_map[site].items():
                #nr_pair should be of the form ((n),(r))
                F_b[int(nr_to_index(nr_pair[0][0], nr_pair[1][0]))][site] = prob
    else:
        for site in range(site_count):
            for nr_key in Fb_map[site].Keys:
                n = nr_key.NX[0]
                r = nr_key.RX[0]
                prob = Fb_map[site][nr_key]
                F_b[nr_to_index(n, r)][site] = prob
    
    return F_b

def rn_to_rn_minus_dim(set_of_rns : dict[tuple[list[float]], float], 
                       dim : int) -> dict[tuple[list[float]], set[tuple]]:
    """
    This is a function defined as
    
    f: Rn;Rn -> Rn-dim;Rn-dim.
    
    This function takes set_of_rns and turns it into a mapping 
    {(nx[:-dim] , rx[:-dim]) : set((nx[-dim:] , rx[-dim:], probability))} 
    where the keys are vectors in Rn-dim, and their popped last elements and 
    the probability is stored as values.

    Args:
        set_of_rns (dict[tuple[list[float]], float]): a mapping in the form of 
                           {(nx , rx) -> probability in R} 
                           where nx and rx are both vectors in Rn
        dim (int): the number of dimensions to reduce from Rn.

    Returns:
        dict[tuple[list[float]], set[tuple]]: map in the form -- 
        {(nx[:-dim] , rx[:-dim]) : set((nx[-dim:] , rx[-dim:], probability))}
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

def qt_2_cs(Qt : np.ndarray) -> List[List[float]]:
    """
    Convert a numpy matrix to a C# List<List<float>>

    Args:
        Qt (np.ndarray): SNP transition matrix

    Returns:
        List[List[float]]: The QT matrix in C# form
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

class BiMarkersTransition:
    """
    Class that encodes the probabilities of transitioning from one (n,r) pair 
    to another under a Biallelic model.

    Includes method for efficiently computing e^Qt

    Inputs:
    1) n-- the total number of samples in the species tree
    2) u-- the probability of going from the red allele to the green one
    3) v-- the probability of going from the green allele to the red one
    4) coal-- the coalescent rate constant, theta

    Assumption: Matrix indexes start with n=1, r=0, so Q[0][0] is Q(1,0);(1,0)

    Q Matrix is given by Equation 15 from:

    David Bryant, Remco Bouckaert, Joseph Felsenstein, Noah A. Rosenberg, 
    Arindam RoyChoudhury, Inferring Species Trees Directly from Biallelic 
    Genetic Markers: Bypassing Gene Trees in a Full Coalescent Analysis, 
    Molecular Biology and Evolution, Volume 29, Issue 8, August 2012, 
    Pages 1917–1932, https://doi.org/10.1093/molbev/mss086
    """

    def __init__(self, n : int, u : float, v : float, coal : float) -> None:
        """
        Initialize the Q matrix

        Args:
            n (int): sample count
            u (float): probability of a lineage going from red to green
            v (float): probability of a lineage going from green to red
            coal (float): coal rate, theta.
        Returns:
            N/A
        """

        # Build Q matrix
        self.n = n 
        self.u = u
        self.v = v
        self.coal = coal

        rows = int(.5 * self.n * (self.n + 3))
        self.Q : np.ndarray = np.zeros((rows, rows))
        
        # n ranges from 1 to individuals sampled (both inclusive)
        for n_prime in range(1, self.n + 1):  
            # r ranges from 0 to n (both inclusive)
            for r_prime in range(n_prime + 1):  
                
                # get indeces from n,r pair 
                n_r = nr_to_index(n_prime, r_prime)
                nm_rm = nr_to_index(n_prime - 1, r_prime - 1)
                n_rm = nr_to_index(n_prime, r_prime - 1)
                nm_r = nr_to_index(n_prime - 1, r_prime)
                n_rp = nr_to_index(n_prime, r_prime + 1)
                

                #### EQ 15 ####
                
                # THE DIAGONAL. always calculated
                self.Q[n_r][n_r] = - (n_prime * (n_prime - 1) / coal) \
                                       - (v * (n_prime - r_prime)) \
                                       - (r_prime * u)

                # These equations only make sense if r isn't 0 
                # (and the second, if n isn't 1).
                if 0 < r_prime <= n_prime:
                    if n_prime > 1:
                        self.Q[n_r][nm_rm] = (r_prime - 1) * n_prime / coal
                    self.Q[n_r][n_rm] = (n_prime - r_prime + 1) * v

                # These equations only make sense if r is strictly less than n 
                # (and the second, if n is not 1).
                if 0 <= r_prime < n_prime:
                    if n_prime > 1:
                        self.Q[n_r][nm_r] = (n_prime - 1 - r_prime) \
                                            * n_prime / coal
                    self.Q[n_r][n_rp] = (r_prime + 1) * u

    def expt(self, t : float = 1) -> np.ndarray:
        """
        Compute e^(Q*t) efficiently.
        
        Args:
            t (float): time, generally in coalescent units. Optional, defaults 
                       to 1, in which case e^Q is computed.
        
        Returns:
            np.ndarray: e^(Q*t).
        """
        return expm(self.Q * t)

    def cols(self) -> int:
        """
        return the dimension of the Q matrix
        
        Args:
            N/A
        Returns:
            int: the number of columns in the Q matrix
        """
        return self.Q.shape[1]

    def getQ(self) -> np.ndarray:
        """
        Retrieve the Q matrix.

        Args:
            N/A
        Returns:
            np.ndarray: The Q matrix.
        """
        return self.Q

########################
### RULE EVALUATIONS ###
########################

#TODO: Params are inconsistently ordered

def eval_Rule1(F_b : dict,
               nx : list[int], 
               rx: list[int], 
               n_xtop : int, 
               r_xtop: int, 
               Qt: np.ndarray,
               mx : int) -> list:
    """
    Given all the information on the left side of the Rule 1 equation, compute 
    the right side probability

    Args:
        F_b (dict): F(x, x_bottom) vpi map
        nx (list[int]): a vector containing a 1-1 correspondence of n values to 
                   population interfaces
        rx (list[int]): a vector containing a 1-1 correspondence of r values to 
                   population interfaces
        n_xtop (int): number of lineages at x_top
        r_xtop (int): number of lineages at x_top that are "red"
        Qt (np.ndarray): e^(Q*t) where Q is the transition matrix, and t is 
                         the branch length
        mx (int): number of possible lineages at the node

    Returns:
        list: The n and r vectors for the vpi after the top of the branch has
              been included, plus the evaluation of the vpi.
    """
    evaluation = 0
    
    # n_b always at least 1
    for n_b in range(n_xtop, mx + 1):  
        for r_b in range(0, n_b + 1):
            
            index = nr_to_index(n_b, r_b)
            
            # Q(n,r);(n_t, r_t)
            exp_val = Qt[index][nr_to_index(n_xtop, r_xtop)]  
            
            n_vec = tuple(np.append(nx, n_b))
            r_vec = tuple(np.append(rx, r_b))
            
            try:
                evaluation += F_b[(n_vec, r_vec)] * exp_val
            except KeyError:
                evaluation += 0
                
    n_vec_top = tuple(np.append(nx, n_xtop))
    r_vec_top = tuple(np.append(rx, r_xtop))
    
    return [(n_vec_top, r_vec_top), evaluation]
    
def eval_Rule2(F_t_x : dict, 
               F_t_y : dict, 
               nx : list[int],
               ny : list[int], 
               rx: list[int], 
               ry : list[int], 
               n_zbot : int,
               r_zbot: int) -> list:
    """
    Given the left side information for the Rule 2 equation, 
    calculate the right side probability

    Args:
        F_t_x (dict): F(x, x_top) (vpi map)
        F_t_y (dict): F(y, y_top) (vpi map)
        nx (list[int]): a vector containing n values for the branches up to 
                        node x
        ny (list[int]): a vector containing n values for the branches up to 
                        node y
        rx (list[int]): a vector containing r values for the branches up to 
                        node x
        ry (list[int]): a vector containing r values for the branches up to 
                        node y
        n_zbot (int): number of lineages at branch z's bottom
        r_zbot (int): number of lineages from n_zbot that are "red"

    Returns:
        list: the vpi vectors for the bottom of the branch that stems from the 
              combination of nodes x and y
    """
    evaluation = 0
    
    #iterate through valid range of n_xtop and r_xtop values
    for n_xtop in range(0, n_zbot + 1):
        for r_xtop in range(0, r_zbot + 1):
            if r_xtop <= n_xtop and r_zbot - r_xtop <= n_zbot - n_xtop:
    
                #RULE 2 EQUATION
                const = comb(n_xtop, r_xtop) \
                        * comb(n_zbot - n_xtop, r_zbot - r_xtop) \
                        / comb(n_zbot, r_zbot)
                try:
                    term1 = F_t_x[(tuple(np.append(nx, n_xtop)), 
                                   tuple(np.append(rx, r_xtop)))]

                    term2 = F_t_y[(tuple(np.append(ny, n_zbot - n_xtop)), 
                                   tuple(np.append(ry, r_zbot - r_xtop)))]

                    evaluation += term1 * term2 * const
                except KeyError: 
                    evaluation += 0

    return [(tuple(np.append(np.append(nx, ny), n_zbot)), 
             tuple(np.append(np.append(rx, ry), r_zbot))), 
             evaluation]

def eval_Rule3(F_t : dict,
               nx : list[int], 
               rx : list[int], 
               n_ybot : int,
               n_zbot : int, 
               r_ybot : int, 
               r_zbot : int, 
               gamma_y : float, 
               gamma_z : float) -> list:
    """
    Given left side information for the Rule 3 equation, calculate the 
    right side probability.

    Args:
        F_t (dict): F (x, x_top)
        nx (list[int]): a vector containing n values for branches up to node x
        rx (list[int]): a vector containing r values for branches up to node x
        n_ybot (int): number of lineages that are inherited from the y branch
        n_zbot (int): number of lineages that are inherited from the z branch
        r_ybot (int): number of the y lineages that are "red"
        r_zbot (int): number of the z lineages that are "red"
        gamma_y (float): inheritance probability for branch y
        gamma_z (float): inheritance probability for branch z

    Returns:
        list: the vpi vectors for the bottom of the branches that stem from the 
               reticulation node above x, and the evaluation for the given r
               values.
    """
    #Rule 3 Equation
    try:
        top_value = F_t[(tuple(np.append(nx, n_ybot + n_zbot)), 
                         tuple(np.append(rx, r_ybot + r_zbot)))]
        evaluation = top_value \
                     * comb(n_ybot + n_zbot, n_ybot) \
                     * pow(gamma_y, n_ybot) \
                     * pow(gamma_z, n_zbot)
    except KeyError:
        evaluation = 0
        
    return [(tuple(np.append(np.append(nx, n_ybot), n_zbot)), 
             tuple(np.append(np.append(rx, r_ybot), r_zbot))), 
             evaluation]
    
def eval_Rule4(F_t: dict, 
               nz : list[int], 
               rz : list[int], 
               n_zbot : int,
               r_zbot : int) -> list:
    """
    Given all the information on the left side of the Rule 4 equation, 
    calculate the right side probability.

    Args:
        F_t (dict): F(z_xtop,ytop), a vpi mapping.
        nz (list[int]): a vector of n values for the branches up to node z
        rz (list[int]): a vector of r values for the branches up to node z
        n_zbot (int): number of lineages at z
        r_zbot (int): number of the n_zbot lineages that are "red"
    Returns:
        list: vpi vectors for the bottom of the branch that stems from node z, 
              plus the evalutation given the n and r values.
    """
    
    evaluation = 0
    
    #Iterate through all possible values of n_xtop and r_xtop
    for n_xtop in range(1, n_zbot + 1):
        for r_xtop in range(0, r_zbot + 1):
            # Ensure the combinatorics is well defined
            if r_xtop <= n_xtop and r_zbot - r_xtop <= n_zbot - n_xtop:  
                
                #RULE 4 EQUATION
                const = comb(n_xtop, r_xtop) \
                        * comb(n_zbot - n_xtop, r_zbot - r_xtop) \
                        / comb(n_zbot, r_zbot)
    
                try:
                    nz_xtop_ytop = tuple(np.append(np.append(nz, n_xtop), 
                                         n_zbot - n_xtop))
                    rz_xtop_ytop = tuple(np.append(np.append(rz, r_xtop), 
                                         r_zbot - r_xtop))
                    
                    evaluation += F_t[(nz_xtop_ytop, rz_xtop_ytop)] * const
                except KeyError:
                    evaluation += 0
                   
    
    return [(tuple(np.append(nz, n_zbot)), 
             tuple(np.append(rz, r_zbot))),
             evaluation] 


###########################
### PARTIAL LIKELIHOODS ###   
###########################

class PartialLikelihoods:
    """
    Class that bookkeeps the vectors of population interfaces (vpis) and their
    associated likelihood values.
    
    Contains methods for evaluating and storing likelihoods based on the rules
    described by Rabier et al.
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty PartialLikelihood obj.

        Args:
            N/A
        Returns:
            N/A
        """
        # A map from a vector of population interfaces (vpi)
        # -- represented as a tuple of strings-- 
        # to probability maps defined by rules 0-4.
        self.vpis : dict = {}
        self.ploidy : int = None
        
        #C sharp SNPEvals object for efficient calcs
        self.evaluator = SNPEvals()
        
    def set_ploidy(self, ploidy : int) -> None:
        """
        Set the ploidy value for the partial likelihoods object.
        
        Args:
            ploidyness (int): ploidy value.
        Returns:
            N/A
        """
        self.ploidy = ploidy
        
    def Rule0(self, 
              reds : np.ndarray, 
              samples : int,
              site_count : int, 
              vector_len : int,
              branch_id : str) -> tuple:
        """
        Given leaf data, compute the initial partial likelihood values 
        for the interface F(x_bot)

        Args:
            reds (np.ndarray): An array of red counts per site.
            samples (int): Number of total samples.
            site_count (int): Number of total sites.
            vector_len (int): Max vector length, based on ploidy and the number 
                              of possible (n,r) pairs.
            branch_index (int): The unique identifier of the branch we are 
                                operating on.

        Returns:
            tuple: vpi key that maps to the partial likelihoods at the 
                   population interface that now includes the top of this 
                   branch, x_top.
        """
        
        if cs:
            red_ct = List[int]()
            for item in reds:
                red_ct.Add(int(item))
            
            #Call the C# function instead of the python
            F_map = self.evaluator.Rule0(red_ct, 
                                         site_count,
                                         vector_len,
                                         samples) 
        else:
            F_map = {}

            for site in range(site_count):
                F_map[site] = {}
                for index in range(vector_len):
                    actual_index = index_to_nr(index)
                    n = actual_index[0]
                    r = actual_index[1]

                    # Rule 0 formula
                    if reds[site] == r and n == samples:
                        F_map[site][(tuple([n]), tuple([r]))] = 1
                    else:
                        F_map[site][(tuple([n]), tuple([r]))] = 0
                
                
        # Generate the new vpi key  
        vpi_key = tuple(["branch_" + str(branch_id) + ": bottom"])  
    
        # Map the vector values to the vpi key
        self.vpis[vpi_key] = F_map
        
        return vpi_key

    def Rule1(self,
              vpi_key_x : tuple, 
              branch_id_x : int,
              m_x : int, 
              Qt : np.ndarray) -> tuple:
        """
        Given a branch x, and partial likelihoods for the population interface 
        that includes x_bottom, we'd like to compute the partial likelihoods for 
        the population interface that includes x_top.
        
        This uses Rule 1 from (1)
        

        Args:
            vpi_key_x (tuple): the key to the vpi map, the value of which is a 
                             mapping containing mappings from vectors 
                             (nx, n_xbot; rx, r_xbot) to probability values 
                             for each site
            branch_id_x (int): the unique id of branch x
            m_x (int): number of possible lineages at the branch x
            Qt (np.ndarray): the transition rate matrix exponential

        Returns:
            tuple: vpi key that maps to the partial likelihoods at the 
                   population interface that now includes the top of this 
                   branch, x_top.
        """
        
        
        # Check if vectors are properly ordered
        if "branch_" + str(branch_id_x) + ": bottom" != vpi_key_x[-1]:
            vpi_key_temp = self.reorder_vpi(vpi_key_x,
                                            site_count, 
                                            branch_id_x, 
                                            False)
            del self.vpis[vpi_key_x]
            vpi_key_x = vpi_key_temp
            
            
        F_b = self.vpis[vpi_key_x]
        
        if not cs:
            F_t = {}
            
            # Do calculations for each site
            for site in range(site_count):
                #Gather all combinations of nx, rx values 
                nx_rx_map = rn_to_rn_minus_dim(F_b[site], 1)
                
                #initialize the site mapping
                F_t[site] = {}
                
                for vectors in nx_rx_map.keys():
                    nx = list(vectors[0])
                    rx = list(vectors[1])
                    #Iterate over the possible values for n_xtop and r_xtop
                    for ft_index in range(n_to_index(m_x + 1)):
                        actual_index = index_to_nr(ft_index)
                        n_top = actual_index[0]
                        r_top = actual_index[1]
                        
                        # Evaluate the function using Rule1, and insert 
                        # that value into F_t
                        entry = eval_Rule1(F_b[site], 
                                           nx, 
                                           rx, 
                                           n_top, 
                                           r_top, 
                                           Qt,
                                           m_x)
                        F_t[site][entry[0]] = entry[1]

                    
                #Handle the (0,0) case. merely copy the probabilities
                for vecs in F_b[site].keys():
                    nx = list(vecs[0])
                    rx = list(vecs[1])
                    if nx[-1] == 0 and rx[-1] == 0:
                        F_t[site][vecs] = F_b[site][vecs]
        else:
            # Use C sharp evaluator instead
            F_t = self.evaluator.Rule1(F_b, site_count, m_x, qt_2_cs(Qt))
        
        # Replace the instance of x_bot with x_top
        new_vpi_key = list(vpi_key_x)
        edit_index = vpi_key_x.index("branch_" + str(branch_id_x) + ": bottom")
        new_vpi_key[edit_index] = "branch_" + str(branch_id_x) + ": top"
        new_vpi_key = tuple(new_vpi_key)
        
        # Put the map back
        self.vpis[new_vpi_key] = F_t
        del self.vpis[vpi_key_x]
        
        return new_vpi_key
                
    def Rule2(self, 
              vpi_key_x : tuple, 
              vpi_key_y : tuple,  
              branch_id_x : str,
              branch_id_y : str, 
              branch_id_z : str) -> tuple:
        """
        Given branches x and y that have no leaf descendents in common and a 
        parent branch z, and partial likelihood mappings for the population 
        interfaces that include x_top and y_top, we would like to calculate 
        the partial likelihood mapping for the population interface
        that includes z_bottom.
        
        This uses Rule 2 from (1)

        Args:
            vpi_key_x (tuple): The vpi that contains x_top
            vpi_key_y (tuple): The vpi that contains y_top
            branch_id_x (str): the unique id of branch x
            branch_id_y (str): the unique id of branch y
            branch_id_z (str): the unique id of branch z
        

        Returns:
            tuple: the vpi key that is the result of applying rule 2 to 
                   vpi_x and vpi_y. Should include z_bot.
        """
        
        #Reorder the vpis if necessary
        if "branch_" + str(branch_id_x) + ": top" != vpi_key_x[-1]:
            
            vpi_key_xtemp = self.reorder_vpi(vpi_key_x, 
                                             site_count, 
                                             branch_id_x,
                                             True)
            del self.vpis[vpi_key_x]
            vpi_key_x = vpi_key_xtemp
        
        if "branch_" + str(branch_id_y) + ": top" != vpi_key_y[-1]:
            
            vpi_key_ytemp = self.reorder_vpi(vpi_key_y, 
                                             site_count,
                                             branch_id_y, 
                                             True)
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
                            actual_index = index_to_nr(index)
                            n_bot = actual_index[0]
                            r_bot = actual_index[1]
                            # Evaluate the formula given in rule 2, 
                            # and insert as an entry in F_b
                            entry = eval_Rule2(F_t_x[site], F_t_y[site],
                                               nx, ny, n_bot, rx, ry, r_bot)
                            F_b[site][entry[0]] = entry[1]
        else:
            F_b = self.evaluator.Rule2(F_t_x, F_t_y, site_count, vector_len)
        
        #Combine the vpis
        new_vpi_key_x= list(vpi_key_x)
        new_vpi_key_x.remove("branch_" + str(branch_id_x) + ": top")
        
        new_vpi_key_y= list(vpi_key_y)
        new_vpi_key_y.remove("branch_" + str(branch_id_y) + ": top")
        
        #Create new vpi key, (vpi_x, vpi_y, z_branch_bottom)
        z_name = "branch_" + str(branch_id_z) + ": bottom"
        vpi_y = np.append(new_vpi_key_y, z_name)
        new_vpi_key = tuple(np.append(new_vpi_key_x, vpi_y))
        
        
        #Update the vpi tracker
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key_x]
        del self.vpis[vpi_key_y]
                         
        return new_vpi_key

    def Rule3(self, 
              vpi_key_x : tuple, 
              branch_id_x : str,
              branch_id_y : str, 
              branch_id_z : str,
              g_this : float,
              g_that : float,
              mx : int) -> tuple:
        """
        Given a branch x, its partial likelihood mapping at x_top, and parent 
        branches y and z, we would like to compute the partial likelihood 
        mapping for the population interface x, y_bottom, z_bottom.

        This uses Rule 3 from (1)
        
        Args:
            vpi_key_x (tuple): the vpi containing x_top
            branch_id_x (str): the unique id of branch x
            branch_id_y (str): the unique id of branch y
            branch_id_z (str): the unique id of branch z
            g_this (float): gamma inheritance probability for branch y
            g_that (float): gamma inheritance probability for branch z
            mx (int): number of possible lineages at x.

        Returns:
            tuple: the vpi key that now corresponds to F (x, y_bot, z_bot)
        """
        
        F_t_x = self.vpis[vpi_key_x]
        
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
                    for n_y in range(mx + 1):
                        for n_z in range(mx - n_y + 1):
                            if n_y + n_z >= 1:
                                for r_y in range(n_y + 1):
                                    for r_z in range(n_z + 1):
                                        # Evaluate the formula in rule 3 
                                        # and add the result to F_b
                                        entry = eval_Rule3(F_t_x[site],
                                                           nx,
                                                           rx, 
                                                           n_y, 
                                                           n_z, 
                                                           r_y, 
                                                           r_z, 
                                                           g_this, 
                                                           g_that)
                                        F_b[site][entry[0]] = entry[1]
        else:
            F_b = self.evaluator.Rule3(F_t_x, 
                                       site_count, 
                                       vector_len, 
                                       mx, 
                                       g_this, 
                                       g_that)
        
        #Create new vpi key                      
        new_vpi_key = list(vpi_key_x)
        new_vpi_key.remove("branch_" + str(branch_id_x) + ": top")
        new_vpi_key.append("branch_" + str(branch_id_y) + ": bottom")
        new_vpi_key.append("branch_" + str(branch_id_z) + ": bottom")
        new_vpi_key = tuple(new_vpi_key)
        
        # Update vpi tracker
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key_x]
        
        return new_vpi_key               
            
    def Rule4(self, 
              vpi_key_xy : tuple, 
              branch_index_x : int, 
              branch_index_y : int, 
              branch_index_z : int) -> tuple:
        """
        Given a branches x and y that share common leaf descendants and that 
        have parent branch z, compute F z, z_bot via 
        Rule 4 described by (1)

        Args:
            vpi_key_x (tuple): vpi containing x_top and y_top.
            branch_index_x (int): the index of branch x
            branch_index_y (int): the index of branch y
            branch_index_z (int): the index of branch z

        Returns:
            tuple: vpi key for F(z,z_bot)
        """

        if "branch_" + str(branch_index_y) + ": top" != vpi_key_xy[-1]:
            vpi_key_temp = self.reorder_vpi(vpi_key_xy,
                                            site_count, 
                                            branch_index_y, 
                                            True)
            del self.vpis[vpi_key_xy]
            vpi_key_xy = vpi_key_temp
        
        F_t = self.vpis[vpi_key_xy]
        
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
                        actual_index = index_to_nr(index)
                        n_bot = actual_index[0]
                        r_bot = actual_index[1]
                        # Evaluate the formula given in rule 2,
                        # and insert as an entry in F_b
                        entry = eval_Rule4(F_t[site], nx, rx, n_bot, r_bot)
                        F_b[site][entry[0]] = entry[1]
        else:      
            F_b = self.evaluator.Rule4(F_t, site_count, vector_len)
        
        #Create new vpi
        new_vpi_key = list(vpi_key_xy)
        new_vpi_key.remove("branch_" + str(branch_index_x) + ": top")
        new_vpi_key.remove("branch_" + str(branch_index_y) + ": top")
        new_vpi_key.append("branch_" + str(branch_index_z) + ": bottom")
        new_vpi_key = tuple(new_vpi_key)
    
        # Update vpi tracker
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key_xy]
        
        return new_vpi_key
    
    def reorder_vpi(self, 
                    vpi_key: tuple, 
                    branch_index : int, 
                    for_top : bool) -> tuple:
        """
        For use when a rule requires a certain ordering of a vpi, and the
        current vpi does not satisfy it.
        
        I.E, For Rule1, have vpi (branch_1_bottom, branch_2_bottom) but 
        need to calculate for branch 1 top. 
        
        The vpi needs to be reordered to (branch_2_bottom, branch_1_bottom), 
        and the vectors in the partial likelihood mappings need to be reordered 
        to match.

        Args:
            vpi_key (tuple): a vpi tuple
            branch_index (int): branch index of the branch that needs 
                                to be in the front
            for_top (bool): bool indicating whether we are looking for 
                            branch_index_top or branch_index_bottom in the 
                            vpi key.

        Returns:
            tuple: the new, reordered vpi key.
        """
        #print("REORDER 1")
        if for_top:
            name = "branch_" + str(branch_index) + ": top"
            former_index = list(vpi_key).index(name)
        else:
            name = "branch_" + str(branch_index) + ": bottom"
            former_index = list(vpi_key).index(name)
            
        new_vpi_key = list(vpi_key)
        new_vpi_key.append(new_vpi_key.pop(former_index))
        
        F_map = self.vpis[vpi_key]
        
        if not cs:
            new_F = {}
            
            # Reorder all the vectors based on the location of the 
            # interface within the vpi
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
    
        return tuple(new_vpi_key)
    
    def get_key_with(self, branch_id : str) -> tuple:
        """
        From the set of vpis, grab the one (should only be one) that contains 
        the branch identified by branch_index

        Args:
            branch_index (int): unique branch identifier

        Returns:
            tuple: the vpi key corresponding to branch_index, or None if no 
                   such vpi currently exists
        """
    
        for vpi_key in self.vpis:
            top = "branch_" + branch_id + ": top"
            bottom = "branch_" + branch_id + ": bottom"
            
            #Return vpi if there's a match
            if top in vpi_key or bottom in vpi_key:
                return vpi_key
        
        #No vpis were found containing branch_index
        return None
            

######################
### Model Building ###
######################

class U(Parameter):
    """
    Parameter for the red->green lineage transition probability.
    """
    def __init__(self, value : float) -> None:
        """
        Initialize the U parameter with a given value.

        Args:
            value (float): The value of the U parameter.
        Returns:
            N/A
        """
        super().__init__("u", value)

class V(Parameter):
    """
    Parameter for the green->red lineage transition probability.
    """
    def __init__(self, value : float) -> None:
        """
        Initialize the V parameter with a given value.

        Args:
            value (float): The value of the V parameter.
        Returns:
            N/A
        """
        super().__init__("v", value)

class Coal(Parameter):
    """
    Coalescent rate parameter, theta.
    """
    def __init__(self, value : float) -> None:
        """
        Initialize the coalescent rate parameter with a given value.

        Args:
            value (float): The value of the coalescent rate parameter.
        Returns:
            N/A
        """
        super().__init__("coal", value)

class Samples(Parameter):
    """
    Parameter for the number of total samples (sequences) present.
    """
    def __init__(self, value : int) -> None:
        """
        Initialize the samples parameter with a given value.
        
        Args:
            value (int): The number of samples.
        Returns:
            N/A
        """
        super().__init__("samples", value)

class SiteParameter(Parameter):
    """
    Parameter for the number of sites (sequence length).
    """
    def __init__(self, value : int) -> None:
        """
        Initialize the site parameter with a given value.

        Args:
            value (int): The number of sites.
        Returns:
            N/A
        """
        super(Parameter).__init__("sitect", value)
        
class BiMarkersTransitionMatrixNode(CalculationNode):
    """
    Node that encodes the transition matrix, Q.
    """
    def __init__(self) -> None:
        """
        Initialize an empty container for the transition matrix, Q.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
    
    def calc(self) -> BiMarkersTransition:
        """
        Grab the model parameters, then compute and store Q.

        Args:
            N/A
        Returns:
            BiMarkersTransition: The Q matrix that was just computed and cached.
        """
        params = self.get_parameters()
        return self.cache(BiMarkersTransition(params["samples"], 
                                              params["u"], 
                                              params["v"], 
                                              params["coal"]))
            
    def sim(self) -> None:
        """
        N/A
        
        Args:
            N/A
        Returns:
            N/A
        """
        pass
    
    def get(self) -> BiMarkersTransition:
        """
        If no changes to model parameters have been made, returns a cached Q.
        Otherwise, grabs the updated parameters and recomputes Q, caches it,
        and returns it.

        Args:
            N/A
        Returns:
            BiMarkersTransition: Up to date Q matrix.
        """
        if self.dirty:
            return self.calc()
        else:
            return self.cached
            
    def update(self) -> None:
        """
        Tell nodes upstream that depend on Q that their likelihoods need to be
        recalculated.
        
        Args:
            N/A
        Returns:
            N/A
        """
        self.upstream()

class VPIAccumulator(Accumulator):
    """
    Class that holds a reference to the PartialLikelihood object, that tracks
    vpis with their partial likelihoods.
    """
    def __init__(self) -> None:
        """
        Initialize the VPIAccumulator with an empty PartialLikelihood object.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__("VPI", PartialLikelihoods())
    
# class BiMarkersNode(ANetworkNode):
#     def __init__(self, 
#                  in_edges : list[Edge],
#                  out_edges : list[Edge],
#                  name : str = None, 
#                  node_type : str = None) -> None:
       
#         super().__init__(name, node_type)
#         self.in_edges = in_edges
#         self.out_edges = out_edges
        
    
#     def red_count(self) -> np.ndarray:
#         """
#         Only defined for leaf nodes, this method returns the count of red 
#         alleles for each site for the associated species. The dimension of the
#         resulting array will be (sequence count, sequence length)
#         where sequence count is the number of data sequences associated with
#         the species, and the sequence length is simply the length of each 
#         sequence (these will all be the same).

#         Returns:
#             np.ndarray: An array that describes the red allele counts per 
#                         sequence and per site.
#         """
#         if len(self.get_children()) == 0:
#             spec : ExtantSpecies = self.get_model_children(ExtantSpecies)[0]
#             seqs : list[DataSequence] = spec.get_seqs()
            
#             tot = np.zeros(len(seqs[0].get_seq()))
#             for seq_rec in seqs:
#                 tot = np.add(tot, np.array(seq_rec.get_numerical_seq()))
            
#             return tot
#         else:
#             warnings.warn("Red count method called on a network node that has\
#                            network children. Needs to be called on a leaf!")
        
#     def calc(self):
#         """
#         Calculates both the top and bottom partial likelihoods, 
#         based on Eq 14 and 19.

#         Returns a list of length 2, element [0] is the bottom likelihoods, 
#         element [1] is the top likelihoods.
        
#         Calculated using eqs 12,14,16,19 from David Bryant, Remco Bouckaert, 
#         Joseph Felsenstein, Noah A. Rosenberg, Arindam RoyChoudhury, 
#         Inferring Species Trees Directly from Biallelic Genetic Markers: 
#         Bypassing Gene Trees in a Full Coalescent Analysis, Molecular Biology 
#         and Evolution, Volume 29, Issue 8, August 2012, Pages 1917-1932, 
#         https://doi.org/10.1093/molbev/mss086
        
#         Also, Rule 3,4 for networks Rabier CE, Berry V, Stoltz M, Santos JD, 
#         Wang W, et al. (2021) On the inference of complex phylogenetic networks 
#         by Markov Chain Monte-Carlo. PLOS Computational Biology 17(9): e1008380.
#         https://doi.org/10.1371/journal.pcbi.1008380
#         """
#         ###############
#         #### SETUP ####
#         ###############
        
#         # Grab vpi tracker
#         vpi_acc : VPIAccumulator = self.get_model_children(VPIAccumulator)[0]
#         pl : PartialLikelihoods = vpi_acc.data
        
#         # Grab site count
#         sitect_param : SiteParameter = self.get_model_children(SiteParameter)[0]
#         site_count = sitect_param.get_value()
        
#         # Grab vector length at this node
#         vector_len = n_to_index(self.possible_lineages() + 1)  
        
#         # ID each branch
#         par_branches : dict[BiMarkersNode, list[str]] = {}
#         parents : list[BiMarkersNode] = []
#         for par in self.get_model_parents(BiMarkersNode):
#             branch_id = "<" + str(par.__hash__) + ", " \
#                             + str(self.__hash__) + ">"
#             par_branches[par] = branch_id
#             parents.append(par)
        
#         child_branches : dict[BiMarkersNode, list[str]] = {}
#         children : list[BiMarkersNode] = {}
#         for child in self.get_model_children(BiMarkersNode):
#             branch_id = "<" + str(self.__hash__) + ", " \
#                             + str(child.__hash__) + ">"
#             child_branches[child] = branch_id
#             children.append(child) 
        
        

#         ###########################################################
#         #### IF THIS NODE IS A LEAF, THERE IS ONLY ONE BRANCH, ####
#         #### AND ME MUST APPLY RULE 0. ############################
#         ###########################################################
#         if len(self.get_children()) == 0:
#             branch_id = par_branches[list(par_branches.keys())[0]]
#             F_key = pl.Rule0(self.red_count(), 
#                              self.samples(),
#                              site_count, 
#                              vector_len, 
#                              branch_id)  
            
#         #### Case 2, the branch is for an internal node, so bottom 
#         # likelihoods need to be computed based on child tops
#         else:
#             # EQ 19
#             if len(parents) == 2:
#                 #RULE 3
#                 y : BiMarkersNode = parents[0]
#                 z : BiMarkersNode = parents[1]
                
#                 x_id = child_branches[children[0]]
#                 y_id = par_branches[y]
#                 z_id = par_branches[z]
#                 F_t_x_key = pl.get_key_with(x_id)
                
#                 possible_lineages = self.possible_lineages() 
                
#                 this : Edge = [e for e in self.in_edges if e.src == y][0]
#                 that : Edge = [e for e in self.in_edges if e.src == z][0]
                
#                 g_this = this.gamma
#                 g_that = that.gamma
                
#                 if g_this + g_that != 1:
#                     raise ModelError("Set of inheritance probabilities do not \
#                         sum to 1 for node<" + self.name + ">")
                
#                 F_b_key = pl.Rule3(F_t_x_key, 
#                                    x_id,
#                                    y_id,
#                                    z_id,
#                                    g_this,
#                                    g_that,
#                                    possible_lineages)
                                
#                 #Do the rule 1 calculations for the sibling branch
#                 q = self.get_model_children(BiMarkersTransitionMatrixNode)
#                 Q : BiMarkersTransitionMatrixNode = q[0]
#                 QT = Q.get()
#                 z_qt = QT.expt(that.length)
                
#                 F_t_key_sibling = pl.Rule1(F_b_key, 
#                                            z_id, 
#                                            z.possible_lineages(),
#                                            z_qt)
                
#                 F_key = F_t_key_sibling
             
#             elif len(node_par.children) == 2:
                
#                 y_branch : SNPBranchNode = node_par.get_branch_from_child(net_children[0])
#                 F_t_y_key = y_branch.get()
#                 y_branch_index = y_branch.index
                
#                 z_branch : SNPBranchNode = node_par.get_branch_from_child(net_children[1], avoid_index=y_branch_index)
#                 F_t_z_key = z_branch.get()
#                 z_branch_index = z_branch.index
                
#                 #Find out whether lineage y and z have leaves in common 
#                 if not net_children[1].leaf_descendants.isdisjoint(net_children[0].leaf_descendants): #If two sets are not disjoint
#                     print("Y BRANCH INDEX: " + str(y_branch_index))
#                     print("Z BRANCH INDEX: " + str(z_branch_index))
#                     F_b_key = self.vpi_tracker.Rule4(F_t_z_key, site_count, vector_len, y_branch_index, z_branch_index, self.index)
#                 else: # Then use Rule 2
#                     F_b_key = self.vpi_tracker.Rule2(F_t_y_key, F_t_z_key, site_count, vector_len, y_branch_index, z_branch_index, self.index)
#                     #raise ModelError("temp catch")
#                 F_key = F_b_key
#             else:
#                 #A node should only have one child if it is the root node. simply pass along the vpi
#                 F_key = node_par.get_branch_from_child(net_children[0]).get()
                    
#         # TOP: Compute the top likelihoods based on the bottom likelihoods w/ eq 14&16
#         if node_par.parents is not None:
#             F_key = self.vpi_tracker.Rule1(F_key, site_count, vector_len, node_par.possible_lineages(), self.Qt, self.index)
#             self.updated = False
#         else:
#             self.updated = False
    
#         # print("F_T (at site 0)")
#         # print(F_key)
#         # print(self.vpi_tracker.vpis[F_key][0])
        
#         return F_key
    
#     def calc_leaf_descendants(self) -> set[Node]:
#         """
#         Calculate the leaves that are descendants of a lineage/node.
        
#         Returns:
#             leaf_descendants (set) : a set of node descendants
#         """
#         for child in self.get_children():
#             if len(child.get_children()) == 0:
#                 self.leaf_descendants.add(child)
#             else:
#                 #The union of all its children's descendants
#                 child_desc = child.calc_leaf_descendants()
#                 self.leaf_descendants = self.leaf_descendants.union(child_desc)
        
#         return self.leaf_descendants
        
#     def get(self) -> tuple:
#         if self.updated:
#             return self.calc()
#         else:
#             return self.cached

#     def possible_lineages(self) -> int:
#         """
#         Calculate the number of lineages that flow through this node.
#         For non-reticulation nodes, if branch x has children y,z:

#         Returns:
#             int: number of lineages
#         """
#         if len(self.get_children()) == 0:
#             return self.samples()
#         else:
#             return sum([child.samples() for child in self.leaf_descendants])
    
#     def samples(self)->int:
#         if len(self.get_children()) == 0:
#             seqs = self.get_model_children(ExtantSpecies)[0].get_seqs()
#             return sum([rec.ploidy() for rec in seqs]) 
#         else:
#             Warning("Calling samples method on a node that is not a leaf")
            
class BiMarkersLikelihood(CalculationNode):
    """
    Root likelihood algorithm.
    This node is also the root of the model graph.
    """
    
    def __init__(self) -> None:
        """
        Initialize the root likelihood algorithm node.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
    
    def calc(self) -> float:
        """
        Run the root likelihood calculation and return the final MCMC bimarkers
        likelihood value for the network.

        Args:
            N/A
        Returns:
            float: a negative log likelihood, the closer to 0 the more likely
                   that the network is the true network that represents 
                   the data.
        """
        
        #Grab Q matrix
        tn : BiMarkersTransitionMatrixNode 
        tn = self.get_model_children(BiMarkersTransitionMatrixNode)[0]
        q_null_space = scipy.linalg.null_space(tn.get().getQ())
        
        # normalized so the first two values sum to one
        x = q_null_space / (q_null_space[0] + q_null_space[1]) 

        # .get() returns a list of all vpi keys for branches exiting a node.
        # For the root, there will only be one vpi.
        root_node : BiMarkersNode = self.get_model_children(BiMarkersNode)[0]
        root_vpi_key = root_node.get()[0]
        vpi_acc : VPIAccumulator = self.get_model_children(VPIAccumulator)[0]
        F_b_map = vpi_acc.get_data().vpis[root_vpi_key]
        
        #Grab all hyperparams
        params = self.get_parameters()
        
        #Convert mapping to an array of the right size
        F_b = to_array(F_b_map, 
                       n_to_index(params["samples"] + 1),
                       params["sitect"]) 

        L = np.zeros(params["sitect"])
       
        # EQ 20, Root probabilities
        for site in range(params["sitect"]):
            L[site] = np.dot(F_b[:, site], x)
    
        #print("NON-LOG PROBABILITY: " + str(np.sum(L)))
        return self.cache(np.sum(np.log(L)))
    
    def get(self) -> float:
        """
        Gets the final log likelihood of the network.
        Calculates it, or returns the cached value.

        Args:
            N/A
        Returns:
            float: a negative log likelihood, the closer to 0 the more likely
                   that the network is the true network that represents 
                   the data.
        """
        if self.dirty:
            self.calc()
        else:
            return self.cached
    
    def sim(self) -> None:
        """
        N/A
        
        Args:
            N/A
        Returns:
            N/A
        """
        pass
    
    def update(self) -> None:
        """
        N/A
        
        Args:
            N/A
        Returns:
            N/A
        """
        pass

##############################
#### SNP MODEL COMPONENTS ####
##############################

class VPIComponent(ModelComponent):
    """
    Model component that hooks up the VPI tracker to the network and the root
    probability calculator.
    """
    
    def __init__(self, dependencies: set[type]) -> None:
        """
        Initialize the VPI component with a set of dependencies.

        Args:
            dependencies (set[type]): a set of types that this component
                                       depends on.
        Returns:
            N/A
        """
        super().__init__(dependencies)
        
    def build(self, model : Model) -> None:
        """
        Build the VPI tracker, and hook it up to the root probability calculator
        and all network nodes.

        Args:
            model (Model): the model that the VPI tracker will be hooked up to.
        Returns:
            N/A
        """
        vpi_acc = VPIAccumulator()
        
        # Hook to root probability
        root_prob = model.all_nodes[BiMarkersLikelihood][0]
        vpi_acc.join(root_prob)
        
        # Hook to all network nodes
        join_network(vpi_acc, model)
        
        # Bookkeep
        model.all_nodes[VPIAccumulator].append(vpi_acc)

class SNPRootComponent(ModelComponent):
    """
    Model component that hooks up the SNP likelihood algorithm to the root
    of the network.
    """
    def __init__(self, dependencies: set[type]) -> None:
        """
        Initialize the SNP root component with a set of dependencies.

        Args:
            dependencies (set[type]): a set of types that this component depends
                                       on.
        Returns:
            N/A  
        """ 
        super().__init__(dependencies)
        
    def build(self, model : Model) -> None:
        """
        Build the SNP likelihood algorithm, and hook it up to the root of the
        network.

        Args:
            model (Model): the model that the SNP likelihood algorithm will be
                            hooked up to.
        Returns:
            N/A
        """
        root_prob = BiMarkersLikelihood()
        
        # Hook to root of network as parent
        net_root : ModelNode = model.nodetypes["root"][0]
        net_root.join(root_prob)
        
        # Bookkeep
        model.all_nodes[BiMarkersLikelihood].append(root_prob)
    
class SNPTransitionComponent(ModelComponent):
    """
    Model component that hooks up the SNP transition matrix, Q, to each network
    node and the root likelihood calculator.
    """
    def __init__(self, dependencies: set[type]) -> None:
        """
        Initialize the SNP transition component with a set of dependencies.

        Args:
            dependencies (set[type]): Set of types that this component depends
                                        on.
        Returns:
            N/A
        """
        super().__init__(dependencies)
    
    def build(self, model : Model) -> None:
        """
        Build the SNP transition matrix, Q, and hook it up to each network node
        and the root probability calculator.
        Args:
            model (Model): the model that the SNP transition matrix will be
                            hooked up to.   
        Returns:
            N/A
        """
        q_node = BiMarkersTransitionMatrixNode()
        
        # Hook to root probability
        root_prob = model.all_nodes[BiMarkersLikelihood][0]
        q_node.join(root_prob)
        
        # Hook to each Network Node 
        join_network(q_node, model)
        
        # Bookkeep
        model.all_nodes[BiMarkersTransitionMatrixNode].append(q_node)
    
class SNPParamComponent(ModelComponent):
    """
    Model component that hooks up model parameters (u, v, coal, and others) to
    any node that requires it for calculation of various likelihoods and 
    constructs.
    """
    
    def __init__(self, dependencies : set[type], params : dict) -> None:
        """
        Initialize the SNP parameter component with a set of dependencies and
        a dictionary of parameters.

        Args:
            dependencies (set[type]): Set of types that this component depends
                                        on.
            params (dict): Dictionary of parameters that will be used in the
                           model.
        Returns:
            N/A
        """
        super().__init__(dependencies)  
        self.params = params
    
    def build(self, model : Model) -> None:
        """
        Build the SNP model parameters and hook them up to the SNP transition
        matrix, Q, the root probability calculator, and the network.

        Args:
            model (Model): the model that the SNP model parameters will be
                            hooked up to.
        Returns:
            N/A
        """     
        # All parameters
        u_node = U(self.params["u"])
        v_node = V(self.params["v"])
        coal_node = Coal(self.params["coal"])
        samples_node = Samples(self.params["samples"])
        site_node = SiteParameter(self.params["sites"])
        
        #Hook to Q matrix node
        trans_nodes = model.all_nodes[BiMarkersTransitionMatrixNode]    
        q_node : BiMarkersTransitionMatrixNode = trans_nodes[0]
        
        u_node.join(q_node)
        v_node.join(q_node)
        coal_node.join(q_node)
        samples_node.join(q_node)
        
        #Hook to root probability calculator
        root_prob = model.all_nodes[BiMarkersLikelihood][0]
        samples_node.join(root_prob)
        site_node.join(root_prob)
        
        # Hook to network
        join_network(site_node, model)
        join_network(samples_node, model)
        
        # Bookkeep
        model.all_nodes[Parameter].extend([u_node,
                                          v_node, 
                                          coal_node, 
                                          samples_node, 
                                          site_node])


def build_model(filename : str, 
                net : Network,
                u : float = .5 ,
                v : float = .5, 
                coal : float = 1, 
                grouping : dict = None, 
                auto_detect : bool = False) -> Model:
    """
    
    Args:
        filename (str): string path destination of a nexus file that contains 
                        SNP data
        net (Network): A phylogenetic network.
        u (float, optional): Parameter for the probability of an
                             allele changing from red to green. Defaults to .5.
        v (float, optional): Parameter for the probability of an
                             allele changing from green to red. Defaults to .5.
        coal (float, optional): Parameter for the rate of coalescence. 
                                Defaults to 1.
        grouping (dict, optional): Grouping of data sequences. Defaults to None.
        auto_detect (bool, optional): Flag that, if enabled, will automatically
                                      group data sequences together based on 
                                      naming similarity. Defaults to False.
        
    Returns:
        Model: A SNP model
    """
    # Parse the data file into a sequence alignment
    aln = MSA(filename, grouping = grouping, grouping_auto_detect = auto_detect)
    
    # Set up all model components
    network = NetworkComponent(net = net)
    
    msa = MSAComponent({NetworkComponent}, aln, grouping = grouping)

    param = SNPParamComponent({NetworkComponent,
                               MSAComponent,
                               SNPRootComponent,
                               SNPTransitionComponent}, 
                              {"samples": aln.total_samples(),
                               "u": u, 
                               "v": v, 
                               "coal" : coal,
                               "sites" : aln.dim()[1],
                               "grouping" : grouping})
    
    vpi = VPIComponent({NetworkComponent, SNPRootComponent})
    root = SNPRootComponent({NetworkComponent})
     
    # Build model
    snp_model : Model = ModelFactory(network, msa, param, vpi, root).build()
    
    return snp_model

###########################
### METHOD ENTRY POINTS ###
###########################

def MCMC_BIMARKERS(filename: str, 
                   u : float = .5 ,
                   v : float = .5, 
                   coal : float = 1, 
                   grouping : dict = None, 
                   auto_detect : bool = False) -> dict[Network, float]:
    
    """
    Given a set of taxa with SNP data, perform a Markov Chain Monte Carlo
    chain to infer the most likely phylogenetic network that describes the
    taxa and data.

    Args:
        filename (str): string path destination of a nexus file that contains 
                        SNP data
        u (float, optional): Parameter for the probability of an
                             allele changing from red to green. Defaults to .5.
        v (float, optional): Parameter for the probability of an
                             allele changing from green to red. Defaults to .5.
        coal (float, optional): Parameter for the rate of coalescence. 
                                Defaults to 1.
        grouping (dict, optional): TODO: Figure out an apt description. Defaults to None.
        auto_detect (bool, optional): Flag that, if enabled, will automatically
                                      group data sequences together based on 
                                      naming similarity. Defaults to False.
        
    Returns:
        dict[Network, float]: The log likelihood (a negative number) of the most 
                              probable network, along with the network itself 
                              that achieved that score.
    """

    # Parse the data file into a sequence alignment
    aln = MSA(filename, grouping = grouping, grouping_auto_detect = auto_detect)
    
    # Generate starting network and place into model component
    start_net = CBDP(1, .5, aln.num_groups()).generate_network()
    
    snp_model = build_model(filename, 
                            start_net,
                            u, 
                            v, 
                            coal, 
                            grouping, 
                            auto_detect)
    
    mh = MetropolisHastings(ProposalKernel(),
                            data = Matrix(aln, Alphabet("SNP")), 
                            num_iter = 800,
                            model = snp_model) 
     
    result_state = mh.run()
    result_model = result_state.current_model

    return {result_model.network : result_model.likelihood()}

def SNP_LIKELIHOOD(filename : str,
                   u : float = .5 ,
                   v : float = .5, 
                   coal : float = 1, 
                   grouping : dict = None, 
                   auto_detect : bool = False) -> float:
    """
    Given a set of taxa with SNP data and a phylogenetic network, calculate the 
    likelihood of the network given the data using the SNP likelihood algorithm.

    Args:
        filename (str): string path destination of a nexus file that 
                        contains SNP data and a network
        u (float, optional): Parameter for the probability of an
                             allele changing from red to green. Defaults to .5.
        v (float, optional): Parameter for the probability of an
                             allele changing from green to red. Defaults to .5.
        coal (float, optional): Parameter for the rate of coalescence. 
                                Defaults to 1.
        grouping (dict, optional): Grouping of data sequences. Defaults to None.
        auto_detect (bool, optional): Flag that, if enabled, will automatically
                                      group data sequences together based on 
                                      naming similarity. Defaults to False.
        
    Returns:
        float: The log likelihood (a negative number) of the network.
    """
    
    net = NetworkParser(filename).get_network(0)
        
    snp_model = build_model(filename, 
                            net,
                            u, 
                            v, 
                            coal, 
                            grouping, 
                            auto_detect)
    
    return snp_model.likelihood()
       
def SNP_LIKELIHOOD_DATA(filename : str,
                         set_reds : dict,
                         u : float = .5 ,
                         v : float = .5, 
                         coal : float = 1,
                         grouping : dict = None, 
                         auto_detect : bool = False) -> float:
    """
    THIS FUNCTION IS ONLY FOR TESTING PURPOSES
    Given a set of taxa with SNP data and a phylogenetic network, calculate the 
    likelihood of the network given the data using the SNP likelihood algorithm.

    Args:
        filename (str): string path destination of a nexus file that 
                        contains SNP data and a network
        set_reds (dict): A dictionary of taxa labels and their associated 
                         red allele data.
        u (float, optional): Parameter for the probability of an
                             allele changing from red to green. Defaults to .5.
        v (float, optional): Parameter for the probability of an
                             allele changing from green to red. Defaults to .5.
        coal (float, optional): Parameter for the rate of coalescence. 
                                Defaults to 1.
        grouping (dict, optional): TODO: Figure out an apt description. Defaults to None.
        auto_detect (bool, optional): Flag that, if enabled, will automatically
                                      group data sequences together based on 
                                      naming similarity. Defaults to False.
        
    Returns:
        float: The log likelihood (a negative number) of the network.
    """
    
    net = NetworkParser(filename).get_network(0)
        
    snp_model = build_model(filename, 
                            net,
                            u, 
                            v, 
                            coal, 
                            grouping, 
                            auto_detect)
    
    for taxa in snp_model.all_nodes[ExtantSpecies]:
        leaf : ExtantSpecies = taxa
        leaf.update(DataSequence(set_reds[leaf.label], leaf.label))
    
    return snp_model.likelihood()
            
            
        
        
