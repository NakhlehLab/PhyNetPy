from math import sqrt, comb, pow
import numpy as np

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


def Rule0(node_par, site_count, vector_len):

    F_b = np.zeros((vector_len, site_count))
    
    # Compute leaf partials via EQ 12
    reds = node_par.red_count()

    for site in range(site_count):
        for index in range(vector_len):
            actual_index = undo_index(index)
            n = actual_index[0]
            r = actual_index[1]

            # EQUATION 12
            if reds[site] == r and n == node_par.samples():
                F_b[index][site] = 1
                
    return F_b

def Rule1(F_b : np.ndarray, site_count : int, vector_len : int, m_y : int, Qt : np.ndarray) -> np.ndarray:
    
    # ONLY CALCULATE F_T FOR NON ROOT BRANCHES
    F_t = np.zeros((vector_len, site_count))
    
    # Do this for each marker
    for site in range(site_count):
        for ft_index in range(0, vector_len):
            tot = 0
            actual_index = undo_index(ft_index)
            n_t = actual_index[0]

            for n_b in range(n_t, m_y + 1):  # n_b always at least 1
                for r_b in range(0, n_b + 1):
                    index = map_nr_to_index(n_b, r_b)
                    exp_val = Qt[index][ft_index]  # Q(n,r);(n_t, r_t)

                    tot += exp_val * F_b[index][site]

            F_t[ft_index][site] = tot
    
    return F_t
            
            
def Rule2(F_t_y : np.ndarray, F_t_z : np.ndarray, site_count : int, vector_len : int) -> np.ndarray:
    
    F_b = np.zeros((vector_len, site_count))
    for site in range(site_count):
        for index in range(vector_len):
            actual_index = undo_index(index)
            n = actual_index[0]
            r = actual_index[1]
            tot = 0

            # EQUATION 19
            for n_y in range(1, n):
                for r_y in range(0, r + 1):
                    if r_y <= n_y and r - r_y <= n - n_y:  # Ensure that the combinatorics makes sense
                        # Compute the constant term
                        const = comb(n_y, r_y) * comb(n - n_y, r - r_y) / comb(n, r)

                        # Grab Ftz(n_y, r_y)
                        term1 = F_t_z[map_nr_to_index(n_y, r_y)][site]

                        # Grab Fty(n - n_y, r - r_y)
                        term2 = F_t_y[map_nr_to_index(n - n_y, r - r_y)][site]

                        tot += term1 * term2 * const

            F_b[index][site] = tot
            
    return F_b

def Rule3(F_t_x : dict, g_this : float, g_that : float, site_count : int):
    print("Hello from rule 3")
        
def Rule4():
    print("Hello from rule 4")
        



    



def to_array(Fb_map :dict, vector_len, site_count):
    
    F_b = np.zeros((vector_len, site_count))  
    for site in range(site_count):
        for nr_pair, prob in Fb_map[site].items():
            #nr_pair should be of the form ((n),(r))
            F_b[int(map_nr_to_index(nr_pair[0][0], nr_pair[1][0]))][site] = prob
    
    return F_b



def rn_to_rn_minus_one(set_of_rns : dict):
    """
    This is a function defined as
    
    f: Rn;Rn -> Rn-1;Rn-1.
    
    set_of_rns should be a mapping in the form of {(nx , rx) -> probability in R} where nx and rx are both vectors in Rn
    
    This function takes set_of_rns and turns it into a mapping {(nx[:-1] , rx[:-1]) : set((nx[-1] , rx[-1], probability))} where the keys
    are vectors in Rn-1, and their popped last elements and the probability is stored as values.

    Args:
        set_of_rns (dict): a mapping in the form of {(nx , rx) -> probability in R} where nx and rx are both vectors in Rn
    """
    
    rn_minus_one = {}
    
    for vectors, prob in set_of_rns.items():
        nx = vectors[0]
        rx = vectors[1]
        
        new_value = (nx[-1], rx[-1], prob)
        new_key = (nx[:-1], rx[:-1])
        if new_key in rn_minus_one.keys():
            rn_minus_one[new_key].add(new_value)
        else:
            init_value = set()
            init_value.add(new_value)
            rn_minus_one[new_key] = init_value

    return rn_minus_one


def rn_to_rn_minus_two(set_of_rns : dict):
    """
    This is a function defined as
    
    f: Rn;Rn -> Rn-2;Rn-2.
    
    set_of_rns should be a mapping in the form of {(nx , rx) -> probability in R} where nx and rx are both vectors in Rn
    
    This function takes set_of_rns and turns it into a mapping {(nx[:-2] , rx[:-2]) : set((nx[-2:] , rx[-2:], probability))} where the keys
    are vectors in Rn-2, and their popped last 2 elements and the probability is stored as values.

    Args:
        set_of_rns (dict): a mapping in the form of {(nx , rx) -> probability in R} where nx and rx are both vectors in Rn
    """
    
    rn_minus_two = {}
    
    for vectors, prob in set_of_rns.items():
        nx = vectors[0]
        rx = vectors[1]
        #The vectors must be long enough
        new_value = (nx[-2:], rx[-2:], prob)
        new_key = (nx[:-2], rx[:-2])
        if new_key in rn_minus_two.keys():
            rn_minus_two[new_key].add(new_value)
        else:
            init_value = set()
            init_value.add(new_value)
            rn_minus_two[new_key] = init_value

    return rn_minus_two



def eval_Rule1(F_b : dict, nx : list, n_xtop : int, rx: list, r_xtop: int, Qt: np.ndarray, mx : int) -> dict:
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

def eval_Rule2(F_t_x : dict, F_t_y : dict, nx : list, ny : list, n_zbot : int, rx: list, ry : list, r_zbot: int, mx : int, my: int) -> dict:
    evaluation = 0
    for n_xtop in range(1, n_zbot):
        for r_xtop in range(0, r_zbot + 1):
            if r_xtop <= n_xtop and r_zbot - r_xtop <= n_zbot - n_xtop:
                # for n_xtop in range(max(0, n_zbot - my), min(mx, n_zbot) + 1): #inclusive
                #     for r_xtop in range(max(0, n_xtop + r_zbot - n_zbot), min(n_xtop, r_zbot) + 1): #also inclusive
           
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
    #Rule 3 Equation
    try:
        evaluation = F_t[(tuple(np.append(nx, n_ybot + n_zbot)), tuple(np.append(rx, r_ybot + r_zbot)))] * comb(n_ybot + n_zbot, n_ybot) * pow(gamma_y, n_ybot) * pow(gamma_z, n_zbot)
    except KeyError:
        evaluation = 0
        
    return [(tuple(np.append(np.append(nx, n_ybot), n_zbot)), tuple(np.append(np.append(rx, r_ybot), r_zbot))), evaluation]
    

def eval_Rule4(F_t: dict, nz: list, rz:list, n_zbot:int, r_zbot: int)-> dict:
    evaluation = 0
    for n_xtop in range(0, n_zbot + 1):
        for r_xtop in range(0, r_zbot + 1):
            if r_xtop <= n_xtop and r_zbot - r_xtop <= n_zbot - n_xtop:
                
                #RULE 4 EQUATION
                const = comb(n_xtop, r_xtop) * comb(n_zbot - n_xtop, r_zbot - r_xtop) / comb(n_zbot, r_zbot)
                print(tuple(np.append(np.append(nz, n_xtop), n_zbot - n_xtop)))
                print(tuple(np.append(np.append(rz, r_xtop), r_zbot - r_xtop)))
                try:
                    term1= F_t[(tuple(np.append(np.append(nz, n_xtop), n_zbot - n_xtop)), tuple(np.append(np.append(rz, r_xtop), r_zbot - r_xtop)))]
                    evaluation += term1 * const
                    if evaluation != 0:
                        print("COMPUTING SOME NUMBERS")
                    
                except KeyError:
                    evaluation += 0
                    print("OOPSIES")
    
    return [(tuple(np.append(nz, n_zbot)), tuple(np.append(rz, r_zbot))), evaluation] 




class PartialLikelihoods:
    
    def __init__(self) -> None:
        
        # A map from a vector of population interfaces (vpi)-- represented as a tuple of strings-- to probability maps
        # defined by rules 0-4.
        self.vpis : dict = {}
        
    def Rule0(self, node_par, site_count, vector_len, branch_index : int)->tuple:

        F_b = {}
        
        # Compute leaf partials via EQ 12
        reds = node_par.red_count()

        for site in range(site_count):
            F_b[site] = {}
            for index in range(vector_len):
                actual_index = undo_index(index)
                n = actual_index[0]
                r = actual_index[1]

                # EQUATION 12
                if reds[site] == r and n == node_par.samples():
                    F_b[site][(tuple([n]),tuple([r]))] = 1
                else:
                    F_b[site][(tuple([n]),tuple([r]))] = 0
        
        vpi_key = tuple(["branch_" + str(branch_index) + ": bottom"])       
        self.vpis[vpi_key] = F_b
        
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
        
        F_t = {}
        F_b : dict = self.vpis[vpi_key]
        
        print("CALCULATING WITH RULE 1 FOR VPI:" + str(vpi_key))
        print("BRANCH THAT WERE CALCULATING THE TOP FOR:" + str(branch_index))
        
        if "branch_" + str(branch_index) + ": bottom" != vpi_key[-1]:
            
            vpi_key_temp = self.reorder_vpi_rule1(vpi_key, site_count, branch_index)
            del self.vpis[vpi_key]
            vpi_key = vpi_key_temp
        
        for site in range(site_count):
            #Gather all combinations of nx, rx values 
            nx_rx_map = rn_to_rn_minus_one(F_b[site])
            
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
        
        new_vpi_key = list(vpi_key)
        new_vpi_key[vpi_key.index("branch_" + str(branch_index) + ": bottom")] = "branch_" + str(branch_index) + ": top"
        new_vpi_key = tuple(new_vpi_key)
        
        self.vpis[new_vpi_key] = F_t
        del self.vpis[vpi_key]

        return new_vpi_key
                
                
    def Rule2(self, vpi_key_x : tuple, vpi_key_y :tuple, site_count : int, vector_len : int, mx: int, my: int, branch_index_x:int, branch_index_y:int, branch_index_z:int) -> tuple:
        """
        Given branches x and y that have no leaf descendents in common and a parent branch z, and partial likelihood mappings for the population 
        interfaces that include x_top and y_top, we would like to calculate the partial likelihood mapping for the population interface
        that includes z_bottom.
        
        This uses Rule 2 from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932, Rabier et. al.

        Args:
            F_t_x (dict): A mapping containing mappings from vectors (nx, n_xtop; rx, r_xtop) to probability values for each site
            F_t_y (dict): A mapping containing mappings from vectors (nx, n_ytop; rx, r_ytop) to probability values for each site
            site_count (int): number of total sites in the multiple sequence alignment
            vector_len (int): number of possible lineages at the root
            mx (int): possible lineages at x
            my (int): possible lineages at y

        Returns:
            dict: A mapping F_b, in the same format as F_t_x and F_t_y, that represents the partial likelihoods for the population interface
                x, y, z_bottom.
        """
        
        F_b = {}
        F_t_x : dict = self.vpis[vpi_key_x]
        F_t_y : dict = self.vpis[vpi_key_y]
        
        print("VPI Y: " + str(vpi_key_y) + " LAST ENTRY SHOULD BE " + str(branch_index_y))
        print("VPI X: " + str(vpi_key_x) + " LAST ENTRY SHOULD BE " + str(branch_index_x))
        
        if "branch_" + str(branch_index_x) + ": top" != vpi_key_x[-1]:
            
            vpi_key_xtemp = self.reorder_vpi_rule2(vpi_key_x, site_count, branch_index_x)
            del self.vpis[vpi_key_x]
            vpi_key_x = vpi_key_xtemp
        
        if "branch_" + str(branch_index_y) + ": top" != vpi_key_y[-1]:
            
            vpi_key_ytemp = self.reorder_vpi_rule2(vpi_key_y, site_count, branch_index_y)
            del self.vpis[vpi_key_y]
            vpi_key_y = vpi_key_ytemp
        
        for site in range(site_count):
            nx_rx_map_y = rn_to_rn_minus_one(F_t_y[site])
            nx_rx_map_x = rn_to_rn_minus_one(F_t_x[site])
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
                        entry = eval_Rule2(F_t_x[site], F_t_y[site], nx, ny, n_bot, rx, ry, r_bot, mx, my)
                        F_b[site][entry[0]] = entry[1]
        
        new_vpi_key_x= list(vpi_key_x)
        new_vpi_key_x.remove("branch_" + str(branch_index_x) + ": top")
        
        new_vpi_key_y= list(vpi_key_y)
        new_vpi_key_y.remove("branch_" + str(branch_index_y) + ": top")
        
        new_vpi_key = tuple(np.append(new_vpi_key_x, np.append(new_vpi_key_y, "branch_" + str(branch_index_z) + ": bottom")))
        
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key_x]
        del self.vpis[vpi_key_y]
                            
        return new_vpi_key

    def Rule3(self, vpi_key : tuple, g_this : float, g_that : float, site_count : int, m: int, branch_index_x:int, branch_index_y:int, branch_index_z:int)->tuple:
        """
        Given a branch x, its partial likelihood mapping at x_top, and parent branches y and z, we would like to compute
        the partial likelihood mapping for the population interface x, y_bottom, z_bottom.

        This uses Rule 3 from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932, Rabier et. al.
        
        Args:
            F_t_x (dict): A mapping containing mappings from vectors (nx, n_xtop; rx, r_xtop) to probability values for each site
            g_this (float): gamma inheritance probability for branch y
            g_that (float): gamma inheritance probability for branch z
            site_count (int): number of sites
            m (int): number of possible lineages at x.

        Returns:
            dict: A mapping in the same format as F_t_x, that represents the partial likelihoods at the 
            population interface x, y_bottom, z_bottom
        """
        F_b = {}
        F_t_x : dict = self.vpis[vpi_key]
        
        for site in range(site_count):
            nx_rx_map = rn_to_rn_minus_one(F_t_x[site])
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
                                
        new_vpi_key= list(vpi_key)
        new_vpi_key.remove("branch_" + str(branch_index_x) + ": top")
        new_vpi_key.append("branch_" + str(branch_index_y) + ": bottom")
        new_vpi_key.append("branch_" + str(branch_index_z) + ": bottom")
        
        new_vpi_key = tuple(new_vpi_key)
        
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key]
                
        return new_vpi_key               
            
    def Rule4(self, vpi_key : tuple, site_count : int, vector_len : int, branch_index_x : int, branch_index_y : int, branch_index_z : int):
        
        F_b = {}
        F_t : dict = self.vpis[vpi_key]
       
        
        print("VPI: " + str(vpi_key) + " LAST ENTRY SHOULD BE " + str(branch_index_x))
        
        if "branch_" + str(branch_index_x) + ": top" != vpi_key[-1]:
            
            vpi_key_temp = self.reorder_vpi_rule2(vpi_key, site_count, branch_index_x)
            del self.vpis[vpi_key]
            vpi_key = vpi_key_temp
        
        
        for site in range(site_count):
            nx_rx_map = rn_to_rn_minus_two(F_t[site])
            
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
        
        new_vpi_key = list(vpi_key)
        new_vpi_key.remove("branch_" + str(branch_index_x) + ": top")
        new_vpi_key.remove("branch_" + str(branch_index_y) + ": top")
        new_vpi_key.append("branch_" + str(branch_index_z) + ": bottom")
        
        new_vpi_key = tuple(new_vpi_key)
        
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key]
                            
        return new_vpi_key
    
    def reorder_vpi_rule1(self, vpi_key, site_count, branch_index):
        
        former_index = list(vpi_key).index("branch_" + str(branch_index) + ": bottom")
        new_vpi_key = list(vpi_key)
        new_vpi_key.append(new_vpi_key.pop(former_index))
        F = self.vpis[vpi_key]
        new_F = {}
        
        for site in range(site_count):
            new_F[site] = {}
            for vectors, prob in F[site].items():
                nx = list(vectors[0])
                rx = list(vectors[1])
                new_nx = list(nx)
                new_rx = list(rx)
                
                new_nx.append(new_nx.pop(former_index))
                new_rx.append(new_rx.pop(former_index))
                
                new_F[site][(tuple(new_nx), tuple(new_rx))] = prob
        
    
        self.vpis[tuple(new_vpi_key)] = new_F
        return tuple(new_vpi_key)
    
    def reorder_vpi_rule2(self, vpi_key, site_count, branch_index):
        
        former_index = list(vpi_key).index("branch_" + str(branch_index) + ": top")
        new_vpi_key = list(vpi_key)
        new_vpi_key.append(new_vpi_key.pop(former_index))
        F = self.vpis[vpi_key]
        new_F = {}
        
        for site in range(site_count):
            new_F[site] = {}
            for vectors, prob in F[site].items():
                nx = list(vectors[0])
                rx = list(vectors[1])
                new_nx = list(nx)
                new_rx = list(rx)
                
                new_nx.append(new_nx.pop(former_index))
                new_rx.append(new_rx.pop(former_index))
                
                new_F[site][(tuple(new_nx), tuple(new_rx))] = prob
        
    
        self.vpis[tuple(new_vpi_key)] = new_F
        return tuple(new_vpi_key)
    
    def get_key_with(self, branch_index):
        for vpi_key in self.vpis:
            top = "branch_" + str(branch_index) + ": top"
            bottom = "branch_" + str(branch_index) + ": bottom"
            
            if top in vpi_key or bottom in vpi_key:
                return vpi_key
        return None
            
                
                
                
            
            
        
        
