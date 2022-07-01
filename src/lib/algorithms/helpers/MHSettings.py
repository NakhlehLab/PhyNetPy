import os
import sys
import math
import copy

DEBUG_MODE = False
EPSILON = 2.220446049250313E-16
_MCMC = True
_Forbid_Net_Net = True
_Forbid_Net_Triangle = True
_CHAIN_LEN = 500000
_BURNIN_LEN = 200000
_SAMPLE_FREQUENCY = 500
_SEED = 12345678
_NUM_THREADS = os.cpu_count()
_OUT_DIRECTORY = sys.getProperty("user.home") ##??
    
    
_MC3_CHAINS = None
_NET_MAX_RETI = 4
_TAXON_MAP = None

#pop size
_ESTIMATE_POP_SIZE = True
_CONST_POP_SIZE = True
_ESTIMATE_POP_PARAM = True
## priors
_POISSON_PARAM = 1.0
_TIMES_EXP_PRIOR = False
_DIAMETER_PRIOR = False
_NETWORK_SIZE_PRIOR = True
## Substitution model
_SUBSTITUTION_MODEL = "JC"
_BASE_FREQS = None
_TRANS_RATES = None
## starting state
_POP_SIZE_MEAN = 0.036
_START_NET = None
_START_GT_LIST = None
## summary
_TOPK_NETS = 10
## diploid phasing
_DIPLOID_SPECIES = None
## divergence time window size
_TIME_WINDOW_SIZE = 0.04
_POP_SIZE_WINDOW_SIZE = 0.04

## --- net ---
NET_INTI_SCALE = 0.95
DEFAULT_NET_LEAF_HEIGHT = 0
DEFAULT_NET_ROOT_HEIGHT = 6
NET_MAX_HEIGHT = 10 ##TODO: MCMC->1000
## --- tree ---
TREE_INTI_SCALE = 1.05
DEFAULT_TREE_LEAF_HEIGHT = 0
ROOT_TIME_UPPER_BOUND = 10
## --- moves ---
INVALID_MOVE = -math.inf 
MOVE_TYPE = ["TREE", "NETWORK", "ALL", "PRIOR"]
TARGET_ACRATE = 0.345
Transform = [None, "Log", "Sqrt"]
## --- MCMC chain ---
SWAP_FREQUENCY = 100
## --- priors ---
EXP_PARAM = 10 ## Mr.Bayes
GAMMA_SHAPE = 2 ## *BEAST
## --- substitution model ---
ESTIMATE_SUBSTITUTION = False ## TODO future improvement
## --- samples ---
SAMPLETYPE = ["Tree", "Network", "ArrayParam" , "Param"]
## --- move weights ---
DISABLE_PARAMETER_MOVES = False
DISABLE_TOPOLOGY_MOVES = False
SAMPLE_SPLITTING = False ## experimental!

DIMENSION_CHANGE_WEIGHT = 0.005

Net_Op_Weights = [
            0.10, 0.04, 0.00,
            0.04, 0.05,
            0.20, 0.15, 0.03, 0.10, DIMENSION_CHANGE_WEIGHT,
            0.06 - DIMENSION_CHANGE_WEIGHT, 0.10, DIMENSION_CHANGE_WEIGHT, 0.06 - DIMENSION_CHANGE_WEIGHT
]
    ## ChangePopSize ScalePopSize ScaleAll ScaleTime ScaleRootTime
    ## ChangeTime SlideSubNet SwapNodes MoveTail AddReticulation
    ## FlipReticulation MoveHead DeleteReticulation ChangeInheritance
Net_Tree_Op_Weights = [
            0.15, 0.04, 0.00,
            0.04, 0.05,
            0.30, 0.27, 0.06, 0.07 - DIMENSION_CHANGE_WEIGHT * 2, DIMENSION_CHANGE_WEIGHT * 2
]

SEARCH_DIMENSION_CHANGE_WEIGHT = 0.03

Search_Net_Op_Weights = [
            0.02, 0.01, 0.00,
            0.04, 0.05,
            0.20, 0.15, 0.03, 0.10, SEARCH_DIMENSION_CHANGE_WEIGHT,
            0.16 - SEARCH_DIMENSION_CHANGE_WEIGHT, 0.10, SEARCH_DIMENSION_CHANGE_WEIGHT, 0.16 - SEARCH_DIMENSION_CHANGE_WEIGHT, 0.0
]
    ## ChangePopSize ScalePopSize ScaleAll ScaleTime ScaleRootTime
    ## ChangeTime SlideSubNet SwapNodes MoveTail AddReticulation
    ## FlipReticulation MoveHead DeleteReticulation ChangeInheritance
Search_Net_Tree_Op_Weights = [
            0.03, 0.01, 0.00,
            0.04, 0.05,
            0.30, 0.27, 0.06, 0.07 - SEARCH_DIMENSION_CHANGE_WEIGHT * 2, SEARCH_DIMENSION_CHANGE_WEIGHT * 2
]

PopSize_Op_Weights = [0.5, 1.0]


def getOperationWeights(weights, start, end):
        if(DISABLE_PARAMETER_MOVES):
                weights[0] = 0.0
                weights[1] = 0.0
                weights[2] = 0.0
                weights[3] = 0.0
                weights[4] = 0.0
                weights[5] = 0.0
        

        if(DISABLE_TOPOLOGY_MOVES):

                for i in range (6, math.min(end, 13)):
                        weights[i] = 0.0
            

                weights[0] = 0.0
                weights[1] = 0.0
                weights[2] = 0.0
                weights[3] = 0.0
                weights[4] = 0.0
                weights[5] = 0.0
        

        arr = []
        sum = 0
        for i in range(start, end):
            sum += weights[i]
        
        for i in range(len(weights)):
            if (i < start):
                arr[i] = 0
            elif (i >= end - 1):
                arr[i] = 1
            else:
                arr[i] = weights[i] / sum + (i == 0 ? 0 : arr[i-1]) ##wtf?
            
        
        return arr
    

def getOperationWeights2(weights):
        if(DISABLE_PARAMETER_MOVES):
            weights[0] = 0.0
            weights[1] = 0.0
            weights[2] = 0.0
            weights[3] = 0.0
            weights[4] = 0.0
            weights[5] = 0.0
        

        if(DISABLE_TOPOLOGY_MOVES):
                for i in range (6, math.min(len(weights), 13)):
                        weights[i] = 0.0           

        arr = []
        sum = 0
        for weight in weights:
            sum += weight
        
        for i in range(len(weights)):
            arr[i] = weights[i] / sum + (i == 0 ? 0 : arr[i-1]) ##wtF???
        
        arr[len(weights)-1] = 1
        return arr
    

def sum(array):
        return array[0] + sum(array[1:])



def deepCopyArray(array):
        return copy.deepcopy(array)

def shallowCopyArray(array):
        return copy.copy(array)


def calcDelta(operator,  logAlpha):
        target = TARGET_ACRATE
        count = (operator._rejCorrectionCounter + operator._acCorrectionCounter + 1.0)

        match (operator._transform): ## string values?
            case "Log":
                count = math.log(count + 1.0) ##?
        
            case "Sqrt":
                count = math.sqrt(count)
                     
        
        deltaP = (math.exp(math.min(logAlpha, 0)) - target) / count

        return (deltaP > -.MAX_VALUE and deltaP < .MAX_VALUE) ? deltaP : 0 ##WTF?
    

def plotNetwork(netStr):
        net = Networks.readNetwork(netStr)
        for n in Networks.postTraversal(net):
            node = n
            for p in node.getParents():
                par = p
                node.setParentSupport(par, node.NO_SUPPORT)
                node.setParentProbability(par, node.NO_PROBABILITY)
            
        
        return net.to()
    

def equals(a,  b):
        return math.abs(a-b) < EPSILON
    

def varyPopSizeAcrossBranches():
        return _ESTIMATE_POP_SIZE and not _CONST_POP_SIZE
    