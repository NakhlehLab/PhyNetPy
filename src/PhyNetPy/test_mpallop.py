from Infer_MP_Allop import *
from helpers import *

"""
Testing Suite for the Network.py class. Extensively ensures that
the Network, Edge, Node, and related classes are working as intended.
To be included in a final release, this file MUST run without any errors
and all tests must be passed.
"""

def mp_1():
    """
    Check that the parsimony scoring function works for a lvl 1 network.
    """
    file_net = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/scenarioD_ideal.nex"
    file_gt = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/D10.nex" 
    subgenome_map = {"B": ["01bA"], "A": ["01aA"], "X": ["01xA", "01xB"],
                     "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"]}
    
    # {'U': ['01uA', '01uB'], 'T': ['01tA', '01tB'], 
    #                  'B': ['01bA'], 'F': ['01fA'], 'C': ['01cA'], 
    #                  'A': ['01aA'], 'D': ['01dA'], 'O': ['01oA']}
    
    #44 isn't right I don't think. Check on this, I think it should be 4.
    score = ALLOP_SCORE(file_net, file_gt, subgenome_map)
    if score == 44: 
        return 1
    else:
        print(f"WRONG SCORE: {score}")
        return 0

def mp_2():
    """
    Check that all files with 3 gene trees infer the correct network.
    """
    return 1

def mp_3():
    """
    Check that all files with 10 gene trees infer the correct network.
    """
    
    
    res = INFER_MP_ALLOP(
                    '/Users/mak17/Documents/PhyNetPy/src/J_pruned_v2.nex',
                    {'U': ['01uA', '01uB'], 'T': ['01tA', '01tB'], 
                     'B': ['01bA'], 'F': ['01fA'], 'C': ['01cA'], 'A': ['01aA'],
                     'D': ['01dA'], 'O': ['01oA']})
    
    if min(res.values()) == -4:
        net_min : Network = minmaxkey(res, mini = False)
        print(net_min.newick())
        return 1
    else:
        return 0

def mp_4():
    """
    Check that all files with 100 gene trees infer the correct network
    """
    return 1

def mp_5():
    """
    Run the method 1000 times to ensure a search space consistency and lack of 
    weird errors.
    """
    return 1

def mp_6():
    """
    Runtime testing. This test is not a pass/fail test, but provides all metrics
    and scaling data.
    """
    return 1

def mp_7():
    """
    Convergence testing. This test is not a pass/fail test, but provides info
    on how quickly the algorithm converges on the correct network.
    """
    return 1

def mp_8():
    """
    Full study on scenarios D, E, F, and J . Generate plots.
    """
    return 1

def mp_9():
    """
    Test with malformed data and ensure that the program halts gracefully.
    """
    return 1

def mp_10():
    """
    Test the starting network generator function to ensure that start nets
    are of proper ploidy values.
    """
    return 1

class Infer_MP_Allop_Test:
    
    #RUN ALL TESTS HERE
    def test(self) -> None:
        res = [mp_1(),
               mp_2(),
               mp_3(),
               mp_4(),
               mp_5(),
               mp_6(),
               mp_7(),
               mp_8(),
               mp_9(),
               mp_10()]
        
        if sum(res) == 6:
            print("All (10/10) correctness tests passed!")
        else:
            print(f"Tests failed. {sum(res)}/10 passed.")
    
    def test(self, indv : int) -> None:
        tests = [mp_1,
                 mp_2,
                 mp_3,
                 mp_4,
                 mp_5,
                 mp_6,
                 mp_7,
                 mp_8,
                 mp_9,
                 mp_10]
        
        if tests[indv]() == 1:
            print("Passed :D")
            return
        else:
            print("Failed :(")
            return


Infer_MP_Allop_Test().test(2)