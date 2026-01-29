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

from PhyNetPy.Infer_MP_Allop import *
import cProfile
import time
import pytest
"""
Testing Suite for the Network.py class. Extensively ensures that
the Network, Edge, Node, and related classes are working as intended.
To be included in a final release, this file MUST run without any errors
and all tests must be passed.
"""

def _minmaxkey(mapping : dict[object, Union[int, float]],
               mini : bool = True) -> object:
    """
    Return the object in a mapping with the minimum or maximum value associated
    with it.

    Args:
        mapping (dict[object, int  |  float]): A mapping from objects to 
                                               numerical values
        mini (bool, optional): If True, return the object with the minimum
                               value. If False, return the object with the
                               maximum value. Defaults to True.
    Returns:
        object: The object with the minimimum or maximum value.
    """

    cur = math.inf
    cur_key = None
    if not mini:
        cur = cur * -1
        
    for key, value in mapping.items():
        if mini:
            if value < cur:
                cur = value
                cur_key = key
        else:
            if value > cur:
                cur = value
                cur_key = key
    
    return cur_key

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
    if score == 3: 
        return 1
    else:
        print(f"WRONG SCORE: {score}")
        return 0

def mp_2():
    """
    Check that all files with 3 gene trees infer the correct network.
    """
    
    file_net = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/scenarioD_ideal.nex"
    file_gt = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/D10.nex" 
    subgenome_map : dict[str, list[str]] = {"B": ["01bA"], "A": ["01aA"], "X": ["01xA", "01xB"],
                     "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"]}
    
    # start = time.time()
    res = INFER_MP_ALLOP_BOOTSTRAP(file_net,
                                   file_gt,
                                   subgenome_map)
    
    # end = time.time()
    
    # print(f"Method time: {end - start}")
    
    net_min : Network = _minmaxkey(res, mini = False)
    # print(net_min.newick())
    # print(res[net_min])
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
        net_min : Network = _minmaxkey(res, mini = False)
        print(net_min.newick())
        return 1
    else:
        net_min : Network = _minmaxkey(res, mini = False)
        print(net_min.newick())
        return 0

def mp_4():
    """
    Test on external data set (5 trees) that did not pass PhyloNet.
    """

    gt = GeneTrees(NetworkParser('/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/external_5.nex').get_all_networks(), external_naming)
    
    for tree in gt.trees:
        print(tree.newick())
       
    print(gt.mp_allop_map()) 
    
    start_t = time.time()
    res = INFER_MP_ALLOP(
                    '/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/external_5.nex',
                    gt.mp_allop_map())
    end_t = time.time()
    
    print(f"External with 5 GT run time: {end_t - start_t}")
    print(f"Results: {res}")
    return 1

def mp_5():
    """
    Scenario J Runtime test (100 Genes) r1 t20
    """
    gt = GeneTrees(NetworkParser('/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/J_100.nex').get_all_networks())
    
    start_t = time.time()
    res = INFER_MP_ALLOP(
                    '/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/J_100.nex',
                    gt.mp_allop_map())
    end_t = time.time()
    
    print(f"J with 100 GT run time: {end_t - start_t}")
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

@pytest.mark.skip(reason="Skipping Infer MP Allop tests for now")
class Test_Infer_MP_Allop:
    
    # RUN ALL TESTS HERE
    # def test(self) -> None: # type: ignore
    #     res = [mp_1(),
    #            mp_2(),
    #            mp_3(),
    #            mp_4(),
    #            mp_5(),
    #            mp_6(),
    #            mp_7(),
    #            mp_8(),
    #            mp_9(),
    #            mp_10()]
        
    #     if sum(res) == 10:
    #         print("All (10/10) correctness tests passed!")
    #     else:
    #         print(f"Tests failed. {sum(res)}/10 passed.")
    
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

# tester = Test_Infer_MP_Allop()
# time_start = time.time()
# for _ in range(20):
#     tester.test(2)
# time_end = time.time()
# print(f"Time taken: {time_end - time_start}")
#Infer_MP_Allop_Test().test(3)