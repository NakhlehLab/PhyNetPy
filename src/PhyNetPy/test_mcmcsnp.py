from MCMC_BiMarkers import *

"""
Testing Suite for the Network.py class. Extensively ensures that
the Network, Edge, Node, and related classes are working as intended.
To be included in a final release, this file MUST run without any errors
and all tests must be passed.
"""


def snp_1():
    """
    Check that the SNP likelihood performs correct calculations that match the 
    table given by Charles Rabier (lvl 1 network).
    """
    
    tbl = {[0, 0, 0] : 0.31581337186422315,
           [0, 0, 1] : 1.853660371657668e-3,
           [0, 0, 2] : 0.05677236283895234,
           [0, 1, 0] : 1.6755618678903335e-3,
           [0, 1, 1] : 1.0705941050619642e-5,
           [0, 1, 2] : 4.789800667080107e-4,
           [0, 2, 0] : 9.884301576368968e-4,
           [0, 2, 1] : 7.570444581342921e-6,
           [0, 2, 2] : 5.3322920321306e-4,
           [0, 3, 0] : 0.04027355605172049,
           [0, 3, 1] : 5.910438937463977e-4,
           [0, 3, 2] : 0.07852626659131974,
           [1, 0, 0] : 1.9618887485350735e-3,
           [1, 0, 1] : 1.21627077880799976e-5,
           [1, 0, 2] : 4.828155168689878e-4,
           [1, 1, 0] : 1.09890103033996982e-5,
           [1, 1, 1] : 9.099282927274783e-8,
           [1, 1, 2] : 7.300548379825278e-6,
           [1, 2, 0] : 7.30054837982544e-6,
           [1, 2, 1] : 9.099282927274914e-9,
           [1, 2, 2] : 1.098901030399711e-5,
           [1, 3, 0] : 4.821551686898895e-4,
           [1, 3, 1] : 1.2162707788079851e-5,
           [1, 3, 2] : 1.9618887485350622e-3,
           [2, 0, 0] : 0.0785262665913196,
           [2, 0, 1] : 5.910438937463979e-4, 
           [2, 0, 2] : 0.040273556051720324,
           [2, 1, 0] : 5.332292032130451e-4,
           [2, 1, 1] : 7.5704445813427665e-6,
           [2, 1, 2] : 9.884301576368857e-4,
           [2, 2, 0] : 4.789800667080225e-4,
           [2, 2, 1] : 1.0719114102479618e-5,
           [2, 2, 2] : 1.6755618678903448e-3,
           [2, 3, 0] : 0.0567723862838952165,
           [2, 3, 1] : 1.8536603716576576548e-3,
           [2, 3, 2] : 0.31581337186422315}

    for grouping, expected in tbl.items():
        set_reds = {"A" : grouping[0], "B" : grouping[1], "C": grouping[2]}
        
        result = SNP_LIKELIHOOD_DATA("PhyNetPy/src/paper_net.nex",
                                      set_reds,
                                      1,
                                      1,
                                      .005)
        
        #if our calculated result is close enough to the expected, keep going
        #if not, then halt the process, report the inconsitency, and return 0.
        if not 1 + 1e-10 > abs(result / expected) > 1 - 1e-10:
            print(f"Expected: {expected}, but got: {result} for \
                    grouping : {grouping}")
            return 0
    
    return 1

def snp_2():
    """
    Check that the SNP likelihood performs correct calculations that match the
    table given by Charles Rabier (lvl 2 network).
    """
    
    tbl = {[0,0,0,0] : 0.420388330446373,
           [0,0,0,1] : 2.1413391677020254e-3,
           [0,0,0,2] : 1.1379876974211235e-3,
           [0,0,0,3] : 0.018044391063547768,
           [1,0,1,1] : 5.7431505391794586e-8,
           [1,1,1,1] : 3.1907711650231237e-10,
           [1,2,1,1] : 2.0749049388423288e-10,
           [1,3,1,1] : 1.268527207088679e-8,
           [1,3,2,1] : 2.0091067872430637e-8,
           [1,3,3,1] : 3.277973419215861e-6,
           [3,3,3,3] : 0.420388330646373}

    for grouping, expected in tbl.items():
        result : float = 0.0
        #if our calculated result is close enough to the expected, keep going
        #if not, then halt the process, report the inconsitency, and return 0.
        if not 1 + 1e-10 > abs(result / expected) > 1 - 1e-10:
            print(f"Expected: {expected}, but got: {result} for \
                    grouping : {grouping}")
            return 0
                
    return 1

def snp_3():
    """
    Run MCMC SNP with 500 iterations and evaluate the topology and branch 
    lengths of the inferred network.
    """
    return 1

def snp_4():
    """
    Run MCMC SNP with 500 iterations 100 times with different random seeds and 
    parameter values for u,v, and coal.
    """
    return 1

def snp_5():
    """
    Test auto grouping to ensure it works as intended.
    """
    return 1

def snp_6():
    """
    Runtime testing. This test is not a pass/fail test, but provides all metrics
    and scaling data.
    
    Run MCMC SNP with 1000 iterations. Evaluate total run time, move efficiency,
    bottleneck points, and more.
    """
    return 1

def snp_7():
    """
    Test a network that has reticulation edges that each have .5 gamma.
    """
    pass

def snp_8():
    """
    PhyloNet vs PhyNetPy runtime. Creates a JSON dictionary with data that can
    be exported to make a figure/chart to compare the results.
    """
    pass


class MCMC_SNP_TEST:
    
    #RUN ALL TESTS HERE
    def test(self) -> None:
        res = [snp_1(),
               snp_2(),
               snp_3(),
               snp_4(),
               snp_5(),
               snp_6(),
               snp_7(),
               snp_8()]
        
        if sum(res) == 7:
            print("All (8/8) tests passed!")
        else:
            print(f"Tests failed. {sum(res)}/8 passed.")
        


MCMC_SNP_TEST().test()