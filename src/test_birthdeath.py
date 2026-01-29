import pytest
from PhyNetPy.BirthDeath import *


################
### HELPERS ####
################

def nonsense_cbdp(birthrate : float, deathrate : float, n : int, s : float)-> bool:
    if birthrate <= deathrate:
        #Trees will die off if death rate is too high.
        return True
    if birthrate <=0 or deathrate < 0 or n < 2 or s <= 0 or s > 1:
        #Bounds on numerical inputs
        return True
    return False
    
################
#### TESTS #####
################

def test_yule():
    """
    Test the Yule model with various birth rates and time constraints.
    
    This test checks:
    - Invalid birth rates and taxa counts raise errors
    - Time conditioning with invalid values raises errors
    - Time conditioning with large expected tip counts raises errors
    - Time conditioning with valid values generates trees
    - Number of tips generated matches the expected number of tips
    """
    n_s = [-3, 0, 1, 2, 3, 5, 10]
    t_s = [-10, 0, 1, 10, 100]
    birth_rates = [-1, 0, .01, .1, 1, 10]
    
    for birth_rate in birth_rates:
        for n in n_s:
            if n < 2 or birth_rate <= 0:
                with pytest.raises(BirthDeathSimulationError):
                    yule_tree = Yule(birth_rate, n).generate_network()
            else:
                yule_tree = Yule(birth_rate, n)
                assert len(yule_tree.generate_network().get_leaves()) == n
        for t in t_s:
            
            if t <= 0 or birth_rate <= 0:
                with pytest.raises(BirthDeathSimulationError):
                    yule_tree = Yule(birth_rate, time=t).generate_network()
            else:
                est_tips = estimate_expected_tips(birth_rate, t)
        
                if est_tips > TIP_ERROR_THRESHOLD:
                    with pytest.raises(BirthDeathSimulationError):
                        yule_tree = Yule(birth_rate, time=t).generate_network()
                else:
                    yule_tree = Yule(birth_rate, time=t).generate_network()
                    for leaf in yule_tree.get_leaves():
                        assert leaf.get_time() == t
                
def test_cbdp():
    """
    Test the CBDP model with various birth rates and time constraints.
    
    This test checks:
    - Invalid birth/death/sampling rates and taxa counts raise errors
    - Trees end up with a live lineage count (NOT the same as number of leaves in the network) equal to the goal taxa 
    """
    
    b = [-1, 0, 0.01, 0.1, 1, 10]
    mu = [-1 ,0, 0.01, 0.1, 1, 10]
    n = [-3, 0, 1, 2, 3, 5, 10]
    sample = [-.1, 0, 0.1, 0.5, 1, 4]
    
    for brate in b:
        for drate in mu:
            for taxa_ct in n:
                for srate in sample:
                    if nonsense_cbdp(brate, drate, taxa_ct, srate):
                        with pytest.raises(BirthDeathSimulationError):
                            cbdp = CBDP(brate, drate, taxa_ct, srate).generate_network()
                            
                    else:
                        cbdp = CBDP(brate, drate, taxa_ct, srate)
                        assert len(live_species(cbdp.generate_network().V())) == taxa_ct


def test_generation_and_clearing():
    """
    Tests that generating a bulk amount of networks and accessing/clearing them 
    works as expected.
    
    Tests both Yule and CBDP.
    """
    #Test Yule
    yuleprocess = Yule(.5, 10)
    yuleprocess.generate_networks(10)
    assert(len(yuleprocess.generated_networks) == 10)
    yuleprocess.clear_generated()
    assert(len(yuleprocess.generated_networks) == 0)
    
    #Test CBDP
    cbdpprocess = CBDP(.5, .05, 10)
    cbdpprocess.generate_networks(10)
    assert(len(cbdpprocess.generated_networks) == 10)
    cbdpprocess.clear_generated()
    assert(len(cbdpprocess.generated_networks) == 0)
    
#Check non none rng
def test_rng_consistency():
    """
    Tests that with a specific seed, you get identical results with both Yule 
    CBDP process with the same parameters.
    """
    seed = 1
    yp = Yule(.5, 10, rng = np.random.default_rng(seed))
    net1 = yp.generate_network()
    yp2 = Yule(.5, 10, rng = np.random.default_rng(seed))
    net2 = yp2.generate_network()
    #TODO: get an isomorphism checker if one doesn't already exist in network module
    assert(len(net1.V()) == len(net2.V()))
    assert(len(net1.E()) == len(net2.E()))

#Check None time and n
def test_lack_of_input():
    with pytest.raises(BirthDeathSimulationError):
        yp = Yule(.5, None, None)
    
    

