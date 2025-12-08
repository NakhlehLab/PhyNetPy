from Infer_MP_Allop import *
import cProfile
import time
import PhyloNet
from GraphUtils import *
from Network import *
from matplotlib import pyplot as plt

#Newick string for the true network
true_network = parse_newick_string("(O,((F,(((T,U)UID_7)#UID_3,#UID_3)UID_12)internal1,((B,A)UID_4,(D,C)UID_10)UID_9)UID_11)root;")


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


def phynetpy_demo():
    """
    Check that our 10 gene tree example infers the correct network.
    """ 
    runtimes = []
    phynetpy_networks = []
    for _ in range(10):
        start_time = time.time()
        res = INFER_MP_ALLOP(
                        '/Users/mak17/Documents/PhyNetPy/src/J_pruned_v2.nex',
                        {'U': ['01uA', '01uB'], 'T': ['01tA', '01tB'], 
                        'B': ['01bA'], 'F': ['01fA'], 'C': ['01cA'], 'A': ['01aA'],
                        'D': ['01dA'], 'O': ['01oA']})
        end_time = time.time()
        runtimes.append(end_time - start_time)
        
        net_min : Network = _minmaxkey(res, mini = False)
        phynetpy_networks.append(net_min)
        
    return runtimes, [net.nakhleh_distance(true_network) for net in phynetpy_networks]
        


def phylonet_demo():
    """
    1. Run PhyloNet's MP-Allop on the 10 gene tree example. Do this 10 times and record the average, min, max, and std deviation of the runtime.
    2. Collect runtime data for the call to PhyloNet's MP-Allop.
    3. Run PhyloNet's luay distance on each of the 10 inferred networks and compare to the true network.
    4. Record the average, min, max, and std deviation of the luay distance.
    5. Return the statistics for both runtime and topological accuracy
    
    Args:
        N/A
    Returns:
        tuple[list[float], list[float]]: The statistics for both runtime and topological accuracy.
    
    """
    def parse_inferred_network(output : list[str]) -> Network:
        """
        Parse the inferred network from the output of PhyloNet's MP-Allop.
        """
        breaker = False
        for line in output:
            if breaker:
                print(line.strip())
                clean_newick = extract_topology(line.strip())
                return parse_newick_string(clean_newick)
            if line.strip().startswith('Inferred Network #1:'):
                breaker = True
        raise Exception(f"PhyloNet out: {output}")
    
    phylonet_networks = []
    runtimes = []
    
    #Run PhyloNet's MP-Allop 10 times
    for _ in range(10):
        #Run PhyloNet's MP-Allop
        start_time = time.time()
        phylonet_output = PhyloNet.run("PhyNetPy/src/PhyNetPy/phylonet_demo.nex")
        end_time = time.time()
        
        phylonet_networks.append(parse_inferred_network(phylonet_output))
        runtimes.append(end_time - start_time)
    
    luay_distances = [net.nakhleh_distance(true_network) for net in phylonet_networks]
    return runtimes, luay_distances
        
    
    
def compare_demo(phynet_only = True):
    
    #Run the PhyNetPy demo
    tPhyNet, nPhyNet = phynetpy_demo()
    
    # Plot 1: PhyNetPy Runtime
    fig = plt.figure(figsize=(5, 7))
    ax1 = fig.add_axes([0.15, 0.1, 0.8, 0.8])  # Adjusted margins for labels
    bp = ax1.boxplot(tPhyNet)
    
    # Add title and axis labels
    plt.title("PhyNetPy Runtime Performance", fontsize=14, fontweight='bold')
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    
    # Set x-tick label
    plt.xticks(ticks=[1], labels=["PhyNetPy"])
    
    # Set y-axis ticks and limits
    plt.yticks(ticks=[0, 2, 4, 6, 8, 10])
    plt.ylim(-1, 11)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    plt.show()
    
    # Plot 2: PhyNetPy Accuracy
    fig2 = plt.figure(figsize=(5, 7))
    ax2 = fig2.add_axes([0.15, 0.1, 0.8, 0.8])  # Adjusted margins
    bp = ax2.boxplot(nPhyNet)
    
    # Add title and axis labels
    plt.title("PhyNetPy Network Inference Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Nakhleh Distance from True Network", fontsize=12)
    
    # Set x-tick label
    plt.xticks(ticks=[1], labels=["PhyNetPy"])
    
    # Set y-axis ticks and limits
    plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6])
    plt.ylim(-0.5, 6.5)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    plt.show()
    
    if not phynet_only:
        tPhylo, nPhylo = phylonet_demo()
    
        # Plot 3: PhyloNet Runtime
        fig3 = plt.figure(figsize=(5, 7))
        ax3 = fig3.add_axes([0.15, 0.1, 0.8, 0.8])
        bp = ax3.boxplot(tPhylo)
        
        plt.title("PhyloNet Runtime Performance", fontsize=14, fontweight='bold')
        plt.xlabel("Algorithm", fontsize=12)
        plt.ylabel("Time (seconds)", fontsize=12)
        
        plt.xticks(ticks=[1], labels=["PhyloNet"])
        plt.yticks(ticks=[0, 100, 200, 300, 400, 500])
        plt.ylim(0, 550)
        
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        
        # Plot 4: Comparison of Accuracy
        fig4 = plt.figure(figsize=(10, 7))
        ax4 = fig4.add_axes([0.1, 0.1, 0.85, 0.8])
        
        # Plot both PhyloNet and PhyNetPy for comparison
        bp2 = ax4.boxplot([nPhylo, nPhyNet], widths=0.6)
        
        plt.title("Network Inference Accuracy Comparison", fontsize=14, fontweight='bold')
        plt.xlabel("Algorithm", fontsize=12)
        plt.ylabel("Nakhleh Distance from True Network", fontsize=12)
        
        plt.xticks(ticks=[1, 2], labels=["PhyloNet", "PhyNetPy"])
        plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6])
        plt.ylim(-0.5, 6.5)
        
        plt.grid(axis='y', alpha=0.3)
        plt.show()
    
compare_demo(False)

