"""
This file is a general template for developing a phylogenetic network method using
PhyNetPy's ModelFactory and ModelGraph framework.

Author-- Mark Kessler
Date-- 10/23/23
Version Info-- 0.1.0
Approved to Release Date : N/A
"""

## STEP 1 : Create a specialty network node ##
"""
Develop a Network Node class (subclass of some abstract network node for developers? idk)

1) Calc needs to simply call its likelihood function on a list of child.get() calls
    If this is all that needs to be passed as args to the likelihood function, nothing needs to be changed and you can simply
    use the default SampleNetworkNode class.
    
    Otherwise, create a new class, MyNetworkNode(SampleNetworkNode) that reimplements the calc method, while passing additional objects as required

"""






