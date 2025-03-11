"""
This file contains testing modules for each class that is being packaged into
any release. In order for a release to be made, EVERY test in this file MUST be 
passed.

Modules:

- DataStructures (Matrix, MSA, DataSequence, Alphabet etc)
- Biology (IUPAC)
- Simulations
- Network
- NetworkOps (Network moves + methods)
- Parsimony (Infer_MP_Allop)
- MCMC (MCMCBiMarkers, MCMCSeq, etc)
- Search (HillClimb, SimAnnealing , MH, ProposalKernel)
- Modeling (Model, ModelFactory, GTR)
- IO (Parser, Newick, Nexus, etc)
- DevTools (Visualization tools, diagnostic tools, logger)
"""


import Biology


