"""Identifiability under co-estimation: does higher signal recover the retic?

For increasing sequence length (which sharpens each gene tree and so preserves
the reticulation's coalescent signal through co-estimation), run BOTH a chain
started from the true network and one started from the UPGMA tree, with the
informed kernel.  Report each chain's reticulation posterior.  Agreement =
the sampler is correct; the reticulation should re-emerge once the signal
survives gene-tree averaging.
"""
import os, sys, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus

true_net = build_true_network()

def num_retic(net):
    return sum(1 for v in net.V() if v.is_reticulation())

for sites in [1500]:
    data = simulate_multilocus(true_net, MAPPING, n_loci=30, seq_length=sites,
                               theta=0.02, model=JC69(), seed=7)
    # from truth
    kw = data.to_mcmc_seq_kwargs(); kw["species_net"] = copy.deepcopy(true_net)
    st = MCMC_SEQ(**kw, priors=MCMCSeqPriors(max_reticulations=2))
    rt = st.search(num_iter=15000, burn_in=5000, sample_freq=50, seed=2)
    # from tree
    sf = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=MCMCSeqPriors(max_reticulations=2))
    rf = sf.search(num_iter=15000, burn_in=5000, sample_freq=50, seed=2)
    print(f"sites={sites}:")
    print(f"  from TRUTH: MAP logP={rt.map_log_posterior:.1f} retic(MAP)={num_retic(rt.map_network)} "
          f"post={rt.reticulation_posterior()}")
    print(f"  from TREE : MAP logP={rf.map_log_posterior:.1f} retic(MAP)={num_retic(rf.map_network)} "
          f"post={rf.reticulation_posterior()}")
