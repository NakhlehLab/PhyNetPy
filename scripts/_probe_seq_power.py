"""Does the reticulation become supported with more loci?

For each locus count, start MCMC_SEQ from the TRUE network and report whether
the chain retains the reticulation (fraction of post-burn-in samples with a
reticulation) and the MAP reticulation count.  If more data -> retained
reticulation, the earlier 0-retic result was a statistical power issue, not a
sampler/likelihood bug.
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus

true_net = build_true_network()

def num_retic(net):
    return sum(1 for v in net.V() if v.is_reticulation())

for n_loci in [10, 40, 100]:
    data = simulate_multilocus(true_net, MAPPING, n_loci=n_loci, seq_length=500,
                               theta=0.02, model=JC69(), seed=999)
    kw = data.to_mcmc_seq_kwargs()
    kw["species_net"] = copy.deepcopy(true_net)
    s = MCMC_SEQ(**kw, priors=MCMCSeqPriors(max_reticulations=2))
    res = s.search(num_iter=15000, burn_in=4000, sample_freq=50, seed=1)
    retics = [smp.num_reticulations for smp in res.samples]
    frac = sum(1 for r in retics if r >= 1) / max(1, len(retics))
    print(f"loci={n_loci:3d}: MAP logP={res.map_log_posterior:.2f} "
          f"retic(MAP)={num_retic(res.map_network)} frac_retic={frac:.2f}")
