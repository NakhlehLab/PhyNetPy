"""Can the SEQ chain DISCOVER the reticulation from a plain tree start?

Uses a data-rich regime (many loci) where the reticulation is strongly
supported, and starts the chain from the default UPGMA *tree* (0 reticulations).
If it climbs to a reticulation, RJMCMC mixing is adequate; if it never does
despite strong signal, we have a genuine add-move barrier to fix.

Also prints, for reference, the log posterior the chain reaches when *started*
from the true network -- an upper-ish bound on what good mixing should attain.
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus

true_net = build_true_network()

def num_retic(net):
    return sum(1 for v in net.V() if v.is_reticulation())

N_LOCI, SITES, SEED = 60, 500, 7
data = simulate_multilocus(true_net, MAPPING, n_loci=N_LOCI, seq_length=SITES,
                           theta=0.02, model=JC69(), seed=SEED)

# Reference: start from truth.
kw = data.to_mcmc_seq_kwargs()
kw_true = dict(kw); kw_true["species_net"] = copy.deepcopy(true_net)
ref = MCMC_SEQ(**kw_true, priors=MCMCSeqPriors(max_reticulations=2))
r_ref = ref.search(num_iter=12000, burn_in=3000, sample_freq=50, seed=1)
print(f"[start=TRUE] MAP logP={r_ref.map_log_posterior:.2f} retic(MAP)={num_retic(r_ref.map_network)}")

# Real test: start from tree, must discover reticulation.
disc = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=MCMCSeqPriors(max_reticulations=2))
r_disc = disc.search(num_iter=40000, burn_in=10000, sample_freq=50, seed=1)
retic_frac = sum(1 for s in r_disc.samples if s.num_reticulations >= 1) / max(1, len(r_disc.samples))
print(f"[start=tree] MAP logP={r_disc.map_log_posterior:.2f} retic(MAP)={num_retic(r_disc.map_network)} "
      f"frac_retic_samples={retic_frac:.2f}")
print("retic posterior:", r_disc.reticulation_posterior())
print(f"gap true-start minus tree-start = {r_ref.map_log_posterior - r_disc.map_log_posterior:.2f}")
