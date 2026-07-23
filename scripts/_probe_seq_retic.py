"""Is the true reticulation a stable high-posterior mode for MCMC_SEQ?

Runs two SEQ chains on the same simulated data:
  (A) started from the TRUE 1-reticulation network,
  (B) started from the default UPGMA species tree (0 reticulations).
Reports the MAP posterior + reticulation count of each, and whether chain A
holds onto the reticulation.  If A keeps a high-posterior reticulation but B
never finds it, the problem is proposal/mixing, not scoring.
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, MAPPING, TRUE_GAMMA_MAJOR
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus

true_net = build_true_network()
data = simulate_multilocus(true_net, MAPPING, n_loci=10, seq_length=400,
                           theta=0.02, model=JC69(), seed=12345)

def major_gamma(net):
    gs = []
    for v in net.V():
        if v.is_reticulation():
            for e in net.in_edges(v):
                g = e.get_gamma()
                if g is not None:
                    gs.append(float(g))
    return max(gs) if gs else None

def num_retic(net):
    return sum(1 for v in net.V() if v.is_reticulation())

ITERS, BURN = 20000, 5000

# (A) start from the true network
kw = data.to_mcmc_seq_kwargs()
kw_true = dict(kw); kw_true["species_net"] = copy.deepcopy(true_net)
samplerA = MCMC_SEQ(**kw_true, priors=MCMCSeqPriors(max_reticulations=2))
resA = samplerA.search(num_iter=ITERS, burn_in=BURN, sample_freq=50, seed=1)
retics_A = [s.num_reticulations for s in resA.samples]
frac_retic_A = sum(1 for r in retics_A if r >= 1) / max(1, len(retics_A))
print(f"[A start=TRUE] MAP logP={resA.map_log_posterior:.3f} "
      f"retic(MAP)={num_retic(resA.map_network)} gamma={major_gamma(resA.map_network)} "
      f"frac_samples_with_retic={frac_retic_A:.2f}")

# (B) start from default UPGMA tree
samplerB = MCMC_SEQ(**data.to_mcmc_seq_kwargs(), priors=MCMCSeqPriors(max_reticulations=2))
resB = samplerB.search(num_iter=ITERS, burn_in=BURN, sample_freq=50, seed=1)
retics_B = [s.num_reticulations for s in resB.samples]
frac_retic_B = sum(1 for r in retics_B if r >= 1) / max(1, len(retics_B))
print(f"[B start=tree] MAP logP={resB.map_log_posterior:.3f} "
      f"retic(MAP)={num_retic(resB.map_network)} gamma={major_gamma(resB.map_network)} "
      f"frac_samples_with_retic={frac_retic_B:.2f}")

print(f"true gamma_major={TRUE_GAMMA_MAJOR}")
print(f"posterior gap (A-B) = {resA.map_log_posterior - resB.map_log_posterior:.3f}")
