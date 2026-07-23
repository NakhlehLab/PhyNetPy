"""Quick end-to-end probe of the three MCMC paths on tiny settings.

Throwaway triage script: confirms which of MCMC_GT / MCMC_SEQ / MCMC_BIMARKERS
actually run to completion today, and surfaces the first exception if not.
"""
from __future__ import annotations

import time
import traceback

from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.infer import (
    MCMC_SEQ, MCMCSeqPriors, MCMC_GT, MCMC_GTPriors, JC69, simulate_multilocus,
)

TRUE_NETWORK = (
    "((((A:0.04,B:0.04)AB:0.03)#H1:0.02[&gamma=0.65],C:0.09)ABC:0.04,"
    "(#H1:0.04[&gamma=0.35],D:0.11)DR:0.02)R;"
)
MAPPING = {sp: [sp] for sp in ("A", "B", "C", "D")}


def probe_gt():
    net = Network.from_newick(TRUE_NETWORK)
    data = simulate_multilocus(net, MAPPING, n_loci=8, seq_length=200,
                               theta=0.02, model=JC69(), seed=1)
    gts = GeneTrees(gene_tree_list=list(data.gene_trees),
                    species_gene_mapping=MAPPING)
    mcmc = MCMC_GT.from_consensus(gts, MAPPING, priors=MCMC_GTPriors())
    t0 = time.perf_counter()
    res = mcmc.search(method="mh", num_iter=2000, burn_in=500, thin=20,
                      max_reticulations=2, seed=1)
    dt = time.perf_counter() - t0
    print(f"[GT ]  ok  {2000} it in {dt:.2f}s = {dt/2000*1e3:.3f} ms/it  "
          f"logP={res.best_log_posterior:.2f}  acc={res.acceptance_rate:.3f}")


def probe_seq():
    net = Network.from_newick(TRUE_NETWORK)
    data = simulate_multilocus(net, MAPPING, n_loci=6, seq_length=200,
                               theta=0.02, model=JC69(), seed=2)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=2))
    t0 = time.perf_counter()
    res = sampler.search(num_iter=2000, burn_in=500, sample_freq=20, seed=2)
    dt = time.perf_counter() - t0
    print(f"[SEQ]  ok  {2000} it in {dt:.2f}s = {dt/2000*1e3:.3f} ms/it  "
          f"MAPlogP={res.map_log_posterior:.2f}")


def probe_snp():
    import os
    from phynetpy.SNPSimulator import simulate, random_network
    from phynetpy.infer import SNP_LIKELIHOOD
    net = random_network(n=6, level=1, seed=3)
    samples = {leaf.label: 1 for leaf in net.get_leaves()}
    sim = simulate(n=6, s=500, net=net, samples=samples,
                   u=1.0, v=1.0, coal=0.005, seed=3)
    path = os.path.join("runs", "_probe_snp.nex")
    os.makedirs("runs", exist_ok=True)
    sim.write_nexus(path)
    t0 = time.perf_counter()
    ll = SNP_LIKELIHOOD(path, u=1.0, v=1.0, coal=0.005, samples=samples,
                        sequential=True)
    dt = time.perf_counter() - t0
    print(f"[SNP-LL] ok  logL={ll:.3f} in {dt:.2f}s")

    # Now try the actual MCMC entry point.
    from phynetpy.infer import MCMC_BIMARKERS
    t0 = time.perf_counter()
    scores = MCMC_BIMARKERS(path, u=1.0, v=1.0, coal=0.005)
    dt = time.perf_counter() - t0
    print(f"[SNP-MCMC] ok  {len(scores)} nets in {dt:.2f}s")


for name, fn in [("GT", probe_gt), ("SEQ", probe_seq), ("SNP", probe_snp)]:
    try:
        fn()
    except Exception:
        print(f"[{name}] FAILED:")
        traceback.print_exc()
