"""Locate the SNP per-call blowup as reticulations/lineages grow."""
import os, sys, time, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phynetpy.SNPSimulator import simulate, random_network
from phynetpy.BiMarkers import (
    _snp_log_likelihood, _build_snp_model, _compute_max_lineages,
    _compute_network_level, state_dim,
)
from phynetpy.MSA import MSA

for level in [0, 1, 2]:
    for n_taxa in [6]:
        net = random_network(n=n_taxa, level=level, seed=7 + level)
        samples = {leaf.label: 1 for leaf in net.get_leaves()}
        sim = simulate(n=n_taxa, s=200, net=net, samples=samples,
                       u=1.0, v=1.0, coal=0.005, seed=1)
        path = os.path.join("runs", f"probe_snp_l{level}.nex")
        os.makedirs("runs", exist_ok=True)
        sim.write_nexus(path)
        aln = MSA(path)
        m = _build_snp_model(net, aln)
        max_n = _compute_max_lineages(m, samples)
        lvl = _compute_network_level(m)
        # warm
        _snp_log_likelihood(net, aln, 1.0, 1.0, 0.005, samples, verbose=False)
        N = 5
        t0 = time.perf_counter()
        for _ in range(N):
            _snp_log_likelihood(net, aln, 1.0, 1.0, 0.005, samples, verbose=False)
        dt = time.perf_counter() - t0
        print(f"level={lvl} taxa={n_taxa} max_lineages={max_n} "
              f"state_dim={state_dim(max_n)} : {1000*dt/N:.1f} ms/call")

# Profile the level-2 case
net = random_network(n=6, level=2, seed=9)
samples = {leaf.label: 1 for leaf in net.get_leaves()}
sim = simulate(n=6, s=200, net=net, samples=samples, u=1.0, v=1.0, coal=0.005, seed=1)
path = os.path.join("runs", "probe_snp_l2b.nex")
sim.write_nexus(path)
aln = MSA(path)
_snp_log_likelihood(net, aln, 1.0, 1.0, 0.005, samples, verbose=False)
pr = cProfile.Profile(); pr.enable()
for _ in range(5):
    _snp_log_likelihood(net, aln, 1.0, 1.0, 0.005, samples, verbose=False)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15)
print(s.getvalue())
