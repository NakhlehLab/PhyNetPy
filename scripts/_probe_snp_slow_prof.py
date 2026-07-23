"""Capture the first slow level-2 SNP network, then profile it in isolation."""
import os, sys, time, cProfile, pstats, io, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phynetpy.BiMarkers as B
from scripts.mcmc_harness import build_true_network, TAXA
from phynetpy.SNPSimulator import simulate
from phynetpy.infer import MCMC_BIMARKERS

captured = {}
_orig = B._snp_log_likelihood
def _timed(net, aln, u, v, coal, samples, **kw):
    if captured:
        return -1000.0  # short-circuit once captured so MCMC finishes fast
    t0 = time.perf_counter()
    r = _orig(net, aln, u, v, coal, samples, **kw)
    dt = time.perf_counter() - t0
    if dt > 0.2:
        captured["net"] = copy.deepcopy(net)
        captured["aln"] = aln
        captured["samples"] = samples
        captured["dt"] = dt
    return r
B._snp_log_likelihood = _timed

true_net = build_true_network()
samples = {leaf.label: 1 for leaf in true_net.get_leaves()}
sim = simulate(n=len(TAXA), s=300, net=true_net, samples=samples,
               u=1.0, v=1.0, coal=0.005, seed=12345)
os.makedirs("runs", exist_ok=True)
path = os.path.join("runs", "probe_slow.nex")
sim.write_nexus(path)
MCMC_BIMARKERS(path, u=1.0, v=1.0, coal=0.005, num_iter=4000, burn_in=300,
               sample_freq=20, seed=12345, samples=samples, max_reticulations=2)

B._snp_log_likelihood = _orig
if "net" not in captured:
    print("no slow state captured"); sys.exit(0)

net = captured["net"]; aln = captured["aln"]; samples = captured["samples"]
m = B._build_snp_model(net, aln)
print(f"captured dt={1000*captured['dt']:.0f}ms level={B._compute_network_level(m)} "
      f"max_n={B._compute_max_lineages(m, samples)}")
print("NEWICK:", net.newick())
pr = cProfile.Profile(); pr.enable()
for _ in range(3):
    _orig(net, aln, 1.0, 1.0, 0.005, samples, verbose=False)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20)
print(s.getvalue())
