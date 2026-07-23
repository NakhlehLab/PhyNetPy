"""Time a single SNP likelihood call + profile expm share on the harness net."""
import os, sys, time, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, TAXA
from phynetpy.SNPSimulator import simulate
from phynetpy.BiMarkers import _snp_log_likelihood
from phynetpy.MSA import MSA

true_net = build_true_network()
samples = {leaf.label: 1 for leaf in true_net.get_leaves()}
sim = simulate(n=len(TAXA), s=400, net=true_net, samples=samples,
               u=1.0, v=1.0, coal=0.005, seed=12345)
os.makedirs("runs", exist_ok=True)
path = os.path.join("runs", "probe_snp.nex")
sim.write_nexus(path)
aln = MSA(path)

# warm
ll = _snp_log_likelihood(true_net, aln, 1.0, 1.0, 0.005, samples, verbose=False)
print("log-lik:", ll)

N = 20
t0 = time.perf_counter()
for _ in range(N):
    _snp_log_likelihood(true_net, aln, 1.0, 1.0, 0.005, samples, verbose=False)
dt = time.perf_counter() - t0
print(f"{N} calls in {dt:.3f}s = {1000*dt/N:.2f} ms/call")

pr = cProfile.Profile()
pr.enable()
for _ in range(N):
    _snp_log_likelihood(true_net, aln, 1.0, 1.0, 0.005, samples, verbose=False)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18)
print(s.getvalue())
