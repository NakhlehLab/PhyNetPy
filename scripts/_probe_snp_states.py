"""Instrument SNP scorer: record per-call time, level, max_n during MCMC."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phynetpy.BiMarkers as B
from scripts.mcmc_harness import build_true_network, TAXA
from phynetpy.SNPSimulator import simulate
from phynetpy.infer import MCMC_BIMARKERS

records = []
_orig = B._snp_log_likelihood
def _timed(net, aln, u, v, coal, samples, **kw):
    t0 = time.perf_counter()
    try:
        return _orig(net, aln, u, v, coal, samples, **kw)
    finally:
        dt = time.perf_counter() - t0
        try:
            m = B._build_snp_model(net, aln)
            mx = B._compute_max_lineages(m, samples)
            lvl = B._compute_network_level(m)
        except Exception:
            mx, lvl = -1, -1
        records.append((dt, lvl, mx))
        if dt > 0.2:
            try:
                nwk = net.newick()
            except Exception:
                nwk = "?"
            print(f"  SLOW {1000*dt:.0f}ms level={lvl} max_n={mx} "
                  f"state_dim={B.state_dim(mx) if mx>0 else 0}", flush=True)
            print(f"       {nwk}", flush=True)
B._snp_log_likelihood = _timed

true_net = build_true_network()
samples = {leaf.label: 1 for leaf in true_net.get_leaves()}
sim = simulate(n=len(TAXA), s=300, net=true_net, samples=samples,
               u=1.0, v=1.0, coal=0.005, seed=12345)
os.makedirs("runs", exist_ok=True)
path = os.path.join("runs", "probe_states.nex")
sim.write_nexus(path)

t0 = time.perf_counter()
MCMC_BIMARKERS(path, u=1.0, v=1.0, coal=0.005, num_iter=3000, burn_in=500,
               sample_freq=20, seed=12345, samples=samples, max_reticulations=2)
dt = time.perf_counter() - t0

records.sort(reverse=True)
print(f"total {dt:.1f}s over {len(records)} scorer calls")
print("slowest 10 calls (ms, level, max_lineages):")
for r in records[:10]:
    print(f"  {1000*r[0]:8.1f} ms  level={r[1]}  max_n={r[2]}  state_dim={B.state_dim(r[2]) if r[2]>0 else 0}")
import statistics
print(f"median ms/call: {1000*statistics.median([r[0] for r in records]):.2f}")
# histogram by level
from collections import Counter
by_level = {}
for dt_, lvl, mx in records:
    by_level.setdefault(lvl, []).append(dt_)
for lvl in sorted(by_level):
    ts = by_level[lvl]
    print(f"  level {lvl}: {len(ts)} calls, total {sum(ts):.1f}s, "
          f"mean {1000*sum(ts)/len(ts):.1f}ms, max {1000*max(ts):.1f}ms")
