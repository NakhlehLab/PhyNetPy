"""Find the thinning that decorrelates SPR topology samples, then confirm the
independent-sample chi-square is well-behaved across many seeds.  If p is
~Uniform[0,1] across seeds (not clustered near 0), the move satisfies detailed
balance and the original failure was autocorrelation in an under-thinned chain.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import chisquare
from phynetpy.Network import Network, Node, Edge
from phynetpy.ModelMove import SPR
from phynetpy.ModelGraph import Model


def start():
    n = {l: Node(l) for l in ["R", "I", "A", "B", "C"]}
    net = Network(); net.add_nodes(*n.values())
    net.add_edges([
        Edge(n["R"], n["I"], length=1.0), Edge(n["R"], n["C"], length=1.0),
        Edge(n["I"], n["A"], length=1.0), Edge(n["I"], n["B"], length=1.0)])
    return net


def cherry_code(net):
    for v in net.V():
        lk = [k for k in net.get_children(v) if not net.get_children(k)]
        if len(lk) == 2:
            return "".join(sorted(k.label for k in lk))
    return "?"


def chain(move_seed, accept_seed, n_iters):
    """Return the full (unthinned) post-burn topology-code stream."""
    model = Model(rng=np.random.default_rng(move_seed))
    model.network = start(); model.update_network()
    arng = np.random.default_rng(accept_seed)
    logt = lambda net: -sum(float(e.get_length() or 0.0) for e in net.E())
    cur = logt(model.network)
    burn = n_iters // 6
    stream = []
    for it in range(n_iters):
        mv = SPR(); mv.execute(model)
        prop = logt(model.network)
        la = (prop - cur) + mv.log_hastings_ratio()
        if la >= 0 or math.log(arng.random()) < la:
            cur = prop
        else:
            mv.undo(model)
        if it >= burn:
            stream.append(cherry_code(model.network))
    return stream


def lag1(codes, target):
    ind = np.array([1.0 if c == target else 0.0 for c in codes])
    if ind.std() == 0:
        return 0.0
    return float(np.corrcoef(ind[:-1], ind[1:])[0, 1])


# 1) autocorrelation vs thinning on one long stream
stream = chain(2024, 17, 300000)
print("lag-1 autocorrelation of AC-indicator vs thinning:")
for thin in [1, 5, 10, 20, 40, 60]:
    print(f"  thin={thin:3d}: ac1={lag1(stream[::thin], 'AC'):.3f}")

# 2) chi-square p across many seeds at a decorrelating thin
thin = 40
print(f"\nindependent-sample chi-square (thin={thin}) across seeds:")
ps = []
for ms in range(10):
    s = chain(1000 + ms, 5000 + ms, 300000)[::thin]
    codes, cnt = np.unique(s, return_counts=True)
    obs = cnt.astype(float)
    p = float(chisquare(obs, np.full(len(obs), obs.sum() / len(obs))).pvalue)
    ps.append(p)
    print(f"  seed={ms}: n={int(obs.sum())} freqs={np.round(obs/obs.sum(),4)} p={p:.4g}")
ps = np.array(ps)
print(f"\nmin p={ps.min():.4g}  median p={np.median(ps):.4g}  "
      f"(H0: p~Uniform[0,1]; a real bias would pin every p near 0)")
