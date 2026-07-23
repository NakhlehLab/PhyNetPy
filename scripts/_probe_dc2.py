import sys, os, time, copy, math, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from phynetpy.Network import Network, Node, Edge
from phynetpy.ModelGraph import Model
from phynetpy.ModelMove import SPR


def three_taxon():
    labels = ["R", "I", "A", "B", "C"]
    nodes = {l: Node(l) for l in labels}
    net = Network()
    net.add_nodes(*nodes.values())
    net.add_edges([
        Edge(nodes["R"], nodes["I"], length=1.0),
        Edge(nodes["R"], nodes["C"], length=1.0),
        Edge(nodes["I"], nodes["A"], length=1.0),
        Edge(nodes["I"], nodes["B"], length=1.0),
    ])
    return net


def dc_time(net, N=200):
    t0 = time.perf_counter()
    for _ in range(N):
        copy.deepcopy(net)
    return 1000 * (time.perf_counter() - t0) / N


def count_objects(net):
    import gc
    from collections import Counter
    seen = set()
    stack = [net]
    typecount = Counter()
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        typecount[type(o).__name__] += 1
        try:
            refs = gc.get_referents(o)
        except Exception:
            refs = []
        for r in refs:
            if id(r) not in seen and not isinstance(r, (str, int, float, bool, type(None), type)):
                stack.append(r)
    return len(seen), typecount


def main():
    model = Model(rng=np.random.default_rng(2024))
    model.network = three_taxon()
    model.update_network()
    ar = np.random.default_rng(17)

    def tgt(n):
        return -sum(float(e.get_length() or 0.0) for e in n.E())

    n0, tc0 = count_objects(model.network)
    print(f"iter    0: deepcopy={dc_time(model.network):.3f} ms  objs={n0}")
    cur = tgt(model.network)
    for it in range(1, 801):
        mv = SPR(); mv.execute(model); pr = tgt(model.network)
        la = (pr - cur) + mv.log_hastings_ratio()
        if la >= 0.0 or math.log(ar.random()) < la:
            cur = pr
        else:
            mv.undo(model)
        if it in (100, 800):
            n, tc = count_objects(model.network)
            print(f"iter {it:4d}: deepcopy={dc_time(model.network):.3f} ms  objs={n}  nodes={len(list(model.network.V()))}")
            print("   top types:", tc.most_common(8))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
