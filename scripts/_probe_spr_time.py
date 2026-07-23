import sys, os, time, traceback
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


def cherry(net):
    for v in net.V():
        leaf_kids = [k for k in net.get_children(v) if not net.get_children(k)]
        if len(leaf_kids) == 2:
            return frozenset(k.label for k in leaf_kids)
    return frozenset()


def main():
    model = Model(rng=np.random.default_rng(2024))
    model.network = three_taxon()
    model.update_network()
    accept_rng = np.random.default_rng(17)

    def log_target(net):
        return -sum(float(e.get_length() or 0.0) for e in net.E())

    cur = log_target(model.network)
    N = 5000
    t0 = time.perf_counter()
    nnodes = []
    for it in range(N):
        move = SPR()
        move.execute(model)
        prop = log_target(model.network)
        la = (prop - cur) + move.log_hastings_ratio()
        import math
        if la >= 0.0 or math.log(accept_rng.random()) < la:
            cur = prop
        else:
            move.undo(model)
        if it % 1000 == 0:
            nnodes.append(len(list(model.network.V())))
    dt = time.perf_counter() - t0
    print(f"{N} iters in {dt:.2f}s = {1000*dt/N:.3f} ms/it")
    print("node counts sampled:", nnodes)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
