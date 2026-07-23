import sys, os, time, copy, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from phynetpy.Network import Network, Node, Edge
from phynetpy.ModelGraph import Model


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


def bench(label, fn, N=2000):
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    dt = time.perf_counter() - t0
    print(f"{label}: {1000*dt/N:.4f} ms/call ({N} calls, {dt:.2f}s)")


def main():
    net = three_taxon()
    model = Model(rng=np.random.default_rng(0))
    model.network = net
    model.update_network()

    bench("deepcopy(net)", lambda: copy.deepcopy(net))
    bench("net.E()", lambda: net.E())
    bench("net.V()", lambda: net.V())
    bench("model.update_network()", lambda: model.update_network())
    bench("net.newick()", lambda: net.newick())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
