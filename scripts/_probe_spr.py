import sys, os, traceback
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
    print("start cherry:", set(cherry(model.network)), "newick:", model.network.newick())
    print("prunable:", [(e.src.label, e.dest.label) for e in SPR._prunable_edges(model.network)])

    seen = {}
    for i in range(20):
        move = SPR()
        move.execute(model)
        c = cherry(model.network)
        hr = move.log_hastings_ratio()
        seen[frozenset(c)] = seen.get(frozenset(c), 0) + 1
        print(f"  iter {i}: cherry={set(c)} loghr={hr:.4f} newick={model.network.newick()}")
        move.undo(model)  # always undo to see raw forward diversity
        print(f"           after undo cherry={set(cherry(model.network))}")
    print("forward-proposal cherries:", {tuple(sorted(k)): v for k, v in seen.items()})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
