import sys, os, math, gc, traceback
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


def path_to(root, target):
    """BFS over referents; return the chain of (type, hint) from root to target."""
    seen = {id(root)}
    parent = {id(root): None}
    stack = [root]
    while stack:
        o = stack.pop()
        if o is target:
            # reconstruct
            chain = []
            cur = o
            while cur is not None:
                chain.append(cur)
                p = parent[id(cur)]
                cur = p
            return list(reversed(chain))
        for r in gc.get_referents(o):
            if isinstance(r, (str, int, float, bool, type(None), type)):
                continue
            if id(r) not in seen:
                seen.add(id(r))
                parent[id(r)] = o
                stack.append(r)
    return None


def main():
    model = Model(rng=np.random.default_rng(2024))
    model.network = three_taxon()
    model.update_network()
    ar = np.random.default_rng(17)

    def tgt(n):
        return -sum(float(e.get_length() or 0.0) for e in n.E())

    cur = tgt(model.network)
    for it in range(1, 401):
        mv = SPR(); mv.execute(model); pr = tgt(model.network)
        la = (pr - cur) + mv.log_hastings_ratio()
        if la >= 0.0 or math.log(ar.random()) < la:
            cur = pr
        else:
            mv.undo(model)

    net = model.network
    live = set(id(n) for n in net.V())
    # find reachable Node objects
    seen = set(); stack = [net]; leaked = []
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, Node) and id(o) not in live:
            leaked.append(o)
        for r in gc.get_referents(o):
            if not isinstance(r, (str, int, float, bool, type(None), type)):
                stack.append(r)
    print(f"live nodes={len(live)}  leaked reachable nodes={len(leaked)}")
    if leaked:
        target = leaked[0]
        print("example leaked node label:", getattr(target, "label", "?"))
        chain = path_to(net, target)
        if chain:
            for i, o in enumerate(chain):
                hint = ""
                if isinstance(o, Node):
                    hint = f" label={getattr(o,'label','?')}"
                elif isinstance(o, Edge):
                    hint = f" {getattr(o.src,'label','?')}->{getattr(o.dest,'label','?')}"
                elif isinstance(o, dict):
                    hint = f" keys~{list(o.keys())[:3]}"
                print(f"  [{i}] {type(o).__module__}.{type(o).__name__}{hint}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
