"""Minimal single-reticulation normalization test.

Network (3 taxa, 1 reticulation):

        R
       / \
      P   Q
     /|   |\
    A H   H B      (H is shared: P->H gamma, Q->H 1-gamma)
      |
      C

Sum over all 2^3 = 8 site patterns of P(pattern | net) must be 1 for any
gamma, branch lengths, and coal.
"""
import sys, os, itertools, math, traceback
sys.path.insert(0, os.path.dirname(__file__))


def build_min_net(gamma, b=0.05):
    from phynetpy.Network import Network, Node, Edge
    net = Network()
    R, P, Q, H = Node("R"), Node("P"), Node("Q"), Node("H", is_reticulation=True)
    A, B, C = Node("A"), Node("B"), Node("C")
    net.add_nodes(R, P, Q, H, A, B, C)
    net.add_edges([
        Edge(R, P, length=b), Edge(R, Q, length=b),
        Edge(P, A, length=2 * b), Edge(Q, B, length=2 * b),
        Edge(P, H, length=b, gamma=gamma),
        Edge(Q, H, length=b, gamma=1.0 - gamma),
        Edge(H, C, length=b),
    ])
    return net


def write_pattern_nexus(path, taxa, pat, newick):
    with open(path, "w", encoding="utf-8") as f:
        f.write("#NEXUS\n\nBEGIN TAXA;\n")
        f.write(f"DIMENSIONS NTAX={len(taxa)};\n")
        f.write(f"TAXLABELS {' '.join(taxa)};\nEND;\n\n")
        f.write("BEGIN DATA;\n  Dimensions nchar=1;\n")
        f.write("  Format datatype=snp missing=? gap=- matchchar=.;\n  Matrix\n")
        for i, t in enumerate(taxa):
            f.write(f"    {t} {pat[i]}\n")
        f.write("  ;\nEND;\n\nBEGIN TREES;\n")
        f.write(f"Tree net = {newick}\nEND;\n")


def norm(net, taxa, u, v, coal, samples=None):
    from phynetpy.BiMarkers import _snp_log_likelihood
    from phynetpy.SNPSimulator import _network_to_rich_newick
    from phynetpy.MSA import MSA
    newick = _network_to_rich_newick(net)
    if samples is None:
        samples = {t: 1 for t in taxa}
    total = 0.0
    os.makedirs("runs", exist_ok=True)
    # enumerate every red-count pattern: taxon t contributes 0..samples[t]
    ranges = [range(samples[t] + 1) for t in taxa]
    for pat in itertools.product(*ranges):
        path = os.path.join("runs", "min.nex")
        write_pattern_nexus(path, taxa, pat, newick)
        aln = MSA(path)
        total += math.exp(_snp_log_likelihood(net, aln, u, v, coal, samples))
    return total


def net4_hcherry(gamma, b=0.05):
    """R->P1,P2 ; P1->A,H ; P2->B,H ; H->CD ; CD->C,D  (H feeds a cherry)."""
    from phynetpy.Network import Network, Node, Edge
    net = Network()
    R, P1, P2, H = Node("R"), Node("P1"), Node("P2"), Node("H", is_reticulation=True)
    CD = Node("CD")
    A, B, C, D = Node("A"), Node("B"), Node("C"), Node("D")
    net.add_nodes(R, P1, P2, H, CD, A, B, C, D)
    net.add_edges([
        Edge(R, P1, length=b), Edge(R, P2, length=b),
        Edge(P1, A, length=3 * b), Edge(P2, B, length=3 * b),
        Edge(P1, H, length=b, gamma=gamma),
        Edge(P2, H, length=b, gamma=1.0 - gamma),
        Edge(H, CD, length=b), Edge(CD, C, length=b), Edge(CD, D, length=b),
    ])
    return net


def net5_pcherry(gamma, b=0.05):
    """R->P1,P2 ; P1->AB,H ; P2->E,H ; AB->A,B ; H->CD ; CD->C,D.

    Both a parent (P1) and the hybrid feed cherries -- closest small analogue
    of the 6-taxon truth.
    """
    from phynetpy.Network import Network, Node, Edge
    net = Network()
    R, P1, P2 = Node("R"), Node("P1"), Node("P2")
    H = Node("H", is_reticulation=True)
    AB, CD = Node("AB"), Node("CD")
    A, B, C, D, E = (Node("A"), Node("B"), Node("C"), Node("D"), Node("E"))
    net.add_nodes(R, P1, P2, H, AB, CD, A, B, C, D, E)
    net.add_edges([
        Edge(R, P1, length=b), Edge(R, P2, length=b),
        Edge(P1, AB, length=2 * b), Edge(AB, A, length=b), Edge(AB, B, length=b),
        Edge(P2, E, length=4 * b),
        Edge(P1, H, length=b, gamma=gamma),
        Edge(P2, H, length=b, gamma=1.0 - gamma),
        Edge(H, CD, length=b), Edge(CD, C, length=b), Edge(CD, D, length=b),
    ])
    return net


def main():
    cases = [
        ("min3   ", ["A", "B", "C"], build_min_net),
        ("net4hc ", ["A", "B", "C", "D"], net4_hcherry),
        ("net5pc ", ["A", "B", "C", "D", "E"], net5_pcherry),
    ]
    print("### symmetric u=v=1 ###")
    for name, taxa, builder in cases:
        print(f"\n[{name.strip()}]  taxa={taxa}")
        for coal in (0.01, 0.1, 1.0, 2.0):
            row = []
            for gamma in (0.0, 0.5, 1.0):
                net = builder(gamma)
                s = norm(net, taxa, 1.0, 1.0, coal)
                row.append(f"g={gamma}:{s:.5f}")
            print(f"  coal={coal:<5}  " + "  ".join(row))

    print("\n### asymmetric u=0.7 v=1.6 ###")
    for name, taxa, builder in cases:
        print(f"\n[{name.strip()}]  taxa={taxa}")
        for coal in (0.1, 1.0):
            row = []
            for gamma in (0.0, 0.3, 0.5, 0.75, 1.0):
                net = builder(gamma)
                s = norm(net, taxa, 0.7, 1.6, coal)
                row.append(f"g={gamma}:{s:.5f}")
            print(f"  coal={coal:<5}  " + "  ".join(row))

    print("\n### multi-sample (2 copies for C) on net4hc, u=v=1 ###")
    taxa = ["A", "B", "C", "D"]
    samples = {"A": 1, "B": 1, "C": 2, "D": 1}
    for coal in (0.1, 1.0):
        row = []
        for gamma in (0.0, 0.5, 1.0):
            net = net4_hcherry(gamma)
            s = norm(net, taxa, 1.0, 1.0, coal, samples=samples)
            row.append(f"g={gamma}:{s:.5f}")
        print(f"  coal={coal:<5}  " + "  ".join(row))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
