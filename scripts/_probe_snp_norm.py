"""Rigorous correctness probe for the biallelic-marker likelihood.

A correct SNP site likelihood P(pattern | net) is a probability distribution
over site patterns, so summing exp(loglik) over *all* possible patterns must
equal 1 (no ascertainment) -- or equal the total polymorphic mass if the
likelihood conditions on the marker being variable.  This test is independent
of any simulator, so it isolates likelihood correctness from data-generation
model mismatch.
"""
import sys, os, itertools, math, traceback
sys.path.insert(0, os.path.dirname(__file__))

from mcmc_harness import build_true_network, TAXA


def write_pattern_nexus(path, taxa, patterns, newick):
    """patterns: list of tuples (one per site) of 0/1 per taxon (taxa order)."""
    n_sites = len(patterns)
    with open(path, "w", encoding="utf-8") as f:
        f.write("#NEXUS\n\nBEGIN TAXA;\n")
        f.write(f"DIMENSIONS NTAX={len(taxa)};\n")
        f.write(f"TAXLABELS {' '.join(taxa)};\nEND;\n\n")
        f.write("BEGIN DATA;\n")
        f.write(f"  Dimensions nchar={n_sites};\n")
        f.write("  Format datatype=snp missing=? gap=- matchchar=.;\n")
        f.write("  Matrix\n")
        for i, t in enumerate(taxa):
            seq = "".join(str(patterns[s][i]) for s in range(n_sites))
            f.write(f"    {t} {seq}\n")
        f.write("  ;\nEND;\n\n")
        f.write("BEGIN TREES;\n")
        f.write(f"Tree net = {newick}\n")
        f.write("END;\n")


def norm_sum(net, taxa, samples, u, v, coal):
    from phynetpy.BiMarkers import _snp_log_likelihood
    from phynetpy.SNPSimulator import _network_to_rich_newick
    from phynetpy.MSA import MSA
    newick = _network_to_rich_newick(net)
    all_patterns = list(itertools.product([0, 1], repeat=len(taxa)))
    total_p = 0.0
    per = []
    for pat in all_patterns:
        path = os.path.join("runs", "pat.nex")
        os.makedirs("runs", exist_ok=True)
        write_pattern_nexus(path, taxa, [pat], newick)
        aln = MSA(path)
        ll = _snp_log_likelihood(net, aln, u, v, coal, samples, verbose=False)
        p = math.exp(ll)
        per.append((pat, ll, p))
        total_p += p
    return total_p, per


def three_taxon_tree(coal_h=0.02):
    """A tiny 3-taxon caterpillar ((A,B),C) for an independent check."""
    from phynetpy.Network import Network, Node, Edge
    net = Network()
    A, B, C = Node("A"), Node("B"), Node("C")
    I0, I1 = Node("I0"), Node("I1")
    net.add_nodes(A, B, C, I0, I1)
    net.add_edges([
        Edge(I0, A, length=coal_h), Edge(I0, B, length=coal_h),
        Edge(I1, I0, length=coal_h), Edge(I1, C, length=2 * coal_h),
    ])
    return net


def set_retic_gamma(net, g_major):
    """Set the reticulation's two parent-edge gammas to (g_major, 1-g_major)."""
    for v in net.V():
        if v.is_reticulation():
            in_edges = net.in_edges(v)
            # deterministic order: sort by source label
            in_edges = sorted(in_edges, key=lambda e: e.src.label)
            in_edges[0].set_gamma(g_major)
            in_edges[1].set_gamma(1.0 - g_major)


def main():
    from phynetpy.BiMarkers import _snp_starting_tree
    import copy

    true_net = build_true_network()
    taxa = list(TAXA)
    samples = {t: 1 for t in taxa}

    for coal in (0.005, 1.0, 2.0):
        for (u, v) in ((1.0, 1.0),):
            print(f"\n=== coal={coal} u={u} v={v} ===")

            # 3-taxon tree (independent sanity check; sum over 8 patterns)
            t3 = three_taxon_tree()
            s3, _ = norm_sum(t3, ["A", "B", "C"], {"A":1,"B":1,"C":1}, u, v, coal)
            print(f"  3-taxon tree          sum_patterns P = {s3:.6f}")

            # 6-taxon caterpillar tree (no reticulation)
            cat = _snp_starting_tree(taxa)
            sc, _ = norm_sum(cat, taxa, samples, u, v, coal)
            print(f"  6-taxon cat tree      sum_patterns P = {sc:.6f}")

            # 6-taxon network, gamma -> 1.0/0.0 (degenerate: should behave
            # like a tree and still sum to 1 if split/merge conserve prob.)
            deg = copy.deepcopy(true_net)
            set_retic_gamma(deg, 1.0)
            sd, _ = norm_sum(deg, taxa, samples, u, v, coal)
            print(f"  6-taxon net gamma=1.0 sum_patterns P = {sd:.6f}")

            # 6-taxon true network (1 reticulation, gamma 0.65/0.35)
            st, per = norm_sum(true_net, taxa, samples, u, v, coal)
            inv0 = next(p for pat, ll, p in per if sum(pat) == 0)
            inv1 = next(p for pat, ll, p in per if sum(pat) == len(taxa))
            print(f"  6-taxon true net      sum_patterns P = {st:.6f}  "
                  f"(all0={inv0:.4g} all1={inv1:.4g})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
