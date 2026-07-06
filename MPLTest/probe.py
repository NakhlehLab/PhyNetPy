"""Probe: confirm PhyNetPy can parse the CAMUS Wolbachia gene trees and the
CAMUS output networks (extended Newick), and that MPL scoring runs."""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.IO import convert_newick

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "camus_testdata")


def read_lines(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def try_parse_network(nwk: str):
    # Try raw, then via convert_newick (PhyNetPy standard) for #H extended newick.
    errs = {}
    for label, s in (("raw", nwk), ("convert", None)):
        try:
            if label == "convert":
                s = convert_newick(nwk, standard="PhyNetPy")
            net = Network.from_newick(s)
            return label, net
        except Exception as e:
            errs[label] = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"could not parse network: {errs}")


def main():
    # Gene trees
    gt_lines = read_lines(os.path.join(DATA, "gene-trees.nwk"))
    print(f"gene tree lines: {len(gt_lines)}")
    t0 = time.time()
    g0 = Network.from_newick(gt_lines[0])
    leaves0 = sorted(n.label for n in g0.get_leaves())
    print(f"first gene tree parsed: {len(leaves0)} leaves in {time.time()-t0:.3f}s")
    print("taxa:", leaves0)

    # Constraint tree
    ctree = read_lines(os.path.join(DATA, "constraint.nwk"))[0]
    label, cnet = try_parse_network(ctree)
    cleaves = sorted(n.label for n in cnet.get_leaves())
    print(f"\nconstraint tree parsed via '{label}': {len(cleaves)} leaves")
    print("constraint == genetree taxa:", set(cleaves) == set(leaves0))

    # CAMUS networks
    for fn in ["network.nwk", "net_q2_t05_max.nwk", "net_q2_t05_norm.nwk",
               "net_q2_t05_sym_a01.nwk"]:
        lines = read_lines(os.path.join(DATA, fn))
        print(f"\n{fn}: {len(lines)} networks")
        for i, ln in enumerate(lines):
            try:
                lab, net = try_parse_network(ln)
                nret = sum(1 for v in net.V() if v.is_reticulation())
                nleaf = sum(1 for _ in net.get_leaves())
                print(f"  net[{i}] parsed via '{lab}': leaves={nleaf} retic={nret}")
            except Exception as e:
                print(f"  net[{i}] FAILED: {e}")


if __name__ == "__main__":
    main()
