"""
Test MPL_reference scoring against 5 known networks with expected log-likelihoods.

Uses 1000 gene trees from subgeneset_3_ret1.txt and the 5 inferred networks
from 5_nets.txt. Each network has an expected "Total log probability" that
was produced by a reference implementation. We verify that our MPL scorer
reproduces those values.

Taxa (single-allele, identity mapping):
    t14, t15, t49, t68, t69, t72, t75, t91, t114, t133
"""

from __future__ import annotations

import os
import math
import time

from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL import MPL
from phynetpy.IO import convert_newick

TESTFILES = os.path.join(os.path.dirname(__file__), "testfiles")

TAXA = ["t14", "t15", "t49", "t68", "t69", "t72", "t75", "t91", "t114", "t133"]
MAPPING = {t: [t] for t in TAXA}

NETWORKS_AND_EXPECTED = [
    (
        "(((((t14:1.0,t75:1.0):0.8767426254184691,(t69:1.0,t114:1.0):2.0170864999696456):5.213391915869064,(t91:1.0)#H1:1.0::0.3011071643186969):0.8237808968694593,t133:1.0):2.054746072134383,(((t15:1.0,#H1:1.0::0.6988928356813031):1.481149764930965,(t72:1.0,t49:1.0):3.0976006419526203):1.2243080352496307,t68:1.0):0.12981149034576273);",
        -56625.66771610746,
    ),
    (
        "(((((t91:1.0)#H1:1.0::0.6794204549833502,t15:1.0):1.6467579736815752,(t49:1.0,t72:1.0):3.0810338707973433):1.2448178522157847,t68:1.0):0.12897581257103757,((t133:1.0,((t69:1.0,t114:1.0):2.0171458621797465,(t75:1.0,t14:1.0):0.8768041758502694):5.822974539769796):0.0011774181844964955,#H1:1.0::0.3205795450166497):2.027626210985976);",
        -56834.854952896036,
    ),
    (
        "(((((t91:1.0)#H1:1.0::0.6796574824419208,t15:1.0):1.641136791735814,(t49:1.0,t72:1.0):3.112036824582357):1.2429307879371827,t68:1.0):0.13009032050420152,((#H1:1.0::0.32034251755807913,t133:1.0):0.0011774181844964955,((t69:1.0,t114:1.0):2.017144112586501,(t14:1.0,t75:1.0):0.8768024093904544):5.824104159721604):2.031502942440107);",
        -56835.51118261383,
    ),
    (
        "(((((t91:1.0)#H1:1.0::0.7003657952229894,t15:1.0):1.4729051378404348,(t72:1.0,t49:1.0):3.1110182687811236):1.223131289061642,t68:1.0):0.129700998724336,((#H1:1.0::0.2996342047770106,(t14:1.0,((t69:1.0,t114:1.0):2.015898657163544,t75:1.0):0.0011774181844964955):5.305114920848316):0.8410777092373762,t133:1.0):2.0541360855910633);",
        -57258.53277086764,
    ),
    (
        "(((((t75:1.0,t14:1.0):0.8768505106788502,(t69:1.0,t114:1.0):2.0170554334700754):4.98605616510801,(t91:1.0)#H1:1.0::0.2697707099856099):1.068656533278907,t133:1.0):2.0486427396063154,(((t15:1.0,(t72:1.0,t49:1.0):3.1061788071082423):0.0011774181844964955,#H1:1.0::0.7302292900143901):1.2754697724540025,t68:1.0):0.12668290661078757);",
        -57367.764080519984,
    ),
]


def _load_gene_trees() -> GeneTrees:
    """Parse 1000 gene trees from the test file."""
    gt_path = os.path.join(TESTFILES, "subgeneset_3_ret1.txt")
    trees = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(Network.from_newick(line))
    gts = GeneTrees(gene_tree_list=trees)
    gts.species_gene_mapping = MAPPING
    return gts


def _parse_network(phylonet_newick: str) -> Network:
    """Parse a PhyloNet-format extended Newick (with ::gamma) into a Network."""
    phynetpy_newick = convert_newick(phylonet_newick, standard="PhyNetPy")
    return Network.from_newick(phynetpy_newick)


def main():
    print("Loading 1000 gene trees...")
    t0 = time.time()
    gts = _load_gene_trees()
    print(f"  Loaded {len(gts.trees)} gene trees in {time.time()-t0:.1f}s")
    print(f"  Taxa: {sorted(gts.taxa_names)}")
    print()

    all_pass = True
    for i, (nwk, expected) in enumerate(NETWORKS_AND_EXPECTED, 1):
        net = _parse_network(nwk)
        leaves = sorted(n.label for n in net.get_leaves())
        print(f"Network #{i}:")
        print(f"  Leaves: {leaves}")

        t0 = time.time()
        mpl = MPL(net, gts, MAPPING)
        score = mpl.score()
        elapsed = time.time() - t0

        diff = abs(score - expected)
        rel_err = diff / abs(expected) if expected != 0 else diff
        status = "PASS" if rel_err < 1e-4 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  Computed:  {score:.6f}")
        print(f"  Expected:  {expected:.6f}")
        print(f"  Abs diff:  {diff:.6f}")
        print(f"  Rel error: {rel_err:.2e}")
        print(f"  Time:      {elapsed:.2f}s")
        print(f"  [{status}]")
        print()

    if all_pass:
        print("ALL 5 NETWORKS PASSED")
    else:
        print("SOME NETWORKS FAILED — see above")


if __name__ == "__main__":
    main()
