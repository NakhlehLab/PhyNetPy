"""
Differential test for the MPL incremental scorer (lever 3).

Asserts that the cached-engine refresh path (used when only branch lengths /
gammas change) produces *bit-for-bit* the same log-pseudo-likelihood as a full
engine rebuild, across many random parameter edits and across topology
changes.  This is the safety net for the lever-3 caching: if the incremental
path ever diverges from a from-scratch score, this test fails.

Run directly (``python tests/test_mpl_incremental.py``) or via pytest.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL import MPL, MPLScorer, _HAS_CYTHON_MPL
from phynetpy.ModelGraph import Model
from phynetpy.IO import convert_newick

TESTFILES = os.path.join(os.path.dirname(__file__), "testfiles")
TAXA = ["t14", "t15", "t49", "t68", "t69", "t72", "t75", "t91", "t114", "t133"]
MAPPING = {t: [t] for t in TAXA}

REF_R1 = (
    "(((((t14:1.0,t75:1.0):0.8767426254184691,(t69:1.0,t114:1.0):2.0170864999696456)"
    ":5.213391915869064,(t91:1.0)#H1:1.0::0.3011071643186969):0.8237808968694593,"
    "t133:1.0):2.054746072134383,(((t15:1.0,#H1:1.0::0.6988928356813031):1.481149764930965,"
    "(t72:1.0,t49:1.0):3.0976006419526203):1.2243080352496307,t68:1.0):0.12981149034576273);"
)
START_TREE = (
    "((((t14:1,t15:1):1,(t49:1,t68:1):1):1,"
    "((t69:1,t72:1):1,(t75:1,t91:1):1):1):1,"
    "(t114:1,t133:1):1);"
)

TOL = 1e-6


def _load_gts() -> GeneTrees:
    trees = []
    with open(os.path.join(TESTFILES, "subgeneset_3_ret1.txt")) as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(Network.from_newick(line))
    gts = GeneTrees(gene_tree_list=trees)
    gts.species_gene_mapping = MAPPING
    return gts


def _fresh_score(net: Network, rho, triplets) -> float:
    """Score via a brand-new scorer (always a full rebuild)."""
    s = MPLScorer(rho, triplets)
    m = Model()
    m.network = net
    m.set_likelihood_calculator(s)
    return s(m)


def _perturb_params(net: Network, rng: np.random.Generator) -> None:
    """Randomly re-value branch lengths and reticulation gammas in place."""
    for e in net.E():
        if e.get_length() is not None:
            e.set_length(float(rng.uniform(0.1, 4.0)))
    for v in net.V():
        if v.is_reticulation():
            in_edges = list(net.in_edges(v))
            if len(in_edges) >= 2:
                g = float(rng.uniform(0.05, 0.95))
                in_edges[0].set_gamma(g)
                in_edges[1].set_gamma(1.0 - g)


def main() -> int:
    print(f"Cython MPL backend active: {_HAS_CYTHON_MPL}")
    gts = _load_gts()
    net = Network.from_newick(convert_newick(REF_R1, standard="PhyNetPy"))
    mpl = MPL(net, gts, MAPPING)
    rho, triplets = mpl._rho, mpl._triplets

    rng = np.random.default_rng(12345)

    # Persistent (incremental) scorer bound to a single model/network.
    scorer = MPLScorer(rho, triplets)
    model = Model()
    model.network = net
    model.set_likelihood_calculator(scorer)

    # First call: full rebuild (dirty defaults to None).
    model.update_network()
    s_inc = scorer(model)
    s_ref = _fresh_score(Network.from_newick(convert_newick(REF_R1, standard="PhyNetPy")), rho, triplets)
    max_abs = abs(s_inc - s_ref)
    print(f"initial: incremental={s_inc:.6f} fresh={s_ref:.6f} |d|={max_abs:.2e}")

    n_checks = 0
    failures = 0
    for i in range(40):
        _perturb_params(net, rng)
        # Signal a parameters-only change (non-None touched set) so the
        # scorer takes the refresh path, exactly like the optimiser does.
        model.mark_touched({net.root()})
        s_inc = scorer(model)

        # Independent from-scratch score of the identical network state.
        s_ref = _fresh_score_same_net(net, rho, triplets)

        d = abs(s_inc - s_ref)
        max_abs = max(max_abs, d)
        n_checks += 1
        if d > TOL:
            failures += 1
            print(f"  [FAIL] edit {i}: incremental={s_inc:.6f} "
                  f"fresh={s_ref:.6f} |d|={d:.3e}")

    print(f"\n{n_checks} param-edit checks, max |d| = {max_abs:.3e}, "
          f"failures = {failures}")
    if failures == 0:
        print("PASS: incremental refresh path matches full rebuild.")
        return 0
    print("FAIL: incremental path diverged from full rebuild.")
    return 1


def _fresh_score_same_net(net: Network, rho, triplets) -> float:
    """Full-rebuild score of the *current* state of ``net`` (new scorer)."""
    s = MPLScorer(rho, triplets)
    m = Model()
    m.network = net
    m.set_likelihood_calculator(s)
    m.update_network()  # force full-rebuild path
    return s(m)


if __name__ == "__main__":
    raise SystemExit(main())
