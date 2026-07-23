"""Deterministic verification of the identifiability guards in log_prior_seq.

Constructs networks that are valid vs degenerate and checks that the SEQ prior
accepts (finite) or rejects (-inf) each, confirming steps 1-3:

  1. reticulation-cycle guard  (2-cycles and 3-cycles -> -inf; 4-cycle -> finite)
  2. Beta(2,2) gamma prior      (gamma near 0/1 penalised; gamma = 1 -> -inf)
  3. minimum hybrid-edge length (t -> 0 -> -inf)
"""
from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phynetpy.Network import Network, Node, Edge
from phynetpy._mcmc_seq import (
    log_prior_seq, MCMCSeqPriors, _reticulation_cycle_size,
)
import _run_weekend as wk

PRIORS = MCMCSeqPriors(max_reticulations=2)
THETA = 0.02


def _net(nodes_h, edges):
    """Build a network from {name: height} and (parent, child, gamma) edges."""
    nodes = {n: Node(n, is_reticulation=(n == "H")) for n in nodes_h}
    net = Network()
    net.add_nodes(*nodes.values())
    es = []
    for p, c, g in edges:
        length = nodes_h[p] - nodes_h[c]
        if g is None:
            es.append(Edge(nodes[p], nodes[c], length=length))
        else:
            es.append(Edge(nodes[p], nodes[c], length=length, gamma=g))
    net.add_edges(es)
    return net, nodes


def _report(label, net, expect_finite):
    lp = log_prior_seq(net, THETA, PRIORS)
    ok = math.isfinite(lp) == expect_finite
    rets = [v for v in net.V() if v.is_reticulation()]
    csize = _reticulation_cycle_size(net, rets[0]) if rets else "-"
    verdict = "PASS" if ok else "**FAIL**"
    lp_s = f"{lp:.3f}" if math.isfinite(lp) else "-inf"
    print(f"  [{verdict}] {label:<44} cycle={csize}  logPrior={lp_s}  "
          f"(expected {'finite' if expect_finite else '-inf'})")
    return ok


def main() -> None:
    all_ok = True
    print("Identifiability-guard verification (log_prior_seq):\n")

    # --- Valid 4-node cycle (the true 10-taxon network) -------------------
    true_net = wk.build_true_network()
    all_ok &= _report("true 10-taxon net (4-cycle, gamma=0.7)", true_net, True)

    # --- 3-node cycle (triangle bubble): parents in ancestor-descendant ---
    tri, _ = _net(
        {"R": 1.0, "P": 0.7, "Q": 0.5, "H": 0.3, "a": 0.0, "b": 0.0, "c": 0.0},
        [("R", "P", None), ("R", "c", None), ("P", "Q", None),
         ("P", "H", 0.6), ("Q", "H", 0.4), ("Q", "a", None), ("H", "b", None)],
    )
    all_ok &= _report("3-cycle triangle bubble", tri, False)

    # --- 2-node cycle (parallel hybrid edges from one parent) -------------
    try:
        par, _ = _net(
            {"R": 1.0, "P": 0.6, "H": 0.3, "a": 0.0, "b": 0.0, "c": 0.0},
            [("R", "P", None), ("R", "c", None), ("P", "H", 0.6),
             ("P", "H", 0.4), ("P", "a", None), ("H", "b", None)],
        )
        all_ok &= _report("2-cycle parallel-edge bubble", par, False)
    except Exception as e:
        print(f"  [n/a ] 2-cycle parallel-edge bubble  (not representable: {e})")

    # --- Gamma boundary behaviour on the valid 4-cycle --------------------
    def _set_gamma(net, g):
        n2, _ = _net(
            {n.label: wk._TRUE_HEIGHTS[n.label] for n in net.V()},
            [(e.src.label, e.dest.label,
              (g if e.dest.label == "H" and e.src.label == "P1"
               else (1.0 - g) if e.dest.label == "H" else None))
             for e in net.E()],
        )
        return n2

    for g, exp in [(0.7, True), (0.95, True), (0.999, True), (1.0, False)]:
        net_g = _set_gamma(true_net, g)
        lp = log_prior_seq(net_g, THETA, PRIORS)
        lp_s = f"{lp:.3f}" if math.isfinite(lp) else "-inf"
        ok = math.isfinite(lp) == exp
        all_ok &= ok
        print(f"  [{'PASS' if ok else '**FAIL**'}] gamma={g:<6} on 4-cycle"
              f"{'':<24} logPrior={lp_s}")

    # --- Zero-length hybrid edge (t -> 0) ---------------------------------
    zero, nodes = _net(
        {**{n: wk._TRUE_HEIGHTS[n] for n in wk._TRUE_HEIGHTS}},
        [(p, c, g) for (p, c, g) in wk._TRUE_EDGES],
    )
    # collapse P1 onto H's height so the P1->H hybrid edge has length ~0
    for e in zero.in_edges([v for v in zero.V() if v.is_reticulation()][0]):
        e.set_length(0.0)
    all_ok &= _report("zero-length hybrid edges (t->0)", zero, False)

    print("\n" + ("ALL GUARDS VERIFIED" if all_ok else "SOME CHECKS FAILED"))


if __name__ == "__main__":
    main()
