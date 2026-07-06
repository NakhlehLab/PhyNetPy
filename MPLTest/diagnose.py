"""
Follow-up diagnostics for why PhyNetPy's MPL search returns r=0 on the CAMUS
Wolbachia data, and whether it is fixable:

  (A) FT search with optimize_scope="reticulation" (optimise the new
      reticulation's incident branch lengths + gamma per topology, not just
      gamma) -- does the search then accept reticulations?

  (B) CAMUS-seeded refine: start the FT search FROM the CAMUS r=2 network and
      see whether we recover / beat its optimised MPL score (-411175) -- i.e.
      is the starting *network* (not the search engine) the binding constraint?
"""
from __future__ import annotations

import contextlib
import io as _io
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL import MPLScorer, compute_gene_tree_triplets
from phynetpy.ModelGraph import Model
from phynetpy._optimize import optimize_network_parameters
from phynetpy._mcmc_gt import _populate_default_branch_lengths
from phynetpy.infer import InferNetwork_ML

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "camus_testdata")
_SUPPORT = re.compile(r"\)(\d+(?:\.\d+)?)(?=[:,);])")
clean = lambda s: _SUPPORT.sub(")", s)
OPT = dict(max_rounds=8, branch_iters=8)


def read_lines(p):
    return [l.strip() for l in open(p) if l.strip()]


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(_io.StringIO()):
        yield


def main():
    trees = [Network.from_newick(clean(l))
             for l in read_lines(os.path.join(DATA, "gene-trees.nwk"))]
    taxa = sorted(n.label for n in trees[0].get_leaves())
    mapping = {t: [t] for t in taxa}
    gts = GeneTrees(gene_tree_list=trees)
    gts.species_gene_mapping = mapping
    pre = compute_gene_tree_triplets(gene_trees=gts, mapping=mapping,
                                     species_labels=taxa)
    rho = pre.rho_by_triplet
    active = [t for t in pre.triplets if any(rho[t][i] > 0.0 for i in range(3))]

    def constraint():
        return Network.from_newick(
            clean(read_lines(os.path.join(DATA, "constraint.nwk"))[0]))

    def camus(line_idx):
        return Network.from_newick(
            read_lines(os.path.join(DATA, "network.nwk"))[line_idx])

    def score_opt(net):
        _populate_default_branch_lengths(net)
        sc = MPLScorer(rho, active)
        m = Model(rng=np.random.default_rng(0))
        m.network = net
        m.set_likelihood_calculator(sc)
        return optimize_network_parameters(m, sc, mapping, scope="all", **OPT)

    # ── (A) FT search, scope="reticulation" ───────────────────────────
    print("=== (A) FT search, optimize_scope='reticulation' ===", flush=True)
    for maxret in (1, 2):
        inf = InferNetwork_ML(constraint(), gts, mapping,
                              max_reticulations=maxret)
        t0 = time.time()
        with quiet():
            res = inf.search(num_runs=1, num_iter=150, max_failures=80,
                             pseudo=True, fix_st=True, optimize_params=True,
                             optimize_scope="reticulation", final_optimize=True,
                             seed=0, **OPT)
        print(f"  maxret={maxret}: logPL={res.best_log_likelihood:.2f}  "
              f"retic={res.num_reticulations}  "
              f"examined={res.num_networks_examined}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # ── (B) CAMUS-seeded refine ───────────────────────────────────────
    print("\n=== (B) CAMUS-seeded FT refine (start FROM CAMUS r=2 net) ===",
          flush=True)
    camus_r2 = camus(1)  # line 2 = r=2
    base = score_opt(camus(1))
    print(f"  CAMUS r=2 optimised score (target): {base:.2f}", flush=True)
    inf = InferNetwork_ML(camus(1), gts, mapping, max_reticulations=2)
    t0 = time.time()
    with quiet():
        res = inf.search(num_runs=1, num_iter=150, max_failures=80,
                         pseudo=True, fix_st=True, optimize_params=True,
                         optimize_scope="reticulation", final_optimize=True,
                         seed=0, **OPT)
    print(f"  refined: logPL={res.best_log_likelihood:.2f}  "
          f"retic={res.num_reticulations}  "
          f"examined={res.num_networks_examined}  ({time.time()-t0:.1f}s)",
          flush=True)


if __name__ == "__main__":
    main()
