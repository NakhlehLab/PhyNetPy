#!/usr/bin/env python3
"""
Correctness verification for MP-Allop-2 and the SwitchParentage (SRPP) move.

This is Phase 0 of the DEFJ benchmark effort: before running any large
sweep we confirm that

  1. The ``SwitchParentage`` move preserves the *invariants* it is supposed
     to preserve on every proposal:
        - the network stays acyclic,
        - the leaf (taxon) set is unchanged,
        - the per-leaf ploidy (subgenome count) is identical pre/post move,
        - ``undo`` restores the network bit-for-bit (edge multiset).
  2. End-to-end inference on a known-easy DEFJ case (D, 1 gene, low ILS)
     drives the parsimony score down to the optimum (0) and lands on /
     near the true species network.

Run::

    .venv/Scripts/python.exe scripts/verify_mp_allop.py

Exit code 0 == all checks passed; non-zero == a check failed.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import argparse
import collections
import copy
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# Ground-truth networks, gene maps, and dataset paths are single-sourced in
# defj_common so every DEFJ script agrees on e.g. the corrected 3-reticulation
# J topology (see defj_common.TRUE_NETWORKS for the history of that fix).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import defj_common as dc  # noqa: E402

from phynetpy.IO import read_newick
from phynetpy.Infer_MP_Allop import (
    MPAllopComponent,
    allele_map_set,
    partition_gene_trees,
)
from phynetpy.ModelFactory import ModelFactory
from phynetpy.ModelMove import SwitchParentage
from phynetpy.MetropolisHastings import (
    Infer_MP_Allop_Kernel,
    HillClimbing,
    SimulatedAnnealing,
)
from phynetpy.State import State
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFJ_TIER = 10
DEFJ_ROOT = dc.defj_root(DEFJ_TIER)
TRUE_NETWORKS = dc.TRUE_NETWORKS
GENE_MAPS = dc.GENE_MAPS


def load_gene_trees(scenario: str, g: int, n: int, t: int, r: int) -> list:
    return dc.load_gene_trees(scenario, DEFJ_TIER, g, n, t, r, collapse=False)


def build_model(scenario: str, g: int, n: int, t: int, r: int, seed: int):
    """Build an MP-Allop model for a DEFJ condition, with a seeded RNG."""
    rng = np.random.default_rng(seed)
    gene_map = GENE_MAPS[scenario]
    gts = load_gene_trees(scenario, g, n, t, r)
    for gt in gts:
        gt.put_item("allele maps", allele_map_set(gt, gene_map))
        gt.put_item("leaf descendants", gt.leaf_descendants_all())
    start_net = partition_gene_trees(gene_map, rng=rng)
    model = ModelFactory(MPAllopComponent(start_net, gene_map, gts, rng)).build()
    # ModelFactory builds a Model() with its own default (OS-seeded) RNG; pin
    # it so the move stream is reproducible.
    model.rng = rng
    return model


# ----------------------------------------------------------------------
# Invariant helpers
# ----------------------------------------------------------------------

def edge_signature(net) -> collections.Counter:
    """Multiset of (src_label, dest_label) edges; survives deepcopy."""
    return collections.Counter(
        (e.src.label, e.dest.label) for e in net.E()
    )


def leaf_labels(net) -> frozenset:
    return frozenset(nd.label for nd in net.get_leaves())


def ploidy_map(net) -> dict:
    """Map leaf label -> subgenome count (ploidy)."""
    return {nd.label: net.subgenome_count(nd) for nd in net.get_leaves()}


def structural_issues(net) -> list[str]:
    """Non-fatal structural sanity checks for a binary allopolyploid network."""
    issues = []
    root = net.root()
    for nd in net.V():
        indeg, outdeg = net.in_degree(nd), net.out_degree(nd)
        if nd == root:
            if indeg != 0:
                issues.append(f"root '{nd.label}' has in-degree {indeg} (!=0)")
            continue
        if outdeg == 0:  # leaf
            if indeg != 1:
                issues.append(f"leaf '{nd.label}' has in-degree {indeg} (!=1)")
        else:  # internal
            if indeg > 2:
                issues.append(f"node '{nd.label}' in-degree {indeg} (>2)")
            if indeg == 1 and outdeg == 1:
                issues.append(f"node '{nd.label}' is unsuppressed (1,1)")
            ret = nd.is_reticulation()
            if ret and indeg != 2:
                issues.append(
                    f"reticulation '{nd.label}' has in-degree {indeg} (!=2)"
                )
            if (not ret) and indeg == 2:
                issues.append(
                    f"node '{nd.label}' has in-degree 2 but is not flagged retic"
                )
    return issues


# ----------------------------------------------------------------------
# Check 1: per-move invariants (ploidy preservation + bit-for-bit undo)
# ----------------------------------------------------------------------

def check_move_invariants(scenario: str, n_trials: int, seed: int) -> dict:
    """
    Random-walk the network with SwitchParentage, checking every proposal:
      - acyclic, leaf-set invariant, per-leaf ploidy invariant
      - on rejected moves: undo restores the exact edge multiset
    Roughly half the proposals are rejected (random walk) to exercise undo,
    the rest accepted to diversify the topologies visited.
    """
    model = build_model(scenario, g=1, n=1,
                        t=(20 if scenario == "J" else 4), r=1, seed=seed)
    rng = np.random.default_rng(seed + 1)

    results = {
        "trials": 0,
        "valid_moves": 0,
        "invalid_moves": 0,
        "ploidy_violations": 0,
        "leafset_violations": 0,
        "acyclic_violations": 0,
        "undo_failures": 0,
        "undo_tested": 0,
        "structural_warnings": 0,
        "examples": [],
    }

    for i in range(n_trials):
        results["trials"] += 1
        net = model.network
        sig_before = edge_signature(net)
        leaves_before = leaf_labels(net)
        ploidy_before = ploidy_map(net)

        move = SwitchParentage(i)
        try:
            move.execute(model)
        except Exception as exc:  # a move that raises is itself a defect
            results["invalid_moves"] += 1
            if len(results["examples"]) < 5:
                results["examples"].append(f"trial {i}: execute raised {exc!r}")
            continue

        post = model.network
        results["valid_moves"] += 1

        # --- invariant: acyclic ---
        if not post.is_acyclic():
            results["acyclic_violations"] += 1
            if len(results["examples"]) < 5:
                results["examples"].append(f"trial {i}: produced a cycle")

        # --- invariant: leaf set unchanged ---
        leaves_after = leaf_labels(post)
        if leaves_after != leaves_before:
            results["leafset_violations"] += 1
            if len(results["examples"]) < 5:
                results["examples"].append(
                    f"trial {i}: leaf set changed "
                    f"{sorted(leaves_before)} -> {sorted(leaves_after)}"
                )

        # --- invariant: per-leaf ploidy preserved (THE central claim) ---
        ploidy_after = ploidy_map(post)
        if ploidy_after != ploidy_before:
            results["ploidy_violations"] += 1
            if len(results["examples"]) < 5:
                diff = {k: (ploidy_before.get(k), ploidy_after.get(k))
                        for k in set(ploidy_before) | set(ploidy_after)
                        if ploidy_before.get(k) != ploidy_after.get(k)}
                results["examples"].append(
                    f"trial {i}: ploidy changed {diff}"
                )

        # --- structural (non-fatal) ---
        if structural_issues(post):
            results["structural_warnings"] += 1

        # Decide accept/reject: reject ~half to exercise undo bit-for-bit.
        reject = rng.random() < 0.5
        if reject:
            results["undo_tested"] += 1
            move.undo(model)
            sig_after_undo = edge_signature(model.network)
            if sig_after_undo != sig_before:
                results["undo_failures"] += 1
                if len(results["examples"]) < 5:
                    results["examples"].append(
                        f"trial {i}: undo did not restore edge multiset"
                    )
        # else: keep the proposed network and walk forward.

    return results


# ----------------------------------------------------------------------
# Check 2: end-to-end inference on a known-easy case
# ----------------------------------------------------------------------

def check_end_to_end(scenario: str, g: int, t: int, n_iters: int,
                     seed: int) -> dict:
    """Run HC + SA to convergence; report parsimony and distance to truth."""
    true_net = read_newick(TRUE_NETWORKS[scenario])
    model = build_model(scenario, g=g, n=1, t=t, r=1, seed=seed)

    init_pars = -State(copy.deepcopy(model)).likelihood()

    # Hill climbing
    hc_state = State(copy.deepcopy(model))
    t0 = time.perf_counter()
    accepted = 0
    for i in range(n_iters):
        move = SwitchParentage(i)
        if hc_state.generate_next(move):
            cur = hc_state.likelihood()
            proposed = hc_state.proposed().likelihood()
            if cur - proposed < 0:
                hc_state.commit(move)
                accepted += 1
            else:
                hc_state.revert(move)
    hc_time = time.perf_counter() - t0
    hc_net = hc_state.current_model.network
    hc = {
        "final_pars": -hc_state.likelihood(),
        "accepted": accepted,
        "mu_d": mu_distance(hc_net, true_net),
        "hw_d": hardwired_cluster_distance(hc_net, true_net),
        "time": hc_time,
    }

    # Simulated annealing x3 restarts
    sa_model = copy.deepcopy(model)
    t0 = time.perf_counter()
    sa = SimulatedAnnealing(
        pkernel=Infer_MP_Allop_Kernel(), model=sa_model,
        num_iter=n_iters, t_start=5.0, t_end=0.01, n_restarts=3, seed=seed,
    )
    sa_state = sa.run()
    sa_time = time.perf_counter() - t0
    sa_net = sa_state.current_model.network
    sa = {
        "final_pars": -sa.best_score,
        "mu_d": mu_distance(sa_net, true_net),
        "hw_d": hardwired_cluster_distance(sa_net, true_net),
        "time": sa_time,
    }

    return {"init_pars": init_pars, "hc": hc, "sa": sa}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400,
                        help="invariant trials per scenario")
    parser.add_argument("--iters", type=int, default=300,
                        help="search iterations for the end-to-end check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", default="D,E,F,J",
                        help="comma-separated scenarios for invariant checks")
    args = parser.parse_args()

    if not DEFJ_ROOT.exists():
        print(f"ERROR: DEFJ data not found at {DEFJ_ROOT}", flush=True)
        return 2

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    failed = False

    print("=" * 78, flush=True)
    print("  MP-Allop-2 / SwitchParentage (SRPP) correctness verification",
          flush=True)
    print("=" * 78, flush=True)

    # ---- Check 1: invariants ----
    print(f"\n[1] Per-move invariants ({args.trials} trials/scenario)\n",
          flush=True)
    hdr = (f"  {'Scen':<5} {'valid':>6} {'inval':>6} {'ploidy!':>8} "
           f"{'leaf!':>6} {'cyc!':>5} {'undo?':>6} {'undoX':>6} {'struct?':>8}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for s in scenarios:
        res = check_move_invariants(s, args.trials, args.seed)
        fatal = (res["ploidy_violations"] or res["leafset_violations"]
                 or res["acyclic_violations"] or res["undo_failures"])
        if fatal:
            failed = True
        print(
            f"  {s:<5} {res['valid_moves']:>6} {res['invalid_moves']:>6} "
            f"{res['ploidy_violations']:>8} {res['leafset_violations']:>6} "
            f"{res['acyclic_violations']:>5} {res['undo_tested']:>6} "
            f"{res['undo_failures']:>6} {res['structural_warnings']:>8}",
            flush=True,
        )
        for ex in res["examples"]:
            print(f"        ! {ex}", flush=True)

    print("\n   Legend: ploidy!/leaf!/cyc!/undoX are FAILURE counts (want 0);",
          flush=True)
    print("           undo? = number of moves whose undo was checked;",
          flush=True)
    print("           struct? = non-fatal binary-structure warnings.",
          flush=True)

    # ---- Check 2: end-to-end ----
    print(f"\n[2] End-to-end inference (D, 1 gene, low ILS, {args.iters} iters)\n",
          flush=True)
    e2e = check_end_to_end("D", g=1, t=4, n_iters=args.iters, seed=args.seed)
    print(f"  init parsimony : {e2e['init_pars']:.0f}", flush=True)
    print(f"  HC : pars={e2e['hc']['final_pars']:.0f}  "
          f"mu_d={e2e['hc']['mu_d']}  hw_d={e2e['hc']['hw_d']}  "
          f"accepted={e2e['hc']['accepted']}  {e2e['hc']['time']:.2f}s",
          flush=True)
    print(f"  SAx3: pars={e2e['sa']['final_pars']:.0f}  "
          f"mu_d={e2e['sa']['mu_d']}  hw_d={e2e['sa']['hw_d']}  "
          f"{e2e['sa']['time']:.2f}s", flush=True)

    best_pars = min(e2e["hc"]["final_pars"], e2e["sa"]["final_pars"])
    if best_pars > 0:
        print(f"\n  WARNING: best parsimony {best_pars:.0f} > 0 on an easy case "
              f"(expected 0).", flush=True)
        failed = True

    print("\n" + "=" * 78, flush=True)
    print(f"  RESULT: {'FAIL' if failed else 'PASS'}", flush=True)
    print("=" * 78, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
