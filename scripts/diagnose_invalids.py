"""Diagnose why moves produce invalid topologies.

Runs each move type many times on a test network and categorizes failures.
"""

import copy
import random

from phynetpy._mpl import MPL
from phynetpy.IO import read_newick_file
from phynetpy.ModelGraph import Model
from phynetpy.ModelMove import (
    AddReticulation, RemoveReticulation, FlipReticulation,
    SPR, ChangeNodeHeight, ChangeReticSource, ChangeReticDest,
    ChangeInheritanceProb,
)
from phynetpy.infer import MPLScorer

GT_FILE = "tests/testfiles/subgeneset_3_ret1.txt"
TAXA = ["t14", "t15", "t49", "t68", "t69", "t72", "t75", "t91", "t114", "t133"]
MAPPING = {t: [t] for t in TAXA}
TRIALS = 100


def check_invariants(net):
    """Return list of violated invariants."""
    violations = []

    roots = [n for n in net.V() if net.in_degree(n) == 0]
    if len(roots) != 1:
        violations.append(f"root_count={len(roots)}")

    for n in net.V():
        ind = net.in_degree(n)
        outd = net.out_degree(n)

        if ind == 0:
            if outd < 2:
                violations.append(f"root({n.label})_out={outd}")
        elif n.is_reticulation():
            if ind != 2:
                violations.append(f"retic({n.label})_in={ind}")
            if outd != 1:
                violations.append(f"retic({n.label})_out={outd}")
        elif outd == 0:
            if ind != 1:
                violations.append(f"leaf({n.label})_in={ind}")
        else:
            if ind != 1:
                violations.append(f"internal({n.label})_in={ind}")
            if outd < 2:
                violations.append(f"internal({n.label})_out={outd}")

    try:
        if not net.is_acyclic():
            violations.append("has_cycle")
    except Exception as e:
        violations.append(f"acyclic_check_error:{e.__class__.__name__}")

    return violations


def build_model():
    gts = read_newick_file(GT_FILE, return_type="genetrees",
                           species_gene_mapping=MAPPING)
    start_tree = gts.build_majority_rule_consensus_tree()
    mpl = MPL(start_tree, gts, MAPPING)
    scorer = MPLScorer(mpl._rho, mpl._triplets)
    model = Model()
    model.network = copy.deepcopy(mpl.net)
    model.set_likelihood_calculator(scorer)
    return model


def test_move_class(move_cls, model, trials):
    results = {"ok": 0, "exception": {}, "invariant": {}, "noop": 0}

    for _ in range(trials):
        m = copy.deepcopy(model)
        net_before_nodes = len(list(m.network.V()))

        kwargs = {}
        if move_cls is AddReticulation:
            kwargs["max_reticulations"] = 2

        move = move_cls(**kwargs)
        try:
            m = move.execute(m)
        except Exception as e:
            key = f"{e.__class__.__name__}: {str(e)[:60]}"
            results["exception"][key] = results["exception"].get(key, 0) + 1
            continue

        net_after_nodes = len(list(m.network.V()))
        if net_after_nodes == net_before_nodes and move.undo_info is None:
            results["noop"] += 1
            continue

        violations = check_invariants(m.network)
        if violations:
            key = "; ".join(violations[:3])
            results["invariant"][key] = results["invariant"].get(key, 0) + 1
        else:
            try:
                if not m.network.is_acyclic():
                    results["invariant"]["cycle"] = \
                        results["invariant"].get("cycle", 0) + 1
                else:
                    results["ok"] += 1
            except Exception:
                results["ok"] += 1

    return results


def print_network_profile(net, label="Network"):
    """Print degree profile of non-leaf nodes."""
    print(f"\n{label}: {len(list(net.V()))} nodes, "
          f"{len(list(net.E()))} edges")
    violations = check_invariants(net)
    if violations:
        print(f"  VIOLATIONS: {violations}")
    else:
        print(f"  All invariants pass")
    for n in net.V():
        ind = net.in_degree(n)
        outd = net.out_degree(n)
        ntype = "root" if ind == 0 else ("leaf" if outd == 0 else (
            "retic" if n.is_reticulation() else "internal"))
        if ntype not in ("leaf",):
            elen = []
            for e in net.in_edges(n):
                elen.append(e.get_length())
            for e in net.out_edges(n):
                elen.append(e.get_length())
            has_none = any(l is None for l in elen)
            print(f"  {n.label:15s} type={ntype:8s} "
                  f"in={ind} out={outd} none_lengths={has_none}")


def main():
    random.seed(42)
    print("Building model...")
    model = build_model()
    print_network_profile(model.network, "Starting tree")

    print("\n--- Adding reticulations to create a network ---")
    for i in range(2):
        for attempt in range(20):
            m = copy.deepcopy(model)
            move = AddReticulation(max_reticulations=2)
            m = move.execute(m)
            v = check_invariants(m.network)
            retics = sum(1 for n in m.network.V() if n.is_reticulation())
            if retics > i and not v:
                model = m
                print(f"  Reticulation {i+1} added on attempt {attempt+1}")
                break

    print_network_profile(model.network, "Network with reticulations")

    move_classes = [
        AddReticulation, RemoveReticulation, FlipReticulation,
        SPR, ChangeNodeHeight, ChangeReticSource, ChangeReticDest,
        ChangeInheritanceProb,
    ]

    print(f"\n{'='*70}")
    print(f"Running {TRIALS} trials per move type on network WITH reticulation")
    print(f"{'='*70}")

    for cls in move_classes:
        print(f"\n--- {cls.__name__} ---")
        res = test_move_class(cls, model, TRIALS)
        print(f"  OK: {res['ok']}  |  NoOp: {res['noop']}")
        if res["exception"]:
            print(f"  Exceptions:")
            for k, v in sorted(res["exception"].items(),
                                key=lambda x: -x[1]):
                print(f"    [{v}x] {k}")
        if res["invariant"]:
            print(f"  Invariant violations:")
            for k, v in sorted(res["invariant"].items(),
                                key=lambda x: -x[1]):
                print(f"    [{v}x] {k}")


if __name__ == "__main__":
    main()
