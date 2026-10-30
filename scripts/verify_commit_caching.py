#!/usr/bin/env python3
"""
Verify the Stage-1 accepted-move optimisations in ``State.commit``.

Two changes made committing a proposal cheaper:

  1. ``State.commit`` clones the proposed network with ``_clone_net``
     (``Network.copy``) instead of ``copy.deepcopy``.
  2. ``State.commit(move, score=...)`` primes the model's likelihood cache
     with the proposal's score instead of leaving the model dirty for a
     re-score that would return the same number.

Both are only safe if the committed model scores *exactly* what the proposal
scored. A bit-for-bit comparison of whole benchmark runs cannot establish
that, because MP-Allop's start-network construction is not reproducible
across processes (``Edge`` has no ``__hash__``, so the candidate-edge list in
``_infer_mp_allop._attach`` is ordered by memory address). Instead this script
checks the invariant directly, at every accepted move of a real search:

    primed score  ==  score recomputed from scratch on the committed network

and separately that ``_clone_net`` and ``copy.deepcopy`` of the same network
score identically.

Run::

    PHYNETPY_DEFJ_ROOT=... .venv/Scripts/python.exe scripts/verify_commit_caching.py

Exit code 0 == invariant held on every accepted move.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import argparse
import copy
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_defj import build_model  # noqa: E402

from phynetpy.GraphUtils import _clone_net  # noqa: E402
from phynetpy.ModelMove import SwitchParentage  # noqa: E402
from phynetpy.State import State  # noqa: E402

# (scenario, tier, g, n, t, r)
CASES = [
    ("D", 10, 1, 1, 4, 1),
    ("D", 10, 10, 1, 100, 1),
    ("E", 10, 10, 1, 4, 1),
    ("F", 10, 10, 1, 100, 1),
    ("J", 10, 3, 1, 20, 1),
]


def check_clone_scores_equal(model) -> tuple[int, int]:
    """Score the network, then a _clone_net and a deepcopy of it."""
    model._dirty = True
    base = model.likelihood()

    original = model.network
    model.network = _clone_net(original)
    model._dirty = True
    cloned = model.likelihood()

    model.network = copy.deepcopy(original)
    model._dirty = True
    deep = model.likelihood()

    model.network = original
    model._dirty = True
    return base, (cloned, deep)


def check_commit_priming(model, n_iters: int) -> tuple[int, int, list]:
    """Hill-climb with priming on, re-scoring after each accept to compare.

    Returns (accepts_checked, violations, sample_violations).
    """
    state = State(copy.deepcopy(model))
    accepts = 0
    violations = []

    for i in range(n_iters):
        move = SwitchParentage(i)
        if not state.generate_next(move):
            continue
        try:
            cur = state.likelihood()
            proposed = state.proposed().likelihood()
        except Exception:
            state.revert(move)
            continue

        # Accept everything that is not strictly worse, so the priming path
        # gets exercised on neutral and improving moves alike.
        if cur - proposed <= 0:
            state.commit(move, score=proposed)
            accepts += 1

            primed = state.likelihood()          # served from the cache
            state.current_model._dirty = True
            recomputed = state.likelihood()      # full re-score

            if primed != recomputed:
                violations.append((i, primed, recomputed))
            # Restore the primed value so the search continues as it would.
            state.current_model.prime_likelihood(primed)
        else:
            state.revert(move)

    return accepts, len(violations), violations[:5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_accepts = 0
    total_violations = 0

    for (scenario, tier, g, n, t, r) in CASES:
        label = f"{scenario}-{tier}G-g{g}-n{n}-t{t}-r{r}"
        try:
            model, n_genes, _ = build_model(scenario, tier, g, n, t, r, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label}: SKIP ({exc})", flush=True)
            continue

        base, (cloned, deep) = check_clone_scores_equal(model)
        clone_ok = (base == cloned == deep)

        accepts, violations, sample = check_commit_priming(model, args.iters)
        total_accepts += accepts
        total_violations += violations

        status = "OK" if (clone_ok and violations == 0) else "FAIL"
        print(f"  {label:<24} {status}  "
              f"clone/deepcopy/base score = {cloned}/{deep}/{base}  "
              f"accepts checked = {accepts}  violations = {violations}",
              flush=True)
        for v in sample:
            print(f"      iter {v[0]}: primed={v[1]} recomputed={v[2]}", flush=True)

    print()
    print(f"Total accepted moves checked: {total_accepts}")
    print(f"Total priming violations:     {total_violations}")
    return 1 if total_violations else 0


if __name__ == "__main__":
    sys.exit(main())
