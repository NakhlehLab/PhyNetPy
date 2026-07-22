#!/usr/bin/env python3
"""
Profile Infer_MP_Allop on the paper_net topology with a valid polyploid gene map.

``tests/testfiles/paper_net.nex`` alone uses species-labeled tips (A,B,C) on a
bubble network; MUL expansion yields four leaves, so a diploid identity map
cannot score. This script uses the same network from that file with
``B`` mapped to two gene copies (``B_a``, ``B_b``) and three binary gene trees
on four tips — the setup expected by Allop_MUL / MPAllopScorer.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import io
from pathlib import Path

import numpy as np

from phynetpy.IO import read_nexus, read_newick
from phynetpy.Infer_MP_Allop import InferMPAllop


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iter", type=int, default=400, help="Hill-climbing iterations"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed"
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="Write pstats.Stats binary (e.g. paper_mp.prof) for snakeviz",
    )
    args = parser.parse_args()

    nex = _project_root() / "tests" / "testfiles" / "paper_net.nex"
    start = read_nexus(str(nex))[0]
    gene_map = {"A": ["A_a"], "B": ["B_a", "B_b"], "C": ["C_a"]}
    gt_strs = [
        "((A_a,B_a),(B_b,C_a));",
        "((A_a,C_a),(B_a,B_b));",
        "((B_a,B_b),(A_a,C_a));",
    ]
    gts = [read_newick(s) for s in gt_strs]
    rng = np.random.default_rng(args.seed)

    infer = InferMPAllop(start, gene_map, gts, iter_ct=args.iter, rng=rng)

    prof = cProfile.Profile()
    prof.enable()
    score = infer.run()
    prof.disable()

    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(40)
    print(s.getvalue())

    print("---")
    print(f"final likelihood (negated parsimony): {score}")
    print(f"leaderboard networks: {len(infer.results)}")
    end = infer.mp_allop_model.network
    n_ret = sum(1 for v in end.V() if v.is_reticulation())
    print(f"final network reticulation nodes: {n_ret}")

    if args.profile_out:
        prof.dump_stats(str(args.profile_out))
        print(f"Wrote profile to {args.profile_out}")


if __name__ == "__main__":
    main()
