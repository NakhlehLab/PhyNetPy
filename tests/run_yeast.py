#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Copyright 2025 Mark Kessler, Luay Nakhleh.  All rights reserved.
##  See "LICENSE.txt" for terms and conditions of usage.
##############################################################################

"""
Run MCMC_SEQ on a PhyloNet-format multilocus NEXUS file (e.g. the canonical
yeast example ``tests/data/MCMCseq_example0.nex``).

This both (a) demonstrates that PhyNetPy ingests *genuine* PhyloNet MCMC_SEQ
input and (b) gives a small, reusable parser for the multilocus NEXUS dialect:

* the ``Begin data; ... Matrix ... ;End;`` block, where loci are delimited by
  ``[locusName, length, ...]`` marker lines and each following ``taxon
  sequence`` row contributes that locus's alignment, and
* the ``BEGIN PHYLONET; MCMC_SEQ ... -tm <A:a1,a2; B:b1;> ...; END;`` block,
  whose ``-tm`` directive (when present) maps species -> allele labels.  With
  no ``-tm`` each taxon is its own species (one allele).

Usage:
    py tests/run_yeast.py
    py tests/run_yeast.py --nex tests/data/MCMCseq_example0.nex --iters 60000
"""

from __future__ import annotations

import argparse
import os
import re

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69


def parse_multilocus_nexus(path: str):
    """Parse a PhyloNet multilocus NEXUS file.

    Args:
        path: Path to the ``.nex`` file.

    Returns:
        ``(loci, mapping)`` where ``loci`` is a list of ``{taxon: sequence}``
        dicts (one per locus, in file order) and ``mapping`` is
        ``{species: [allele, ...]}`` (identity when no ``-tm`` is given).
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    # --- data matrix ---------------------------------------------------
    m = re.search(r"Matrix(.*?);\s*End;", text, re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("No 'Matrix ... ;End;' data block found.")
    body = m.group(1)

    loci: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("["):          # locus delimiter -> new locus
            current = {}
            loci.append(current)
            continue
        parts = line.split()
        if len(parts) >= 2 and current is not None:
            taxon, seq = parts[0], "".join(parts[1:])
            current[taxon] = current.get(taxon, "") + seq
    loci = [lc for lc in loci if lc]       # drop any empty trailing locus
    if not loci:
        raise ValueError("Parsed zero loci from the data block.")

    # --- optional -tm species map -------------------------------------
    taxa = list(loci[0].keys())
    mapping: dict[str, list[str]] = {t: [t] for t in taxa}
    tm = re.search(r"-tm\s*<([^>]*)>", text, re.IGNORECASE)
    if tm:
        mapping = {}
        for entry in tm.group(1).split(";"):
            entry = entry.strip()
            if not entry:
                continue
            sp, alleles = entry.split(":")
            mapping[sp.strip()] = [a.strip() for a in alleles.split(",") if a.strip()]
    return loci, mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--nex", default=os.path.join(here, "data",
                                                   "MCMCseq_example0.nex"))
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--burnin", type=int, default=15000)
    ap.add_argument("--thin", type=int, default=25)
    ap.add_argument("--theta", type=float, default=0.02)
    ap.add_argument("--max-retic", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    loci, mapping = parse_multilocus_nexus(args.nex)
    total_bp = sum(len(next(iter(lc.values()))) for lc in loci)
    print(f"Loaded {os.path.basename(args.nex)}: "
          f"{len(loci)} loci, {total_bp} bp total, "
          f"{len(mapping)} species ({', '.join(sorted(mapping))})")
    for i, lc in enumerate(loci):
        print(f"  locus {i}: {len(next(iter(lc.values())))} bp x {len(lc)} taxa")

    sampler = MCMC_SEQ(
        loci, mapping, model=JC69(), theta=args.theta,
        priors=MCMCSeqPriors(max_reticulations=args.max_retic),
    )
    print(f"\nStarting logP : {sampler.score():.3f}")
    print(f"Running MCMC_SEQ: {args.iters} iters "
          f"(burn-in {args.burnin}, thin {args.thin}) ...")
    result = sampler.search(
        num_iter=args.iters, burn_in=args.burnin, sample_freq=args.thin,
        seed=args.seed, progress=True,
    )

    print()
    print(f"MAP network : {result.map_network.newick()}")
    print(f"MAP logP    : {result.map_log_posterior:.3f}")
    print(f"MAP theta   : {result.map_theta:.6f}")
    print(f"acceptance  : {result.acceptance_rate:.3f}")
    print()
    print(result.summary())

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        result.write_log(args.out + ".log")
        result.write_networks(args.out + ".trees")
        print(f"\nWrote {args.out}.log and {args.out}.trees (open in Tracer).")


if __name__ == "__main__":
    main()
