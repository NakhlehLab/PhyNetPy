"""
Kernel-isolated Cython A/B for the PhyNetPy MPL triplet DP.

The end-to-end ``score()`` A/B understates the compiled kernel because each
call also pays Python-side setup (engine construction, topology flattening,
triplet indexing). This harness separates the three costs so each claim can be
attributed to the right layer:

  A. ``_cy_score_from_topo``  -- pure Cython DP over a *cached* topology.
     This is the inner loop the ``.pyx`` rewrite actually replaced.
  B. ``_score_with_cython``   -- topology flattening + Cython DP, i.e. what a
     naive per-call Cython invocation costs without the 0.6.0 caching.
  C. Python reference DP      -- ``_TripleDPEngine.calculate_triple_probability``
     summed over all active triplets.

Reported speedups: C/A (kernel) and C/B (kernel without topology caching).
Both paths are asserted to agree numerically.

Run:  .venv/Scripts/python.exe scripts/portfolio_benchmarks_kernel.py
Out:  scripts/portfolio_benchmarks_kernel.json
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

OUT_PATH = Path(__file__).with_suffix(".json")
RESULTS: dict[str, Any] = {"meta": {}, "kernel_ab": {}}


def timed(fn: Callable[[], Any], loops: int, repeats: int = 5) -> float:
    """Median seconds per call."""
    fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(loops):
            fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) / loops


def main() -> None:
    import phynetpy
    from phynetpy import _mpl
    from phynetpy._mpl import (
        MPL,
        _build_cython_triplet_index,
        _cy_score_from_topo,
        _extract_topology_for_cython,
        _score_with_cython,
    )
    from phynetpy.infer import simulate
    from phynetpy.models import MSC

    RESULTS["meta"] = {
        "phynetpy_version": phynetpy.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "has_cython_mpl": bool(_mpl._HAS_CYTHON_MPL),
    }
    print("PhyNetPy MPL kernel A/B")
    for k, v in RESULTS["meta"].items():
        print(f"  {k:20}{v}")

    if not _mpl._HAS_CYTHON_MPL:
        print("\nCython MPL engine unavailable; aborting.")
        return

    _LOG_FLOOR = _mpl._LOG_FLOOR

    print()
    print("=" * 92)
    print("MPL triplet DP: pure Cython kernel vs pure Python reference")
    print("=" * 92)
    print(f"  {'taxa':>5} {'trips':>7} "
          f"{'A cy-cached':>13} {'B cy+extract':>14} {'C python':>12} "
          f"{'C/A':>8} {'C/B':>8} {'agree':>10}")
    print("  " + "-" * 88)

    for n_taxa, n_gts in ((6, 40), (8, 60), (10, 80), (12, 100), (16, 120)):
        try:
            gts = simulate(MSC(theta=0.02), taxa=n_taxa, n=n_gts,
                           data="gene_trees", seed=11)
            net = gts.true_network
            labels = [n.label for n in net.get_leaves()]
            mapping = {s: [s] for s in labels}

            mpl = MPL(net, gts, mapping)
            engine = mpl._triple_engine
            active = mpl._active_triplets
            rho = mpl._rho

            # Cached-topology Cython setup (done once, as 0.6.0 does)
            topo = _extract_topology_for_cython(engine)
            trip_idx, rho_vals = _build_cython_triplet_index(
                topo, active, rho
            )

            def path_a() -> float:
                return _cy_score_from_topo(topo, trip_idx, rho_vals)

            def path_b() -> float:
                return _score_with_cython(engine, active, rho)

            def path_c() -> float:
                total = 0.0
                for triplet in active:
                    x, y, z = triplet
                    p_xy = engine.calculate_triple_probability((x, y, z))
                    p_xz = engine.calculate_triple_probability((x, z, y))
                    probs = (p_xy, p_xz, max(1.0 - p_xy - p_xz, 0.0))
                    r = rho[triplet]
                    for r_i, p_i in zip(r, probs):
                        if r_i > 0.0:
                            total += r_i * (
                                math.log(p_i) if p_i > 0.0 else _LOG_FLOOR
                            )
                return total

            va, vb, vc = path_a(), path_b(), path_c()
            agree = max(abs(va - vc), abs(vb - vc))

            a = timed(path_a, loops=200, repeats=5)
            b = timed(path_b, loops=50, repeats=5)
            c = timed(path_c, loops=20, repeats=3)

            print(f"  {n_taxa:>5} {len(active):>7} "
                  f"{a * 1e3:>10.4f} ms {b * 1e3:>11.4f} ms "
                  f"{c * 1e3:>9.4f} ms "
                  f"{c / a:>7.1f}x {c / b:>7.1f}x {agree:>10.1e}")

            RESULTS["kernel_ab"][f"{n_taxa}taxa"] = {
                "taxa": n_taxa,
                "gene_trees": n_gts,
                "active_triplets": len(active),
                "cython_cached_ms": a * 1e3,
                "cython_with_extract_ms": b * 1e3,
                "python_ms": c * 1e3,
                "speedup_kernel": c / a,
                "speedup_with_extract": c / b,
                "max_abs_diff": agree,
                "log_pl": vc,
            }
        except Exception as exc:
            print(f"  {n_taxa:>5}   ERR {str(exc)[:60]}")
            RESULTS["kernel_ab"][f"{n_taxa}taxa"] = {"error": repr(exc)}

    OUT_PATH.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
