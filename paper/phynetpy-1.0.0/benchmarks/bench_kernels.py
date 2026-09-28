"""
Compiled-kernel A/B: the Cython pseudo-likelihood engine against the
pure-Python reference dynamic program.

``phynetpy._mpl`` keeps both implementations of the triplet dynamic program and
selects between them on the module flag ``_HAS_CYTHON_MPL``.  Toggling that flag
gives a controlled comparison: identical input, identical search path, only the
inner DP changes.  Each configuration also checks that the two paths return the
same log pseudo-likelihood, so the speed measurement carries a correctness
assertion with it.

Reaching into a private flag is deliberate.  The alternative -- uninstalling the
compiled extension -- would change more than the one variable under test.

Run: python paper/phynetpy-1.0.0/benchmarks/bench_kernels.py
"""

from __future__ import annotations

from typing import Any

from common import banner, environment, timed, write

RESULTS: dict[str, Any] = {"meta": {}, "mpl_engine": {}}

CONFIGURATIONS = ((6, 40), (8, 60), (10, 80), (12, 100))
SEED = 11
# Matches bench_phylonet.py: theta = 0.5 leaves roughly half the gene trees
# discordant with the species topology, so the dynamic program is exercised on
# a realistic mix rather than on identical inputs.
THETA = 0.5


def main() -> None:
    from phynetpy import _mpl
    from phynetpy.criteria import PseudoLikelihood
    from phynetpy.infer import score, simulate
    from phynetpy.models import MSC

    RESULTS["meta"] = environment()
    RESULTS["meta"]["measurement"] = {
        "quantity": "log pseudo-likelihood of a fixed network (triplet DP)",
        "toggle": "phynetpy._mpl._HAS_CYTHON_MPL",
        "seed": SEED,
        "model": f"MSC(theta={THETA})",
        "cython_available": bool(_mpl._HAS_CYTHON_MPL),
    }

    if not _mpl._HAS_CYTHON_MPL:
        print("Compiled MPL engine unavailable; nothing to compare.")
        RESULTS["mpl_engine"]["error"] = "compiled MPL engine unavailable"
        write(RESULTS, "kernels")
        return

    banner("Pseudo-likelihood triplet DP: compiled kernel vs pure Python")
    print(
        f"  {'taxa':>5} {'genes':>7} {'Cython (ms)':>13} {'Python (ms)':>13}"
        f" {'speedup':>9} {'|diff|':>11}"
    )
    print("  " + "-" * 64)

    for n_taxa, n_gts in CONFIGURATIONS:
        gts = simulate(
            MSC(theta=THETA), taxa=n_taxa, n=n_gts, data="gene_trees", seed=SEED
        )
        net = gts.true_network
        crit = PseudoLikelihood()

        def call() -> float:
            return score(net, gts, model=MSC(theta=THETA), criterion=crit)

        _mpl._HAS_CYTHON_MPL = True
        cy_value = call()
        cy_stats = timed(call, loops=5, repeats=3)

        _mpl._HAS_CYTHON_MPL = False
        py_value = call()
        py_stats = timed(call, loops=3, repeats=3)

        _mpl._HAS_CYTHON_MPL = True

        cy_ms = cy_stats["per_call_us"] / 1000
        py_ms = py_stats["per_call_us"] / 1000
        diff = abs(cy_value - py_value)

        RESULTS["mpl_engine"][f"{n_taxa}taxa_{n_gts}genes"] = {
            "taxa": n_taxa,
            "gene_trees": n_gts,
            "cython_ms": cy_ms,
            "python_ms": py_ms,
            "speedup": py_ms / cy_ms,
            "log_pl_cython": cy_value,
            "log_pl_python": py_value,
            "abs_diff": diff,
            "protocol": {"cython": cy_stats, "python": py_stats},
        }
        print(
            f"  {n_taxa:>5} {n_gts:>7} {cy_ms:>13.3f} {py_ms:>13.3f}"
            f" {py_ms / cy_ms:>8.2f}x {diff:>11.2e}"
        )

    write(RESULTS, "kernels")


if __name__ == "__main__":
    main()
