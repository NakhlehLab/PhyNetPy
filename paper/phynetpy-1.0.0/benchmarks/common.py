"""
Shared timing and provenance helpers for the PhyNetPy paper benchmarks.

Every benchmark in this directory writes a JSON file whose ``meta`` block
records enough of the environment to reproduce the measurement: package
versions, interpreter, OS, CPU, and the timing protocol itself.  The
manuscript's tables and figures are generated from those JSON files by
``make_artifacts.py``, so no number in the paper is typed by hand.

Timing protocol
---------------
``timed`` runs a warm-up call, then ``repeats`` batches of ``loops`` calls
each, and reports the *median* batch.  The median (not the minimum) is
reported in the paper because it is the more honest summary of what a user
experiences; the minimum is retained in the JSON for reference.

Startup cost is deliberately excluded from these microbenchmarks -- the
warm-up call pays for import, JIT-free Cython module loading, and first-touch
allocation.  Benchmarks where startup *is* part of the measured quantity
(notably the PhyloNet comparison, which pays JVM startup per invocation) say
so explicitly in their own metadata.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

BENCH_DIR = Path(__file__).resolve().parent


def environment() -> dict[str, Any]:
    """Provenance block written into every benchmark JSON."""
    import numpy
    import scipy

    import phynetpy

    try:
        import networkx

        nx_version = networkx.__version__
    except Exception:  # pragma: no cover
        nx_version = None

    return {
        "phynetpy_version": phynetpy.__version__,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "networkx": nx_version,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "timing": {
            "clock": "time.perf_counter",
            "statistic": "median of repeats",
            "warmup_calls": 1,
            "startup_included": False,
            "parallelism": "single process, single thread unless noted",
        },
    }


def timed(
    fn: Callable[[], Any], loops: int, repeats: int = 5
) -> dict[str, float]:
    """Time ``fn``: ``repeats`` batches of ``loops`` calls, median batch.

    Args:
        fn: Zero-argument callable to time.
        loops: Calls per batch.
        repeats: Number of batches.

    Returns:
        Per-call statistics in microseconds, plus the raw loop and repeat
        counts so the measurement can be audited.
    """
    fn()  # warm caches and first-touch allocation
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(loops):
            fn()
        samples.append(time.perf_counter() - t0)
    median = statistics.median(samples)
    return {
        "loops": loops,
        "repeats": repeats,
        "median_total_s": median,
        "per_call_us": median / loops * 1e6,
        "best_per_call_us": min(samples) / loops * 1e6,
        "worst_per_call_us": max(samples) / loops * 1e6,
    }


def banner(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def row(label: str, *cols: str) -> None:
    print(f"  {label:<34}" + "".join(f"{c:>13}" for c in cols))


def write(results: dict[str, Any], name: str) -> Path:
    """Write ``results`` to ``<benchmarks>/<name>.json`` and return the path."""
    out = BENCH_DIR / f"{name}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return out


def load(name: str) -> dict[str, Any]:
    """Read back a benchmark JSON by bare name."""
    return json.loads((BENCH_DIR / f"{name}.json").read_text(encoding="utf-8"))
