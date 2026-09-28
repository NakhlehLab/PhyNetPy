"""
Reproduce every number in the manuscript.

Runs each benchmark in turn and then regenerates the tables and figures.  The
PhyloNet head-to-head is skipped with a warning if no jar is configured, since
it is the only step that needs software outside this repository.

Run::

    set PHYLONET_JAR=...\\phylonet.jar
    python paper/phynetpy-1.0.0/benchmarks/run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = (
    ("bench_core.py", "core numerical and data-structure benchmarks", True),
    ("bench_kernels.py", "compiled-kernel A/B", True),
    ("bench_snp.py", "biallelic-marker likelihood timing", True),
    ("bench_testsuite.py", "test suite and CI facts", True),
    ("bench_phylonet.py", "PhyloNet head-to-head", False),
    ("make_artifacts.py", "tables and figures", True),
)


def main() -> int:
    failures: list[str] = []
    for script, description, always in STEPS:
        if script == "bench_phylonet.py" and not os.environ.get("PHYLONET_JAR"):
            print(
                f"\n=== SKIP {script} ({description}) ===\n"
                "    PHYLONET_JAR is not set; the head-to-head comparison "
                "needs a PhyloNet jar.",
                file=sys.stderr,
            )
            continue
        print(f"\n=== {script} ({description}) ===", flush=True)
        proc = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
        if proc.returncode != 0:
            failures.append(script)
            if always:
                print(f"    {script} failed", file=sys.stderr)

    if failures:
        print(f"\nFailed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll benchmarks complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
