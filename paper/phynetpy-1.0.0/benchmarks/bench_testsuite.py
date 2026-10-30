"""
Collect test-suite and continuous-integration facts for the validation table.

Runs the suite the way the project's own CI runs it, parses pytest's summary
line, and reads the CI matrix out of the workflow file rather than restating it
from memory.  This is what keeps Table 3 honest between revisions: re-running
this script is the only way the numbers change.

Also records the per-area subtotals the manuscript quotes (substitution models,
network comparison, the PhyloNet cross-check) by collecting those files
individually.

Run: python paper/phynetpy-1.0.0/benchmarks/bench_testsuite.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import banner, environment, write

REPO = Path(__file__).resolve().parents[3]

# Per-area subtotals the manuscript refers to by name.
AREAS = {
    "substitution_models": ["tests/test_gtr.py"],
    "network_comparison": [
        "tests/test_network_distances.py",
        "tests/test_reticulation_comparison.py",
        "tests/test_blobs_and_subnet.py",
    ],
    "phylonet_crosscheck": ["tests/test_crosscheck_phylonet.py"],
    "network_core": ["tests/test_network.py", "tests/test_network_moves.py"],
    "inference_api": ["tests/test_infer_api.py"],
    "memory_and_mutation_cost": ["tests/test_memory.py"],
}

SUMMARY = re.compile(
    r"(?:(\d+) passed)?(?:, )?(?:(\d+) failed)?(?:, )?(?:(\d+) skipped)?"
    r"(?:, )?(?:(\d+) deselected)?(?:, )?(?:(\d+) error)?"
)


def pytest(args: list[str]) -> tuple[str, float]:
    """Run pytest with ``args``; return (last summary line, wall seconds)."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q", "--no-header"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    tail = ""
    for line in reversed(lines):
        if "passed" in line or "collected" in line or "error" in line:
            tail = line.strip()
            break
    return tail, elapsed


def parse(summary: str) -> dict[str, int]:
    """Pull the counts out of a pytest summary line."""
    out = {"passed": 0, "failed": 0, "skipped": 0, "deselected": 0, "errors": 0}
    for key, pattern in (
        ("passed", r"(\d+) passed"),
        ("failed", r"(\d+) failed"),
        ("skipped", r"(\d+) skipped"),
        ("deselected", r"(\d+) deselected"),
        ("errors", r"(\d+) error"),
    ):
        match = re.search(pattern, summary)
        if match:
            out[key] = int(match.group(1))
    return out


def collected(paths: list[str]) -> int:
    """Number of tests pytest collects from ``paths``."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q", "--collect-only"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) tests? collected", proc.stdout)
    return int(match.group(1)) if match else -1


def ci_matrix() -> dict[str, Any]:
    """Read the CI configuration rather than restating it."""
    workflow = REPO / ".github" / "workflows" / "ci.yml"
    text = workflow.read_text(encoding="utf-8")
    versions = re.search(r"python-version:\s*\[([^\]]+)\]", text)
    # Only the keys nested under the top-level ``jobs:`` mapping are jobs;
    # ``on:`` has same-indentation children that are triggers, not jobs.
    jobs_block = text.split("\njobs:", 1)[1] if "\njobs:" in text else ""
    return {
        "workflow": str(workflow.relative_to(REPO)).replace("\\", "/"),
        "python_versions": (
            [v.strip().strip('"') for v in versions.group(1).split(",")]
            if versions
            else []
        ),
        "jobs": re.findall(r"^  (\w+):$", jobs_block, flags=re.MULTILINE),
        "runs_on": sorted(set(re.findall(r"runs-on:\s*(\S+)", text))),
        "windows_smoke": "windows-latest" in text,
        "packaging_checks": [
            name
            for name in ("check_manifest", "build", "twine check")
            if name.replace("_", "-") in text or name in text
        ],
        "slow_suite_opt_in": "workflow_dispatch" in text and "run_slow" in text,
    }


def main() -> None:
    results: dict[str, Any] = {"meta": environment()}
    results["meta"]["measurement"] = {
        "quantity": "test counts and wall time for the default suite",
        "command": 'pytest -m "not slow" -q',
        "note": (
            "The PhyloNet cross-check module skips unless a PhyloNet jar and "
            "BEAGLE are installed, so the skipped count is environment "
            "dependent. This run had both present."
        ),
    }

    banner("Test suite")

    total = collected([])
    print(f"  collected (all markers)      {total}")

    summary, elapsed = pytest(["-m", "not slow"])
    default = parse(summary)
    print(f"  default suite                {summary}")
    print(f"  wall time                    {elapsed:.1f} s")

    results["suite"] = {
        "collected_total": total,
        "default_run": default,
        "default_summary_line": summary,
        "wall_seconds": elapsed,
    }

    banner("Per-area subtotals")
    areas: dict[str, Any] = {}
    for name, paths in AREAS.items():
        count = collected(paths)
        areas[name] = {"files": paths, "collected": count}
        print(f"  {name:<28}{count:>5}")
    results["areas"] = areas

    banner("Continuous integration")
    ci = ci_matrix()
    results["ci"] = ci
    for key, value in ci.items():
        print(f"  {key:<22}{value}")

    write(results, "testsuite")


if __name__ == "__main__":
    main()
