"""Integration cross-check of the MCMC_SEQ likelihood against PhyloNet itself.

This suite runs the compiled ``CrossCheck`` Java harness
(``tests/crosscheck/CrossCheck.java``) against the real PhyloNet jar + BEAGLE
and asserts that PhyNetPy's MSNC density and Felsenstein log-likelihood agree
with PhyloNet's own implementations on a battery of adversarial states (GTR,
multiple alleles, stacked reticulations, boundary population sizes, invalid
embeddings, ambiguity codes/gaps, saturation, constant sites, larger trees).

The whole module **auto-skips** unless the environment is set up:

* the PhyloNet jar exists (``PHYLONET_JAR`` env var, or the default path), and
* the BEAGLE native dir exists (``BEAGLE_DIR`` env var, or the default path).

When those are present, the fixture compiles ``CrossCheck.java`` for a Java 8
runtime (requires ``javac`` on PATH). Targeting Java 8 keeps the harness
compatible when the compiler is newer than the runtime, which is common on
Windows machines with both a JDK and a legacy Java installation. Because it
shells out to Java + BEAGLE this is an opt-in integration test rather than a
unit test.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# The runner module lives beside the Java harness; make it importable.
_CROSS = Path(__file__).resolve().parent / "crosscheck"
sys.path.insert(0, str(_CROSS))

import run_crosscheck as rc  # noqa: E402

_JAR = Path(rc.JAR)
_BEAGLE = Path(rc.BEAGLE_DIR)

pytestmark = pytest.mark.skipif(
    not (_JAR.exists() and _BEAGLE.exists()),
    reason=(
        "PhyloNet cross-check skipped: set PHYLONET_JAR and BEAGLE_DIR (or "
        "install to the default paths) to enable. "
        f"jar={_JAR} exists={_JAR.exists()}; "
        f"beagle={_BEAGLE} exists={_BEAGLE.exists()}"
    ),
)

# Beagle needs its plugin dir on PATH (not just java.library.path) on Windows.
os.environ["PATH"] = str(_BEAGLE) + os.pathsep + os.environ.get("PATH", "")


@pytest.fixture(scope="module")
def java_results(tmp_path_factory):
    """Compile for Java 8 and run the harness once; return its results."""
    cls = _CROSS / "CrossCheck.class"
    javac = shutil.which("javac")
    if javac is None:
        if not cls.exists():
            pytest.skip("CrossCheck.class missing and javac not on PATH")
    else:
        subprocess.run(
            [
                javac,
                "-source",
                "8",
                "-target",
                "8",
                "-cp",
                str(_JAR),
                "-d",
                str(_CROSS),
                str(_CROSS / "CrossCheck.java"),
            ],
            check=True,
        )
    spec = tmp_path_factory.mktemp("crosscheck") / "cases.spec"
    rc.write_spec(spec)
    return rc.run_java(spec)


def _param_ids():
    params = []
    for c in rc.CASES:
        params.append((c["name"], "MSNC"))
        if c.get("seqs"):
            params.append((c["name"], "FELSEN"))
    return params


_PARAMS = _param_ids()


@pytest.mark.parametrize(
    "case_name,factor",
    _PARAMS,
    ids=[f"{n}-{f}" for n, f in _PARAMS],
)
def test_phylonet_parity(java_results, case_name, factor):
    """PhyNetPy must match PhyloNet's value for this (case, factor)."""
    case = next(c for c in rc.CASES if c["name"] == case_name)
    jv = java_results.get((case_name, factor))
    assert jv is not None, f"no PhyloNet result for {case_name}/{factor}"
    assert not isinstance(jv, str), f"PhyloNet harness error: {jv}"

    pv = rc.python_values(case)[factor]

    if math.isinf(jv) or math.isinf(pv):
        assert jv == pv, f"{case_name}/{factor}: java={jv} py={pv}"
    else:
        assert abs(jv - pv) < 1e-5, (
            f"{case_name}/{factor}: java={jv:.10f} py={pv:.10f} "
            f"diff={abs(jv - pv):.2e}"
        )
