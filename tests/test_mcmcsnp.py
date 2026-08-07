"""
Test suite for the SNP Likelihood algorithm (phynetpy.BiMarkers).

Includes:
    - A smoke test that scores the packaged NEXUS fixtures.
    - Scalability / stress tests with simulated data at various taxa and site
      counts.
    - Edge-case tests (gamma = 0.5 reticulations).

The entire class is currently **skipped** (``@pytest.mark.skip``) because the
underlying inference pipeline is still under active development.  Remove the
skip marker once the pipeline is stable.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

import os
import time

import pytest
from pathlib import Path

from phynetpy.criteria import Likelihood
from phynetpy.data import BiallelicMarkers
from phynetpy.infer import score
from phynetpy.IO import read_nexus
from phynetpy.models import MSC
from phynetpy.SNPSimulator import simulate, random_network


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
_TESTFILES_DIR = _PACKAGE_ROOT / "testfiles"
# Generated per-run output (not a fixture); lives alongside the other
# gitignored run artifacts under runs/.
_STRESS_TEST_DIR = _PROJECT_ROOT / "runs" / "mcmc_snp_stress_tests"

_nexus_path = _TESTFILES_DIR / "paper_net.nex"
_large_nexus_path = _TESTFILES_DIR / "paper_net_largeseq.nex"


def _score_nexus(path, *, u, v, coal, samples, **search):
    """Score the network in a NEXUS file against the SNP matrix beside it.

    These fixtures keep the network and its marker matrix in one file, so
    this splits them back onto the two arguments the ``score`` verb takes:
    the network to evaluate, and the data to evaluate it against.
    """
    net = read_nexus(str(path))[0]
    markers = BiallelicMarkers.from_file(str(path), samples=samples)
    return score(
        net, markers,
        model=MSC(u=u, v=v, coal=coal),
        criterion=Likelihood(),
        **search,
    )


# ---------------------------------------------------------------------------
# Individual test routines (called by the class below)
# ---------------------------------------------------------------------------

def _snp_likelihood_smoke():
    """Smoke test: score both the standard and large nexus fixtures."""
    for path in (_large_nexus_path, _nexus_path):
        print(_score_nexus(
            path.absolute(), u=1, v=1, coal=0.005,
            samples={"A": 2, "B": 2, "C": 2},
        ))
    return 1


def _snp_scalability_stress():
    """Scalability stress test.

    Generates random level-2 networks at 10, 25, and 50 taxa, simulates
    SNP data at 1 000, 2 000, and 10 000 sites, scores each combination,
    and reports timing.  This is NOT a pass/fail test.
    """
    os.makedirs(_STRESS_TEST_DIR, exist_ok=True)

    taxa_counts = [10, 25, 50]
    site_counts = [1000, 2000, 10000]
    lvl = 2
    seed = 42
    u, v, coal = 1.0, 1.0, 0.005

    results = []
    print("\n" + "=" * 72)
    print("  SNP LIKELIHOOD SCALABILITY TEST")
    print("  Level-2 networks | samples=1 per taxon | u=1, v=1, coal=0.005")
    print("=" * 72)

    for n_taxa in taxa_counts:
        print(f"\n--- Generating level-{lvl} network with {n_taxa} taxa ---")
        net = random_network(n=n_taxa, level=lvl, seed=seed + n_taxa)
        samples = {leaf.label: 1 for leaf in net.get_leaves()}

        for n_sites in site_counts:
            print(f"\n  >> {n_taxa} taxa, {n_sites} sites:")
            sim_seed = seed + n_taxa * 1000 + n_sites

            t0 = time.perf_counter()
            sim = simulate(
                n=n_taxa, s=n_sites, net=net,
                samples=samples, u=u, v=v, coal=coal, seed=sim_seed,
            )
            t_sim = time.perf_counter() - t0
            print(f"     Simulation: {t_sim:.3f}s")

            nex_file = str(
                _STRESS_TEST_DIR / f"stress_{n_taxa}taxa_{n_sites}sites_lvl{lvl}.nex"
            )
            sim.write_nexus(nex_file)
            print(f"     Written to: {nex_file}")

            try:
                t0 = time.perf_counter()
                log_lik = _score_nexus(
                    nex_file, u=u, v=v, coal=coal,
                    samples=samples, sequential=True,
                )
                t_lik = time.perf_counter() - t0
                print(f"     Likelihood: {log_lik:.6f}  Time: {t_lik:.3f}s")
                results.append((n_taxa, n_sites, log_lik, t_lik, None))
            except Exception as e:
                t_lik = time.perf_counter() - t0
                print(f"     ERROR after {t_lik:.3f}s: {e}")
                results.append((n_taxa, n_sites, None, t_lik, str(e)))

    # Summary table
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'Taxa':>6} {'Sites':>7} {'Log-Lik':>14} {'Time (s)':>10} {'Status':>10}")
    print("  " + "-" * 55)
    for n_taxa, n_sites, log_lik, t_lik, err in results:
        if err is None:
            print(f"  {n_taxa:>6} {n_sites:>7} {log_lik:>14.4f} {t_lik:>10.3f} {'OK':>10}")
        else:
            print(f"  {n_taxa:>6} {n_sites:>7} {'N/A':>14} {t_lik:>10.3f} {'FAIL':>10}")
    print("=" * 72)
    return 1


def _snp_gamma_half():
    """Verify likelihood computation when all reticulation gammas are 0.5."""
    n_taxa, n_sites = 10, 1000
    u, v, coal = 1.0, 1.0, 0.005

    net = random_network(n=n_taxa, level=1, gamma_range=(0.5, 0.5), seed=99)
    samples = {leaf.label: 1 for leaf in net.get_leaves()}

    sim = simulate(
        n=n_taxa, s=n_sites, net=net,
        samples=samples, u=u, v=v, coal=coal, seed=99,
    )

    os.makedirs(_STRESS_TEST_DIR, exist_ok=True)
    nex_file = str(_STRESS_TEST_DIR / "gamma_half_test.nex")
    sim.write_nexus(nex_file)

    try:
        result = _score_nexus(
            nex_file, u=u, v=v, coal=coal,
            samples=samples, sequential=True,
        )
        print(f"Gamma=0.5 test: log-likelihood = {result:.6f}")
        return 1
    except Exception as e:
        print(f"Gamma=0.5 test FAILED: {e}")
        return 0


# ---------------------------------------------------------------------------
# Pytest class (currently skipped)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="MCMC SNP pipeline is under active development")
class TestMCMC_SNP:
    """Aggregated SNP likelihood test runner.

    Remove the ``@pytest.mark.skip`` decorator once the inference pipeline
    is stable enough for CI.
    """

    def test_snp_suite(self) -> None:
        """Run the smoke, scalability, and gamma-half tests."""
        results = [_snp_likelihood_smoke(), _snp_scalability_stress(), _snp_gamma_half()]
        assert sum(results) == 3, f"SNP tests failed: {sum(results)}/3 passed"
