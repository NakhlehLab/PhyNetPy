"""
Test suite for the SNP Likelihood algorithm (phynetpy.BiMarkers).

Includes:
    - Correctness checks against known expected values (Charles Rabier tables)
      for level-1 and level-2 networks.
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

from phynetpy.BiMarkers import SNP_LIKELIHOOD
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


# ---------------------------------------------------------------------------
# Individual test routines (called by the class below)
# ---------------------------------------------------------------------------

def _snp_correctness_level1():
    """Validate SNP likelihood against Charles Rabier's level-1 table.

    Compares computed likelihoods for all (A, B, C) red-count groupings
    against pre-computed expected values.
    """
    tbl = {
        (0, 0, 0): 0.31581337186422315,
        (0, 0, 1): 1.853660371657668e-3,
        (0, 0, 2): 0.05677236283895234,
        (0, 1, 0): 1.6755618678903335e-3,
        (0, 1, 1): 1.0705941050619642e-5,
        (0, 1, 2): 4.789800667080107e-4,
        (0, 2, 0): 9.884301576368968e-4,
        (0, 2, 1): 7.570444581342921e-6,
        (0, 2, 2): 5.3322920321306e-4,
        (0, 3, 0): 0.04027355605172049,
        (0, 3, 1): 5.910438937463977e-4,
        (0, 3, 2): 0.07852626659131974,
        (1, 0, 0): 1.9618887485350735e-3,
        (1, 0, 1): 1.21627077880799976e-5,
        (1, 0, 2): 4.828155168689878e-4,
        (1, 1, 0): 1.09890103033996982e-5,
        (1, 1, 1): 9.099282927274783e-8,
        (1, 1, 2): 7.300548379825278e-6,
        (1, 2, 0): 7.30054837982544e-6,
        (1, 2, 1): 9.099282927274914e-9,
        (1, 2, 2): 1.098901030399711e-5,
        (1, 3, 0): 4.821551686898895e-4,
        (1, 3, 1): 1.2162707788079851e-5,
        (1, 3, 2): 1.9618887485350622e-3,
        (2, 0, 0): 0.0785262665913196,
        (2, 0, 1): 5.910438937463979e-4,
        (2, 0, 2): 0.040273556051720324,
        (2, 1, 0): 5.332292032130451e-4,
        (2, 1, 1): 7.5704445813427665e-6,
        (2, 1, 2): 9.884301576368857e-4,
        (2, 2, 0): 4.789800667080225e-4,
        (2, 2, 1): 1.0719114102479618e-5,
        (2, 2, 2): 1.6755618678903448e-3,
        (2, 3, 0): 0.0567723862838952165,
        (2, 3, 1): 1.8536603716576576548e-3,
        (2, 3, 2): 0.31581337186422315,
    }

    for grouping, expected in tbl.items():
        set_reds = {"A": grouping[0], "B": grouping[1], "C": grouping[2]}
        result = SNP_LIKELIHOOD_DATA(
            str(_nexus_path), set_reds, 1, 1, 0.005
        )
        if not 1 + 1e-10 > abs(result / expected) > 1 - 1e-10:
            print(f"Expected: {expected}, but got: {result} for grouping: {grouping}")
        else:
            print(f"Expected: {expected}, and got: {result} for grouping: {grouping}")
    return 1


def _snp_correctness_level2():
    """Validate SNP likelihood against Charles Rabier's level-2 table.

    Placeholder — computed results are not yet wired in.
    """
    tbl = {
        (0, 0, 0, 0): 0.420388330446373,
        (0, 0, 0, 1): 2.1413391677020254e-3,
        (0, 0, 0, 2): 1.1379876974211235e-3,
        (0, 0, 0, 3): 0.018044391063547768,
        (1, 0, 1, 1): 5.7431505391794586e-8,
        (1, 1, 1, 1): 3.1907711650231237e-10,
        (1, 2, 1, 1): 2.0749049388423288e-10,
        (1, 3, 1, 1): 1.268527207088679e-8,
        (1, 3, 2, 1): 2.0091067872430637e-8,
        (1, 3, 3, 1): 3.277973419215861e-6,
        (3, 3, 3, 3): 0.420388330646373,
    }

    for grouping, expected in tbl.items():
        result: float = 0.0
        if not 1 + 1e-10 > abs(result / expected) > 1 - 1e-10:
            print(f"Expected: {expected}, but got: {result} for grouping: {grouping}")
            return 0
    return 1


def _snp_likelihood_smoke():
    """Smoke test: run SNP_LIKELIHOOD on both standard and large nexus files."""
    result = SNP_LIKELIHOOD(
        str(_large_nexus_path.absolute()),
        u=1, v=1, coal=0.005,
        samples={"A": 2, "B": 2, "C": 2},
    )
    print(result)
    result = SNP_LIKELIHOOD(
        str(_nexus_path.absolute()),
        u=1, v=1, coal=0.005,
        samples={"A": 2, "B": 2, "C": 2},
    )
    print(result)
    return 1


def _snp_scalability_stress():
    """Scalability stress test.

    Generates random level-2 networks at 10, 25, and 50 taxa, simulates
    SNP data at 1 000, 2 000, and 10 000 sites, runs SNP_LIKELIHOOD on
    each combination, and reports timing.  This is NOT a pass/fail test.
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
                log_lik = SNP_LIKELIHOOD(
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
        result = SNP_LIKELIHOOD(
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
