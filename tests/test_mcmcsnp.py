"""
Test suite for the SNP Likelihood algorithm (phynetpy.BiMarkers).

Includes:
    - A smoke test that scores the packaged NEXUS fixtures.
    - Scalability / stress tests with simulated data at various taxa and site
      counts.
    - Edge-case tests (gamma = 0.5 reticulations).

Finite-likelihood fixture checks run in normal CI. Scalability and larger
reticulation simulations remain marked ``slow``.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

import math
import os
import time
from types import SimpleNamespace

import pytest
from pathlib import Path

import phynetpy.BiMarkers as biomarkers
from phynetpy.criteria import Likelihood
from phynetpy.data import BiallelicMarkers
from phynetpy.infer import score
from phynetpy.IO import read_nexus
from phynetpy.models import BranchLengthUnit, MSC
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


def _score_nexus(path, *, u, v, theta, samples, **search):
    """Score the network in a NEXUS file against the SNP matrix beside it.

    These fixtures keep the network and its marker matrix in one file, so
    this splits them back onto the two arguments the ``score`` verb takes:
    the network to evaluate, and the data to evaluate it against.
    """
    net = read_nexus(str(path))[0]
    net.set_branch_length_unit(BranchLengthUnit.SUBSTITUTIONS_PER_SITE)
    markers = BiallelicMarkers.from_file(str(path), samples=samples)
    return score(
        net, markers,
        model=MSC(theta=theta, u=u, v=v),
        criterion=Likelihood(),
        **search,
    )


def _snp_scalability_stress():
    """Scalability stress test.

    The default release gate exercises 10-taxon data through 10,000 sites and
    a 25-taxon/1,000-site level-2 network. Set
    ``PHYNETPY_EXTREME_SNP_STRESS=1`` to add the resource-dependent 25/50-taxon
    cases used for hardware benchmarking.
    """
    os.makedirs(_STRESS_TEST_DIR, exist_ok=True)

    stress_grid = {
        10: [1000, 2000, 10000],
        25: [1000],
    }
    if os.environ.get("PHYNETPY_EXTREME_SNP_STRESS") == "1":
        stress_grid[25] = [1000, 2000, 10000]
        stress_grid[50] = [1000, 2000, 10000]
    lvl = 2
    seed = 42
    u, v, theta = 1.0, 1.0, 0.005

    results = []
    print("\n" + "=" * 72)
    print("  SNP LIKELIHOOD SCALABILITY TEST")
    print("  Level-2 networks | samples=1 per taxon | u=1, v=1, theta=0.005")
    print("=" * 72)

    for n_taxa, site_counts in stress_grid.items():
        print(f"\n--- Generating level-{lvl} network with {n_taxa} taxa ---")
        net = random_network(n=n_taxa, level=lvl, seed=seed + n_taxa)
        samples = {leaf.label: 1 for leaf in net.get_leaves()}

        for n_sites in site_counts:
            print(f"\n  >> {n_taxa} taxa, {n_sites} sites:")
            sim_seed = seed + n_taxa * 1000 + n_sites

            t0 = time.perf_counter()
            sim = simulate(
                n=n_taxa, s=n_sites, net=net,
                samples=samples, u=u, v=v, theta=theta, seed=sim_seed,
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
                    nex_file, u=u, v=v, theta=theta,
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
    return results


def _snp_gamma_half():
    """Verify likelihood computation when all reticulation gammas are 0.5."""
    n_taxa, n_sites = 10, 1000
    u, v, theta = 1.0, 1.0, 0.005

    net = random_network(n=n_taxa, level=1, gamma_range=(0.5, 0.5), seed=99)
    samples = {leaf.label: 1 for leaf in net.get_leaves()}

    sim = simulate(
        n=n_taxa, s=n_sites, net=net,
        samples=samples, u=u, v=v, theta=theta, seed=99,
    )

    os.makedirs(_STRESS_TEST_DIR, exist_ok=True)
    nex_file = str(_STRESS_TEST_DIR / "gamma_half_test.nex")
    sim.write_nexus(nex_file)

    return _score_nexus(
        nex_file, u=u, v=v, theta=theta,
        samples=samples, sequential=True,
    )


# ---------------------------------------------------------------------------
# Pytest class
# ---------------------------------------------------------------------------

def test_batch_size_rejects_unallocatable_single_site(monkeypatch) -> None:
    """The resource guard must run before coercing a zero batch to one."""
    monkeypatch.setattr(
        biomarkers,
        "_estimate_peak_vpi_memory",
        lambda model, samples, n_sites: (800, [1, 100]),
    )
    monkeypatch.setattr(biomarkers, "_compute_network_level", lambda model: 2)
    monkeypatch.setattr(
        biomarkers,
        "_compute_max_lineages",
        lambda model, samples: 10,
    )
    fake_device = SimpleNamespace(mem_info=(1_000, 1_000))
    monkeypatch.setattr(
        biomarkers,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(Device=lambda index: fake_device),
        ),
    )
    monkeypatch.setattr(
        biomarkers,
        "GPU_SPECS",
        SimpleNamespace(available=True),
    )

    with pytest.raises(biomarkers.SNPResourceError, match="single site"):
        model = SimpleNamespace(nodetypes={"leaf": []})
        biomarkers._compute_batch_size(model, {}, 10, use_gpu=True)


class TestBiMarkersLikelihood:
    """Marker likelihood smoke tests plus opt-in stress coverage."""

    @pytest.mark.parametrize("path", [_large_nexus_path, _nexus_path])
    def test_fixture_likelihood_is_finite(self, path) -> None:
        value = _score_nexus(
            path.absolute(), u=1.0, v=1.0, theta=0.005,
            samples={"A": 2, "B": 2, "C": 2},
        )
        assert math.isfinite(value)

    @pytest.mark.slow
    def test_scalability_stress(self) -> None:
        results = _snp_scalability_stress()
        unexpected = [
            error
            for *_, error in results
            if error is not None
            and "GPU memory was exhausted" not in error
            and "available memory" not in error
            and "too complex for available hardware" not in error
        ]
        assert any(error is None for *_, error in results)
        assert not unexpected

    @pytest.mark.slow
    def test_gamma_half(self) -> None:
        assert math.isfinite(_snp_gamma_half())
