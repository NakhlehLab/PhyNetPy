"""Tests for :mod:`phynetpy._chain_analysis` (MCMC post-analysis / Tracer interop).

Coverage:

* ``TestDiagnostics`` -- ESS / ACT / standard-error-of-mean validated against
  closed-form AR(1) theory and the i.i.d. limit (the same numbers Tracer
  reports), plus constant-trace edge handling.
* ``TestHPD`` -- the HPD interval is the narrowest interval with the requested
  coverage.
* ``TestLogIO`` -- ``write_tracer_log`` / ``read_tracer_log`` round-trip, and
  the file is in the tab-delimited ``state``-first format Tracer expects.
* ``TestTreesIO`` -- the NEXUS ``.trees`` writer emits one tree per sample.
* ``TestResultIntegration`` -- ``MCMCSeqResult`` / ``MCMCGTResult`` expose
  working ``trace_table`` / ``write_log`` / ``summary`` / ``write_networks``.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phynetpy.Network import Network
from phynetpy._chain_analysis import (
    ParameterSummary,
    ChainSummary,
    autocorrelation_time,
    effective_sample_size,
    gelman_rubin,
    geweke,
    hpd_interval,
    read_tracer_log,
    standard_error_of_mean,
    write_tracer_log,
    write_trees_nexus,
)


def _ar1(n: int, phi: float, seed: int = 0) -> list[float]:
    """A stationary AR(1) series x_t = phi x_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    x = 0.0
    out = []
    # discard a burn-in so the series starts at stationarity
    for _ in range(2000):
        x = phi * x + rng.standard_normal()
    for _ in range(n):
        x = phi * x + rng.standard_normal()
        out.append(x)
    return out


# ═══════════════════════════════════════════════════════════════════════
class TestDiagnostics:
    def test_iid_ess_near_n(self):
        rng = np.random.default_rng(1)
        x = list(rng.standard_normal(8000))
        ess = effective_sample_size(x)
        # An independent chain has ESS approximately equal to N.
        assert 0.9 * len(x) <= ess <= 1.1 * len(x)

    @pytest.mark.parametrize("phi", [0.5, 0.8, 0.95])
    def test_ar1_ess_matches_theory(self, phi):
        n = 20000
        x = _ar1(n, phi, seed=7)
        ess = effective_sample_size(x)
        # ESS_theory = N * (1 - phi) / (1 + phi)
        expected = n * (1.0 - phi) / (1.0 + phi)
        assert ess == pytest.approx(expected, rel=0.25)

    @pytest.mark.parametrize("phi", [0.5, 0.8])
    def test_ar1_act_matches_theory(self, phi):
        x = _ar1(20000, phi, seed=11)
        act = autocorrelation_time(x, step_size=1)
        # ACT_theory (step_size 1) = (1 + phi) / (1 - phi)
        expected = (1.0 + phi) / (1.0 - phi)
        assert act == pytest.approx(expected, rel=0.3)

    def test_act_scales_with_step_size(self):
        x = _ar1(8000, 0.8, seed=3)
        assert autocorrelation_time(x, 10) == pytest.approx(
            10 * autocorrelation_time(x, 1)
        )

    def test_sem_is_autocorrelation_corrected(self):
        # For a correlated chain the corrected SEM exceeds the naive one.
        x = np.asarray(_ar1(8000, 0.9, seed=5))
        naive = float(x.std(ddof=1) / math.sqrt(x.size))
        assert standard_error_of_mean(x) > 1.5 * naive

    def test_constant_trace(self):
        x = [3.0] * 100
        assert math.isnan(effective_sample_size(x))
        assert math.isnan(autocorrelation_time(x))
        assert standard_error_of_mean(x) == 0.0

    def test_geweke_converged_chain_small_z(self):
        rng = np.random.default_rng(2)
        x = list(rng.standard_normal(10000))
        assert abs(geweke(x)) < 2.0

    def test_geweke_detects_drift(self):
        # A linear trend (non-stationary) should trip the diagnostic.
        x = list(np.linspace(0.0, 50.0, 5000) + np.random.default_rng(0).standard_normal(5000))
        assert abs(geweke(x)) > 2.0


# ═══════════════════════════════════════════════════════════════════════
class TestGelmanRubin:
    """Multi-chain R-hat: ~1 for converged chains, >>1 for disagreement."""

    def test_single_chain_is_nan(self):
        assert math.isnan(gelman_rubin([list(range(100))]))

    def test_converged_chains_near_one(self):
        rng = np.random.default_rng(7)
        # Four independent chains from the SAME stationary distribution.
        chains = [list(rng.standard_normal(5000)) for _ in range(4)]
        rhat = gelman_rubin(chains)
        assert rhat == pytest.approx(1.0, abs=0.05)

    def test_disagreeing_chains_large_rhat(self):
        rng = np.random.default_rng(0)
        # Chains stuck in different basins (different means) -> R-hat >> 1,
        # the signature of a network sampler trapped in distinct topology
        # modes.
        chains = [
            list(rng.standard_normal(2000) + offset)
            for offset in (-10.0, 0.0, 10.0, 20.0)
        ]
        assert gelman_rubin(chains) > 1.2

    def test_unequal_lengths_use_common_minimum(self):
        rng = np.random.default_rng(3)
        chains = [
            list(rng.standard_normal(3000)),
            list(rng.standard_normal(1500)),
        ]
        rhat = gelman_rubin(chains)
        assert not math.isnan(rhat) and rhat > 0.0


# ═══════════════════════════════════════════════════════════════════════
class TestHPD:
    def test_single_sample(self):
        assert hpd_interval([4.2]) == (4.2, 4.2)

    def test_full_coverage(self):
        lo, hi = hpd_interval([1.0, 2.0, 3.0, 4.0], prob=1.0)
        assert (lo, hi) == (1.0, 4.0)

    def test_narrowest_interval(self):
        # 20 tightly-clustered points near 0 plus 5 far outliers (both signs):
        # the 80% HPD (= 20 of 25 points) must sit on the cluster, not span a
        # symmetric quantile interval that swallows an outlier.
        cluster = list(np.linspace(-0.05, 0.05, 20))
        outliers = [50.0, 80.0, 100.0, -60.0, -90.0]
        lo, hi = hpd_interval(cluster + outliers, prob=0.8)
        assert hi - lo < 1.0
        assert lo <= 0.0 <= hi

    def test_invalid_prob(self):
        with pytest.raises(ValueError):
            hpd_interval([1.0, 2.0], prob=0.0)


# ═══════════════════════════════════════════════════════════════════════
class TestLogIO:
    def test_round_trip(self, tmp_path):
        states = [0, 100, 200, 300]
        traces = {
            "posterior": [-10.5, -9.2, -9.8, -8.1],
            "likelihood": [-12.0, -11.0, -11.5, -10.0],
            "theta": [0.02, 0.021, 0.019, 0.022],
        }
        path = tmp_path / "chain.log"
        write_tracer_log(states, traces, str(path), comments=["unit test"])
        got_states, got_traces = read_tracer_log(str(path))
        assert got_states == states
        assert got_traces.keys() == traces.keys()
        for k in traces:
            assert got_traces[k] == pytest.approx(traces[k])

    def test_header_format(self, tmp_path):
        path = tmp_path / "c.log"
        write_tracer_log([0, 1], {"posterior": [-1.0, -2.0]}, str(path))
        lines = path.read_text().splitlines()
        header = [ln for ln in lines if not ln.startswith("#")][0]
        assert header.split("\t") == ["state", "posterior"]

    def test_length_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError):
            write_tracer_log([0, 1], {"x": [1.0]}, str(tmp_path / "bad.log"))

    def test_nan_round_trips(self, tmp_path):
        path = tmp_path / "n.log"
        write_tracer_log([0, 1], {"x": [1.0, float("nan")]}, str(path))
        _, traces = read_tracer_log(str(path))
        assert traces["x"][0] == 1.0
        assert math.isnan(traces["x"][1])


# ═══════════════════════════════════════════════════════════════════════
class TestTreesIO:
    def test_one_tree_per_sample(self, tmp_path):
        states = [0, 100, 200]
        newicks = [
            "((A:1,B:1):1,C:2);",
            "((A:1,C:1):1,B:2);",
            "((B:1,C:1):1,A:2)",  # missing ';' should be repaired
        ]
        path = tmp_path / "out.trees"
        write_trees_nexus(states, newicks, str(path), taxa=["A", "B", "C"])
        text = path.read_text()
        assert text.startswith("#NEXUS")
        assert text.count("TREE STATE_") == 3
        assert "BEGIN TREES;" in text and "END;" in text
        # the repaired tree must end with a semicolon
        assert "STATE_200 = ((B:1,C:1):1,A:2);" in text


# ═══════════════════════════════════════════════════════════════════════
class TestResultIntegration:
    def _make_seq_result(self):
        from phynetpy.infer import MCMCSeqResult, MCMCSeqSample

        net = Network.from_newick("((A:0.01,B:0.01)I1:0.02,C:0.03)R;")
        rng = np.random.default_rng(0)
        samples = []
        for i in range(300):
            it = i * 50
            samples.append(
                MCMCSeqSample(
                    iteration=it,
                    log_posterior=float(-100 + rng.standard_normal()),
                    log_likelihood=float(-110 + rng.standard_normal()),
                    network_newick="((A:0.01,B:0.01)I1:0.02,C:0.03)R;",
                    theta=float(0.02 + 0.001 * rng.standard_normal()),
                    num_reticulations=0,
                )
            )
        return MCMCSeqResult(
            map_network=net,
            map_log_posterior=-95.0,
            map_theta=0.02,
            samples=samples,
            acceptance_rate=0.3,
            num_iterations=20000,
        )

    def test_trace_table_columns(self):
        res = self._make_seq_result()
        states, traces = res.trace_table()
        assert len(states) == 300
        assert set(traces) == {
            "posterior",
            "likelihood",
            "prior",
            "theta",
            "reticulationCount",
        }
        # prior == posterior - likelihood
        for p, l, pr in zip(
            traces["posterior"], traces["likelihood"], traces["prior"]
        ):
            assert pr == pytest.approx(p - l)

    def test_summary_and_step(self):
        res = self._make_seq_result()
        summary = res.summary()
        assert isinstance(summary, ChainSummary)
        assert summary.step_size == 50
        assert "theta" in summary.parameters
        ps = summary["theta"]
        assert isinstance(ps, ParameterSummary)
        assert ps.lower_hpd <= ps.mean <= ps.upper_hpd
        assert ps.ess > 0
        # printable table
        assert "parameter" in str(summary)

    def test_write_log_and_networks(self, tmp_path):
        res = self._make_seq_result()
        log = tmp_path / "seq.log"
        trees = tmp_path / "seq.trees"
        res.write_log(str(log))
        res.write_networks(str(trees))
        states, traces = read_tracer_log(str(log))
        assert len(states) == 300
        assert "theta" in traces
        assert trees.read_text().count("TREE STATE_") == 300
