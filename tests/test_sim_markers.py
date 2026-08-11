"""Exact biallelic-marker simulation and unit-contract tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phynetpy.BiMarkers import BiMarkersTransition
from phynetpy.Network import Network
from phynetpy.SNPSimulator import SimulatedSNPData, simulate as simulate_snp
from phynetpy.data import BiallelicMarkers
from phynetpy.criteria import Bayesian, Likelihood
from phynetpy.infer import infer, score, simulate
from phynetpy.models import BranchLengthUnit, MSC
from phynetpy._sim_markers import (
    _mutate_biallelic_state,
    simulate_biallelic_markers,
)


def _timed_net(newick: str) -> Network:
    return Network.from_newick(
        newick,
        branch_length_unit=BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
    )


def _two_taxon_net() -> Network:
    return _timed_net("(A:0.01,B:0.01)Root;")


def test_seed_reproducibility_and_multisample_counts() -> None:
    net = _two_taxon_net()
    mapping = {"A": ["A_0", "A_1"], "B": ["B_0", "B_1", "B_2"]}
    kwargs = dict(theta=0.02, u=0.7, v=1.3)

    first = simulate_biallelic_markers(
        net, 25, mapping, rng=np.random.default_rng(17), **kwargs
    )
    second = simulate_biallelic_markers(
        net, 25, mapping, rng=np.random.default_rng(17), **kwargs
    )

    assert first == second
    assert all(0 <= count <= 2 for count in first["A"])
    assert all(0 <= count <= 3 for count in first["B"])


def test_legacy_public_wrapper_uses_exact_multisample_path() -> None:
    net = _two_taxon_net()
    result = simulate_snp(
        2,
        12,
        net,
        samples={"A": 2, "B": 3},
        theta=0.02,
        seed=11,
    )
    expected = simulate_biallelic_markers(
        net,
        12,
        {"A": ["A_0", "A_1"], "B": ["B_0", "B_1", "B_2"]},
        theta=0.02,
        u=1.0,
        v=1.0,
        rng=np.random.default_rng(11),
    )
    assert result.theta == pytest.approx(0.02)
    assert result.data == expected


def test_nexus_uses_single_hex_character_per_count(tmp_path) -> None:
    path = tmp_path / "counts.nex"
    simulated = SimulatedSNPData(
        network=_two_taxon_net(),
        data={"A": [10, 15], "B": [0, 1]},
        n_taxa=2,
        n_sites=2,
        samples={"A": 15, "B": 15},
        u=1.0,
        v=1.0,
        theta=0.02,
        seed=1,
    )
    simulated.write_nexus(str(path))
    markers = BiallelicMarkers.from_file(
        str(path), samples=simulated.samples
    )
    observed = {
        record.get_name(): record.get_numerical_seq()
        for record in markers.alignment.get_records()
    }
    assert observed == simulated.data


@pytest.mark.parametrize("state,row", [(0, 1), (1, 2)])
def test_two_state_ctmc_matches_bryant_transition(state: int, row: int) -> None:
    u, v, t = 0.8, 1.2, 0.37
    transition = BiMarkersTransition(1, u, v, 0.02).expt(t)
    expected_red = float(transition[row, 2])

    class ThresholdRNG:
        def __init__(self, value: float) -> None:
            self.value = value

        def random(self) -> float:
            return self.value

    assert _mutate_biallelic_state(
        state, t, u, v, ThresholdRNG(expected_red / 2.0)
    ) == 1
    assert _mutate_biallelic_state(
        state, t, u, v, ThresholdRNG((1.0 + expected_red) / 2.0)
    ) == 0


def test_reticulation_simulation_and_simulate_to_score_are_finite() -> None:
    net = _timed_net(
        "((A:0.01,(B:0.005)#H1:0.005[&gamma=0.3]):0.01,"
        "(#H1:0.01[&gamma=0.7],C:0.015):0.005)Root;"
    )
    mapping = {"A": ["A"], "B": ["B"], "C": ["C"]}
    model = MSC(
        theta=0.02,
        u=1.0,
        v=1.0,
        branch_thetas={
            "A": 0.01,
            "B": 0.03,
            ("__root__", "Root"): 0.04,
        },
    )
    markers = simulate(
        model,
        net,
        n=8,
        data="markers",
        mapping=mapping,
        seed=5,
    )

    value = score(net, markers, model=model, criterion=Likelihood())
    assert math.isfinite(value)
    with pytest.raises(NotImplementedError, match="branch_thetas"):
        infer(markers, model=model, criterion=Bayesian())


@pytest.mark.parametrize(
    "unit",
    [BranchLengthUnit.UNSPECIFIED, BranchLengthUnit.COALESCENT_2N],
)
def test_marker_simulation_rejects_wrong_or_missing_units(unit) -> None:
    net = Network.from_newick(
        "(A:0.01,B:0.01)Root;",
        branch_length_unit=unit,
    )
    with pytest.raises(ValueError, match="branch-length units|requires"):
        simulate_biallelic_markers(
            net,
            2,
            {"A": ["A"], "B": ["B"]},
            theta=0.02,
            u=1.0,
            v=1.0,
            rng=np.random.default_rng(1),
        )


@pytest.mark.slow
def test_simulated_markers_prefer_true_tree_statistically() -> None:
    """Fixed-seed integration check kept outside default CI."""

    true_net = _timed_net("((A:0.02,B:0.02)I:0.02,C:0.04)Root;")
    wrong_net = _timed_net("((A:0.02,C:0.02)I:0.02,B:0.04)Root;")
    mapping = {"A": ["A"], "B": ["B"], "C": ["C"]}
    model = MSC(theta=0.02, u=1.0, v=1.0)
    markers = simulate(
        model,
        true_net,
        n=400,
        data="markers",
        mapping=mapping,
        seed=431,
    )

    true_score = score(
        true_net, markers, model=model, criterion=Likelihood()
    )
    wrong_score = score(
        wrong_net, markers, model=model, criterion=Likelihood()
    )
    assert true_score > wrong_score
