"""
Test suite for the BirthDeath module (phynetpy.BirthDeath).

Validates the correctness and robustness of birth-death process simulators
used to generate random phylogenetic trees/networks:

    - **Yule** (pure-birth) model: taxa-conditioned and time-conditioned generation.
    - **CBDP** (Constant-rate Birth-Death with sampling) model.
    - Bulk generation and clearing of networks.
    - Reproducibility via explicit RNG seeds.
    - Proper error handling for invalid / missing parameters.
"""

import pytest
import numpy as np

from phynetpy.BirthDeath import (
    BirthDeathSimulationError,
    CBDP,
    TIP_ERROR_THRESHOLD,
    Yule,
    estimate_expected_tips,
    live_species,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_nonsense_cbdp(birthrate: float,
                       deathrate: float,
                       n: int,
                       s: float) -> bool:
    """Return True when the CBDP parameter combination is biologically
    nonsensical and should be rejected by the constructor.

    Rules:
        - Birth rate must exceed death rate (otherwise lineages die off).
        - All rates and counts must satisfy basic positivity / range bounds.
    """
    if birthrate <= deathrate:
        return True
    if birthrate <= 0 or deathrate < 0 or n < 2 or s <= 0 or s > 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Yule Model Tests
# ---------------------------------------------------------------------------

class TestYule:
    """Tests for the pure-birth (Yule) process."""

    def test_taxa_conditioned_generation(self):
        """Valid taxa-conditioned Yule processes should produce the requested
        number of leaves; invalid parameters should raise."""
        n_values = [-3, 0, 1, 2, 3, 5, 10]
        birth_rates = [-1, 0, 0.01, 0.1, 1, 10]

        for birth_rate in birth_rates:
            for n in n_values:
                if n < 2 or birth_rate <= 0:
                    with pytest.raises(BirthDeathSimulationError):
                        Yule(birth_rate, n).generate_network()
                else:
                    tree = Yule(birth_rate, n).generate_network()
                    assert len(tree.get_leaves()) == n

    def test_time_conditioned_generation(self):
        """Time-conditioned Yule processes should stop at the specified time
        and reject invalid time/rate combinations."""
        t_values = [-10, 0, 1, 10, 100]
        birth_rates = [-1, 0, 0.01, 0.1, 1, 10]

        for birth_rate in birth_rates:
            for t in t_values:
                if t <= 0 or birth_rate <= 0:
                    with pytest.raises(BirthDeathSimulationError):
                        Yule(birth_rate, time=t).generate_network()
                else:
                    est_tips = estimate_expected_tips(birth_rate, t)
                    if est_tips > TIP_ERROR_THRESHOLD:
                        with pytest.raises(BirthDeathSimulationError):
                            Yule(birth_rate, time=t).generate_network()
                    else:
                        tree = Yule(birth_rate, time=t).generate_network()
                        for leaf in tree.get_leaves():
                            assert leaf.get_time() == t


# ---------------------------------------------------------------------------
# CBDP Model Tests
# ---------------------------------------------------------------------------

class TestCBDP:
    """Tests for the Constant-rate Birth-Death with sampling Process."""

    def test_parameter_validation_and_leaf_count(self):
        """Nonsensical parameter combos should raise; valid ones should
        yield the correct number of live lineages."""
        b_vals = [-1, 0, 0.01, 0.1, 1, 10]
        mu_vals = [-1, 0, 0.01, 0.1, 1, 10]
        n_vals = [-3, 0, 1, 2, 3, 5, 10]
        sample_vals = [-0.1, 0, 0.1, 0.5, 1, 4]

        for brate in b_vals:
            for drate in mu_vals:
                for taxa_ct in n_vals:
                    for srate in sample_vals:
                        if _is_nonsense_cbdp(brate, drate, taxa_ct, srate):
                            with pytest.raises(BirthDeathSimulationError):
                                CBDP(brate, drate, taxa_ct, srate).generate_network()
                        else:
                            net = CBDP(brate, drate, taxa_ct, srate).generate_network()
                            assert len(live_species(net.V())) == taxa_ct


# ---------------------------------------------------------------------------
# Bulk Generation & Clearing
# ---------------------------------------------------------------------------

class TestBulkGeneration:
    """Tests for batch network generation and clearing."""

    def test_yule_bulk_generation_and_clear(self):
        """Generating 10 Yule networks stores them; clearing empties the list."""
        yp = Yule(0.5, 10)
        yp.generate_networks(10)
        assert len(yp.generated_networks) == 10

        yp.clear_generated()
        assert len(yp.generated_networks) == 0

    def test_cbdp_bulk_generation_and_clear(self):
        """Generating 10 CBDP networks stores them; clearing empties the list."""
        cp = CBDP(0.5, 0.05, 10)
        cp.generate_networks(10)
        assert len(cp.generated_networks) == 10

        cp.clear_generated()
        assert len(cp.generated_networks) == 0


# ---------------------------------------------------------------------------
# RNG Consistency
# ---------------------------------------------------------------------------

class TestRNGConsistency:
    """Verify reproducibility when an explicit random seed is provided."""

    def test_yule_same_seed_same_topology(self):
        """Two Yule runs with identical seeds should produce networks with
        the same node/edge counts (a proxy for identical topology)."""
        seed = 1
        net1 = Yule(0.5, 10, rng=np.random.default_rng(seed)).generate_network()
        net2 = Yule(0.5, 10, rng=np.random.default_rng(seed)).generate_network()

        assert len(net1.V()) == len(net2.V())
        assert len(net1.E()) == len(net2.E())


# ---------------------------------------------------------------------------
# Missing Parameters
# ---------------------------------------------------------------------------

class TestMissingInput:
    """Ensure constructors reject ambiguous / incomplete parameter sets."""

    def test_yule_requires_n_or_time(self):
        """Yule with neither n nor time should raise."""
        with pytest.raises(BirthDeathSimulationError):
            Yule(0.5, None, None)
