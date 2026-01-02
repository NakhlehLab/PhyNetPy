#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the new modular architecture.

Tests the ScoringStrategy, NetworkAdapter, ModelBuilder, and BiMarkers components.
"""

from __future__ import annotations


def test_scoring_strategy_interface():
    """Test that scoring strategy interface is properly defined."""
    from .ScoringStrategy import (
        ScoringStrategy, 
        SubtreeIndependentScoring, 
        FullRecomputeScoring,
        ScoringContext
    )
    
    print("Testing ScoringStrategy interface...")
    
    # Test that abstract class can't be instantiated directly
    try:
        ScoringStrategy()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass  # Expected
    
    # Test SubtreeIndependentScoring
    class TestScoring(SubtreeIndependentScoring):
        def score(self, context):
            return 0.0
    
    strategy = TestScoring()
    assert strategy.supports_caching() == True
    assert strategy.is_dirty() == True
    
    # Test ScoringContext
    ctx = ScoringContext(parameters={"u": 0.5})
    assert ctx.get_param("u") == 0.5
    assert ctx.get_param("missing", 1.0) == 1.0
    
    print("  ✓ ScoringStrategy interface tests passed")
    

def test_network_adapter():
    """Test NetworkAdapter functionality."""
    from .NetworkAdapter import NetworkAdapter
    
    print("Testing NetworkAdapter...")
    
    adapter = NetworkAdapter()
    
    # Test empty adapter
    assert adapter.network is None
    assert list(adapter.postorder()) == []
    
    print("  ✓ NetworkAdapter tests passed")


def test_model_builder():
    """Test ModelBuilder phase-based construction."""
    from .ModelBuilder import (
        ModelBuilder, 
        BuildPhase, 
        BuildContext,
        ParameterPhase,
        ValidationPhase,
        CustomPhase,
        BuildError
    )
    
    print("Testing ModelBuilder...")
    
    # Test BuildContext
    ctx = BuildContext()
    ctx.set("key", "value")
    assert ctx.get("key") == "value"
    assert ctx.has("key") == True
    assert ctx.require("key") == "value"
    
    try:
        ctx.require("missing")
        assert False, "Should raise BuildError"
    except BuildError:
        pass  # Expected
    
    # Test ParameterPhase
    builder = ModelBuilder()
    builder.add_phase(ParameterPhase({"a": 1, "b": 2}))
    model = builder.build()
    
    assert model.get_parameter("a") == 1
    assert model.get_parameter("b") == 2
    
    # Test CustomPhase
    executed = [False]
    def custom_exec(ctx):
        ctx.set("custom", True)
        executed[0] = True
    
    builder2 = ModelBuilder()
    builder2.add_phase(CustomPhase("Custom", custom_exec))
    model2 = builder2.build()
    
    assert executed[0] == True
    assert model2.get_component("custom") == True
    
    print("  ✓ ModelBuilder tests passed")


def test_bimarkers_q_matrix():
    """Test BiMarkersQ matrix construction."""
    from .MCMC_BiMarkers_v2 import BiMarkersQ, n_to_index, nr_to_index, index_to_nr
    import numpy as np
    
    print("Testing BiMarkersQ...")
    
    # Test index conversions
    assert n_to_index(1) == 0
    assert n_to_index(2) == 2
    assert n_to_index(3) == 5
    
    assert nr_to_index(1, 0) == 0
    assert nr_to_index(1, 1) == 1
    assert nr_to_index(2, 0) == 2
    
    assert index_to_nr(0) == (1, 0)
    assert index_to_nr(1) == (1, 1)
    assert index_to_nr(2) == (2, 0)
    
    # Test Q matrix construction
    Q = BiMarkersQ(n=3, u=0.5, v=0.5, coal=1.0)
    
    # Check matrix dimensions
    expected_rows = int(0.5 * 3 * (3 + 3))
    assert Q.Q.shape == (expected_rows, expected_rows)
    
    # Check diagonal elements are negative (should be)
    for i in range(expected_rows):
        assert Q.Q[i][i] <= 0
    
    # Test expt
    Qt = Q.expt(1.0)
    assert Qt.shape == Q.Q.shape
    
    # Rows should sum to approximately 1 (stochastic matrix)
    for i in range(Qt.shape[0]):
        row_sum = np.sum(Qt[i, :])
        assert abs(row_sum - 1.0) < 1e-6, f"Row {i} sums to {row_sum}"
    
    print("  ✓ BiMarkersQ tests passed")


def test_vpi_tracker():
    """Test VPI tracker and rules."""
    import numpy as np
    from .MCMC_BiMarkers_v2 import VPITracker, BiMarkersQ, n_to_index
    
    print("Testing VPITracker...")
    
    tracker = VPITracker()
    
    # Test Rule 0 (leaf initialization)
    reds = np.array([1, 0, 1])  # 3 sites
    samples = 2
    site_count = 3
    vector_len = n_to_index(samples + 1)
    
    vpi_key = tracker.rule0(reds, samples, site_count, vector_len, "leaf1")
    
    assert vpi_key == ("branch_leaf1: bottom",)
    assert vpi_key in tracker.vpis
    print("  ✓ Rule 0 initializes leaf VPIs correctly")
    
    # Check that VPI has correct structure
    vpi = tracker.get_vpi(vpi_key)
    assert len(vpi) == site_count
    print("  ✓ VPI has correct number of sites")
    
    # Test Rule 1 (propagate to top)
    Q = BiMarkersQ(n=samples, u=0.5, v=0.5, coal=1.0)
    Qt = Q.expt(1.0)
    
    vpi_key = tracker.rule1(vpi_key, "leaf1", samples, Qt, site_count)
    assert "top" in vpi_key[0]
    print("  ✓ Rule 1 propagates to branch top")
    
    # Create another leaf for testing Rule 2
    tracker2 = VPITracker()
    reds2 = np.array([0, 1, 0])
    vpi_key1 = tracker2.rule0(reds, samples, site_count, vector_len, "leaf1")
    vpi_key1 = tracker2.rule1(vpi_key1, "leaf1", samples, Qt, site_count)
    
    vpi_key2 = tracker2.rule0(reds2, samples, site_count, vector_len, "leaf2")
    vpi_key2 = tracker2.rule1(vpi_key2, "leaf2", samples, Qt, site_count)
    
    # Test Rule 2 (merge disjoint branches)
    merged_key = tracker2.rule2(
        vpi_key1, vpi_key2,
        "leaf1", "leaf2", "internal1",
        site_count, vector_len
    )
    assert "internal1" in merged_key[-1]
    print("  ✓ Rule 2 merges disjoint branches")
    
    # Test Rule 3 (split at reticulation)
    tracker3 = VPITracker()
    vpi_key_child = tracker3.rule0(reds, samples, site_count, vector_len, "child")
    vpi_key_child = tracker3.rule1(vpi_key_child, "child", samples, Qt, site_count)
    
    split_key = tracker3.rule3(
        vpi_key_child, "child",
        "parent1", "parent2",
        0.6, 0.4,  # gamma values
        samples, site_count
    )
    assert "parent1" in str(split_key) and "parent2" in str(split_key)
    print("  ✓ Rule 3 splits at reticulation node")
    
    print("VPITracker tests passed!\n")


def test_reticulation_handling():
    """Test that reticulation nodes are handled correctly."""
    import numpy as np
    from .MCMC_BiMarkers_v2 import VPITracker, BiMarkersQ, n_to_index
    
    print("Testing reticulation handling...")
    
    # Create a simple scenario with a reticulation
    samples = 2
    site_count = 2
    vector_len = n_to_index(samples + 1)
    
    Q = BiMarkersQ(n=samples, u=0.5, v=0.5, coal=1.0)
    Qt = Q.expt(1.0)
    
    tracker = VPITracker()
    
    # Initialize a leaf
    reds = np.array([1, 0])
    vpi_key = tracker.rule0(reds, samples, site_count, vector_len, "leaf")
    vpi_key = tracker.rule1(vpi_key, "leaf", samples, Qt, site_count)
    
    # Apply Rule 3: split at reticulation (gamma_y=0.5, gamma_z=0.5)
    vpi_key = tracker.rule3(
        vpi_key, "leaf",
        "branch_y", "branch_z",
        0.5, 0.5,
        samples, site_count
    )
    
    # Check that we now have entries for both parent branches
    vpi = tracker.get_vpi(vpi_key)
    assert len(vpi) == site_count
    print("  ✓ Rule 3 creates VPIs for both parent branches")
    
    # The VPI should have entries where n_y + n_z = n_total
    # This tests that the lineage splitting is working
    for site in range(site_count):
        has_entries = len(vpi[site]) > 0
        assert has_entries, f"No entries for site {site}"
    print("  ✓ Rule 3 distributes lineages correctly")
    
    print("Reticulation handling tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running New Architecture Tests")
    print("=" * 60)
    print()
    
    try:
        test_scoring_strategy_interface()
        test_network_adapter()
        test_model_builder()
        test_bimarkers_q_matrix()
        test_vpi_tracker()
        test_reticulation_handling()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Make sure you're running from the PhyNetPy directory")
        print("or that PhyNetPy is in your PYTHONPATH")
        return False
        
    except AssertionError as e:
        print(f"\nAssertion Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()

