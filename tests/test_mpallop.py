"""
Test suite for the Maximum Parsimony Allopolyploidy inference module
(phynetpy.Infer_MP_Allop).

Includes:
    - Parsimony scoring against known scenario networks.
    - Network inference with bootstrap support.
    - Larger-scale inference tests (10, 100 gene trees).
    - Runtime / convergence benchmarks (stubs).
    - Robustness tests for malformed input and starting-network generation.

The entire class is currently **skipped** (``@pytest.mark.skip``) because
the module is under active development and requires local data files.
Remove the skip marker once the pipeline and test data are stable.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

import math
import time
from typing import Union

import pytest

# Guard: skip the entire module if Infer_MP_Allop is not available
pytest.importorskip("phynetpy.Infer_MP_Allop", reason="Infer_MP_Allop module removed")

from phynetpy.Infer_MP_Allop import *  # noqa: E402, F403
from phynetpy.IO import read_nexus  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minmaxkey(mapping: dict[object, Union[int, float]],
               mini: bool = True) -> object:
    """Return the key in *mapping* whose value is the minimum (or maximum).

    Args:
        mapping: A dictionary from objects to numerical values.
        mini: If True return the key with the **minimum** value; if False
              return the key with the **maximum** value.

    Returns:
        The key corresponding to the extreme value.
    """
    cur = math.inf if mini else -math.inf
    cur_key = None

    for key, value in mapping.items():
        if (mini and value < cur) or (not mini and value > cur):
            cur = value
            cur_key = key

    return cur_key


# ---------------------------------------------------------------------------
# Individual test routines
# ---------------------------------------------------------------------------

def _mp_score_level1() -> int:
    """Check parsimony score on scenario-D ideal network (level-1)."""
    file_net = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/scenarioD_ideal.nex"
    file_gt = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/D10.nex"
    subgenome_map = {
        "B": ["01bA"], "A": ["01aA"],
        "X": ["01xA", "01xB"], "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"],
    }

    score = ALLOP_SCORE(file_net, file_gt, subgenome_map)
    if score == 3:
        return 1
    print(f"WRONG SCORE: {score}")
    return 0


def _mp_infer_bootstrap() -> int:
    """Infer network with bootstrap support on scenario-D data."""
    file_net = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/scenarioD_ideal.nex"
    file_gt = "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/D10.nex"
    subgenome_map: dict[str, list[str]] = {
        "B": ["01bA"], "A": ["01aA"],
        "X": ["01xA", "01xB"], "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"],
    }

    res = INFER_MP_ALLOP_BOOTSTRAP(file_net, file_gt, subgenome_map)
    _minmaxkey(res, mini=False)
    return 1


def _mp_infer_10_gene_trees() -> int:
    """Infer network from 10 gene trees (scenario J pruned)."""
    res = INFER_MP_ALLOP(
        "/Users/mak17/Documents/PhyNetPy/src/J_pruned_v2.nex",
        {
            "U": ["01uA", "01uB"], "T": ["01tA", "01tB"],
            "B": ["01bA"], "F": ["01fA"], "C": ["01cA"],
            "A": ["01aA"], "D": ["01dA"], "O": ["01oA"],
        },
    )

    if min(res.values()) == -4:
        net_min: Network = _minmaxkey(res, mini=False)  # type: ignore[name-defined]
        print(net_min.newick())
        return 1
    net_min: Network = _minmaxkey(res, mini=False)  # type: ignore[name-defined]
    print(net_min.newick())
    return 0


def _mp_external_5_trees() -> int:
    """Test inference on an external data set with 5 gene trees."""
    gt = GeneTrees(  # type: ignore[name-defined]
        read_nexus(
            "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/external_5.nex"
        ),
        external_naming,  # type: ignore[name-defined]
    )
    for tree in gt.trees:
        print(tree.newick())
    print(gt.mp_allop_map())

    start_t = time.time()
    res = INFER_MP_ALLOP(
        "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/external_5.nex",
        gt.mp_allop_map(),
    )
    print(f"External with 5 GT run time: {time.time() - start_t}")
    print(f"Results: {res}")
    return 1


def _mp_scenario_j_100_genes() -> int:
    """Runtime test: scenario J with 100 gene trees."""
    gt = GeneTrees(  # type: ignore[name-defined]
        read_nexus("/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/J_100.nex")
    )

    start_t = time.time()
    INFER_MP_ALLOP(
        "/Users/mak17/Documents/Lab-PhyNetPy/PhyNetPy/src/J_100.nex",
        gt.mp_allop_map(),
    )
    print(f"J with 100 GT run time: {time.time() - start_t}")
    return 1


# Stubs for future work
def _mp_runtime_study() -> int:
    """(Stub) Runtime scaling study across varying gene-tree counts."""
    return 1

def _mp_convergence_study() -> int:
    """(Stub) Convergence analysis measuring iterations to correct topology."""
    return 1

def _mp_full_scenario_study() -> int:
    """(Stub) Full study on scenarios D, E, F, and J with plots."""
    return 1

def _mp_malformed_data() -> int:
    """(Stub) Ensure graceful failure on malformed input data."""
    return 1

def _mp_starting_network_ploidy() -> int:
    """(Stub) Verify starting networks satisfy ploidy constraints."""
    return 1


# ---------------------------------------------------------------------------
# Pytest class (currently skipped)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Infer MP Allop module under development; requires local data")
class TestInferMPAllop:
    """Aggregated MP allopolyploidy inference tests.

    Remove the ``@pytest.mark.skip`` decorator once the module is stable
    and the required data files are available in CI.
    """

    _TESTS = [
        _mp_score_level1,
        _mp_infer_bootstrap,
        _mp_infer_10_gene_trees,
        _mp_external_5_trees,
        _mp_scenario_j_100_genes,
        _mp_runtime_study,
        _mp_convergence_study,
        _mp_full_scenario_study,
        _mp_malformed_data,
        _mp_starting_network_ploidy,
    ]

    def test_individual(self, indv: int = 0) -> None:
        """Run a single test by index (useful for isolated debugging)."""
        assert self._TESTS[indv]() == 1, f"Test {indv} failed"
