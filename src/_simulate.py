#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##############################################################################

"""
The third verb: simulate.

The generative inverse of :func:`phynetpy.infer.infer` -- the same three
axes, run in the opposite direction.  Putting the biology on its own axis is
what makes this nearly free: ``simulate`` reuses the entire model axis and
returns objects on the data axis, so its output feeds straight back into
:func:`~phynetpy.infer.score` and :func:`~phynetpy.infer.infer`::

    sim = simulate(MSC(), true_net, n=100, data="gene_trees", mapping=mapping)
    back = infer(sim, model=MSC(), criterion=MDC())

That composition is the point.  A null-model or recovery check is now two
calls with no glue code, because the type that comes out of ``simulate`` is
the type that goes into the verbs.

.. warning::
   Simulation requires networks tagged with
   :class:`~phynetpy.models.BranchLengthUnit.SUBSTITUTIONS_PER_SITE`.
   Gene-tree criteria use ``COALESCENT_2N`` instead; convert explicitly with
   :func:`phynetpy.models.convert_network_branch_lengths`.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional

import numpy as np

from .data import Alignment, BiallelicMarkers, Data, GeneTrees
from .models import (
    Allopolyploid,
    BranchLengthUnit,
    MSC,
    Model,
    resolve_model,
)
from .Network import Network

__all__ = ["simulate"]


#: Aliases for the ``data`` argument, mapping user spellings to the data-axis
#: type to produce.
_DATA_KINDS = {
    "gene_trees": "gene_trees",
    "genetrees": "gene_trees",
    "trees": "gene_trees",
    "alignment": "alignment",
    "alignments": "alignment",
    "sequences": "alignment",
    "seq": "alignment",
    "markers": "markers",
    "biallelic": "markers",
    "biallelicmarkers": "markers",
    "snp": "markers",
    "snps": "markers",
}

#: Default population mutation rate when the model does not set one.  Matches
#: ``_sim_seq.simulate_multilocus``.
_DEFAULT_THETA = 0.02


def _yule_species_tree(taxa: Any, birth_rate: float, seed: Any) -> Network:
    """Draw a species tree under a pure-birth (Yule) process.

    Used when no *network* is supplied.  Pure birth rather than birth-death:
    :class:`phynetpy.BirthDeath.CBDP` keeps extinct lineages as tips (so
    ``taxa=5`` would not mean five tips) and draws from the global ``random``
    module (so it could not honour *seed*).  Simulate along your own network
    if you need extinction.

    Args:
        taxa: Number of tips, or the list of tip labels to use.
        birth_rate: Speciation rate; must be positive.
        seed: Random seed.

    Returns:
        Network: The simulated species tree, relabelled if *taxa* was a list.

    Raises:
        ValueError: If the rate or the tip count are inconsistent.
    """
    from .BirthDeath import Yule

    labels: Optional[List[str]] = None
    if isinstance(taxa, int):
        n_taxa = taxa
    else:
        labels = [str(label) for label in taxa]
        n_taxa = len(labels)

    if n_taxa < 2:
        raise ValueError(f"a species tree needs at least 2 taxa; got {n_taxa}.")
    if birth_rate <= 0:
        raise ValueError(f"birth_rate must be positive; got {birth_rate}.")

    network = Yule(
        birth_rate, n=n_taxa, rng=np.random.default_rng(seed),
    ).generate_network()
    network.set_branch_length_unit(BranchLengthUnit.SUBSTITUTIONS_PER_SITE)

    if labels is not None:
        for leaf, label in zip(sorted(network.get_leaves(), key=lambda v: v.label),
                               labels):
            network.update_node_name(leaf, label)
    return network


def simulate(
    model: Any = None,
    network: Optional[Network] = None,
    n: int = 1,
    data: str = "gene_trees",
    *,
    mapping: Optional[Dict[str, List[str]]] = None,
    seq_length: int = 500,
    seed: Any = None,
    taxa: Any = None,
    birth_rate: float = 1.0,
    **params: Any,
) -> Data:
    """Generate data under ``model`` along ``network``.

    Args:
        model: The generative process (:class:`~phynetpy.models.MSC`).  A
            string shortcut such as ``"MSC"`` is accepted; ``None`` defaults
            to ``MSC()``.  Process parameters are read off the model:
            ``theta``/``branch_thetas`` for the coalescent and ``u``/``v`` for
            markers.
        network: The species network to simulate along.  Needs branch
            lengths tagged as ``SUBSTITUTIONS_PER_SITE`` and inheritance
            probabilities on reticulation in-edges. ``None`` draws a species
            *tree* under a pure-birth process instead, which requires *taxa*.
        n: How much data to generate.  Number of gene trees, number of loci,
            or number of marker sites, depending on *data*.
        data: Which data-axis type to return -- ``"gene_trees"``,
            ``"alignment"``, or ``"markers"``.
        mapping: Species -> list of allele labels to sample.  ``None`` samples
            one allele per leaf, named after the leaf.
        seq_length: Sites per locus, for ``data="alignment"``.
        seed: Random seed for reproducibility.
        taxa: Tip count, or list of tip labels, for the species tree drawn
            when *network* is ``None``.
        birth_rate: Speciation rate for that pure-birth (Yule) process.
        **params: Extra arguments forwarded to the underlying simulator
            (``substitution_model``, ``samples``, ...).

    Returns:
        Data: A :class:`~phynetpy.data.GeneTrees`,
        :class:`~phynetpy.data.Alignment`, or
        :class:`~phynetpy.data.BiallelicMarkers`, carrying *mapping*, so the
        result feeds directly into :func:`~phynetpy.infer.infer` and
        :func:`~phynetpy.infer.score`.  The generating network is attached as
        ``.true_network`` so a recovery check needs nothing else.

    Raises:
        TypeError: If neither *network* nor *taxa* is given.
        ValueError: If *data* is not a recognised kind, *n* is not positive,
            or the species-tree parameters are inconsistent.
        NotImplementedError: If the model has no simulator for this data
            kind.

    Examples:
        A recovery check -- simulate under a known network, then try to get
        it back::

            sim = simulate(MSC(theta=0.02), true_net, n=50, mapping=mapping)
            result = infer(sim, criterion=PseudoLikelihood())

        A null check on a species tree PhyNetPy drew itself::

            sim = simulate(MSC(), taxa=6, n=200)
            result = infer(sim, criterion=PseudoLikelihood())
            compare_networks(result.best, sim.true_network)
    """
    model_obj: Model = resolve_model(model)
    if "branch_thetas" in params:
        raise TypeError(
            "configure fixed population rates with "
            "MSC(branch_thetas=...), not simulate(branch_thetas=...)."
        )

    if n <= 0:
        raise ValueError(f"n must be positive; got {n}.")
    if network is None:
        if taxa is None:
            raise TypeError(
                "simulate() needs either a network to simulate along, or "
                "taxa=<count or labels> to draw one under a pure-birth "
                "process."
            )
        network = _yule_species_tree(taxa, birth_rate, seed)

    key = str(data).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _DATA_KINDS:
        known = sorted(set(_DATA_KINDS.values()))
        raise ValueError(
            f"unknown data kind {data!r}; expected one of {known}."
        )
    kind = _DATA_KINDS[key]

    if isinstance(model_obj, Allopolyploid):
        raise NotImplementedError(
            "simulation under the allopolyploid model is not implemented "
            "yet; only MSC() has simulators. The model axis is ready for it "
            "-- the generative code is what is missing."
        )
    if not isinstance(model_obj, MSC):
        raise NotImplementedError(
            f"no simulator for {type(model_obj).__name__}."
        )

    # No mapping means one sampled allele per leaf, named after the leaf.
    resolved_mapping = mapping if mapping is not None else {
        leaf.label: [leaf.label] for leaf in network.get_leaves()
    }
    theta = model_obj.theta if model_obj.theta is not None else _DEFAULT_THETA

    if kind == "gene_trees":
        result = _simulate_gene_trees(
            model_obj, network, n, resolved_mapping, theta, seed,
        )
    elif kind == "alignment":
        result = _simulate_alignment(
            model_obj, network, n, resolved_mapping, theta, seq_length, seed,
            params,
        )
    else:
        result = _simulate_markers(
            model_obj, network, n, resolved_mapping, theta, seed, params,
        )

    # Ground truth, so a recovery check is two calls with nothing in between.
    result.true_network = network
    return result


def _simulate_gene_trees(
    model: MSC,
    network: Network,
    n: int,
    mapping: Dict[str, List[str]],
    theta: float,
    seed: Any,
) -> GeneTrees:
    """Simulate ``n`` gene trees under the MSNC along ``network``."""
    from ._sim_seq import simulate_gene_tree

    rng = np.random.default_rng(seed)
    trees = [
        simulate_gene_tree(
            network, mapping, theta, rng,
            branch_thetas=model.branch_thetas,
        )
        for _ in range(n)
    ]
    # Simulated gene trees are coalescent genealogies, so they carry real
    # branch lengths -- unlike gene trees read from a topology-only file.
    return GeneTrees(trees, mapping, has_branch_lengths=True)


def _simulate_alignment(
    model: MSC,
    network: Network,
    n: int,
    mapping: Dict[str, List[str]],
    theta: float,
    seq_length: int,
    seed: Any,
    params: Dict[str, Any],
) -> Alignment:
    """Simulate ``n`` loci of sequence data along ``network``."""
    from ._sim_seq import simulate_multilocus

    simulated = simulate_multilocus(
        network, mapping, n, seq_length,
        theta=theta,
        model=params.get("substitution_model"),
        branch_thetas=model.branch_thetas,
        seed=seed,
    )
    return Alignment(
        simulated.loci, mapping, substitution_model=simulated.model,
    )


def _simulate_markers(
    model: MSC,
    network: Network,
    n: int,
    mapping: Dict[str, List[str]],
    theta: float,
    seed: Any,
    params: Dict[str, Any],
) -> BiallelicMarkers:
    """Simulate ``n`` biallelic marker sites along ``network``."""
    from .MSA import MSA, DataSequence
    from ._sim_markers import simulate_biallelic_markers

    samples = params.get("samples")
    marker_mapping = {sp: list(labels) for sp, labels in mapping.items()}
    if samples is None:
        samples = {sp: len(labels) for sp, labels in marker_mapping.items()}
    else:
        try:
            samples = {
                str(sp): operator.index(count)
                for sp, count in samples.items()
            }
        except TypeError as exc:
            raise ValueError("marker sample counts must be integers.") from exc
        if set(samples) != set(marker_mapping):
            raise ValueError(
                "samples keys must exactly match the marker mapping."
            )
        for species, count in samples.items():
            if count <= 0:
                raise ValueError(
                    f"sample count for {species!r} must be positive."
                )
            labels = marker_mapping[species]
            if len(labels) == 1 and labels[0] == species and count > 1:
                marker_mapping[species] = [
                    f"{species}_{i}" for i in range(count)
                ]
            elif len(labels) != count:
                raise ValueError(
                    f"mapping for {species!r} has {len(labels)} labels but "
                    f"samples requests {count}."
                )

    data = simulate_biallelic_markers(
        network,
        n,
        marker_mapping,
        theta=theta,
        u=model.u,
        v=model.v,
        branch_thetas=model.branch_thetas,
        rng=np.random.default_rng(seed),
    )
    records = [
        DataSequence(data[label], label) for label in sorted(data)
    ]
    return BiallelicMarkers(
        MSA(data=records), marker_mapping, samples=samples
    )
