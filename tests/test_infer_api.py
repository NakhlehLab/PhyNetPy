"""
Test suite for the two-verb inference API.

Covers the parts of the design that are *contracts* rather than numerics:

    - Dispatch: the registry resolves legal triples and refuses the rest,
      with the three failure modes kept distinct (``TypeError`` for
      undefined, ``ValueError`` for an unsatisfiable branch-length policy,
      ``NotImplementedError`` for unimplemented).
    - The data axis: type detection, mapping placement, and the
      ``has_branch_lengths`` flag that has to be captured at parse time.
    - The criterion axis: string shortcuts, ``Bayesian`` wrapping an
      objective, and ``scorable``.
    - ``InferenceResult``: uniform shape, score direction, and passthrough
      to the engine's native result.
    - ``StartMode.AUGMENT``: the result really does contain the backbone.

Numerical correctness of each engine lives in that engine's own test module
(``test_mpl.py``, ``test_mcmc_gt.py``, ...); these tests only check that the
verbs route to it and report what it returned.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import warnings

import pytest

from phynetpy.criteria import (
    Bayesian,
    CriterionError,
    Likelihood,
    MDC,
    PseudoLikelihood,
    resolve_criterion,
)
from phynetpy.data import (
    Alignment,
    BiallelicMarkers,
    Data,
    DataError,
    GeneTrees,
)
from phynetpy.infer import (
    InferenceResult,
    Start,
    StartMode,
    infer,
    registered_cells,
    score,
    simulate,
    validity_matrix,
)
from phynetpy.models import (
    Allopolyploid,
    BranchLengthUnit,
    MSC,
    Model,
    ModelSpecError,
    convert_network_branch_lengths,
    resolve_model,
)
from phynetpy.GraphUtils import network_clusters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Four taxa, majority topology (((A,B),C),D) with two discordant trees, so a
# search has something to prefer.
_GT_NEWICKS = [
    "(((A:0.5,B:0.5):0.5,C:1.0):0.5,D:1.5);",
    "(((A:0.5,B:0.5):0.5,C:1.0):0.5,D:1.5);",
    "(((A:0.5,B:0.5):0.5,C:1.0):0.5,D:1.5);",
    "((A:1.0,(B:0.5,C:0.5):0.5):0.5,D:1.5);",
    "(((A:0.5,C:0.5):0.5,B:1.0):0.5,D:1.5);",
]

_GT_NEWICKS_NO_LENGTHS = [
    "(((A,B),C),D);",
    "(((A,B),C),D);",
    "((A,(B,C)),D);",
]


@pytest.fixture
def gts() -> GeneTrees:
    """Gene trees with branch lengths."""
    return GeneTrees.from_newick(_GT_NEWICKS)


@pytest.fixture
def gts_topologies() -> GeneTrees:
    """Gene trees without branch lengths (topologies only)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GeneTrees.from_newick(_GT_NEWICKS_NO_LENGTHS)


@pytest.fixture
def seed_net(gts: GeneTrees):
    """Majority-rule consensus of the gene trees, as a starting network."""
    net = gts.build_majority_rule_consensus_tree()
    net.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
    return net


def _as_substitution_net(network, theta: float = 0.02):
    """Copy a coalescent-unit fixture onto the timed-MSC scale."""

    return convert_network_branch_lengths(
        network,
        BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        theta=theta,
    )


# ---------------------------------------------------------------------------
# The data axis
# ---------------------------------------------------------------------------

class TestDataAxis:

    def test_gene_trees_is_data(self, gts):
        assert isinstance(gts, Data)
        assert gts.taxa == {"A", "B", "C", "D"}
        assert len(gts.trees) == len(_GT_NEWICKS)

    def test_branch_lengths_detected_from_source(self, gts, gts_topologies):
        # This is the whole reason the flag is captured at parse time: the
        # reader back-fills missing lengths with 1.0, so both objects have a
        # length on every edge by the time anyone can look.
        assert gts.has_branch_lengths is True
        assert gts_topologies.has_branch_lengths is False

    def test_branch_length_flag_from_file(self, tmp_path):
        with_lengths = tmp_path / "with.nwk"
        without = tmp_path / "without.nwk"
        with_lengths.write_text("((A:1,B:1):1,C:2);\n")
        without.write_text("((A,B),C);\n")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert GeneTrees.from_file(with_lengths).has_branch_lengths is True
            assert GeneTrees.from_file(without).has_branch_lengths is False

    def test_mapping_lives_on_the_data(self, gts):
        # The verbs take no mapping argument; it travels with the sample.
        assert gts.mapping is None
        assert gts.resolved_mapping() == {n: [n] for n in ["A", "B", "C", "D"]}

        explicit = {"AB": ["A", "B"], "CD": ["C", "D"]}
        gts.mapping = explicit
        assert gts.mapping == explicit
        assert gts.resolved_mapping() == explicit
        # Kept in step with the inherited attribute the engines read.
        assert gts.species_gene_mapping == explicit

    def test_mapping_via_constructor(self):
        mapping = {"AB": ["A", "B"], "C": ["C"], "D": ["D"]}
        loaded = GeneTrees.from_newick(_GT_NEWICKS, mapping)
        assert loaded.resolved_mapping() == mapping

    def test_empty_rejected(self):
        with pytest.raises(DataError):
            GeneTrees([])

    def test_non_network_input_rejected(self):
        with pytest.raises(DataError, match="Network objects"):
            GeneTrees(["((A,B),C);"])  # type: ignore[list-item]

    def test_reticulate_input_rejected(self):
        from phynetpy.GraphUtils import add_hybrid
        from phynetpy.Network import Network

        net = Network.from_newick("(((A:1,B:1):1,C:2):1,D:3);")
        leaf_of = {leaf.label: leaf for leaf in net.get_leaves()}
        add_hybrid(
            net,
            next(iter(net.in_edges(leaf_of["A"]))),
            next(iter(net.in_edges(leaf_of["C"]))),
        )
        assert any(net.in_degree(v) >= 2 for v in net.V())

        # A reticulate network is a species network, not a gene tree; taking
        # it here would silently score the wrong thing.
        with pytest.raises(DataError, match="reticulation"):
            GeneTrees([net])

    def test_sequence_data_has_no_branch_lengths(self):
        aln = Alignment([{"a1": "ACGT", "b1": "ACGA"}])
        assert aln.has_branch_lengths is False
        assert aln.n_loci == 1
        assert aln.n_sites == 4
        assert aln.taxa == {"a1", "b1"}

    def test_markers_default_to_one_sample_per_taxon(self):
        from phynetpy.MSA import DataSequence, MSA

        msa = MSA(data=[DataSequence([0, 1, 2], "A"),
                        DataSequence([1, 1, 0], "B")])
        markers = BiallelicMarkers(msa)
        assert markers.samples == {"A": 1, "B": 1}
        assert markers.n_sites == 3

    def test_markers_reject_non_msa(self):
        with pytest.raises(DataError):
            BiallelicMarkers("markers.nex")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The model and criterion axes
# ---------------------------------------------------------------------------

class TestModelAxis:

    def test_defaults_to_msc(self):
        assert isinstance(resolve_model(None), MSC)

    @pytest.mark.parametrize("spec", ["MSC", "msc", "MSNC", "coalescent"])
    def test_string_shortcuts(self, spec):
        assert isinstance(resolve_model(spec), MSC)

    @pytest.mark.parametrize("spec", ["allopolyploid", "allop", "POLYPLOID"])
    def test_allopolyploid_shortcuts(self, spec):
        assert isinstance(resolve_model(spec), Allopolyploid)

    def test_unknown_string_rejected(self):
        with pytest.raises(ModelSpecError):
            resolve_model("gene duplication")

    def test_instance_passes_through(self):
        model = MSC(theta=0.02)
        assert resolve_model(model) is model

    def test_non_positive_rates_rejected(self):
        with pytest.raises(ModelSpecError):
            MSC(theta=0.0)
        with pytest.raises(ModelSpecError):
            MSC(u=-1.0)
        with pytest.raises(ModelSpecError):
            MSC(theta=float("nan"))
        with pytest.raises(ModelSpecError, match="keys"):
            MSC(branch_thetas={object(): 0.02})

    def test_subgenome_map_resolves_from_data(self, gts):
        gts.mapping = {"AB": ["A", "B"], "C": ["C"], "D": ["D"]}
        assert Allopolyploid().resolve_subgenome_map(gts) == gts.mapping

        explicit = {"X": ["A"], "Y": ["B", "C", "D"]}
        assert Allopolyploid(explicit).resolve_subgenome_map(gts) == explicit

    def test_models_model_is_not_modelgraph_model(self):
        import phynetpy

        # The naming collision the subpackages exist to avoid: the top-level
        # ``Model`` must stay the probabilistic graphical model.
        assert phynetpy.Model is not Model
        assert phynetpy.Model.__module__.endswith("ModelGraph")


class TestCriterionAxis:

    def test_defaults_to_likelihood(self):
        assert isinstance(resolve_criterion(None), Likelihood)

    @pytest.mark.parametrize(
        "spec,expected",
        [("MDC", MDC), ("parsimony", MDC), ("ML", Likelihood),
         ("MPL", PseudoLikelihood), ("pseudo", PseudoLikelihood),
         ("bayesian", Bayesian), ("mcmc", Bayesian)],
    )
    def test_string_shortcuts(self, spec, expected):
        assert isinstance(resolve_criterion(spec), expected)

    def test_unknown_string_rejected(self):
        with pytest.raises(CriterionError):
            resolve_criterion("maximum vibes")

    def test_accepts_data_encodes_what_is_defined(self):
        # Parsimony over an alignment is not unimplemented, it is undefined.
        assert MDC.accepts_data == (GeneTrees,)
        assert Alignment in Likelihood.accepts_data
        assert Alignment not in PseudoLikelihood.accepts_data

    def test_mdc_ignores_branch_lengths_by_construction(self):
        assert MDC().use_branch_lengths is False

    def test_bayesian_delegates_to_its_objective(self):
        bayes = Bayesian(objective=PseudoLikelihood())
        assert bayes.accepts_data == PseudoLikelihood.accepts_data
        assert bayes.use_branch_lengths is bayes.objective.use_branch_lengths

    def test_bayesian_is_not_scorable(self):
        assert Bayesian().scorable is False
        assert Likelihood().scorable is True

    def test_bayesian_cannot_wrap_bayesian(self):
        with pytest.raises(CriterionError):
            Bayesian(objective=Bayesian())

    def test_bayesian_rejects_inconsistent_chain_budget(self):
        with pytest.raises(CriterionError):
            Bayesian(chain_length=100, burnin=100)

    def test_unimplemented_options_refuse_rather_than_ignore(self):
        with pytest.raises(CriterionError):
            MDC(weighting="bootstrap")
        with pytest.raises(CriterionError):
            PseudoLikelihood(subsets="quartets")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestDispatch:

    def test_illegal_combination_is_a_type_error(self, gts):
        # MDC needs gene-tree topologies; an alignment cannot supply them.
        aln = Alignment([{"A": "ACGT", "B": "ACGA"}])
        with pytest.raises(TypeError, match="not defined on"):
            infer(aln, criterion=MDC())

    def test_unimplemented_combination_is_not_implemented(self, gts):
        # Valid in principle (MDC is defined on gene trees under the MSC);
        # PhyNetPy just has no deep-coalescence engine for it.
        with pytest.raises(NotImplementedError, match="valid in principle"):
            infer(gts, model=MSC(), criterion=MDC())

    def test_unsatisfiable_branch_length_policy_is_a_value_error(
        self, gts_topologies, seed_net,
    ):
        with pytest.raises(ValueError, match="carries none"):
            score(
                seed_net, gts_topologies,
                criterion=Likelihood(use_branch_lengths=True),
            )

    def test_branch_lengths_allowed_when_present(self, gts, seed_net):
        # Same criterion, data that can satisfy it: no error from dispatch.
        assert isinstance(
            score(seed_net, gts, criterion=Likelihood(use_branch_lengths=True)),
            float,
        )

    def test_non_axis_arguments_rejected(self, gts):
        with pytest.raises(TypeError, match="phynetpy.data.Data"):
            infer(_GT_NEWICKS, criterion=Likelihood())  # type: ignore[arg-type]

    def test_score_refuses_bayesian(self, gts, seed_net):
        with pytest.raises(TypeError, match="scores nothing on its own"):
            score(seed_net, gts, criterion=Bayesian())

    def test_registry_is_the_validity_matrix(self):
        matrix = validity_matrix(MSC)

        # Illegal cells ("x") are exactly the accepts_data violations.
        assert matrix["Alignment"]["MDC"] == "x"
        assert matrix["Alignment"]["PseudoLikelihood"] == "x"
        assert matrix["BiallelicMarkers"]["MDC"] == "x"

        # Unimplemented-but-legal cells are marked, not hidden.
        assert matrix["GeneTrees"]["MDC"] == "-"
        assert matrix["Alignment"]["Likelihood"] == "-"
        assert matrix["BiallelicMarkers"]["PseudoLikelihood"] == "-"

        # Implemented cells name the method that runs.
        assert matrix["GeneTrees"]["Likelihood"] == "InferNetwork_ML"
        assert matrix["GeneTrees"]["PseudoLikelihood"] == "InferNetwork_MPL"
        assert matrix["GeneTrees"]["Bayesian"] == "MCMC_GT"
        assert matrix["Alignment"]["Bayesian"] == "MCMC_SEQ"
        assert matrix["BiallelicMarkers"]["Likelihood"] == "MLE_BiMarkers"
        assert matrix["BiallelicMarkers"]["Bayesian"] == "MCMC_BiMarkers"

    def test_allopolyploid_matrix(self):
        assert validity_matrix(Allopolyploid)["GeneTrees"]["MDC"] == "MP_Allop"

    def test_every_registered_cell_names_a_method(self):
        cells = registered_cells()
        assert cells
        for data_name, model_name, criterion_name, method in cells:
            assert method and method != "Engine"

    def test_old_method_names_are_gone(self):
        import phynetpy
        import phynetpy.infer as infer_module

        for name in (
            "MPL", "MCMC_GT", "MCMC_SEQ", "InferNetwork_ML",
            "INFER_MP_ALLOP", "INFER_MP_ALLOP_BOOTSTRAP", "ALLOP_SCORE",
            "MCMC_BIMARKERS", "SNP_LIKELIHOOD",
        ):
            assert not hasattr(phynetpy, name), f"phynetpy.{name} still exists"
            assert not hasattr(infer_module, name), (
                f"phynetpy.infer.{name} still exists"
            )


# ---------------------------------------------------------------------------
# The verbs
# ---------------------------------------------------------------------------

class TestScore:

    def test_pseudo_likelihood(self, gts, seed_net):
        value = score(seed_net, gts, criterion=PseudoLikelihood())
        assert isinstance(value, float)
        assert value <= 0.0

    def test_likelihood(self, gts, seed_net):
        value = score(seed_net, gts, criterion=Likelihood())
        assert isinstance(value, float)
        assert value <= 0.0

    def test_optimize_cannot_make_the_score_worse(self, gts, seed_net):
        import copy

        held = score(copy.deepcopy(seed_net), gts, criterion=Likelihood())
        optimised = score(
            copy.deepcopy(seed_net), gts, criterion=Likelihood(), optimize=True,
        )
        assert optimised >= held - 1e-6

    def test_string_shortcuts_route_the_same_way(self, gts, seed_net):
        import copy

        assert score(
            copy.deepcopy(seed_net), gts, model="MSC", criterion="MPL",
        ) == pytest.approx(
            score(copy.deepcopy(seed_net), gts, criterion=PseudoLikelihood())
        )

    def test_allopolyploid_parsimony_is_an_extra_lineage_count(self, gts):
        subgenomes = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
        value = score(
            gts.build_majority_rule_consensus_tree(), gts,
            model=Allopolyploid(subgenome_map=subgenomes), criterion=MDC(),
        )
        # A parsimony cost: non-negative, and minimised rather than maximised.
        assert value >= 0.0

    def test_optimize_refused_for_parsimony(self, gts):
        subgenomes = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
        with pytest.raises(ValueError, match="no continuous parameters"):
            score(
                gts.build_majority_rule_consensus_tree(), gts,
                model=Allopolyploid(subgenome_map=subgenomes),
                criterion=MDC(), optimize=True,
            )


class TestInfer:

    def test_pseudo_likelihood_returns_inference_result(self, gts):
        result = infer(
            gts, criterion=PseudoLikelihood(),
            num_iter=40, max_reticulations=1, seed=17,
        )
        assert isinstance(result, InferenceResult)
        assert result.method == "InferNetwork_MPL"
        assert result.best is not None
        assert isinstance(result.score, float)
        assert result.lower_is_better is False
        assert result.posterior is None

    def test_likelihood_returns_inference_result(self, gts):
        result = infer(
            gts, criterion=Likelihood(),
            num_iter=15, max_reticulations=1, seed=17,
        )
        assert result.method == "InferNetwork_ML"
        assert isinstance(result.score, float)

    def test_bayesian_populates_the_posterior(self, gts):
        result = infer(
            gts,
            criterion=Bayesian(chain_length=200, burnin=50, sample_freq=10,
                               seed=17),
        )
        assert result.method == "MCMC_GT"
        assert result.posterior is not None
        assert len(result.posterior) > 0

    def test_result_passes_through_to_the_native_object(self, gts):
        result = infer(
            gts,
            criterion=Bayesian(chain_length=200, burnin=50, sample_freq=10,
                               seed=17),
        )
        # The rich per-method surface is not lost to the wrapper.
        assert result.raw is not None
        assert callable(getattr(result, "summary", None))
        with pytest.raises(AttributeError):
            _ = result.definitely_not_an_attribute

    def test_parsimony_result_is_reported_as_minimised(self, gts):
        subgenomes = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
        result = infer(
            gts, model=Allopolyploid(subgenome_map=subgenomes),
            criterion=MDC(), num_iter=20, seed=17,
        )
        assert result.method == "MP_Allop"
        assert result.lower_is_better is True
        assert result.score >= 0.0

    def test_bayesian_pseudo_likelihood_refused_with_a_reason(self, gts):
        with pytest.raises(NotImplementedError, match="calibrated posterior"):
            infer(
                gts,
                criterion=Bayesian(objective=PseudoLikelihood(),
                                   chain_length=200, burnin=50),
            )


class TestStart:

    def test_bare_network_is_a_free_start(self, gts, seed_net):
        result = infer(
            gts, criterion=PseudoLikelihood(), start=seed_net,
            num_iter=20, max_reticulations=1, seed=17,
        )
        assert result.best is not None

    def test_start_defaults_to_free(self, seed_net):
        assert Start(seed_net).mode is StartMode.FREE
        assert Start(seed_net).augment_only is False

    def test_mode_accepts_a_string(self, seed_net):
        assert Start(seed_net, "augment").mode is StartMode.AUGMENT

    def test_bad_start_rejected(self, gts):
        with pytest.raises(TypeError, match="Start or a Network"):
            infer(gts, criterion=PseudoLikelihood(), start="((A,B),C);")

    def test_augment_result_contains_the_backbone(self, gts, seed_net):
        required = network_clusters(seed_net)
        assert required, "backbone must have clusters for this to test anything"

        result = infer(
            gts, criterion=PseudoLikelihood(),
            start=Start(seed_net, mode=StartMode.AUGMENT),
            num_iter=80, max_reticulations=1, seed=17,
        )
        assert required <= network_clusters(result.best)

    def test_start_network_is_not_mutated(self, gts, seed_net):
        before = network_clusters(seed_net)
        infer(
            gts, criterion=PseudoLikelihood(), start=Start(seed_net),
            num_iter=40, max_reticulations=1, seed=17,
        )
        assert network_clusters(seed_net) == before


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

class TestSimulate:

    @pytest.mark.parametrize(
        "kind", ["gene_trees", "alignment", "markers"]
    )
    @pytest.mark.parametrize(
        "unit",
        [
            BranchLengthUnit.UNSPECIFIED,
            BranchLengthUnit.COALESCENT_2N,
        ],
    )
    def test_requires_substitution_units(self, seed_net, kind, unit):
        network, _ = seed_net.copy()
        network.set_branch_length_unit(unit)
        with pytest.raises(ValueError, match="requires|branch-length units"):
            simulate(MSC(theta=0.02), network, n=1, data=kind, seed=3)

    def test_branch_thetas_belong_on_model(self, seed_net):
        with pytest.raises(TypeError, match=r"MSC\(branch_thetas"):
            simulate(
                MSC(theta=0.02),
                _as_substitution_net(seed_net),
                n=1,
                branch_thetas={"A": 0.01},
            )

    def test_gene_trees_round_trip_into_the_verbs(self, seed_net):
        mapping = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
        sim_net = _as_substitution_net(seed_net)
        sim = simulate(MSC(theta=0.02), sim_net, n=6, data="gene_trees",
                       mapping=mapping, seed=3)

        assert isinstance(sim, GeneTrees)
        assert len(sim.trees) == 6
        # Simulated genealogies carry real coalescent branch lengths, so a
        # branch-length-requiring criterion is satisfiable on them.
        assert sim.has_branch_lengths is True
        assert sim.resolved_mapping() == mapping

        # The whole point of sharing the axes: the output is a legal input.
        assert isinstance(score(seed_net, sim, criterion=PseudoLikelihood()), float)

    def test_alignment(self, seed_net):
        mapping = {"A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"]}
        sim = simulate(MSC(theta=0.02), _as_substitution_net(seed_net),
                       n=2, data="alignment",
                       mapping=mapping, seq_length=40, seed=3)
        assert isinstance(sim, Alignment)
        assert sim.n_loci == 2
        assert sim.n_sites == 80

    def test_markers(self, seed_net):
        sim = simulate(
            MSC(), _as_substitution_net(seed_net), n=25,
            data="markers", seed=3,
        )
        assert isinstance(sim, BiallelicMarkers)
        assert sim.n_sites == 25

    def test_truth_is_attached_for_recovery_checks(self, seed_net):
        for kind in ("gene_trees", "alignment", "markers"):
            truth = _as_substitution_net(seed_net)
            sim = simulate(MSC(theta=0.02), truth, n=3, data=kind,
                           seq_length=20, seed=3)
            assert sim.true_network is truth

    def test_simulate_draws_its_own_species_tree(self):
        sim = simulate(MSC(theta=0.02), taxa=5, n=4, seed=11)

        assert isinstance(sim, GeneTrees)
        truth = sim.true_network
        assert len(truth.get_leaves()) == 5
        # Pure birth, so the truth is a tree.
        assert all(truth.in_degree(v) <= 1 for v in truth.V())
        # Alleles are named after the drawn tips, so the result is scoreable
        # without the caller having to learn the generated labels.
        assert sim.taxa == {leaf.label for leaf in truth.get_leaves()}
        scoring_net = convert_network_branch_lengths(
            truth,
            BranchLengthUnit.COALESCENT_2N,
            theta=0.02,
        )
        assert isinstance(
            score(scoring_net, sim, criterion=PseudoLikelihood()), float
        )

    def test_species_tree_honours_supplied_labels(self):
        sim = simulate(MSC(), taxa=["A", "B", "C", "D"], n=3, seed=5)
        assert {leaf.label for leaf in sim.true_network.get_leaves()} == {
            "A", "B", "C", "D",
        }

    def test_drawn_species_tree_is_reproducible(self):
        first = simulate(MSC(), taxa=5, n=2, seed=7).true_network
        second = simulate(MSC(), taxa=5, n=2, seed=7).true_network
        # Compared on clusters, not on the Newick string: children are stored
        # in a set, so one topology has several equally valid renderings.
        assert network_clusters(first) == network_clusters(second)

    def test_network_or_taxa_is_required(self):
        with pytest.raises(TypeError, match="needs either a network"):
            simulate(MSC(), None, n=3)

    @pytest.mark.parametrize("kwargs, message", [
        ({"taxa": 1}, "at least 2 taxa"),
        ({"taxa": 4, "birth_rate": 0.0}, "birth_rate must be positive"),
    ])
    def test_inconsistent_species_tree_params_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            simulate(MSC(), n=3, **kwargs)

    def test_unknown_data_kind_rejected(self, seed_net):
        with pytest.raises(ValueError, match="unknown data kind"):
            simulate(MSC(), seed_net, n=3, data="quartets")

    def test_non_positive_n_rejected(self, seed_net):
        with pytest.raises(ValueError, match="must be positive"):
            simulate(MSC(), seed_net, n=0)

    def test_allopolyploid_simulation_reports_honestly(self, seed_net):
        with pytest.raises(NotImplementedError, match="not implemented"):
            simulate(Allopolyploid(), seed_net, n=3)
