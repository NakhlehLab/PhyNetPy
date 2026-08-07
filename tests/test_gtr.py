"""
Test suite for the substitution models in phynetpy.GTR.

The models form a nesting hierarchy, and each one is a continuous-time Markov
chain, which gives plenty of independent ways to catch a wrong formula:

Rate matrix (Q) properties, for every model:
    - Rows sum to 0 (Q is a valid generator)
    - Time reversibility: ``pi_i Q_ij == pi_j Q_ji``
    - ``pi`` is the stationary distribution: ``pi^T Q == 0``
    - Unit mean rate: ``-sum_i pi_i Q_ii == 1``
    - Every exchangeability parameter reaches the cell it names

Transition matrix (P(t) = e^(Qt)) properties, for every model:
    - Agrees with ``scipy.linalg.expm(Q t)`` to machine precision. This is the
      key check on the closed forms: six of the eight models bypass expm
      entirely, so each closed form is validated against the thing it replaces.
    - Row stochastic and non-negative
    - ``P(0) == I`` and ``P(t) -> 1 pi^T`` as ``t -> inf``
    - Chapman-Kolmogorov: ``P(s) P(t) == P(s + t)``

Model nesting (a wrong parameter->cell mapping breaks these even when the
matrix invariants above still hold):
    - JC == K80(kappa=1) == F81(uniform pi) == K81(all equal) == ...
    - TN93(alpha_R == alpha_Y) == HKY
    - HKY(uniform pi) == K80
    - F81 == HKY(kappa=1)

Cross-validation:
    - Agreement with the independent implementation used by the inference
      engine (``phynetpy._seq_likelihood``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from phynetpy.GTR import (
    GTR,
    JC,
    K80,
    HKY,
    F81,
    K81,
    SYM,
    TN93,
    SubstitutionModelError,
)

# Base frequencies that are deliberately uneven, to catch anything that only
# works when pi is uniform.
PI = [0.37, 0.23, 0.27, 0.13]

# Branch lengths spanning "almost no change" to "fully saturated".
TIMES = [0.0, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 50.0]

TOL = 1e-11


def _models() -> list[tuple[str, GTR, list[float], bool]]:
    """
    Build one instance of every model.

    Returns:
        list[tuple[str, GTR, list[float], bool]]: Name, model, its base
            frequencies, and whether it uses a closed-form ``expt``.
    """
    return [
        ("JC",   JC(),                              [0.25] * 4, True),
        ("K80",  K80(4.0),                          [0.25] * 4, True),
        ("F81",  F81(PI),                           PI,         True),
        ("HKY",  HKY(PI, 3.5),                      PI,         True),
        ("K81",  K81(5.0, 1.0, 2.0),                [0.25] * 4, True),
        ("TN93", TN93(PI, 5.0, 2.0),                PI,         True),
        ("SYM",  SYM([1., 2., 3., 4., 5., 6.]),     [0.25] * 4, False),
        ("GTR",  GTR(PI, [1., 2., 3., 4., 5., 6.]), PI,         False),
    ]


def _ids() -> list[str]:
    """
    Pytest ids for the model fixtures.

    Returns:
        list[str]: One name per model.
    """
    return [name for name, _, _, _ in _models()]


@pytest.fixture(params = _models(), ids = _ids())
def model(request) -> tuple[str, GTR, np.ndarray, bool]:
    """
    Parameterized fixture yielding each substitution model in turn.

    Args:
        request: Pytest request carrying the parameter tuple.
    Returns:
        tuple[str, GTR, np.ndarray, bool]: Name, model, base frequencies as an
            array, and whether ``expt`` is closed form.
    """
    name, m, pi, closed = request.param
    return name, m, np.asarray(pi, dtype = float), closed


# ---------------------------------------------------------------------------
# Q matrix invariants
# ---------------------------------------------------------------------------

class TestRateMatrix:
    """Invariants that any valid reversible substitution generator satisfies."""

    def test_shape(self, model) -> None:
        """Q must be a plain 4x4 float matrix."""
        _, m, _, _ = model
        Q = np.asarray(m.getQ(), dtype = float)
        assert Q.shape == (4, 4)

    def test_rows_sum_to_zero(self, model) -> None:
        """Off-diagonals must be offset exactly by the diagonal."""
        _, m, _, _ = model
        Q = np.asarray(m.getQ(), dtype = float)
        assert np.allclose(Q.sum(axis = 1), 0.0, atol = TOL)

    def test_off_diagonals_non_negative(self, model) -> None:
        """Rates out of a state cannot be negative."""
        _, m, _, _ = model
        Q = np.asarray(m.getQ(), dtype = float)
        off = Q[~np.eye(4, dtype = bool)]
        assert (off >= 0.0).all()

    def test_time_reversible(self, model) -> None:
        """``pi_i Q_ij == pi_j Q_ji``: the defining property of this family."""
        _, m, pi, _ = model
        Q = np.asarray(m.getQ(), dtype = float)
        flux = pi[:, None] * Q
        assert np.allclose(flux, flux.T, atol = TOL)

    def test_pi_is_stationary(self, model) -> None:
        """The supplied base frequencies must be Q's equilibrium."""
        _, m, pi, _ = model
        Q = np.asarray(m.getQ(), dtype = float)
        assert np.allclose(pi @ Q, 0.0, atol = TOL)

    def test_unit_mean_rate(self, model) -> None:
        """Normalization makes t a branch length in substitutions per site."""
        _, m, pi, _ = model
        Q = np.asarray(m.getQ(), dtype = float)
        assert -float(pi @ np.diag(Q)) == pytest.approx(1.0, abs = TOL)


class TestRateIndexing:
    """
    The exchangeability list is AC, AG, AT, CG, CT, GT (upper-triangular
    row-major). A scrambled mapping is the easiest way to silently break every
    model at once, so pin the mapping down cell by cell.
    """

    # (parameter index, the pair of states it must control)
    CELLS = [(0, (0, 1)), (1, (0, 2)), (2, (0, 3)),
             (3, (1, 2)), (4, (1, 3)), (5, (2, 3))]

    @pytest.mark.parametrize("idx,cell", CELLS)
    def test_parameter_controls_its_own_cell(self, idx, cell) -> None:
        """Bumping one rate must change exactly its own (symmetric) pair."""
        base = [1.0] * 6
        bumped = list(base)
        bumped[idx] = 7.0

        # Uniform pi keeps Q symmetric so the comparison is unambiguous.
        ref = np.asarray(GTR([0.25] * 4, base).getQ(), dtype = float)
        got = np.asarray(GTR([0.25] * 4, bumped).getQ(), dtype = float)

        i, j = cell
        changed = ~np.isclose(got, ref, atol = 1e-12)
        # Normalization rescales everything, so instead of "only these cells
        # moved", assert the bumped pair became the strict maximum.
        off = got[~np.eye(4, dtype = bool)]
        assert got[i][j] == pytest.approx(got[j][i], abs = TOL)
        assert got[i][j] == pytest.approx(off.max(), abs = TOL)
        assert changed.any()

    def test_all_six_parameters_are_used(self) -> None:
        """No exchangeability may be ignored, and none may be double-counted."""
        seen = set()
        for idx in range(6):
            rates = [1.0] * 6
            rates[idx] = 3.0
            Q = np.asarray(GTR([0.25] * 4, rates).getQ(), dtype = float)
            i, j = np.unravel_index(
                np.argmax(np.where(np.eye(4, dtype = bool), -np.inf, Q)),
                (4, 4),
            )
            seen.add(frozenset((int(i), int(j))))
        assert len(seen) == 6

    def test_transitions_are_ag_and_ct(self) -> None:
        """
        Indices 1 and 4 must be the transitions A<->G and C<->T. Every DNA
        subclass's equivalency pattern depends on this.
        """
        rates = [1.0, 9.0, 1.0, 1.0, 9.0, 1.0]
        Q = np.asarray(GTR([0.25] * 4, rates).getQ(), dtype = float)
        # A<->G is (0,2); C<->T is (1,3).
        assert Q[0][2] == pytest.approx(Q[1][3], abs = TOL)
        assert Q[0][2] > Q[0][1]
        assert Q[0][2] > Q[0][3]
        assert Q[1][3] > Q[1][2]


# ---------------------------------------------------------------------------
# P(t) invariants, and the closed forms
# ---------------------------------------------------------------------------

class TestTransitionMatrix:
    """Properties of ``P(t) = e^(Qt)``, including the closed-form shortcuts."""

    @pytest.mark.parametrize("t", TIMES)
    def test_matches_matrix_exponential(self, model, t) -> None:
        """
        The headline check: closed forms must reproduce expm(Q t) exactly.

        Six models skip the matrix exponential entirely, so this is what proves
        each shortcut is algebraically right rather than merely plausible.
        """
        _, m, _, _ = model
        Q = np.asarray(m.getQ(), dtype = float)
        got = np.asarray(m.expt(t), dtype = float)
        assert np.allclose(got, expm(Q * t), atol = TOL)

    @pytest.mark.parametrize("t", TIMES)
    def test_row_stochastic(self, model, t) -> None:
        """Each row is a probability distribution over destination states."""
        _, m, _, _ = model
        P = np.asarray(m.expt(t), dtype = float)
        assert np.allclose(P.sum(axis = 1), 1.0, atol = TOL)
        assert (P >= -1e-15).all()
        assert (P <= 1.0 + 1e-15).all()

    def test_identity_at_zero(self, model) -> None:
        """No time, no change."""
        _, m, _, _ = model
        assert np.allclose(np.asarray(m.expt(0.0), dtype = float),
                           np.eye(4), atol = TOL)

    def test_converges_to_stationary(self, model) -> None:
        """As t grows, every row forgets its origin and converges to pi."""
        _, m, pi, _ = model
        P = np.asarray(m.expt(500.0), dtype = float)
        assert np.allclose(P, np.tile(pi, (4, 1)), atol = 1e-8)

    @pytest.mark.parametrize("s,t", [(0.01, 0.02), (0.1, 0.3),
                                     (0.5, 0.5), (1.0, 2.0)])
    def test_chapman_kolmogorov(self, model, s, t) -> None:
        """``P(s) P(t) == P(s + t)`` for a genuine Markov semigroup."""
        _, m, _, _ = model
        left = np.asarray(m.expt(s), dtype = float) \
               @ np.asarray(m.expt(t), dtype = float)
        right = np.asarray(m.expt(s + t), dtype = float)
        assert np.allclose(left, right, atol = TOL)

    def test_negative_time_clamped(self, model) -> None:
        """Negative branch lengths must not produce a non-stochastic matrix."""
        _, m, _, _ = model
        assert np.allclose(np.asarray(m.expt(-1.0), dtype = float),
                           np.eye(4), atol = TOL)

    def test_reversibility_of_p(self, model) -> None:
        """Detailed balance carries over to P: ``pi_i P_ij == pi_j P_ji``."""
        _, m, pi, _ = model
        P = np.asarray(m.expt(0.4), dtype = float)
        flux = pi[:, None] * P
        assert np.allclose(flux, flux.T, atol = TOL)

    def test_repeated_calls_are_consistent(self, model) -> None:
        """The cached eigendecomposition must not corrupt later results."""
        _, m, _, _ = model
        first = np.asarray(m.expt(0.3), dtype = float).copy()
        m.expt(1.7)
        m.expt(0.0)
        assert np.allclose(np.asarray(m.expt(0.3), dtype = float),
                           first, atol = TOL)


class TestClosedForms:
    """The closed forms against their textbook statements, written out here."""

    @pytest.mark.parametrize("t", [0.05, 0.3, 1.0, 4.0])
    def test_jc_textbook(self, t) -> None:
        """JC: ``P_ii = 1/4 + 3/4 e^(-4t/3)``, ``P_ij = 1/4 - 1/4 e^(-4t/3)``."""
        P = np.asarray(JC().expt(t), dtype = float)
        e = math.exp(-4.0 * t / 3.0)
        assert P[0][0] == pytest.approx(0.25 + 0.75 * e, abs = TOL)
        assert P[0][1] == pytest.approx(0.25 - 0.25 * e, abs = TOL)

    @pytest.mark.parametrize("kappa", [1.0, 2.0, 4.0, 10.0])
    def test_k80_textbook(self, kappa) -> None:
        """
        K80 (Kimura 1980), in Kimura's own notation: transition rate alpha and
        transversion rate beta, normalized so ``alpha + 2 beta == 1``.

            P(same)        = 1/4 + 1/4 e^(-4 beta t)
                                 + 1/2 e^(-2 (alpha + beta) t)
            P(transition)  = 1/4 + 1/4 e^(-4 beta t)
                                 - 1/2 e^(-2 (alpha + beta) t)
            P(transversion)= 1/4 - 1/4 e^(-4 beta t)
        """
        m = K80(kappa)
        alpha, beta = m.alpha, m.beta

        for t in (0.1, 0.7, 2.5):
            P = np.asarray(m.expt(t), dtype = float)
            e1 = math.exp(-4.0 * beta * t)
            e2 = math.exp(-2.0 * (alpha + beta) * t)
            assert P[0][0] == pytest.approx(0.25 + 0.25 * e1 + 0.5 * e2,
                                           abs = TOL)
            # A<->G is a transition, A<->C a transversion.
            assert P[0][2] == pytest.approx(0.25 + 0.25 * e1 - 0.5 * e2,
                                           abs = TOL)
            assert P[0][1] == pytest.approx(0.25 - 0.25 * e1, abs = TOL)

    @pytest.mark.parametrize("t", [0.05, 0.3, 1.0, 4.0])
    def test_f81_textbook(self, t) -> None:
        """F81: ``P_ij = pi_j + (delta_ij - pi_j) e^(-ut)``, u = 1/(1-sum pi^2)."""
        m = F81(PI)
        pi = np.asarray(PI, dtype = float)
        u = 1.0 / (1.0 - float(pi @ pi))
        expected = math.exp(-u * t) * np.eye(4) \
                   + (1.0 - math.exp(-u * t)) * np.tile(pi, (4, 1))
        assert np.allclose(np.asarray(m.expt(t), dtype = float),
                           expected, atol = TOL)

    def test_f81_reduces_to_jc(self) -> None:
        """F81 with uniform frequencies is exactly JC (u = 4/3)."""
        for t in (0.1, 0.9, 3.0):
            assert np.allclose(np.asarray(F81([0.25] * 4).expt(t), dtype = float),
                               np.asarray(JC().expt(t), dtype = float),
                               atol = TOL)


class TestK80Parameterization:
    """
    K80 is parameterized by kappa, the transition/transversion ratio, and
    exposes Kimura's alpha (transition rate) and beta (transversion rate) as
    derived normalized rates.
    """

    KAPPAS = [0.5, 1.0, 2.0, 4.0, 10.0]

    @pytest.mark.parametrize("kappa", KAPPAS)
    def test_alpha_over_beta_is_kappa(self, kappa) -> None:
        """The defining relationship: ``kappa == alpha / beta``."""
        m = K80(kappa)
        assert m.alpha / m.beta == pytest.approx(kappa, abs = TOL)

    @pytest.mark.parametrize("kappa", KAPPAS)
    def test_rates_are_normalized(self, kappa) -> None:
        """Unit mean rate for K80 means ``alpha + 2 beta == 1``."""
        m = K80(kappa)
        assert m.alpha + 2.0 * m.beta == pytest.approx(1.0, abs = TOL)

    @pytest.mark.parametrize("kappa", KAPPAS)
    def test_rates_are_the_q_entries(self, kappa) -> None:
        """
        alpha and beta are Kimura's *rate matrix* entries, so they must equal
        the corresponding cells of Q rather than the raw exchangeabilities.
        """
        m = K80(kappa)
        Q = np.asarray(m.getQ(), dtype = float)
        assert Q[0][2] == pytest.approx(m.alpha, abs = TOL)   # A<->G transition
        assert Q[1][3] == pytest.approx(m.alpha, abs = TOL)   # C<->T transition
        assert Q[0][1] == pytest.approx(m.beta, abs = TOL)    # A<->C transversion
        assert Q[0][3] == pytest.approx(m.beta, abs = TOL)    # A<->T transversion

    @pytest.mark.parametrize("kappa", KAPPAS)
    def test_transitions_exceed_transversions(self, kappa) -> None:
        """kappa > 1 must mean transitions really are the faster substitution."""
        m = K80(kappa)
        if kappa > 1.0:
            assert m.alpha > m.beta
        elif kappa < 1.0:
            assert m.alpha < m.beta
        else:
            assert m.alpha == pytest.approx(m.beta, abs = TOL)

    @pytest.mark.parametrize("kappa", KAPPAS)
    def test_rate_pair_recovers_kappa(self, kappa) -> None:
        """
        A caller holding Kimura's two rates converts by dividing, and the
        overall scale of the pair must not matter since normalization absorbs
        it. This is the documented migration path from the old rate-pair form.
        """
        ref = K80(kappa)
        for scale in (0.25, 1.0, 17.0):
            m = K80((kappa * scale) / scale)
            assert m.kappa == pytest.approx(kappa, abs = TOL)
            assert np.allclose(np.asarray(m.getQ(), dtype = float),
                               np.asarray(ref.getQ(), dtype = float),
                               atol = TOL)

    def test_kappa_one_is_jukes_cantor(self) -> None:
        """kappa = 1 removes the transition bias entirely."""
        m = K80(1.0)
        assert m.alpha == pytest.approx(m.beta, abs = TOL)
        assert np.allclose(np.asarray(m.getQ(), dtype = float),
                           np.asarray(JC().getQ(), dtype = float), atol = TOL)

    def test_kappa_is_keyword_addressable(self) -> None:
        """``K80(kappa = 4)`` reads better at call sites than a bare number."""
        assert np.allclose(np.asarray(K80(kappa = 4.0).getQ(), dtype = float),
                           np.asarray(K80(4.0).getQ(), dtype = float),
                           atol = TOL)

    def test_alpha_and_beta_are_read_only(self) -> None:
        """
        kappa is the single source of truth. alpha and beta are derived from it,
        so assigning to them must fail rather than leave Q disagreeing with the
        rates the caller thinks are in effect.
        """
        m = K80(4.0)
        for name in ("alpha", "beta"):
            with pytest.raises(AttributeError):
                setattr(m, name, 0.5)

    def test_old_two_positional_form_fails_loudly(self) -> None:
        """
        The previous signature was ``K80(alpha, beta)`` with alpha meaning
        *transversion* -- the reverse of Kimura's notation. Since the argument
        order carried the opposite meaning, the old call must raise rather than
        silently invert a user's transition/transversion ratio.
        """
        with pytest.raises(TypeError):
            K80(0.2, 0.8)


class TestModelNesting:
    """
    Every model degenerates to a simpler one under the right parameters. These
    catch parameter-to-cell mix-ups that the matrix invariants cannot see.
    """

    JC_EQUIVALENTS = [
        ("K80",  lambda: K80(1.0)),
        ("F81",  lambda: F81([0.25] * 4)),
        ("HKY",  lambda: HKY([0.25] * 4, 1.0)),
        ("K81",  lambda: K81(1.0, 1.0, 1.0)),
        ("SYM",  lambda: SYM([1.0] * 6)),
        ("TN93", lambda: TN93([0.25] * 4, 1.0, 1.0)),
        ("GTR",  lambda: GTR([0.25] * 4, [1.0] * 6)),
    ]

    @pytest.mark.parametrize("name,build", JC_EQUIVALENTS,
                             ids = [n for n, _ in JC_EQUIVALENTS])
    def test_collapses_to_jc(self, name, build) -> None:
        """All rates equal and pi uniform means the model *is* Jukes-Cantor."""
        m = build()
        for t in (0.1, 0.7, 2.0):
            assert np.allclose(np.asarray(m.expt(t), dtype = float),
                               np.asarray(JC().expt(t), dtype = float),
                               atol = TOL), f"{name} != JC at t={t}"

    def test_tn93_with_equal_ratios_is_hky(self) -> None:
        """TN93 collapses to HKY when kappa_r == kappa_y."""
        a, b = TN93(PI, 3.0, 3.0), HKY(PI, 3.0)
        for t in (0.1, 0.6, 2.0):
            assert np.allclose(np.asarray(a.expt(t), dtype = float),
                               np.asarray(b.expt(t), dtype = float),
                               atol = TOL)

    def test_hky_with_uniform_pi_is_k80(self) -> None:
        """HKY with uniform pi is K80 at the same transition/transversion ratio."""
        hky, k80 = HKY([0.25] * 4, 4.0), K80(4.0)
        for t in (0.1, 0.6, 2.0):
            assert np.allclose(np.asarray(hky.expt(t), dtype = float),
                               np.asarray(k80.expt(t), dtype = float),
                               atol = TOL)

    def test_f81_is_hky_with_kappa_one(self) -> None:
        """F81 is HKY with no transition/transversion bias."""
        f81, hky = F81(PI), HKY(PI, 1.0)
        for t in (0.1, 0.6, 2.0):
            assert np.allclose(np.asarray(f81.expt(t), dtype = float),
                               np.asarray(hky.expt(t), dtype = float),
                               atol = TOL)

    def test_k81_with_equal_transversions_is_k80(self) -> None:
        """K81's two transversion classes merging gives K80."""
        k81, k80 = K81(4.0, 1.0, 1.0), K80(4.0)
        for t in (0.1, 0.6, 2.0):
            assert np.allclose(np.asarray(k81.expt(t), dtype = float),
                               np.asarray(k80.expt(t), dtype = float),
                               atol = TOL)


class TestEngineAgreement:
    """
    Cross-validate against ``phynetpy._seq_likelihood``, the independently
    written model used by the sequence-likelihood inference engine.
    """

    def test_gtr_matches_engine(self) -> None:
        """Free pi and all six rates free."""
        from phynetpy._seq_likelihood import GTR as EngineGTR
        rates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        mine, theirs = GTR(PI, rates), EngineGTR(PI, rates)
        assert np.allclose(np.asarray(mine.getQ(), dtype = float),
                           theirs.Q, atol = TOL)
        for t in (0.1, 0.5, 2.0):
            assert np.allclose(np.asarray(mine.expt(t), dtype = float),
                               theirs.p_matrix(t), atol = TOL)

    def test_jc_matches_engine(self) -> None:
        """JC against the engine's JC69."""
        from phynetpy._seq_likelihood import JC69
        mine, theirs = JC(), JC69()
        for t in (0.1, 0.5, 2.0):
            assert np.allclose(np.asarray(mine.expt(t), dtype = float),
                               theirs.p_matrix(t), atol = TOL)

    def test_hky_matches_engine(self) -> None:
        """HKY against the engine's HKY85, which is parameterized by kappa."""
        from phynetpy._seq_likelihood import HKY85
        kappa = 3.5
        mine, theirs = HKY(PI, kappa), HKY85(kappa, PI)
        assert np.allclose(np.asarray(mine.getQ(), dtype = float),
                           theirs.Q, atol = TOL)
        for t in (0.1, 0.5, 2.0):
            assert np.allclose(np.asarray(mine.expt(t), dtype = float),
                               theirs.p_matrix(t), atol = TOL)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestInputHandling:
    """Construction, validation, and mutation of model parameters."""

    def test_frequencies_with_float_rounding_accepted(self) -> None:
        """
        ``[0.4, 0.3, 0.2, 0.1]`` sums to 0.9999999999999999 in IEEE-754, so an
        exact ``== 1`` test would reject a perfectly ordinary input.
        """
        freqs = [0.4, 0.3, 0.2, 0.1]
        assert sum(freqs) != 1.0            # guards the premise of this test
        HKY(freqs, 4.0)
        F81(freqs)
        TN93(freqs, 4.0, 5.0)
        GTR(freqs, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_k80_accepts_any_positive_kappa(self) -> None:
        """
        Normalization absorbs the overall scale, so kappa is unconstrained apart
        from being positive. No value should need to be pre-scaled by the caller.
        """
        for kappa in (1e-6, 0.5, 1.0, 2.0, 5.0, 100.0, 1e6):
            m = K80(kappa)
            assert m.kappa == pytest.approx(kappa)

    def test_column_vector_input_accepted(self) -> None:
        """
        Column vectors iterate as length-1 arrays rather than scalars, which
        used to poison Q. They must be flattened instead.
        """
        m = GTR(np.ones((4, 1)) * 0.25, np.ones((6, 1)))
        Q = np.asarray(m.getQ(), dtype = float)
        assert Q.shape == (4, 4)
        assert np.allclose(Q, np.asarray(JC().getQ(), dtype = float),
                           atol = TOL)

    def test_hyperparams_roundtrip(self) -> None:
        """``get_hyperparams`` returns flat float lists."""
        freqs, trans = GTR(PI, [1., 2., 3., 4., 5., 6.]).get_hyperparams()
        assert [type(x) for x in freqs] == [float] * 4
        assert [type(x) for x in trans] == [float] * 6

    def test_state_count(self) -> None:
        """DNA models report 4 states."""
        assert JC().state_count() == 4

    @pytest.mark.parametrize("freqs", [
        [0.25, 0.25, 0.25],              # too short
        [0.5, 0.5, 0.5, 0.5],            # does not sum to 1
        [0.25, 0.25, 0.25, 0.25, 0.0],   # too long
    ])
    def test_bad_frequencies_rejected(self, freqs) -> None:
        """Malformed frequency vectors must raise, not silently misbehave."""
        with pytest.raises(SubstitutionModelError):
            GTR(freqs, [1.0] * 6)

    @pytest.mark.parametrize("rates", [[1.0] * 5, [1.0] * 7, []])
    def test_bad_transition_count_rejected(self, rates) -> None:
        """A wrong number of exchangeabilities must raise."""
        with pytest.raises(SubstitutionModelError):
            GTR([0.25] * 4, rates)

    @pytest.mark.parametrize("kappa", [0.0, -1.0, -0.5])
    def test_k80_rejects_non_positive_kappa(self, kappa) -> None:
        """A zero or negative rate ratio is not a model."""
        with pytest.raises(SubstitutionModelError):
            K80(kappa)

    @pytest.mark.parametrize("build", [
        lambda: K80(),
        lambda: HKY(PI),
        lambda: TN93(PI, 2.0),
        lambda: K81(2.0, 1.0),
    ], ids = ["K80", "HKY", "TN93", "K81"])
    def test_rate_parameters_are_required(self, build) -> None:
        """No model silently guesses a rate parameter the caller omitted."""
        with pytest.raises(TypeError):
            build()

    @pytest.mark.parametrize("build", [
        lambda: HKY(PI, 3.0),
        lambda: TN93(PI, 3.0, 3.0),
        lambda: K80(3.0),
        lambda: K81(3.0, 1.0, 1.0),
    ], ids = ["HKY", "TN93", "K80", "K81"])
    def test_constrained_models_cannot_break_their_pattern(self, build) -> None:
        """
        Taking named rate parameters instead of a 6-element list makes an
        invalid equivalency pattern unrepresentable: the four transversions and
        the two transitions come out tied by construction, with no validation
        needed.
        """
        m = build()
        _, trans = m.get_hyperparams()
        assert len(trans) == 6
        # Transversions are indices 0, 2, 3, 5; transitions are 1 and 4.
        assert len({trans[0], trans[2], trans[3], trans[5]}) == 1
        assert trans[1] == trans[4]

    def test_tn93_allows_unequal_transition_ratios(self) -> None:
        """The whole point of TN93: kappa_r may differ from kappa_y."""
        m = TN93(PI, 5.0, 2.0)
        _, trans = m.get_hyperparams()
        assert trans[1] == pytest.approx(5.0)     # A<->G, purine
        assert trans[4] == pytest.approx(2.0)     # C<->T, pyrimidine
        Q = np.asarray(m.getQ(), dtype = float)
        # Dividing out the destination-frequency factor recovers the ratio gap.
        assert Q[0][2] / PI[2] != pytest.approx(Q[1][3] / PI[3], abs = 1e-9)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_rate_parameters_must_be_positive_and_finite(self, bad) -> None:
        """Every named rate parameter is validated the same way."""
        for build in (lambda: K80(bad),
                      lambda: HKY(PI, bad),
                      lambda: TN93(PI, bad, 2.0),
                      lambda: TN93(PI, 2.0, bad),
                      lambda: K81(bad, 1.0, 1.0),
                      lambda: K81(1.0, bad, 1.0),
                      lambda: K81(1.0, 1.0, bad)):
            with pytest.raises(SubstitutionModelError):
                build()

    def test_rate_parameters_must_be_numbers(self) -> None:
        """A non-numeric rate raises the module's own error type."""
        with pytest.raises(SubstitutionModelError):
            K80("fast")


class TestSetHyperparams:
    """Mutating a model must rebuild Q and invalidate any cached decomposition."""

    def test_gtr_updates_q(self) -> None:
        """Changing rates changes Q."""
        m = GTR(PI, [1.0] * 6)
        before = np.asarray(m.getQ(), dtype = float).copy()
        m.set_hyperparams({"transitions": [1.0, 5.0, 1.0, 1.0, 5.0, 1.0]})
        assert not np.allclose(before, np.asarray(m.getQ(), dtype = float))

    def test_gtr_matches_fresh_instance(self) -> None:
        """A mutated model equals one built with the new parameters."""
        m = GTR(PI, [1.0] * 6)
        rates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        m.set_hyperparams({"transitions": rates})
        fresh = GTR(PI, rates)
        assert np.allclose(np.asarray(m.getQ(), dtype = float),
                           np.asarray(fresh.getQ(), dtype = float), atol = TOL)
        assert np.allclose(np.asarray(m.expt(0.5), dtype = float),
                           np.asarray(fresh.expt(0.5), dtype = float),
                           atol = TOL)

    def test_expt_cache_invalidated(self) -> None:
        """A stale eigendecomposition must not survive a parameter change."""
        m = GTR(PI, [1.0] * 6)
        m.expt(0.5)                      # populate the cache
        rates = [1.0, 6.0, 1.0, 1.0, 6.0, 1.0]
        m.set_hyperparams({"transitions": rates})
        assert np.allclose(np.asarray(m.expt(0.5), dtype = float),
                           np.asarray(GTR(PI, rates).expt(0.5), dtype = float),
                           atol = TOL)

    # Each model paired with an update and the equivalent fresh construction.
    ROUNDTRIPS = [
        ("K80",  lambda: K80(1.0),          {"kappa": 4.0},
                 lambda: K80(4.0)),
        ("F81",  lambda: F81(PI),           {"base frequencies": [0.25] * 4},
                 lambda: F81([0.25] * 4)),
        ("HKY",  lambda: HKY(PI, 1.0),      {"kappa": 3.5},
                 lambda: HKY(PI, 3.5)),
        ("TN93", lambda: TN93(PI, 1.0, 1.0),
                 {"kappa_r": 5.0, "kappa_y": 2.0},
                 lambda: TN93(PI, 5.0, 2.0)),
        ("K81",  lambda: K81(1.0, 1.0, 1.0),
                 {"alpha": 5.0, "beta": 1.0, "gamma": 2.0},
                 lambda: K81(5.0, 1.0, 2.0)),
        ("SYM",  lambda: SYM([1.0] * 6),
                 {"transitions": [1., 2., 3., 4., 5., 6.]},
                 lambda: SYM([1., 2., 3., 4., 5., 6.])),
        ("GTR",  lambda: GTR(PI, [1.0] * 6),
                 {"transitions": [1., 2., 3., 4., 5., 6.]},
                 lambda: GTR(PI, [1., 2., 3., 4., 5., 6.])),
    ]

    @pytest.mark.parametrize("name,build,update,expected", ROUNDTRIPS,
                             ids = [r[0] for r in ROUNDTRIPS])
    def test_update_matches_fresh_instance(self, name, build, update,
                                          expected) -> None:
        """
        Updating a model must leave it indistinguishable from one built with the
        new values -- in Q *and* in P(t), which also proves the cached
        eigendecomposition and any closed-form parameters were refreshed.
        """
        m, fresh = build(), expected()
        m.expt(0.5)                      # populate the cache before mutating
        m.set_hyperparams(update)
        assert np.allclose(np.asarray(m.getQ(), dtype = float),
                           np.asarray(fresh.getQ(), dtype = float), atol = TOL)
        for t in (0.1, 0.5, 2.0):
            assert np.allclose(np.asarray(m.expt(t), dtype = float),
                               np.asarray(fresh.expt(t), dtype = float),
                               atol = TOL)

    def test_k81_partial_update_keeps_other_rates(self) -> None:
        """Naming one K81 rate class must not disturb the other two."""
        m = K81(5.0, 1.0, 2.0)
        before = (m.beta, m.gamma)
        m.set_hyperparams({"alpha": 9.0})
        # Values are renormalized, so compare the ratio rather than the value.
        assert m.beta / m.gamma == pytest.approx(before[0] / before[1],
                                                abs = TOL)
        assert m.alpha / m.beta == pytest.approx(9.0 / 1.0, abs = TOL)

    def test_invalid_update_raises(self) -> None:
        """Validation applies on update, not just construction."""
        with pytest.raises(SubstitutionModelError):
            K80(2.0).set_hyperparams({"kappa": -1.0})
        with pytest.raises(SubstitutionModelError):
            HKY(PI, 2.0).set_hyperparams({"base frequencies": [0.5] * 4})


class TestHyperparamNames:
    """
    Every model declares the parameter names it accepts. An unrecognized name
    must raise rather than be ignored: silently dropping it would leave the
    caller believing a change took effect, and for the closed-form models it
    could desync ``getQ()`` from ``expt()``.
    """

    EXPECTED = {
        "JC":   (),
        "K80":  ("kappa",),
        "F81":  ("base frequencies",),
        "HKY":  ("kappa", "base frequencies"),
        "TN93": ("kappa_r", "kappa_y", "base frequencies"),
        "K81":  ("alpha", "beta", "gamma"),
        "SYM":  ("transitions",),
        "GTR":  ("states", "base frequencies", "transitions"),
    }

    def test_declared_names(self, model) -> None:
        """Each model advertises exactly its own free parameters."""
        name, m, _, _ = model
        assert set(m.HYPERPARAMS) == set(self.EXPECTED[name])

    def test_unknown_name_raises(self, model) -> None:
        """A misspelled or foreign parameter name is an error."""
        _, m, _, _ = model
        with pytest.raises(SubstitutionModelError):
            m.set_hyperparams({"not_a_real_parameter": 1.0})

    def test_declared_names_are_accepted(self, model) -> None:
        """Conversely, every declared name must be settable."""
        name, m, pi, _ = model
        values = {
            "kappa": 2.0, "kappa_r": 2.0, "kappa_y": 3.0,
            "alpha": 2.0, "beta": 1.0, "gamma": 1.5,
            "states": 4, "base frequencies": list(pi),
            "transitions": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
        }
        m.set_hyperparams({k: values[k] for k in m.HYPERPARAMS})

    def test_jc_rejects_all_changes(self) -> None:
        """
        JC has no free parameters. It previously inherited the general setter,
        so changing "transitions" would alter Q while ``expt`` kept returning
        Jukes-Cantor probabilities -- a silent disagreement between the two.
        """
        jc = JC()
        assert jc.HYPERPARAMS == ()
        for params in ({"transitions": [1.0, 5.0, 1.0, 1.0, 5.0, 1.0]},
                       {"base frequencies": [0.4, 0.3, 0.2, 0.1]},
                       {"kappa": 4.0}):
            with pytest.raises(SubstitutionModelError):
                jc.set_hyperparams(params)

    def test_q_and_expt_stay_in_sync(self, model) -> None:
        """
        After any accepted update, the closed forms must still agree with the
        matrix exponential of the *current* Q. This is the invariant that the
        name checking exists to protect.
        """
        _, m, pi, _ = model
        values = {
            "kappa": 3.0, "kappa_r": 4.0, "kappa_y": 2.0,
            "alpha": 4.0, "beta": 1.0, "gamma": 2.0,
            "states": 4, "base frequencies": list(pi),
            "transitions": [1.0, 3.0, 2.0, 1.0, 4.0, 1.0],
        }
        m.set_hyperparams({k: values[k] for k in m.HYPERPARAMS})
        Q = np.asarray(m.getQ(), dtype = float)
        for t in (0.1, 0.7, 2.0):
            assert np.allclose(np.asarray(m.expt(t), dtype = float),
                               expm(Q * t), atol = TOL)


class TestK81RateClasses:
    """
    K81's three rate classes must reach the correct nucleotide pairs. A
    mis-mapping here still yields a valid reversible Q, so only pair-by-pair
    checks catch it.
    """

    ALPHA, BETA, GAMMA = 5.0, 1.0, 2.0

    def test_classes_map_to_correct_pairs(self) -> None:
        """alpha on the transitions, beta on A-C/G-T, gamma on A-T/C-G."""
        m = K81(self.ALPHA, self.BETA, self.GAMMA)
        Q = np.asarray(m.getQ(), dtype = float)
        # Uniform pi and unit mean rate make the normalized rates sum to 1,
        # so each Q entry is its rate divided by the rate total.
        total = self.ALPHA + self.BETA + self.GAMMA
        a, b, g = (self.ALPHA / total, self.BETA / total, self.GAMMA / total)
        # Transitions: A<->G and C<->T.
        assert Q[0][2] == pytest.approx(a, abs = TOL)
        assert Q[1][3] == pytest.approx(a, abs = TOL)
        # beta class: A<->C and G<->T.
        assert Q[0][1] == pytest.approx(b, abs = TOL)
        assert Q[2][3] == pytest.approx(b, abs = TOL)
        # gamma class: A<->T and C<->G.
        assert Q[0][3] == pytest.approx(g, abs = TOL)
        assert Q[1][2] == pytest.approx(g, abs = TOL)

    def test_rates_round_trip(self) -> None:
        """
        Rates read back exactly as passed. Storing them raw is what lets a
        partial update be interpreted against the rates already in place.
        """
        m = K81(self.ALPHA, self.BETA, self.GAMMA)
        assert (m.alpha, m.beta, m.gamma) == (self.ALPHA, self.BETA,
                                              self.GAMMA)

    @pytest.mark.parametrize("scale", [0.1, 1.0, 7.0, 1000.0])
    def test_scale_invariant(self, scale) -> None:
        """Only the ratios between the three rates are identifiable."""
        ref = np.asarray(K81(self.ALPHA, self.BETA, self.GAMMA).getQ(),
                         dtype = float)
        got = np.asarray(K81(self.ALPHA * scale, self.BETA * scale,
                             self.GAMMA * scale).getQ(), dtype = float)
        assert np.allclose(got, ref, atol = TOL)

    def test_three_distinct_classes(self) -> None:
        """
        With distinct rates the three classes must stay distinct, or a
        collapsed class would hide a parameter-to-cell mix-up.
        """
        Q = np.asarray(K81(self.ALPHA, self.BETA, self.GAMMA).getQ(),
                       dtype = float)
        assert len({round(Q[0][2], 12), round(Q[0][1], 12),
                    round(Q[0][3], 12)}) == 3


class TestScaleInvariance:
    """
    Q is normalized to unit mean rate, so the overall magnitude of any rate
    argument is discarded. Scaling all rates must leave the model unchanged.
    """

    @pytest.mark.parametrize("scale", [0.01, 0.5, 3.0, 250.0])
    def test_gtr_rates_scale_free(self, scale) -> None:
        """GTR's six exchangeabilities are only identifiable up to scale."""
        base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ref = np.asarray(GTR(PI, base).getQ(), dtype = float)
        got = np.asarray(GTR(PI, [r * scale for r in base]).getQ(),
                         dtype = float)
        assert np.allclose(got, ref, atol = TOL)

    @pytest.mark.parametrize("scale", [0.01, 0.5, 3.0, 250.0])
    def test_sym_rates_scale_free(self, scale) -> None:
        """Same for SYM."""
        base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ref = np.asarray(SYM(base).getQ(), dtype = float)
        got = np.asarray(SYM([r * scale for r in base]).getQ(), dtype = float)
        assert np.allclose(got, ref, atol = TOL)
