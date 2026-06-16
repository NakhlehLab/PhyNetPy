#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

"""
Author : Mark Kessler
First Included in Version : 0.5.0

Post-analysis of MCMC chains: convergence diagnostics and interop with the
standard phylogenetics tooling.

This module is the reporting layer for PhyNetPy's Bayesian samplers
(:class:`~phynetpy.infer.MCMC_SEQ`, :class:`~phynetpy.infer.MCMC_GT`).  It
provides two things:

1. **Tracer interoperability.**  :func:`write_tracer_log` emits a
   BEAST-style tab-delimited ``.log`` file that
   `Tracer <https://www.beast2.org/tracer-2/>`_ opens directly, and
   :func:`read_tracer_log` parses such files back into numeric traces (so you
   can re-analyse logs produced elsewhere).  :func:`write_trees_nexus` writes a
   NEXUS ``.trees`` file of the sampled networks for DensiTree / TreeAnnotator.

2. **Native diagnostics.**  :func:`effective_sample_size`,
   :func:`autocorrelation_time`, :func:`hpd_interval` and :func:`geweke`
   reproduce the numbers Tracer reports, using the same algorithms as BEAST's
   ``TraceCorrelation`` so the values agree.  :func:`summarize_traces` rolls
   these into a per-parameter :class:`ChainSummary` with a printable table, so
   users get solid reporting without leaving Python.

The diagnostics operate on a *trace table* -- a mapping from column name to a
sequence of per-sample real values -- so they are agnostic to which sampler
produced the chain.  Sampler result objects expose ``trace_table()`` /
``write_log()`` / ``summary()`` helpers that build the table for you.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

# Largest lag considered when estimating the autocorrelation time.  Matches
# BEAST's ``TraceCorrelation.MAX_LAG`` so ESS values line up with Tracer.
_MAX_LAG = 2000

# Tracer flags parameters whose ESS falls below this as poorly mixed.
LOW_ESS_THRESHOLD = 200.0


# ======================================================================
# Core diagnostics (faithful to BEAST 1's dr.inference.trace.TraceCorrelation)
# ======================================================================

def _as_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Trace values must be one-dimensional.")
    return arr


def _gamma_and_varstat(arr: np.ndarray) -> tuple[float, float]:
    """Return ``(gamma0, varStat)`` from BEAST's initial-positive-sequence sum.

    ``gamma0`` is the lag-0 autocovariance (the sample variance computed with
    an ``N`` denominator); ``varStat`` is ``gamma0`` plus twice the sum of the
    autocovariances accumulated under BEAST's "fancy stopping criterion": pairs
    of consecutive lags are added at even lags and the sum is truncated as soon
    as a pair turns non-positive (Geyer's initial positive sequence).
    """
    n = arr.size
    mean = float(arr.mean())
    dev = arr - mean
    max_lag = min(n - 1, _MAX_LAG)

    gamma = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        # autocovariance at this lag, normalised by (n - lag) as in BEAST
        gamma[lag] = float(np.dot(dev[: n - lag], dev[lag:])) / (n - lag)

    var_stat = gamma[0]
    for lag in range(1, max_lag + 1):
        if lag % 2 == 0:
            pair = gamma[lag - 1] + gamma[lag]
            if pair > 0.0:
                var_stat += 2.0 * pair
            else:
                break
    return float(gamma[0]), float(var_stat)


def effective_sample_size(values: Sequence[float]) -> float:
    """Effective sample size of a univariate trace.

    Uses BEAST/Tracer's estimator ``ESS = N * gamma(0) / varStat`` where
    ``varStat`` is the initial-positive-sequence autocovariance sum.  A
    perfectly independent chain yields ``ESS == N``; heavy autocorrelation
    drives it toward 1.

    Args:
        values: The sampled values of one parameter, in draw order.

    Returns:
        The effective sample size.  ``nan`` for a constant trace (no variance,
        ESS undefined -- Tracer shows a dash) and for fewer than two samples.
    """
    arr = _as_array(values)
    if arr.size < 2:
        return float("nan")
    gamma0, var_stat = _gamma_and_varstat(arr)
    if gamma0 == 0.0 or var_stat == 0.0:
        return float("nan")
    return float(arr.size * gamma0 / var_stat)


def autocorrelation_time(values: Sequence[float], step_size: int = 1) -> float:
    """Integrated autocorrelation time (in states).

    Args:
        values: The sampled values, in draw order.
        step_size: Number of MCMC states between consecutive samples (the
            sampling / thinning interval).  Scales the result into state units
            exactly as Tracer does.

    Returns:
        ``ACT = step_size * varStat / gamma(0)``; ``nan`` for a constant trace.
    """
    arr = _as_array(values)
    if arr.size < 2:
        return float("nan")
    gamma0, var_stat = _gamma_and_varstat(arr)
    if gamma0 == 0.0:
        return float("nan")
    return float(step_size * var_stat / gamma0)


def standard_error_of_mean(values: Sequence[float]) -> float:
    """Monte-Carlo standard error of the mean, ``sqrt(varStat / N)``.

    This is the autocorrelation-corrected error Tracer reports, *not* the naive
    ``stdev / sqrt(N)`` (which ignores between-sample dependence).
    """
    arr = _as_array(values)
    if arr.size < 2:
        return float("nan")
    _, var_stat = _gamma_and_varstat(arr)
    return float(math.sqrt(var_stat / arr.size)) if var_stat > 0 else 0.0


def hpd_interval(
    values: Sequence[float], prob: float = 0.95
) -> tuple[float, float]:
    """Highest-posterior-density (HPD) interval of a trace.

    The narrowest interval that contains a ``prob`` fraction of the samples --
    the same credible interval Tracer displays (e.g. "95% HPD interval").

    Args:
        values: The sampled values.
        prob: Target coverage in ``(0, 1]`` (default 0.95).

    Returns:
        ``(lower, upper)`` bounds.  For a single sample both equal that value.
    """
    if not 0.0 < prob <= 1.0:
        raise ValueError("prob must be in (0, 1].")
    arr = np.sort(_as_array(values))
    n = arr.size
    if n == 0:
        raise ValueError("Cannot compute an HPD interval of no samples.")
    if n == 1:
        return float(arr[0]), float(arr[0])
    n_in = max(1, int(math.ceil(prob * n)))
    if n_in >= n:
        return float(arr[0]), float(arr[-1])
    widths = arr[n_in - 1 :] - arr[: n - n_in + 1]
    k = int(np.argmin(widths))
    return float(arr[k]), float(arr[k + n_in - 1])


def geweke(
    values: Sequence[float], first: float = 0.1, last: float = 0.5
) -> float:
    """Geweke (1992) convergence z-score.

    Compares the mean of the first ``first`` fraction of the chain to the mean
    of the last ``last`` fraction, standardised by their (spectral) standard
    errors.  ``|z| < 2`` is consistent with convergence.

    Args:
        values: The sampled values, in draw order.
        first: Fraction of the chain taken from the start.
        last: Fraction taken from the end.

    Returns:
        The z-score, or ``nan`` if the chain is too short / degenerate.
    """
    arr = _as_array(values)
    n = arr.size
    if n < 4 or first <= 0 or last <= 0 or first + last > 1:
        return float("nan")
    a = arr[: max(2, int(first * n))]
    b = arr[n - max(2, int(last * n)) :]
    _, va = _gamma_and_varstat(a)
    _, vb = _gamma_and_varstat(b)
    se = math.sqrt(va / a.size + vb / b.size)
    if se == 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / se)


def gelman_rubin(chains: Sequence[Sequence[float]]) -> float:
    r"""Gelman-Rubin potential scale reduction factor (R-hat).

    The standard multi-chain convergence diagnostic (Gelman & Rubin 1992):
    run ``m >= 2`` independent chains from over-dispersed starting points,
    each of length ``n``, and compare the between-chain variance ``B`` to
    the within-chain variance ``W``:

    .. math::

        \hat{V} = \frac{n - 1}{n} W + \frac{B}{n}, \qquad
        \hat{R} = \sqrt{\hat{V} / W}.

    Values near ``1.0`` (a common rule of thumb is ``< 1.01`` or
    ``< 1.05``) indicate the chains have mixed to a common distribution;
    values well above 1 mean they have not converged.  Unlike single-chain
    diagnostics (ESS, Geweke) this directly detects multi-modal failure
    where each chain is internally stable but they disagree -- exactly the
    failure mode of a network sampler stuck in different topology basins.

    Args:
        chains: A sequence of ``m`` chains, each a sequence of per-sample
            values for one parameter (in draw order).  Chains may differ
            in length; the common minimum length is used.

    Returns:
        The R-hat statistic, or ``nan`` if fewer than two chains, fewer
        than two samples per chain, or the within-chain variance is zero.
    """
    arrs = [_as_array(c) for c in chains]
    m = len(arrs)
    if m < 2:
        return float("nan")
    n = min(a.size for a in arrs)
    if n < 2:
        return float("nan")
    arrs = [a[:n] for a in arrs]
    means = np.array([a.mean() for a in arrs], dtype=np.float64)
    variances = np.array([a.var(ddof=1) for a in arrs], dtype=np.float64)

    grand_mean = float(means.mean())
    between = n / (m - 1) * float(np.sum((means - grand_mean) ** 2))
    within = float(variances.mean())
    if within == 0.0:
        return float("nan")
    var_plus = (n - 1) / n * within + between / n
    return float(math.sqrt(var_plus / within))


# ======================================================================
# Per-parameter and per-chain summaries
# ======================================================================

@dataclass
class ParameterSummary:
    """Tracer-style summary statistics for one traced parameter.

    Attributes mirror the columns Tracer shows in its "Estimates" table.
    """

    name: str
    n: int
    mean: float
    stderr_of_mean: float
    stdev: float
    median: float
    lower_hpd: float
    upper_hpd: float
    hpd_prob: float
    ess: float
    act: float
    geweke_z: float
    minimum: float
    maximum: float

    @property
    def ess_ok(self) -> bool:
        """Whether ESS clears Tracer's poor-mixing threshold (200)."""
        return not math.isnan(self.ess) and self.ess >= LOW_ESS_THRESHOLD


def summarize(
    name: str,
    values: Sequence[float],
    *,
    step_size: int = 1,
    hpd_prob: float = 0.95,
) -> ParameterSummary:
    """Compute a :class:`ParameterSummary` for one trace."""
    arr = _as_array(values)
    n = arr.size
    if n == 0:
        raise ValueError(f"No samples for parameter {name!r}.")
    lo, hi = hpd_interval(arr, hpd_prob)
    return ParameterSummary(
        name=name,
        n=n,
        mean=float(arr.mean()),
        stderr_of_mean=standard_error_of_mean(arr),
        stdev=float(arr.std(ddof=1)) if n > 1 else 0.0,
        median=float(np.median(arr)),
        lower_hpd=lo,
        upper_hpd=hi,
        hpd_prob=hpd_prob,
        ess=effective_sample_size(arr),
        act=autocorrelation_time(arr, step_size),
        geweke_z=geweke(arr),
        minimum=float(arr.min()),
        maximum=float(arr.max()),
    )


@dataclass
class ChainSummary:
    """A table of :class:`ParameterSummary` objects for a whole chain."""

    parameters: dict[str, ParameterSummary] = field(default_factory=dict)
    n_samples: int = 0
    step_size: int = 1

    def __getitem__(self, name: str) -> ParameterSummary:
        return self.parameters[name]

    def __iter__(self):
        return iter(self.parameters.values())

    @property
    def min_ess(self) -> float:
        """Smallest ESS across all parameters (the binding diagnostic)."""
        vals = [p.ess for p in self.parameters.values() if not math.isnan(p.ess)]
        return min(vals) if vals else float("nan")

    @property
    def converged(self) -> bool:
        """Heuristic: every parameter clears the ESS threshold."""
        return all(p.ess_ok for p in self.parameters.values())

    def low_ess(self) -> list[str]:
        """Names of parameters whose ESS is below the threshold."""
        return [n for n, p in self.parameters.items() if not p.ess_ok]

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Plain nested dict (handy for JSON / DataFrame construction)."""
        return {
            n: {
                "mean": p.mean,
                "stderr_of_mean": p.stderr_of_mean,
                "stdev": p.stdev,
                "median": p.median,
                f"hpd{int(p.hpd_prob * 100)}_lower": p.lower_hpd,
                f"hpd{int(p.hpd_prob * 100)}_upper": p.upper_hpd,
                "ess": p.ess,
                "act": p.act,
                "geweke_z": p.geweke_z,
                "min": p.minimum,
                "max": p.maximum,
            }
            for n, p in self.parameters.items()
        }

    def __str__(self) -> str:
        hdr = (
            f"{'parameter':<18}{'mean':>12}{'stderr':>11}{'stdev':>11}"
            f"{'median':>12}{'95% HPD lo':>13}{'95% HPD hi':>13}"
            f"{'ESS':>10}{'mix':>6}"
        )
        lines = [
            f"Chain summary: {self.n_samples} samples, "
            f"sampling interval {self.step_size}",
            hdr,
            "-" * len(hdr),
        ]
        for p in self.parameters.values():
            ess = "nan" if math.isnan(p.ess) else f"{p.ess:.1f}"
            lines.append(
                f"{p.name:<18}{p.mean:>12.5g}{p.stderr_of_mean:>11.4g}"
                f"{p.stdev:>11.4g}{p.median:>12.5g}{p.lower_hpd:>13.5g}"
                f"{p.upper_hpd:>13.5g}{ess:>10}{'ok' if p.ess_ok else 'LOW':>6}"
            )
        low = self.low_ess()
        if low:
            lines.append("")
            lines.append(
                f"WARNING: low ESS (< {LOW_ESS_THRESHOLD:.0f}) for: "
                + ", ".join(low)
                + " -- run the chain longer."
            )
        return "\n".join(lines)


def summarize_traces(
    traces: Mapping[str, Sequence[float]],
    *,
    step_size: int = 1,
    hpd_prob: float = 0.95,
    skip: Iterable[str] = (),
) -> ChainSummary:
    """Summarize every column of a trace table.

    Args:
        traces: Mapping from parameter name to its per-sample values.
        step_size: Sampling / thinning interval (states between samples).
        hpd_prob: Coverage for the reported HPD interval.
        skip: Column names to exclude (e.g. the ``state`` index column).

    Returns:
        A :class:`ChainSummary`.
    """
    skip_set = set(skip)
    params: dict[str, ParameterSummary] = {}
    n_samples = 0
    for name, values in traces.items():
        if name in skip_set:
            continue
        params[name] = summarize(
            name, values, step_size=step_size, hpd_prob=hpd_prob
        )
        n_samples = max(n_samples, params[name].n)
    return ChainSummary(parameters=params, n_samples=n_samples, step_size=step_size)


# ======================================================================
# Tracer .log I/O
# ======================================================================

def write_tracer_log(
    states: Sequence[int],
    traces: Mapping[str, Sequence[float]],
    path: str,
    *,
    comments: Optional[Sequence[str]] = None,
    state_column: str = "state",
) -> None:
    """Write a BEAST/Tracer-compatible tab-delimited ``.log`` file.

    The first column is the integer chain state; remaining columns are the
    traced parameters in ``traces`` insertion order.  The file opens directly
    in Tracer.

    Args:
        states: Per-sample chain-state (iteration) indices.
        traces: Mapping from parameter name to per-sample values; every column
            must have the same length as ``states``.
        path: Output path (conventionally ending in ``.log``).
        comments: Optional ``#``-prefixed header lines (e.g. provenance).
        state_column: Name of the leading index column (Tracer expects
            ``state``).
    """
    n = len(states)
    cols = list(traces.keys())
    for c in cols:
        if len(traces[c]) != n:
            raise ValueError(
                f"Trace column {c!r} has {len(traces[c])} rows, expected {n}."
            )
    with open(path, "w", encoding="utf-8") as fh:
        if comments:
            for line in comments:
                fh.write(f"# {line}\n")
        fh.write(state_column + "\t" + "\t".join(cols) + "\n")
        for i in range(n):
            row = [str(int(states[i]))]
            for c in cols:
                v = traces[c][i]
                row.append(repr(float(v)) if v == v else "NaN")
            fh.write("\t".join(row) + "\n")


def read_tracer_log(path: str) -> tuple[list[int], dict[str, list[float]]]:
    """Read a BEAST/Tracer ``.log`` file.

    Skips ``#`` comment lines, treats the first remaining row as the header and
    its first column as the integer chain state.

    Args:
        path: Path to the ``.log`` file.

    Returns:
        ``(states, traces)`` where ``states`` is the list of state indices and
        ``traces`` maps each remaining column name to its list of floats.
    """
    states: list[int] = []
    header: Optional[list[str]] = None
    cols: dict[str, list[float]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                for name in header[1:]:
                    cols[name] = []
                continue
            states.append(int(float(fields[0])))
            for name, val in zip(header[1:], fields[1:]):
                cols[name].append(float(val))
    if header is None:
        raise ValueError(f"No data found in log file {path!r}.")
    return states, cols


# ======================================================================
# NEXUS .trees I/O (sampled networks; DensiTree / TreeAnnotator)
# ======================================================================

def write_trees_nexus(
    states: Sequence[int],
    newicks: Sequence[str],
    path: str,
    *,
    taxa: Optional[Sequence[str]] = None,
    prefix: str = "STATE",
) -> None:
    """Write sampled networks as a NEXUS ``TREES`` block.

    Produces a file consumable by DensiTree / TreeAnnotator and other NEXUS
    readers.  Reticulate (extended-Newick) strings are written verbatim, which
    plain tree viewers may not render but which round-trips losslessly.

    Args:
        states: Per-sample chain-state indices (used to name each tree).
        newicks: Extended-Newick strings, one per sample, aligned with
            ``states``.
        path: Output path (conventionally ``.trees`` or ``.nex``).
        taxa: Optional explicit taxon list for the ``TAXA`` block; if omitted no
            ``TAXA`` block is written (most readers cope).
        prefix: Tree-name prefix; tree ``i`` is named ``<prefix>_<state>``.
    """
    if len(states) != len(newicks):
        raise ValueError("states and newicks must have equal length.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#NEXUS\n\n")
        if taxa:
            fh.write("BEGIN TAXA;\n")
            fh.write(f"    DIMENSIONS NTAX={len(taxa)};\n")
            fh.write("    TAXLABELS " + " ".join(taxa) + ";\n")
            fh.write("END;\n\n")
        fh.write("BEGIN TREES;\n")
        for st, nwk in zip(states, newicks):
            s = nwk.strip()
            if not s.endswith(";"):
                s += ";"
            fh.write(f"    TREE {prefix}_{int(st)} = {s}\n")
        fh.write("END;\n")
