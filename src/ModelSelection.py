"""
Author : Mark Kessler
Last Stable Edit : 4/24/26
First Included in Version : 0.3.2

Docs   - [x]
Tests  - [ ]
Design - [x]

Model-selection helpers for phylogenetic network search.

This module provides a generic, search-method-agnostic way to pick the
number of reticulations that is actually justified by the data.  Given
any search procedure that (a) accepts a ``max_reticulations=k`` knob
and (b) returns the best log-likelihood (or log-pseudo-likelihood) it
found, :func:`reticulation_sweep` runs the search over a range of
``k`` values, collects the best score at each ``k``, and reports the
trade-off in three complementary ways:

    1. Raw log-likelihood curve (for the slope/elbow heuristic).
    2. AIC = ``2p - 2 logL``.
    3. BIC = ``p ln(n) - 2 logL``.

where ``p`` is the parameter count
(``base_params + params_per_reticulation * k``) and ``n`` is the data
size (for MPL, typically the number of gene trees).

Note on MPL (pseudo-likelihood): AIC/BIC are strictly derived for true
likelihoods.  Applying them to the log-pseudo-likelihood produced by
MPL is a pragmatic approximation -- the ranking over ``k`` is still
informative, but the absolute AIC/BIC values should not be compared to
those of a fully-specified likelihood model.  The slope/elbow heuristic
has no such caveat and is often the most honest of the three.

Typical use with the pseudo-likelihood criterion::

    from phynetpy.infer import infer
    from phynetpy.criteria import PseudoLikelihood
    from phynetpy.ModelSelection import reticulation_sweep

    def run_k(k: int, seed: int) -> float:
        return infer(
            gene_trees, criterion=PseudoLikelihood(),
            method="sa", num_iter=20000, max_reticulations=k, seed=seed,
        ).score

    result = reticulation_sweep(
        run_k, k_values=[0, 1, 2, 3],
        seeds=[42, 7, 101],
        data_size=len(gene_trees.trees),
        params_per_reticulation=3,
    )
    result.print_summary()
    result.save_csv("runs/retic_sweep.csv")
    result.plot("runs/retic_sweep.png")
    print("recommended k by BIC:", result.best_by("bic"))
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence


# Selection rule accepted by :meth:`SweepResult.best_by`.  Unrelated to
# :class:`phynetpy.criteria.Criterion`: this names how to pick ``k`` from a
# finished sweep, not what objective the sweep optimised.
Criterion = Literal["logL", "aic", "bic", "elbow"]


@dataclass
class SweepRow:
    """One row of the reticulation sweep: results for a single ``k``.

    Attributes:
        k: Reticulation count this row was produced with.
        best_log_lik: Best log-likelihood observed across seeds at
            this ``k`` (higher is better for MPL).
        all_log_liks: Per-seed log-likelihoods, in the order the seeds
            were evaluated.  Useful for diagnosing multimodality.
        n_params: Effective parameter count used for AIC/BIC.
        aic: ``2 * n_params - 2 * best_log_lik``.
        bic: ``n_params * ln(data_size) - 2 * best_log_lik``.
        elapsed_s: Wall-clock time spent on this ``k`` (all seeds).
        delta_log_lik: Marginal gain over the previous ``k`` row; set
            by :func:`reticulation_sweep` after all rows are built.
            ``None`` for the first row.
    """

    k: int
    best_log_lik: float
    all_log_liks: List[float] = field(default_factory=list)
    n_params: int = 0
    aic: float = 0.0
    bic: float = 0.0
    elapsed_s: float = 0.0
    delta_log_lik: Optional[float] = None


@dataclass
class SweepResult:
    """Container for reticulation-sweep rows with selection/plotting helpers.

    Built by :func:`reticulation_sweep`.  Exposes
    :meth:`best_by` for criterion-based ``k`` recommendation,
    :meth:`print_summary` for a console table, :meth:`save_csv` for a
    machine-readable dump, and :meth:`plot` for a visual report.

    Attributes:
        rows: Ordered list of :class:`SweepRow`, one per ``k``.
        data_size: ``n`` used in the BIC formula.
        params_per_reticulation: Parameters added by each reticulation.
        base_params: Backbone (k=0) parameter count.
        log_lik_label: Human-readable name for the y-axis / summary
            column (e.g. ``"log-pseudo-likelihood"``).
    """

    rows: List[SweepRow]
    data_size: int
    params_per_reticulation: int
    base_params: int
    log_lik_label: str = "log-likelihood"

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def best_by(self, criterion: Criterion) -> int:
        """Return the recommended ``k`` under the given criterion.

        Args:
            criterion: One of

                * ``"logL"``:  ``argmax_k logL(k)`` (ignores parsimony;
                  always picks the largest ``k`` in a non-overfit
                  regime).
                * ``"aic"``:   ``argmin_k AIC(k)``.
                * ``"bic"``:   ``argmin_k BIC(k)``.
                * ``"elbow"``: smallest ``k`` at which the next-step
                  gain in log-likelihood falls below
                  ``elbow_tol_frac`` of the maximum gain across the
                  sweep.  This matches the classic "knee plot"
                  heuristic.

        Returns:
            Recommended reticulation count.

        Raises:
            ValueError: If the sweep is empty or ``criterion`` is
                unrecognised.
        """
        if not self.rows:
            raise ValueError("empty SweepResult")
        if criterion == "logL":
            return max(self.rows, key=lambda r: r.best_log_lik).k
        if criterion == "aic":
            return min(self.rows, key=lambda r: r.aic).k
        if criterion == "bic":
            return min(self.rows, key=lambda r: r.bic).k
        if criterion == "elbow":
            return self._elbow_k()
        raise ValueError(
            f"unknown criterion {criterion!r}; expected one of "
            "'logL', 'aic', 'bic', 'elbow'"
        )

    def _elbow_k(self, elbow_tol_frac: float = 0.1) -> int:
        """Return the knee/elbow ``k`` from the log-likelihood curve.

        Walks the marginal gains ``logL(k_{i+1}) - logL(k_i)`` left to
        right and returns the first ``k_i`` whose next-step gain
        drops below ``elbow_tol_frac`` times the largest marginal
        gain in the sweep.  Falls back to the largest ``k`` when the
        curve never flattens.

        Args:
            elbow_tol_frac: Fraction of the maximum marginal gain
                used as the "flat" threshold.  Default 0.1 means
                "accept the current k when the next step contributes
                less than 10% of the best step seen".

        Returns:
            Recommended ``k`` under the elbow heuristic.
        """
        if len(self.rows) < 2:
            return self.rows[0].k
        deltas = [r.delta_log_lik or 0.0 for r in self.rows[1:]]
        max_d = max(deltas) if deltas else 0.0
        if max_d <= 0:
            return self.rows[0].k
        threshold = elbow_tol_frac * max_d
        for i, d in enumerate(deltas):
            if d < threshold:
                return self.rows[i].k
        return self.rows[-1].k

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def print_summary(self, file=None) -> None:
        """Pretty-print the sweep table, deltas, and recommendations.

        Args:
            file: Optional text stream to print to.  Defaults to
                ``sys.stdout``.
        """
        from sys import stdout

        out = file or stdout
        header = (
            f"{'k':>3}  {'best_logL':>14}  {'delta':>10}  "
            f"{'params':>6}  {'AIC':>12}  {'BIC':>12}  {'time_s':>8}"
        )
        print(header, file=out)
        print("-" * len(header), file=out)
        for r in self.rows:
            delta_str = (
                f"{r.delta_log_lik:+10.4f}"
                if r.delta_log_lik is not None
                else f"{'-':>10}"
            )
            print(
                f"{r.k:>3}  {r.best_log_lik:>14.4f}  {delta_str}  "
                f"{r.n_params:>6}  {r.aic:>12.4f}  {r.bic:>12.4f}  "
                f"{r.elapsed_s:>8.1f}",
                file=out,
            )
        print(file=out)
        print(
            f"  recommended k (argmax logL) : {self.best_by('logL')}",
            file=out,
        )
        print(f"  recommended k (min AIC)     : {self.best_by('aic')}", file=out)
        print(f"  recommended k (min BIC)     : {self.best_by('bic')}", file=out)
        print(f"  recommended k (elbow)       : {self.best_by('elbow')}", file=out)
        print(
            f"  (data_size={self.data_size}, "
            f"params/retic={self.params_per_reticulation}, "
            f"base_params={self.base_params})",
            file=out,
        )

    def save_csv(self, path: str | Path) -> None:
        """Write the sweep rows to ``path`` as CSV, one row per ``k``.

        Parent directories are created on demand.  Per-seed scores are
        joined into a single ``;``-separated field so the CSV stays
        flat.

        Args:
            path: Output CSV path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "k", "best_log_lik", "delta_log_lik", "n_params",
                "aic", "bic", "elapsed_s", "all_log_liks",
            ])
            for r in self.rows:
                w.writerow([
                    r.k,
                    f"{r.best_log_lik:.6f}",
                    ("" if r.delta_log_lik is None
                     else f"{r.delta_log_lik:.6f}"),
                    r.n_params,
                    f"{r.aic:.6f}",
                    f"{r.bic:.6f}",
                    f"{r.elapsed_s:.3f}",
                    ";".join(f"{x:.6f}" for x in r.all_log_liks),
                ])

    def plot(
        self,
        path: str | Path | None = None,
        *,
        show: bool = False,
        title: str | None = None,
    ) -> None:
        """Render the log-likelihood / AIC / BIC curves versus ``k``.

        Writes a PNG to ``path`` (if provided) and/or opens an
        interactive window (``show=True``).  Each recommended ``k``
        is marked with a vertical dashed line colour-matched to its
        criterion.

        Args:
            path: Optional PNG output path.
            show: When True, also display an interactive window.
            title: Optional figure title.
        """
        import matplotlib.pyplot as plt  # type: ignore

        ks = [r.k for r in self.rows]
        logLs = [r.best_log_lik for r in self.rows]
        aics = [r.aic for r in self.rows]
        bics = [r.bic for r in self.rows]

        fig, (ax_lik, ax_ic) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

        ax_lik.plot(ks, logLs, marker="o", color="tab:blue",
                    label=self.log_lik_label)
        best_lik_k = self.best_by("logL")
        ax_lik.axvline(best_lik_k, color="tab:blue", alpha=0.3,
                       linestyle="--",
                       label=f"argmax logL (k={best_lik_k})")
        elbow_k = self.best_by("elbow")
        ax_lik.axvline(elbow_k, color="tab:green", alpha=0.3,
                       linestyle="--",
                       label=f"elbow (k={elbow_k})")
        ax_lik.set_ylabel(self.log_lik_label)
        ax_lik.grid(True, alpha=0.3)
        ax_lik.legend(loc="lower right")

        ax_ic.plot(ks, aics, marker="s", color="tab:orange", label="AIC")
        ax_ic.plot(ks, bics, marker="^", color="tab:red", label="BIC")
        best_aic_k = self.best_by("aic")
        best_bic_k = self.best_by("bic")
        ax_ic.axvline(best_aic_k, color="tab:orange", alpha=0.3,
                      linestyle="--", label=f"min AIC (k={best_aic_k})")
        ax_ic.axvline(best_bic_k, color="tab:red", alpha=0.3,
                      linestyle="--", label=f"min BIC (k={best_bic_k})")
        ax_ic.set_xlabel("reticulations (k)")
        ax_ic.set_ylabel("information criterion (lower is better)")
        ax_ic.grid(True, alpha=0.3)
        ax_ic.legend(loc="lower right")

        if title:
            fig.suptitle(title)
        fig.tight_layout()

        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def reticulation_sweep(
    search_fn: Callable[[int, int], float],
    k_values: Sequence[int],
    *,
    seeds: Sequence[int] = (0,),
    data_size: int = 1,
    params_per_reticulation: int = 3,
    base_params: int = 0,
    log_lik_label: str = "log-likelihood",
    progress: bool = True,
) -> SweepResult:
    """Run ``search_fn`` over each ``k in k_values`` and summarize.

    Args:
        search_fn: Callable ``f(k, seed) -> float`` that performs one
            search with ``max_reticulations == k`` and returns the best
            log-likelihood found.  The caller is responsible for
            constructing a fresh search object / starting network for
            each invocation if that's appropriate for the method.
        k_values: Reticulation counts to sweep over
            (e.g. ``range(0, 4)``).
        seeds: RNG seeds to run at each ``k``.  If multiple seeds are
            provided the best (highest) log-likelihood across seeds is
            taken as the representative score for that ``k``.
        data_size: ``n`` used in BIC (``p ln(n) - 2 logL``).  For MPL
            this is typically ``len(gene_trees.trees)``.
        params_per_reticulation: Number of free parameters each
            additional reticulation contributes.  For MPL, 3 (one
            gamma + two new branch lengths) is a sensible default;
            1 (gamma only) is the most conservative choice.
        base_params: Parameters attributable to the backbone tree
            (commonly ``2 * n_taxa - 3`` for an unrooted binary tree;
            pass ``0`` to ignore -- only differences across ``k``
            matter for the AIC/BIC comparison.)
        log_lik_label: Y-axis label, e.g. "log-pseudo-likelihood".
        progress: Print a one-line progress update per search.

    Returns:
        A :class:`SweepResult` with the best score and derived stats at
        each ``k``.
    """
    if not k_values:
        raise ValueError("k_values must be non-empty")
    if not seeds:
        raise ValueError("seeds must be non-empty")

    rows: List[SweepRow] = []
    for k in k_values:
        # Run every seed at this k; take the max as the representative
        # score (we're interested in what the method *can* achieve at
        # a given k, not in the average of good and bad runs).
        scores: List[float] = []
        t0 = time.time()
        for seed in seeds:
            score = float(search_fn(k, seed))
            scores.append(score)
            if progress:
                print(
                    f"  [sweep] k={k}  seed={seed}  "
                    f"{log_lik_label}={score:.4f}",
                    flush=True,
                )
        elapsed = time.time() - t0
        best = max(scores)
        n_p = base_params + params_per_reticulation * k
        aic = 2.0 * n_p - 2.0 * best
        bic = n_p * math.log(max(data_size, 1)) - 2.0 * best
        rows.append(SweepRow(
            k=k,
            best_log_lik=best,
            all_log_liks=scores,
            n_params=n_p,
            aic=aic,
            bic=bic,
            elapsed_s=elapsed,
        ))

    # Backfill marginal gains now that every row exists.
    for i in range(1, len(rows)):
        rows[i].delta_log_lik = (
            rows[i].best_log_lik - rows[i - 1].best_log_lik
        )

    return SweepResult(
        rows=rows,
        data_size=data_size,
        params_per_reticulation=params_per_reticulation,
        base_params=base_params,
        log_lik_label=log_lik_label,
    )
