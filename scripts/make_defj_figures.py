#!/usr/bin/env python3
"""
Build accuracy and runtime figures + LaTeX summary tables for the DEFJ
benchmark, comparing MP-Allop-2 (Hill Climbing and Simulated Annealing x3)
against PhyloNet's InferNetwork_MP_Allopp.

Reads:
    runs/defj/mp_allop_results.csv   (from benchmark_defj.py)
    runs/defj/mp_w*.csv              (parallel worker shards, auto-merged)
    runs/defj/phylonet_results.csv   (from run_phylonet_defj.py)

Writes PNGs and .tex tables into paper_figures/. Tolerant of partial / missing
data: if the PhyloNet CSV is absent, only the MP-Allop methods are plotted.

Run::

    .venv/Scripts/python.exe scripts/make_defj_figures.py

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import ptitprince as pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defj_common as dc  # noqa: E402

ROOT = dc.project_root()
RUNS = ROOT / "runs" / "defj"
OUT = ROOT / "paper_figures"

METHOD_ORDER = ["MP-HC", "MP-SA3", "PhyloNet"]
# Set at runtime; when True, the PhyloNet CSV is ignored entirely.
MP_ONLY = False
# Set at runtime by --mp-csv. When non-empty, exactly these MP-Allop CSVs are
# read and the default mp_allop_results.csv + mp_w*.csv shards are ignored,
# so a fresh sweep is never silently merged with an older one.
MP_CSVS: list[Path] = []

# ── Colour palette (colour-blind friendly, Nature-style) ──────────────────
METHOD_COLORS = {
    "MP-HC":    "#0072B2",   # deep blue
    "MP-SA3":   "#009E73",   # teal green
    "PhyloNet": "#D55E00",   # vermillion
}
SCEN_ORDER = ["D", "E", "F", "J"]

# ── Box-plot glyphs ──────────────────────────────────────────────────────
# A box plot shows the *median* as the line inside the box, which is easy to
# miss and easy to mistake for the mean. Every box plot here therefore also
# draws the mean as a white diamond with a dark edge: filled-light-on-dark
# reads clearly against both the teal and vermillion fills at 8 pt / 300 dpi.
#
# Outliers used to be diamonds too, which made a flier sitting near the box
# indistinguishable from the mean marker. They are now crosses, so the two
# glyph shapes carry different meanings.
MEAN_PROPS = {
    "marker": "D",
    "markerfacecolor": "white",
    "markeredgecolor": "#1a1a1a",
    "markeredgewidth": 0.7,
    "markersize": 3.4,
}
FLIER_PROPS = {
    "marker": "x",
    "markerfacecolor": "none",
    "markeredgecolor": "#4d4d4d",
    "markeredgewidth": 0.55,
    "markersize": 2.8,
}


MEAN_LEGEND_LABEL = "Mean"
# Spelled out in the figure title rather than the legend: a long legend label
# widens an outside legend enough that tight_layout collapses the axes.
MEAN_NOTE = "mean = white diamond, median = line in box"


def _mean_legend_handle():
    """Proxy artist so the mean marker can be named in a legend."""
    return plt.Line2D([], [], linestyle="none",
                      label=MEAN_LEGEND_LABEL, **MEAN_PROPS)


def _savefig(fig, out: Path, attempts: int = 6, **kwargs) -> None:
    """Save ``fig``, retrying past transient OS file locks.

    The repo lives under a OneDrive-synced ``Documents`` tree with Defender
    active, and either can briefly hold a handle on a PNG the moment it is
    written. That surfaces here as ``OSError: [Errno 22] Invalid argument``
    from PIL's ``open(path, "w+b")`` on a path that is perfectly valid, and it
    lands on a different figure each run. Retrying clears it.
    """
    for attempt in range(attempts):
        try:
            fig.savefig(out, **kwargs)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.4 * (attempt + 1))


def _shared_legend(fig, hue_order: list[str]) -> None:
    """Replace per-axes legends with one figure-level legend below the panels.

    Faceted figures here use small panels; anchoring a legend outside a single
    axes makes ``tight_layout`` shrink that axes until the figure size goes
    degenerate and ``savefig`` fails with EINVAL. One legend for the whole
    figure avoids that, and is the conventional layout for a panel grid.
    """
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS.get(m, "#888888"),
                      edgecolor="#333333", linewidth=0.7, label=m)
        for m in hue_order
    ]
    labels = list(hue_order)
    handles.append(_mean_legend_handle())
    labels.append(MEAN_LEGEND_LABEL)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=True, bbox_to_anchor=(0.5, -0.02))


def _add_mean_to_legend(ax, outside: bool = True) -> None:
    """Re-draw ``ax``'s legend with the mean-marker entry appended."""
    leg = ax.get_legend()
    if leg is None:
        return
    handles = list(leg.legend_handles)
    labels = [t.get_text() for t in leg.get_texts()]
    handles.append(_mean_legend_handle())
    labels.append(MEAN_LEGEND_LABEL)
    if outside:
        ax.legend(handles, labels, title="", loc="upper left",
                  bbox_to_anchor=(1.02, 1))
    else:
        ax.legend(handles, labels, title="", loc="best")

# ── Global matplotlib style (Nature / PLOS ONE aesthetic) ────────────────
FONT_SIZE   = 8      # pt  – matches typical journal body text
TITLE_SIZE  = 9
LABEL_SIZE  = 8
TICK_SIZE   = 7
LEGEND_SIZE = 7
DPI         = 300

def _apply_style() -> None:
    """Apply a clean, publication-ready rcParams style."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          FONT_SIZE,
        "axes.titlesize":     TITLE_SIZE,
        "axes.labelsize":     LABEL_SIZE,
        "xtick.labelsize":    TICK_SIZE,
        "ytick.labelsize":    TICK_SIZE,
        "legend.fontsize":    LEGEND_SIZE,
        "legend.frameon":     True,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#cccccc",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.7,
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        "axes.grid":          True,
        "grid.color":         "#e5e5e5",
        "grid.linewidth":     0.5,
        "grid.linestyle":     "-",
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })


def _numify(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_data() -> pd.DataFrame:
    """Return a long-form frame with a unified 'method' column."""
    frames = []

    if MP_CSVS:
        for path in MP_CSVS:
            frames.append(pd.read_csv(path))
    else:
        # Primary MP-Allop CSV (single-process runs).
        mp_path = RUNS / "mp_allop_results.csv"
        if mp_path.exists():
            frames.append(pd.read_csv(mp_path))

        # Parallel worker shards (benchmark_defj.py --out runs/defj/mp_wN.csv).
        for shard in sorted(RUNS.glob("mp_w*.csv")):
            frames.append(pd.read_csv(shard))

    if frames:
        mp = pd.concat(frames, ignore_index=True, sort=False)
        # Drop duplicate keys if a merged file overlaps worker shards.
        key = ["tier", "scenario", "g", "n", "t", "r", "method"]
        mp = mp.drop_duplicates(subset=key, keep="last")
        mp = mp[(mp["error"].isna()) | (mp["error"].astype(str) == "")]
        mp = _numify(mp, ["mu_d", "hw_d", "final_pars", "seconds",
                          "g", "n", "t", "r", "tier", "n_retics"])
        mp["method"] = mp["method"].map({"HC": "MP-HC", "SA3": "MP-SA3"})
        frames = [mp]
    else:
        frames = []

    pn_path = RUNS / "phylonet_results.csv"
    if pn_path.exists() and not MP_ONLY:
        pn = pd.read_csv(pn_path)
        pn = pn[(pn["error"].isna()) | (pn["error"].astype(str) == "")]
        pn = _numify(pn, ["mu_d", "hw_d", "final_pars", "seconds",
                          "g", "n", "t", "r", "tier", "n_retics"])
        pn["method"] = "PhyloNet"
        frames.append(pn)

    if not frames:
        raise SystemExit("No result CSVs found in runs/defj/")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.dropna(subset=["mu_d"])
    return df


def _present_methods(df: pd.DataFrame) -> list[str]:
    return [m for m in METHOD_ORDER if m in set(df["method"])]


def _method_palette(methods: list[str]) -> dict[str, str]:
    return {m: METHOD_COLORS[m] for m in methods if m in METHOD_COLORS}


# ── Metric display helpers ────────────────────────────────────────────────
METRIC_LABELS = {
    "mu_d":  r"$\mu$-distance to true network",
    "hw_d":  "Hardwired-cluster distance to true network",
}
METRIC_TITLES = {
    "mu_d":  r"Species-network error ($\mu$-distance)",
    "hw_d":  "Species-network error (hardwired-cluster distance)",
}

def _ylabel(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)

def _fig_title(metric: str, subtitle: str = "") -> str:
    base = METRIC_TITLES.get(metric, metric)
    return f"{base}\n{subtitle}" if subtitle else base


# ── Shared raincloud helper ───────────────────────────────────────────────
def _raincloud(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    x_order: list,
    hue_order: list[str],
    orient: str = "v",
) -> None:
    """
    Raincloud = half-violin (ptitprince) + raw jittered strip + box overlay.
    Each method is drawn separately at a slight horizontal offset so they
    don't overlap.
    """
    palette = _method_palette(hue_order)
    # ptitprince width and offset bookkeeping
    width_viol = 0.35
    # draw raincloud with dodge
    pt.RainCloud(
        data=data,
        x=x,
        y=y,
        hue="method",
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        orient=orient,
        width_viol=width_viol,
        width_box=0.12,
        move=0.18,
        point_size=1.8,
        alpha=0.70,
        dodge=True,
        linewidth=0.8,
        box_linewidth=0.8,
        box_showmeans=True,
        box_meanprops=MEAN_PROPS,
        box_flierprops=FLIER_PROPS,
        ax=ax,
    )


# ── Figure A: accuracy by scenario (raincloud) ───────────────────────────
def fig_accuracy_by_scenario(df: pd.DataFrame, metric: str) -> None:
    """Raincloud plot: every run, grouped by scenario and method (n=1)."""
    _apply_style()
    sub = df[df["n"] == 1].copy()
    if sub.empty:
        return
    scen   = [s for s in SCEN_ORDER if s in set(sub["scenario"])]
    hue_order = _present_methods(sub)

    # one panel per scenario, shared y-axis
    fig, axes = plt.subplots(
        1, len(scen),
        figsize=(2.5 * len(scen), 3.6),
        sharey=True, squeeze=False,
    )
    for k, s in enumerate(scen):
        ax = axes[0][k]
        d = sub[sub["scenario"] == s].copy()
        d["_x"] = "  "   # single-group raincloud; method encoded by hue
        _raincloud(ax, d, x="_x", y=metric, x_order=["  "],
                   hue_order=hue_order)
        ax.set_xlabel(f"Scenario {s}", labelpad=4)
        ax.set_title(f"({chr(65 + k)})", loc="left", fontsize=TITLE_SIZE,
                     fontweight="bold")
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)
        if k == 0:
            ax.set_ylabel(_ylabel(metric), labelpad=6)
        else:
            ax.set_ylabel("")
        # legend only on last panel
        leg = ax.get_legend()
        if k < len(scen) - 1:
            if leg: leg.remove()
        else:
            if leg:
                leg.set_title("")
                for lh in leg.legend_handles:
                    lh.set_alpha(1.0)
                    lh._sizes = [25]
                _add_mean_to_legend(ax, outside=False)

    fig.suptitle(_fig_title(metric, f"single individual (n = 1); {MEAN_NOTE}"),
                 y=1.01, fontsize=TITLE_SIZE)
    fig.tight_layout(w_pad=0.3)
    out = OUT / f"defj_accuracy_by_scenario_{metric}.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


# ── Figure B: accuracy vs ILS (faceted box plots) ────────────────────────
def fig_accuracy_vs_ils(df: pd.DataFrame, metric: str) -> None:
    """Faceted box plots: error vs ILS level, one panel per scenario (n=1, g=1)."""
    _apply_style()
    sub = df[
        (df["n"] == 1) & (df["g"] == 1)
        & (df["scenario"].isin(["D", "E", "F"]))
    ].copy()
    if sub.empty:
        print("  (skip accuracy_vs_ils: no D/E/F g1 n1 data yet)")
        return
    sub["t_label"] = sub["t"].astype(int).map({4: "Low\n(t=4)", 20: "Med\n(t=20)",
                                                100: "High\n(t=100)"})
    t_order   = ["Low\n(t=4)", "Med\n(t=20)", "High\n(t=100)"]
    scen      = [s for s in ["D", "E", "F"] if s in set(sub["scenario"])]
    hue_order = _present_methods(sub)
    palette   = _method_palette(hue_order)

    fig, axes = plt.subplots(
        1, len(scen),
        figsize=(2.6 * len(scen), 3.4),
        sharey=True, squeeze=False,
    )
    for k, s in enumerate(scen):
        ax = axes[0][k]
        d  = sub[sub["scenario"] == s]
        sns.boxplot(
            data=d, x="t_label", y=metric,
            hue="method", order=t_order, hue_order=hue_order,
            palette=palette,
            width=0.55, linewidth=0.8,
            showmeans=True, meanprops=MEAN_PROPS, flierprops=FLIER_PROPS,
            ax=ax,
        )
        ax.set_title(f"({chr(65 + k)}) Scenario {s}", loc="left",
                     fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("ILS level", labelpad=4)
        if k == 0:
            ax.set_ylabel(_ylabel(metric), labelpad=6)
        else:
            ax.set_ylabel("")
        leg = ax.get_legend()
        if leg:
            leg.remove()

    _shared_legend(fig, hue_order)
    fig.suptitle(_fig_title(metric, f"1 gene tree, n = 1; {MEAN_NOTE}"),
                 y=1.02, fontsize=TITLE_SIZE)
    fig.tight_layout(w_pad=0.4, rect=(0, 0.06, 1, 1))
    out = OUT / f"defj_accuracy_vs_ils_{metric}.png"
    _savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── Figure C: accuracy vs gene count (faceted box plots) ─────────────────
def fig_accuracy_vs_genes(df: pd.DataFrame, metric: str) -> None:
    """Faceted box plots: error vs gene count, one panel per scenario (n=1)."""
    _apply_style()
    sub = df[df["n"] == 1].copy()
    if sub.empty:
        return
    sub["g_label"] = sub["g"].astype(int).astype(str)
    g_order   = [str(g) for g in sorted(sub["g"].dropna().astype(int).unique())]
    scen      = [s for s in SCEN_ORDER if s in set(sub["scenario"])]
    hue_order = _present_methods(sub)
    palette   = _method_palette(hue_order)

    fig, axes = plt.subplots(
        1, len(scen),
        figsize=(2.6 * len(scen), 3.6),
        sharey=True, squeeze=False,
    )
    for k, s in enumerate(scen):
        ax = axes[0][k]
        d  = sub[sub["scenario"] == s]
        sns.boxplot(
            data=d, x="g_label", y=metric,
            hue="method", order=g_order, hue_order=hue_order,
            palette=palette,
            width=0.55, linewidth=0.8,
            showmeans=True, meanprops=MEAN_PROPS, flierprops=FLIER_PROPS,
            ax=ax,
        )
        ax.set_title(f"({chr(65 + k)}) Scenario {s}", loc="left",
                     fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Number of gene trees", labelpad=4)
        if k == 0:
            ax.set_ylabel(_ylabel(metric), labelpad=6)
        else:
            ax.set_ylabel("")
        leg = ax.get_legend()
        if leg:
            leg.remove()

    _shared_legend(fig, hue_order)
    fig.suptitle(_fig_title(metric, f"single individual (n = 1); {MEAN_NOTE}"),
                 y=1.02, fontsize=TITLE_SIZE)
    fig.tight_layout(w_pad=0.4, rect=(0, 0.06, 1, 1))
    out = OUT / f"defj_accuracy_vs_genes_{metric}.png"
    _savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── Figure D: ILS x gene-count grid ──────────────────────────────────────
#
# The marginal figures above each collapse one of the two factors that drive
# accuracy: ``fig_accuracy_vs_genes`` pools all three ILS levels into every
# box, and ``fig_accuracy_vs_ils`` fixes g = 1. Pooling ILS is what makes
# MP-Allop's spread look large -- t = 4 and t = 100 are close to disjoint
# distributions (at 10 genes, ~94% of low-ILS runs recover the network
# exactly while no high-ILS run does), so a single box over both is a
# mixture, not a measure of run-to-run stability. This grid separates them.

ILS_LABELS = {4: "Low ILS (t=4)", 20: "Med ILS (t=20)", 100: "High ILS (t=100)"}


def _grid(df: pd.DataFrame, value: str, log_y: bool):
    """Shared ILS-row x scenario-column panel grid; returns (fig, ok)."""
    sub = df[(df["n"] == 1) & (df["scenario"].isin(["D", "E", "F"]))].copy()
    sub = sub.dropna(subset=[value, "t", "g"])
    if sub.empty:
        return None, False

    sub["g_label"] = sub["g"].astype(int).astype(str)
    g_order   = [str(g) for g in sorted(sub["g"].astype(int).unique())]
    ils       = [t for t in (4, 20, 100) if t in set(sub["t"].astype(int))]
    scen      = [s for s in ["D", "E", "F"] if s in set(sub["scenario"])]
    hue_order = _present_methods(sub)
    palette   = _method_palette(hue_order)

    fig, axes = plt.subplots(
        len(ils), len(scen),
        figsize=(2.5 * len(scen), 2.2 * len(ils)),
        sharey=True, sharex=True, squeeze=False,
    )
    for i, t in enumerate(ils):
        for j, s in enumerate(scen):
            ax = axes[i][j]
            d = sub[(sub["t"].astype(int) == t) & (sub["scenario"] == s)]
            if not d.empty:
                sns.boxplot(
                    data=d, x="g_label", y=value,
                    hue="method", order=g_order, hue_order=hue_order,
                    palette=palette, width=0.6, linewidth=0.7,
                    showmeans=True, meanprops=MEAN_PROPS,
                    flierprops=FLIER_PROPS,
                    ax=ax,
                )
            if log_y:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                    lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}"
                ))
            if i == 0:
                ax.set_title(f"Scenario {s}", fontsize=TITLE_SIZE,
                             fontweight="bold")
            ax.set_xlabel("Number of gene trees" if i == len(ils) - 1 else "",
                          labelpad=4)
            ax.set_ylabel(ILS_LABELS[t] if j == 0 else "", labelpad=6)
    _shared_legend(fig, hue_order)
    return fig, True


def fig_accuracy_grid(df: pd.DataFrame, metric: str) -> None:
    """Accuracy with ILS and gene count separated: ILS rows, scenario columns."""
    _apply_style()
    fig, ok = _grid(df, metric, log_y=False)
    if not ok:
        print(f"  (skip accuracy_grid_{metric}: no D/E/F n1 data yet)")
        return
    fig.suptitle(
        f"{_ylabel(metric)} by ILS level and gene count (n = 1)\n{MEAN_NOTE}",
        y=1.005, fontsize=TITLE_SIZE,
    )
    fig.tight_layout(w_pad=0.3, h_pad=0.5, rect=(0, 0.05, 1, 1))
    out = OUT / f"defj_accuracy_grid_{metric}.png"
    _savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_runtime_grid(df: pd.DataFrame) -> None:
    """Runtime with ILS and gene count separated: ILS rows, scenario columns."""
    _apply_style()
    fig, ok = _grid(df, "seconds", log_y=True)
    if not ok:
        print("  (skip runtime_grid: no D/E/F n1 data yet)")
        return
    for ax in fig.axes:
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel() + "\nseconds (log)", labelpad=6)
    fig.suptitle("Wall-clock runtime by ILS level and gene count "
                 f"(n = 1)\n{MEAN_NOTE}",
                 y=1.005, fontsize=TITLE_SIZE)
    fig.tight_layout(w_pad=0.3, h_pad=0.5, rect=(0, 0.05, 1, 1))
    out = OUT / "defj_runtime_grid.png"
    _savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── Runtime figures (box plots, log scale) ───────────────────────────────
def fig_runtime_vs_genes(df: pd.DataFrame) -> None:
    _apply_style()
    sub = df.dropna(subset=["seconds"]).copy()
    if sub.empty:
        return
    sub["g_label"]   = sub["g"].astype(int).astype(str)
    g_order          = [str(g) for g in sorted(sub["g"].dropna().astype(int).unique())]
    hue_order        = _present_methods(sub)
    palette          = _method_palette(hue_order)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    sns.boxplot(
        data=sub, x="g_label", y="seconds",
        hue="method", order=g_order, hue_order=hue_order,
        palette=palette, width=0.55, linewidth=0.8,
        showmeans=True, meanprops=MEAN_PROPS, flierprops=FLIER_PROPS,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}"
    ))
    ax.set_xlabel("Number of gene trees", labelpad=4)
    ax.set_ylabel("Wall-clock time (s, log scale)", labelpad=6)
    ax.set_title(f"Runtime vs gene count\n({MEAN_NOTE})", fontsize=TITLE_SIZE)
    _add_mean_to_legend(ax, outside=False)
    fig.tight_layout()
    out = OUT / "defj_runtime_vs_genes.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_runtime_by_scenario(df: pd.DataFrame) -> None:
    _apply_style()
    sub = df.dropna(subset=["seconds"]).copy()
    if sub.empty:
        return
    scen      = [s for s in SCEN_ORDER if s in set(sub["scenario"])]
    hue_order = _present_methods(sub)
    palette   = _method_palette(hue_order)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    sns.boxplot(
        data=sub, x="scenario", y="seconds",
        hue="method", order=scen, hue_order=hue_order,
        palette=palette, width=0.55, linewidth=0.8,
        showmeans=True, meanprops=MEAN_PROPS, flierprops=FLIER_PROPS,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}"
    ))
    ax.set_xlabel("Scenario", labelpad=4)
    ax.set_ylabel("Wall-clock time (s, log scale)", labelpad=6)
    ax.set_title(f"Runtime by scenario\n({MEAN_NOTE})", fontsize=TITLE_SIZE)
    _add_mean_to_legend(ax, outside=False)
    fig.tight_layout()
    out = OUT / "defj_runtime_by_scenario.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


def write_tables(df: pd.DataFrame) -> None:
    methods = _present_methods(df)
    scen = [s for s in SCEN_ORDER if s in set(df["scenario"])]

    def cell(vals):
        if len(vals) == 0:
            return "--"
        if len(vals) == 1:
            return f"{vals.mean():.2f}"
        return f"{vals.mean():.2f} $\\pm$ {vals.std():.2f}"

    # Accuracy table (mu_d).
    lines = [
        "% Auto-generated by scripts/make_defj_figures.py",
        "\\begin{tabular}{l" + "c" * len(methods) + "}",
        "\\toprule",
        "Scenario & " + " & ".join(methods) + " \\\\",
        "\\midrule",
    ]
    for s in scen:
        row = [s]
        for m in methods:
            row.append(cell(df[(df["scenario"] == s) & (df["method"] == m)]["mu_d"]))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "defj_accuracy_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {OUT / 'defj_accuracy_table.tex'}")

    # Runtime table (seconds).
    lines = [
        "% Auto-generated by scripts/make_defj_figures.py",
        "\\begin{tabular}{l" + "c" * len(methods) + "}",
        "\\toprule",
        "Scenario & " + " & ".join(f"{m} (s)" for m in methods) + " \\\\",
        "\\midrule",
    ]
    for s in scen:
        row = [s]
        for m in methods:
            row.append(cell(df[(df["scenario"] == s) & (df["method"] == m)]["seconds"]))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "defj_runtime_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {OUT / 'defj_runtime_table.tex'}")


def write_summary(df: pd.DataFrame) -> None:
    lines = ["DEFJ benchmark summary (rows used after dropping errors):", ""]
    counts = df.groupby(["method"]).size()
    for m, c in counts.items():
        lines.append(f"  {m}: {c} runs")
    lines.append("")
    lines.append("Mean mu_d / hw_d / seconds by method:")
    agg = df.groupby("method")[["mu_d", "hw_d", "seconds"]].mean()
    lines.append(agg.to_string())
    (OUT / "defj_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {OUT / 'defj_summary.txt'}")
    print("\n" + "\n".join(lines))


def main() -> int:
    import argparse
    global OUT, MP_ONLY, MP_CSVS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mp-only", action="store_true",
                        help="ignore the PhyloNet CSV; plot MP-Allop only")
    parser.add_argument("--mp-csv", type=Path, nargs="+", default=None,
                        help="read MP-Allop results from exactly these CSVs "
                             "instead of mp_allop_results.csv + mp_w*.csv "
                             "(avoids merging a fresh sweep with an old one)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="output directory (default: paper_figures/, or "
                             "paper_figures/mp_only/ when --mp-only)")
    args = parser.parse_args()

    MP_ONLY = args.mp_only
    MP_CSVS = list(args.mp_csv) if args.mp_csv else []
    if args.outdir is not None:
        OUT = args.outdir
    elif MP_ONLY:
        OUT = ROOT / "paper_figures" / "mp_only"

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} valid result rows "
          f"({df['method'].value_counts().to_dict()})")
    print(f"Output dir: {OUT}")
    for metric in ("mu_d", "hw_d"):
        fig_accuracy_by_scenario(df, metric)
        fig_accuracy_vs_ils(df, metric)
        fig_accuracy_vs_genes(df, metric)
        fig_accuracy_grid(df, metric)
    fig_runtime_vs_genes(df)
    fig_runtime_by_scenario(df)
    fig_runtime_grid(df)
    write_tables(df)
    write_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
