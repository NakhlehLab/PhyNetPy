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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defj_common as dc  # noqa: E402

ROOT = dc.project_root()
RUNS = ROOT / "runs" / "defj"
OUT = ROOT / "paper_figures"

METHOD_ORDER = ["MP-HC", "MP-SA3", "PhyloNet"]
# Set at runtime; when True, the PhyloNet CSV is ignored entirely.
MP_ONLY = False
METHOD_COLORS = {"MP-HC": "#1f77b4", "MP-SA3": "#2ca02c", "PhyloNet": "#d62728"}
SCEN_ORDER = ["D", "E", "F", "J"]


def _numify(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_data() -> pd.DataFrame:
    """Return a long-form frame with a unified 'method' column."""
    frames = []

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


def _swarmplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str = "method",
    x_order: list | None = None,
    hue_order: list[str] | None = None,
    dodge: bool = True,
) -> None:
    """Beeswarm of individual runs; one point per replicate/condition."""
    if hue_order is None:
        hue_order = _present_methods(data)
    sns.swarmplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=x_order,
        hue_order=hue_order,
        palette=_method_palette(hue_order),
        dodge=dodge,
        size=2.0,
        alpha=0.65,
        ax=ax,
        warn_thresh=2000,
    )


def fig_accuracy_by_scenario(df: pd.DataFrame, metric: str) -> None:
    """Swarm plot: every run's metric, grouped by scenario and method (n=1)."""
    sub = df[df["n"] == 1].copy()
    if sub.empty:
        return
    scen = [s for s in SCEN_ORDER if s in set(sub["scenario"])]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _swarmplot(ax, sub, x="scenario", y=metric, x_order=scen)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(f"{metric} to true network")
    ax.set_title(f"Species-network error ({metric}), single individual (n=1)")
    ax.legend(title="", loc="upper right")
    fig.tight_layout()
    out = OUT / f"defj_accuracy_by_scenario_{metric}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_accuracy_vs_ils(df: pd.DataFrame, metric: str) -> None:
    """Swarm plot vs ILS level t for D/E/F (n=1, g=1)."""
    sub = df[(df["n"] == 1) & (df["g"] == 1)
             & (df["scenario"].isin(["D", "E", "F"]))].copy()
    if sub.empty:
        print("  (skip accuracy_vs_ils: no D/E/F g1 n1 data yet)")
        return
    sub["t"] = sub["t"].astype(int).astype(str)
    t_order = sorted(sub["t"].unique(), key=int)
    scen = [s for s in ["D", "E", "F"] if s in set(sub["scenario"])]
    fig, axes = plt.subplots(1, len(scen), figsize=(4.5 * len(scen), 4.2),
                             sharey=True, squeeze=False)
    for k, s in enumerate(scen):
        ax = axes[0][k]
        d = sub[sub["scenario"] == s]
        _swarmplot(ax, d, x="t", y=metric, x_order=t_order)
        ax.set_title(f"Scenario {s}")
        ax.set_xlabel("ILS level (t)")
        if k == 0:
            ax.set_ylabel(metric)
        if k != len(scen) - 1:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    fig.suptitle(f"Network error vs ILS ({metric}), n=1, 1 gene")
    fig.tight_layout()
    out = OUT / f"defj_accuracy_vs_ils_{metric}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_accuracy_vs_genes(df: pd.DataFrame, metric: str) -> None:
    """Swarm plot vs gene count, pooled over scenarios (n=1)."""
    sub = df[df["n"] == 1].copy()
    if sub.empty:
        return
    sub["g"] = sub["g"].astype(int).astype(str)
    g_order = sorted(sub["g"].unique(), key=int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _swarmplot(ax, sub, x="g", y=metric, x_order=g_order)
    ax.set_xlabel("Number of gene trees (g)")
    ax.set_ylabel(metric)
    ax.set_title(f"Network error vs gene count ({metric}), n=1")
    ax.legend(title="", loc="upper right")
    fig.tight_layout()
    out = OUT / f"defj_accuracy_vs_genes_{metric}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_runtime_vs_genes(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["seconds"]).copy()
    if sub.empty:
        return
    sub["g"] = sub["g"].astype(int).astype(str)
    g_order = sorted(sub["g"].unique(), key=int)
    sub["log_seconds"] = np.log10(sub["seconds"].clip(lower=0.1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _swarmplot(ax, sub, x="g", y="log_seconds", x_order=g_order)
    ax.set_xlabel("Number of gene trees (g)")
    ax.set_ylabel("log10 wall-clock seconds")
    ax.set_title("Runtime vs gene count")
    ax.legend(title="", loc="upper left")
    fig.tight_layout()
    out = OUT / "defj_runtime_vs_genes.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_runtime_by_scenario(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["seconds"]).copy()
    if sub.empty:
        return
    scen = [s for s in SCEN_ORDER if s in set(sub["scenario"])]
    sub["log_seconds"] = np.log10(sub["seconds"].clip(lower=0.1))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _swarmplot(ax, sub, x="scenario", y="log_seconds", x_order=scen)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("log10 wall-clock seconds")
    ax.set_title("Runtime by scenario")
    ax.legend(title="", loc="upper left")
    fig.tight_layout()
    out = OUT / "defj_runtime_by_scenario.png"
    fig.savefig(out, dpi=200)
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
    global OUT, MP_ONLY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mp-only", action="store_true",
                        help="ignore the PhyloNet CSV; plot MP-Allop only")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="output directory (default: paper_figures/, or "
                             "paper_figures/mp_only/ when --mp-only)")
    args = parser.parse_args()

    MP_ONLY = args.mp_only
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
    fig_runtime_vs_genes(df)
    fig_runtime_by_scenario(df)
    write_tables(df)
    write_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
