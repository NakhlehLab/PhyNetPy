# PhyNetPy 1.0.0 software paper

LaTeX source for the PhyNetPy software paper, targeting Oxford *Bioinformatics*.
Compiles with `pdflatex`/`xelatex` + BibTeX, or by uploading the directory to
Overleaf. No non-standard packages.

```bash
# any TeX distribution
latexmk -pdf main.tex

# or, self-contained
tectonic -X compile main.tex
```

## Layout

| Path | Contents |
| --- | --- |
| `main.tex` | Document root: preamble, title, `\input` of each section. |
| `sections/` | One file per section. Prose only. |
| `tables/` | `\input`-ed tables. **All but `comparison.tex` are generated.** |
| `figures/` | `architecture`, `transition`, `workflow` are standalone TikZ sources plus their PDFs; `performance.pdf` is generated. |
| `benchmarks/` | Measurement scripts, their JSON output, and the generators that turn JSON into tables and figures. |
| `listings/` | Every code listing in the paper, in executable form. |
| `references.bib` | Bibliography. Entries keyed `TODO_*` are placeholders. |
| `INVENTORY.md` | Claim/evidence traceability. Internal; not submitted. |
| `check_prose.py` | Flags inflated language and stock transitions. |

## Regenerating every number

No numeric result in the paper is typed by hand. To reproduce all of them:

```bash
export PHYLONET_JAR=/path/to/phylonet.jar   # optional; skipped if unset
python benchmarks/run_all.py                # benchmarks + tables + figures
python benchmarks/make_analyses_table.py    # Table 1, from the live registry
python listings/paper_listings.py           # executes every code listing
python check_prose.py                       # writing check
```

`run_all.py` runs each `bench_*.py`, then `make_artifacts.py`, which rewrites
`tables/*.tex` and `figures/performance.pdf` from the JSON. Generated tables
carry a header saying so. The PhyloNet comparison is the only step needing
software from outside this repository.

The standalone figures are compiled separately, since they change rarely:

```bash
cd figures && tectonic -X compile architecture.tex   # and transition, workflow
```

## Before submission

`INVENTORY.md` §14 and §15 list what remains. The blocking items:

1. **Two defects found while benchmarking** (`INVENTORY.md` §14.1, §14.2). The
   pseudo-likelihood search reports scores far worse than the generating
   network's and does not improve with more iterations, and the
   `PseudoLikelihood` criterion does not enforce branch-length units the way
   `Likelihood` does. Section 5.3 reports the resulting accuracy deficit as an
   open question; resolving the cause changes what that section says.
2. **Version.** Everything was measured on 0.6.0. The paper describes 1.0.0,
   which adds the two unbuilt registry cells. Re-run the benchmarks against the
   tagged release and update the abstract, Section 5 and the reproducibility
   section.
3. **References.** Every `TODO_*` key in `references.bib`, plus the `note`
   fields marked `{TODO}` on real entries.
4. **PhyloNet rows** marked `[TODO: verify]` in `tables/comparison.tex`.
5. **Archival DOI** for the release.

`TODO` markers are deliberately left visible in the compiled PDF so none can be
missed; search the text for `TODO` to enumerate them.
