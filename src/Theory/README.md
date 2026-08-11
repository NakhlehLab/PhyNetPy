# `Theory/`

A workbench for deep, self-contained attacks on open algorithmic problems in
phylogenetics that fit PhyNetPy's design. Work here is *research first*: a
rigorous written treatment (problem statement, mathematics, prior work,
proposed method, and a validation/integration plan) precedes any code, and a
side prototype is built and validated in isolation before it is proposed for
integration into the `phynetpy` package proper.

Nothing in this folder is part of the public API. Modules here import from
`phynetpy` freely but are not imported by it.

## Active problem (under review — see note)

- **Network Consensus Geometry (NCG)** --
  [`network_consensus_geometry.md`](network_consensus_geometry.md).
  The motivating question -- summarise a *sample* of networks (Bayesian
  posterior or bootstrap) into a central "consensus" network with support, the
  way trees get a majority-rule consensus -- is real. But a literature check
  (Barthélemy--McMorris 1986; Erdős--Semple--Steel 2019; Bai et al. 2021;
  Huber--Moulton--Spillner 2023) found that the core construction and its
  hardest sub-problem (decoding a µ-representation back to a network) are
  **already published** for tree-child / orchard / 1-nested networks. The
  document is now an honest state-of-the-art map that separates what is known
  from the narrow, still-uncertain gaps (level-$\ge 2$ realizability under
  aggregation; the estimator-theory angle) and recommends either narrowing to
  a verified-open sub-question or pivoting to a less crowded candidate problem.
  **No claim here is a result.**
