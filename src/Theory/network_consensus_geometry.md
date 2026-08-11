# Network Consensus Geometry — research-status note

**Honest assessment of "summarise a sample of networks into a consensus network
with support," after a literature check.**

> This file replaces an earlier draft that presented this problem as if it had a
> clean, novel solution. A related-work pass (below) shows that most of that
> draft was either definitional, forty years old, or already published in
> 2019–2024. This note documents precisely what is known, what is therefore
> *not* a contribution, and the narrow, genuinely-uncertain gap that might
> remain — with the risks stated plainly. Nothing here is claimed as a result.

---

## 1. The problem (unchanged, and real)

A Bayesian (`infer(..., criterion=Bayesian())`) or bootstrap network analysis
returns a *sample* of networks on a fixed taxon set $X$. Trees have a canonical
summary — the majority-rule consensus tree plus per-clade support — and networks
lack an equally settled one in practice. The two user questions are:

- **(Q1)** a single central "consensus" network to report;
- **(Q2)** a support value per reticulation, comparable across studies.

The problem is real. The question is whether *we* have anything new to say about
it, given what follows.

---

## 2. Reality check — what the literature already contains

**(R1) Majority-rule is the median consensus.** Barthélemy & McMorris (1986):
the majority-rule tree is the median of the sample under the Robinson–Foulds
(symmetric-difference) distance, i.e. it minimises summed distance to the
sample. Any "Fréchet-median-under-a-symmetric-difference-metric" framing is this
result. **Consequence:** the median/majority framing is not novel.

**(R2) µ-distance is a symmetric-difference metric — by definition.** Cardona,
Rosselló & Valiente (2008, 2009) define $d_\mu$ as the multiset symmetric
difference of µ-vectors; that it equals the $\ell_1$ distance between µ-vector
*histograms* is immediate from the definition, not a theorem. **Consequence:**
the "linearisation observation" in the first draft was definitional.

**(R3) The µ-decoding / realizability problem is largely solved for the relevant
classes.**
- Erdős, Semple & Steel (2019, *Math. Biosci.* 313): a binary **orchard**
  network is uniquely determined by its ancestral profile (µ-representation)
  *among all networks*, and there is a **polynomial-time reconstruction
  algorithm**. Orchard $\supsetneq$ tree-child, with an unbounded number of
  reticulations.
- Bai, Erdős, Semple & Steel (2021, *Math. Biosci.* 332): extends this to
  **stack-free** orchard networks; without the stack-free condition the
  representation determines the network only up to resolution of high-in-degree
  vertices.
- arXiv:2412.05107 (2024): plain µ does **not** encode general orchard networks;
  a *modified* µ-representation (with in-degrees) encodes strongly
  reticulation-visible semi-binary stack-free orchard networks.

**Consequence:** the "deep open core" of the first draft (decode a µ-representation
back to a network) is, for orchard/tree-child networks, a solved poly-time
problem. My "greedy bottom-up decoding" conjecture is essentially the existing
cherry-picking reduction.

**(R4) The consensus-network construction already exists (for level-1).** Huber,
Moulton & Spillner (2023, *JGAA* 27(7)): given a collection of **1-nested**
networks, threshold the features of a realizable encoding at frequencies
$(p,q) = (\tfrac12, \tfrac23)$ and build the unique consensus network; it runs
in $O(t|X|^2 + |X|^3)$ and **reduces to the majority-rule tree** when the inputs
are trees. This is exactly the "threshold-then-realise consensus" the first
draft proposed. PhyloFusion (Huson et al. 2024, *Syst. Biol.*) constructs
tree-child networks from sets of trees.

**Consequence:** the consensus-from-a-sample idea, including its majority-rule
reduction and the need to characterise realizable thresholded feature sets, is
published for 1-nested networks.

**(R5) Reference-based support already ships.** PhyloNetworks'
`hybridclades_support` and PhyloNet's averaged credible set already give
feature-frequency support. The honest gap here is only *reference-freeness*,
which is a refinement, not a new problem.

---

## 3. What of the first draft survives

Almost nothing as a *result*. What survives is:

- a correct **map of the metric substrate** already in PhyNetPy
  (`GraphUtils.mu_distance`, `_mu_vectors`, `ReticulationComparison`), with the
  important correction that plain µ is metric-complete only on **tree-child**
  networks (R2, R3) — so any consensus built on `mu_distance` must either
  restrict to tree-child or switch to the extended/modified µ-representation;
- the observation that PhyNetPy is unusually well-positioned to *implement* an
  existing consensus method (it has the µ-machinery, a sample source, and the
  reticulation-matching comparison), which is an engineering contribution, not a
  scientific one.

---

## 4. The narrow gap that might remain (stated as questions, with risk)

Two directions are *not* directly closed by R1–R5. Both are uncertain; neither
is claimed to work.

### 4.1 Consensus for level ≥ 2 orchard networks (combinatorial)

Huber–Moulton–Spillner is restricted to **1-nested** (level-1) networks, and
their key technical step is a *non-trivial* lemma: the frequency-thresholded
set-pair system is realizable only for specific thresholds ($p \ge \tfrac12$,
$q \ge \tfrac23$), not for the naive $\tfrac12$. The analogous question for
**µ-representations of higher-level orchard networks** — "for which thresholds
is the thresholded (or coordinate-wise-median) µ-multiset realizable as an
orchard network, and if none, what is the complexity of projecting onto the
realizable set?" — is not answered by anything found above, because the
Erdős–Semple–Steel decoder assumes a *valid* profile as input and says nothing
about projecting an aggregated, possibly-invalid one.

- **Open question OQ-1.** Does there exist a threshold rule under which the
  aggregated µ-representation of a sample of level-$k$ orchard networks is always
  realizable (generalising HMS's $(\tfrac12,\tfrac23)$)?
- **Open question OQ-2.** If not, what is the complexity of
  $\arg\min_{N}\lVert h_N - h^\star\rVert_1$ over orchard networks (the
  projection)? Plausibly NP-hard, given that related network-from-features
  problems are NP-hard for non-dense inputs (van Iersel et al.).

**Risk.** The honest prior is that this is either (a) a modest generalisation of
HMS that a specialist could see quickly, or (b) genuinely hard, in which case the
deliverable is a hardness result — respectable but not the "cutting-edge method
for biologists" originally scoped. Either way it needs a proper related-work
pass against the van Iersel / Semple / Moulton school before we invest.

### 4.2 Consensus as a *statistical estimator* (the actually under-served angle)

R1–R5 are **purely combinatorial**: "given a set of networks, combine them."
None treats the consensus as an *estimator of a distribution* with statistical
properties. The under-served questions:

- **OQ-3 (consistency).** Is the µ-median network a consistent estimator of the
  true network as the posterior/bootstrap concentrates, and at what rate?
- **OQ-4 (the network stickiness phase transition).** Trees exhibit
  "stickiness": the Fréchet mean collapses to the star tree under dispersion.
  For networks the analogue is collapse to the backbone tree when reticulation
  placement is uncertain. Is there a sharp phase transition in an
  entropy-of-reticulation-placement parameter at which reticulations drop out of
  the consensus? A quantitative such result would be new.
- **OQ-5 (calibrated support).** Turn reference-free feature frequency into a
  *calibrated* support (coverage-correct under a null), which the existing
  frequency counts are not.

**Risk.** This is a statistics contribution that leans on the combinatorial
consensus already existing (HMS provides it for level-1). It is more likely to
be defensible than §4.1, but it is thinner, and OQ-4 in particular may be hard
to state cleanly for networks.

---

## 5. Recommendation

Given R1–R5, I do **not** recommend proceeding as if this problem is open. Two
honest options:

1. **Narrow and verify.** Commit only to OQ-1/OQ-2 (level-$\ge 2$ realizability)
   *or* OQ-3/OQ-4 (estimator theory), and first do a dedicated related-work pass
   (van Iersel, Semple, Steel, Moulton, Huber, Scornavacca; the
   `PhyloNetworks`/`SplitsTree` tool literature) to confirm the specific
   sub-question is open before any derivation. Treat a clean negative
   (hardness) result as an acceptable outcome.

2. **Pivot.** Move to one of the other candidate problems whose frontier is less
   crowded — most plausibly the **identifiability / network anomaly zone**
   (Candidate 2), where the higher-level cases are demonstrably open, or the
   **#P-hard likelihood / FPT-in-level** direction (Candidate 3). These were
   flagged in the original candidate list and were not undercut by this search.

Either way, the discipline going forward: **no framing as a result until a
targeted novelty check clears it.** That is the change of process this note
exists to enforce.

---

## References

*Consensus and medians.*
Barthélemy & McMorris (1986), *The median procedure for n-trees*, J. Classif. --
Huber, Moulton & Spillner (2023), *Computing consensus networks for collections
of 1-nested phylogenetic networks*, JGAA 27(7):541–563 (arXiv:2107.09696). --
Huson et al. (2024), *PhyloFusion*, Syst. Biol. (syaf049).

*µ-representations / reconstruction.*
Cardona, Rosselló & Valiente (2008), *Comparison of tree-child phylogenetic
networks*, IEEE/ACM TCBB. --
Cardona, Llabrés, Rosselló & Valiente (2009), *Metrics for phylogenetic networks
I*, IEEE/ACM TCBB. --
Erdős, Semple & Steel (2019), *A class of phylogenetic networks reconstructable
from ancestral profiles*, Math. Biosci. 313:33–40 (arXiv:1901.04064). --
Bai, Erdős, Semple & Steel (2021), *Defining phylogenetic networks using
ancestral profiles*, Math. Biosci. 332:108537. --
Cardona, Pons, Scornavacca et al. (2023), *Comparison of orchard networks using
their extended µ-representation*, arXiv:2302.10015. --
(2024) *Metrics for classes of semi-binary phylogenetic networks using
µ-representations*, arXiv:2412.05107.

*Tree geometry / stickiness (for the estimator angle).*
Billera, Holmes & Vogtmann (2001); Sturm (2003); Owen & Provan (2011); Barden,
Le & Owen; Skwerer et al. (2024), arXiv:2407.03977.

*PhyNetPy internals.* `GraphUtils.mu_distance`, `_mu_vectors`,
`dominant_tree`; `ReticulationComparison.compare_networks`; `ModelMove`;
`infer(..., criterion=Bayesian())`.
