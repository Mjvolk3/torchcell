---
id: v4i1j4ypcxlximjoyoeullm
title: CGT-Metabolism — pigment prediction, metabolome transfer, and the enzyme-constrained model class
desc: 'Consolidates the metabolism work: the 1-week SIMB pigment/metabolome-transfer demo, and the CGT-Metabolism model class it is built inside'
updated: 1785037045821
created: 1785037045821
---

Consolidation of [[scratch.2026.07.25.010919-adding-metabolism]] (dataset substrate + defects),
[[scratch.2026.07.25.172340-adding-metabolism-explainer]] (**the mathematics — Q1–Q15 answer every
design question; this note does not repeat the derivations**), and
[[plan.simb-2026-multimodal-cgt.2026.07.21]] WS4/WS8/WS11b/WS12/WS16.

**Two tracks, deliberately separated by deadline.**

| track | deliverable | deadline |
| --- | --- | --- |
| **A — SIMB demo** | betaxanthin + β-carotene prediction, and *does the Mülleder metabolome help either?* | **~2026-07-31** (submit ~08-02) |
| **B — CGT-Metabolism** | metabolite entities, soft catalysis, amortized flux sampling, enzyme constraints | after SIMB |

Track A ships inside the **new model class** so B is a continuation, not a rewrite. Track A needs
**no flux layer**.

---

## Track A — the SIMB demonstration

### The science: a designed positive/negative control pair

The question "does the metabolome help production prediction" is vague. The two pigments turn it
into a **controlled contrast with a mechanism**, because one has its precursor measured
genome-wide and the other does not.

**Betaxanthin ⟸ tyrosine — and tyrosine is measured in 4,678 strains.**

- Betaxanthin is synthesized *from tyrosine*: tyrosine → L-DOPA (CYP76AD1) → betalamic acid (DOD),
  condensing with cyclo-DOPA.
- **`tyrosine` is one of Mülleder's 19 amino acids** (`torchcell/datasets/scerevisiae/mulleder2016.py:98`),
  measured as an absolute intracellular concentration (mM) across **4,678 single deletions**.
- **The cassette is itself tyrosine-pathway deregulation.** Cachera's construct is not just the two
  plant genes — it carries **ARO4^K229L** and **ARO7^G141S**
  (`cachera2023.py:67-68,97-113`), the classic feedback-resistant alleles of DAHP synthase and
  chorismate mutase, whose whole purpose is to relieve tyrosine inhibition and push flux into the
  aromatic pathway. **The strain designers already bet on tyrosine flux; we can now measure whether
  the model recovers that.**

**β-carotene ⟸ acetyl-CoA / GGPP — and nothing measures those genome-wide.** The route is
acetyl-CoA → mevalonate → FPP → GGPP → phytoene → lycopene → β-carotene. Mülleder measures amino
acids only. Acetyl-CoA appears solely in `MetaboliteZelezniak2018` — and there in only **18 of 95**
strains. So β-carotene has **no genome-wide precursor measurement at all**.

**Therefore the experiment predicts its own asymmetry**, which is far stronger than a single lift:

$$\Delta_{\text{betaxanthin}}=r(\text{joint})-r(\text{alone})\ \gg\ \Delta_{\beta\text{-carotene}}$$

A lift on betaxanthin and a null on β-carotene is a *mechanistic* result (transfer happens through
the shared precursor, not through generic multitask regularization). A lift on both would mean the
gain is generic — also publishable, but a different claim. A null on both is a clean negative.

### THE GATE: deletion-keyed aggregation. Without it this experiment has zero data

`GenotypeAggregator` keys on the **full** perturbed gene set, and the pigment strains carry their
cassette as `gene_addition` perturbations (β-carotene 3 genes, betaxanthin 4 + 2 alleles). So
pigment genotypes are disjoint from single-KO metabolome genotypes. Measured
(`experiments/019-simb-multimodal/results/fig6_overlap_census.json`):

| aggregation key | betaxanthin ∩ metabolome | β-carotene ∩ metabolome |
| --- | ---: | ---: |
| full gene set (today) | **0** | **0** |
| **deletion-only (needed)** | **4,439** | **4,226** |

**This is the single blocking implementation item.** It is the axis-aware-key TODO WS2b deferred,
and it is exactly Design Decision 3 (cassette = reference-strain background: we predict pigment as
a function of the *deletion* on a fixed pathway-carrying reference strain). Nothing else in Track A
matters until this lands.

### Datasets

Build `fig6_pigment_transfer` — three datasets, deletion-keyed:

| dataset | n | target | shape | notes |
| --- | ---: | --- | --- | --- |
| `BetaxanthinCachera2023Dataset` | 4,735 | `metabolite_level[betaxanthin]` | scalar | population-centered fluorescence (can be negative); **per-record SE, n up to 16** |
| `CarotenoidOzaydin2013Dataset` | 4,474 | `visual_score` | scalar | **ordinal −5…+5**, subjective; `visual_score_min` on replicated rows |
| `AminoAcidMulleder2016Dataset` | 4,678 | `metabolite_level` | **vec(19)** | absolute mM; contains **tyrosine**; `metabolite_level_se=None`, n=1 |

**Not in Track A:** environment/chemogenomic datasets (explicitly deferred — "we aren't looking to
do environment yet"), isobutanol, Zelezniak, da Silveira, Yoshida, Xue, the proteomes. The ME
superset inventory is recorded in the explainer Q15 for Track B.

Units are mutually incomparable (fluorescence centered at 0, ordinal −5…+5, mM), so **three
separate heads, never one shared scalar head.**

### Model class — `CellGraphTransformerMetabolism`

Fork from `torchcell/models/equivariant_cell_graph_transformer.py`. Track A activates only the
heads; the metabolism layer is scaffolded and inert.

| component | Track A | Track B |
| --- | --- | --- |
| ENC + graph-regularized attention + PERT operator | copied verbatim, parity-tested | unchanged |
| `PerGeneHead` (S0) | copied | unchanged |
| **scalar product heads** (`betaxanthin`, `beta_carotene`) | **new — explicit scalar path** | unchanged |
| **19-dim metabolite head** (Mülleder, key-sorted, fixed columns) | **new** | replaced by the aligned per-metabolite head |
| flux layer (box, $Sv$, capacity, budget, $\mu$/$\Delta$) | **scaffold only, inert** | activated |

Mülleder needs **no Yeast9 alignment** in Track A — 19 fixed key-sorted columns, dense in every
record. That sidesteps the entire `target_metabolite_ids`/`col_idx` gap for the conference.

### Harness defects that must be fixed first (all verified, see the 010919 note)

1. `head_phenotypes` routes `per_metabolite: [metabolite]` but the real label is **`metabolite_level`**
   → head silently unsupervised, zero gradient, no warning
   (`conf/train_cgt_multitask.yaml:102`; `train_cgt_multitask.py:324`).
2. `_vector_phenotype_keys(..., scan=5000)` only scans the first 5,000 LMDB rows
   (`train_cgt_multitask.py:267`) — with ~9.2k pigment rows first, every metabolome row falls past
   the cap.
3. `is_scalar` is hardcoded true only for `gene_interaction` (`:316`) → a scalar target assigned into
   a vector head **broadcasts across all columns** instead of erroring. `visual_score` is routed
   nowhere.
4. Per-phenotype standardization must cover the new heads (generic by head name today, so config-only).

### Noise ceilings first (cheap; they reframe every result)

Mirroring `{morphology,expression}_noise_ceiling.py`:

- **betaxanthin** — per-record SE with n up to 16 → a real, well-powered ceiling.
- **β-carotene** — ordinal with `visual_score_min` → the ceiling is **rank agreement, not Pearson**.
  Decide the metric before training: report **Spearman**, and treat the target as ordinal.
- **Mülleder** — `n_replicates=1`, no SE → no within-dataset ceiling; use the tyrosine-vs-betaxanthin
  correlation as the external sanity check instead.

### Baseline: Merzbacher 2025 (the bar is low)

`merzbacherAccuratePredictionGene2025` — *"Accurate prediction of gene deletion phenotypes with Flux
Cone Learning,"* Nat Commun 2025, doi 10.1038/s41467-025-63436-9. OCR'd at
`$DATA_ROOT/torchcell-library/merzbacherAccuratePredictionGene2025/`.

- Uses **Cachera 2023**, restricted to the **811 deletions that are Yeast9 metabolic genes** (of
  4,223 in their copy of the screen).
- **Reconciliation:** we hold 4,735 deletions, 906 in Yeast9 (19.1 %); they hold 4,223 / 811
  (19.2 %) — **identical ratio**, so the gap is filtering, not a different dataset.
- **3-class classification** (low/medium/high; thresholds set *"qualitatively"* to make 67 % medium;
  138/545/128). They **tried regression and abandoned it**.
- **69.8 % accuracy against a 67.2 % majority-class rate — a 2.6-point margin.** High-producer
  accuracy 11–30 %. **No correlation or AUC reported.**
- Split: a single class-stratified random 80/20 at gene level, **no CV, no seed, not released**;
  reported test N is internally inconsistent (659 vs 649; 20 % of 811 = 162).

**Comparison protocol:** (1) reproduce their setting — intersect to Yeast9 genes, document the
906-vs-811 difference, apply their binning, stratified 80/20, report accuracy + per-class; (2) beat
it on their own 811; (3) report what they cannot — **regression on all ~4,735**, and the ~3,900
non-metabolic deletions their flux cone has no representation for; (4) add a gene-disjoint split
alongside, since theirs is random and leaks related genes.

**Their method is the un-amortized version of Track B's flux sampler** — OptGPSampler MCMC over
each deletion's flux cone, then shallow sklearn on the samples. Worth stating plainly in the paper.

### Runs — GPU 0 on GilaHyper is free

Verified: GPU 0 at 526 MiB / 0 % (GPUs 1–3 held by job 1322, the `019-joint-dec003` sweep). All four
conditions are ~4.5–4.7 k single-KO records, so they run sequentially on one card.

| # | condition | active heads | tests |
| --- | --- | --- | --- |
| A1 | betaxanthin alone | `betaxanthin` | baseline vs Merzbacher |
| A2 | betaxanthin + metabolome | `betaxanthin`, `mulleder19` | **Δ = the tyrosine transfer** |
| A3 | β-carotene alone | `beta_carotene` | baseline |
| A4 | β-carotene + metabolome | `beta_carotene`, `mulleder19` | **Δ = the negative control** |

Identical splits and seeds across A1–A4 (the 019 `_v2` lesson: an unmatched split invalidates Δ).
Metric: Pearson **and** Spearman per target; Spearman is primary for β-carotene.
First milestone is a **`--fast_dev_run` fit-check on each head** — "does it train at all" before "is
it good."

---

## Track B — CGT-Metabolism (after SIMB)

Full mathematics and every derivation: [[scratch.2026.07.25.172340-adding-metabolism-explainer]].
Settled decisions, with the reasons recorded there:

| decision | resolution |
| --- | --- |
| entity layers | genes + **metabolites**; **no enzyme nodes** — catalysis is relational, and a GPR-derived node set would bake the annotation into the most rigid part of the architecture |
| $S$ vs $\rho$ | **hard chemistry, soft biology** — $S$ never softens; $\rho$ is a prior whose zeros mean *untested* |
| promiscuity | $\Pi=\Pi^{\mathrm{GPR}}+\Delta\Pi$, $\Delta\Pi\ge0$, $\ell_1$-penalized → the annotation is a **floor**, mass can never leave known reactions; hard and soft run as parallel heads |
| the box | **dynamic**: $\bar v^u_j(\varepsilon,p,\mathrm{seq})$ folds medium, deletion, and capacity into the parameterization → exact, and one fewer loss weight |
| exactness budget | decoupled constraints → parameterization; **coupled** ($Sv=0$, shared budget) → objective |
| growth objective | **dropped.** Fitness (~$10^7$ records, all genes) supervises $v_{\mathrm{bio}}$ directly and is not blind to non-metabolic genes |
| $k_{\mathrm{cat}}$ | constrains capacity — **the only source of magnitude in the whole model** (4,129/4,131 Yeast9 bounds carry none) |
| $K_M$ | **required, not deferred** — Wu 2026: promiscuity is a $K_m$ effect (~2× higher), $k_{\mathrm{cat}}$ indistinguishable, so a $k_{\mathrm{cat}}$-only model gives promiscuous flux away free |
| thermodynamics | per-metabolite $\mu_i$; loop-freedom falls out of $\Delta$ being a potential difference, no integer variables |

### Two things Track B must do that are now concrete

**1. Refactor `YeastGEM` into a general GEM interface.** Two independent reasons:

- **The SBML drops the thermodynamics.** `data/databases/model_metDeltaG.csv` holds **2,389 real
  $\Delta_fG'^\circ$ values (85.1 % of 2,806 metabolites)** and `model_rxnDeltaG.csv` holds 3,210
  $\Delta_rG'^\circ$ — shipped inside yeast-GEM 9.0.2, invisible to us because
  `YeastGEM.model` reads `yeast-GEM.xml` and SBML is the one export without them (`grep -c deltaG
  yeast-GEM.xml` → 0). The `.yml`/`.mat` exports carry them.
- **Species transferability.** Only two ingredients are organism-specific — a GEM and one scalar
  ($P_{\mathrm{avail}}$). $\Delta_fG'^\circ$ is chemistry, and $k_{\mathrm{cat}}$/$K_M$/$\mathrm{MW}$
  come from sequence. So the constraint layer belongs in
  `torchcell/metabolism/constraints.py` as **pure functions of a GEM**, with no yeast constants in
  the model file.

**2. Amortized flux sampling — and the trap in it.** *A per-reaction marginal distribution is not a
distribution over flux vectors.* $Sv=0$ constrains the **joint**; sampling each $v_j$ from an
independent marginal almost surely violates it. Merzbacher hit the same wall from the other side and
reported it — their deep models failed *"attributed to the fluxes being linearly correlated through
$Sv=0$."* Those correlations *are* the constraint. So:

- put the distribution on a **latent** and push it through a deterministic map, never on
  per-reaction marginals;
- sampling $z\in\mathbb{R}^{1538}$ in $\ker S$ and setting $v=\mathcal{N}z$ makes **every sample
  exactly mass-balanced** — which flips the earlier box-vs-null-space recommendation *for the
  sampling variant*, since bounds then need a hinge;
- either way, report the per-sample feasibility residual, and evaluate as
  **width(model interval) vs width(FVA interval)** per reaction = how much the data bought.

### Deferred within B

Yeast-MetaTwin as a replacement $S$ (16,244 metabolites / 59,865 reactions — **decide, do not
drift**); env/chemogenomic datasets (need two whole missing adapter families:
`EnvironmentResponsePhenotype` and CRISPRa/i/d); the corrected iBioFoundry COBRA media port
(`experiments/007-kuzmin-tm/scripts/setup_media_conditions.py` is stale with a hardcoded
out-of-repo path, and the fixed formulations live only in the external `iBioFoundry-AI` repo).

### Reading to acquire (none of these are mirrored)

Sánchez 2017 (original GECKO — also the source of Yeast9's minimal medium), Domenzain 2022 (GECKO
2.0, where ecYeastGEM's $k_{\mathrm{cat}}$/$P_{\mathrm{avail}}$ come from), **Elsemman 2022 (already
in Zotero, just unassigned to a collection — one-line fix)**, Chen & Nielsen 2022 (review).
Already mirrored: `domenzainComputationalBiologyPredicts2025`, `wuSystematicallyExploringYeast2026`,
`yuanOpenEnzymeDatabase2026` (the Open Enzyme Database), `longEnzymeEngineeringDatabase2026`.

## Framing correction (2026-07-25)

Wu 2026 is **not** an argument about noise or uncertainty in metabolism. It is evidence that **the
real state of metabolism is closer to what they present**: 93 % of the reaction network unmodelled
(4,131 known vs 55,734 predicted), Yeast9 covering ~7 % of the known metabolome, ~53 % of modelled
enzymes acting as generalists, 52 % of underground reactions re-linking metabolites Yeast9 already
contains, and Yeast9 overstating fragility 2×. The design consequence is not "add slack for
uncertainty" but **"the annotated network is a subset of the real one, so treat $\rho$ as a floor
and let the model add to it."**
