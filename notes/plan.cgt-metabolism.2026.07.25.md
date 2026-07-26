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

### 2026.07.25 — GATE PASSED, and it uncovered a genotype collision

`DeletionKeyedGenotypeAggregator` landed in `torchcell/data/genotype_aggregate.py` as a subclass
(the aggregator is passed as a *class* and instantiated with only `root=` at
`neo4j_cell.py:431`, so a constructor flag would need plumbing through 20+ call sites). Verified
against the real source LMDBs by
`experiments/019-simb-multimodal/scripts/verify_deletion_keyed_aggregation.py` →
`results/deletion_keyed_aggregation_census.json`:

| pair | full-key | deletion-key |
| --- | ---: | ---: |
| betaxanthin ∩ metabolome | **0** | **4,432** |
| β-carotene ∩ metabolome | **0** | **4,221** |

Within 7 and 5 of the live-DB census (4,439 / 4,226) — the small gap is the query's `gene_set`
filter, which this raw-LMDB check does not apply.

**Unexpected finding — full-key aggregation silently MERGES two distinct betaxanthin strains.**
Betaxanthin has 4,735 records but only **4,734** unique full-keys, versus 4,735 deletion-keys.
The colliding bucket, confirmed by inspection:

```text
deletions: ['YBR249C']   all: ['CYP76AD1', 'DOD', 'YBR249C', 'YPR060C']
deletions: ['YPR060C']   all: ['CYP76AD1', 'DOD', 'YBR249C', 'YPR060C']
```

Because the cassette carries **ARO4 (YBR249C)** and **ARO7 (YPR060C)** as feedback-resistant
*alleles*, both gene names are already in every strain's perturbation set — so the ARO4-deletion
strain and the ARO7-deletion strain have **identical full gene-name sets** and get averaged
together. This is a correctness bug, not just lost co-location, and it lands on precisely the two
tyrosine-pathway genes the betaxanthin story is about. Deletion-keying resolves it (4,735 distinct).
It also confirms the `GenotypeAggregator` TODO was right: the full-set key is unsafe the moment a
non-deletion axis touches a gene that is also deletable.

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

## 2026.07.25 — iBioFoundry β-carotene case study: media + four negative results

The user supplied the iBF `s-cerevsisiae-beta-carotene-knockout` code dump. It contains the
**corrected `media_setup.py`** flagged as missing earlier, plus a run history with results that
change the Fig-6 narrative. Not yet ported; recorded here first.

### The media layer we should port

`media_setup.py` layers recipes, each calling the previous: `reset_media` (close all exchanges) →
`setup_minimal_media` (glucose + NH4 + O2 + H2O + P/S + H+ + 9 trace metals) → `setup_ynb_media`
(+9 vitamins) → `setup_sc_media` (+20 AAs + uracil + adenine) → `setup_sc_ura_media` (SC − uracil,
for URA3-plasmid selection) → `setup_sc_ura_adenine_media` (SC − uracil − adenine). Supplements are
capped at **5 % of the glucose uptake rate**, sourced to Suthers et al. 2020
(`suthersGenomescaleMetabolicReconstruction2020`). **Correct DOI is `10.1016/j.mec.2020.e00148`**
(*Metabolic Engineering Communications* **11**, e00148) — the iBF `media_setup.py` docstring cites
`10.1016/j.ymben.2020.03.010`, which is an unrelated paper; we already flagged this in
[[torchcell.datamodels.media-components]] and `torchcell/datamodels/media.py:21` has it right.

### Suthers 2020 checked against the source — the 5 % rule does NOT mean what our code does

Already mirrored (`$DATA_ROOT/torchcell-library/suthersGenomescaleMetabolicReconstruction2020/`,
captured 2026-07-14 via the `database` collection, born-digital `pdftotext` OCR). §2.5 "Modeling
simulations", verbatim:

> "During initial testing … the carbon substrate uptake rate was set to a value 3.3 mmol gDW⁻¹ hr⁻¹;
> we chose this value as a rough estimate for glucose uptake … and arbitrarily applied it to each
> carbon substrate. … **For growth predictions involving rich media, supplementary compound uptake
> rates were set to 0.165 mmol gDW⁻¹ hr⁻¹ (i.e., 5 % of default substrate uptake rate of 3.3 mmol
> gDW⁻¹ hr⁻¹).** … The undefined composition of yeast extract in Yeast-Peptone-Dextrose (YPD) media
> was assumed to be that of YNB media plus 20 amino acids and D-glucose. … Glucose uptake rate was
> set to 10.0 mmol gDW⁻¹ hr⁻¹ during OptKnock simulations."

**The 5 % is anchored to the DEFAULT 3.3, giving a fixed absolute 0.165 — it is not 5 % of whatever
glucose bound you set.** The paper's own OptKnock runs use glucose 10.0 and do **not** rescale
supplements. Our code does rescale: `experiments/007-kuzmin-tm/scripts/setup_media_conditions.py:71,124`
computes `glucose_rate * 0.05`, and iBF's `media_setup.py` does the same at `glucose_rate=10.0` →
supplements at **0.5, i.e. 3.03× the sourced value**. `torchcell/datamodels/media.py:19` repeats the
"5 % of glucose" phrasing.

**Decide and document before porting.** Constant-ratio scaling is a defensible modeling choice, but
it is not what the source did, and tripling amino-acid availability changes biosynthetic burden —
material for an amino-acid readout like Mülleder. Per CLAUDE.md this cannot ship as "sourced".

Three further findings that shape the port:

- **YPD-as-stand-in has a citable basis after all.** Suthers explicitly models YPD as *"YNB media plus
  20 amino acids and D-glucose"*, declaring yeast extract's composition undefined and substituting
  that set. So an approximation is defensible **provided it is named as an assumption**, which is
  exactly what iBF's retirement of the `setup_ypd_media` name enforces. Peptone is never modeled.
- **The SC recipe is not in Suthers at all.** SC is used throughout their Fig. 4 but its composition
  appears nowhere in the paper or SI. Our SC (`setup_sc_media` = YNB + 20 AAs + uracil + adenine)
  therefore needs its own source — the standard Difco/Sigma formulation already cited in
  `media.py:18`. This matters: **SC is betaxanthin's medium**, the Track A headline dataset.
- **The real derivation is one deferral further out.** The 5 % rule carries no justification in
  Suthers ("rough estimate", "arbitrarily applied"); it points to **Dinh et al. 2019**
  (*Metab Eng Commun* **9**, e00101), which is **not in our mirror**. Per the CLAUDE.md
  follow-deferrals rule, that is the paper to capture if we want the convention's actual basis.

Suthers SI is open-access and retrievable in one scriptable call —
`curl -s "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7586132/supplementaryFiles"` —
including `mmc8.zip` with `iIsor850.{json,xml}` (the actual exchange bounds). Not yet mirrored;
`manifest.json` has `si_data_sources: []`. Note `mmc5.xlsx` is growth data *labelled by* medium, not
recipes — no media-composition table exists in the SI.

### Our datasets need exactly four media, and one of them cannot be represented

| medium | datasets | maps to |
| --- | --- | --- |
| **SC** | Cachera betaxanthin · Lopez isobutanol · Kemmeren expression | `setup_sc_media` ✅ |
| **SC-URA** | Ozaydin β-carotene | `setup_sc_ura_media` ✅ |
| **SM** | Mülleder AA · Zelezniak metabolome+proteome · Messner proteome | `setup_ynb_media` (decide + document: SM = YNB + glucose, no AAs) |
| **YPD** | **Ohya morphology (4,718)** · da Silveira lipids · Yoshida organic acids | **NOT REPRESENTABLE** |

**YPD: we approximate it, and Suthers tells us exactly how (RESOLVED).** iBF's `media_setup.py`
retires the name — *"Real YPD contains peptone-derived peptides, yeast-extract lipids, and undefined
nutrients that Yeast9 has no way to represent"* — but retiring the name is not the same as refusing
to model it. Suthers §2.5 states the substitution outright:

> "The undefined composition of yeast extract in Yeast-Peptone-Dextrose (YPD) media was assumed to be
> that of **YNB media plus 20 amino acids and D-glucose**."

So the YPD stand-in is **YNB + 20 AAs, no nucleobases** — which is *numerically identical* to iBF's
`setup_sc_ura_adenine_media` (its docstring even says so: "functionally identical to our
pre-retirement `setup_ypd_media` helper"). Port it as a **separately named** entry point,
`setup_ypd_approx_media`, delegating to the same bound-setter but carrying the Suthers quote in its
docstring. Same bounds, different claim, different provenance — the name must record what it asserts,
so a reader can never mistake it for real YPD. Peptone is never modeled, by us or by Suthers.

**Consequence worth knowing before investing in WS6: our four media collapse to ~two bound-vectors.**

| medium | exchanges opened beyond YNB |
| --- | --- |
| SC | 20 AAs + uracil + adenine |
| SC-URA | 20 AAs + adenine |
| YPD (approx) | 20 AAs |
| SM | — (YNB only) |

The three amino-acid media differ from each other by **one or two nucleobase exchanges**; the only
large axis is **±20 amino acids** (SM vs the rest). So exchange-bound ε-conditioning carries roughly
one bit of real information across our datasets — worth doing for correctness, but it will not
explain much variance. It also *shrinks* the Track A media confound: betaxanthin(SC) vs Mülleder(SM)
and β-carotene(SC-URA) vs Mülleder(SM) differ by the same 20-amino-acid step plus/minus adenine, so
the two arms are near-identical in medium distance and the confound cancels almost exactly in the
paired contrast.

### The media confound in Track A — and why the paired design absorbs it

Track A's three datasets sit on **three different media**: betaxanthin SC, β-carotene SC-URA,
Mülleder SM. Medium is perfectly collinear with dataset, so ε-conditioning cannot separate them —
a single-arm "metabolome helped" result would be confounded with a medium change.

The positive/negative control pair rescues this. Both arms pair against the *same* Mülleder/SM, and
SC vs SC-URA differ only in uracil, so the two arms are near-matched and the confound largely
cancels in the difference of differences $\Delta_{\text{betax}}-\Delta_{\beta\text{-car}}$. **This is
a second, independent reason to report the contrast rather than either lift alone.**

### Four iBF negative results that strengthen the Fig-6 story

1. **MCS finds ZERO growth-coupled single-KO designs.** The first run produced cell-killers because
   it specified only a SUPPRESS module; corrected with an inverted target (`{bc<=0.01, biomass>=1.5}`)
   plus a PROTECT module (`{biomass>=1.5}`), re-run at MAX_COST=1 on both bases (jobs 3056/3057,
   1.8 h each) → **0 solutions**. Under the single-KO experimental constraint, growth-coupled
   β-carotene production is unreachable on Yeast9+Crt+SC-Ura. Network redundancy bypasses any single cut.
2. **FSEOF never increases the β-carotene envelope.** All 50 biomass-feasible candidates give
   `bc_max_ko == bc_max_base == 2.851` exactly.
3. **PAH1 KO *shrinks* the envelope ~9 %** (−0.27 on 2.851) despite collaborator lab evidence
   favouring it — a documented FBA-vs-wet-lab disagreement.
4. **Flux sampling is intractable** on Yeast9 (4,131 rxns) under GLPK — OptGP/ACHR warmup is
   ~`2·n_reactions` LP solves per cone; a minimal init ran >15 min CPU before being killed.

**Consequence:** for β-carotene single-KO design, constraint-based methods produce *nothing usable* —
zero MCS designs, zero FSEOF ceiling shifts, and a wrong-signed FBA call on the one candidate lab
data supports. Meanwhile we hold **4,474 measured deletions**. That is a far stronger Fig-6 framing
than "our model also does this": it is the case where mechanistic design has no answer and data does.

### They already ran Flux Cone Learning — and it failed on β-carotene

Using Merzbacher's own pre-computed WT Yeast9 flux samples (Zenodo 10.5281/zenodo.15761895, mirrored
locally at `databases/fcl/paper_merzbacher_2025/`) with Ozaydin colour labels 3-class binned:
**balanced accuracy 0.362 (HGB) / 0.354 (RF) against a 0.33 random baseline**, 89 % of predictions
collapsing to "medium". The paper reports ~0.70 on *betaxanthin*.

Their diagnosis — and it corroborates our precursor asymmetry from the *label* side:

- visual ordinal scoring has higher measurement error than fluorescence;
- class imbalance 9/67/24 high/medium/low;
- **many low-colour deletions are flagged petite** — mitochondrial dysfunction starves the MVA
  pathway, a non-metabolic confound flux geometry cannot see;
- carotenoid (MVA→FPP) and betaxanthin (tyrosine) use different precursor pathways.

**Two actionable consequences for Track A.** (a) **Filter or condition on petite** — our
`CarotenoidOzaydin2013Dataset` already emits `comment_annotations` with `flag_petite` plus six other
QC booleans (`ozaydin2013.py:116-124`), so this is available now; without it we are partly predicting
mitochondrial dysfunction rather than carotenoid flux. (b) Expect β-carotene to be the weak arm for
**two independent reasons** — no measured precursor *and* a noisy ordinal readout — which makes the
noise-ceiling work (rank agreement, not Pearson) essential rather than optional.

## Framing correction (2026-07-25)

Wu 2026 is **not** an argument about noise or uncertainty in metabolism. It is evidence that **the
real state of metabolism is closer to what they present**: 93 % of the reaction network unmodelled
(4,131 known vs 55,734 predicted), Yeast9 covering ~7 % of the known metabolome, ~53 % of modelled
enzymes acting as generalists, 52 % of underground reactions re-linking metabolites Yeast9 already
contains, and Yeast9 overstating fragility 2×. The design consequence is not "add slack for
uncertainty" but **"the annotated network is a subset of the real one, so treat $\rho$ as a floor
and let the model add to it."**
