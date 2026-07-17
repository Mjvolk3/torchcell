---
id: mvr869l8ndgm1snlspq3x6v
title: "Results 6 -- Experimental Plan: Regulatory-Network Double-KO Chassis Test"
desc: ''
updated: 1784306533850
created: 1784306533850
---

Planning note for the Results §6 experimental section. Companion to the
metabolome-dataset triage in [[paper.north-star.dataset-triage]] and to the analysis of
the Domenzain 2025 "103 valuable chemicals" ecFactory paper
(`domenzainComputationalBiologyPredicts2025`, PNAS, DOI 10.1073/pnas.2417322122).

## 2026.07.17 - Regulatory-network double-KO chassis experiment

### Thesis: predict metabolic epistasis that constraint-based models cannot

Demonstrate the core torchcell claim on a bioproduction task: **predict the effect of a
gene-pair knockout on an intermediate-metabolite pool where that effect is driven by
regulatory-network epistasis — the regime constraint-based metabolic models (CBM)
structurally cannot reach.**

Domenzain 2025 ecFactory is the CBM baseline to beat. It is enzyme-constrained FBA/FSEOF
over ecYeastGEM: it flux-scans and enzyme-capacity-ranks **single** OE/KD/KO targets for
103 chemicals. The paper states its own limit plainly — "the impact of these
modifications on other biological processes, such as regulatory networks, is not
accounted for in the metabolic model." A **double KO whose effect on a precursor pool is
non-additive** — because the two genes interact through signaling, allosteric feedback,
or network buffering — is invisible to that framework: it scores the pair as ~additive or
reroutes flux around it. torchcell learns that interaction structure from Costanzo/Kuzmin
SGA (fitness epistasis) and, combined with single-KO metabolomics, extrapolates to
double-KO metabolite epistasis.

Manipulating the regulatory network is the explicit goal. That points the wet-lab arm at
**kinase/phosphatase knockouts** and at the one dataset that measures their metabolic
consequences directly.

### Two arms

**Arm A — wet-lab: organic-acid chassis via regulatory (kinase) double-KOs.**

- Target intermediates: **succinate** and **malate** — build both, pick the lead by
  prediction strength. Both are C4 platform acids and central-carbon chassis nodes.
- Dataset anchor: **Zelezniak 2018** (~95 kinase/phosphatase KO strains × 50-metabolite
  SRM panel) — measures succinate and malate directly, plus 11 of the 12 central-carbon
  precursor nodes the paper names as platform control points (all but succinyl-CoA). This
  is *the* regulatory-network metabolome dataset: its perturbations ARE the signaling layer.
- Interaction structure: Costanzo 2016 + Kuzmin 2018/2020 SGA (genome-wide digenic/trigenic).
- Prediction: rank gene pairs with **positive epistasis on the succinate (or malate)
  pool** — double KOs whose predicted pool exceeds the additive expectation of the singles.
- CBM filter: keep only pairs ecFactory does *not* flag as individual targets → the
  "CBM cannot see it" set.
- Readout: **HPLC organic-acid column** (RI/UV, e.g. Aminex HPX-87H) — the simplest,
  most standard assay available.
- Win condition: a double KO that beats the additive prediction *and* was absent from
  ecFactory's target list.

**Arm B — in-silico: does adding the metabolome data improve production-target prediction?**

- A cheap ablation needing no wet lab that directly justifies putting Mülleder + Zelezniak
  into this database build.
- Test: with vs without the metabolome datasets, does prediction of production targets
  already in the pipeline improve — specifically **isobutanol** (precursor pool =
  **valine**, in Mülleder + Zelezniak) and **betaxanthin** (precursor pool = **tyrosine**,
  in Mülleder + Zelezniak)?
- Logic: the free precursor pool is a learned intermediate feature for the downstream
  product. If adding the pool measurements lifts isobutanol/betaxanthin prediction, the
  AA-metabolome data demonstrably helps — the same precursor→product logic that makes
  aromatic/BCAA intermediates good chassis targets in Arm A.

### Dataset mapping (grounded)

| Role | Dataset | State | What it supplies |
|---|---|---|---|
| Reg-network metabolome (Arm A) | Zelezniak 2018 | BUILT (PR #35/#41) | ~95 kinase/phosphatase KOs × 50 metabolites incl. **succinate, malate**, citrate, fumarate, 2-OG, OAA, PEP, E4P, R5P, pyruvate, acetyl-CoA + aromatic/BCAA AAs |
| AA-precursor metabolome (Arm B) | Mülleder 2016 | ADDING | 4,678 genome-wide single KOs × 19 AAs incl. **Tyr, Phe, Trp, Val, Leu, Ile**; reference = population mean |
| Epistasis structure | Costanzo 2016 + Kuzmin 2018/2020 SGA | BUILT | genome-wide digenic (+ trigenic) genetic interactions |
| CBM baseline (external) | Domenzain 2025 ecFactory / ecYeastGEM | reference | single OE/KD/KO flux predictions for 103 chemicals |

**Load-bearing clarification: Mülleder does NOT contain organic acids** — it is the 19
amino acids only. So **Arm A (succinate/malate) rides on Zelezniak, not Mülleder**;
Mülleder powers Arm B (Val→isobutanol, Tyr→betaxanthin). Keep the two arms' data
sources distinct.

### Precursor → product use cases (corrected + expanded)

Correction to an earlier over-read: of the four production targets, **tyrosine is the
precursor for betaxanthin only** (1 of 4), not four. The mapping is one precursor per
product:

| Product | Precursor pool | In a dataset? | Note |
|---|---|---|---|
| betaxanthin (target) | **tyrosine** | Mülleder + Zelezniak (pool); Cachera 2023 (product) | betalain via L-DOPA |
| isobutanol (target) | **valine** | Mülleder + Zelezniak (pool) | Ehrlich pathway |
| beta-carotene (target) | **mevalonate/FPP** | **none** (pool gap) | Ozaydin 2013 = product readout, not pool |
| 2-phenylethanol (NOT a target) | **phenylalanine** | Mülleder + Zelezniak (pool) | see source below |

**Where the 2-phenylethanol case comes from:** it is Domenzain 2025's own validated
example, not one of our targets. Paper p.5, "In Silico Predictions Capture Validated
Metabolic Engineering Strategies": "7 out of the 12 predicted gene targets to increase
2-phenylethanol have been previously engineered..." (target list in SI Appendix Table S2).
It is Phe-derived and Phe is in Mülleder — so it is a strong *candidate new use case*, not
a current target.

**Additional intermediate use cases** (all pool-measured in Mülleder and/or Zelezniak),
for a few more shots on goal:

- **Phenylalanine → 2-phenylethanol / cinnamate / styrene** (Domenzain-validated; aromatic).
- **Arginine / ornithine → spermidine, putrescine** — Domenzain's *other* validated case
  (spermidine, 9 of 85 targets). Ornithine + arginine in Zelezniak, arginine in Mülleder.
  Polyamines are 6 of the paper's 103.
- **Tryptophan → indole alkaloids / auxin** (aromatic; in Mülleder + Zelezniak).
- **Valine / leucine / isoleucine → fusel alcohols** (isobutanol + isoamyl alcohol; BCAA).
- **Central-carbon nodes E4P / PEP / acetyl-CoA** (Zelezniak) — the paper's chassis control
  points, but harder to assay (LC-MS, not HPLC-UV).

**Downstream of tyrosine specifically, in our datasets:** only **betaxanthin** (Cachera
2023, our target) is a *measured tyrosine-derived product*. The broader tyrosine space
(tyrosol, hydroxytyrosol, p-coumarate, flavonoids, alkaloids) is **not** in any
current/planned dataset — those would be pure model extrapolation from the tyrosine pool,
not data-anchored. For data-anchored use cases, pivot to the sibling aromatics (Phe/Trp)
and the polyamine precursors above rather than deeper tyrosine derivatives.

### Are we missing key datasets before this build?

**For the chosen succinate/malate reg-network experiment: no core gap.** Zelezniak
(kinase-KO organic-acid metabolome) + SGA (epistasis) are exactly the two ingredients, and
both are built. The only *constraint* is that Zelezniak is ~95 kinase/phosphatase KOs — the
sole gene-indexed organic-acid metabolome in the triage — so Arm A's double-KO candidates
are kinase × (kinase or SGA-covered metabolic) pairs. That constraint is *aligned* with the
reg-network goal, not a deficiency.

**Enhancements worth adding (not blockers):**

- **Zhu/Loewen kinase-phosphatase lipidomics** ([[paper.north-star.dataset-triage]] row 6,
  Top-10 #4; ~129 kinase/phosphatase mutants × lipidome; not built) — a *second*
  regulatory-network metabolite class (fatty acids/lipids) on the same signaling axis as
  Zelezniak. Strongest single addition if the reg-network story is the headline.
- **Ambroset/Fay mQTL** (row 19; natural-variation metabolome; not built) — a natural-variant
  complement to Mülleder for a model-generalization check across perturbation types.
- **Leutert 2023 phosphoproteome** (row 56; WT-only, no deletion axis) — the direct
  regulatory-signal readout layer; useful as a feature/prior, not a genotype-perturbation set.

**Genuine gap — the isoprenoid/mevalonate pool (defer terpenes):** no dataset measures the
mevalonate/IPP/FPP pool across KOs. In the triage, dedicated terpenoid engineering papers
were **excluded** as single-strain demos; the only terpene *screen* candidate is
**Trikka/Makris sclareol** (row 25, diterpene, SI-table-only, not built); **Ozaydin 2013
beta-carotene** (built) is a *production* readout, not a pool metabolome. So the
beta-carotene/mevalonate chassis route has no pool anchor and is correct to defer. If
pursued later, adding a KO × isoprenoid-intermediate metabolomics dataset is the prerequisite.

### Detection methods

- succinate, malate → **HPLC organic-acid column** (Aminex HPX-87H, RI/UV) — simplest,
  standard. *(chosen arm)*
- tyrosine, phenylalanine → **HPLC-UV** (~274/280 nm, aromatic absorbance; no derivatization).
- valine / BCAA → HPLC with derivatization or LC-MS (not natively UV-active).
- mevalonate, phosphosugars (E4P/PEP) → LC-MS (harder) — another reason to defer terpenes.

### Honest caveats

- Mülleder and Zelezniak are **single-KO** (and Zelezniak is kinase-only); neither measured
  a double KO. The double-KO prediction is an **extrapolation** — single-KO metabolomics ×
  SGA epistasis structure — which is precisely what the wet-lab arm validates. That is both
  the scientific risk and the scientific value.
- Arm A's candidate space is bounded by the intersection of Zelezniak's kinase perturbations
  and SGA's interaction coverage; enumerate that intersection before committing strains.
- ecFactory predictions for terpenes/flavonoids are already strong (protein-constrained
  families), so those are the *worst* places to look for a CBM blind spot; amino-acid and
  organic-acid families (slightly/stoichiometrically constrained, feedback- and
  signaling-regulated) are the *right* places — consistent with choosing succinate/malate
  and the aromatic pools.

### Links

- [[paper.north-star.dataset-triage]] — dataset inventory + Top-10 priorities + gaps.
- [[paper.north-star]] — bioproduction-chassis positioning vs Qian 2026.
- `domenzainComputationalBiologyPredicts2025` — ecFactory / ecYeastGEM CBM baseline.
- [[torchcell.datasets.scerevisiae.zelezniak2018]] — reg-network metabolome (Arm A).
- [[torchcell.datasets.scerevisiae.mulleder2016]] — AA-precursor metabolome (Arm B).

## 2026.07.17 - CABBI alignment, precursor-pool proof-of-concept, cross-host strategies

### Framing: proof-of-concept on NATIVE precursor pools, no heterologous pathway

torchcell here is a **proof-of-concept for the virtual cell** (CABBI framing), not a
metabolic-engineering feat. Deliberately keep any heterologous pathway *out of the
prediction*: a heterologous enzyme adds reactions outside the trained network and the
metabolic model, so a claim about the heterologous product extrapolates beyond what the
model represents. Instead predict/measure **native precursor pools**, which sit entirely
inside the genotype → native-phenotype space the datasets cover; the pool doubles as a
chassis-capacity proxy for whatever product is bolted on downstream. This is why the
experiment targets pools (succinate, malate, acetyl-CoA, valine) rather than engineering
the products directly.

### CABBI alignment (grounded)

CABBI's target portfolio (grounded from cabbi.bio + genomicscience.energy.gov): **organic
acids** (3-HP → acrylic; citramalate → methacrylate; succinate, malate, citric, pyruvic in
*Issatchenkia orientalis*), **oleochemicals / fatty alcohols** (biodiesel, jet fuel,
lubricants), and the **triacetic acid lactone (TAL)** polyketide; hosts *I. orientalis*,
*Rhodosporidium toruloides*, *Yarrowia lipolytica*, and *S. cerevisiae*.

- **On-axis** (central-carbon / amino-acid-derived, all pool-anchored in our data):
  succinate, malate (Arm A); isobutanol (← valine); citramalate (← pyruvate + acetyl-CoA);
  3-HP (← aspartate / malonyl-CoA); fatty alcohols / TAL (← acetyl-CoA / malonyl-CoA).
- **Off-axis** (specialty/pharma, not CABBI-central): tyrosine-derived aromatics —
  betalains, benzylisoquinoline alkaloids, flavonoids, stilbenoids, tyrosol, vitamin E.
  Keep these as the general-bioproduction story, not the CABBI-grant story.

### Strategy — acetyl-CoA → TAL, *S. cerevisiae* discovery → *I. orientalis* transfer

**Author-flagged: acetyl-CoA is the limiting node for TAL production in our *I.
orientalis* strain.** TAL (4-hydroxy-6-methyl-2-pyrone) is made by a heterologous
2-pyrone synthase from one acetyl-CoA (starter) + two malonyl-CoA (extender, itself from
acetyl-CoA via ACC1), so acetyl-CoA is the central precursor feeding both units.

- **Plan:** use torchcell to find *S. cerevisiae* gene KOs (single + regulatory-network
  double) predicted to **raise the acetyl-CoA pool**; map the hits to their *I. orientalis*
  orthologs; test whether the acetyl-CoA-boosting strategy transfers to relieve the TAL
  bottleneck in IO.
- **Why it fits the proof-of-concept:** the *prediction* is on the **native acetyl-CoA
  pool** — no heterologous enzyme in the *S. cerevisiae* discovery loop; the heterologous
  2-PS is only the downstream converter in the IO production strain. Clean genotype →
  native-phenotype prediction plus a concrete **cross-host generalization test** — exactly
  the virtual-cell claim CABBI cares about.
- **Data anchor:** acetyl-CoA is measured in **Zelezniak 2018** (kinase/phosphatase-KO
  metabolome), so torchcell can learn KO → acetyl-CoA and rank regulatory double-KOs.
- **Cross-host step:** *S. cerevisiae* hit → *I. orientalis* ortholog mapping
  (kinase-signaling is broadly conserved across yeasts; use an orthology resource). This
  is the concrete instance of the host-transfer bridge.
- **Detection:** acetyl-CoA is an unstable CoA thioester → LC-MS/MS in *S. cerevisiae*
  (or use TAL titer as an acetyl-CoA reporter if 2-PS is expressed in the discovery host);
  TAL by HPLC/LC-MS in IO.
- **Caveats:** (1) **malonyl-CoA, the co-precursor, is NOT measured in any of our
  datasets** — the model sees acetyl-CoA but not the ACC1-derived extender pool; (2)
  acetyl-CoA is compartmentalized (cytosolic vs mitochondrial) and TAL synthesis draws on
  the *cytosolic* pool, which a whole-cell pool measurement may not resolve.

### Per-row dataset mapping (CABBI products → native pool → our datasets)

| CABBI product | Native pool | Our dataset(s) | Perturbation / readout |
|---|---|---|---|
| succinate | succinate | Zelezniak 2018 · Yoshida 2012 | kinase-KO SRM metabolome · 17 gene-deletion HPLC titers |
| malate | malate | Zelezniak 2018 · Yoshida 2012 | same two |
| citramalate | pyruvate + acetyl-CoA | Zelezniak 2018 (both) · Yoshida 2012 (pyruvate) | citramalate itself: none (heterologous CimA) |
| isobutanol | valine / α-ketoisovalerate | Mülleder 2016 (valine, genome-wide) · Zelezniak 2018 (valine + pyruvate) · López 2024 | López = YKO isobutanol/BCAA biosensor (α-ketoisovalerate) |
| 3-HP | malonyl-CoA / aspartate | Mülleder 2016 (aspartate) · Zelezniak 2018 (aspartate + acetyl-CoA) | **malonyl-CoA: none**; 3-HP itself: none |
| fatty alcohols / TAL | acetyl-CoA / malonyl-CoA | Zelezniak 2018 (acetyl-CoA) · da Silveira 2014 (lipidome) · Xue 2025 (FFA titers) | **malonyl-CoA pool: none** |

Cross-cutting: the double-KO epistasis prediction for every row rides on **Costanzo 2016 +
Kuzmin 2018/2020 SGA** (they measure no metabolite → method layer, not a per-row pool
source). **Xue 2025** is itself a *combinatorial-deletion* (multi-KO) titer set — direct
multi-KO evidence for the fatty-acid / TAL row, not just single-KO extrapolation.

### Dataset corrections (supersede the earlier "Dataset mapping" table)

The earlier table under-credited our holdings. We DO have loaders (all wired via the
`ce90fcdb` CABBI-metabolism-adapters landing) for:

- **Yoshida 2012** ([[torchcell.datasets.scerevisiae.yoshida2012]]) — WT + 17 gene
  deletions × organic-acid **HPLC titers** (acetate, citrate, malate, pyruvate, succinate,
  mM). Direct deletion → succinate/malate map with our planned assay.
- **da Silveira dos Santos 2014** ([[torchcell.datasets.scerevisiae.dasilveira2014]]) —
  kinase/phosphatase **lipidome** (the reg-network lipid dataset earlier miscalled "not
  built" — it exists).
- **Montaño López 2024** ([[torchcell.datasets.scerevisiae.lopez2024]]) — genome-wide YKO
  **isobutanol/BCAA biosensor** (Leu3p/LEU1-yEGFP reporting α-ketoisovalerate).
- **Xue 2025** ([[torchcell.datasets.scerevisiae.xue2025]]) — in-house **FFA
  combinatorial-deletion** titers (chassis + up to 3 stacked TF deletions).

State caveats: `dasilveira2014` had an LMDB-build caveat (needs an injected
`SCerevisiaeGenome`); `lopez2024` / `xue2025` are DOI-less in-house sources with privacy
handling pending — confirm before counting them in this build.

### Gaps (defer or add)

- **malonyl-CoA pool** — not measured; blocks a *direct* precursor anchor for 3-HP, TAL,
  and fatty alcohols (proxy through acetyl-CoA instead).
- **isoprenoid / mevalonate pool** — not measured; terpene chassis stays deferred.
- Non-native products themselves (citramalate, 3-HP, TAL) are absent from *S. cerevisiae*
  data by construction — the strategy is native-precursor-pool prediction + heterologous
  conversion downstream, never predicting the non-native product directly.
