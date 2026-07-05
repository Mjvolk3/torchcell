---
id: l8393oimoopk2ab7dmwakw7
title: Ozaydin2013
desc: ''
updated: 1783210882206
created: 1783210882206
---

## 2026.07.04 - WS7 build: Ozaydin beta-carotene visual screen

Roadmap [[plan.schematization-ingestion-roadmap.2026.06.23]] WS7. The abstract's
β-carotene case. Maps to the new `VisualScorePhenotype` (WS4).

### Source + provenance (full extraction machinery exercised)

- Paper + SI found in Zotero (group `6582362`) by DOI `10.1016/j.ymben.2012.07.010`.
  Zotero holds the **PDF only**; the data SI is NOT in Zotero.
- **SI xlsx** (`1-s2.0-S109671761200081X-mmc1.xlsx`, 367 KB, sha256 `4818726e…`)
  fetched from the **Elsevier ESM** direct URL (`ars.els-cdn.com/...mmc1.xlsx`).
- **paper.pdf** pulled from Zotero via `torchcell.literature.ZoteroLibrary`; **MinerU
  OCR** (`swanki-mineru` env) → `paper.md` to capture the strain construction.
- All three in `$DATA_ROOT/torchcell-library/ozaydinCarotenoidbasedPhenotypicScreen2013a/`
  with a `manifest.json` (role + sha256 + retrieval per file).

### The screen (from OCR'd paper.md — corrects the iBF note)

Base strain **BY4741** (Open Biosystems YKO collection) transformed with plasmid
**YB/I/BTS1** = `YEplac195 TDH3p-crtYB-CYC1t; TDH3p-crtI-CYC1t; TDH3p-BTS1-CYC1t`
(Verwaal 2007): crtYB + crtI from *X. dendrorhous* + an extra copy of the **native
GGPP synthase BTS1** (NOT crtE — the iBF note said CrtE; the OCR shows BTS1). Colony
color on a **-5..+5** scale (WT carrying the plasmid = 0) is a visual proxy for
carotenoid (β-carotene) accumulation. Scored on SC-URA agar, 30 °C. This heterologous
cassette is the background a future CBM adds on top of Yeast9; it is captured in
paper.md so the CBM is buildable later (per user: capture-complete, modeling-light).

### Dataset (`CarotenoidOzaydin2013Dataset`)

Per-ORF aggregation of SI Sheet 1: `visual_score = max` numeric color across replicate
plates, `visual_score_min = min`, `n_replicates = count`, QC flags parsed from the
Comment free-text. Sheet 2 (TOP200) merged for gene name/function/category + `in_top200`.
Base strain (**BY4741 / BY4730**) captured per record. `target_metabolite_id` left
`None` (Yeast9 mapping deferred).

**4474 records** (ORFs with a numeric color). Excluded, never silently: text-only rows
(`pet`/`tiny`/`_`) and **malformed ORF names** (e.g. `YLR287-A` missing the W/C — NOT
guess-repaired). Score distribution matches the paper (0 = 2020 majority, spanning
-5..+5). L0-L4 all PASS (`torchcell/verification/visual_score.py`); L4 gene overlap
with the Ohya deletion collection = **0.946**.

### Schema note (WS4)

`VisualScorePhenotype` (schema.py): ordinal `visual_score` within a declared
`[score_scale_min, score_scale_max]`, `score_semantics`, `target_product`
(+ optional Yeast9 `target_metabolite_id`), `n_replicates`, `qc_flags`, `score_text`.
A defensive fix to `MicroarrayExpressionPhenotype.n_replicates` (raise a clean
ValueError on non-dict, not crash on `.items()`) was needed so pydantic union
resolution can skip that member when a scalar-`n_replicates` phenotype is the match.

## 2026.07.05 - Follow-up Fixes + Plasmid Provenance + Cassette-Perturbation Design

### Done + committed (branch `fix/ws8-ozaydin-cachera-followups`)

- **PubMed ID.** `Publication` now carries `pubmed_id="22918085"` +
  `pubmed_url`. PMID<->DOI (`10.1016/j.ymben.2012.07.010`) confirmed via NCBI
  E-utilities (title "Carotenoid-based phenotypic screen...", Metabolic
  Engineering 2013). Not guessed.
- **`qc_flags` -> `comment_annotations`.** `VisualScorePhenotype`'s 7 booleans
  are parsed from the free-text Comment column and are a MIX, NOT all QC: true QC
  (`flag_qc_failure`, `flag_het_diploid`), secondary growth/physiology PHENOTYPES
  (`flag_petite`, `flag_tiny`, `flag_slow_growth`), interpretation caveats
  (`flag_sterile`, `flag_unusual_color`). Renaming avoids callers filtering on
  these as if they were all quality failures. Rebuilt Ozaydin LMDB (to a scratch
  root) round-trips the renamed field; 260/4474 records carry >=1 true annotation.
  NOTE: the canonical LMDB at `$DATA_ROOT/data/torchcell/carotenoid_ozaydin2013`
  still holds the OLD `qc_flags` key and MUST be rebuilt in place when this lands.

### Plasmid / cassette definition -- sourced from OUR mirror (provenance-first)

From `torchcell-library/ozaydinCarotenoidbasedPhenotypicScreen2013a/paper.md`
(MinerU OCR of the paper PDF):

- **Screen plasmid YB/I/BTS1** (Table 2, verbatim): `YEplac195 TDH3p-crtYB-CYC1t;
  TDH3p-crtI-CYC1t; TDH3p-BTS1-CYC1t`, ref Verwaal et al. (2007).
  - Backbone **YEplac195** = URA3-marked **2micron (episomal, multi-copy)** vector
    -> scored on SC-URA (selective). `localization = episomal_2micron`.
  - Heterologous: **crtYB** (bifunctional phytoene synthase/lycopene cyclase),
    **crtI** (phytoene desaturase), both from *Xanthophyllomyces dendrorhous*.
  - Native, extra copy: **BTS1** (GGPP synthase, YPL069C) -- NOT crtE.
  - Promoters/terminators: all TDH3p / CYC1t.
- **Deposition (paper line 38, verbatim):** "Information about all the strains and
  plasmids have been deposited in the public instance of the JBEI Registry
  (<https://public-registry.jbei.org/>)." Detailed construction is in the paper SI.
- **Full sequence (#4, external, not yet mirrored):** JBEI Registry + Verwaal et al.
  2007 (Appl. Environ. Microbiol. 73, 4342-4350). Our mirror has the COMPOSITION
  (enough to name genes + localization for the schema); the backbone SEQUENCE is
  the remaining external dig.

### Heterologous-cassette perturbation -- DESIGN SIGN-OFF STILL PENDING

The schema has only loss-of-function perturbations (Deletion/Damp/Allele/Ts/
Suppressor), all subclassing `GenePerturbation` whose `systematic_gene_name`
validator requires the native yeast pattern `Y[A-P][LR]\d{3}[WC]...`. Heterologous
genes (crtYB, crtI) fail it, so today the cassette lives ONLY in docstrings -- the
engineered background is invisible in the data. The cassette is a CONSTANT chassis
across all ~4800 strains AND the reference; the per-strain variable is the single KO.

Three shapes were put to the user (2026.07.05); recommendation = **Option A**:

- **A. Per-gene addition perturbation (recommended).** New `GeneAdditionPerturbation`
  family alongside deletions in the same `perturbations` list; per-gene
  `localization` (episomal_2micron | chromosomal_integration) + `source_organism` +
  `construct`; validator branches to allow heterologous names. Most reusable, keeps
  all genetic mods in one consumed place; cassette genes repeat on the reference.
- **B. Genotype-level background.** Structured `engineered_background` field on
  `Genotype` (identical on the reference), NOT in the perturbation list. Cleanly
  separates assay chassis from the screened KO, but perturbation-consuming models
  won't see the cassette genes.
- **C. Single cassette-object perturbation.** One `HeterologousCassettePerturbation`
  in the list carrying localization + backbone/locus + nested member genes. Keeps
  the "one construct" grouping but nests structure + breaks one-pert=one-gene.

Open sub-decision (native extras): BTS1 extra copy (Ozaydin) and ARO4^K229L /
ARO7^G141S (Cachera) are NATIVE genes -- model with the same addition type flagged
`is_heterologous=False`, or reuse the existing `AllelePerturbation`? Deferred to the
shape decision.
