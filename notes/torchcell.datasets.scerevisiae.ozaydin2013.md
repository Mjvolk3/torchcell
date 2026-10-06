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

### Heterologous-cassette perturbation -- DESIGN ADOPTED (Option A; see 2026.07.15 below)

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

### 2026.07.05 - Plasmid availability (answer: NO downloadable assembled plasmid)

Verified by web research (do they provide the plasmid so it can be downloaded?):

- **Verwaal et al. 2007 deposited NO plasmid sequences and NO accession numbers**
  (open-access PMC1932764). It only points to NCBI for the individual carotenogenic
  genes; no GenBank for the YEplac195-YB/I / YB/I/BTS1 / YB/I/E constructs.
- **Screen plasmid YB/I/BTS1 is PHYSICAL-only at EUROSCARF = accession P30796**
  (siblings: YB/I = P30795, YB/I/E = P30797). Euroscarf distributes the physical
  plasmid; no full-construct GenBank download.
- **To get a raw plasmid sequence for the collapse path, two options:**
  (a) reconstruct in silico from parts -- YEplac195 backbone (NCBI, gi:415332, has a
  GenBank) + crtYB/crtI accessions (*X. dendrorhous*) + BTS1/YPL069C (S288C) +
  TDH3p/CYC1t (S288C); or (b) order P30796 from Euroscarf and sequence it.
- Net: Ozaydin's "plasmid-as-raw" is a MANUAL reconstruction/deposit job (no direct
  download), unlike Cachera where a GenBank map exists. Both still end at our mirror
  with sha256 provenance; only the retrieval recipe differs (reconstruct vs download).

## 2026.07.15 - Pre-adapter audit: design adopted + sha256 + rebuild

Pre-adapter cleanup ahead of the graph-DB rebuild
([[plan.ozaydin-cachera-preadapter-cleanup.2026.07.15]]). Loader is current and rebuilds
clean: **4474 usable ORFs** (Sheet 1 whole-collection screen; text-only/malformed rows
excluded and logged -- this is the full screen, not a PDF selection).

### Cassette-perturbation design -- ADOPTED (Option A)

The three-option decision above is settled and **implemented in code**
(`ozaydin2013.py::_carotenogenic_cassette`): the YB/I/BTS1 chassis is three per-gene
`GeneAdditionPerturbation`s in the genotype `perturbations` list --
`crtYB` + `crtI` (`is_heterologous=True`, `source_organism="Xanthophyllomyces
dendrorhous"`) and native `BTS1`/YPL069C (`is_heterologous=False`) -- all with
`localization="episomal_2micron"`, `construct_name="YB/I/BTS1"`. The native-extras
sub-decision resolved to: **model native additions with the same `GeneAdditionPerturbation`
type flagged `is_heterologous=False`** (BTS1 here; Cachera's ARO4^K229L/ARO7^G141S use the
same type with `variant=`), NOT a separate native-addition type and NOT `AllelePerturbation`
(these are extra/ectopic copies, not native-locus edits). So the Ozaydin genotype is
`{kanmx_deletion: 1, gene_addition: 3}`. `plasmid_contig_id`/`locus_tag` stay `None` until
the plasmid-sequence store lands (Ozaydin plasmid is physical-only, reconstruct-from-parts).

### sha256 + rebuild

Added `_SI_SHA256` pin + verification to `download()` (the stored SI is canonical, not the
Elsevier URL). The on-disk canonical LMDB predated the required `Media.is_synthetic` field
and failed schema round-trip; rebuilt in place under `$DATA_ROOT` as part of this cleanup.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `_SI_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Strain of the scored rows, mixed-strain scores refused (issue #528)

- Before: `records.setdefault` kept the first row's strain, so an ORF scored on two backgrounds was labeled with whichever row came first.
- Now: the record's strain is the strain its numeric scores were taken on; numeric scores on two strains raise `MixedStrainScoresError` ("Ozaydin: <ORF> has numeric color scores on strains [...]; one record carries one reference genome"). An ORF with no numeric score keeps its first row's strain and is excluded anyway.
- Record-neutral on the pinned SI: one ORF is listed on two strains (YML086C: BY4730 scored 1, BY4741 reads `pet`), its numeric score is BY4730's and BY4730 is its first row; the whole per-ORF aggregate of 4,975 ORFs is identical before and after (`ozaydin_agg_snapshot.py` in the fix scratchpad).
- Left open (record-changing): equal replicate scores give `visual_score_min` equal to the score on 34 records (the schema documents None only for a single replicate, so this is consistent with it); the free-text `Media(name="SC-URA")` stub and the cassette's `plasmid_contig_id`, `locus_tag` and `integration_locus` touch all 4,474 records.
- Tests: `test_numeric_scores_on_two_strains_refuse_the_build_by_name`, `test_the_strain_is_the_one_the_score_was_taken_on`, `test_replicates_on_one_strain_aggregate_and_or_the_flags`.

## 2026.10.02 - Sourced medium replaces the inline stub (issue #622)

Issue #622: the screen medium was `Media(name="SC-URA", state="solid", is_synthetic=True)`, a stub with no components and no provenance, and its name matched no library object.

Methods line 42 (mirror `paper.md`, sha256 `df2f2b63...`): transformants were "spotted onto selective medium agar plates (SC-URA)", and the carotenoid color was scored after "about 5 ul of this culture was spotted onto SC-URA plates. After 2 day of growth at 30 C". The paper prints no SC recipe and no agar amount. The loader now emits `OZAYDIN_SC_URA_AGAR`: the library `SC_URA` (base `SC`, uracil dropped) made `solid`, plus an agar row with NO concentration, quoting line 42 (`SOURCED_VALUES`). Agar is therefore an open gap (`Media.open_gaps`) rather than a borrowed 2%. Loader-local because the library `SC_URA` is liquid and no other served dataset states this plate.

Measured by `experiments/036-dataset-fixes-before-kg-build/scripts/media_stubs_seven_loaders.py` (output `experiments/036-dataset-fixes-before-kg-build/results/media_stubs_seven_loaders.csv`): scratch build 4,474 records (same as the dev store); every one of the 8,948 experiment and reference environments carries `OZAYDIN_SC_URA_AGAR` (32 components, 1 dropout, 2 provenance entries, 31 open gaps), against the stub on all 8,948 in the dev store. Its `media_identity` matches no library object (the state and the agar row differ from `SC_URA`). The Finding test `test_medium_is_a_free_text_stub_not_the_library_sc_ura` is replaced by `test_medium_is_the_sourced_sc_ura_agar_plate`.
