---
id: nynnujfognarend64bh087p
title: Zelezniak2018
desc: ''
updated: 1783350470165
created: 1783350470165
---

## 2026.07.06 - Built + Verified (WS9 kinase-KO proteome; L0-L4 PASS)

Quantitative SWATH-MS proteome of the yeast kinase-knockout collection. Second `/goal`
dataset (the mis-stated "Kemmeren" -> Zelezniak2018). Introduces a NEW phenotype family
`ProteinAbundancePhenotype` (protein abundance keyed by systematic ORF).

### Provenance (verified)

- **Paper:** Zelezniak et al. 2018, *Cell Systems* 7(3):269-283, "Machine Learning
  Predicts the Yeast Metabolome from the Quantitative Proteome of Kinase Knockouts."
  DOI `10.1016/j.cels.2018.08.001`, PMID **30195436**, open access PMC6167078.
- **Raw data (scriptable, hash-pinned):** Zenodo record 1320289 (concept DOI
  `10.5281/zenodo.1320288`, CC-BY-4.0), `proteins_dataset.data_prep.tsv`,
  14,313,785 bytes, sha256 `9ff81ecb1e2dd44d2f6e072ce5b628f0be1abdf57cdbd90d645db4d1fb64bfeb`,
  `retrieval_method=direct_url` (sha256-verified on download). Raw MS: PRIDE PXD010529.

### Structure + mapping

Long format: `ORF` (726 proteins), `sample`, `replicate`, `KO_ORF` (98 total: 97 kinase
KOs plus **WT**), `KO_gene_name`, `value` (batch-corrected/SVA label-free log signal). 264264
rows, 0 NaN, all ORF/KO_ORF systematic. Complete matrix: every strain has all 726
proteins; 2-12 replicate samples per (strain, protein) -> **SE always defined**.

- Aggregate per (KO_ORF, ORF) -> mean / SE=SD/sqrt(n) / n. 97 experiments (WT excluded).
- **Reference = the measured WT strain** (12 samples) -- a real control profile (unlike
  Mulleder's population-mean proxy). `ProteinAbundancePhenotype`:
  `protein_abundance={ORF->mean}`, `protein_abundance_se`, `n_replicates`,
  `measurement_type="swath_ms_label_free_log_signal_sva"`.
- Background: **BY4741-pHLUM** (prototrophic via the pHLUM minichromosome restoring
  HIS3/LEU2/URA3/MET17); `strain="BY4741"`, pHLUM not yet modeled as a GeneAddition.
- `Publication(pubmed_id="30195436", doi="10.1016/j.cels.2018.08.001")`.

### Schema addition

New `ProteinAbundancePhenotype` + `ProteinAbundanceExperiment(Reference)` (mirrors
`MetabolitePhenotype`), registered in the 3 unions + `EXPERIMENT_TYPE_MAP` /
`EXPERIMENT_REFERENCE_TYPE_MAP`; passes the schema-invariant gate. New verifier
`torchcell/verification/protein.py` (L0-L4, `reference_finite` for absolute abundances)
plus `run_protein` in `runners.py`. Verify: L0 97; L1 97==97 + 97 unique KO; L2 70422
values + SE; L3 reference_finite + measurement_type_consistent; L4 gene_containment
0.979 of the 97 kinases in Ohya. ALL PASS.

REVIEW FLAGS: (1) pHLUM prototrophy not modeled as GeneAddition (`strain="BY4741"`);
(2) metabolome sheet (`metabolites_dataset.data_prep.tsv`, ~46 LC-SRM metabolites) NOT
ingested -- optional MetabolitePhenotype follow-up; (3) absolute log signal stored (raw
form) -- log2(strain/WT) is derivable downstream, not baked.

- [x] ProteinAbundancePhenotype + loader + verifier + build + L0-L4 + registration
- [ ] metabolome sheet as a MetabolitePhenotype (optional)
- [ ] pHLUM prototrophy-restoring markers as GeneAddition

## 2026.07.07 - Metabolite dataset (WS9 kinase-KO metabolome; L0-L4 PASS)

`MetaboliteZelezniak2018Dataset` -- the metabolome sibling of the proteome loader, added
to the SAME module `torchcell/datasets/scerevisiae/zelezniak2018.py` (reuses the existing
`MetabolitePhenotype` / `MetaboliteExperiment` schema; NO schema change). First dataset to
populate `target_metabolite_ids` (metabolite -> Yeast9 `s_NNNN`). This finishes WS9.

### Raw file + provenance (verified)

- Same Zenodo record 1320289 (concept DOI 10.5281/zenodo.1320288), file
  `metabolites_dataset.data_prep.tsv`.
- The proteome loader's `?download=1` URL 403s for this file; use the Zenodo API content
  endpoint (Mozilla UA):
  `https://zenodo.org/api/records/1320289/files/metabolites_dataset.data_prep.tsv/content`.
- Size 229637 bytes; `sha256 = c4429fd8cef675d96ffacba1ed51e52ea483fd72d6978a22c04fa405f4e1b07d`
  (pinned as `METABOLITE_DATA_SHA256`, verified on download). Zenodo intermittently
  rate-limits ("403 unusual traffic"); retry with backoff.

### Column mapping (actual columns differ from the Zenodo README)

- Columns: `metabolite_id, kegg_id, official_name, dataset, genotype, replicate, value`.
- `genotype` = the strain (95 systematic kinase ORFs + literal `WT`) -- this is the strain
  column, NOT a KO_ORF column. All 95 non-WT validate against `_SYSTEMATIC_RE`.
- `metabolite_id` = BiGG-style id, 50 total; 5 are co-elution merges joined with `;`
  (`3pg;2pg`, `g6p;g6p-B`, `g6p;f6p;g6p-B`, `xu5p-D;ru5p-D`, `ala-L;ala-B`) -- KEPT verbatim
  as dict keys (honest to source; resolved to `s_NNNN` via the FIRST sub-id).
- `value` -> `metabolite_level` (mean over pooled replicate rows); range ~0.004-58995.

### Pooling-across-protocol decision

The `dataset` column is the "Protocol used for generation" (1/2/3). There is no per-record
column distinguishing protocols downstream, so `_aggregate` POOLS rows across BOTH `dataset`
(protocol) and `replicate` per (metabolite, strain): `metabolite_level` = pooled mean,
`metabolite_level_se` = sample_SD * n^-0.5 when n>1 else NaN, `n_replicates` = pooled row
count. Pooled-row distribution across all (strain, metabolite): n=1 -> 1347, n=2 -> 55,
n=3 -> 458, n=4 -> 148. Records where every metabolite has n=1 (77 of 95) collapse
`metabolite_level_se` to `None` (SE keys must be a subset of level keys; all-NaN -> None).

### measurement_type rationale

`measurement_type = "srm_ms_signal_batch_corrected"`. The README defines `value` as
"metabolite signal obtained from SRM-MS/MS experiment, corrected for batch effects" -- an
ARBITRARY batch-corrected SRM signal, NOT a concentration. The measurement_type records
this so the numbers are never silently compared to true concentrations (e.g. Mulleder mM).

### s_NNNN mapping (first real Yeast9 CBM linkage)

Module-level helper `build_metabolite_s_id_map({metabolite_id: kegg_id})` (reusable, e.g.
by Mulleder) reads `YeastGEM().model.metabolites`. Hybrid, model-sourced (ids come ONLY
from the model, never invented): prefer KEGG (`kegg.compound` annotation, matched on the
first `;`-separated `kegg_id` token), fall back to BiGG (`bigg.metabolite`, first token of
`metabolite_id`); cytosol (`c`) preferred, else the available compartment. All 50/50 ids
resolve. 49 map to cytosolic `s_NNNN`; the sole non-cytosolic exception is `b124tc`
(But-1-ene-1,2,4-tricarboxylate, KEGG C04002 -> mitochondrial `s_0454`, no cytosolic form).
`s7p` matched via BiGG (no KEGG annotation on its cytosolic species). `target_metabolite_ids`
is populated per-record, subset to the metabolites that strain measured.

### Reference = measured WT (sparse), restricted per strain

Reference (`reference_centered=False`) = the measured `genotype == "WT"` strain (a real
control), NOT a centered 0. Targeted-metabolome coverage is sparse and per-strain: strains
measure 13-50 metabolites each, and WT measured only 45 of the 50 ids (NEVER
`adp/amp/atp/e4p/fum`). So a strain can measure metabolites WT lacks (85/95 strains do).
Each record's reference is the WT baseline RESTRICTED to that strain's measured metabolites
(reference keys = experiment ∩ WT, a subset; every reference value a real WT measurement,
never invented). Metabolites a strain measures but WT lacks simply have no WT baseline.

Because the reference keys are a strict subset (not equal) for this sparse dataset, the
verifier's `_l3_reference_finite` was relaxed from key-EQUALITY to key-SUBSET + non-empty +
finite (Mulleder's dense case still satisfies subset + equal cardinality, so it still
passes).

### Verify (L0-L4 PASS)

95 strains x (13-50) metabolites. L0 95 records; L1 95==95 + 95 unique KO; L2 2008 level
values + 661 SE; L3 reference_finite (key-subset) + measurement_type_consistent
`srm_ms_signal_batch_corrected`; L4 gene_containment 0.979 of the 95 kinases in Ohya. ALL
PASS. Registered via `@register_dataset` + `MetaboliteZelezniak2018Dataset` export in
`datasets/scerevisiae/__init__.py`; spec added to `METABOLITE_DATASETS` in
`verification/runners.py` (`reference_centered=False`, expected_count 95). mypy --strict
clean.

- [x] MetaboliteZelezniak2018Dataset + s_NNNN mapping + build + L0-L4 + registration (WS9)
- [ ] pHLUM prototrophy-restoring markers as GeneAddition (shared with proteome)

## 2026.09.23 - SM media components (issue 143)

- Both loaders now emit `SM_DEFERRED`, a DISTINCT media node from the Mulleder/Messner `SM`:
  this paper's "Strains and Culture" names the medium and states no recipe, deferring to
  Mulleder et al. 2012 (not mirrored), so the composition is a single
  `composition_deferred` component and the grams are not borrowed from the other papers.
  Reasoning in [[torchcell.datamodels.media-components]]. Both LMDBs rebuilt + L0-L4 re-run.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `DATA_SHA256`, `METABOLITE_DATA_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Proteome aggregation refuses a blank value or a repeated replicate id

Previous behavior: `ProteomeZelezniak2018Dataset._aggregate` grouped by protein and took `count` as `n`, never reading `replicate`. A repeated (ORF, KO_ORF, replicate) row was counted as an extra replicate (larger `n`, smaller SE), a blank value silently left `n` one short, and a protein blank in every row of a strain aborted later in `ProteinAbundancePhenotype` validation ("n_replicates for YBR002C must be >= 1").

Fix (issue #520): `_aggregate(sub, strain)` refuses, naming the strain (WT included), the row count and the first offending (protein, replicate), for any blank value and for any repeated (protein, replicate) id. The all-blank case now refuses there too.

Record-neutral, measured on the pinned `proteins_dataset.data_prep.tsv` (sha256 `9ff81ecb...`): 264,264 rows, 0 blank values, 0 rows in repeated (ORF, KO_ORF, replicate) groups, 0 all-blank (strain, protein) cells, 0 cells with n < 2 of 71,148. Script: the b4-zelezniak scratchpad `classify_zelezniak.py`.

Tests: `test_proteome_repeated_replicate_id_refuses_naming_the_strain`, `test_proteome_repeated_replicate_id_in_the_wt_reference_refuses`, `test_proteome_blank_value_refuses_instead_of_shrinking_n`, `test_proteome_all_blank_protein_refuses_with_a_loader_message`.

Not changed, observed while measuring: the metabolome file pools rows across the `dataset` protocol column, and 378 rows share a (metabolite, genotype, replicate) id across protocols (0 when `dataset` is included in the key). For `3pg;2pg` the two protocols differ by about three orders of magnitude (WT replicate 1: 850.27 under protocol 1, 0.388 under protocol 2), so pooling them into one mean and SD is worth a separate review. Out of scope for issue #520.

Review follow-up (same day): `inf` passed the blank check, so a non-finite value now refuses the same way, naming the strain and the first cell. Recounted on the pinned file: 0 non-finite values of 264,264 (float64 column). Test: `test_proteome_non_finite_value_refuses`.

## 2026.10.02 - Metabolome protocols kept separate (issue 595)

Issue #595. `MetaboliteZelezniak2018Dataset._aggregate` grouped one strain's rows by `metabolite_id` only, so rows from the three LC-SRM protocols in the `dataset` column were averaged into one mean and SE. The paper (mirrored `paper.md`, sha256 `072bfb2d...`, STAR Methods, Metabolomics) states that Dataset 1 is not calibrated ("quantified by external calibration (except Dataset 1)"), was measured on the proteomics cultures, and that Datasets 2 and 3 come from re-grown cultures on different chromatography (ion-pair for 2, HILIC amino acids for 3). The pooled mean mixed two unit systems, the stored SE was the between-protocol scale gap, and every record's WT reference was pooled the same way. The old label "ARBITRARY batch-corrected SRM signal, NOT a concentration" also covered protocols 2 and 3, which are calibrated.

Fix:

- One record per (strain, protocol). The aggregation key includes `dataset`, so `n_replicates` counts one protocol's rows.
- The WT reference is aggregated per protocol, and each record is referenced only against the WT of its own protocol (restricted to the metabolites the strain measured, as before).
- The protocol is carried on the existing `MetabolitePhenotype.measurement_type` field (no schema change), on both the experiment and the reference. `ZELEZNIAK_METABOLITE_PROTOCOLS` (pydantic `ZelezniakMetaboliteProtocol`) holds per protocol: `measurement_type`, `calibrated`, `unit` or a typed `unit_gap`, and `SourcedValue` quotes (sha256-pinned, audited by `test_protocol_quotes_are_verbatim_in_the_mirrored_paper`).
  - 1: `lc_srm_signal_batch_corrected_uncalibrated_dataset1`, uncalibrated, unit "arbitrary units (batch-corrected SRM peak signal, NOT a concentration)".
  - 2: `lc_srm_ion_pair_external_calibration_batch_corrected_unit_unstated_dataset2`, calibrated, unit gap.
  - 3: `lc_srm_hilic_external_calibration_batch_corrected_unit_unstated_dataset3`, calibrated, unit gap.
- `CALIBRATED_UNIT_GAP` (`deferred_pending_source_review`): the Methods state the calibration standards (500 uM to 100 nM) and a dilution plus cell-volume conversion for Figures 4F-4H, but neither the paper nor the processed file states the unit of the deposited calibrated value or whether that conversion was applied. Resolve with the Zenodo record 1320289 README (not mirrored) or the authors' `kinase_metabolism` code.
- New refusals, each with an exact-message test: a `dataset` value with no sourced protocol, strain rows on a protocol with no WT rows, a (strain, protocol) sharing no metabolite with its WT, a repeated (metabolite, replicate) id inside one protocol. None fires on the pinned release.
- Verifier: `verify_metabolite_dataset(..., protocol_measurement_types=...)` keys L1 uniqueness on (strain, protocol) and requires each record's type to be declared and its reference to share it. The registry entry declares the three types and `expected_count` 129.

Measured by `experiments/036-dataset-fixes-before-kg-build/scripts/zelezniak2018_protocol_split.py` (raw `metabolites_dataset.data_prep.tsv` sha256 `c4429fd8...`, old dev-tree LMDB, new scratch build of this loader), results in `experiments/036-dataset-fixes-before-kg-build/results/zelezniak2018_protocol_split.json` and `zelezniak2018_protocol_split_examples.csv`:

- Raw: 3,522 rows, 2,053 (genotype, metabolite) cells, 189 cells pool protocols 1 and 2 (704 rows; 10 metabolites, 19 genotypes); ratio of per-protocol means in a pooled cell min 5.0x, median 139.6x, max 2,194x; 115 cells over 100x, 10 over 1,000x. 0 repeated (dataset, metabolite, genotype, replicate) rows.
- Old store: 95 records, one measurement_type; 18 records carry 179 pooled experiment cells; all 95 carry pooled reference cells (950).
- New store: 129 records (protocol 1: 95, protocol 2: 18, protocol 3: 16); 0 duplicate (strain, protocol) keys; 2,187 experiment cells and 1,946 reference cells each equal the mean, n and SE of exactly one protocol's raw rows (0 mismatches, so 0 cells pool protocols); 0 records whose reference is on another protocol. `verify_metabolite_dataset` with the registry spec passes on the scratch build.
- Examples: WT `3pg;2pg` was mean 425.329, SE 424.941, n 2; now 850.270 (protocol 1, n 1) and 0.388 (protocol 2, n 1), each in its own protocol's references. YIL042C `r5p` was mean 283.205, SE 282.456, n 4; now 1130.574 (protocol 1, n 1) and 0.749, SE 0.086, n 3 (protocol 2).

Open:

- The calibrated unit is a typed gap until the Zenodo README or the authors' code is mirrored (needs a retrieval go-ahead).
- Downstream readers of the pooled values (experiment 019 metabolome heads and fig6 queries, amino-acid and metabolite comparisons) now see up to three records per strain on different scales. `MeanExperimentDeduplicator` groups by `experiment_type` plus genes, so applied to this dataset it would merge a strain's protocol records again (read from `torchcell/data/mean_experiment_deduplicate.py`; not run).
- `DATASET_FULL_RECORDS` in `torchcell/knowledge_graphs/build_time_projection.py` still says 95; it is a gathered snapshot of the dev tree and is regenerated after the dev rebuild.
