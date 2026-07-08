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
