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
