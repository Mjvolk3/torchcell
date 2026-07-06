---
id: wls9f96e2x0wkq7ojlvm5ch
title: Mulleder2016
desc: ''
updated: 1783309845159
created: 1783309845159
---

## 2026.07.05 - Recon + Build Plan (WS9 amino-acid metabolome)

Genome-wide amino-acid metabolome of the yeast deletion collection. Rounds out the
metabolomics/omics-for-pretraining set alongside Kemmeren (expression, already landed).
Maps to `MetabolitePhenotype` (WS4). Citation key `mullederFunctionalMetabolomicsDescribes2016`.

### Provenance (verified this session)

- **Paper:** Mulleder et al. 2016, *Cell* 165:1282, "Functional Metabolomics Describes
  the Yeast Biosynthetic Regulome." DOI `10.1016/j.cell.2016.09.007`, PMID **27693354**,
  open access PMC5055083.
- **Raw data (scriptable, hash-pinned):** Mendeley Data DOI `10.17632/bnzdhd6ck8.1`
  ("The Saccharomyces cerevisiae amino acid metabolome", CC BY 4.0). File
  `Table_S3_Complete_Dataset.xls`, **8,502,272 bytes**, sha256
  `a7fcb4bc8aa5e394e7f6e2b99e327eaa88fa04111ab5602fc7cb3445f653802e`. Direct download:
  `https://data.mendeley.com/public-files/datasets/bnzdhd6ck8/files/621f3646-9e51-488a-b6b8-f6427b40fc87/file_downloaded`
  (listed via `https://data.mendeley.com/public-api/datasets/bnzdhd6ck8/files?folder_id=root&version=1`).
  `retrieval_method = direct_url`. Downloaded + sha256-verified 2026-07-05.
- No OCR needed for the DATA (structured .xls). OCR the paper PDF only for Methods
  provenance (replicate structure -- see open question).

### Table_S3 structure (inspected)

8 sheets. The loader consumes **`intracellular_concentration_mM`**: one row per ORF
(4678 strains) x **19 amino acids** in mM (batch-normalised, adjusted for dilution,
extraction volume, cell number + volume). The 19 = 20 standard AAs minus **cysteine**
(excluded, oxidation) -- note the paper text says "18", the DATA has 19; source the
count from the data. Other sheets: `data_raw` (raw uM, incl. 237 QC-sample rows +
slow-growers, 4831 unique ORF), `robust_summary_statistics` (POPULATION mean/SD/RSD per
AA, Minimum Covariance Determinant), `Z-score`, `p_value_Z-test`, `Mahalanobis_distance`,
`p_value_X2-test`.

### OPEN PROVENANCE QUESTION -- n_replicates / SE (do NOT guess)

The per-strain `intracellular_concentration_mM` sheet has **one value per (strain, AA)**;
`data_raw` shows **4637/4678 strains with a single measurement** (170 with 2, 20 with 3,
4 with 4). So the genome-wide screen is effectively **n=1 per strain**; the paper's
"mean +/- SD, n=3" refers to targeted VALIDATION, not the screen. **No per-strain SE
column is released** (only population-level robust SD per AA). Per CLAUDE.md "value not a
released per-record column": must READ THE METHODS (OCR paper) to confirm the screen
replicate structure before setting `n_replicates`; likely `n_replicates={aa:1}` and
`metabolite_level_se=None` (honest), with the population robust SD recorded as context
only, NOT as per-record error. RESOLVE via OCR before finalizing.

### MetabolitePhenotype mapping (plan)

- `metabolite_level = {amino_acid -> mM}` (19 AAs) from intracellular_concentration_mM.
- `measurement_type = "intracellular_concentration_mM"`.
- `n_replicates` / `metabolite_level_se`: per open question above.
- `target_metabolite_ids`: **populatable** -- amino acids ARE native Yeast9 metabolites
  (map each AA -> `s_NNNN`). First dataset where CBM linkage is real, not deferred.
- Genotype: single-gene KO per strain (ORF = systematic name; ~1029 unannotated gene
  names, key on ORF). **Background = PROTOTROPHIC deletion collection** (auxotrophy
  restored episomally, Mulleder et al. 2012) -- NOT standard BY4741; reference/genotype
  must reflect prototrophy restoration. Grown exponentially in MINIMAL medium.
- `Publication(pubmed_id="27693354", doi="10.1016/j.cell.2016.09.007")`.

### Build checklist

1. [ ] Deposit raw `Table_S3_Complete_Dataset.xls` + provenance record to the raw-data
   mirror; mirror the paper PDF (Zotero + torchcell-library) + MinerU OCR -> paper.md.
2. [ ] Resolve n_replicates/SE from Methods OCR (open question).
3. [ ] Decide amino-acid -> Yeast9 `s_NNNN` map (target_metabolite_ids).
4. [ ] Model prototrophic background on genotype/reference.
5. [ ] Loader `torchcell/datasets/scerevisiae/mulleder2016.py` (mirror cachera2023.py);
   register in datasets `__init__`.
6. [ ] Add to `torchcell.verification` metabolite runner; build LMDB; verify L0-L4.
7. [ ] Land via rebase + ff (own worktree `feat/ws9-mulleder-metabolome`).
