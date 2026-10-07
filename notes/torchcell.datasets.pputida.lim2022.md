---
id: r3ozrjj2rw791itano5n5wi
title: Lim2022
desc: ''
updated: 1791382918594
created: 1791382918594
---

## 2026.10.07 - putidaPRECISE321 loader (plan Step 8, rank 7)

Source: `torchcell/datasets/pputida/lim2022.py` (`PutidaPrecise321Lim2022Dataset`)
Tests: `tests/torchcell/datasets/pputida/test_lim2022.py`
Plan: [[plan.bacteria-ontology-genome]] section 4 and Step 8 (row 7, status `aggregation`).
Skeleton: [[torchcell.datasets.bacteria_common]]; schema: [[torchcell.datamodels.bacterial-perturbation-ontology]].

Lim et al. 2022, "Machine-learning from Pseudomonas putida KT2440 transcriptomes reveals
its transcriptional regulatory network", Metab. Eng., doi:10.1016/j.ymben.2022.04.004
(citation key `limMachinelearningPseudomonasPutida2022`, `paper.md` sha256
`d5f49a99...`).

### Aggregation finding (measured from the Supplementary Data sample sheet)

The sheet `1-Sample_list` of `si/si2.xlsx` has 550 rows; 321 carry
`putidaPRECISE321 = 1`, over 118 conditions and 21 projects, which matches the paper
("It consists of 321 samples from 118 unique experimental conditions across 21
projects"). The `Note` column marks the samples generated for this paper ("in-house
data"): **96 of the 321 are new, 225 are reprocessed.**

| source (sheet `DOI`, else `PMID`, else project) | projects | samples | conditions | new here |
|---|---|---|---|---|
| Lim 2022 in-house, NREL | Muconate | 80 | 27 | yes |
| Lim 2022 in-house, UNMC | Aromatic | 12 | 4 | yes |
| Lim 2022 in-house, UCSD | ALE | 4 | 2 | yes |
| doi:10.1099/mic.0.000875 (Pobre 2020) | Bioreactor2 | 36 | 12 | no |
| doi:10.1128/AEM.03236-16 (Bojanovic 2017) | Multistress | 21 | 7 | no |
| doi:10.1038/s41598-018-23858-6 (Miyazaki 2018) | Mobile_gene | 18 | 5 | no |
| doi:10.3389/fmicb.2020.01099 (Nakamura 2020) | HNS_protein | 18 | 9 | no |
| doi:10.1039/D0GC01663B (Lim 2020) | ALE | 16 | 7 | no |
| doi:10.1021/acssuschemeng.1c03765 (Lim 2021) | ALE | 14 | 7 | no |
| pmid:30298597 | Carbon | 11 | 4 | no |
| doi:10.1016/j.ymben.2020.01.001 (Bentley 2020) | Muconate | 9 | 3 | no |
| doi:10.1128/AEM.01210-18 (Xiao 2018) | FinR | 6 | 2 | no |
| doi:10.3389/fmicb.2020.573857 (Lei 2020) | Zn | 6 | 2 | no |
| pmid:25711694 | Crc | 6 | 3 | no |
| doi:10.1039/C8EE00460A (Jayakody 2018) | Glycolaldehyde | 4 | 2 | no |
| no publication in the sheet: BioProject PRJNA535401 | FleQ | 9 | 3 | no |
| no publication in the sheet: PRJEB9788 | ZnO | 8 | 3 | no |
| no publication in the sheet: PRJNA338876, PRJNA338850 (a DTU PhD thesis URL in `DOI`) | Phase | 8 | 3 | no |
| no publication in the sheet: PRJNA480068 | Myristic_acid | 6 | 2 | no |
| no publication in the sheet: PRJDB5812 | Mobile_gene2 | 6 | 2 | no |
| no publication in the sheet: one BioProject per sample (PRJNA455030-455037) | Fuel | 6 | 3 | no |
| no publication in the sheet: one BioProject per sample (PRJNA455888-455897) | RelA | 9 | 3 | no |
| no publication in the sheet: one BioProject per sample (PRJNA457015-457248) | Carbon2 | 6 | 2 | no |
| no publication in the sheet: PRJNA520374, PRJNA520375 | Bioreactor1 | 2 | 1 | no |

The full machine-written table, with every BioProject and verbatim `DOI` cell, is
`preprocess/build_report.json` (`aggregation.by_source`).

**The paper's own counts disagree with each other.** The Introduction says "of which 305
were previously published and 16 were newly generated in this study"; the Methods count
"97 samples were newly generated in this study" among the 541 collected; the sheet flags
97 in-house rows of which 96 are in the compendium. The 16 equals the in-house samples
whose `CenterName` is `SBRG` (12 aromatic, 4 ALE); the 80 NREL muconate samples carry no
DOI, `Note` "in-house data, carbon metabolism engineering", and share BioProject
PRJNA796354 with the 16. The SI says "The 81 samples for the 'muconate' project were
prepared as described previously (Bentley et al., 2020)". The loader follows the sheet's
`Note` column, so a record's in-house status is the sheet's, not the Introduction's.

**Net new to torchcell: all of it.** No served dataset carries a P. putida transcriptome,
none of the 21 reprocessed sources (12 named by a DOI or PMID, 9 by BioProject only) is its own row among the fifty, and the expansion document
already merged the one duplicate (putidaPRECISE321 returned under two keys).

**What the loader stores.** One record per compendium sample whose genotype can be
written against KT2440 (180 of 321). Each record's `Publication` names its source: the
sheet's DOI, else its first PMID, else Lim 2022 (in-house samples, and public samples whose
row names no publication; their processed profile is first published by Lim 2022). Per
sample, `preprocess/sample_ledger.json` holds the SRX, project, condition, replicate,
reference condition, the `SourceStudy` (DOI cell verbatim, PMID, BioProject, GEO series,
center, in-house flag), the publication, the count column, the strain call and the drop
reason. Records are per sample, not per condition: the release is per sample, the phenotype
class has no replicate field, and summing counts across replicates would fabricate a
library.

### Data and retrieval

Raw mirror `$DATA_ROOT/torchcell-raw/limMachinelearningPseudomonasPutida2022/`, written by
`deposit_raw_mirror` (idempotent by sha256, refuses a differing file):

| file | sha256 | retrieval |
|---|---|---|
| `si/si2.xlsx` | `10e81a18fdfd08b9e27581f877dd0ae2a169946508b1ed5dabbe444210420970` | `elsevier_mmc(pii="S1096717622000635", filename="mmc2.xlsx")`, the library mirror's record |
| `data/counts.csv` | `59a8a1ac6b45c44a490d15a8c24a58c1083b623ed215aec758cad687cd8dbfc0` | `direct_url` of `raw.githubusercontent.com/SBRG/modulome_ppu/f63a0df.../data/raw_data/counts.csv` |

The GitHub repository is the one the paper's Data availability names
(`Source codes for iModulon analysis and figures are available at https://github.com/SBRG/modulome_ppu.`);
`f63a0df` (2022-04-28) is its last commit and no release or tag exists. Its
`data/raw_data/log_tpm.csv` equals the SI X matrix to 1.8e-15, so the SI is the expression
authority and the repository adds only the counts.

- **Unit, back-solved.** The Methods say counts were "converted into $\log _ { 2 }$
  transcripts per million (TPM)" without the pseudocount. Measured: `2**X - 1` sums to
  1,000,000 in every one of the 321 columns (and `2**X` to 1,005,564), so X is
  log2(TPM + 1). Stored: `expression_tpm = 2**X - 1`, `measurement_type="rnaseq_tpm"`.
- **Counts pairing, measured.** `pair_count_columns` accepts a count column for a sample
  when `log2(k * c / L + 1)` reproduces X below 1e-6 (`k` the median TPM-to-rate ratio,
  `L` the pinned annotation's gene length), and requires exactly one such column. Result:
  291 samples pair with their own column, 30 with an in-house `SBRG_*` column (14
  galactose/xylose `SRX88945xx` samples absent from `counts.csv` by name, and the 16
  ionic-liquid `SRX82457xx` samples), max deviation 5.0e-9. For those 16, the same-named
  SRX column exists and does NOT reproduce the published values (deviation 0.19 to 1.15).
  One gene is left out of the comparison: PP_1495 (prfB), whose release gene-table length
  is 72 bp (the first segment of a frameshifted CDS) while the annotation's feature is
  1,096 bp; the implied featureCounts length is 1,095.
- **Reference genome.** "Sequencing reads were aligned to the reference genome
  (AE015451.2)", the single chromosome of `GCA_000007565.2` (`pputida_KT2440_ASM756v2`),
  so `genome_reference = assembly_reference("KT2440")`. "Additionally, non-P. putida
  KT2440 samples were excluded for consistency of gene expression profiles."

### Identifier histogram (`reconcile_locus_tags`)

- **X gene ids**: 5,564 unique, 5,564 CURRENT at the locus-tag layer, 0 remapped, 0
  outside the namespace. Threshold `MIN_RESOLVED_FRACTION = 1.0` (any unresolved id stops
  the build).
- **Deleted-gene symbols from condition labels**: 8 unique; 2 resolve at the gene-symbol
  layer (`relA` -> PP_1656, `fleQ` -> PP_4373, which the paper also writes "FleQ
  (PP_4373)"); 6 are RETIRED (`crc`, `crcY`, `crcZ`, `finR`, `turA`, `turB`): the KT2440
  GenBank annotation carries none of these symbols and the release's own gene table has
  no row for them either. Their samples are dropped (below), not guessed onto a locus.
- Measured on the built records: the six `RelA:Del_relA*` samples hold 20 reads on PP_1656
  (0.40 TPM) against 694 in a wild-type Multistress sample, which is what a deletion looks
  like.

### Strain calls and drops

Every compendium condition is in exactly one of `REFERENCE_CONDITIONS` (62) or
`NON_REFERENCE_CONDITIONS` (56); an unlisted condition raises. A reference call rests on
the compendium labels and biosample fields (none of the source papers is mirrored).

| drop reason | samples | conditions |
|---|---|---|
| `engineered_strain_code` (muconate CJ522, GB032, GB045, GB062) | 71 | 24 |
| `non_reference_background` (UWC1, Mobile_gene and Mobile_gene2) | 24 | 7 |
| `plasmid_content` (pCAR1 conditions, multicopy wspR) | 15 | 7 |
| `engineered_evolved_sugar_strain` (Lim 2021 xylose and galactose ALE) | 14 | 7 |
| `deleted_gene_unresolved` (finR, turA, turB, crc, crcZ+crcY) | 11 | 5 |
| `evolved_isolate` (IL_T8, IL_T9) | 4 | 2 |
| `sequence_variant_allele` (IL_PP5350, a 9 bp in-frame deletion) | 2 | 1 |
| **total dropped** | **141** | 53 |

Stored: **180 records**, 171 on the reference strain and 9 single deletions (6 relA, 3
fleQ), over 65 conditions and 19 projects; 31 are in-house, 149 reprocessed.

### Replicate structure

"samples with low correlation within biological replicates $( \mathbb { R } ^ { 2 } < 0 .
9 5 )$ were discarded"; "For 321 samples that passed the aforementioned QC metrics, unique
project and condition identifiers were given (project:condition, Supplementary Data)."
Each record is one biological-replicate library; replicates share `full_name` in the
ledger. Compendium conditions: 71 with 3 replicates, 40 with 2, 7 with 4. Stored
conditions: 46 with 3, 17 with 2, 2 with 4. `RNASeqExpressionPhenotype` has no
`n_samples` field, so the count lives in the ledger, not on the record. `n_mapped_reads`
is a typed gap (`deferred_pending_source_review`): the repository's `multiqc_stats.tsv`
holds it and is not mirrored.

### Environment

The sheet has no medium, temperature or dose column. Each condition's environment is a
composition-deferred placeholder `Media` naming the condition and where its recipe is
stated (the SynLethDB precedent; `state="liquid"` and `is_synthetic=False` are not
sourced, the schema requires both), with a typed temperature gap
(`not_carried_by_curation` for reprocessed samples, `not_reported_by_primary` for
in-house ones). One placeholder per condition, so two conditions of one study never
collapse onto one environment.

The aromatic project is typed from SI Method 1: "Cells were grown in the M9 medium with
2.5 g/L of either coumarate, ferulate, a mixture of coumarate and ferulate, or glucose."
The medium is `LIM2022_M9_NO_CARBON` (the SI's "2 g/L (NH4)2SO4, 6.8 g/L Na2HPO4, 3 g/L
KH2PO4, 0.5 g/L NaCl, 2 mM MgSO4, 0.1 mM CaCl2, 500 μL/L 2000× trace element solution",
ammonium chloride as the dropout, the stated trace composition kept as one sub-mix) and
the carbon source is `EnvironmentPhysicalPerturbation(factor=carbon_source)` at 2.5 g/L;
the mixture carries two agents with gapped magnitudes. Readings recorded on the quote:
"coumarate" as p-coumaric acid, "ferulate" as ferulic acid. The 30 C of the UCSD protocol
is not asserted for the aromatic (UNMC) or the four in-house ALE samples.

### Reference

One reference per project (19), its own baseline (`reference_condition`, the condition
the paper centers each project on): `expression_tpm = 2**mean(X) - 1` over the
baseline's replicates (the level the centering subtracts), counts their rounded mean,
environment the baseline's, genome the KT2440 pin.

### Build and verification (dev tree)

`python -m torchcell.database.build_dataset_lmdb --dataset PutidaPrecise321Lim2022Dataset`:
180 records in 43 s, 19 references, gene set {PP_1656, PP_4373};
`python -m torchcell.provenance.build_manifest`: `putida_precise321_lim2022` fresh.

`verify_records` (L0 to L4, report in `preprocess/verification_report.json`): PASS. L0 180
validate; L1 count 180 and 180 distinct count profiles; L2 1,001,520 TPMs finite and
non-negative, counts non-negative integers, 180 of 180 sum to 1e6 TPM; L3 one measurement
type, references finite, one assembly pin (`pputida_KT2440_ASM756v2`,
`GCA_000007565.2`); L4 5,564 measured and 2 perturbed genes, 0 outside the 5,786-locus
KT2440 universe (`_gene_set_for_reference` on the stored pin).

The family verifier `verify_rnaseq_dataset` FAILS its L1 `strain_uniqueness` on these
records by construction ("180 records without a strain"): it keys on a perturbation
`strain_id` and one record per (strain, environment), which fits Caudal's one isolate per
record and not replicate-level records or empty wild-type genotypes. The Step 9 runner
should call `verify_records` (or a sample-keyed L1) for this dataset.

### Gaps and follow-ups

- The 12 source papers are not mirrored; mirroring them would type the 61 deferred
  conditions (medium, additions, doses, timing) and could recover the six retired
  deletion symbols and the engineered strain genotypes (Bentley 2020 for 71 samples).
- Suggested `media.py` addition (not made here, out of this branch's scope):
  `M9_NREL_NOCARBON_LIM2022`, Lim 2022 SI Method 1's NREL M9 without glucose, whose
  trace-element composition is stated in full (the library's `M9_NREL_LIM2025` defers it
  to Lim 2020 and Linger 2014).
- `torchcell/datasets/pputida/__init__.py` still says "No loader has landed yet"; left
  untouched to avoid conflicts with the sibling P. putida branches.
