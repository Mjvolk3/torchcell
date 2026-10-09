---
id: 4e40s2ulotevlklvuxxbcnt
title: Lamoureux2023
desc: ''
updated: 1791383246402
created: 1791383246402
---

## 2026.10.07 - PRECISE-1K loader (rank 5 of the fifty)

Source: `torchcell/datasets/ecoli/lamoureux2023.py` (`RnaseqLamoureux2023Dataset`, root `data/torchcell/rnaseq_lamoureux2023`).
Tests: `tests/torchcell/datasets/ecoli/test_lamoureux2023.py`.
Plan: [[plan.bacteria-ontology-genome]] section 4; skeleton [[torchcell.datasets.bacteria_common]]; schema [[torchcell.datamodels.bacterial-perturbation-ontology]]; media [[torchcell.datamodels.media]].

Lamoureux et al. 2023, "A multi-scale expression and regulation knowledge base for Escherichia coli", Nucleic Acids Res., doi:10.1093/nar/gkad750. Citation key `lamoureuxMultiscaleExpressionRegulation2023`; `paper.md` sha256 `ca41cceba80fe6af16dfc8ffc671c2b9d0dac5d9eb409956891ba4bbedc147a7`, `si/si1.md` sha256 `e2fbd8b86605a9133845c793a681f7359ae4155abfc297d15977576c979b7e3b`.

One record per RNA-seq library: an MG1655 genotype (wild type or whole-gene deletions), the environment its metadata row states, and `RNASeqExpressionPhenotype` over 4,257 genes. Experiment and reference classes are `BacterialRNASeqExpressionExperiment` / `BacterialRNASeqExpressionExperimentReference`, so the assembly pin survives serialization.

### Source and retrieval

The paper's Data Availability: "All data (aside from raw RNA-seq data) and code for analysis and figures are available on Zenodo at: https: //doi.org/10.5281/zenodo.8284223." That Zenodo version holds one file, the GitHub release archive of `SBRG/precise1k` v1.0 (`isSupplementTo` <https://github.com/SBRG/precise1k/tree/v1.0>, archive root `SBRG-precise1k-71e1157`): 278,428,285 bytes, sha256 `7c7008f2c8bcd66aebecbdb97b8c1a0314637e3873ec09e12c26d0ccaaa35172`, matching Zenodo's own md5 `91e7629732198b1b3b50207d2abdb848`.

Only the four members the loader reads are mirrored, under `$DATA_ROOT/torchcell-raw/lamoureuxMultiscaleExpressionRegulation2023/`:

| member (`data/precise1k/`) | sha256 | read for |
|---|---|---|
| `log_tpm_qc.csv` | `8bdb286a...` | the values: PRECISE-1K, 4,257 genes x 1,035 samples |
| `counts.csv` | `f6307411...` | `expression_count` (featureCounts, 4,355 x 1,055) |
| `metadata_qc.csv` | `68a0c5aa...` | genotype, environment, replicate grouping |
| `log_tpm_qc_w_short_low_fpkm.csv` | `512c9fa8...` | the pseudocount back-solve only (4,355 genes) |

Each file's `RetrievalRecord` is `method=zenodo`, `retriever=torchcell.literature.retrieve.zip_member`, params `{url, member, container_sha256}`. `deposit_raw_mirror(archive_path=...)` verifies the archive and each member, is idempotent by sha256 and refuses a differing existing file.

Finding, measured 2026-10-07: Zenodo answers **HTTP 403** to the shared retriever's User-Agent (`retrieve._UA`, a Chrome 120 string) and 206 to `curl`, `python-httpx`, `torchcell-literature` and `Mozilla/5.0 torchcell-literature`. So the recorded retrieval fails as the code stands. With `_UA` set to `torchcell-literature (+https://github.com/Mjvolk3/torchcell)` in-process, the recorded `zip_member` retrieval of `metadata_qc.csv` reproduced the pinned bytes (container sha256 asserted). The archive used for the deposit was fetched with `curl -sL -o precise1k-v1.0.zip https://zenodo.org/api/records/8284223/files/SBRG/precise1k-v1.0.zip/content`. The fix belongs in `retrieve.py` (not this loader's file) and is in the PR body.

### Strain and identifiers

- Strain: "We constructed PRECISE-1K to enable a multi-scale analysis of transcription and regulation in E. coli K-12 MG1655".
- Alignment: "reads were aligned to the $E .$ . coli K-12 MG1655 reference genome (RefSeq accession number NC 000913.3)". NC_000913.3 is the chromosome of ASM584v2, the `ecoli_K12_MG1655_ASM584v2` set; `genome_reference = assembly_reference("MG1655")` (GenBank `GCA_000005845.2`).

The expression matrix is keyed by b-numbers; `reconcile_locus_tags` against MG1655:

| status | n | layer |
|---|---|---|
| current | 4,155 | locus tag |
| non_gene_feature (pseudogene) | 99 | 98 locus tag, 1 gene synonym |
| retired, kept as given | 3 (`b3036`, `b4223`, `b4590`) | not found |

One remap: release `b3681` is a gene synonym of pseudogene `b4556` and is stored as `b4556`. Resolved 4,254 / 4,257 = 0.99930 against the 0.99 floor (`EXPRESSION_MIN_RESOLVED`).

Deleted genes: the 42 symbols of the kept deletion strains all resolve (38 at the gene-symbol layer, 4 at the gene-synonym layer); the floor is 1.0, since a deletion cannot be written on a name the genome lacks. `perturbed_gene_name` is the genome's own `/gene` symbol, which resolves back to the same b-number, so four differ from the release's spelling: `ybaO` -> `decR` (b0447), `ybiH` -> `cecR` (b0796), `ydhB` -> `punR` (b1659), `yiaJ` -> `plaR` (b3574).

### Phenotype and unit

The paper: "the final expression dataset was reported in units of $\log _ { 2 }$ -transformed Transcripts Per Million $( \log _ { 2 } [ \mathrm { T P M } ] )$". No pseudocount is printed and the cited pipeline (ref 22, Sastry 2021 bioRxiv) is not mirrored, so it was **back-solved**: a TPM sums to 1e6 over the genes it was computed on, and over the 4,355-gene pre-filter matrix `sum(2**x - 1)` has median 1,000,000.0000000008 and is within 1 TPM of 1e6 for 921 of 1,035 samples (min 999,182.9, max 1,015,305.3; the other 114 deviate by -0.08% to +1.5%, not investigated). `log_tpm_qc.csv` equals the pre-filter matrix exactly on every shared cell. So `x = log2(TPM + 1)`, and the stored `expression_tpm = 2**x - 1`, the schema field's unit; `measurement_type = "rnaseq_tpm"`. The 98 genes the paper removed ("extremely low-expression transcripts $\mathrm { ( F P K M } < 1 0 )$ were also removed to reduce noise.") keep their TPM share, so a stored sample sums to less than 1e6. `expression_count` is the gene's `counts.csv` value. `n_mapped_reads` is a typed `deferred_pending_source_review` gap (the release's `multiqc_stats.tsv`, not consumed).

### Replicate structure

- "Minimum Pearson correlation with biological replicates (if any) 0.95 (if more than two biological replicates, keep samples with high correlation in ‘greedy’ manner, dropping samples that have at least one sub-threshold correlation with all other replicates)" (Methods).
- "Replicates are tightly correlated, with a median Pearson’s $r$ of 0.99 (Figure 1C)."; Figure 1 legend: "Samples included in PRECISE-1K are required to have replicate correlations of at least 0.95."
- The metadata defines a condition as `full_name` = `project:condition`, one row per library with a `rep_id` and a `Biological Replicates` count.

Each record is ONE biological-replicate library; `RNASeqExpressionPhenotype` has no `n_samples`, so none is claimed. Built: 121 conditions, of sizes 2 (107), 1 (9), 3 (4), 6 (1, `ica:wt_glc`). The metadata count equals the kept group size for 118; `ica:wt_glc` lists 2 and 4 across its six rows, one 3-library condition lists 4 and one single lists 2 (consistent with the greedy QC removal, not verified). Every condition's replicates carry identical condition cells (checked at build). `preprocess/replicate_groups.json` (verbatim cells, samples, record indices) and `preprocess/record_samples.json` (LMDB index -> sample id) carry the grouping, since records hold no sample id.

### Reference

The release centers its ICA input on the control condition ("After centering the Public K-12 dataset to the PRECISE-1K control condition"); measured: `log_tpm_qc - log_tpm_norm_qc` equals the mean of `p1k_00001` and `p1k_00002` (`control:wt_glc`, wild-type MG1655, M9 + 2 g/L glucose + Sauer trace elements) for all 1,035 samples, max absolute difference 3.6e-15. So every record shares one reference: that environment, and `expression_tpm = 2**mean(x) - 1` over the two controls with the mean count rounded half to even. The build refuses a control whose `project_reference` cell names anything else.

### Genotype settling (first matching rule)

| rule | n | why |
|---|---|---|
| `evolved_isolate` | 421 | `Evolved Sample` Endpoint or Midpoint; acquired mutations not in the release |
| `strain_bw25113` | 148 | another assembly set; writing it on MG1655 is the cross-strain inference D9 refuses |
| `heterologous_expression_construct` | 94 | `pColi` study (carbenicillin, rhamnose induction): construct, source organism and sequence not in the release; ref 37 (Tan 2020) not mirrored |
| `strain_dgf298` | 26 | genome-reduced strain, deleted regions not in the release |
| `strain_w3110` | 26 | no assembly set in the tier |
| `point_mutation_allele` | 9 | `rpoBE546V`, `rpoBE672K`; no bacterial substitution leaf |
| `partial_gene_edit` | 8 | `crp_delAr1`, `crp_delAr2`, `crp_delAr1delAr2`; no coordinates |
| `strain_gmos` | 0 | every `GMOS` row is also evolved, so the evolved rule takes it |
| **kept: wild type** | 121 | `Strain Description` exactly `Escherichia coli K-12 MG1655` |
| **kept: deletions** | 120 | 104 single, 12 double, 4 triple; tokens `del_<gene>` or `del<gene>` (`delyieP`); `Escherichia coli del_pdhR` is MG1655 by its `Strain` cell |

An unknown strain, an unknown `Evolved Sample` value or an edit token of no known form raises; nothing is guessed.

### Environment settling and encoding

| rule | n | why |
|---|---|---|
| `medium_not_in_library` | 53 | `CAMHB`, `RPMI+10%LB`, `TMA`, `W2`, `Medium C`, `01xLB`, `M9P`: no `MEDIA_LIBRARY` key and no recipe stated anywhere mirrored |
| `supplement_label_ambiguous` | 4 | `Acetoacetate/LiCl(10mM)`: two compounds, one dose |
| `culture_not_batch` | 3 | chemostat (dilution rates 0.31, 0.44); `Environment` has no culture-type slot and the `CultureEnvironment` fields do not survive the bacterial experiment's `Environment`-typed slot |
| `oxygen_regime_not_stated` | 2 | blank `Electron Acceptor` and no `anaero` in the condition (`minicoli:mg1655_glc`); `aerobicity` cannot be a typed gap |

Encoding of a kept row (the paper prints no recipe; the media agent recorded this, and the per-sample values are the metadata cells):

- `Media`: loader-local, `base_medium` `M9` (209 records) or `LB` (32), state liquid. Components: a `composition_deferred` base ("M9 base, PRECISE-1K formulation"); the named trace-element stock (`sauer trace element mixture`, `aebersold ...`, `... w/o MgSO4`) as a deferred `trace_element`; the selection antibiotic (`Kanamycin`, with 50 ug/mL when stated) as a `selection_agent`. Eight distinct media, all `derived:M9` or `derived:LB` under the verifier.
- Carbon and nitrogen sources are varied across the compendium, so each is an `EnvironmentPhysicalPerturbation` (`carbon_source`, `nitrogen_source`) with its compound as `agent` and the cell's amount in g/L (the column headers say g/L); `glucose(.2%)` is 0.2 % w/v; a bare `glucose` (26 records) has a typed `magnitude` gap.
- pH is an `EnvironmentPhysicalPerturbation(ph)` on every record with a typed `agent` gap.
- Each supplement compound (`a + b` names two) is a `SmallMoleculePerturbation`. mg/mL and mg/L are restated in ug/mL by exact decimal arithmetic; a dose with no unit (`lactic acid(5)`, `HCl(3)`) or no amount (`adenosine()`, `dibucaine`, `salicylate`) is `DoseBasis.fixed`, and its verbatim cell stays in `replicate_groups.json`.
- Chemical formulas are spelled out (`NH4Cl` ammonium chloride, `KNO3` potassium nitrate, `FeCl2`, `FeCl3`, `HCl`); abbreviations (`PQ`, `DPD`) and amino acids without a stereo prefix are kept as written and carry the resolver's `deferred_pending_source_review` InChIKey gap (22 distinct environment compounds, plus kanamycin).
- `aerobicity`: `O2` is aerobic (237); an `anaero` condition is anaerobic (4), with `KNO3(20mM)` as a 20 mM potassium nitrate perturbation where named.
- `duration_hours` is a typed gap on every environment: cells were harvested at "$( \mathrm { O D } _ { 6 0 0 } \sim 0 . 5 $ , unless otherwise specified in sample metadata file)".

The encoding is injective on the condition cells: 121 release conditions give 114 distinct (genotype, environment) pairs, and every pair that pools several conditions pools conditions with identical cells (`control:wt_glc` + `ica:wt_glc` + `ytf:wt_glc`; `misc:wt_no_te` + `oxyR:wt_glc`; `abx_media:m9_ctrl` + `eep:BOP27` + `minspan:wt_glc` + `ssw:wt_glc`; `rpoB:wt_lb` + `tcs:wt_lb`). 67 distinct environments.

### Build and verification (2026-10-07)

`python -m torchcell.database.build_dataset_lmdb --dataset RnaseqLamoureux2023Dataset`: **241 records** (of 1,035; 794 dropped by the rules above), 1 reference, gene set 42, 7 s. `torchcell.provenance.build_manifest`: `rnaseq_lamoureux2023` fresh.

Verifier, run over the stored records (`verify_rnaseq_dataset` + `shared_rule_results` with the MG1655 resolver and locus-tag set; the RNA-seq runner's registry is yeast-only):

| level | check | result |
|---|---|---|
| L0 | structural | PASS, 241 validated |
| L1 | count | PASS, 241 |
| L1 | strain_uniqueness | **FAIL by construction**: requires a `strain_id` on a perturbation, which wild-type records (none) and `BacterialDeletionPerturbation` do not carry; written for one record per isolate |
| L1 | provenance_gaps | PASS: 869 gaps (duration_hours 241, pH agent 241, n_mapped_reads 241, inchikey 120, magnitude 26) |
| L1 | canonical_gene_names | PASS, 42 |
| L2 | tpm / count fidelity | PASS, 1,025,937 values each |
| L3 | measurement_type, reference_finite, compound_identity, media_compound_identity, media_membership | PASS |
| L4 | gene containment (MG1655) | PASS, 4,254 / 4,257 = 0.99930 (the three retired) |
| L4 | perturbation genes current | PASS, 42 / 42 |

The replacement for `strain_uniqueness` is the injectivity check above, pinned in the data-gated test.

### Superset and deduplication (checklist item 7)

The paper: "PRECISE-1K constitutes a nearly 4-fold increase in size from the original 278-sample PRECISE(1)" (ref 1, Sastry 2019). Measured on the same archive (`data/precise/sample_table.csv`, read from the downloaded archive, not mirrored): 266 of PRECISE's 278 sample ids are PRECISE-1K sample ids; the other 12 are not in the compendium (presumably removed by its QC, not checked). PRECISE-1K is therefore the loaded superset and PRECISE gets no loader of its own; the 12 are the only PRECISE libraries this leaves out.

### Open items (named, not guessed)

- BW25113 (148 samples, mostly Keio-background two-component-system deletions) is a separate loader against the BW25113 set, with its own `REFERENCE_STRAIN`; its expression is still keyed by MG1655 b-numbers (reads were aligned to NC_000913.3), which is the design question to settle first.
- Seven base media need `MEDIA_LIBRARY` entries with sourced recipes (53 samples); none is stated in the mirrored paper, SI or release.
- The 421 evolved isolates need their mutation lists (ALEdb or the per-project papers), and the 94 pColi samples their construct sequences.
- `retrieve._UA` 403 at Zenodo (above).
- The RNA-seq verifier's `strain_uniqueness` and `RNASEQ_DATASETS` registry need a bacterial, replicate-aware entry.
- 22 environment compounds and kanamycin await compound-table rows.

## 2026.10.08 - The Public K-12 arm is now its own dataset

[[torchcell.datasets.ecoli.lamoureux2023_public_k12]]
(`RnaseqPublicK12Lamoureux2023Dataset`, 240 records) serves the same release's other arm,
the 1,675 reprocessed public K-12 samples. Three things about this loader changed with it,
and none alters a stored PRECISE-1K record:

- **The raw mirror is now shared.** Four more members live under the same citation key
  (`data/k12_modulome/{metadata_qc,counts,multiqc_stats}` and
  `data/annotation/gene_info.csv`), and one `manifest.json` pins all eight.
  `deposit_public_raw_mirror` is additive, so the four pins here survive it; the data-gated
  mirror test asserts a superset rather than equality.
- **`parse_amount` grew a keyword-only `extra_units`.** The public cells carry `g/L` and
  `ng/mL`, which no PRECISE-1K cell of a parsed column does (measured over `Carbon Source`,
  `Nitrogen Source`, `Supplement`, `Antibiotic for selection` and `Electron Acceptor`: the
  only matches are `mg/L`). The default is `UNIT_TOKENS`, so this loader's parsing is
  unchanged and an unknown token still raises.
- **`MAPPED_READS_GAP` is resolvable, and the other arm resolves it.** `n_mapped_reads` is
  the featureCounts `Assigned` total of the release's MultiQC table, and the Public K-12
  loader reads its arm's copy (`data/k12_modulome/multiqc_stats.tsv`), where the total
  equals the sum of the stored counts for 1,675 of 1,675 samples. The equivalent file for
  this arm, `data/precise1k/multiqc_stats.tsv`, is still not consumed, so the gap stands
  here; filling it is a dev-LMDB rebuild of this dataset, not an edit.

The open item above, "BW25113 (148 samples) is a separate loader", is now 148 + 361 = 509
samples across the two arms, which raises its priority rather than its difficulty.

## 2026.10.09 - The released growth-rate column is now its own dataset

`metadata_qc.csv` carries a `Growth Rate (1/hr)` column this loader does not read. 354 of
its 1,035 rows release a value, every one a `p1k_*` id, and the 103 the genotype and
environment rules above admit are exactly the built records of this store that carry one.
They are now served separately, as 89 absolute `EnvironmentResponsePhenotype` rates, by
`GrowthRateLamoureux2023Dataset` ([[torchcell.datasets.ecoli.lamoureux2023_growth]]); 14
of the 103 release exactly `0.0` and are dropped as indistinguishable from an unrecorded
cell.

Two facts about THIS loader the growth-rate loader had to measure. The reference it
declares, the `control:wt_glc` pair `p1k_00001` and `p1k_00002`, releases an EMPTY rate
cell, so the growth-rate dataset cannot use it and declares a base condition instead
(wild-type MG1655, M9 + `glucose(2)`, 37 C, pH 7.0: 8 released rates, mean 0.63875 h^-1).
And `settle_genotype` / `settle_environment` / `build_environment` are imported by that
loader rather than restated, so the two datasets keep exactly the same samples.

It is a SEPARATE MODULE, and the reason is measured: this module's schema closure is 61
symbols and equals the 61 the served `rnaseq_lamoureux2023` store records, and adding the
environment-response symbols raises it to 72, which would stale that store for a change
touching none of its 241 records.
