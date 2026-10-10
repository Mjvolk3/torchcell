---
id: 90ps099r045z38w6783e9d8
title: Balakrishnan2022
desc: ''
updated: 1791625319833
created: 1791625319833
---

## 2026.10.10 - Loader, schema decision and L0-L4 (issue #854)

Loader: `torchcell/datasets/ecoli/balakrishnan2022.py`, class
`MrnaFractionBalakrishnan2022Dataset`, dev store
`$DATA_ROOT/data/torchcell/mrna_fraction_balakrishnan2022`. Release inventory:
[[experiments.036-dataset-fixes-before-kg-build.scripts.balakrishnan2022_release_inventory]].
Duplication measurement:
[[experiments.036-dataset-fixes-before-kg-build.scripts.balakrishnan2022_expression_duplication]].

### Schema decision: a sibling leaf, not an optional count

`MrnaNumberFractionPhenotype` (with `MrnaNumberFractionExperiment` and its
assembly-pinned reference, graph class `mrna number fraction phenotype`) holds the
released quantity under its own name: `mrna_number_fraction` in [0, 1], a key subset of
one transcriptome whose sum may not exceed 1, `n_libraries`, and a required
`measurement_type`. Issue #854 recommended making `expression_count` optional on
`RNASeqExpressionPhenotype`; that was not taken, for two measured reasons:

- It moves the schema closure of the four served transcriptomes (Caudal 2024, Caglar
  2017, Lamoureux 2023, Lim 2022), all of which carry counts, for one release that has
  none. The sibling leaf is additive.
- Storing `psi x 1e6` under `expression_tpm` would assert a TPM pipeline (length
  normalization of read counts) the release does not document. The paper defines the
  stored number directly: "RNA-sequencing was used to determine the mRNA number
  fractions ψm,i ≡ [mRi]/[mR] for the corresponding mRNAs, with [mR] ≡ ∑i[mRi] being the
  total mRNA concentration; see SI Methods."

Schema impact (`scripts/schema_impact_check.py --base origin/main`): 0 breaking; 8
impacted dataset groups go stale only through the `ExperimentType` union gaining a
member, the shape every new experiment family has (the promoter-activity precedent).

### Records, measured on the built store

| quantity | count | source of the number |
|---|---|---|
| described samples (description sheet) | 29 | `check_sample_metadata` |
| released sample columns | 29 (`a4_1` twice) | `read_fractions` |
| records | 28 | 29 - 1 duplicate `a4_1` column |
| described but unreleased | 1 (`a3_1`) | ledger `sample_drops` |
| released rows | 4,342 | `dropped_records.json` |
| stored gene keys per record | 4,176 | every record, `samples.csv` |
| refused: b-number of a non-gene feature locus | 74 | `refused_rows.csv` |
| refused: split-fragment row whose b-number is only a feature synonym | 69 | same |
| refused: retired b-number (b3036, b3776, b4223, b4274, b4590) | 5 | same |
| refused: released locus `0` (IS-element rows) | 18 | same |
| stored released zeros (all 28 records) | 2,169 (64 to 90 per record) | `samples.csv` |
| references | 2 (M9: c5 + c0_1; MOPS: r0 + r0_1, `n_libraries=2`) | build log |

The 148 refused b-numbers (74 + 69 + 5) are the inventory's "148 b-numbers do not
resolve", now split by why. Stored fractions sum to 0.9754 to 0.9972 per record (the
refused rows carry the rest).

Correction to #854: the issue states 27 loadable columns. 29 columns minus the one
duplicate is 28. The `a4_1` data is kept under the release's own label: it cannot be
shown to be `a3_1` instead (log10 Pearson r against `a4` 0.9888, against `a3` 0.9882,
measured on the pinned bytes), and both copies carry the `a4_1` header.

### Sourcing

| value | stored as | source (sha256 prefix) | quote |
|---|---|---|---|
| quantity | `mrna_number_fraction` | `PMC9804519.1.txt` (`8f7c13f6`) | "RNA-sequencing was used to determine the mRNA number fractions ψm,i ≡ [mRi]/[mR] ..." |
| measured strain + reference condition | NCM3722 background; c5/c0_1 and r0/r0_1 | same | "For E. coli K-12 strain NCM3722 growing exponentially in glucose minimal medium (reference condition, growth rate 0.91/h)" |
| strains, media, carbon, nitrogen, supplement, growth rate | per record | Table S3 (`9d7df03b`) sheet 1 | row renderings, re-read by `check_sample_metadata` |
| Pu-ptsG (NQ1243, NQ1390) | PromoterReplacement b1101, "Pu", inducible, decreased | Table S3 + Mori 2021 Appendix `si1.docx` (`3c9490d0`) | "Titratable glucose uptake (Pu-ptsG)"; "This is done by replacing the ptsG promoter with a titratable Pu promoter from Pseudomonas putida; ..." |
| Plac-GOGAT (NQ393) | PromoterReplacement b3212, "Plac", inducible, decreased | same | "Titratable ammonia assimilation (Plac-GOGAT)"; "the pre-culture of NQ393 (an NCM3722 derivative with low GOGAT expression in GDH-null background" |
| base media recipes, temperature | deferred component; `ProvenanceGap(temperature)` | paper text | "Experimental methods for cellular growth, RNA sequencing, ... are reported in the Supplementary Material." |

Not typed, by decision (no mirrored source states the allele or the location): the xylR
driver cassettes of NQ1243 (Ptet) and NQ1390 (lacIq promoter), so the two strains share
one typed genotype; and NQ393's GDH-null allele. "3MBA" is stored under that name; no
mirrored source expands it. Three nitrogen cells read "(NH4)2SO5", "SO6", "SO7" on the
consecutive rows `a2_1`, `a3_1`, `a4_1`, a spreadsheet fill series off the "(NH4)2SO4"
above them; the loaded two are read as "(NH4)2SO4" and the verbatim cells stay in
`samples.csv`.

### L0-L4 verification (built store, `build_dataset_lmdb --verify`)

| level | rule | result |
|---|---|---|
| L0 | structural | PASS, 28 records validated |
| L1 | count | PASS, 28 of 28 |
| L1 | replicate_groups | PASS, 28 distinct profiles over 15 (genotype, environment) groups |
| L2 | number_fraction_value_fidelity | PASS, 116,928 values in [0, 1] |
| L3 | fraction_sum_at_most_one | PASS, 56 profiles sum to 0.975420 .. 0.997159 |
| L3 | measurement_type_consistent | PASS, `rnaseq_mrna_number_fraction` |
| L3 | reference_finite | PASS, 116,928 values |
| L4 | gene_containment_assembly | PASS, 1.000 of 4,176 keys are MG1655 (ASM584v2) genes |

### Duplication against served expression stores

Measured by `balakrishnan2022_expression_duplication.py`
(`results/balakrishnan2022_expression_duplication.json`), each TPM profile renormalized
to a fraction over the shared genes:

| served store | shared genes | pairs | exact profile matches (1e-6) | max log10 Pearson |
|---|---|---|---|---|
| `rnaseq_caglar2017` | 0 (REL606 namespace) | none | none | none |
| `rnaseq_lamoureux2023` | 4,118 | 6,748 | 0 | 0.918 |
| `rnaseq_public_k12_lamoureux2023` | 4,176 | 6,720 | 0 | 0.901 |

Within-condition replicate r of this release is 0.989 to 0.993 (c5~c0_1, r0~r0_1,
a2~a2_1, c3~c3_1). Verdict: independent; no served record re-serves a Balakrishnan
library.

## 2026.10.10 - Candidate gate verdict

`candidate-gate gate --row "Balakrishnan" --table bacteria --phenotype-class
MrnaNumberFractionPhenotype --issue 854 --proposed-class
MrnaFractionBalakrishnan2022Dataset --write` wrote
`database/candidates/balakrishnanPrinciplesGeneRegulation2022.json`, outcome
admissible: G1 pass (primary), G2 pass, G3 pass (2 raw files, 8,735 worksheet rows,
intact), G4 pass (no key held by another module), G5 pass. G2 passes through the gate's
K-12 lineage token rule (MG1655 and BW25113 resolve and read), not through an NCM3722
assembly: none is in the genomes tier, and the records pin MG1655 with an NCM3722
background and a genotype gap. G4 compared no dev store; the value comparison is the
separate duplication measurement above (0 exact matches, independent).
