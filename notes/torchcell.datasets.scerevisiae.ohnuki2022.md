---
id: jl6ylsecd9dpcx6ot4nedws
title: Ohnuki2022
desc: ''
updated: 1783884847422
created: 1783884847422
---

## 2026.07.12 - Ohnuki 2022 quadruple-mutant morphology build

`ScmdOhnuki2022Dataset` (`torchcell/datasets/scerevisiae/ohnuki2022.py`). Ohnuki et al.
2022, *npj Syst Biol Appl* 8:3, doi:10.1038/s41540-022-00212-1, PMID 35087094. CalMorph
morphology of gene-deletion strains in a drug-hypersensitive 3Δ background →
`CalMorphPhenotype`. **L0-L4 verified, 1,979 records.** (Compound side NOT built — 7
compounds, no released morphology vectors.)

- **Source (SCMD2, mirror + hash-pin):** `quad1982data.tsv` (1,982 quadruple mutants,
  sha256 `5a1d4500…`) + `wt749data.tsv` (749 3Δ reps, sha256 `4603aadf…`).
- **FULL-FIDELITY 3Δ genotype (reuses Vanacloig 2022's convention for the same background):**
  each strain = 4 deletions — target `KanMxDeletionPerturbation` + PDR1(YGL013C)
  `NatMxDeletion` + PDR3(YBL005W) `MarkerDeletion(KlURA3)` + SNQ2(YDR011W)
  `MarkerDeletion(KlLEU2)` (verified vs strain Y13206). **FLAG for review:**
  `ExperimentReference` can't hold the 3Δ background genotype, so ReferenceGenome=BY4741
  placeholder while the reference PHENOTYPE is the measured 749-rep 3Δ average — an
  asymmetry shared with Vanacloig 2022.
- 3 dropped: YGL141W (NaN in 2 CV traits → dropped whole, no imputation), PDR1/SNQ2 as
  targets (already in background). Env YPD 25 C liquid (sourced). 501 CalMorph traits.

## 2026.07.15 - Add shared-resolver ORF reconciliation (retain-all naming)

The target ORF name is now reconciled to current R64 via the shared
`reconcile_systematic_names` helper (see
[[torchcell.datasets.scerevisiae.gene_name_reconcile]]) before the NaN + 3Δ-background
drops. Previously the loader stored raw ORF names with no R64 reconciliation. Build unchanged
at **1979** (1982 − 1 NaN row YGL141W − 2 background-collision rows), but 7 target names are
now correctly remapped to current R64 ids and 1 (YAR037W) retained as a legacy name. The
3Δ-background collision check now runs on the reconciled name (catches a legacy alias of a
background gene too). Added an optional injectable `genome`.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no (source hashed first; a failed post-copy check left the copy); refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `MUTANT_SHA256`, `WT_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Issue #533 Repeated ORFs and Blank Reference Cells Refused

- A target ORF listed twice (after strip, uppercase and name reconciliation) used to give two records of one quadruple genotype; it is now refused by name. 0 of the 1982 released rows repeat.
- The 3Delta reference mean used to skip a blank cell (`mean` skips NaN) while a blank mutant cell dropped its strain. A blank reference cell is now refused, so the reference is never averaged over a subset of replicates. 0 blank cells in the released 749 x 501 `wt749data.tsv`; the blank mutant drop (YGL141W, 2 cells) is unchanged.
- Tests: `test_repeated_target_orf_refuses`, `test_blank_reference_cell_refuses`, `test_clean_base_only_matrix`.

## 2026.10.12 - Every released record stored as released; no genome at load time (#1045)

Owner rule, 2026-10-12: the data is the data; the genome and the experiment are loaded separately and meet through their gene sets at modeling time, so a loader never consults the annotation to decide whether a record exists or how it is keyed. Sweep row: [[database.genome-gating-sweep.2026.10.12]]. PR #1079 (group 1). The store is rebuilt from the branch with `--retire-existing --verify` (slurm array 4457); the L0-L4 table is appended when the task finishes.

Before: `reconcile_systematic_names` remapped 7 target ORFs to the current id (e.g. YAL058C-A to YAL056C-A, YML033W to YML034W); the two target rows that are also 3Delta background genes (YGL013C PDR1, YDR011W SNQ2) and the one row with blank cells (YGL141W HRD1, 2 of 501 CV traits) were dropped: 1,979 records.

Now: the released target ORF (stripped, upper-cased) is both names; the HRD1 row is stored with its 499 measured traits and the 2 blank cells key-absent; the PDR1 and SNQ2 targets are stored as released (the genotype carries the target deletion and the background deletion of the same gene, which is what the collection says of those strains). 1,982 records.

### Family verification (job 4606): the count pin and the one incomplete record

The morphology runner still pinned the pre-#1045 count 1979 and the `calmorph_completeness` gate refused the HRD1 record whose two blank cells are key-absent. The pin is 1982 and the runner declares that one record (`OHNUKI2022_EXPECTED_INCOMPLETE`), which the rule reads as the loader's own count; verified again by job 4658.

### L0-L4 from the rebuilt stores (slurm arrays 4457 and 4604, family verifiers job 4658)

**data/torchcell/scmd_ohnuki2022**: verdict PASS (/scratch/projects/torchcell-scratch/data/torchcell/scmd_ohnuki2022/preprocess/verification_report.json)

| level | rule | class | outcome | message |
|---|---|---|---|---|
| L0 | `structural` | gate | ok | 1982 records validated |
| L1 | `count` | gate | ok | observed 1982, expected 1982 |
| L1 | `calmorph_completeness` | gate | ok | 1/1982 records missing CalMorph parameters (1 declared by the loader) |
| L1 | `reference_populated` | gate | ok | WT reference carries the full 281+220 vocabulary in all 1982 records |
| L2 | `value_fidelity` | gate | ok | 556942 values checked |
| L2 | `cv_nonnegative` | gate | ok | 436038 values checked |
| L3 | `vocabulary_parity` | gate | ok | CALMORPH_LABELS(281) + CALMORPH_STATISTICS(220) == CALMORPH_PARAMETERS(501), disjoint=True |
| L4 | `gene_containment_scmd_ohnuki2022` | report | ok | 1.000 of scmd_ohnuki2022's 1983 deletion genes are in Ohya (>= 0.9) |

## 2026.11.24 - One record per released 3Delta replicate (#1178, part of #1171)

Per-replicate policy (#1178): the 749 rows of `wt749data.tsv` used to be folded into one plain per-feature mean; each is now its own record. The mutant records do not change.

**Replicate evidence** (mirror `ohnukiHighthroughputPlatformYeast2022/`, `paper.md` sha256 `de2cad9b...`):

- Mutants, one culture each: "We obtained wild-type data 749 times and mutant data once for each mutant." (line 97). `replicate_id = None`, records byte-identical.
- 3Delta replicates: "the morphological data of the 1982 mutants were taken every time using the 749 replicates of 3Δ over 152 days" (line 151); Fig. 4a "749 replicates of $3 \Delta$ (orange, ordered by the date of data acquisition)" (line 59); the SCMD2 page lists "749 replicated wild-type (drug-hypersensitive strain)" (`data/scmd_quadruple_summary.html` line 44, sha256 `66a3df47...`). Each is a culture grown with that day's mutants (names `Q<yymmdd><well>`, 125 distinct dates). The authors combined them only into a date-aware null (one-way ANOVA with "an experimental date as an explanatory variable", line 151) used to normalize the mutants; the release carries the raw per-culture averages.
- Genotype: the 3Delta background alone, the same three deletions every mutant record carries (Y13206, "snq2Δ:: KlLEU2 pdr3Δ:: KlURA3 pdr1Δ:: NATMX", line 128). Withdrawn samples: none quoted (749 in the paper, 749 rows).

**Loader:** records 0..1981 mutants (unchanged), 1982..2730 the 3Delta replicates in release order, `replicate_id` = `NAME`. Reference unchanged (`replicate_id = None`). Blank or repeated `NAME` refuses; `data.csv` is now written after every record is built (it was written before the records, so a refusal could leave it behind).

**Counts and mean reproduction** (same script and output as the Ohya 2005 section): 1,982 -> 2,731 records; first 1,982 identical; the 749 records equal the released cells; mutant groups max abs diff 0.0; 3Delta mean vs the old reference max abs 1.6e-11 (max rel 2.6e-15). `build_time_projection.DATASET_FULL_RECORDS` said 1,979 (stale since #1045 restored three rows); it is now 2,731.

**Verifier** (`run_ohnuki_morphology`, oracle 2,731 and 749, 1 declared incomplete record) on `$DATA_ROOT/tmp/scratch/per-replicate-scmd-morphology/scmd_ohnuki2022`: PASS. L0 2731; L1 count 2731/2731, `record_uniqueness` 2731 keys, `replicate_records` 749 of 749 distinct, completeness 1 declared short (HRD1), reference full; L2 767,411 base and 600,818 CV values; L3 parity; L4 1.000 of 1983 genes in Ohya (report).

**Not done, reported:** the environment is still the name-only stub `Media(name="YPD", state="liquid")` with no components or provenance (#622 pattern; the paper prints the 1/2/2% recipe at line 135). Changing it changes every record, so it is left for its own change.
