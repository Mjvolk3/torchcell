---
id: 4vp0a9tm13r73od3gy8bpqw
title: Xue2025
desc: ''
updated: 1783966590843
created: 1783966590843
---

## 2026.07.13 - Xue 2025 in-house combinatorial FFA-titer dataset

`FattyAcidXue2025Dataset` -- FIRST COMBINATORIAL-genotype metabolite dataset (up to 6
deletions/strain). 176 records + 1 measured WT reference, from the in-house Xue 2025 FFA
overproduction screen.

- **Genotype**: every non-WT strain = the `POX1-FAA1-FAA4` FFA-overproduction baseline
  (3 deletions, implicit in ALL strains) + 0-3 TF deletions decoded from the genotype string
  (`<letters> <N>Δ`, letters via the Abbreviations sheet, `#TF + 3 == N`). `+ve Ctrl` =
  baseline only. One `KanMxDeletionPerturbation` per gene (**markers unknown/mixed for
  in-house combos -> KanMX is a documented REPRESENTATIVE**; state=absent is the asserted
  fact -- can't stack 6 KanMX). 13 genes all resolve to R64. Decode source =
  `experiments/008-xue-ffa` parser.
- **Phenotype**: `MetabolitePhenotype`, `metabolite_level={C14:0,C16:0,C18:0,C16:1,C18:1: mg/L}`,
  `measurement_type="titer_mg_per_l"`, `metabolite_level_se` = sample SD/sqrt(n) PER FFA (n
  computed from actual non-blank reps: mostly 3; 10 strains n=2, `G-O-T 6Δ` n=1 -> SE NaN,
  which the verifier skips). `reference_centered=False`. `target_metabolite_ids=None` (defer
  ChEBI/Yeast9). Env = SC/30C aerobic (in-house-assumed, flagged).
- **Reference**: measured wt BY4741 titers.

**Verifier change**: metabolite L1 `orf_uniqueness` -> `genotype_uniqueness` (keys on the full
deletion-set signature, not per-ORF) so combinatorial strains where an ORF recurs across many
strains pass; backward-compatible (single-deletion datasets reduce to per-gene; confirmed on
isobutanol). L0-L4 all pass (176 records).

**FLAGGED**: (1) in-house/unpublished -> `Publication` anchors to the FFA-chassis paper
Runguphan & Keasling 2014 (PMID 23899824); `Provenance` points to the in-house xlsx (sha256
023de80e). (2) KanMX marker is a representative (real markers mixed/unknown). (3) SC + 30C
assumed. See memory `[[remaining-datasets-blocked-status]]`.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: yes; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `DATA_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.10 - TFC7 is YOR110W, not YNL039W (issue #886)

The release codes its tenth TF as `T = TFC7` (`_EXPECTED_CODE_TO_GENE`, asserted against the `Abbreviations` sheet). In R64-4-1 `TFC7` has two candidate ORFs: it is the standard name of `YOR110W` (TFIIIC subunit tau 55) and a secondary alias of `YNL039W` (BDP1, `Alias=BDP1,B",TFC5,TFC7,...`). `_resolve_systematic` returned `candidates[0]`, and the alias table lists `YNL039W` first, so 46 of 176 records stored BDP1.

Fix: `_resolve_systematic` goes through `resolve_gene_name_strict` ([[torchcell.datasets.gene_alias_resolution]]). The release gives no ORF, so the rule for `TFC7` is the SGD standard name: `AMBIGUOUS_ALIAS_PINS` pins `TFC7 -> YOR110W` quoting `ID=YOR110W;Name=YOR110W;gene=TFC7;` from the R64-4-1 GFF. `process()` now builds every record, runs `check_ambiguous_aliases` over all 988 stored (name, ORF) pairs, and only then opens the store. The other 12 names each have one candidate ORF. A lowercase systematic name now resolves (the case-sensitive `gene_set` test noted in the old test is gone); the release carries none.

Dev store rebuilt with `python -m torchcell.database.build_dataset_lmdb --dataset FattyAcidXue2025Dataset --retire-existing --verify` (old store kept as `processed.superseded.20261010-201759`). Compared record by record against the superseded store:

| | Before | After |
|---|---:|---:|
| records | 176 | 176 |
| `(TFC7, YNL039W)` perturbations | 46 | 0 |
| `(TFC7, YOR110W)` perturbations | 0 | 46 |
| records whose genotype changed | | 46 |
| records with any other field changed (NaN-aware) | | 0 |
| distinct ORFs in the store | 13 | 13 |

L0-L4 on the rebuilt store (`run_metabolite` from `torchcell/verification/runners.py`, narrowed to this store; report `data/torchcell/ffa_xue2025/preprocess/verification_report.json`):

| Level | Rule | Result | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 176 records validated |
| L1 | `count` | PASS | observed 176, expected 176 |
| L1 | `genotype_uniqueness` | PASS | 176 unique strains (deletion sets), one record each |
| L2 | `value_fidelity` | PASS | 880 values checked |
| L2 | `se_nonnegative` | PASS | 875 values checked |
| L3 | `reference_finite` | PASS | reference level finite + key-subset for all 880 values |
| L3 | `measurement_type_consistent` | PASS | single measurement_type: 'titer_mg_per_l' |
| L4 | `gene_containment_scmd_ohya2005` | FAIL | 0.769 of scmd_ohya2005's 13 deletion genes are in Ohya (>= 0.9) |

L4 still fails, as the issue predicted, now for one cause only: the three genes missing from Ohya are `YLR071C` (RGR1), `YOR110W` (TFC7) and `YOR174W` (MED4), the essential genes stored as full deletions in a haploid chassis (#887, an owner decision, not touched here). Before the fix the missing set was `YLR071C`, `YNL039W`, `YOR174W`.

## 2026.10.12 - #887: the 176 records are stored as the spreadsheet reports them

Owner decision of 2026-10-12 on #887: "they must be inviable under certain conditions... again... the data is the data! read the excel spreadsheet. if these are the genes reported then that is what they are... leave a note in the ffa paper to look into this and check under which conditions they are essential." No record is dropped and no record carries a flag. The follow-up for the lab is in [[Strains reported as RGR1, MED4 and TFC7 deletions|experiments.008-xue-ffa.results.002-interaction-models-report#20261012---strains-reported-as-rgr1-med4-and-tfc7-deletions-check-the-conditions-of-essentiality-with-the-strain-builders]].

Measured by [[experiments.036-dataset-fixes-before-kg-build.scripts.keep_records_887_sd_zero_inventory]] (results `experiments/036-dataset-fixes-before-kg-build/results/keep_records_887_sd_zero_inventory.json`, slurm job 4201) off the pinned workbook (sha256 `023de80ec51a4d0d1ccd6fc3506e42a44fcea6d3c3101ee7a83a005e49bbc779`) and the cached SGD locus records.

**What the spreadsheet reports.** The mirror holds this one file for `xue2025`. It names the three genes on the `Abbreviations` sheet (`RGR1` / `R` in A6:B6, `MED4` / `M` in A3:B3, `TFC7` / `T` in A10:B10) and the single-TF strains as deletion counts: `R 4Δ` (titer sheet A72), `M 4Δ` (A11), `T 4Δ` (A33). Every cell of both sheets (2,803) was searched for RGR1, MED4, TFC7, truncation, DAmP, degron, conditional, hypomorph and partial: the only hits are the three gene names on the `Abbreviations` sheet. No other allele is stated.

| Gene | ORF | Records | SGD null `inviable` entry | `gene_essentiality_sgd` | In Ohya 2005 |
|---|---|---:|---|---|---|
| RGR1 | `YLR071C` | 46 | S288C, systematic mutation set, Giaever 2002 (PMID 12140549) | record 796 | no |
| MED4 | `YOR174W` | 46 | S288C, systematic mutation set, Giaever 2002 (PMID 12140549) | record 1123 | no |
| TFC7 | `YOR110W` | 46 | strain "Other", classical genetics, Manaud 1998 (PMID 9584160); none in S288C | none | no |

112 of 176 records carry at least one of the three; 83 carry RGR1 or MED4, which SGD records as inviable nulls in S288C. The other 29 carry only TFC7, whose inviable null SGD records in another background. (The issue's "three essential genes in S288C" was written while TFC7 was stored as `YNL039W`, BDP1, which is in the essentiality store; #886 fixed that.)

**Stored records: unchanged, 176 before and after,** the 138 perturbations of the three genes each a `kanmx_deletion` with `state = "absent"`, as reported. No loader change to records, so the store (array 4011 task 21, `1475b8cf1`) is not rebuilt. Tests: `test_essential_gene_strains_are_kept_as_the_release_states_them` (hermetic) and `test_the_released_workbook_keeps_all_176_records_and_the_112` (`--data`, the pinned workbook) in `tests/torchcell/datasets/scerevisiae/test_xue2025.py`.

L0-L4 on the dev store, read-only (`kg4_pre_build_verification_sweep.py --dataset FattyAcidXue2025Dataset`, job 4201; table by `no_verifier_store_report_tables.py`). L4 `gene_containment_scmd_ohya2005` fails for the three genes, as it must while the strains are reported as deletions of genes the viable haploid collection lacks; the rule is not loosened, and the open question is the lab's (#887 stays open):

`ffa_xue2025`: **FAIL**, 8 rules, 1 failing (report written 2026-10-12T07:15:08)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 176 records validated |
| L1 | `count` | PASS | observed 176, expected 176 |
| L1 | `genotype_uniqueness` | PASS | 176 unique strains (deletion sets), one record each |
| L2 | `value_fidelity` | PASS | 880 values checked |
| L2 | `se_nonnegative` | PASS | 875 values checked |
| L3 | `reference_finite` | PASS | reference level finite + key-subset for all 880 values |
| L3 | `measurement_type_consistent` | PASS | single measurement_type: 'titer_mg_per_l' |
| L4 | `gene_containment_scmd_ohya2005` | **FAIL** | 0.769 of scmd_ohya2005's 13 deletion genes are in Ohya (>= 0.9) |

## 2026.10.12 - Every released name stored as released on the released-gene-name leaf (#1045, PR #1085)

The loader no longer takes a genome. The release names its 13 genes by common name only, so every deletion is stored on `ReleasedGeneNamePerturbation` under that name (kind `deletion`, marker KanMX); TFC7, the standard name of YOR110W and an alias of YNL039W, is stored as the symbol the release wrote and the #886 pin table is gone. `preprocess/data.csv` lists `genotype,n_deletions,genes` with the sorted released names. The count stays 176; the L0-L4 table from the rebuild follows below once job 4606 finishes.

### L0-L4 from the rebuilt stores (slurm arrays 4457 and 4604, family verifiers job 4658)

**ffa_xue2025**: verdict PASS (/scratch/projects/torchcell-scratch/data/torchcell/ffa_xue2025/preprocess/verification_report.json)

| level | rule | class | outcome | message |
|---|---|---|---|---|
| L0 | `structural` | gate | ok | 176 records validated |
| L1 | `count` | gate | ok | observed 176, expected 176 |
| L1 | `genotype_uniqueness` | gate | ok | 176 unique strains (deletion sets), one record each |
| L2 | `value_fidelity` | gate | ok | 880 values checked |
| L2 | `se_nonnegative` | gate | ok | 875 values checked |
| L3 | `reference_finite` | gate | ok | reference level finite + key-subset for all 880 values |
| L3 | `measurement_type_consistent` | gate | ok | single measurement_type: 'titer_mg_per_l' |
| L4 | `gene_containment_scmd_ohya2005` | report | finding | 0.000 of scmd_ohya2005's 13 deletion genes are in Ohya (>= 0.9) |
