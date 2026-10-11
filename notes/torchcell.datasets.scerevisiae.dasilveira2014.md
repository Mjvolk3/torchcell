---
id: n6srg6orgqp9kedz8yhg3b4
title: Dasilveira2014
desc: ''
updated: 1783835203529
created: 1783835203529
---

## 2026.07.12 - da Silveira 2014 lipidomics build

`MetaboliteDaSilveira2014Dataset` (`torchcell/datasets/scerevisiae/dasilveira2014.py`).
da Silveira dos Santos et al. 2014, *Mol Biol Cell* 25(20):3234, doi:10.1091/mbc.E14-03-0851,
PMID 25143408. Kinase/phosphatase deletion lipidomics → `MetabolitePhenotype`. **L0-L4
verified, 127 mutant records × up to 147 lipid species.**

- **Source (open-access SI, mirror + hash-pin):** Table S4 `TableS4_complete_dataset_all_lipids.xlsx`
  (sha256 `91409229…bced3894`, Quant sheet = relative abundance a.u.) + Table S10 ChEBI ids,
  fetched once from the Europe PMC supplementary ZIP (PMC4196872).
- **WT reference discovered in-data:** the Quant sheet has **3 WT control rows** (Y7092,
  Y7220, BY4741) — NOT ORFs, so excluded from mutant records; their per-lipid MEAN is the
  measured WT reference (`reference_centered=False`, restricted per record to measured
  lipids). This resolves the paper's "127 mutants" (130 rows − 3 WT = 127) and is strictly
  more faithful than a mutant population mean.
- **Value:** relative abundance in arbitrary units (`measurement_type="lipidomics_ms_relative_abundance_au"`,
  NOT concentration). `n_replicates=2` (biological duplicates, sourced verbatim; up-to-6
  technical reps not independent → not counted); no per-strain SE released → `se=None`.
- **Target ids DEFERRED:** lipids are acyl-resolved species mapping to ChEBI (Table S10,
  147/147), not Yeast9 `s_NNNN` — so `target_metabolite_ids=None`; the lipid→ChEBI map is
  emitted to `preprocess/lipid_chebi.csv` for a follow-up.
- Background `BY4741` (evidence-based: BY4741 WT control row), YPD liquid 30 C. 0 ORFs dropped.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: yes; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `DATA_SHA256`, `CHEBI_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Refusals before the store opens (issue #537)

Previous behavior: a lipid that no WT control row measured was stored on mutants with no reference value; a mutant with every lipid blank reached the phenotype validator inside the open write transaction, leaving `data.csv` with `n_lipids` 0 and an empty `processed/lmdb` that a retry served as 0 records; `gene_set` membership was case-sensitive, so `ybr001c` was unresolved.

Fix: both cases are refused with a named `RuntimeError` before `data.csv` or the store is written, and names are uppercased before the `gene_set` and alias lookups. Measured on the pinned Table S4 (`Quant` sheet): 147 lipids, all measured in at least one WT row; 127 mutant rows, none all-blank, every name uppercase and resolved; the built records do not change.

Tests: `test_a_lipid_no_wt_row_measured_is_refused_before_anything_is_written`, `test_padded_and_lowercase_systematic_names_resolve`, `test_an_all_blank_mutant_row_is_refused_before_anything_is_written`.

## 2026.10.01 - Review follow-up on PR #591

- Two mutant rows resolving to one ORF are refused with a named `RuntimeError` instead of keeping the first with a warning. Measured on the pinned Table S4: 0 such pairs of 127 mutant rows, so the built records do not change.
- `lipid_chebi.csv` is written after every record is built, so a refused matrix writes nothing under `preprocess/` or `processed/`; on success the bytes are unchanged (`test_side_files`).

## 2026.10.02 - Sourced medium replaces the inline stub (issue #622)

Issue #622: the loader built `Media(name="YPD", state="liquid", is_synthetic=False)` inline, a stub with no components and no provenance.

The Methods print their own "YPD", and it is not the library YPD (mirror `paper.md` line 168, sha256 `87b6e92b...`; the PDF text layer reads the same): 2% glucose, 1% Bacto Peptone, 2% Bacto Yeast Extract (yeast extract and peptone swapped against the library's 1% / 2%), 10 mM MES, "40 mg/ml" L-tryptophan, uracil and adenine; cultures grown to early exponential phase in it at 30 C. The loader now emits a loader-local `DA_SILVEIRA_YPD` (`base_medium="YPD"`, liquid) with seven components, each quoted in `SOURCED_VALUES`. Its `media_identity` is not `YPD_LIQUID`'s.

Held as open gaps rather than values:

- L-tryptophan: printed "40 mg/ml", which is 40 g/L. Not stored as a concentration. Hypothesis (untested): a typo for 40 mg/l. Needs the user's call.
- uracil and adenine: no amount printed.
- MES resolves to no InChIKey in the compound table, so the resolver attaches its `deferred_pending_source_review` gap.

Measured by `experiments/036-dataset-fixes-before-kg-build/scripts/media_stubs_seven_loaders.py` (output `experiments/036-dataset-fixes-before-kg-build/results/media_stubs_seven_loaders.csv`): scratch build 127 records (same as the dev store); all 254 environments carry `DA_SILVEIRA_YPD` (7 components, 5 open gaps), against the stub on all 254 in the dev store.

## 2026.10.10 - Refusing gene-name resolution (issue #886)

`_resolve_systematic` took the first `alias_to_systematic` candidate for an ORF cell that was not a live gene. It now goes through `resolve_gene_name_strict` ([[torchcell.datasets.gene_alias_resolution]]): a live ORF stays itself, a name with one candidate ORF resolves to it, an ambiguous name without a pin raises `GeneNameRefused`, and a cell R64 does not know at all is still counted unresolved and dropped, as before.

Two of the stored Standard Names are ambiguous in R64-4-1: `YPK1` (YJL093C, YKL126W, YNL307C) and `SLT2` (YAL014C, YHR030C). Their ORFs came from the `Systematic Name` column, not from the alias table, so first-match never touched them; the sweep (PR #890) saw them because it read `perturbed_gene_name`. `AMBIGUOUS_ALIAS_PINS` now records them under the rule `paper_gene_list`, quoting the table's own rows (`YKL126W | YPK1`, `YHR030C | SLT2`), and `process()` runs `check_ambiguous_aliases` over every stored (Standard Name, ORF) pair against the name -> ORF pairing read from the Quant sheet before the store opens. Both are also the R64 standard-name owners.

Measured on 2026.10.10: all 127 mutant `Systematic Name` cells are live R64 ORFs, so the strict resolver returns the same ORF for each (0 of 127 changed). A build of this branch into a scratch root from the same raw files matches the dev store record for record (127 of 127, 0 fields changed), so the dev store was not rebuilt; `--list-stale --include-private` does not name it.

L0-L4 on the dev store (`run_metabolite`, narrowed to this store; report `data/torchcell/metabolite_dasilveira2014/preprocess/verification_report.json`):

| Level | Rule | Result | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 127 records validated |
| L1 | `count` | PASS | observed 127, expected 127 |
| L1 | `genotype_uniqueness` | PASS | 127 unique strains (deletion sets), one record each |
| L2 | `value_fidelity` | PASS | 17638 values checked |
| L2 | `se_nonnegative` | PASS | 0 values checked |
| L3 | `reference_finite` | PASS | reference level finite + key-subset for all 17638 values |
| L3 | `measurement_type_consistent` | PASS | single measurement_type: 'lipidomics_ms_relative_abundance_au' |
| L4 | `gene_containment_scmd_ohya2005` | PASS | 0.992 of scmd_ohya2005's 127 deletion genes are in Ohya (>= 0.9) |
