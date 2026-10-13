---
id: czf1blmi8sxbwq7yjor01eq
title: Yeastphenome
desc: ''
updated: 1790814227090
created: 1790814227090
---

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against per-screen `valuez_sha256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.12 - Every released record stored as released; no genome at load time (#1045)

Owner rule, 2026-10-12: the data is the data; the genome and the experiment are loaded separately and meet through their gene sets at modeling time, so a loader never consults the annotation to decide whether a record exists or how it is keyed. Sweep row: [[database.genome-gating-sweep.2026.10.12]]. PR #1079 (group 1). The store is rebuilt from the branch with `--retire-existing --verify` (slurm array 4457); the L0-L4 table is appended when the task finishes.

Before: a column was stored only when its zygosity was haploid or homozygous diploid, its phenotype a growth readout, its medium a base in the two library tables, and its condition one dosed agent in `_UNIT_MAP`; every other growth column was counted and left out: 53 condition columns (237,558 values), 14 media columns (63,859), 2 heterozygous columns (6,862).

Now (`_parse_conditions`, `_parse_media`): a condition string is a comma-joined list; `standard` is the base condition with no edit; `temperature [N C]` sets the environment temperature (no temperature gap then); `time [N h]` / `[N gen]` set the durations; `oxygen [0%]` sets `aerobicity="anaerobic"`; each dosed agent is a `SmallMoleculePerturbation` (defensins stay `BiologicPerturbation`); `%`, `% v/v`, `% w/v` join `_UNIT_MAP`; a repeated identical component counts once. A `het` column is an `EngineeredCopyNumberPerturbation` (2 -> 1, KanMX) in BY4743. A medium base outside the two tables is classified by prefix (`YP` rich; `SC`, `SD`, `YNB`, `CSM`, `MM`, `SynBase` synthetic), anything else stores `is_synthetic` as a typed gap, and a 4-field header stores a `Media` whose name, state and synthetic flag are typed gaps (the SGD precedent). 133 environments, 524,682 records (was 83 and 296,777).

Not stored, counted in `preprocess/dropped_records.json` under `condition_not_representable`: 19 columns (80,374 released values) whose condition has a component no schema field carries: a dose RANGE (`hydrogen peroxide [1-1.2 mM]` x7 in berry_gasch_2011, `caspofungin [10-30 ng/ml]`, `[60-120 ng/ml]`), an agent with no dose (`gentamicin`, `Actara`, `Mospilan`, `Confidor`, `16001701`; `Concentration` needs a value or a basis), a unit `ConcentrationUnit` lacks (`ng/ml`: `rapamycin [4 ng/ml]`; `U/ml`: `zymolyase [30 U/ml]`), and a genotype edit written as a condition (`loss + cytoduction of mtDNA`, `expression of human CYP1A2`, `expression of human NAT2`). The two units are an additive enum change for the schema PR; the rest need a carrier. One `expression of mtDNA (ARG8m)` column (4,473 values) is a reporter readout, not growth, and stays out of this growth loader. A row id with no ORF form is counted under `deferred_to_released_gene_name_leaf` (0 today).

## 2026.10.12 - The deferred name is a record on the released-gene-name leaf (#1045, PR #1085)

A row id with no ORF form is stored on `ReleasedGeneNamePerturbation`, kind `heterozygous_deletion` for a `het` column and `deletion` otherwise, marker KanMX (0 of the released rows); the ledger key is `stored_on_released_gene_name_leaf`. The count stays 524,682.

### Family verification (job 4606) and what it took to pass

The first family-verifier pass over the rebuilt store (slurm 4606) failed three gates, each a property of the released matrices rather than a defect: `pair_uniqueness` found 14,508 records repeating an earlier (study, strain, condition) triple (the curated matrices release the same strain under one parsed condition in more than one column of a screen; the repeated records are identical in units, medium and edit, measured over all 524,682 records), `environment_perturbed` found 8,975 records with no environmental edit (the base-condition columns), and `media_membership` found every medium free text because `_parse_media` set no `base_medium`. The loader now anchors each released medium token at the `MEDIA_LIBRARY` key its leading word or classifying prefix names (`YPD`, `SC`, `SD`, `YNB`, `YP` for `YPG [3%]`, `LB`), keeping the released name as `name` and substituting no library object; the runner spec declares the 14,508 repeated pairs (`expected_duplicated_pairs`) and the 8,975 unperturbed records (`expected_unperturbed`), both read by the verifier as the loader's own count, so an undeclared repeat still fails. Store rebuilt by slurm 4657 and verified by 4658; the L0-L4 table follows.
