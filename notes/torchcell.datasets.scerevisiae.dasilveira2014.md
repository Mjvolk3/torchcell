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

## 2026.10.12 - Names stored as released; the loader reads no genome

Owner, 2026-10-12: "the data is the data!!!! it is what they reported". A mutant row is stored under its `Systematic Name` cell (stripped and upper-cased, the case the perturbation schema's systematic-name format requires) and its `Standard Name` cell (the systematic name when blank). The loader no longer calls `gene_alias_resolution`, no longer drops a row whose ORF R64 does not know, no longer refuses an ambiguous standard name without a pin, and no longer takes a `genome`. Two rows naming one ORF are still refused (one strain, one record); a lipid no WT row measured and an all-blank mutant row are still refused.

Measured 2026-10-12 against the built dev store (`$DATA_ROOT/data/torchcell/metabolite_dasilveira2014`): its 127 stored (systematic, standard) pairs equal the 127 released (`Systematic Name`, `Standard Name`) pairs of the pinned Table S4, so the change moves no stored record and the store is not rebuilt. Comparing those names with R64 is now the reported `canonical_gene_names` / `current_genome_genes` findings of its verification ([[torchcell.verification.report]]).

## 2026.11.24 - Every Quant row is a record; the three wild types join the store (#1171, #1178)

Per-replicate policy (#1178, owner 2026-11-17 and 2026-11-24), item 7: a control row the source releases is a record whenever its genotype is statable. Table S4's `Quant` sheet has three wild-type rows, `Y7092` (WT1), `Y7220` (WT2) and `BY4741` (WT3), which the loader read only into the reference. They are now three records with an EMPTY genotype, and each record's `genome_reference.strain` is the strain id its row gives (the mutants keep `BY4741`). 127 -> 130 records, stored in release row order: the wild types are records 116-118, and the 11 mutant rows after them move from 116-126 to 119-129.

The three are three strains, not three replicates of one, so `replicate_id` is `None` on each, and each `Quant` value is the release's own aggregate over two cultures (below). `phenotype_reference` is unchanged: the per-lipid mean over the three wild-type rows, restricted to the lipids the record measured, `replicate_id = None`.

### What a replicate is here (sources read in full)

Mirror `paper.md` (sha256 `87b6e92b...`), line 31: "We analyzed the lipid profile of 129 yeast knockout strains and wild type in biological duplicates with up to six technical replicates per biological replicate." Line 176: "Two independent biological replicates were analyzed, each of which comprised up to six technical replicates."

The Supplemental Information PDF was NOT in either mirror. It is listed in the article's supplementary material (PMC4196872 XML, `supp_E14-03-0851v1_CombinedSupMats.pdf`) and was fetched from the PMC Open Access AWS bucket: `curl -o supp_E14-03-0851v1_CombinedSupMats.pdf https://pmc-oa-opendata.s3.amazonaws.com/PMC4196872.1/supp_E14-03-0851v1_CombinedSupMats.pdf`, retrieved 2026-11-25T02:14:57Z, sha256 `f5b2afec39d23135b54a64e3a661ed618a063ab7bff4ad55abf0ca10b27fc4f4`. Verbatim (pdftotext -layout):

- p. 36: "Knockout strains were plated in YPD plates and isolated colonies were used for the two independent cell cultures." A biological replicate is an independent culture.
- p. 36: "The screening was conducted in 12 batches of lipid analysis. Each batch contained a wild-type strain and up to 23 mutants. For each biological replicate (BR), the MS analysis was done up to six sequential repetitions, herein called technical replicates (TR). ... From that, we computed the median of the values obtained in the TRs for each lipid in each sample." The technical level is a median inside each culture.
- p. 37: "The corrected values are herein referred as “Quant”." (after the interplate correction).
- p. 37: "A cutoff r=0.7 was established and two strains were excluded from the analysis (cla4Δ and ptc7Δ)."

Table S4 releases one `Quant` value per strain x lipid; the per-culture (BR1, BR2) values are in no SI table. So the 127 mutant records and the 3 wild-type records are aggregate-only as released (policy item 8), `n_replicates = 2` per lipid, `metabolite_level_se = None`. For a wild-type row `n_replicates = 2` is the conservative lower end of an unstated count: a wild-type strain ran in every one of the 12 batches, and the three wild-type rows do not say how many cultures each pools.

The 129-vs-127 discrepancy flagged since 2026.07.12 is resolved: Table S1 lists 129 mutants including CLA4 (YNL298W) and PTC7 (YHR076W), the two the authors excluded at the r = 0.7 QC, and Table S4 releases the other 127. Nothing is dropped by the loader.

The SI tables the plan listed as unmirrored were fetched from the same bucket (`https://pmc-oa-opendata.s3.amazonaws.com/PMC4196872.1/<file>`, same timestamp). Journal file `s03`, `s04`, `s06` and `s10` are byte-identical to the mirrored Tables S1, S2, S4 and S10 (sha256 `c6a69801...`, `3c7d55f2...`, `91409229...`, `29a7ed0a...`). The others:

| File | Table | sha256 | Content |
|---|---|---|---|
| `s05.xlsx` | S3 | `dd5eee82c8f6e14ef012389d60767c14ffb4ea650bf7420a63ac300e2bf9e62f` | per-lipid and per-category mean, quartiles, SD, thresholds over all strains |
| `s07.xlsx` | S5 | `f2a1e379ad87d77d4c52950f33afb68dc5b99cae653f4ecc5f0fe337ccbbc180` | lipid categories, same 130 rows as S4, sheets HIT, Quant, Norm, log2, SSMD |
| `s08.xlsx` | S6 | `ac097b7bf627e53b726d6b9e6de031b1ae0a7aa79c94b2501e046635646ef320` | hits and relative impact per strain |
| `s09.xlsx` | S7 | `5cd16af9ba8884c7f76c3805b62e832ad82435fc40f2843d38dac3a8d514d3fb` | WT, kcs1, pho85, vip1 sphingolipids: Mean and SD only |
| `s11.xlsx` | S11 | `f8cac929336abd93affd8b9b268cc1f16f24518571ad12b4b04826cc7fd6094c` | CED / PPA / snf1 follow-up (FY1679 background): `Detailed data set` sheet has one row per biological replicate, 18 rows = 6 strains x 3 (`WT-1` ... `PPA_snf1-3`), each labelled `Median` |

None of S3, S5, S6, S7 is per-replicate screen data. S11 IS per-replicate, but it is a separate experiment (Fig. 5, six FY1679-background strains, three cultures each) that no loader reads; it is a candidate for a new dataset class, not part of this conversion. The fetched files stay outside the raw mirror, which holds only what the loader consumes.

### Verification and the mean check

Scratch build `$DATA_ROOT/tmp/scratch/per-replicate-dasilveira2014/metabolite_dasilveira2014` (pre-change build beside it, `before_metabolite_dasilveira2014`). `verify_metabolite_dataset` with the updated `runners.py` spec (`expected_count` 130): PASS, L0 130 records, L1 count 130 / 130, L1 `genotype_uniqueness` 130 unique keys, L2 `value_fidelity` 18,072 values, L3 `reference_finite` 18,072 values, L4 `gene_containment_scmd_ohya2005` 0.992 of 127 genes.

The L1 uniqueness key in `torchcell/verification/metabolite.py` now carries the reference genome's `strain` (the #1057 precedent in `environment_response`), because three empty-genotype records are told apart only by the strain their reference names. A dataset with one reference strain keys as before.

Grouping the 130 records by (genotype, environment) gives 128 groups: 127 single mutant groups whose value and reference equal the pre-change store record for record (max abs difference 0 over 17,638 values), and one group of the three wild types, whose per-lipid mean reproduces the old (and current) reference over all 147 lipids with max abs difference 7.3e-12 (float summation order).
