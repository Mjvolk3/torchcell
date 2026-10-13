---
id: rvk0wf220zszqlmx1zc3p5e
title: Royet2025
desc: ''
updated: 1791614341867
created: 1791614341867
---

## 2026.10.10 - Row 53 loaded: KT2440 mariner Tn-seq in four metals

Loader `torchcell/datasets/pputida/royet2025.py`, class `EnvMetalTnseqRoyet2025Dataset`,
dev store `$DATA_ROOT/data/torchcell/pputida_env_metal_tnseq_royet2025`. Royet et al. 2025,
Environmental Microbiology 27:e70095, doi:10.1111/1462-2920.70095, PMC12041740, citation key
`royetHighThroughputTnSeqScreens2025`.

### Mirror route

Zotero-mirrored: the paper, its OCR and all ten SI objects were already in
`$DATA_ROOT/torchcell-library/royetHighThroughputTnSeqScreens2025/` (2026-10-07). The two
consumed workbooks were re-retrieved with `torchcell.literature.retrieve.pmc_cloud_object`
on 2026-10-10 (`python -m torchcell.datasets.pputida.royet2025 deposit --retrieve-into ...`)
and deposited in the raw mirror `$DATA_ROOT/torchcell-raw/royetHighThroughputTnSeqScreens2025/`
with a manifest; the bytes equal the literature mirror's capture.

| file | role | sha256 |
|---|---|---|
| `EMI-27-e70095-s006.xlsx` (library `si/si9.xlsx`) | Table S5, the stored values | `7c32029a...ec40eb` |
| `EMI-27-e70095-s005.xlsx` (library `si/si7.xlsx`) | Table S3, the replicate pools | `283c919e...857418` |

### Release inventory (measured)

Script `experiments/036-dataset-fixes-before-kg-build/scripts/royetHighThroughputTnSeqScreens2025_release_inventory.py`,
results `experiments/036-dataset-fixes-before-kg-build/results/royetHighThroughputTnSeqScreens2025_release_inventory.json`.

- Table S5 has four RESAMPLING sheets (`LB-Co`, `LB-Cu`, `LB-Zn`, `LB-Cd`), 5,729 genes
  each, the same genes in the same order, 0 missing `log2FC` or `q-value` cells. Every
  number is stored as a TEXT cell. The `Log2FC all metals` summary sheet agrees with the
  four sheets in every one of its 45,832 cells.
- The q <= 0.05 counts per sheet are 9, 14, 3 and 8, exactly the Results' "we identified
  9 genes involved in cobalt tolerance, 14 in copper tolerance, 3 in zinc tolerance, and 8
  in cadmium tolerance."
- `Mean A` (the LB arm) differs between sheets for 1,052 to 1,116 genes, because TRANSIT
  normalizes each LB-vs-metal pair separately; the LB pools are the same two (Table S3).
- Table S3: two pools per arm (`GL`, `LB`, `Co`, `Cu`, `Zn`, `Cd`, `#1` and `#2`), 12 in all.
- Table S4: two HMM sheets of 5,729 genes. LB agar: ES 600, NE 4,458, GA 435, GD 223,
  N/A 13. After LB outgrowth: ES 722, NE 4,238, GA 533, GD 223, N/A 13.
- Identifiers: 5,729 of 5,729 `#Orf` tags are current locus tags of the genome tier's
  KT2440 assembly (GCA_000007565.2), 0 remapped.

### Records

One `BacterialEnvironmentResponseExperiment` per (gene, metal). Genotype: one gene-level
`TransposonInsertionPerturbation` (`transposon="mariner (Himar1)"`, no barcode, no
position, no strand, each a typed gap). Phenotype: `EnvironmentResponsePhenotype`,
`log2_ratio`, `assay_type=other` (non-barcoded Tn-seq; the Girgis 2009 precedent),
`n_samples=2`, `sample_unit=biological_replicate`, uncertainty and SE as typed gaps (the
release carries a permutation p and a BH q, which are written to `preprocess/not_stored.json`).
Reference: the parent in the same metal at log2FC 0. Environment: LB (formulation
unstated, the shared `MEDIA_LIBRARY["LB"]` object), 28 C, aerobic, 12 generations, plus
CoCl2 10 uM, CuCl2 2.5 mM, ZnCl2 125 uM or CdCl2 12.5 uM.

| metal | released | stored | dropped (no reads either arm) | of which ES in LB | zero-TA-site | kept exact zeros | kept q <= 0.05 |
|---|---|---|---|---|---|---|---|
| Co | 5,729 | 5,394 | 335 | 246 | 13 | 120 | 9 |
| Cu | 5,729 | 5,401 | 328 | 243 | 13 | 123 | 14 |
| Zn | 5,729 | 5,398 | 331 | 246 | 13 | 342 | 3 |
| Cd | 5,729 | 5,390 | 339 | 254 | 13 | 132 | 8 |
| total | 22,916 | 21,583 | 1,333 | 989 | 52 | 717 | 34 |

Source: `preprocess/dropped_records.json` of the dev store built 2026-10-10 by
`python -m torchcell.database.build_dataset_lmdb --dataset EnvMetalTnseqRoyet2025Dataset --retire-existing --verify`.
The 5,491 distinct genes stored are the 5,729 minus the 238 genes empty in all four metals.

**The one drop rule (owner decision taken by recommendation).** Where both `Mean A` and
`Mean B` are released as `0.0`, no insertion mutant of the gene was read in either arm and
TRANSIT releases `log2FC 0.00`, `q 1.00000`. Storing that zero would assert "no metal
effect" for a gene the screen could not observe (989 of the 1,333 are ES in LB, 52 have no
TA site at all). The row's original framing ("the 22,916 records include every null") is
corrected here: 21,583 cells are measurements, including 717 measured exact zeros.

### Schema need: met by an existing class

The tranche-2 triage recorded for this row: "a fitness record whose genotype is a gene
rather than a strain, because a non-barcoded insertion pool is never resolvable to a
clone". `TransposonInsertionPerturbation` already carries that case with its per-mutant
fields left `None`, as Girgis 2009 and Borchert 2024 store it. No schema change, no issue
filed. `scripts/schema_impact_check.py --base origin/main`: "No schema contract changes vs
origin/main."

### Duplication against Borchert 2024: independent

Measured on the Borchert release (`fModule_Metadata.xlsx`, sha256 `4d649385...`) and the
dev store's `source_studies.json`: 0 of the 332 compendium samples are on LB (media are
MOPS minimal 108, MOPS glucose no-nitrogen 104, M9 78, M9 1% glucose 22, RCH2 20), 0 name a
metal, the libraries are `Putida_ML5` and `Putida_ML5_JBEI` (barcoded), and the source
studies are borchert2023, borchert2024, schmidt2022 and thompson2020. So no (gene,
condition) pair is shared and no correlation is measurable. All 4,732 compendium genes are
among Royet's 5,729 (997 Royet-only), which is the shared gene universe, not a shared
measurement. Nothing is subsumed and nothing is partitioned.

### Not stored

- Table S4's HMM essentiality calls (two growth stages, four states). `GeneEssentialityPhenotype`
  is boolean, so ES vs not-ES would load honestly but would discard the GD/GA grades; it
  would also be a second dataset class with its own adapter and pins. Left as an open
  owner decision.
- `Mean A`, `Mean B`, `Delta sum`, `p-value`, `q-value`: no field on
  `EnvironmentResponsePhenotype`; written verbatim (as parsed) to `preprocess/not_stored.json`.

### L0 to L4 (dev store, 2026-10-10)

| level | rule | result |
|---|---|---|
| L0 | structural | ok, 21,583 records validated |
| L1 | count | ok, 21,583 observed, 21,583 expected |
| L1 | pair_uniqueness | ok, 21,583 unique (study, strain, condition) |
| L1 | provenance_gaps | ok, 43,166 documented gaps over 21,583 records, 0 deferred |
| L1 | canonical_gene_names | ok, 5,491 systematic names, each current |
| L2 | value_fidelity | ok, 21,583 values |
| L2 | se_nonnegative | ok, 0 values (none released) |
| L2 | interval_orientation | ok, 0 intervals |
| L2 | uncertainty_sanity | ok, 21,583 records n_samples >= 2 with no uncertainty |
| L3 | measurement_type_consistent | ok, `log2_ratio` |
| L3 | reference_zero | ok, all 21,583 |
| L3 | environment_perturbed | ok, all 21,583 carry a metal |
| L3 | compound_identity | ok, 21,583 references carry a structure identifier |
| L3 | media_compound_identity | ok |
| L3 | media_membership | ok, 1 shared medium |
| L3 | provenance_audit | ok, 13 of 13 sourced values verbatim in `paper.md` |
| L4 | gene_containment | ok, 1.000 of 5,491 genes in the KT2440 assembly |
| L4 | current_genome_genes | ok, 5,491 of 5,491 |

Report: `$DATA_ROOT/data/torchcell/pputida_env_metal_tnseq_royet2025/preprocess/verification_report.json`.

## 2026.10.10 - Table S5's p-value and q-value stored on every record (#877)

Issue #877; carrier added by PR #880 (#863): `environment_response_p_value`,
`environment_response_p_value_adjusted`, `p_value_adjustment_method` on
`EnvironmentResponsePhenotype`. Every one of the 21,583 kept records now carries its cell's
released `p-value` and `q-value` verbatim, with `p_value_adjustment_method="benjamini_hochberg"`.
The reference phenotype carries none (a log2FC of 0 by construction is not a test).

### The correction, sourced twice

- **Publisher PDF.** Table 1, footnote f, page 8 of `paper.pdf` (sha256 `d6d54b16...64498f09`),
  as `pdftotext -layout` (poppler 21.01.0) reads it: "f p-­Values adjusted for multiple
  comparisons using the Benjamini-­Hochberg procedure (see Transit manual)." The MinerU OCR
  (`paper.md`) dropped Table 1's footnotes, so the auditable `paper.md` quote is the
  Results' "FDR adjusted $p$ -value $( q$ -value)" (`SOURCED_VALUES["p_value_adjustment"]`,
  whose note records the footnote), and the footnote is kept as `TABLE1_FOOTNOTE_F`.
- **Back-solve** (`adjustment_back_solve`, written to `preprocess/adjustment_back_solve.json`).
  Within each metal sheet, BH over the released p-values is compared with the released
  q-values. p is printed at 1e-4 resolution and q to 5 decimals, so a faithful BH lands
  within 5e-6; the build refuses above `ADJUSTMENT_TOLERANCE = 1e-5`.

| family | LB-Co | LB-Cu | LB-Zn | LB-Cd |
|---|---|---|---|---|
| 5,729 (the released rows) | 1.67e-4 | 1.73e-4 | 7.5e-5 | 1.75e-4 |
| 5,730 (released + 1 unreleased test) | 5.0e-6 | 5.0e-6 | 0 | 3.3e-6 |

(max |BH - released q| per sheet, from the build's `preprocess/adjustment_back_solve.json`:
`released_family_max_abs_deviation` and `per_sheet_max_abs_deviation`.) So the released
q-values are BH over a family of 5,730 tests, one more than any table releases: Table S5's
four sheets and its summary sheet, and both Table S4 sheets, list the same 5,729 genes.
Hypothesis (untested): TRANSIT's annotation carried one gene that was removed from the
released workbook. The loader encodes this as `UNRELEASED_TESTS = 1`; the family is the whole sheet,
the 1,333 dropped empty genes included, so no q is recomputed over the kept subset.

### not_stored.json

`columns` is now `Mean A`, `Mean B`, `Delta sum` for all 22,916 cells. The 1,333 dropped
cells' p and q (all released as 1 and 1) move to `dropped_cells_test`, so nothing released is
lost. The `environment_response_uncertainty` gap note now points at the new fields.

### Schema impact

`scripts/schema_impact_check.py --base fix/env-response-p-value-863`: "No schema contract
changes". Against `origin/main` the only change is #880's (`EnvironmentResponsePhenotype`, 36
impacted dataset groups, 0 breaking), which already lists Royet 2025 and rebuilds with KG 4.0.

### L0 to L4 (dev store rebuilt 2026-10-10)

`python -m torchcell.database.build_dataset_lmdb --dataset EnvMetalTnseqRoyet2025Dataset --retire-existing --verify`;
LMDB entries 21,583; `--list-stale --include-private` does not name the dataset.

| level | rule | result |
|---|---|---|
| L0 | structural | ok, 21,583 records validated |
| L1 | count | ok, 21,583 observed, 21,583 expected |
| L1 | pair_uniqueness | ok, 21,583 unique (study, strain, condition) |
| L1 | provenance_gaps | ok, 43,166 documented gaps over 21,583 records, 0 deferred |
| L1 | canonical_gene_names | ok, 5,491 systematic names, each current |
| L2 | value_fidelity | ok, 21,583 values |
| L2 | se_nonnegative | ok, 0 values (none released) |
| L2 | interval_orientation | ok, 0 intervals |
| L2 | uncertainty_sanity | ok, 21,583 records n_samples >= 2 with no uncertainty |
| L2 | released_test_fidelity (new) | ok, 21,583 of 21,583 carry the released p-value and q-value verbatim |
| L3 | measurement_type_consistent | ok, `log2_ratio` |
| L3 | reference_zero | ok, all 21,583 |
| L3 | environment_perturbed | ok, all 21,583 carry a metal |
| L3 | compound_identity | ok, 21,583 references carry a structure identifier |
| L3 | media_compound_identity | ok |
| L3 | media_membership | ok, 1 shared medium |
| L3 | p_value_adjustment_back_solve (new) | ok, BH over a family of 5,730 reproduces 4 sheets x 5,729 rows to 5.0e-6 |
| L3 | provenance_audit | ok, 15 of 15 sourced values verbatim in `paper.md` |
| L4 | gene_containment | ok, 1.000 of 5,491 genes in the KT2440 assembly |
| L4 | current_genome_genes | ok, 5,491 of 5,491 |

Report: `$DATA_ROOT/data/torchcell/pputida_env_metal_tnseq_royet2025/preprocess/verification_report.json`.

## 2026.10.12 - Table S4: the HMM essentiality state of every gene (#909, PR #1083)

`GradedGeneEssentialityRoyet2025Dataset` (`data/torchcell/graded_gene_essentiality_royet2025`, organism `pputida`) stores Table S4 (`EMI-27-e70095-s002.xlsx`, sha256 `b786a2a1b995a736b96a894e7b623a1ca1ec419827373114f03b9bd597991f42`, 713,725 bytes, `pmc_cloud`, retrieved 2026-10-07), now deposited in the raw mirror beside S5 and S6 (`NOT_MIRRORED` no longer lists it; the literature mirror's copy is `si/si8.xlsx`). Each of the two sheets, "HMM analysis - LB agar" and "HMM analysis - outgrowth in LB", gives every one of the 5,729 KT2440 genes a TRANSIT HMM state with its TA sites, the sites labeled each state, the mean insertion rate and the mean read count. Records: 5,729 x 2 = 11,458 `BacterialGradedGeneEssentialityExperiment`, one per (gene, stage), the gene under the released Gene ID and Gene name with no annotation lookup.

| stage | ES | NE | GA | GD | N/A |
|---|---|---|---|---|---|
| LB agar | 600 | 4,458 | 435 | 223 | 13 |
| outgrowth in LB | 722 | 4,238 | 533 | 223 | 13 |

Measured on the workbook by `read_table_s4`; the LB agar ES and NE counts are asserted against the Results' own statement (`hmm_counts`). The 13 `N/A` rows per sheet are the genes with no TA site, stored as the released state `N/A`.

- **Evidence.** `tnseq_hmm_state` under `insertion_depletion`: the scoring rule is the Results' TRANSIT HMM sentence (`_Q_HMM`), the state definitions are Table 1's footnote c, quoted from paper.pdf page 8 in the `hmm_counts` note because the OCR dropped the footnote (`TABLE1_FOOTNOTE_C`, soft hyphens as the PDF text has them). The eight per-gene statistics go on the descriptor, with the label-site counts summing to the TA sites.
- **Environment.** LB agar with gentamicin, 24 h at 28 C, for the plated pool (`plating`, `plated_pool`); for the second sheet, ~10^7 mutants inoculated in 25 mL LB and grown at 28 C, shaken, for 12 generations, the LB arm of the screen (`outgrowth`). The medium carries no amounts because the paper gives none. Gentamicin is unresolved in the compound table, so the candidate verdict is `admissible_with_gaps` on #1063.
- Reference: the unperturbed KT2440 with `essentiality_call = "NE"`.
- `all_raw_files()` replaces the module constant so the deposit, retrieval and CLI iterate S4, S5 and S6 and tests can substitute the list.
- Verification: `verify_graded_build` (registered in `runners.BACTERIAL_GRADED_ESSENTIALITY_DATASETS`).

### L0-L4 of the graded store

Built and verified by slurm array 4584 (task 2, `--retire-existing --verify`, source `1c55407e4`); rendered by `experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_report_tables.py graded_gene_essentiality_royet2025`.

`graded_gene_essentiality_royet2025`: **PASS**, 31 rules, 0 failing (report written 2026-10-12T23:40:35)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 11458 records validated |
| L1 | `count` | PASS | observed 11458, expected 11458 |
| L2 | `calls_match_table_s4` | PASS | 11458 records are the 11458 Table S4 rows, call and statistics |
| L3 | `evidence_is_tnseq_hmm_state` | PASS | 11458 of 11458 records carry the tnseq_hmm_state statements; every reference is the parent called NE |
| L1 | `provenance_gaps` | PASS | 11458 documented provenance gaps over 5729/11458 records; 1 deferred field(s): ['inchikey']; 401030 undeclared None values over 27 carrier fields (top: Compound.inchi x45832, Compound.smiles x34374, Compound.chebi_id ... |
| L1 | `canonical_gene_names` | PASS | 5729 systematic names, one canonical spelling each (no resolver supplied: spelling checked, annotation not) |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 17187 compound references carry a structure identifier; 5729 declare a typed gap (1 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 5729 records on a shared MEDIA_LIBRARY medium, 5729 on a medium deriving from one (2 distinct media) |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L4 | `perturbed_gene_containment_assembly` | PASS | 1.000 of 5729 measured genes are loci of pputida_KT2440_ASM756v2 (>= 1.0) |
