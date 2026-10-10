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
