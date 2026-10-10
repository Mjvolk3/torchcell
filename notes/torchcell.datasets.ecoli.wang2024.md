---
id: zxgdgcy1hjie8piy33h27zv
title: Wang2024
desc: ''
updated: 1791614347340
created: 1791614347340
---

## 2026.10.10 - Loader for the rifampicin dose-by-time Tn-seq screen (row 52)

Wang, Fu, Shi, Zhao and Lyu 2024, Microbiology Spectrum 12:e02895-23 (doi:10.1128/spectrum.02895-23, PMC10782999, citation key `wangGenomeWideScreenRevealsCellular2024`). Row 52 of the ranked bacterial candidate table. Loader `torchcell/datasets/ecoli/wang2024.py`, class `EnvChemgenWang2024Dataset`, dev root `data/torchcell/ecoli_env_chemgen_wang2024`.

### Mirror route

The paper is not in the literature mirror (no `torchcell-library` manifest carries the DOI). Raw-mirror deposit on the Mohiuddin 2022 pattern, through `torchcell.literature.retrieve.pmc_cloud_object` (PMC Article Datasets bucket prefix `PMC10782999.1`, scriptable), into `$DATA_ROOT/torchcell-raw/wangGenomeWideScreenRevealsCellular2024/`:

| file | role | sha256 |
|---|---|---|
| `data/spectrum.02895-23-s0002.xlsx` (Table S2) | raw_data | `143c77c80093adbead1293f79100a5c9a0bf6cbe8d6e7208d0e29710773f9f98` |
| `paper/PMC10782999.1.pdf` | paper_pdf | `8c24d5fbf0bad2b56dc2c8669ba2c6eade9a11cd2a981888b27e46d9a2f153ae` |
| `paper/PMC10782999.1.txt` | paper_text (quote anchor) | `0e499878b45bd741ed81bfe557013980faca39ea72006b4b9a30bd9312d89b0d` |
| `si/spectrum.02895-23-s0002.legends.txt` | si_text, derived (ProcessingRecord) | `c78aab7d2a945f6153bcb1a2f47a615ce7d8b58ad51f8915dcc05ca13434e94c` |

The bucket's own md5 for each object matched the downloaded bytes. Tables S3 and S4 and the SI PDF are not consumed and not mirrored (`NOT_MIRRORED`): S3 is the hit call over S2, S4 is DAVID enrichment of S3.

Re-run: `python -m torchcell.datasets.ecoli.wang2024 deposit --retrieve --download-dir <dir>`.

### Release inventory (measured)

`experiments/036-dataset-fixes-before-kg-build/scripts/wangGenomeWideScreenRevealsCellular2024_release_inventory.py`, results in `experiments/036-dataset-fixes-before-kg-build/results/wangGenomeWideScreenRevealsCellular2024_release_inventory.json`.

- Six sheets (`0.25xMIC-1hour` ... `20xMIC-3hours`), each 4,419 data rows x 11 columns, 4,419 distinct `#Orf`, zero non-numeric value cells: 26,514 cells.
- The input columns (`Sites`, `Mean Ctrl`, `Sum Ctrl`) are identical across the six sheets: one input sample is the control of every comparison.
- The input has no reads (`Sum Ctrl` 0) for the same 303 genes on every sheet; 277 to 294 of them per sheet also have none after treatment, and every such row prints log2FC 0.
- `log2FC` is matched within 0.011 by `log2((Mean Exp + 1) / (Mean Ctrl + 1))` on the rounded means for 4,179 to 4,217 of 4,419 rows per sheet and within 0.1 for all but one row. This is an observation about the release, not a sourced formula.
- The paper's own cut (|log2FC| > 1, adjusted p < 0.05) gives 132 / 73 / 178 / 161 / 206 / 158 genes per sheet and 363 in the union; the paper states 365.

### Values and where they come from

Every value is a `SourcedValue` quoting `paper/PMC10782999.1.txt` or the rendered legend rows, all 18 audited PASS against the raw mirror at L3.

- Host MG1655, GCA_000005845.2, no strain background (the Tn5 library was transformed into MG1655 itself). TRANSIT mapped to NC_000913.3.
- Dose: 2, 32 and 160 mg/L rifampicin ("treated with different concentrations of rifampicin (2, 32, and 160 mg/L, Sigma Aldrich, R3501) at 37°C with rotation"), MIC 8 mg/L ("had a rifampicin MIC of 8 mg/L (Fig. S1A)"). Stored as `Concentration(value, ug/mL, basis=DoseBasis.MIC)` with the multiple in the perturbation's `description`, e.g. `rifampicin at 4x MIC (MIC 8 mg/L)`. The absolute concentration IS released, so there is no unit finding to file.
- Exposure 1 or 3 h on `Environment.duration_hours`; LB, 37 C, aerobic. Kanamycin was washed out before the split and is not on the medium.
- Phenotype `EnvironmentResponsePhenotype`, `log2_ratio`, `assay_type=other` (no `AssayType` member for junction sequencing), `n_samples=2` biological replicates (one log2FC over the two replicates' summed counts), `screen_id` = sheet name. Uncertainty and SE are typed `not_reported_by_primary` gaps.
- Reference per condition: the parent at log2FC 0 in the same environment.

### Retention

| rule | cells |
|---|---|
| released | 26,514 |
| `no_insertion_reads_in_either_pool` (Sum Ctrl = Sum Exp = 0; per sheet 287 / 294 / 285 / 277 / 286 / 292) | 1,721 |
| `b_number_is_not_a_locus_tag_of_the_pinned_annotation` (b3036, b4223, b4590, b4700 not found; b3681 a synonym of pseudogene b4556) x 6 | 30 |
| stored | 24,763 |

4,414 of 4,419 b-numbers are locus tags as released, so no record carries a `DerivedIdentifierMapping`. 4,169 genes carry at least one record; the other 245 kept loci have no reads in any sheet. 9 to 26 genes per sheet with no input reads but reads after treatment are kept.

### Not stored: the p-value columns

`p-value` and `Adj. p-value` have no carrier on `EnvironmentResponsePhenotype`. Schema finding filed as issue #863 (`dataset`, `before-next-kg-build`), proposing an additive p-value pair on the class.

### Duplication check

Independent. Spearman r of each Wang condition against the rifampicin records in the dev stores:

- Choe 2025 CRISPRi (3.2 ug/mL, about 3,780 shared MG1655 b-numbers): -0.028 to 0.027.
- Shiver 2016 Keio (4 ug/mL, about 3,420 shared genes joined by symbol): -0.001 to 0.121.

Different library, modality and dose; no row re-serves another.

### Build and verification

`python -m torchcell.database.build_dataset_lmdb --dataset EnvChemgenWang2024Dataset --retire-existing --verify`: 24,763 records, gene_set 4,169, 6 references, 15 s. `--list-stale --include-private` does not name it.

| level | check | result |
|---|---|---|
| L0 | structural | 24,763 records validated |
| L1 | count | 24,763 = expected |
| L1 | pair_uniqueness | 24,763 unique (study, strain, condition) |
| L1 | provenance_gaps | 74,289 documented gaps; deferred field `inchikey` (rifampicin is name-only in the compound table) |
| L1 | canonical_gene_names | 4,169 systematic names, each current |
| L2 | value_fidelity | 24,763 values |
| L2 | uncertainty_sanity | no labeled uncertainty; n_samples 2 on every record |
| L3 | measurement_type_consistent | `log2_ratio` only |
| L3 | reference_zero | 24,763 of 24,763 |
| L3 | environment_perturbed | 24,763 of 24,763 |
| L3 | media_membership | LB Miller, shared library medium |
| L3 | provenance_audit | 18 of 18 sourced values backed by their verbatim quote |
| L4 | gene_containment | 1.000 of 4,169 genes in MG1655 |
| L4 | current_genome_genes | every name a current gene |

Schema impact: `scripts/schema_impact_check.py --base origin/main` reports no schema contract changes. Adapter `torchcell/adapters/wang2024_adapter.py` with conf `ecoli_env_chemgen_wang2024_adapter.yaml`, appended to `kg_bacteria.yaml` after Mohiuddin 2022.
