---
id: 0hnqmukolubbq8k2xro0gpe
title: Nguessan2025
desc: ''
updated: 1791733634016
created: 1791733634016
---

## 2026.10.11 - Loader written; every record waits on the walled Nguyen Ba 2022 genotypes

Yeast expansion wave 2, row 8. N'Guessan et al. 2025, eLife 13:RP93906, doi 10.7554/eLife.93906, citation key `nguessanRefiningResolutionYeast2025`. Loader `torchcell/datasets/scerevisiae/nguessan2025.py` (class `Nguessan2025Dataset`, deliberately unregistered), verifier [[torchcell.verification.segregant_expression_reference_panel]], adapter [[torchcell.adapters.nguessan2025_adapter]], measurements [[experiments.036-dataset-fixes-before-kg-build.scripts.nguessan2025_release_inventory]]. Design inherited from [[torchcell.datasets.scerevisiae.boocock2025]] (its pooled panel) and [[torchcell.datamodels.eqtl-data-model]]. Branch `feat/scerevisiae-nguessan2025`, chained on `feat/scerevisiae-boocock2025` (PR #1011) for the single-cell schema classes.

**State: no dev store.** The genotype of every record is a reference-panel genotype on Nguyen Ba 2022's Dryad deposit, which is walled (#1014). The loader, verifier, adapter, raw mirror, tests and candidate verdict are in place; the build, L0 to L4 on a store, the dataset-to-adapter map entry and the verification-runner entry follow the by-hand deposit.

### What the release is, corrected against the brief and the candidate row

The candidate row named the wrong authors (Boocock, Kruglyak, Albert), a "segregant panel from Bloom lineage" and "~4,500 profiled segregants". Read against the mirror and the release:

- The segregants are batch 1 of the Nguyen Ba 2022 barcoded bulk QTL (BB-QTL) panel, not Bloom's: 4,489 F2 strains of BY4741 x RM11-1a, pooled, grown in YPD to log phase and profiled in one 10x 3' run.
- 18,233 barcodes (cells) passed CellRanger. Each cell was genotyped from its transcribed SNPs by the authors' HMM and assigned to its closest batch-1 strain; the release keeps the assignment (`significant_best_match`, a 0-based batch-1 index, NA when not significant), not the per-cell genotype.
- No genotype is in the GitHub release. The README defers it: "The reference panel genotype can be downloaded at https://datadryad.org/stash/dataset/doi:10.5061/dryad.1rn8pk0vd."

The row's text fields are corrected in `experiments/database/scripts/build_candidate_datasets_table.py` and the tables regenerated (only this row's lines moved, plus the Peter 2018 verdict line that main had not regenerated).

### Measured on the release (`results/nguessan2025_release_inventory.json`)

| quantity | value | paper |
|---|---|---|
| barcodes | 18,233 | "The barcode filtering retained 18,233 barcodes." |
| released genes | 6,240 (every feature nonzero in at least one cell) | |
| median UMI per cell | 5,158 | Table 1, 5,158 |
| median genes per cell | 1,311 | Table 1, 1,311 |
| GEM groups | one (`-1` on all 18,233 barcodes) | "a single scRNA-seq run" |
| cells with a significant match | 14,263 | Figure 2A uses a stricter 13,069 (FDR < 0.05 and a reference less uncertain than the cell) |
| strains they match | 2,951 of 4,489 | |
| mean cells per matched strain | 4.8333 | Table 1, 4.83 |
| cells per matched strain | min 1, median 3, p90 10, max 53; 295 strains above ten | |
| strains with 1 / 2 / 3 / 4 / 5 cells | 649 / 527 / 386 / 282 / 225 | |
| cells without a significant match | 3,970 (median UMI 2,435.5 vs 5,464 assigned; 593 at or above the third UMI quartile, 7,037) | doublets are high-coverage barcodes "without significant genotype association" |
| best match, significant or not | indexes 0..4,487 over 3,225 strains | |
| nonzero genes per record | min 475, median 2,592, max 5,111 | |
| `umi_count` JSON bytes per record | median 36,695, p90 56,367, max 75,110; 109.3 MB over 2,951 records | |
| reference pseudobulk | 6,222 nonzero genes, 80,394,648 UMIs | |
| released genes outside the SGD S288C ORF set | 1 (`YFL057C`, 6,239/6,240 = 0.9998) | |

The genotype side of the cardinality (blocks and bytes per record at the 0.05 grid) is NOT measured: it needs `geno_data_1.txt.gz`. Hypothesis (untested): with 41,594 SNPs against Boocock's 2,190 to 3,268, a low-coverage HMM posterior could reach thousands of grid blocks per strain, so the genotype may dominate the record (Boocock's de novo cells: 533 to 729 blocks, 25 to 34 KB at the median). The inventory script measures it per encoding the moment the four files are on disk.

### Records, by design (counts from the release; no store exists)

One record per batch-1 strain with a significant cell match: 2,951 records over 14,263 cells. `SegregantExpressionExperiment` with:

- **Genotype** `SegregantGenotype`, cross `BBQTL_BYxRM`, `segregant_id` the Dryad `Number`. Parent 1 BY4741 on `sgd_S288C_R64-4-1_20230830` (`S288C_reference_sequence_R64-4-1_20230830.fsa`), parent 2 RM11-1a on `peter2018_1011_assemblies` member `GENOMES_ASSEMBLED/AAA_6.re.fa` (Peter 2018 Table S1 lists RM11-1a as AAA), the same two pins Bloom 2019 and Boocock 2025 use; no new genomes-tier set. Blocks are runs of constant called parent (P(RM) > 0.5 calls RM) and constant P(called parent) on the 0.05 grid; an exact 0.5 is written as parent 1 at 0.5. `marker_matrix_sha256` is the geno file's.
- **Phenotype** `SingleCellExpressionPhenotype`: nonzero counts over the 6,240 released genes summed over the strain's cells, `total_umis` their row sums (equal to the all-feature total, since the authors kept every feature with a UMI in any cell), `library_cell_counts` {"1": n_cells}, `cell_cycle_stage_counts` None with a `ProvenanceGap` (no stage in the release).
- **Environment** YPD (`restated(YPD_LIQUID, ...)`, same `media_identity`), `duration_hours=4`, temperature a `ProvenanceGap`.
- **Reference** the panel pseudobulk: every one of the 14,263 assigned cells summed.

### The Dryad wall (#1014)

Measured 2026-10-11 from GilaHyper: `/api/v2/files/<id>/download` answers HTTP 401 `{"error":"Unauthorized, must have current bearer token"}`; `/stash/downloads/file_stream/<id>` redirects to `/downloads/file_stream/<id>`, which answers HTTP 403; the whole-dataset `/api/v2/datasets/.../download` answers 401. The dataset metadata and the file listing are readable and are deposited. The four files the loader consumes are pinned in the raw-mirror manifest as `manual_browser` records at Dryad's listed sha-256, with the recipe as the retrieval command:

| file | bytes | sha256 (Dryad) |
|---|---|---|
| `data/dryad/README.txt` | 3,400 | `a4514d32...` |
| `data/dryad/SNP_list.txt` | 1,159,378 | `333550aa...` |
| `data/dryad/segregant_info.txt` | 5,691,263 | `0d46317e...` |
| `data/dryad/geno_data_1.txt.gz` | 5,078,870,207 | `25dc8703...` |

Formats are read from the Dryad README only (the usage notes in `data/dryad/dataset_metadata.json`, whose line breaks split the sentences): "The first column contains the identifying number of that strain, as in the 'Number'" and "Each file contains the genotype of 20,000 strains." The SNP list's columns are not named there ("Table with index, genome position, and sequence information"), so `SNP_LIST_COLUMNS = ("index", "chromosome", "position", "BY", "RM")` is an assumption the reader enforces; a different header is refused with a message naming the constant. That geno file 1 holds batch 1 is an expectation; the reader refuses and counts any batch-1 strain it does not hold.

The map from the release's batch-1 index to `Number` is taken as the batch-1 rows of `segregant_info.txt` in file order, the order the authors' code reads batch 1 in (`np.nanargmin(mtx_dist_corrected_genotype_vs_all_reference_lineages[:,0:nb_lineages_in_batch1],axis=1)`; other batches drawn by `np.random.randint(4489, nb_rows_all_genotype_df, 4489)`). This is the one inference in the join, so L4 `cis_concordance` tests it against a permuted strain map rather than assuming it.

### Raw mirror (`$DATA_ROOT/torchcell-raw/nguessanRefiningResolutionYeast2025/`, `manifest.json`)

| path | bytes | sha256 | retrieval |
|---|---|---|---|
| `data/github/Matrix_gene_expression_barcodes_1_to_9000.csv.tar.gz` | 14,178,440 | `f15e49c596a4...` | `direct_url`, arnaud00013/sc-eQTL @ `1e0e2109` |
| `data/github/Matrix_gene_expression_barcodes_9001_to_18233.csv.tar.gz` | 14,574,215 | `bf66d9e667eb...` | same |
| `data/github/Table_expressed_genes.csv` | 135,918 | `e22972a2e9c3...` | same |
| `data/github/Table_single_cell_barcodes_mapping_to_reference_panel_strains_0_based_index.csv` | 567,272 | `db7e6ca5f36e...` | same |
| `code/README.md` | 8,466 | `143d18b8d322...` | same |
| `code/III_Genotype_analysis/generate_dist_cell_to_lineage_corrected_gen.py` | 18,648 | `397b32c21ecf...` | same |
| `data/dryad/dataset_metadata.json` | 8,759 | `ddaeabd11734...` | `direct_url`, Dryad datasets API |
| `data/dryad/version_175928_files.json` | 16,774 | `56f893d83596...` | `direct_url`, Dryad files API |
| four Dryad files above | | | `manual_browser`, pending #1014 |

`1e0e2109` is the commit the paper's Software Heritage citation archives. Raw reads (SRA PRJNA1022775) are not deposited.

### Sourcing (verbatim, `paper.md` sha256 `97cf5c17...`)

- Panel: "We analyzed cells from a single batch (batch 1) of 4489  F2 segregants obtained from a F1 cross between the yeast laboratory strain BY4741 and the vineyard strain RM11-1a generated in a previous study (Nguyen Ba et al., 2022)."
- Genotypes: "Genotypes and fitness data used were the same ones obtained in the previous study." and "Thus, the genotype of the segregants can be represented by the frequency of only one of the parental alleles, which is RM in the dataset."
- Medium: "inoculated approximately $5 ^ { \star } 1 0 ^ { 6 }$ cells in YPD $1 \%$ yeast extract, $2 \%$ peptone, $2 \%$ dextrose) to saturation." and "The next day, about $\scriptstyle 1 0 ^ { 7 }$ cells were passaged to $5 \mathsf { m L }$ of fresh YPD and grew for 4 hours to bring cells to log phase."
- Temperature, searched before gapping it: the Methods name no growth temperature; the one temperature of the protocol is the spheroplasting step, "The cells were incubated at $3 7 ^ { \circ } \mathsf { C }$ for approximately 10 minutes at which point spheroplasting was verified". The "23C-30C" of Figure 2 are Nguyen Ba 2022's fitness assays, not this culture.
- Assignment: "Next, we assigned the single cell to its best match in the studied batch of 4489 trains only if this match is better than the best match in randomly generated batches of the same size".
- Release columns (code/README.md): "the column **significant_best_match** represents the same information but with missing values (NA) when the relatedness between the single cell and its closest strain is not statistically significant".
- Released genes (the lineage script): "#ONLY SELECT GENES THAT ARE EXPRESSED IN AT LEAST 1 CELL".

17 paper quotes and 11 release quotes are `SourcedValue`s; all 28 audit verbatim against the mirror bytes (`test_every_quote_is_verbatim_in_the_mirror_bytes`, run with the mirror mounted).

Found while sourcing, not fixed here: the library's YEPD recipe quote in `torchcell/datamodels/media.py` (`_YEPD_QUOTE`, cited to `yantongSyntheticGeneticArray2006/paper.txt`) elides with "..." and is not a verbatim substring of its source, so it fails `audit_sourced_value`; this loader audits only its own quotes.

### Verification

L0 to L4 has NOT run on a store; there is none. On the synthetic release the tests build (`tests/torchcell/verification/test_segregant_expression_reference_panel.py`), every rule passes except `cis_concordance`, which three synthetic strains cannot power; each rule's failure mode is tested by corrupting one thing. The rules, for the real build:

| level | rule | what it checks |
|---|---|---|
| L0 | `structural` | every record validates as a `segregant_expression` experiment |
| L1 | `count`, `segregant_uniqueness`, shared rules | 2,951 records, one per assigned strain |
| L2 | `value_fidelity` | counts, totals, cells, library partition equal the assigned cells; the stage partition is a typed gap |
| L2 | `mosaic_round_trip` | blocks re-expand to the strain's posterior: called parent exact (0.5 a tie), level within 0.025 |
| L2 | `reference_pseudobulk` | the reference and the sum of the records both equal the 14,263 assigned cells |
| L3 | `release_numbers` | 18,233 barcodes, medians 5,158 and 1,311, mean 4.83 cells per strain, 4,489 batch-1 strains |
| L3 | `sourced_values` | paper quotes in the literature mirror, release quotes in the raw mirror |
| L4 | `parents_pinned`, `marker_reference`, `gene_containment_sgd` | BY4741 on S288C, RM11-1a on AAA; sampled SNPs' BY allele is the S288C base; 6,239/6,240 genes are S288C ORFs (threshold 0.99) |
| L4 | `mat_fixation` | the panel's expression is MATa (a-specific vs alpha-specific UMIs at least 10:1), so P(RM) at the SNP nearest MAT must be fixed (within 0.05 of 0 or 1) |
| L4 | `cis_concordance` | per gene detected in at least half the strains, the absolute Pearson r of log CPM with P(RM) at its nearest SNP exceeds the 99th percentile of the same statistic under 20 permuted strain maps in at least 5% of tested genes (5x the null rate, a threshold set before any build) |

Expression measured on the release supports the MAT rule's premise (inventory `mating_type_genes`): over the 14,263 assigned cells the five a-specific genes in the matrix carry 50,457 reference UMIs (STE2 YFL026W 29,123, detected in 89.94% of the 2,951 strains) and the four alpha-specific ones 247 (MF(ALPHA)1 YPL187W 3 UMIs, detected in 0.10% of strains; SAG1 YJR004C 206), a ratio near 204 against the rule's 10. MFA1 (YDR461W) is not among the released genes.

### Owner decisions taken by recommendation

1. One record per strain (the pooled design), not per cell: the cells' own genotypes are not released, and a per-cell record would need a genotype only the matched strain supplies.
2. The 3,970 cells without a significant match are refused, as the authors refuse them; the 1,538 strains of batch 1 that no cell matched (4,489 - 2,951) have no record.
3. The genotype is the reference panel's released posterior on the 0.05 grid (Boocock's decision 1), not a hard call: Nguyen Ba 2022 released a posterior.
4. An exact 0.5 posterior is stored as parent 1 at 0.5 (Boocock's encoder raises instead because its release has none). How often it occurs is not measured.
5. `cell_cycle_stage_counts` becomes `dict | None` with a required gap rather than a new phenotype leaf: it moves only `SingleCellExpressionPhenotype`, impacting Boocock 2025 and this dataset, where a new leaf would re-stale every loader whose closure holds the unions.
6. The library label is the GEM-group token `1` every barcode carries.
7. `duration_hours=4` from the passage to log phase; the saturation culture before it is not modeled.
8. Parents named BY4741 and RM11-1a, with the paper's own words as `engineered_background`. Their marker genotypes are in Nguyen Ba 2022, which is not mirrored (#1015); the authors' CellRanger reference was built from `BY4742_fixed.fa` (`BY_reference/reference.json`), a discrepancy #1015 records.
9. The dataset is kept out of `@register_dataset`, the dataset-to-adapter map and the verification runner until the deposit: a registered class needs a passing verdict, and a mapped store that cannot be built fails the KG build's freshness check.

### Not loaded

| what | why |
|---|---|
| eQTL, QTL, hotspot and variance-partitioning results (Supplementary files 1 to 3) | derived analyses conditional on method and threshold |
| per-cell HMM genotypes | not released (only the pipeline) |
| BB-QTL fitness of these strains (18 `pheno_data_*` files) | Nguyen Ba 2022's own measurement, its own candidate row |
| DISCERN-imputed expression | a model output, not a measurement |
| raw reads, SRA PRJNA1022775 | no loader reads reads |

### Schema impact (`scripts/schema_impact_check.py --base origin/feat/scerevisiae-boocock2025`)

1 changed symbol, `SingleCellExpressionPhenotype` (field `cell_cycle_stage_counts` widened to `dict[str, int] | None`, `_check_totals` changed), classed BREAKING by the checker; impacted `Boocock2025Dataset` and `Nguessan2025Dataset` only. Boocock's records validate unchanged, but its dev store reads stale after this lands and needs one rebuild before the KG 4.0 build. Against `origin/main` the report also carries every Boocock 2025 schema change, since this branch sits on #1011.

### Candidate gate

`python -m torchcell.candidates gate --row "N'Guessan 2025" --table yeast --phenotype-class SingleCellExpressionPhenotype --write`: G1 pass; G2 pass (S288C and RM11-1a resolve and read); G3 blocked:wall on the four `manual_browser` records; G4 and G5 unmeasured; outcome blocked:wall (`database/candidates/nguessanRefiningResolutionYeast2025.json`). The ledger refused a second 2026.10.11 section ("one ledger per day"), so the verdict is not in [[torchcell.candidates.ledger]] yet.

### After the deposit (#1014)

1. Run the inventory script; record the genotype cardinality, the exact-0.5 rate, and confirm or correct `SNP_LIST_COLUMNS` and that geno file 1 holds batch 1.
2. Rewrite the raw manifest (`deposit_raw_mirror`) so the four records are on disk; re-gate (G3 should pass) and write the verdict.
3. Add `@register_dataset`, the `VERIFICATION_SPEC` entry in `runners.SEGREGANT_EXPRESSION_DATASETS` (with a module dispatch in `run_segregant_expression`), the dataset-to-adapter map entry and the pins in `test_build_time_projection.py` and `test_runners.py`.
4. `build_dataset_lmdb --dataset Nguessan2025Dataset --retire-existing --verify`, then record the L0 to L4 table here.
