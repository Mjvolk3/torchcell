---
id: nyf8kb9s9pw2sxq8dncq3bu
title: Nguessan2025_release_inventory
desc: ''
updated: 1791733657626
created: 1791733657626
---

## 2026.10.11 - What the N'Guessan 2025 release holds, measured before the build

`experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_release_inventory.py` reads the GitHub release from the raw mirror (sha256-checked against `manifest.json`) and writes `experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_release_inventory.json`. Every number in [[torchcell.datasets.scerevisiae.nguessan2025]] comes from that file. Run from the repo root with the worktree on `PYTHONPATH`.

Measured 2026-10-11 (GitHub arnaud00013/sc-eQTL @ `1e0e2109`):

- release: 18,233 barcodes, 6,240 genes, median 5,158 UMIs and 1,311 genes per cell, one GEM group (`1`), best matches indexing 0..4,487 over 3,225 strains; one released gene (`YFL057C`) outside the SGD S288C ORF set;
- assignment: 14,263 cells with a significant match to 2,951 strains, mean 4.8333 (Table 1: 4.83), median 3, p90 10, max 53, 295 strains above ten cells; 3,970 cells unassigned, median UMI 2,435.5 against 5,464 assigned, 593 of them at or above the third UMI quartile (7,037);
- records: 475 to 5,111 nonzero genes (median 2,592); `umi_count` JSON 36,695 bytes at the median, 109.3 MB over 2,951 records; reference 6,222 nonzero genes, 80,394,648 UMIs;
- mating-type genes: a-specific reference UMIs 50,457 (STE2 in 89.94% of strains), alpha-specific 247 (MF(ALPHA)1 in 0.10%).

The Dryad section reads `deposited: false` and names the four missing files (#1014). Once they are on disk the same run adds the batch-1 strain count, the SNP count, the fraction of posteriors exactly 0.5 and inside (0.05, 0.95), and blocks and pickled genotype bytes per strain for a called-parent run and the 0.1, 0.05, 0.02 and 0.01 grids, the cardinality the loader's 0.05 grid was chosen against for Boocock 2025.

## 2026.10.12 - The Dryad side, measured on the deposit

Re-run under slurm (job 4195, 4 min 3 s, 2.9 GB peak RSS) after the four Dryad files arrived (#1014), from a frozen snapshot of commit `79792b952`. The script now counts blocks vectorized, adds the loader's own encoder (`ng.encode_reference_posterior`) beside the grids, and reads every row of `geno_data_1.txt.gz`. The release and assignment sections are unchanged; the Dryad section:

- `segregant_info.txt` batch 1: 4,489 strains, `Number` 0 to 4,488 in file order;
- `SNP_list.txt`: 41,610 rows, 16 of them `END`, 41,594 SNPs;
- `geno_data_1.txt.gz`: 20,000 rows, `Number` 0 to 19,999 ascending and contiguous, 41,594 posteriors on every row; all 4,489 batch-1 strains and all 2,951 assigned strains are in it;
- blocks per strain at the loader's 0.05 grid: min 57, median 483, mean 591.8, p90 1,219, max 2,225; pickled genotype bytes median 20,529, max 96,028, 74.5 MB over 2,951 strains (called parent only: median 63 blocks; 0.1 / 0.02 / 0.01 grids: median 330 / 750 / 1,030);
- fraction of a strain's markers at exactly 0.5: median 0, p90 0, max 0.0134; in (0.05, 0.95): median 0.0258, mean 0.0811, max 0.674.
