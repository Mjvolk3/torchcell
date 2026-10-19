---
id: l06c9llh805cvdjnrxoy89b
title: Cross_dataset_correlation
desc: ''
updated: 1792437323710
created: 1792437323710
---

## 2026.10.19 - Cross-dataset correlations of the raw store

Reads the 043 raw LMDB once (`--stage extract`, job 5890: 173,617 records seen in
1,048 s, MaxRSS 164 GB by `sacct`, most of it the mapped LMDB pages). It caches
record-level matrices of the 11 compared datasets under
`$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/cross_dataset_cache/`.
Then, in `--stage analyze` (final run job 5892, from the cache), it computes per pair:

- (A) per shared genotype, Pearson and Spearman across the shared genes, with at least 20
  finite pairs;
- (B) per shared gene, Pearson across the shared genotypes, with at least 5.

The morphology ridge follows the design of the 019 `morphology_input_screen.py`.
Launcher: [[experiments.043-expression-proteome-morphology.scripts.gh_cross_dataset_correlation]];
rerun the analysis alone with `sbatch --export=ALL,STAGE=analyze`.

Outputs: `results/cross_dataset_correlation.json` (pair statistics, morphology, extraction
metadata, the quantity table), `results/cross_dataset_correlation/<pair>_per_genotype.csv`
and `<pair>_per_gene.csv`. Figures (timestamped, `ASSET_IMAGES_DIR`):

![](./assets/images/043-expression-proteome-morphology/cross_dataset_per_genotype_pearson_2026-10-19-14-37-41.svg)

![](./assets/images/043-expression-proteome-morphology/cross_dataset_mrna_protein_per_gene_2026-10-19-14-37-41.svg)

![](./assets/images/043-expression-proteome-morphology/cross_dataset_morphology_ridge_2026-10-19-14-37-41.svg)

Identifier mapping: Messner 2023 UniProt accessions to ORFs through the SGD GFF (all
mapped); Nadal-Ribelles 2025 standard names to ORFs through the inverted ORF-to-name map
(411 released names unmapped and left out of the comparison).

A smoke run on every 7th record (`--stride 7 --tag _smoke7`) checked the pipeline; its
outputs were moved to `/scratch/projects/torchcell-scratch/deprecated/2026.10.19/xcorr_smoke7/`.
