---
id: hmhdw99ri7n2gy389dg6lca
title: Ecoli_download_tutorial
desc: ''
updated: 1794934986070
created: 1794934986070
---


## 2026.11.17 - Download tutorial for four E. coli datasets

The code of the documentation page `docs/source/datasets/ecoli/download-tutorial.md`. It needs an installed `torchcell` and a route to the public tc-data endpoint, and nothing from the repository: no `DATA_ROOT`, no key, no checkout. Run in an empty directory it downloads four built stores into `./data/torchcell/<slug>/`, loads them with their loader classes, and prints records, tables and one content hash per dataset.

| slug | loader | records |
|---|---|---:|
| `ecoli_growth_rate_campos2018` | `GrowthRateCampos2018Dataset` | 3,664 |
| `ecoli_morphology_campos2018` | `MorphologyCampos2018Dataset` | 3,664 |
| `gene_interaction_babu2014` | `GeneInteractionBabu2014Dataset` | 42,705 |
| `metabolome_fuhrer2017` | `MetabolomeFuhrer2017Dataset` | 3,806 |

Record counts are the `n_experiments` of the endpoint's index rows, printed by the script's index block (`experiments/034-showcase-datasets/results/ecoli_download_tutorial.gilahyper.json`, field `index_rows`).

Design decisions:

- **Standard output is data only.** It holds no timing, no path and no version, so two machines can be compared by the sha256 of what they printed. The seconds each loader took go to `data/torchcell/ecoli_download_tutorial_timings.json` and to standard error.
- **The page shows the script itself.** Each block between a `# [name]` and a `# [/name]` marker is a `literalinclude` on the page, and `section()` prints the banner that [[experiments.034-showcase-datasets.scripts.ecoli_download_tutorial_page]] splits the output on, so the code on the page is the code that produced the output under it.
- **The verify block reproduces the release's content hash.** An experiment id is `sha256(json.dumps(record["experiment"]))`, the id `scripts/package_dataset_lmdb.py` and the graph adapter compute, and `content_sha256` over the ids equals the `content_sha256` of the index row. All four matched on 2026-11-17, and the four values equal the ones `database/releases/2026.10.17-4295f7ca.json` records for these datasets, so a download holds the records knowledge graph 4.0 serves.

Things the output shows that a reader of the data needs:

- 242 of the 3,664 imaging records hold 9 features and 3,422 hold 15 (the `features per strain` line).
- In the Babu records the Hfr Cavalli donor is the first perturbation in some records and the second in others; `collection` on each perturbation says which side it is.
- 21 metabolome records name more than one gene (the `records that name more than one gene` line), the pooled deposit columns the Fuhrer loader documents.
