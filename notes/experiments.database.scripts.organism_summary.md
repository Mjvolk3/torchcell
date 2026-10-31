---
id: 2wsep2bl7hkcy02k0meralu
title: Organism_summary
desc: ''
updated: 1793431572737
created: 1793431572737
---

## 2026.10.31 - One row per organism of a knowledge-graph release

`experiments/database/scripts/organism_summary.py` writes
`experiments/database/results/organism_summary.json`: per organism of a release (default
`2026.10.17-4295f7ca`, knowledge graph 4.0), the datasets and records of the release snapshot,
the distinct experiment types their records instantiate (from `dataset_probe_cache.json`,
`probes-v1`; a dataset the cache does not hold is listed as unprobed and counted in no type),
the assembly sets of the genomes tier (`strain_or_population`, file roles, whether the set is an
assembly or a variant table), and the gene count of the primary reference through its genome
class (`SCerevisiaeGenome`, `EcoliK12MG1655Genome`, `PPutidaKT2440Genome`). The dissertation's
organism table (thesis `common/organisms_table.py`, `tab:organisms`) is rendered from the JSON,
so none of its numbers is typed by hand. The JSON is tracked beside the release snapshots it
summarizes (an exception in `experiments/database/results/.gitignore`).

Measured 2026-10-31 on GilaHyper:

| Organism | Reference genomes | Genes | Datasets | Experiment types | Records |
| :-- | :-- | --: | --: | --: | --: |
| *Saccharomyces cerevisiae* | S288C; 1,011 isolates; CEN.PK113-7D | 6,607 | 70 | 24 | 65,889,107 |
| *Escherichia coli* | K-12 MG1655; B REL606; K-12 BW25113; K-12 W3110 | 4,506 | 68 | 17 | 4,561,865 |
| *Pseudomonas putida* | KT2440 | 5,729 | 32 | 8 | 2,695,603 |
| **Total** | | | **170** | | **73,146,575** |

The one unprobed dataset is `InhibitorBioscreenVolk2021Dataset`, whose records are
`StrainEnvironmentResponseExperiment` (its loader), a type the other yeast datasets already
count, so the 24 stands. The probe cache's instance counts differ from the release for eight
chemogenomic yeast datasets (Hoepfner 2014 holds 30,308,574 rows in the measured files and
3,127,814 records in the graph), which is why records come from the release snapshot and the
cache supplies only the experiment type.
